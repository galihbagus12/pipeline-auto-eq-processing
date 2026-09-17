# ============================================================
# EQTransformer Picking Pipeline
# Model            : iquique pretrained / stead
# Output format    : STEAD-style CSV
# Input structure  : input_folder/<YYYY-MM-DD>/<NET>.<STA>..<YYYY-MM-DD>.mseed
# Station metadata : stations.txt (network, latitude, longitude, elevation_m)
# Extras           : progress bar (tqdm) + checkpoint file (automatic resume)
#
# Output structure :
#   output_folder/<YYYY-MM-DD>/<NET>.<STA>/csv/results_<NET>.<STA>_<date>.csv
#   output_folder/<YYYY-MM-DD>/<NET>.<STA>/plots/overview/...
#   output_folder/<YYYY-MM-DD>/<NET>.<STA>/plots/events/...
#   output_folder/<YYYY-MM-DD>/<NET>.<STA>/plots/summary/...   (per station-day statistics)
#   output_folder/summary/ALL_results_combined.csv             (all stations and days
#                                                                 combined, appended per
#                                                                 file -> resumable)
#   output_folder/checkpoint.json
#
# NOTE: This file is deliberately split into:
#   1) MODULE IMPORTS
#   2) CONFIGURATION (loaded from config/config.yaml -- see CONFIG_YAML_PATH)
#   3) PROCESS (utilities, QC, CSV builder, plotting, main pipeline)
# CONFIG is loaded from the 'picking' section of the shared config/config.yaml
# (see pipeline.py at the project root, which runs every stage from that same
# file) -- edit the YAML to change parameters, not this file; the CONFIG dict
# keys and every function signature below are unchanged, so PROCESS never
# needs to be touched for a config-only change.
# ============================================================

# ============================================================
# 1) MODULE IMPORTS
# ============================================================

import seisbench.models as sbm
from obspy import read, UTCDateTime
import torch
import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import re
import glob
import json
import time
import warnings
warnings.filterwarnings('ignore')
import seisbench

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError(
        "Package 'tqdm' is not installed. Run: pip install tqdm"
    )


# ============================================================
# 2) CONFIGURATION
# ============================================================

CONFIG_YAML_PATH = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")

with open(CONFIG_YAML_PATH, 'r') as _f:
    _all_cfg = yaml.safe_load(_f)

CONFIG = _all_cfg['picking']
CONFIG['network'] = _all_cfg['network']   # single global network code (see stations.txt)

# ============================================================
# 3) PROCESS
# ============================================================

# ------------------------------------------------------------
# 3.1 General utilities (folders, checkpoint, filename parsing)
# ------------------------------------------------------------

def setup_output_folders(output_folder):
    folders = {
        'root'     : output_folder,
        'summary'  : os.path.join(output_folder, "summary"),
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    return folders


def setup_station_day_folders(output_folder, day_str, station_label):
    """
    Create the per-day -> per-station folder structure:

    output_folder/<day_str>/<NET>.<STA>/csv/
    output_folder/<day_str>/<NET>.<STA>/plots/overview/
    output_folder/<day_str>/<NET>.<STA>/plots/events/
    output_folder/<day_str>/<NET>.<STA>/plots/summary/   (per station-day statistics)
    """
    base = os.path.join(output_folder, day_str, station_label)
    folders = {
        'root'     : base,
        'csv'      : os.path.join(base, "csv"),
        'plots'    : os.path.join(base, "plots"),
        'overview' : os.path.join(base, "plots", "overview"),
        'events'   : os.path.join(base, "plots", "events"),
        'summary'  : os.path.join(base, "plots", "summary"),
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    return folders


def append_to_combined_csv(combined_path, df):
    """
    Append the result of one mseed file to the combined CSV in the 'summary'
    folder. Because rows are always appended rather than rewritten from scratch,
    an interrupted run can continue without redoing what was already recorded.
    This works together with checkpoint.json, which prevents the same file from
    being processed and appended twice.
    """
    if df.empty:
        return
    os.makedirs(os.path.dirname(combined_path), exist_ok=True)
    write_header = not os.path.exists(combined_path)
    df.to_csv(combined_path, mode='a', header=write_header, index=False)


def load_stations_metadata(path, network):
    """Load station metadata from stations.txt (columns: station latitude
    longitude elevation_m) -> dict {STA: {network, latitude, longitude,
    elevation_m}}. network is a single global constant (config.yaml's
    top-level 'network' key) since stations.txt itself carries no per-station
    network code -- every station on this array shares one."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"stations.txt not found: {path}")
    data = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            sta, lat, lon, elev = line.split()
            data[sta] = {
                'network': network,
                'latitude': float(lat),
                'longitude': float(lon),
                'elevation_m': float(elev),
            }
    return data


def load_checkpoint(checkpoint_file):
    """Load the list of files already processed (for resuming)."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            return set(data.get('processed', []))
        except Exception:
            return set()
    return set()


def save_checkpoint(checkpoint_file, processed_set):
    """Save the list of processed files to the checkpoint file (JSON)."""
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'processed'   : sorted(processed_set),
            'total'       : len(processed_set),
            'last_update' : str(pd.Timestamp.now()),
        }, f, indent=2)


def parse_mseed_filename(filepath):
    """
    Parse an mseed filename of the common form: NET.STA.LOC.YYYY-MM-DD.mseed
    Example: 3M.EJA13..2015-12-01.mseed -> network=3M, station=EJA13, date=2015-12-01

    Return: (network, station, date_str) -- date_str is None if not found.
    """
    base  = os.path.basename(filepath)
    parts = base.split('.')

    network = parts[0] if len(parts) > 0 else None
    station = parts[1] if len(parts) > 1 else None

    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', base)
    date_str   = date_match.group(1) if date_match else None

    return network, station, date_str


def get_day_folders(input_folder, pattern):
    """Return all per-day sub-folders inside input_folder, sorted."""
    candidates = sorted(glob.glob(os.path.join(input_folder, pattern)))
    return [d for d in candidates if os.path.isdir(d)]


def build_model_call_kwargs(cfg):
    """
    Build the performance kwargs (batch_size, parallelism) passed to
    model.annotate() / model.classify(). parallelism is only included when
    > 1 (cross-file multiprocessing, multi-core CPU only).
    """
    kwargs = {}
    if cfg.get('batch_size'):
        kwargs['batch_size'] = cfg['batch_size']
    if cfg.get('parallelism') and cfg['parallelism'] > 1:
        kwargs['parallelism'] = cfg['parallelism']
    return kwargs


def apply_quality_filters(result_df, cfg):
    """
    Apply quality filters to the picking results:
    1. min_peak_value  -> drop detections whose probability is too low
                           (orphan pick rows without a detection are kept here and
                            are caught by filter #2 when require_p_and_s is on)
    2. require_p_and_s  -> when True, only rows with both a valid P AND S
                           (not NaN) count as an event
    """
    df = result_df[
        result_df['detection_probability'].isna() |
        (result_df['detection_probability'] >= cfg['min_peak_value'])
    ].copy()

    if cfg.get('require_p_and_s'):
        df = df[
            df['p_arrival_time'].notna() & df['s_arrival_time'].notna()
        ]

    return df.reset_index(drop=True)


def check_excessive_gaps(st, max_gap_seconds):
    """
    Check whether the stream contains an inter-segment gap larger than
    max_gap_seconds. Used BEFORE st.merge(fill_value='interpolate') to prevent
    ObsPy from attempting a huge array allocation on corrupt timestamps or
    extreme gaps (which can trigger swapping/thrashing that looks like the
    process hanging at 0% CPU).

    Return: list of (trace_id, gap_seconds) -- empty when safe.
    """
    if not max_gap_seconds:
        return []

    problems = []
    try:
        for gap in st.get_gaps():
            # get_gaps() tuple format: (net, sta, loc, chan, t_start, t_end, delta, n_samples)
            t_start, t_end = gap[4], gap[5]
            gap_duration = float(t_end - t_start)  # seconds
            if gap_duration > max_gap_seconds:
                trace_id = ".".join(str(x) for x in gap[:4])
                problems.append((trace_id, round(gap_duration, 1)))
    except Exception:
        # If get_gaps() itself fails, do not halt the pipeline -- fall through to
        # the normal merge() (this guard is best-effort, not mandatory).
        return []

    return problems


def check_components(st):
    channels = [tr.stats.channel[-1].upper() for tr in st]
    missing  = [c for c in ['Z', 'N', 'E'] if c not in channels]
    return missing


def build_station_info(network_file, station_code, stations_meta, instrument_type):
    """
    Build the station_info dict, preferring metadata from stations.txt.
    Returns None if the station is absent from stations.txt (event is skipped).
    """
    meta = stations_meta.get(station_code)
    if meta is None:
        return None

    return {
        'network'         : meta.get('network', network_file),
        'station'         : station_code,
        'instrument_type' : instrument_type,
        'station_lat'     : meta.get('latitude', ''),
        'station_lon'     : meta.get('longitude', ''),
        'station_elv'     : meta.get('elevation_m', ''),
    }


def compute_snr(st, pick_time, phase, window_signal=1.0, window_noise=1.0):
    """
    Compute a simple SNR for the P or S phase.
    noise  : window before pick_time
    signal : window after pick_time
    """
    try:
        if phase == 'P':
            comps = [tr for tr in st if tr.stats.channel[-1].upper() == 'Z']
        else:
            comps = [tr for tr in st if tr.stats.channel[-1].upper() in ['N', 'E']]

        if not comps:
            return None

        tr = comps[0]

        tr_noise = tr.copy().trim(
            starttime=pick_time - window_noise,
            endtime=pick_time
        )
        tr_signal = tr.copy().trim(
            starttime=pick_time,
            endtime=pick_time + window_signal
        )

        if tr_noise.stats.npts < 2 or tr_signal.stats.npts < 2:
            return None

        rms_noise  = np.sqrt(np.mean(tr_noise.data ** 2))
        rms_signal = np.sqrt(np.mean(tr_signal.data ** 2))

        if rms_noise == 0:
            return None

        snr = round(float(rms_signal / rms_noise), 4)
        return snr

    except Exception:
        return None


# ------------------------------------------------------------
# 3.2 CSV builder (STEAD-style)
# ------------------------------------------------------------

def build_stead_style_csv(picks, detections, st, station_info, date_str):
    """
    Build a STEAD-style DataFrame from picks and detections.

    Output columns:
    file_name, network, station, instrument_type,
    station_lat, station_lon, station_elv,
    event_start_time, event_end_time,
    detection_probability, detection_uncertainty,
    p_arrival_time, p_probability, p_uncertainty, p_snr,
    s_arrival_time, s_probability, s_uncertainty, s_snr
    """

    file_name = f"{station_info['network']}.{station_info['station']}.{date_str}"

    p_picks = [p for p in picks if p.phase == 'P']
    s_picks = [p for p in picks if p.phase == 'S']

    rows = []

    for det in detections:
        t_start = det.start_time
        t_end   = det.end_time

        p_match = None
        for p in p_picks:
            if t_start <= p.peak_time <= t_end:
                if p_match is None or p.peak_value > p_match.peak_value:
                    p_match = p

        s_match = None
        for s in s_picks:
            if t_start <= s.peak_time <= t_end:
                if s_match is None or s.peak_value > s_match.peak_value:
                    s_match = s

        p_snr = compute_snr(st, p_match.peak_time, 'P') if p_match else None
        s_snr = compute_snr(st, s_match.peak_time, 'S') if s_match else None

        det_prob = round(float(det.peak_value), 6) if hasattr(det, 'peak_value') else None
        det_unc  = round(1.0 - det_prob, 6) if det_prob is not None else None

        p_prob = round(float(p_match.peak_value), 6) if p_match else None
        p_unc  = round(1.0 - p_prob, 6) if p_prob is not None else None

        s_prob = round(float(s_match.peak_value), 6) if s_match else None
        s_unc  = round(1.0 - s_prob, 6) if s_prob is not None else None

        row = {
            'file_name'              : file_name,
            'network'                : station_info['network'],
            'station'                : station_info['station'],
            'instrument_type'        : station_info['instrument_type'],
            'station_lat'            : station_info['station_lat'],
            'station_lon'            : station_info['station_lon'],
            'station_elv'            : station_info['station_elv'],

            'event_start_time'       : str(t_start),
            'event_end_time'         : str(t_end),
            'detection_probability'  : det_prob,
            'detection_uncertainty'  : det_unc,

            'p_arrival_time'         : str(p_match.peak_time) if p_match else None,
            'p_probability'          : p_prob,
            'p_uncertainty'          : p_unc,
            'p_snr'                  : p_snr,

            's_arrival_time'         : str(s_match.peak_time) if s_match else None,
            's_probability'          : s_prob,
            's_uncertainty'          : s_unc,
            's_snr'                  : s_snr,
        }
        rows.append(row)

    matched_p_times = set()
    matched_s_times = set()
    for row in rows:
        if row['p_arrival_time']:
            matched_p_times.add(row['p_arrival_time'])
        if row['s_arrival_time']:
            matched_s_times.add(row['s_arrival_time'])

    for p in p_picks:
        if str(p.peak_time) not in matched_p_times:
            p_snr  = compute_snr(st, p.peak_time, 'P')
            p_prob = round(float(p.peak_value), 6)
            rows.append({
                'file_name'             : file_name,
                'network'               : station_info['network'],
                'station'               : station_info['station'],
                'instrument_type'       : station_info['instrument_type'],
                'station_lat'           : station_info['station_lat'],
                'station_lon'           : station_info['station_lon'],
                'station_elv'           : station_info['station_elv'],
                'event_start_time'      : None,
                'event_end_time'        : None,
                'detection_probability' : None,
                'detection_uncertainty' : None,
                'p_arrival_time'        : str(p.peak_time),
                'p_probability'         : p_prob,
                'p_uncertainty'         : round(1.0 - p_prob, 6),
                'p_snr'                 : p_snr,
                's_arrival_time'        : None,
                's_probability'         : None,
                's_uncertainty'         : None,
                's_snr'                 : None,
            })

    df = pd.DataFrame(rows, columns=[
        'file_name', 'network', 'station', 'instrument_type',
        'station_lat', 'station_lon', 'station_elv',
        'event_start_time', 'event_end_time',
        'detection_probability', 'detection_uncertainty',
        'p_arrival_time', 'p_probability', 'p_uncertainty', 'p_snr',
        's_arrival_time', 's_probability', 's_uncertainty', 's_snr',
    ])

    return df


def classify_quality(val):
    if val >= 0.9:   return 'High'
    elif val >= 0.7: return 'Medium'
    elif val >= 0.5: return 'Low'
    else:            return 'Very Low'


# ------------------------------------------------------------
# 3.3 Plotting functions
# ------------------------------------------------------------

def plot_overview(st, annotations, result_df, station_info,
                  date_str, folders, cfg):
    try:
        n_det = len(result_df)
        n_p   = result_df['p_arrival_time'].notna().sum()
        n_s   = result_df['s_arrival_time'].notna().sum()

        fig, axes = plt.subplots(5, 1, figsize=(22, 14), sharex=True)
        fig.suptitle(
            f"EQTransformer ({cfg['model_name']}) | "
            f"Station: {station_info['network']}.{station_info['station']} | "
            f"{date_str}\n"
            f"P-picks: {n_p} | S-picks: {n_s} | Detections: {n_det} | "
            f"P_thr: {cfg['P_threshold']} | S_thr: {cfg['S_threshold']}",
            fontsize=12, fontweight='bold'
        )

        comp_map = {
            'Z': ('k',       'Vertical (Z)'),
            'N': ('navy',    'North (N)'),
            'E': ('darkred', 'East (E)'),
        }

        for idx, comp in enumerate(['Z', 'N', 'E']):
            tr_list = [tr for tr in st if tr.stats.channel[-1].upper() == comp]
            if tr_list:
                tr     = tr_list[0]
                t_plot = tr.times("matplotlib")
                color, label = comp_map[comp]
                axes[idx].plot(t_plot, tr.data, color=color,
                               linewidth=0.3, alpha=0.85)
                axes[idx].set_ylabel(label, fontsize=8)
                axes[idx].grid(True, alpha=0.25)

                for _, row in result_df.iterrows():
                    if pd.notna(row['p_arrival_time']):
                        try:
                            pt = mdates.date2num(
                                pd.Timestamp(row['p_arrival_time']).to_pydatetime()
                            )
                            axes[idx].axvline(x=pt, color='blue',
                                              linewidth=0.6, alpha=0.5)
                        except Exception:
                            pass
                    if pd.notna(row['s_arrival_time']):
                        try:
                            st_ = mdates.date2num(
                                pd.Timestamp(row['s_arrival_time']).to_pydatetime()
                            )
                            axes[idx].axvline(x=st_, color='red',
                                              linewidth=0.6, alpha=0.5)
                        except Exception:
                            pass

        ann_p = annotations.select(channel="*P*")
        ann_s = annotations.select(channel="*S*")
        ann_d = annotations.select(channel="*D*")

        if ann_p:
            axes[3].plot(ann_p[0].times("matplotlib"), ann_p[0].data,
                         'b', linewidth=0.5, label='P prob', alpha=0.8)
        if ann_s:
            axes[3].plot(ann_s[0].times("matplotlib"), ann_s[0].data,
                         'r', linewidth=0.5, label='S prob', alpha=0.8)
        axes[3].axhline(y=cfg['P_threshold'], color='blue',
                        linestyle='--', linewidth=0.8, alpha=0.5,
                        label=f"P thr={cfg['P_threshold']}")
        axes[3].axhline(y=cfg['S_threshold'], color='red',
                        linestyle='--', linewidth=0.8, alpha=0.5,
                        label=f"S thr={cfg['S_threshold']}")
        axes[3].set_ylabel('P/S Probability', fontsize=8)
        axes[3].set_ylim(0, 1)
        axes[3].legend(loc='upper right', fontsize=7, ncol=2)
        axes[3].grid(True, alpha=0.25)

        if ann_d:
            axes[4].fill_between(ann_d[0].times("matplotlib"),
                                 ann_d[0].data, alpha=0.35, color='green')
            axes[4].plot(ann_d[0].times("matplotlib"), ann_d[0].data,
                         'g', linewidth=0.5, label='Detection prob')
        axes[4].axhline(y=cfg['detection_threshold'], color='green',
                        linestyle='--', linewidth=0.8, alpha=0.5,
                        label=f"Det thr={cfg['detection_threshold']}")
        axes[4].set_ylabel('Detection\nProbability', fontsize=8)
        axes[4].set_ylim(0, 1)
        axes[4].legend(loc='upper right', fontsize=7)
        axes[4].set_xlabel('UTC time', fontsize=9)
        axes[4].grid(True, alpha=0.25)

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))

        plt.tight_layout()
        fname = os.path.join(
            folders['overview'],
            f"overview_{station_info['network']}."
            f"{station_info['station']}_{date_str}.{cfg['plot_format']}"
        )
        plt.savefig(fname, dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.close()

    except Exception as e:
        plt.close()
        print(f"    ⚠️  Overview failed: {e}")


def plot_per_event(st, result_df, station_info,
                   date_str, folders, cfg):
    if result_df.empty:
        return

    max_plots = cfg['max_event_plots']
    n_plots   = len(result_df) if max_plots == 0 else min(max_plots, len(result_df))

    for i, (_, row) in enumerate(result_df.head(n_plots).iterrows()):
        try:
            if pd.isna(row['event_start_time']):
                continue

            t_start = row['event_start_time']
            t_end   = row['event_end_time']

            st_cut = st.copy().trim(
                starttime=UTCDateTime(str(t_start)) - 10,
                endtime=UTCDateTime(str(t_end)) + 10
            )
            if len(st_cut) == 0:
                continue

            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

            p_time = row['p_arrival_time']
            s_time = row['s_arrival_time']
            p_prob = row['p_probability']
            s_prob = row['s_probability']
            p_snr  = row['p_snr']
            s_snr  = row['s_snr']

            fig.suptitle(
                f"Event #{i+1:03d} | "
                f"{station_info['network']}.{station_info['station']} | "
                f"{date_str}\n"
                f"Start: {t_start} | End: {t_end}\n"
                f"P: {p_time} (prob={p_prob}, SNR={p_snr}) | "
                f"S: {s_time} (prob={s_prob}, SNR={s_snr})",
                fontsize=9, fontweight='bold'
            )

            comp_info = [
                ('Z', 'black',   f"{station_info['instrument_type']}Z"),
                ('N', 'navy',    f"{station_info['instrument_type']}N"),
                ('E', 'darkred', f"{station_info['instrument_type']}E"),
            ]

            for idx, (comp, color, label) in enumerate(comp_info):
                tr_list = [tr for tr in st_cut
                           if tr.stats.channel[-1].upper() == comp]
                if tr_list:
                    tr     = tr_list[0]
                    t_plot = tr.times("matplotlib")
                    axes[idx].plot(t_plot, tr.data, color=color,
                                   linewidth=0.8)
                    axes[idx].set_ylabel(label, fontsize=8)
                    axes[idx].grid(True, alpha=0.3)

                    if pd.notna(p_time):
                        try:
                            pt = mdates.date2num(
                                pd.Timestamp(str(p_time)).to_pydatetime()
                            )
                            axes[idx].axvline(
                                x=pt, color='blue', linewidth=2,
                                linestyle='--',
                                label=f"P (prob={p_prob:.2f}, SNR={p_snr})"
                                if idx == 0 else ""
                            )
                        except Exception:
                            pass

                    if pd.notna(s_time):
                        try:
                            st2 = mdates.date2num(
                                pd.Timestamp(str(s_time)).to_pydatetime()
                            )
                            axes[idx].axvline(
                                x=st2, color='red', linewidth=2,
                                linestyle='--',
                                label=f"S (prob={s_prob:.2f}, SNR={s_snr})"
                                if idx == 0 else ""
                            )
                        except Exception:
                            pass

                    if idx == 0:
                        handles, labels = axes[idx].get_legend_handles_labels()
                        if handles:
                            axes[idx].legend(loc='upper right', fontsize=8)

                axes[idx].xaxis.set_major_formatter(
                    mdates.DateFormatter('%H:%M:%S')
                )

            axes[-1].set_xlabel('UTC time', fontsize=9)
            plt.tight_layout()

            fname = os.path.join(
                folders['events'],
                f"event_{i+1:03d}_"
                f"{station_info['network']}."
                f"{station_info['station']}_{date_str}.{cfg['plot_format']}"
            )
            plt.savefig(fname, dpi=cfg['plot_dpi'], bbox_inches='tight')
            plt.close()

        except Exception as e:
            plt.close()
            print(f"    ⚠️  Event #{i+1} failed: {e}")
            continue


def plot_summary(result_df, station_info, date_str, folders, cfg):
    if result_df.empty:
        return

    try:
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(
            f"Summary | {station_info['network']}.{station_info['station']} | "
            f"{date_str} | EQT-{cfg['model_name']}",
            fontsize=13, fontweight='bold'
        )

        df = result_df.copy()
        # errors='coerce' -> rows without an event (None/NaN, e.g. an orphan
        # P-pick with no detection) become NaT instead of raising on "None"
        df['event_start_dt'] = pd.to_datetime(df['event_start_time'], errors='coerce')
        df['hour']           = df['event_start_dt'].dt.hour

        ax1 = fig.add_subplot(3, 3, 1)
        ax1.hist(df['hour'].dropna(), bins=24, range=(0, 24),
                 color='green', alpha=0.7, edgecolor='white')
        ax1.set_xlabel('UTC hour', fontsize=9)
        ax1.set_ylabel('Count', fontsize=9)
        ax1.set_title(f'Detections per hour (n={len(df)})', fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(3, 3, 2)
        p_probs = df['p_probability'].dropna()
        if not p_probs.empty:
            ax2.hist(p_probs, bins=20, range=(0, 1),
                     color='blue', alpha=0.7, edgecolor='white')
            ax2.axvline(x=cfg['P_threshold'], color='k',
                        linestyle='--', linewidth=1,
                        label=f"thr={cfg['P_threshold']}")
            ax2.legend(fontsize=8)
        ax2.set_xlabel('P Probability', fontsize=9)
        ax2.set_ylabel('Count', fontsize=9)
        ax2.set_title('P probability distribution', fontsize=9)
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(3, 3, 3)
        s_probs = df['s_probability'].dropna()
        if not s_probs.empty:
            ax3.hist(s_probs, bins=20, range=(0, 1),
                     color='red', alpha=0.7, edgecolor='white')
            ax3.axvline(x=cfg['S_threshold'], color='k',
                        linestyle='--', linewidth=1,
                        label=f"thr={cfg['S_threshold']}")
            ax3.legend(fontsize=8)
        ax3.set_xlabel('S Probability', fontsize=9)
        ax3.set_ylabel('Count', fontsize=9)
        ax3.set_title('S probability distribution', fontsize=9)
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(3, 3, 4)
        p_snr = df['p_snr'].dropna()
        if not p_snr.empty:
            ax4.hist(p_snr, bins=20, color='blue', alpha=0.7, edgecolor='white')
        ax4.set_xlabel('P SNR', fontsize=9)
        ax4.set_ylabel('Count', fontsize=9)
        ax4.set_title('P SNR distribution', fontsize=9)
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(3, 3, 5)
        s_snr = df['s_snr'].dropna()
        if not s_snr.empty:
            ax5.hist(s_snr, bins=20, color='red', alpha=0.7, edgecolor='white')
        ax5.set_xlabel('S SNR', fontsize=9)
        ax5.set_ylabel('Count', fontsize=9)
        ax5.set_title('S SNR distribution', fontsize=9)
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(3, 3, 6)
        det_probs = df['detection_probability'].dropna()
        if not det_probs.empty:
            ax6.hist(det_probs, bins=20, range=(0, 1),
                     color='green', alpha=0.7, edgecolor='white')
            ax6.axvline(x=cfg['detection_threshold'], color='k',
                        linestyle='--', linewidth=1,
                        label=f"thr={cfg['detection_threshold']}")
            ax6.legend(fontsize=8)
        ax6.set_xlabel('Detection Probability', fontsize=9)
        ax6.set_ylabel('Count', fontsize=9)
        ax6.set_title('Detection probability distribution', fontsize=9)
        ax6.grid(True, alpha=0.3)

        ax7 = fig.add_subplot(3, 3, 7)
        valid = df[df['p_probability'].notna() & df['s_probability'].notna()]
        if not valid.empty:
            sc = ax7.scatter(
                valid['p_probability'], valid['s_probability'],
                c=valid['detection_probability'], cmap='viridis',
                s=30, alpha=0.7
            )
            plt.colorbar(sc, ax=ax7, label='Det. Prob')
        ax7.set_xlabel('P Probability', fontsize=9)
        ax7.set_ylabel('S Probability', fontsize=9)
        ax7.set_title('P prob vs S prob', fontsize=9)
        ax7.grid(True, alpha=0.3)

        ax8 = fig.add_subplot(3, 3, (8, 9))
        valid_t = df[df['event_start_dt'].notna()]
        if not valid_t.empty:
            ax8.scatter(
                valid_t['event_start_dt'],
                valid_t['detection_probability'],
                c='green', s=20, alpha=0.6, label='Detection', zorder=3
            )
            p_valid = df[df['p_probability'].notna() &
                         df['event_start_dt'].notna()]
            s_valid = df[df['s_probability'].notna() &
                         df['event_start_dt'].notna()]
            if not p_valid.empty:
                ax8.scatter(
                    p_valid['event_start_dt'],
                    p_valid['p_probability'],
                    c='blue', s=15, alpha=0.5, marker='^', label='P prob'
                )
            if not s_valid.empty:
                ax8.scatter(
                    s_valid['event_start_dt'],
                    s_valid['s_probability'],
                    c='red', s=15, alpha=0.5, marker='v', label='S prob'
                )
        ax8.set_xlabel('UTC time', fontsize=9)
        ax8.set_ylabel('Probability', fontsize=9)
        ax8.set_title('Probability timeline', fontsize=9)
        ax8.legend(fontsize=7)
        ax8.grid(True, alpha=0.3)
        ax8.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax8.xaxis.set_major_locator(mdates.HourLocator(interval=2))

        plt.tight_layout()
        fname = os.path.join(
            folders['summary'],
            f"summary_{station_info['network']}."
            f"{station_info['station']}_{date_str}.{cfg['plot_format']}"
        )
        plt.savefig(fname, dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.close()

    except Exception as e:
        plt.close()
        print(f"    ⚠️  Summary failed: {e}")


# ------------------------------------------------------------
# 3.4 Main pipeline
# ------------------------------------------------------------

def main():
    print("=" * 60)
    print("  EQTransformer Picking Pipeline")
    print(f"  Model    : {CONFIG['model_name']}")
    print(f"  SeisBench: {seisbench.__version__}")
    print("=" * 60)

    # --- Set the number of CPU threads for torch (model compute) ---
    if CONFIG.get('num_threads'):
        torch.set_num_threads(CONFIG['num_threads'])
        print(f"🧵 torch.set_num_threads({CONFIG['num_threads']})")

    folders       = setup_output_folders(CONFIG['output_folder'])
    stations_meta = load_stations_metadata(CONFIG['stations_file'], CONFIG['network'])
    print(f"\n📍 Station metadata loaded: {len(stations_meta)} stations")

    checkpoint = load_checkpoint(CONFIG['checkpoint_file'])
    print(f"🗂️  Checkpoint loaded: {len(checkpoint)} files already processed")

    combined_csv_path = os.path.join(folders['summary'], CONFIG['combined_csv_name'])

    day_folders = get_day_folders(CONFIG['input_folder'], CONFIG['day_folder_pattern'])
    if not day_folders:
        print(f"❌ No day folders found in: {CONFIG['input_folder']}")
        return

    print(f"\n📂 Found {len(day_folders)} day folders")
    if os.path.exists(combined_csv_path):
        n_existing = sum(1 for _ in open(combined_csv_path)) - 1  # minus header
        print(f"📄 Combined CSV already exists: {combined_csv_path} ({n_existing} rows) -> will continue")

    print(f"\n⏳ Loading model EQT-{CONFIG['model_name']}...")
    model = sbm.EQTransformer.from_pretrained(CONFIG['model_name'])
    print(f"   Component order : {model.component_order}")
    print(f"   Sampling rate   : {model.sampling_rate} Hz")

    n_skipped_ckp = 0
    n_skipped_sta = 0
    n_error       = 0

    day_bar = tqdm(day_folders, desc="Days", unit="day", position=0)
    for day_folder in day_bar:
        day_str = os.path.basename(day_folder)
        day_bar.set_postfix_str(day_str)

        mseed_files = sorted(glob.glob(os.path.join(day_folder, CONFIG['file_pattern'])))
        if not mseed_files:
            continue

        file_bar = tqdm(mseed_files, desc=f"  {day_str}", unit="file",
                        position=1, leave=False)

        for filepath in file_bar:
            filename       = os.path.basename(filepath)
            checkpoint_key = f"{day_str}/{filename}"
            file_bar.set_postfix_str(filename)

            # --- Skip if already processed (checkpoint) ---
            if checkpoint_key in checkpoint:
                n_skipped_ckp += 1
                continue

            try:
                t0 = time.time()

                def log_step(msg):
                    if CONFIG.get('verbose_timing'):
                        tqdm.write(f"    [{time.strftime('%H:%M:%S')}] "
                                   f"(+{time.time() - t0:5.1f}s) {checkpoint_key}: {msg}")

                network_file, station_code, date_from_file = parse_mseed_filename(filepath)
                date_str = date_from_file or day_str

                log_step("reading mseed...")
                st = read(filepath)

                gap_problems = check_excessive_gaps(st, CONFIG.get('max_gap_seconds'))
                if gap_problems:
                    log_step(f"⚠️  extreme gap detected {gap_problems} "
                             f"(>{CONFIG['max_gap_seconds']}s) -> file skipped, not merged/interpolated")
                    checkpoint.add(checkpoint_key)
                    save_checkpoint(CONFIG['checkpoint_file'], checkpoint)
                    continue

                log_step(f"merging stream ({len(st)} traces)...")
                st.merge(method=1, fill_value='interpolate')

                missing = check_components(st)
                if missing:
                    # Incomplete components -> mark as done (will not be retried)
                    log_step(f"missing components {missing} -> skip")
                    checkpoint.add(checkpoint_key)
                    save_checkpoint(CONFIG['checkpoint_file'], checkpoint)
                    continue

                instrument_type = st[0].stats.channel[:2]
                station_info = build_station_info(
                    network_file, station_code, stations_meta, instrument_type
                )
                if station_info is None:
                    # Station absent from stations.txt -> skip and mark as done
                    log_step("station not in stations.txt -> skip")
                    n_skipped_sta += 1
                    checkpoint.add(checkpoint_key)
                    save_checkpoint(CONFIG['checkpoint_file'], checkpoint)
                    continue

                station_label = f"{station_info['network']}.{station_info['station']}"

                # Dedicated folder <output>/<day>/<NET.STA>/... for this file
                station_folders = setup_station_day_folders(
                    CONFIG['output_folder'], day_str, station_label
                )

                model_kwargs = build_model_call_kwargs(CONFIG)

                log_step("model.annotate() started (usually the slowest stage)...")
                annotations = model.annotate(st, **model_kwargs)
                log_step("model.annotate() done, model.classify() started...")
                output = model.classify(
                    st,
                    P_threshold=CONFIG['P_threshold'],
                    S_threshold=CONFIG['S_threshold'],
                    detection_threshold=CONFIG['detection_threshold'],
                    overlap=CONFIG['overlap'],
                    **model_kwargs,
                )
                log_step(f"model.classify() done "
                         f"({len(output.picks)} picks, {len(output.detections)} detections)")

                result_df = build_stead_style_csv(
                    output.picks, output.detections, st, station_info, date_str
                )

                result_df = apply_quality_filters(result_df, CONFIG)

                # --- Per station-day CSV ---
                csv_path = os.path.join(
                    station_folders['csv'],
                    f"results_{station_label}_{date_str}.csv"
                )
                result_df.to_csv(csv_path, index=False)

                # --- Append to the combined CSV in summary/ ---
                # (resumable: appended only, never rewritten from scratch)
                append_to_combined_csv(combined_csv_path, result_df)

                # --- Per station-day plots ---
                log_step(f"plotting ({len(result_df)} result rows)...")
                if CONFIG['plot_overview']:
                    plot_overview(st, annotations, result_df,
                                  station_info, date_str, station_folders, CONFIG)
                if CONFIG['plot_per_event']:
                    plot_per_event(st, result_df, station_info,
                                   date_str, station_folders, CONFIG)
                if CONFIG['plot_summary']:
                    plot_summary(result_df, station_info,
                                 date_str, station_folders, CONFIG)
                log_step("done (file complete)")

                # --- Mark this file as fully processed ---
                checkpoint.add(checkpoint_key)
                save_checkpoint(CONFIG['checkpoint_file'], checkpoint)

            except Exception as e:
                n_error += 1
                tqdm.write(f"  ❌ Error on {checkpoint_key}: {e}")
                # Deliberately NOT recorded in the checkpoint so it is retried
                # on the next run.
                continue

        file_bar.close()

    day_bar.close()

    # --- Combined CSV info (appended throughout the run) ---
    if os.path.exists(combined_csv_path):
        n_total_rows = sum(1 for _ in open(combined_csv_path)) - 1  # minus header
        print(f"\n✅ Combined CSV → {combined_csv_path}")
        print(f"   Total rows : {n_total_rows}")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Newly processed files            : {len(checkpoint) - n_skipped_ckp}")
    print(f"  Skipped (checkpoint)             : {n_skipped_ckp}")
    print(f"  Skipped (unknown station)        : {n_skipped_sta}")
    print(f"  Errored (retried on next run)    : {n_error}")
    print(f"  Total recorded in checkpoint     : {len(checkpoint)}")
    print("\n✅ PIPELINE COMPLETE!")


if __name__ == "__main__":
    main()