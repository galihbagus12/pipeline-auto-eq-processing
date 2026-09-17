# ============================================================
# Local Magnitude (ML) Pipeline
#
# Stage 4 of the workflow:
#   1) pick.py         -> phase picking (EQTransformer / SeisBench)
#   2) association.py  -> phase association + quality control
#   3) locator.py       -> hypocentre location (NonLinLoc)
#   4) magnitud.py      -> local magnitude, Wood-Anderson amplitude (this file)
#
# Ported from the reference notebook mag_filter.ipynb, with two inputs
# swapped for this project's actual pipeline outputs:
#   - event/origin data + phase picks now come from the relocation stage
#     (output/relocation/hypocenter_relocated_summary.csv + phase.dat)
#     instead of a standalone phase_fixed.dat
#   - waveforms are read from output/preprocessing/<YYYY-MM-DD>/<NET>.<STA>..<date>.mseed
#     (one folder per day, per pick.py's convention) instead of one folder
#     per station
#
# Method: Wood-Anderson amplitude from the horizontal-component response,
# local peak-to-peak measured near the S pick, ML via the Hutton & Boore
# (1987) formula. One station magnitude per (event, station) pair, then the
# event ML is the median over stations with status == OK and SNR >= min_snr.
#
# Input  : output/relocation/hypocenter_relocated_summary.csv (relocated events)
#          output/relocation/phase.dat                        (P/S picks, hypoDD format)
#          stations.txt                                      (station coordinates)
#          output/preprocessing/<date>/<NET>.<STA>..<date>.mseed (waveforms)
#
# Also renders the deliverable event/magnitude map (PyGMT) as the final step
# of main() -- relocated hypocentres coloured by depth, sized by ML, over
# topography, with stations, cross-section index lines and legends. No
# separate script/run needed: running this file produces the map too.
#
# Output structure :
#   output_folder/csv/ml_station_magnitudes.csv   one row per (event, station)
#   output_folder/csv/ml_event_magnitudes.csv     one row per event
#   output_folder/summary/                        diagnostic plots + event_magnitude_map.png
#   xml_cache_dir/                                 cached instrument responses
#
# NOTE: This file is deliberately split into:
#   1) MODULE IMPORTS
#   2) CONFIGURATION (loaded from config/config.yaml -- see CONFIG_YAML_PATH)
#   3) PROCESS (loading, waveform I/O, magnitude, map figure, main pipeline)
# CONFIG is loaded from the 'magnitud' section of the shared config/config.yaml
# (see pipeline.py at the project root, which runs every stage from that same
# file) -- edit the YAML to change parameters, not this file; the CONFIG dict
# keys and every function signature below are unchanged, so PROCESS never
# needs to be touched for a config-only change.
# ============================================================

# ============================================================
# 1) MODULE IMPORTS
# ============================================================

import json
import math
import os
import tempfile
import warnings
from datetime import timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from obspy import Stream, UTCDateTime, read, read_inventory
from obspy.clients.fdsn import Client
from obspy.geodetics import degrees2kilometers, locations2degrees
from obspy.signal.invsim import estimate_wood_anderson_amplitude_using_response

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Package 'tqdm' is not installed. Run: pip install tqdm")

warnings.filterwarnings('ignore')


# ============================================================
# 2) CONFIGURATION
# ============================================================

CONFIG_YAML_PATH = str(Path(__file__).resolve().parent.parent / "config" / "config.yaml")

with open(CONFIG_YAML_PATH, 'r') as _f:
    _all_cfg = yaml.safe_load(_f)

CONFIG = _all_cfg['magnitud']
CONFIG['network'] = _all_cfg['network']   # single global network code (see stations.txt)


# ============================================================
# 3) PROCESS
# ============================================================

# ------------------------------------------------------------
# 3.1 Loading: stations, relocated events, phase picks
# ------------------------------------------------------------

def load_stations(stations_file, network):
    """stations.txt (columns: station latitude longitude elevation_m) ->
    DataFrame with columns: station, network, latitude, longitude,
    elevation_m. network is a single global constant (config.yaml's
    top-level 'network' key) since stations.txt carries no per-station
    network code."""
    rows = []
    with open(stations_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            sta, lat, lon, elev = line.split()
            rows.append({
                'station'     : sta,
                'network'     : network,
                'latitude'    : float(lat),
                'longitude'   : float(lon),
                'elevation_m' : float(elev),
            })
    return pd.DataFrame(rows)


def load_relocated_events(reloc_summary_csv):
    reloc = pd.read_csv(reloc_summary_csv)
    events = reloc.rename(columns={
        'cuspid'         : 'event_id',
        'lat_rel'        : 'event_lat',
        'lon_rel'        : 'event_lon',
        'dep_rel'        : 'event_depth_km',
        'reloc_datetime' : 'origin_time',
    }).copy()
    events['event_id'] = events['event_id'].astype(int)
    events['origin_time'] = pd.to_datetime(events['origin_time'], utc=True, errors='coerce')
    return events


def parse_phase_dat(phase_dat):
    """Parse a hypoDD-format phase file (relocation.py's phase.dat): blocks
    starting with '# yyyy mm dd HH MM SS.ss lat lon dep ... cuspid' followed
    by 'STATION traveltime weight PHASE' lines."""
    rows = []
    current_event_id = None
    current_origin = None

    with open(phase_dat, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('#'):
                parts = line[1:].strip().split()
                if len(parts) < 14:
                    current_event_id = None
                    current_origin = None
                    continue

                yyyy, mm, dd = int(parts[0]), int(parts[1]), int(parts[2])
                HH, MM = int(parts[3]), int(parts[4])
                SS = float(parts[5])
                sec = int(SS)
                usec = int(round((SS - sec) * 1_000_000))
                if usec == 1_000_000:
                    sec += 1
                    usec = 0

                current_origin = pd.Timestamp(
                    year=yyyy, month=mm, day=dd, hour=HH, minute=MM,
                    second=sec, microsecond=usec, tz='UTC',
                )
                current_event_id = int(parts[-1])
                continue

            parts = line.split()
            if current_event_id is None or current_origin is None or len(parts) < 4:
                continue

            station = parts[0].strip()
            tt = float(parts[1])
            weight = float(parts[2])
            phase = parts[3].strip().upper()
            if phase not in ('P', 'S'):
                continue

            rows.append({
                'event_id'  : current_event_id,
                'station'   : station,
                'phase'     : phase,
                'weight'    : weight,
                'pick_time' : current_origin + pd.to_timedelta(tt, unit='s'),
            })

    return pd.DataFrame(rows)


def build_event_station_table(events, phase_df, station_df, require_s_pick):
    phase_df = phase_df.dropna(subset=['event_id', 'station', 'phase', 'pick_time']).copy()
    phase_df['event_id'] = phase_df['event_id'].astype(int)

    event_ids_final = set(events['event_id'].dropna().astype(int))
    phase_df = phase_df[phase_df['event_id'].isin(event_ids_final)].copy()

    picks = (
        phase_df.sort_values('pick_time')
        .pivot_table(index=['event_id', 'station'], columns='phase', values='pick_time', aggfunc='min')
        .reset_index()
    )
    picks = picks.rename(columns={'P': 'pick_P', 'S': 'pick_S'})
    if 'pick_P' not in picks.columns:
        picks['pick_P'] = pd.NaT
    if 'pick_S' not in picks.columns:
        picks['pick_S'] = pd.NaT

    event_station = picks.merge(
        events[['event_id', 'origin_time', 'event_lat', 'event_lon', 'event_depth_km']],
        on='event_id', how='left',
    )
    event_station = event_station.merge(
        station_df[['station', 'network', 'latitude', 'longitude', 'elevation_m']],
        on='station', how='left',
    )
    event_station = event_station.rename(columns={
        'latitude'    : 'station_lat',
        'longitude'   : 'station_lon',
        'elevation_m' : 'station_elevation_m',
    })

    missing_station = event_station[event_station['station_lat'].isna()]['station'].unique()
    if len(missing_station) > 0:
        print('WARNING: station not found in stations.txt:', missing_station)

    event_station = event_station.dropna(
        subset=['event_lat', 'event_lon', 'event_depth_km', 'station_lat', 'station_lon']
    )
    if require_s_pick:
        event_station = event_station.dropna(subset=['pick_S'])

    return event_station.reset_index(drop=True)


# ------------------------------------------------------------
# 3.2 Distance and magnitude formula
# ------------------------------------------------------------

def to_utcdatetime(x) -> UTCDateTime:
    if pd.isna(x):
        raise ValueError('Time is NaT/NaN')
    return UTCDateTime(pd.Timestamp(x).to_pydatetime())


def hypocentral_distance_km(ev_lat, ev_lon, ev_depth_km, sta_lat, sta_lon, sta_elev_m, min_distance_km):
    deg = locations2degrees(ev_lat, ev_lon, sta_lat, sta_lon)
    horizontal_km = degrees2kilometers(deg)
    vertical_km = float(ev_depth_km) + float(sta_elev_m) / 1000.0
    R = math.sqrt(horizontal_km ** 2 + vertical_km ** 2)
    return max(R, min_distance_km), horizontal_km, vertical_km


def ml_hutton_boore(A_mm, R_km):
    """Hutton & Boore (1987) local magnitude formula."""
    if A_mm <= 0 or R_km <= 0:
        return np.nan
    return np.log10(A_mm) + 1.11 * np.log10(R_km / 100.0) + 0.00189 * (R_km - 100.0) + 3.0


def calculate_ml(A_mm, R_km, ml_formula):
    if ml_formula == 'hutton_boore':
        return ml_hutton_boore(A_mm, R_km)
    raise ValueError(f"Unknown ML_FORMULA: {ml_formula}")


# ------------------------------------------------------------
# 3.3 Waveform I/O
# ------------------------------------------------------------

@lru_cache(maxsize=8)
def _read_mseed_cached(path_str):
    return read(path_str)


def date_strings_between(t1: UTCDateTime, t2: UTCDateTime):
    d0, d1 = t1.datetime.date(), t2.datetime.date()
    out, d = [], d0
    while d <= d1:
        out.append(d.strftime('%Y-%m-%d'))
        d += timedelta(days=1)
    return out


def find_mseed_files(waveform_folder, network, station, t1, t2):
    """filtered/<YYYY-MM-DD>/<NET>.<STA>..<YYYY-MM-DD>.mseed (pick.py's convention:
    one folder per day, not one folder per station)."""
    files = []
    for dstr in date_strings_between(t1, t2):
        day_dir = Path(waveform_folder) / dstr
        if not day_dir.exists():
            continue
        exact = day_dir / f'{network}.{station}..{dstr}.mseed'
        if exact.exists():
            files.append(exact)
            continue
        matches = sorted(day_dir.glob(f'*{station}*.mseed'))
        files.extend(matches)

    uniq, seen = [], set()
    for f in files:
        if str(f) not in seen:
            uniq.append(f)
            seen.add(str(f))
    return uniq


def read_station_window(waveform_folder, network, station, t1, t2):
    files = find_mseed_files(waveform_folder, network, station, t1, t2)
    if not files:
        raise FileNotFoundError(f'No miniSEED found for {station} between {t1} and {t2}')
    st = Stream()
    for f in files:
        st += _read_mseed_cached(str(f)).copy()
    st.merge(method=1, fill_value='interpolate')
    st.trim(starttime=t1, endtime=t2, pad=False)
    return st


def get_station_inventory(xml_cache_dir, network, auspass_fdsn_url, channel_pattern, station, time):
    xml_path = Path(xml_cache_dir) / f'{network}_{station}_response.xml'
    if xml_path.exists():
        return read_inventory(str(xml_path))
    client = Client(auspass_fdsn_url)
    inv = client.get_stations(
        network=network, station=station, channel=channel_pattern,
        starttime=time - 3600, endtime=time + 3600, level='response',
    )
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    inv.write(str(xml_path), format='STATIONXML')
    return inv


# ------------------------------------------------------------
# 3.4 Amplitude measurement (Wood-Anderson)
# ------------------------------------------------------------

def preprocess_counts_trace(tr, freqmin, freqmax, corners):
    tr2 = tr.copy()
    tr2.detrend('demean')
    tr2.detrend('linear')
    tr2.taper(max_percentage=0.05)
    tr2.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=corners, zerophase=True)
    return tr2


def local_peak_to_peak(data, fs, max_timespan_s):
    data = np.asarray(data, dtype=float)
    if len(data) < 5 or not np.isfinite(data).all():
        return None

    diff = np.diff(data)
    sign = np.sign(diff)
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    for i in range(len(sign) - 2, -1, -1):
        if sign[i] == 0:
            sign[i] = sign[i + 1]

    maxima = np.where((sign[:-1] > 0) & (sign[1:] < 0))[0] + 1
    minima = np.where((sign[:-1] < 0) & (sign[1:] > 0))[0] + 1
    if len(maxima) == 0 or len(minima) == 0:
        return None

    extrema = np.sort(np.concatenate([maxima, minima]))
    peak_idx = extrema[np.argmax(np.abs(data[extrema]))]
    opposite = minima if data[peak_idx] >= 0 else maxima
    if len(opposite) == 0:
        return None

    dt = np.abs(opposite - peak_idx) / fs
    candidates = opposite[dt <= max_timespan_s]
    if len(candidates) == 0:
        return None

    opp_idx = candidates[np.argmax(np.abs(data[candidates] - data[peak_idx]))]
    amp_pp = abs(data[peak_idx] - data[opp_idx])
    timespan = abs(peak_idx - opp_idx) / fs
    if amp_pp <= 0 or timespan <= 0 or timespan > max_timespan_s:
        return None

    return {
        'amp_pp_counts' : float(amp_pp),
        'timespan_s'    : float(timespan),
        'peak_idx'      : int(peak_idx),
        'opposite_idx'  : int(opp_idx),
    }


def rms(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.nan
    return float(np.sqrt(np.mean(x ** 2)))


def compute_wa_component(tr, inv, amp_t1, amp_t2, noise_t1, noise_t2, cfg):
    tr_amp = tr.copy().trim(amp_t1, amp_t2, pad=False)
    if len(tr_amp.data) < 10:
        raise ValueError('Amplitude window too short')

    tr_amp_f = preprocess_counts_trace(
        tr_amp, cfg['filter_freqmin'], cfg['filter_freqmax'], cfg['filter_corners']
    )
    fs = float(tr_amp_f.stats.sampling_rate)

    p2p = local_peak_to_peak(tr_amp_f.data, fs, cfg['max_p2p_timespan_s'])
    if p2p is None:
        raise ValueError('No valid local peak-to-peak found')

    response = inv.get_response(tr.id, tr.stats.starttime)
    wa_amp_mm = estimate_wood_anderson_amplitude_using_response(
        response=response, amplitude=p2p['amp_pp_counts'], timespan=p2p['timespan_s']
    )
    signal_peak = float(np.max(np.abs(tr_amp_f.data)))

    noise_rms, snr = np.nan, np.nan
    if noise_t1 is not None and noise_t2 is not None and noise_t2 > noise_t1:
        tr_noise = tr.copy().trim(noise_t1, noise_t2, pad=False)
        if len(tr_noise.data) > 10:
            try:
                tr_noise_f = preprocess_counts_trace(
                    tr_noise, cfg['filter_freqmin'], cfg['filter_freqmax'], cfg['filter_corners']
                )
                noise_rms = rms(tr_noise_f.data)
                if noise_rms > 0:
                    snr = signal_peak / noise_rms
            except Exception:
                pass

    return {
        'trace_id'         : tr.id,
        'channel'          : tr.stats.channel,
        'wa_amp_mm'        : float(wa_amp_mm),
        'amp_pp_counts'    : p2p['amp_pp_counts'],
        'timespan_s'       : p2p['timespan_s'],
        'signal_peak_counts': signal_peak,
        'noise_rms_counts' : noise_rms,
        'snr'              : snr,
    }


# ------------------------------------------------------------
# 3.5 Per (event, station) processing
# ------------------------------------------------------------

def process_event_station(row, cfg):
    out = {
        'event_id'            : int(row['event_id']),
        'station'             : row['station'],
        'origin_time'         : row['origin_time'],
        'event_lat'           : row['event_lat'],
        'event_lon'           : row['event_lon'],
        'event_depth_km'      : row['event_depth_km'],
        'station_lat'         : row['station_lat'],
        'station_lon'         : row['station_lon'],
        'station_elevation_m' : row['station_elevation_m'],
        'status'              : 'INIT',
        'error'               : '',
    }

    try:
        station = row['station']
        origin_time = to_utcdatetime(row['origin_time'])
        pick_P = to_utcdatetime(row['pick_P']) if pd.notna(row['pick_P']) else None
        pick_S = to_utcdatetime(row['pick_S']) if pd.notna(row['pick_S']) else None

        out['pick_P'] = pick_P.isoformat() if pick_P is not None else None
        out['pick_S'] = pick_S.isoformat() if pick_S is not None else None

        if cfg['require_s_pick'] and pick_S is None:
            out['status'] = 'SKIP_NO_S_PICK'
            return out

        if pick_S is not None:
            amp_t1 = pick_S - cfg['amp_pre_s']
            amp_t2 = pick_S + cfg['amp_after_s']
            out['amp_window_ref'] = 'S'
        elif pick_P is not None:
            amp_t1 = pick_P
            amp_t2 = pick_P + cfg['amp_after_p_only']
            out['amp_window_ref'] = 'P'
        else:
            out['status'] = 'SKIP_NO_PICK'
            return out

        if pick_P is not None:
            noise_t1 = pick_P - cfg['noise_before_p1']
            noise_t2 = pick_P - cfg['noise_before_p2']
        else:
            noise_t1 = origin_time - 15
            noise_t2 = origin_time - 5

        read_t1 = min(noise_t1, amp_t1) - 2
        read_t2 = amp_t2 + 2

        out['amp_t1'] = amp_t1.isoformat()
        out['amp_t2'] = amp_t2.isoformat()
        out['noise_t1'] = noise_t1.isoformat()
        out['noise_t2'] = noise_t2.isoformat()

        R_km, Rh_km, Rv_km = hypocentral_distance_km(
            row['event_lat'], row['event_lon'], row['event_depth_km'],
            row['station_lat'], row['station_lon'], row['station_elevation_m'],
            cfg['min_distance_km'],
        )
        out['R_km'], out['Rh_km'], out['Rv_km'] = R_km, Rh_km, Rv_km

        st = read_station_window(cfg['waveform_folder'], cfg['network'], station, read_t1, read_t2)
        inv = get_station_inventory(
            cfg['xml_cache_dir'], cfg['network'], cfg['auspass_fdsn_url'],
            cfg['channel_pattern'], station, origin_time,
        )

        st_h = Stream()
        for ch in cfg['horizontal_channels']:
            st_h += st.select(network=cfg['network'], station=station, channel=ch)

        if len(st_h) == 0:
            out['status'] = 'FAIL_NO_HORIZONTAL_TRACE'
            return out

        comps = []
        for tr in st_h:
            try:
                comps.append(compute_wa_component(tr, inv, amp_t1, amp_t2, noise_t1, noise_t2, cfg))
            except Exception as e:
                comps.append({
                    'trace_id': tr.id, 'channel': tr.stats.channel,
                    'wa_amp_mm': np.nan, 'amp_pp_counts': np.nan, 'timespan_s': np.nan,
                    'signal_peak_counts': np.nan, 'noise_rms_counts': np.nan, 'snr': np.nan,
                    'component_error': f'{type(e).__name__}: {e}',
                })

        for comp in comps:
            ch = comp['channel']
            out[f'A_WA_mm_{ch}'] = comp.get('wa_amp_mm', np.nan)
            out[f'p2p_counts_{ch}'] = comp.get('amp_pp_counts', np.nan)
            out[f'timespan_s_{ch}'] = comp.get('timespan_s', np.nan)
            out[f'snr_{ch}'] = comp.get('snr', np.nan)
            if 'component_error' in comp:
                out[f'error_{ch}'] = comp['component_error']

        valid_comps = [c for c in comps if np.isfinite(c.get('wa_amp_mm', np.nan)) and c.get('wa_amp_mm', 0) > 0]
        if len(valid_comps) == 0:
            out['status'] = 'FAIL_NO_VALID_COMPONENT'
            return out

        best = max(valid_comps, key=lambda c: c['wa_amp_mm'])
        A_mm = best['wa_amp_mm']
        ML_station = calculate_ml(A_mm, R_km, cfg['ml_formula'])

        snr_values = [c.get('snr', np.nan) for c in valid_comps]
        snr_max = np.nanmax(snr_values) if len(snr_values) else np.nan

        out['best_channel'] = best['channel']
        out['A_WA_mm'] = A_mm
        out['ML_station'] = ML_station
        out['n_components_valid'] = len(valid_comps)
        out['snr_max'] = snr_max
        out['valid_snr'] = bool(np.isfinite(snr_max) and snr_max >= cfg['min_snr'])
        out['status'] = 'OK'
        return out

    except Exception as e:
        out['status'] = 'FAIL'
        out['error'] = f'{type(e).__name__}: {e}'
        return out


# ------------------------------------------------------------
# 3.6 Event-level aggregation
# ------------------------------------------------------------

def aggregate_event_magnitudes(station_ml, events, cfg):
    valid = station_ml[station_ml['status'].eq('OK')].copy()
    valid = valid[np.isfinite(valid['ML_station'])]
    valid = valid[np.isfinite(valid['A_WA_mm']) & (valid['A_WA_mm'] > 0)]

    if cfg['use_snr_filter_for_event_ml'] and 'valid_snr' in valid.columns:
        valid = valid[valid['valid_snr'].eq(True)]

    if len(valid) == 0:
        return pd.DataFrame(columns=[
            'event_id', 'origin_time', 'event_lat', 'event_lon', 'event_depth_km',
            'ML_median', 'ML_mean', 'ML_std', 'ML_min', 'ML_max', 'n_station_ML',
            'A_WA_mm_max', 'R_km_min', 'R_km_median', 'snr_max_median',
        ])

    event_ml = (
        valid.groupby('event_id')
        .agg(
            origin_time=('origin_time', 'first'),
            event_lat=('event_lat', 'first'),
            event_lon=('event_lon', 'first'),
            event_depth_km=('event_depth_km', 'first'),
            ML_median=('ML_station', 'median'),
            ML_mean=('ML_station', 'mean'),
            ML_std=('ML_station', 'std'),
            ML_min=('ML_station', 'min'),
            ML_max=('ML_station', 'max'),
            n_station_ML=('station', 'nunique'),
            A_WA_mm_max=('A_WA_mm', 'max'),
            R_km_min=('R_km', 'min'),
            R_km_median=('R_km', 'median'),
            snr_max_median=('snr_max', 'median'),
        )
        .reset_index()
    )

    return event_ml.merge(events[['event_id']], on='event_id', how='right')


# ------------------------------------------------------------
# 3.7 Diagnostic plots (quick QC -- see section 3.8 for the deliverable map)
# ------------------------------------------------------------

def plot_summary(event_ml, summary_dir):
    event_ml = event_ml.dropna(subset=['ML_median'])
    if len(event_ml) == 0:
        print('No valid event ML -- skipping summary plots.')
        return

    os.makedirs(summary_dir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.hist(event_ml['ML_median'], bins=25)
    plt.xlabel('ML median')
    plt.ylabel('Number of events')
    plt.title('Local magnitude distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'ml_histogram.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(event_ml['n_station_ML'], event_ml['ML_median'], s=20)
    plt.xlabel('Number of stations used for ML')
    plt.ylabel('ML median')
    plt.title('ML vs. station count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'ml_vs_station_count.png'), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(event_ml['event_lon'], event_ml['event_lat'], c=event_ml['ML_median'], s=25)
    plt.colorbar(sc, label='ML median')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Event map coloured by ML (quick QC)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'ml_quick_map.png'), dpi=150)
    plt.close()


# ------------------------------------------------------------
# 3.8 Event/magnitude map (PyGMT)
#
# Renders the deliverable map: relocated hypocentres coloured by depth, sized
# by ML, over topography, with stations, cross-section index lines and a
# magnitude/depth legend. Ported from a reference PyGMT script for the same
# East Java region.
#
# Reuses the 'events' and 'event_ml' tables already built by main() -- no
# separate CSV re-read. FAULT_TRACES and the well marker are external data
# that cannot be derived from the network geometry, so they stay as
# explicit, flagged CONFIG values (map_plot_faults / map_plot_well).
# ------------------------------------------------------------

def load_station_coords(stations_file):
    """stations.txt (columns: station latitude longitude elevation_m) ->
    DataFrame with columns: station, longitude, latitude (map plotting only,
    no network/elevation needed)."""
    rows = []
    with open(stations_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            sta, lat, lon, _elev = line.split()
            rows.append({'station': sta, 'longitude': float(lon), 'latitude': float(lat)})
    return pd.DataFrame(rows)


def size_from_magnitude(magnitude, cfg):
    span = cfg['map_ml_ref_max'] - cfg['map_ml_ref_min']
    size = cfg['map_size_min_cm'] + (cfg['map_size_max_cm'] - cfg['map_size_min_cm']) * (
        (magnitude - cfg['map_ml_ref_min']) / span
    )
    return size.clip(lower=cfg['map_size_min_cm'], upper=cfg['map_size_max_cm'])


def build_map_events(events, event_ml, cfg):
    df = events[['event_id', 'event_lat', 'event_lon', 'event_depth_km']].merge(
        event_ml[['event_id', 'ML_median']], on='event_id', how='left'
    )
    df = df.rename(columns={'event_lat': 'lat_rel', 'event_lon': 'lon_rel', 'event_depth_km': 'dep_rel'})
    df = df.dropna(subset=['lon_rel', 'lat_rel', 'dep_rel']).copy()

    df['ML_plot'] = df['ML_median'].fillna(df['ML_median'].median())
    df['size_cm'] = size_from_magnitude(df['ML_plot'], cfg)
    return df


def load_isc_catalog(cfg, region):
    if not cfg['map_isc_csv']:
        return None
    if not os.path.exists(cfg['map_isc_csv']):
        print(f"WARNING: map_isc_csv configured but not found, skipping: {cfg['map_isc_csv']}")
        return None

    df_isc = pd.read_csv(cfg['map_isc_csv'])
    for col in ('magnitude', 'latitude', 'longitude', 'depth_km'):
        df_isc[col] = pd.to_numeric(df_isc[col], errors='coerce')
    df_isc = df_isc.dropna(subset=['latitude', 'longitude', 'magnitude', 'depth_km'])
    df_isc = df_isc[
        df_isc['longitude'].between(region[0], region[1]) & df_isc['latitude'].between(region[2], region[3])
    ].copy()
    df_isc['size_cm'] = size_from_magnitude(df_isc['magnitude'], cfg)
    return df_isc


def derive_region(station_df, map_events, padding_deg):
    """Bounding box of stations + relocated events, padded by padding_deg on
    every side -- keeps the map objective and self-adapting to the dataset,
    rather than a region hard-coded for one specific network."""
    lon_min = min(station_df['longitude'].min(), map_events['lon_rel'].min()) - padding_deg
    lon_max = max(station_df['longitude'].max(), map_events['lon_rel'].max()) + padding_deg
    lat_min = min(station_df['latitude'].min(), map_events['lat_rel'].min()) - padding_deg
    lat_max = max(station_df['latitude'].max(), map_events['lat_rel'].max()) + padding_deg
    return [round(lon_min, 2), round(lon_max, 2), round(lat_min, 2), round(lat_max, 2)]


def derive_cross_section_lines(region):
    """Two north-south and two east-west index lines at the 1/3 and 2/3
    points of the region -- an objective placeholder splitting the mapped
    volume into quadrants for later cross-sections, rather than fixed
    coordinates tied to one specific study area."""
    lon_min, lon_max, lat_min, lat_max = region
    lon_1_3 = lon_min + (lon_max - lon_min) / 3
    lon_2_3 = lon_min + 2 * (lon_max - lon_min) / 3
    lat_1_3 = lat_max - (lat_max - lat_min) / 3
    lat_2_3 = lat_max - 2 * (lat_max - lat_min) / 3
    return {
        'A': lon_1_3, 'D': lon_2_3,   # vertical (N-S) lines
        'B': lat_1_3, 'C': lat_2_3,   # horizontal (E-W) lines
    }


def draw_cross_section_lines(fig, region):
    lon_min, lon_max, lat_min, lat_max = region
    lines = derive_cross_section_lines(region)
    line_pen = '1.2p,black,--'
    font_idx = '14p,Helvetica-Bold,black'
    label_dy = (lat_max - lat_min) * 0.03

    for label, lon in (('A', lines['A']), ('D', lines['D'])):
        fig.plot(x=[lon, lon], y=[lat_min, lat_max], pen=line_pen)
        fig.text(x=lon + 0.1, y=lat_min + label_dy, text=label, font=font_idx, justify='CB', no_clip=True)
        fig.text(x=lon + 0.1, y=lat_max - label_dy, text=f"{label}'", font=font_idx, justify='CT', no_clip=True)

    for label, lat in (('B', lines['B']), ('C', lines['C'])):
        fig.plot(x=[lon_min, lon_max], y=[lat, lat], pen=line_pen)
        fig.text(x=lon_min + 0.15, y=lat + 0.1, text=label, font=font_idx, justify='RM', no_clip=True)
        fig.text(x=lon_max - 0.1, y=lat + 0.1, text=f"{label}'", font=font_idx, justify='LM', no_clip=True)


def nice_tick_interval(vmax, target_ticks=8):
    """Round vmax/target_ticks up to a 1/2/5-times-power-of-ten step, so the
    colorbar shows ~target_ticks annotations regardless of the data's depth
    range -- fixes label collision when a few outlier events (e.g. a poorly
    constrained 400+ km relocation) stretch the depth axis far beyond the
    bulk of the catalogue."""
    if vmax <= 0:
        return 1
    raw_step = vmax / target_ticks
    exponent = math.floor(math.log10(raw_step))
    for mult in (1, 2, 5, 10):
        step = mult * 10 ** exponent
        if step >= raw_step:
            return step
    return 10 * 10 ** exponent


def draw_colorbar_and_scale(fig, depth_max_event):
    major = nice_tick_interval(depth_max_event, target_ticks=8)
    minor = major / 2
    fig.colorbar(frame=[f'xa{major}f{minor}+lDepth (km)'], position='JBC+w15c/0.5c+h+o0c/1.5c')
    fig.basemap(map_scale='jTL+w50k+o0.8c/3.4c+f+lkm')
    fig.basemap(rose='jTL+w1.5c+o1.0c/0.5c+f+l')


def _write_legend_spec(lines):
    """PyGMT's legend() only accepts a spec FILE, not an in-memory string/list,
    so the spec is written to a scratch file and cleaned up by the caller."""
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    tmp.write('\n'.join(lines) + '\n')
    tmp.close()
    return tmp.name


def draw_symbol_legend(fig, cfg, region):
    """Symbol dictionary (stations, events, well, faults) as a GMT legend
    block -- auto-arranged by GMT itself, robust to region size/aspect
    (unlike hand-placed x/y coordinates, which collide once the map's degree
    span differs from what they were tuned for)."""
    lines = [
        'H 11p,Helvetica-Bold Legend',
        'S 0.3c t 0.35c black 0.5p 0.6c Station',
        'S 0.3c c 0.35c red 0.3p 0.6c Relocated event',
    ]
    if cfg.get('map_isc_csv'):
        lines.append('S 0.3c a 0.35c gray70 0.8p 0.6c ISC event (catalogue)')
    if cfg['map_plot_well']:
        lines.append(f"S 0.3c d 0.35c green 1p,darkgreen 0.6c {cfg['map_well_name']}")
    if cfg['map_plot_faults']:
        for fault in cfg['map_fault_traces']:
            lines.append(f"S 0.3c - 0.6c - 1.8p,{fault['color']} 0.6c {fault['name']}")

    spec_path = _write_legend_spec(lines)
    try:
        fig.legend(spec=spec_path, position='JTR+w5c+o0.2c/0.2c', box='+gwhite+p0.5p')
    finally:
        os.remove(spec_path)


def draw_magnitude_legend(fig, cfg, region):
    """Magnitude -> circle-size legend, same GMT legend mechanism as
    draw_symbol_legend(). Three bins spanning [map_ml_ref_min, map_ml_ref_max]
    (matching the reference script's three-tier legend), each sized from its
    own bin midpoint via size_from_magnitude() so it stays consistent with
    however the events themselves are sized."""
    lo, hi = cfg['map_ml_ref_min'], cfg['map_ml_ref_max']
    edges = [lo + (hi - lo) * f for f in (0.0, 1 / 3, 2 / 3, 1.0)]

    lines = ['H 10p,Helvetica-Bold Magnitude (ML)']
    for bin_lo, bin_hi in zip(edges[:-1], edges[1:]):
        size = float(size_from_magnitude(pd.Series([(bin_lo + bin_hi) / 2]), cfg).iloc[0])
        lines.append(f'S 0.3c c {size}c white 0.5p 0.6c {bin_lo:.1f} - {bin_hi:.1f}')

    spec_path = _write_legend_spec(lines)
    try:
        fig.legend(spec=spec_path, position='JBR+w4c+o0.2c/0.2c', box='+gwhite+p0.5p')
    finally:
        os.remove(spec_path)


def build_map_figure(cfg, station_df, map_events, df_isc, region):
    depth_max_event = map_events['dep_rel'].max()

    fig = pygmt.Figure()

    if cfg['map_basemap'] == 'relief':
        # Shaded topography -- needs a remote earth-relief tile the first
        # time it's requested for a given region (pygmt.datasets.load_earth_relief).
        # Requires GMT_DATA_SERVER access; falls back to 'coast' if unavailable.
        grid = pygmt.datasets.load_earth_relief(resolution='03s', region=region)
        pygmt.makecpt(cmap='geo', series=[0, 4000])
        fig.grdimage(
            grid=grid, projection='M25c', region=region, shading=True,
            frame=['xa1f0.5', 'ya1f0.5', 'WSne+tRelocated Hypocenters'],
        )
        fig.coast(shorelines='0.7p,black', water='lightblue', region=region)
    else:
        # Flat land/water fill, drawn entirely from GMT's bundled coastline
        # database -- no network access needed. Default, since GMT's remote
        # data server is disabled on this machine.
        fig.basemap(
            projection='M25c', region=region,
            frame=['xa1f0.5', 'ya1f0.5', 'WSne+tRelocated Hypocenters'],
        )
        fig.coast(
            shorelines='0.7p,black', water='lightblue', land=cfg['map_land_fill'],
            region=region, resolution=cfg['map_coast_resolution'],
        )

    if cfg['map_plot_faults']:
        for fault in cfg['map_fault_traces']:
            fig.plot(x=fault['lon'], y=fault['lat'], pen=f"1.8p,{fault['color']}")

    pygmt.makecpt(cmap='jet', series=[0, depth_max_event], reverse=True)
    fig.plot(
        x=map_events['lon_rel'], y=map_events['lat_rel'], style='cc', size=map_events['size_cm'],
        fill=map_events['dep_rel'], cmap=True, pen='0.15p,black',
    )

    if df_isc is not None and len(df_isc) > 0:
        fig.plot(
            x=df_isc['longitude'], y=df_isc['latitude'], style='a', size=df_isc['size_cm'],
            fill=df_isc['depth_km'], cmap=True, pen='0.8p,black',
        )

    if cfg['map_plot_well']:
        fig.plot(x=[cfg['map_well_lon']], y=[cfg['map_well_lat']], style='d0.5c', fill='green', pen='1p,darkgreen')
        fig.text(
            x=cfg['map_well_lon'] + 0.08, y=cfg['map_well_lat'] + 0.04, text=cfg['map_well_name'],
            font='8p,Helvetica-Bold,darkgreen', justify='LM',
        )

    fig.plot(x=station_df['longitude'], y=station_df['latitude'], style='t0.5c', fill='black', pen='0.5p,black')
    for _, row in station_df.iterrows():
        fig.text(
            x=row['longitude'] + 0.05, y=row['latitude'] + 0.03, text=row['station'],
            font='8p,Helvetica-Bold,black', justify='LM',
        )

    draw_cross_section_lines(fig, region)
    draw_colorbar_and_scale(fig, depth_max_event)
    draw_symbol_legend(fig, cfg, region)
    draw_magnitude_legend(fig, cfg, region)

    return fig


def plot_event_magnitude_map(events, event_ml, cfg):
    station_df = load_station_coords(cfg['stations_file'])
    map_events = build_map_events(events, event_ml, cfg)

    if len(map_events) == 0:
        print('No events with a valid location -- skipping event/magnitude map.')
        return

    region = derive_region(station_df, map_events, cfg['map_region_padding_deg'])
    df_isc = load_isc_catalog(cfg, region)

    print(f"Map region (auto-derived): {region}")
    print(f"Events plotted on map    : {len(map_events)}")
    if df_isc is not None:
        print(f"ISC events plotted on map: {len(df_isc)}")

    os.makedirs(os.path.dirname(cfg['map_output_path']), exist_ok=True)
    fig = build_map_figure(cfg, station_df, map_events, df_isc, region)
    fig.savefig(cfg['map_output_path'], dpi=300)
    print(f"Event/magnitude map saved to: {cfg['map_output_path']}")


# ------------------------------------------------------------
# 3.9 Main pipeline
# ------------------------------------------------------------

def main(cfg):
    csv_dir = os.path.join(cfg['output_folder'], 'csv')
    summary_dir = os.path.join(cfg['output_folder'], 'summary')
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(cfg['xml_cache_dir'], exist_ok=True)

    print('Loading stations, relocated events and phase picks...')
    station_df = load_stations(cfg['stations_file'], cfg['network'])
    events = load_relocated_events(cfg['reloc_summary_csv'])
    phase_df = parse_phase_dat(cfg['phase_dat'])
    event_station = build_event_station_table(events, phase_df, station_df, cfg['require_s_pick'])

    if cfg['max_events'] is not None:
        selected = sorted(event_station['event_id'].unique())[: cfg['max_events']]
        event_station = event_station[event_station['event_id'].isin(selected)].copy()

    print(f"Relocated events        : {len(events)}")
    print(f"Event-station pairs     : {len(event_station)}")
    print(f"Unique events to process: {event_station['event_id'].nunique()}")

    station_out_path = os.path.join(csv_dir, 'ml_station_magnitudes.csv')
    existing = None
    if os.path.exists(station_out_path) and not cfg['overwrite']:
        existing = pd.read_csv(station_out_path)
        done_ok = set(
            zip(existing.loc[existing['status'] == 'OK', 'event_id'],
                existing.loc[existing['status'] == 'OK', 'station'])
        )
        keep_mask = ~event_station.apply(lambda r: (r['event_id'], r['station']) in done_ok, axis=1)
        work_df = event_station[keep_mask].copy()
        print(f"Resuming: {len(done_ok)} pairs already OK, {len(work_df)} left to process.")
    else:
        work_df = event_station

    new_results = []
    for _, row in tqdm(work_df.iterrows(), total=len(work_df), desc='Computing station ML'):
        new_results.append(process_event_station(row, cfg))
    new_df = pd.DataFrame(new_results)

    if existing is not None and len(existing) > 0:
        if len(new_df) > 0:
            existing = existing[
                ~existing.set_index(['event_id', 'station']).index.isin(
                    new_df.set_index(['event_id', 'station']).index
                )
            ]
        station_ml = pd.concat([existing, new_df], ignore_index=True)
    else:
        station_ml = new_df

    station_ml.to_csv(station_out_path, index=False)
    print(f"Station magnitudes saved to: {station_out_path}")
    print(station_ml['status'].value_counts(dropna=False))

    event_ml = aggregate_event_magnitudes(station_ml, events, cfg)
    event_out_path = os.path.join(csv_dir, 'ml_event_magnitudes.csv')
    event_ml.to_csv(event_out_path, index=False)
    print(f"Event magnitudes saved to: {event_out_path}")
    print(f"Events with a valid ML   : {event_ml['ML_median'].notna().sum()} / {len(event_ml)}")

    if cfg['plot_summary']:
        plot_summary(event_ml, summary_dir)
        print(f"Diagnostic plots saved to: {summary_dir}")

    if cfg['plot_map']:
        plot_event_magnitude_map(events, event_ml, cfg)


if __name__ == '__main__':
    main(CONFIG)
