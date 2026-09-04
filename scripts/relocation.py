# ============================================================
# Double-Difference Relocation Pipeline (hypoDD, via relocDD-py)
#
# Stage 4 of the workflow:
#   1) pick.py         -> phase picking (EQTransformer / SeisBench)
#   2) association.py  -> phase association + quality control
#   3) locator.py       -> absolute hypocentre location (NonLinLoc)
#   4) relocation.py    -> double-difference relocation (this file)
#
# hypoDD relocates events *relative to each other* using two kinds of
# differential travel time between nearby event pairs:
#   dt.ct  catalogue differential time, built from absolute picks (phase.dat)
#   dt.cc  cross-correlation differential time, built by correlating the
#          waveform around matching P/S picks of nearby event pairs
# dt.ct is generated automatically by ph2dt (bundled in relocDD-py) from
# phase.dat -- this script only has to build dt.cc itself, which needs
# access to the raw waveforms (ph2dt/hypoDD never touch waveform data).
#
# Input  : output/hypocenter_locator/csv/hypocenter_catalog.csv  (NLLoc events)
#          output/hypocenter_locator/csv/hypocenter_phases.csv   (NLLoc arrivals)
#          output/preprocessing/<date>/<NET>.<STA>..<date>.mseed   (raw waveforms)
#          stations.txt
#
# Output : output/relocation/
#            stations.dat, phase.dat, dt.cc            hypoDD-format inputs
#            ph2dt.inp, hypoDD.inp, run.inp             relocDD-py config
#            event_id_map.csv                           cuspid <-> NLLoc event_id
#            maxsepe_diagnostic.png                      MAXSEPE derivation plot
#            cc_figures/                                 top-N CC pair waveform plots
#            outputs_relocdd/EDD/tradouts/hypoDD.reloc   relocDD-py's raw output
#            hypocenter_relocated_summary.csv             written by relocDD-py
#            relocation_error_summary.csv                 written by relocDD-py
#            relocated_catalog.csv                        final catalogue (this script)
#            relocation_shift.png                         epicentre shift map + histogram
#            shift_statistics.png                          per-event shift + distribution (3-panel)
#            cluster_diagnostics.png / cluster_summary.csv  dt.ct histogram, cluster map & depths
#            rms_before_after.png                          NLLoc vs relocated RMS residual
#            location_errors.png                           horizontal/vertical error (4-panel)
#  (the five plot_* outputs above port the figures from the reference project's
#  jatim_aus/relokasi/plot_output.ipynb, sourced from files this script already
#  produces rather than re-parsing raw NLLoc .hyp text)
#
# NOTE: This file is deliberately split into:
#   1) MODULE IMPORTS
#   2) CONFIGURATION (loaded from config/config.yaml -- see CONFIG_YAML_PATH)
#   3) PROCESS (data prep, cross-correlation, hypoDD I/O, driver, plots)
# CONFIG is loaded from the 'relocation' section of the shared
# config/config.yaml (see pipeline.py at the project root, which runs every
# stage from that same file) -- edit the YAML to change parameters, not this
# file; the CONFIG dict keys and every function signature below are
# unchanged, so PROCESS never needs to be touched for a config-only change.
# CONFIG['layer_top_km'] / CONFIG['layer_vp_km_s'] are the one exception --
# they are not stored directly in the YAML but parsed from
# config/vel_model/jatim_velocity_model.txt (see CONFIG['velocity_model_file'])
# by load_layer_arrays_from_file() below, the same file hypocenter_locator/
# locator.py parses its own CONFIG['velocity_model'] from.
#
# relocDD-py (vendored under output/relocation/relocDD-py/) is a pure-Python
# reimplementation of Waldhauser & Ellsworth's hypoDD -- there is no Fortran
# ph2dt/hypoDD binary installed on this machine, so relocation.py imports
# relocDD-py's run.main() directly instead of shelling out to a binary.
# ============================================================

# ============================================================
# 1) MODULE IMPORTS
# ============================================================

import os
import sys
import glob
import json
import math
import shutil
import warnings
import subprocess
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Package 'tqdm' is not installed. Run: pip install tqdm")

try:
    from obspy import read, Stream, UTCDateTime
    from scipy.signal import correlate
    from scipy.stats import gaussian_kde
except ImportError:
    raise ImportError("Packages 'obspy' and 'scipy' are required. "
                       "Run this script with the pyocto conda environment.")


# ============================================================
# 2) CONFIGURATION
# ============================================================

def load_layer_arrays_from_file(path):
    """Parse an NLLoc LAYER-format file (see config/vel_model/) into the
    (layer_top_km, layer_vp_km_s) parallel-list pair CONFIG expects -- the
    P-velocity/depth layers only; hypoDD uses a single constant Vp/Vs ratio
    (CONFIG['vpvs_ratio']) rather than the file's per-layer Vs column."""
    layer_top_km, layer_vp_km_s = [], []
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != 'LAYER':
                continue
            # LAYER  depth  Vp  Vp_grad  Vs  Vs_grad  density  density_grad
            layer_top_km.append(float(parts[1]))
            layer_vp_km_s.append(float(parts[2]))
    return layer_top_km, layer_vp_km_s


CONFIG_YAML_PATH = "/media/galih/MyBackUp 2024/Jatim_new/config/config.yaml"

with open(CONFIG_YAML_PATH, 'r') as _f:
    _all_cfg = yaml.safe_load(_f)

CONFIG = _all_cfg['relocation']
CONFIG['network'] = _all_cfg['network']   # single global network code (see stations.txt)
CONFIG['layer_top_km'], CONFIG['layer_vp_km_s'] = load_layer_arrays_from_file(CONFIG['velocity_model_file'])


# ============================================================
# 3) PROCESS
# ============================================================

# ------------------------------------------------------------
# 3.1 Folders
# ------------------------------------------------------------

def setup_output_folders(root):
    folders = {
        'root'       : root,
        'cc_figures' : os.path.join(root, 'cc_figures'),
        'relocdd_out': os.path.join(root, 'outputs_relocdd'),
    }
    for path in folders.values():
        os.makedirs(path, exist_ok=True)
    return folders


# ------------------------------------------------------------
# 3.2 Load NLLoc catalogue and stations
# ------------------------------------------------------------

def load_stations_dataframe(path, network):
    """stations.txt (columns: station latitude longitude elevation_m) ->
    DataFrame with columns: station, network, latitude, longitude,
    elevation_m (same shape as hypocenter_locator/locator.py). network is a
    single global constant (config.yaml's top-level 'network' key) since
    stations.txt carries no per-station network code."""
    rows = []
    with open(path, 'r') as f:
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
    return pd.DataFrame(rows).sort_values('station').reset_index(drop=True)


def load_catalog_and_phases(cfg):
    """
    Load hypocenter_catalog.csv / hypocenter_phases.csv, filter to qc_pass
    events, and assign a sequential integer cuspid (hypoDD requires numeric
    event IDs; NLLoc's event_id, e.g. "jatim.20151206.015334.grid0", is not).
    """
    df_cat = pd.read_csv(cfg['catalog_csv'], parse_dates=['origin_time'])
    df_pha = pd.read_csv(cfg['phases_csv'], parse_dates=['pick_time'])

    if cfg['use_qc_pass_only'] and 'qc_pass' in df_cat.columns:
        keep_ids = set(df_cat.loc[df_cat['qc_pass'], 'event_id'])
        df_cat = df_cat[df_cat['event_id'].isin(keep_ids)]

    df_cat = df_cat.sort_values('origin_time').reset_index(drop=True)
    df_cat['cuspid'] = np.arange(1, len(df_cat) + 1)

    id_map = dict(zip(df_cat['event_id'], df_cat['cuspid']))
    df_pha = df_pha[df_pha['event_id'].isin(id_map)].copy()
    df_pha['cuspid'] = df_pha['event_id'].map(id_map)

    return df_cat, df_pha


# ------------------------------------------------------------
# 3.3 MAXSEPE: median 3rd-nearest-neighbour 3D distance
# ------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def pairwise_dist3d_km(df_cat):
    """Full N x N matrix of 3D (Haversine + depth) inter-event distance."""
    lat = df_cat['latitude'].to_numpy()
    lon = df_cat['longitude'].to_numpy()
    dep = df_cat['depth_km'].to_numpy()
    n = len(df_cat)

    horiz = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    ddep = dep[:, None] - dep[None, :]
    dist3d = np.sqrt(horiz ** 2 + ddep ** 2)
    np.fill_diagonal(dist3d, np.nan)
    return dist3d


def derive_maxsepe(df_cat, cfg, folders):
    """
    MAXSEPE = median of the k-th nearest-neighbour 3D distance across the
    catalogue (k = cfg['maxsepe_nn_k'], default 3rd-NN -- robust to isolated
    singletons that would drag the 1st-NN median down artificially).
    """
    if cfg['maxsepe_km'] is not None:
        print(f"   MAXSEPE fixed by CONFIG: {cfg['maxsepe_km']:.2f} km")
        return float(cfg['maxsepe_km'])

    k = cfg['maxsepe_nn_k']
    dist3d = pairwise_dist3d_km(df_cat)
    n = len(df_cat)
    if n <= k:
        raise ValueError(f"Catalogue has only {n} events -- need more than "
                          f"{k} to derive MAXSEPE from the {k}-th nearest neighbour. "
                          f"Set CONFIG['maxsepe_km'] explicitly instead.")

    sorted_dist = np.sort(dist3d, axis=1)
    nn_k = sorted_dist[:, k - 1]
    maxsepe = float(np.round(np.nanmedian(nn_k), 1))

    p25, p75 = np.nanpercentile(nn_k, [25, 75])
    print(f"   {k}-NN 3D distance : P25={p25:.1f}  median={np.nanmedian(nn_k):.1f}  "
          f"P75={p75:.1f} km  ({n} events)")
    print(f"   MAXSEPE (auto, median of {k}-NN): {maxsepe:.1f} km")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(nn_k, bins=30, color='steelblue', alpha=0.8, edgecolor='white')
    ax.axvline(maxsepe, color='crimson', ls='--', lw=2,
               label=f"MAXSEPE = median = {maxsepe:.1f} km")
    ax.set_xlabel(f"{k}-th nearest-neighbour 3D distance (km)")
    ax.set_ylabel("Number of events")
    ax.set_title("MAXSEPE derivation: event-pair separation distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(folders['root'], f"maxsepe_diagnostic.{cfg['plot_format']}"),
                dpi=cfg['plot_dpi'])
    plt.close(fig)

    return maxsepe


# ------------------------------------------------------------
# 3.4 Write hypoDD-format inputs: stations.dat, phase.dat
# ------------------------------------------------------------

def write_stations_dat(stations, path):
    with open(path, 'w') as f:
        for _, row in stations.iterrows():
            f.write(f"{row['station']:<10s}{row['latitude']:>10.4f}"
                    f"{row['longitude']:>11.4f}{row['elevation_m']:>8.1f}\n")


def write_phase_dat(df_cat, df_pha, path):
    """
    One '# YR MO DY HR MN SC.ss LAT LON DEP MAG EH EV RMS ID' header per event,
    followed by 'STA  travel_time_s  weight  PHASE' lines -- the format ph2dt's
    readphase() expects (relocDD-py/ph2dt/ph2dt_files.py).
    """
    pha_by_event = {cid: g for cid, g in df_pha.groupby('cuspid')}
    n_events, n_picks = 0, 0

    with open(path, 'w') as f:
        for _, ev in df_cat.iterrows():
            grp = pha_by_event.get(ev['cuspid'])
            if grp is None or grp.empty:
                continue

            ot = ev['origin_time']
            eh = ev['err_h_km'] if pd.notna(ev.get('err_h_km')) else 0.0
            ev_z = ev['err_z_km'] if pd.notna(ev.get('err_z_km')) else 0.0
            rms = ev['rms_s'] if pd.notna(ev.get('rms_s')) else 0.0

            f.write(f"# {ot.year:4d} {ot.month:02d} {ot.day:02d} "
                    f"{ot.hour:02d} {ot.minute:02d} "
                    f"{ot.second + ot.microsecond / 1e6:5.2f} "
                    f"{ev['latitude']:.4f} {ev['longitude']:.4f} "
                    f"{ev['depth_km']:.3f} 0.0 {eh:.3f} {ev_z:.3f} {rms:.3f} "
                    f"{int(ev['cuspid'])}\n")
            n_events += 1

            for _, p in grp.sort_values(['station', 'phase']).iterrows():
                tt = (p['pick_time'] - ot).total_seconds()
                w = p['weight'] if pd.notna(p.get('weight')) else 1.0
                w = float(np.clip(w, 0.0, 1.0))
                f.write(f"{p['station']:<8s} {tt:7.3f} {w:5.3f} {p['phase']}\n")
                n_picks += 1

    print(f"   phase.dat written: {n_events} events, {n_picks} phase picks")


# ------------------------------------------------------------
# 3.5 Cross-correlation (dt.cc)
# ------------------------------------------------------------

def _date_range_str(t1, t2):
    """t1/t2 are UTCDateTime; .date is a property (datetime.date), not a method."""
    d, end = t1.date, t2.date
    out = []
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def load_pick_window(station, t1, t2, cfg):
    """Read one station's trace over [t1, t2), preferring the configured
    channel priority, falling back to any Z-named channel."""
    # obspy's mseed reader/trim raise ValueError on a plain datetime/Timestamp
    # (it requires UTCDateTime specifically), so convert once up front.
    t1, t2 = UTCDateTime(t1), UTCDateTime(t2)
    base = cfg['waveform_folder']
    found, seen = [], set()
    for dstr in _date_range_str(t1, t2):
        day_dir = os.path.join(base, dstr)
        if not os.path.isdir(day_dir):
            continue
        for pat in cfg['cc_wf_file_patterns']:
            pattern = os.path.join(day_dir, pat.format(sta=station, date=dstr))
            for m in sorted(glob.glob(pattern)):
                if m not in seen:
                    found.append(m)
                    seen.add(m)
            if found:
                break
    if not found:
        return None

    st = Stream()
    for f in found:
        try:
            st += read(f, starttime=t1, endtime=t2)
        except Exception:
            continue
    if len(st) == 0:
        return None
    try:
        st.merge(method=1, fill_value='interpolate')
    except Exception:
        return None

    tr = None
    for ch in cfg['cc_channel_priority']:
        sel = st.select(channel=ch)
        if sel:
            tr = sel[0]
            break
    if tr is None:
        for cand in st:
            if cand.stats.channel.endswith('Z'):
                tr = cand
                break
    if tr is None:
        return None

    tr = tr.copy()
    tr.trim(t1, t2, pad=True, fill_value=0)
    try:
        tr.detrend('demean')
        tr.taper(0.05)
        tr.filter('bandpass', freqmin=cfg['cc_lowcut_hz'], freqmax=cfg['cc_highcut_hz'],
                  corners=4, zerophase=True)
        if abs(tr.stats.sampling_rate - cfg['cc_target_sr_hz']) > 0.01:
            tr.resample(cfg['cc_target_sr_hz'])
    except Exception:
        return None

    data = tr.data.astype(np.float64)
    mx = np.max(np.abs(data))
    if mx < 1e-12:
        return None
    return {'data': data / mx, 'sr': tr.stats.sampling_rate}


def load_event_pick_windows(cuspid, df_pha_event, cfg):
    """
    For one event, cut a window around every P and S pick from the
    continuous waveform. Returns {(station, phase): {"data":, "sr":}}.
    """
    out = {}
    for _, p in df_pha_event.iterrows():
        phase = p['phase']
        pre, post = ((cfg['cc_p_pre_s'], cfg['cc_p_post_s']) if phase == 'P'
                     else (cfg['cc_s_pre_s'], cfg['cc_s_post_s']))
        t1 = p['pick_time'] - timedelta(seconds=pre)
        t2 = p['pick_time'] + timedelta(seconds=post)
        wf = load_pick_window(p['station'], t1, t2, cfg)
        if wf is not None:
            out[(p['station'], phase)] = wf
    return out


def ncc_and_lag(a, b, sr):
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    std_a, std_b = np.std(a) + 1e-12, np.std(b) + 1e-12

    full_cc = correlate(a, b, mode='full')
    ncc = np.clip(full_cc / (n * std_a * std_b), -1, 1)

    lag_idx = int(np.argmax(np.abs(ncc)))
    cc_max = float(ncc[lag_idx])
    center = len(a) - 1
    lag_sec = (lag_idx - center) / sr
    return cc_max, lag_sec, ncc, center


def plot_cc_pair(eid_a, eid_b, dist_km, results, fig_dir, cfg):
    n = len(results)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.8 * n), squeeze=False)
    mean_cc = np.mean([abs(r['cc']) for r in results])
    fig.suptitle(f"Event {eid_a} vs {eid_b} | dist={dist_km:.1f} km | "
                 f"mean|CC|={mean_cc:.3f} | {n} station-phase pairs",
                 fontsize=10, fontweight='bold')

    for row, r in enumerate(results):
        t = np.arange(len(r['data_a'])) / r['sr']
        ax_w, ax_c = axes[row][0], axes[row][1]

        ax_w.plot(t, r['data_a'], color='red', lw=0.9, alpha=0.85, label=f"Event {eid_a}")
        ax_w.plot(t, r['data_b'], color='blue', lw=0.9, alpha=0.75, label=f"Event {eid_b}")
        ax_w.set_ylim(-1.3, 1.3)
        ax_w.set_ylabel(f"{r['sta']} {r['phase']}\namp (norm)", fontsize=8)
        ax_w.grid(True, alpha=0.2)
        if row == 0:
            ax_w.legend(fontsize=7, loc='upper right')

        lag_axis = (np.arange(len(r['ncc'])) - r['center']) / r['sr']
        ax_c.plot(lag_axis, r['ncc'], color='black', lw=0.7)
        color = 'red' if r['cc'] >= 0 else 'blue'
        ax_c.axvline(r['lag_sec'], color=color, ls='--', lw=1.3)
        ax_c.plot(r['lag_sec'], r['cc'], marker='v', color=color, ms=7)
        ax_c.annotate(f"CC={r['cc']:+.3f}\nlag={r['lag_sec']:+.3f}s",
                      xy=(r['lag_sec'], r['cc']), xytext=(10, -20),
                      textcoords='offset points', fontsize=8, color=color)
        ax_c.set_ylim(-1.05, 1.05)
        ax_c.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_dir_path = os.path.join(fig_dir, f"pair_{eid_a}_{eid_b}.{cfg['plot_format']}")
    fig.savefig(fig_dir_path, dpi=cfg['plot_dpi'], bbox_inches='tight')
    plt.close(fig)


def build_dt_cc(df_cat, df_pha, maxsepe_km, cfg, folders):
    """
    Pair every event within maxsepe_km, cross-correlate matching P/S pick
    windows station by station, and write dt.cc for every station-phase with
    |CC| >= cfg['cc_threshold'].
    """
    dist3d = pairwise_dist3d_km(df_cat)
    cuspids = df_cat['cuspid'].to_numpy()
    n = len(df_cat)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dist3d[i, j]
            if d <= maxsepe_km:
                pairs.append((cuspids[i], cuspids[j], float(d)))
    print(f"   {len(pairs)} event pairs within MAXSEPE={maxsepe_km:.1f} km "
          f"(of {n * (n - 1) // 2} possible)")
    if not pairs:
        return {}

    pha_by_cuspid = {cid: g for cid, g in df_pha.groupby('cuspid')}
    needed = {cid for pair in pairs for cid in pair[:2]}

    print("   Cutting waveform windows around P/S picks (cached per event)...")
    wf_cache = {}
    for cid in tqdm(sorted(needed), desc='Load windows', ncols=80):
        grp = pha_by_cuspid.get(cid)
        wf_cache[cid] = load_event_pick_windows(cid, grp, cfg) if grp is not None else {}

    print("   Cross-correlating event pairs...")
    dt_by_pair = defaultdict(list)   # (cid_a, cid_b, dist) -> [row, ...]
    pair_results = []                # for figures: (mean_cc, a, b, dist, results)

    for cid_a, cid_b, dist_km in tqdm(pairs, desc='CC pairs', ncols=80):
        wf_a, wf_b = wf_cache.get(cid_a, {}), wf_cache.get(cid_b, {})
        if not wf_a or not wf_b:
            continue
        common = sorted(set(wf_a.keys()) & set(wf_b.keys()))
        if not common:
            continue

        results = []
        for sta, phase in common:
            a, b = wf_a[(sta, phase)], wf_b[(sta, phase)]
            cc, lag_sec, ncc, center = ncc_and_lag(a['data'], b['data'], a['sr'])
            if abs(cc) < cfg['cc_threshold']:
                continue
            results.append({'sta': sta, 'phase': phase, 'cc': cc, 'lag_sec': lag_sec,
                            'ncc': ncc, 'center': center, 'data_a': a['data'],
                            'data_b': b['data'], 'sr': a['sr']})
            dt_by_pair[(cid_a, cid_b, dist_km)].append({
                'sta': sta, 'lag_sec': lag_sec, 'cc': abs(cc), 'phase': phase,
            })

        if results:
            mean_cc = float(np.mean([abs(r['cc']) for r in results]))
            pair_results.append((mean_cc, cid_a, cid_b, dist_km, results))

    n_lines = sum(len(v) for v in dt_by_pair.values())
    print(f"   {len(dt_by_pair)} pairs / {n_lines} station-phase lines passed "
          f"|CC| >= {cfg['cc_threshold']}")

    if pair_results and cfg['cc_max_plot'] > 0:
        pair_results.sort(key=lambda x: x[0], reverse=True)
        print(f"   Plotting top {min(cfg['cc_max_plot'], len(pair_results))} CC pairs...")
        for mean_cc, a, b, d, results in pair_results[:cfg['cc_max_plot']]:
            plot_cc_pair(a, b, d, results, folders['cc_figures'], cfg)

    return dt_by_pair


def write_dt_cc(dt_by_pair, path):
    """
    relocDD-py's event-pair reader (methodTypes/eventPair/hypoDD_ev.py
    readccs_evpair) requires the header line to be '# EV1ID EV2ID OTC', where
    OTC (origin-time correction) is SUBTRACTED directly from every dt in that
    pair (dt_dt[i] -= otc). It must be 0.0 unless an origin-time correction
    was actually measured -- putting the inter-event distance there (as the
    jatim_aus reference project's dt.cc did) would silently shift every
    differential time in the pair by minus that many seconds. The distance is
    kept as a trailing comment token instead, which the whitespace-split
    parser ignores past index 3.
    """
    n_lines = 0
    with open(path, 'w') as f:
        for (cid_a, cid_b, dist_km), rows in sorted(dt_by_pair.items()):
            f.write(f"# {cid_a:>8d}  {cid_b:>8d}  0.000000  # dist_km={dist_km:.4f}\n")
            for r in rows:
                f.write(f"{r['sta']:<8s}  {r['lag_sec']:10.6f}  {r['cc']:8.5f}  {r['phase']}\n")
                n_lines += 1
    print(f"   dt.cc written: {len(dt_by_pair)} pairs, {n_lines} lines -> {path}")


# ------------------------------------------------------------
# 3.6 relocDD-py config files: ph2dt.inp, hypoDD.inp, run.inp
# ------------------------------------------------------------

def write_ph2dt_inp(path, maxsepe_km, cfg):
    text = (
        "* ph2dt.inp\n"
        "stations.dat\n"
        "phase.dat\n"
        f"{cfg['ph2dt_minwght']} {cfg['ph2dt_maxdist_km']} {maxsepe_km} "
        f"{cfg['maxseps_km']} {cfg['ph2dt_maxngh']} {cfg['ph2dt_minlnk']} "
        f"{cfg['ph2dt_minobs']} {cfg['ph2dt_maxobs']}\n"
    )
    with open(path, 'w') as f:
        f.write(text)


def write_hypodd_inp(path, cfg):
    # relocDD-py's ph2dt hardcodes the catalogue differential-time filename
    # per method (methodTypes/*/ph2dt_*.py): dte.ct for event-pair (EDD),
    # dts.ct for station-pair, dtd.ct for double-pair. hypoDD.inp's filename
    # declaration has to match whatever ph2dt actually wrote, or hypoDD reads
    # nothing.
    ct_filename = {1: 'dte.ct', 2: 'dts.ct', 3: 'dtd.ct'}[cfg['reloctype']]
    lines = [
        "* hypoDD.inp",
        "dt.cc", ct_filename, "event.dat", "stations.dat",
        "hypoDD.loc", "hypoDD.reloc", "hypoDD.sta", "hypoDD.res", "hypoDD.src",
        f"{cfg['hyp_idat']} {cfg['hyp_ipha']} {cfg['hyp_dist_km']}",
        f"{cfg['hyp_obscc']} {cfg['hyp_obsct']}",
        f"{cfg['hyp_istart']} {cfg['hyp_isolv']} {len(cfg['hyp_iter_blocks'])}",
    ]
    for niter, wtccp, wtccs, wrcc, wdcc, wtctp, wtcts, wrct, wdct in cfg['hyp_iter_blocks']:
        lines.append(f"{niter} {wtccp} {wtccs} {wrcc} {wdcc} "
                     f"{wtctp} {wtcts} {wrct} {wdct} {cfg['hyp_damp']}")
    lines.append(f"{len(cfg['layer_top_km'])} {cfg['vpvs_ratio']}")
    lines.append(" ".join(f"{v}" for v in cfg['layer_top_km']))
    lines.append(" ".join(f"{v}" for v in cfg['layer_vp_km_s']))
    lines.append("0")   # CID = 0 -> relocate every cluster
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")


def write_run_inp(path, folders, cfg):
    text = (
        "*\n*\n"
        f"{folders['root']}\n"          # inputfol (ph2dt.inp/hypoDD.inp live here)
        f"{folders['root']}\n"          # datfol   (stations.dat/phase.dat/dt.cc live here)
        f"{folders['relocdd_out']}\n"   # outfol
        f"{cfg['reloctype']}\n"
        "0\n"    # fileout
        "0\n"    # makedata
        "0\n"    # hypoinv
        "0\n"    # noiseswitch
        "0\n"    # noisediff
        f"{cfg['stdcc']}\n"
        f"{cfg['stdct']}\n"
        "0\n"    # nboot
        "0\n"    # nplot
    )
    with open(path, 'w') as f:
        f.write(text)


# ------------------------------------------------------------
# 3.7 Run relocDD-py (ph2dt + hypoDD)
# ------------------------------------------------------------

def run_relocdd(run_inp_path, cfg):
    relocdd_dir = cfg['relocdd_py_dir']
    if not os.path.isdir(relocdd_dir):
        raise FileNotFoundError(
            f"relocDD-py not found at {relocdd_dir}. Vendor the package there first "
            f"(a pure-Python hypoDD reimplementation -- no Fortran ph2dt/hypoDD "
            f"binary is installed on this machine).")

    # relocDD-py/run.py runs main() unconditionally at import time, dispatched
    # from sys.argv (see the bottom of that file) -- it is not guarded by
    # `if __name__ == '__main__'`. Importing it as a module would therefore
    # re-run it a second time using this process's own argv. It has to be
    # invoked as the subprocess it was written for instead.
    result = subprocess.run(
        [sys.executable, 'run.py', run_inp_path, '1', '1'],
        cwd=relocdd_dir, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"relocDD-py run.py exited with code {result.returncode}")


# ------------------------------------------------------------
# 3.8 Build the final relocated catalogue
# ------------------------------------------------------------

def build_relocated_catalog(df_cat, folders, cfg):
    """
    Join relocDD-py's own hypocenter_relocated_summary.csv (cuspid-indexed)
    back to the original NLLoc event_id, so the relocated catalogue stays
    traceable to the .pick files / waveforms it came from.
    """
    summary_path = os.path.join(folders['root'], 'hypocenter_relocated_summary.csv')
    if not os.path.exists(summary_path):
        print("   ⚠️  hypocenter_relocated_summary.csv not found -- "
              "no events survived relocation (check the relocDD-py log above)")
        return pd.DataFrame()

    df_reloc = pd.read_csv(summary_path)
    df_out = df_reloc.merge(
        df_cat[['cuspid', 'event_id', 'latitude', 'longitude', 'depth_km']]
        .rename(columns={'latitude': 'lat_nlloc', 'longitude': 'lon_nlloc',
                         'depth_km': 'depth_nlloc_km'}),
        left_on='cuspid', right_on='cuspid', how='left')

    out_path = os.path.join(folders['root'], 'relocated_catalog.csv')
    df_out.to_csv(out_path, index=False)
    print(f"   ✅ {out_path} : {len(df_out):,} relocated events "
          f"(of {len(df_cat):,} input)")
    return df_out


def plot_relocation_shift(df_out, cfg, folders):
    if df_out.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(df_out['lon_nlloc'], df_out['lat_nlloc'], s=25, color='gray',
              alpha=0.6, label='NLLoc (absolute)')
    ax.scatter(df_out['lon_rel'], df_out['lat_rel'], s=25, color='crimson',
              alpha=0.8, label='hypoDD (relocated)')
    for _, r in df_out.iterrows():
        ax.plot([r['lon_nlloc'], r['lon_rel']], [r['lat_nlloc'], r['lat_rel']],
               color='black', lw=0.4, alpha=0.4)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Epicentre shift: NLLoc -> hypoDD')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(df_out['horiz_shift_km'].dropna(), bins=30, color='steelblue', alpha=0.8,
           edgecolor='white', label='Horizontal shift (km)')
    ax.axvline(df_out['horiz_shift_km'].median(), color='red', ls='--', lw=1.5,
              label=f"median = {df_out['horiz_shift_km'].median():.2f} km")
    ax.set_xlabel('Horizontal shift (km)')
    ax.set_ylabel('Count')
    ax.set_title('Relocation shift distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(folders['root'], f"relocation_shift.{cfg['plot_format']}"),
               dpi=cfg['plot_dpi'])
    plt.close(fig)


# ------------------------------------------------------------
# 3.9 Post-relocation diagnostic plots
#
# Ports the figures from the reference project's jatim_aus/relokasi/
# plot_output.ipynb, but sourced from files this script already produces
# (relocated_catalog.csv, event_id_map.csv's rms_s, relocDD-py's own
# txtfiles/ output) instead of re-parsing raw NLLoc .hyp text and doing a
# fragile nearest-time match against a separate catalog file -- the cuspid
# already ties every table together exactly.
# ------------------------------------------------------------

def read_hypodd_reloc(path):
    cols = ['cuspid', 'lat', 'lon', 'dep', 'x', 'y', 'z', 'ex', 'ey', 'ez',
            'yr', 'mo', 'dy', 'hr', 'mn', 'sec', 'mag', 'cluster']
    df = pd.read_csv(path, sep=r'\s+', header=None, names=cols)
    df['cluster'] = df['cluster'].astype(int)
    return df


def parse_dt_ct(path):
    """Parse a ph2dt-generated dt.ct/dte.ct file into one row per pick."""
    rows = []
    ev1 = ev2 = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('#'):
                p = s.split()
                if len(p) >= 3:
                    ev1, ev2 = int(p[1]), int(p[2])
                continue
            p = s.split()
            if len(p) >= 5 and ev1 is not None:
                sta = p[0]
                t1, t2, w, ph = float(p[1]), float(p[2]), float(p[3]), p[4].upper()
                rows.append([ev1, ev2, sta, ph, t1, t2, t1 - t2, w])
    return pd.DataFrame(rows, columns=['ev1', 'ev2', 'station', 'phase',
                                       't1', 't2', 'dt', 'weight'])


def plot_shift_statistics(df_out, cfg, folders):
    """Per-event horizontal shift / |depth change|, plus their pooled
    distribution with a KDE overlay."""
    d = df_out.dropna(subset=['horiz_shift_km', 'depth_change_km']).copy()
    if d.empty:
        return
    d['abs_depth_change_km'] = d['depth_change_km'].abs()

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    h_med = d['horiz_shift_km'].median()
    ax1.scatter(d['cuspid'], d['horiz_shift_km'], s=25, alpha=0.75, color='#1f77b4')
    ax1.axhline(h_med, color='crimson', ls='--', lw=2, label=f"median = {h_med:.3f} km")
    ax1.set_title('Horizontal shift per event')
    ax1.set_xlabel('Event ID (cuspid)')
    ax1.set_ylabel('Shift (km)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ad_med = d['abs_depth_change_km'].median()
    ax2.scatter(d['cuspid'], d['abs_depth_change_km'], s=25, alpha=0.75, color='#ff7f0e')
    ax2.axhline(ad_med, color='purple', ls='--', lw=2, label=f"median = {ad_med:.3f} km")
    ax2.set_title('|Depth change| per event')
    ax2.set_xlabel('Event ID (cuspid)')
    ax2.set_ylabel('|Depth change| (km)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3.hist(d['horiz_shift_km'], bins=30, alpha=0.5, color='#1f77b4', density=True,
            label='Horizontal shift (km)')
    ax3.hist(d['abs_depth_change_km'], bins=30, alpha=0.5, color='#ff7f0e', density=True,
            label='|Depth change| (km)')
    for series, color in [(d['horiz_shift_km'], '#1f77b4'),
                          (d['abs_depth_change_km'], '#ff7f0e')]:
        if series.nunique() > 1:
            kde = gaussian_kde(series)
            xs = np.linspace(series.min(), series.max(), 200)
            ax3.plot(xs, kde(xs), color=color, lw=2)
    ax3.axvline(h_med, color='#1f77b4', ls='--', lw=1.5)
    ax3.axvline(ad_med, color='#ff7f0e', ls='--', lw=1.5)
    ax3.set_title('Distribution of relocation shift')
    ax3.set_xlabel('Shift (km)')
    ax3.set_ylabel('Density')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(folders['root'], f"shift_statistics.{cfg['plot_format']}"),
               dpi=cfg['plot_dpi'])
    plt.close(fig)


def plot_cluster_diagnostics(folders, cfg):
    """dt.ct differential-time distribution, cluster map and per-cluster
    depth spread, from relocDD-py's own hypoDD.reloc + dt.ct output."""
    reloc_path = os.path.join(folders['relocdd_out'], 'EDD', 'tradouts', 'hypoDD.reloc')
    ct_filename = {1: 'dte.ct', 2: 'dts.ct', 3: 'dtd.ct'}[cfg['reloctype']]
    ct_path = os.path.join(folders['root'], ct_filename)
    if not os.path.exists(reloc_path) or not os.path.exists(ct_path):
        print("   ⚠️  hypoDD.reloc or dt.ct not found -- skipping cluster diagnostics")
        return

    rel = read_hypodd_reloc(reloc_path)
    if rel.empty:
        return
    dt_df = parse_dt_ct(ct_path)

    cluster_tbl = (rel.groupby('cluster')
                   .agg(n_event=('cuspid', 'count'), lon_mean=('lon', 'mean'),
                        lat_mean=('lat', 'mean'), dep_mean=('dep', 'mean'),
                        dep_min=('dep', 'min'), dep_max=('dep', 'max'))
                   .reset_index().sort_values('cluster'))
    cluster_tbl.to_csv(os.path.join(folders['root'], 'cluster_summary.csv'), index=False)
    print(f"   {cluster_tbl['cluster'].nunique()} cluster(s), "
          f"{int(cluster_tbl['n_event'].sum())} events -> cluster_summary.csv")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    if not dt_df.empty:
        bins = np.arange(np.floor(dt_df['dt'].min()) - 0.5, np.ceil(dt_df['dt'].max()) + 0.5, 0.5)
        ax1.hist(dt_df.loc[dt_df['phase'] == 'P', 'dt'], bins=bins, alpha=0.6,
                label='P', color='#1f77b4')
        ax1.hist(dt_df.loc[dt_df['phase'] == 'S', 'dt'], bins=bins, alpha=0.6,
                label='S', color='#ff7f0e')
        ax1.legend(fontsize=8)
    ax1.axvline(0, color='black', lw=1, ls='--')
    ax1.set_title('Catalogue differential-time distribution')
    ax1.set_xlabel('dt = t1 - t2 (s)')
    ax1.set_ylabel('Pick count')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    if not dt_df.empty:
        stats = dt_df.groupby('phase')['dt'].agg(['count', 'mean', 'median']).reset_index()
        x = np.arange(len(stats))
        ax2.bar(x, stats['mean'], color=['#1f77b4', '#ff7f0e'][:len(stats)], alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(stats['phase'])
        for i, r in stats.iterrows():
            ax2.text(i, r['mean'], f"n={int(r['count'])}\nmed={r['median']:.2f}",
                     ha='center', va='bottom', fontsize=9)
    ax2.axhline(0, color='black', lw=1)
    ax2.set_title('Mean dt per phase')
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('dt (s)')
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    clusters = sorted(rel['cluster'].unique())
    cmap = plt.colormaps['tab20'].resampled(max(len(clusters), 1))
    colors = {c: cmap(i) for i, c in enumerate(clusters)}
    for c, g in rel.groupby('cluster'):
        ax3.scatter(g['lon'], g['lat'], s=30, alpha=0.75, color=colors[c])
    for _, r in cluster_tbl.iterrows():
        ax3.text(r['lon_mean'], r['lat_mean'], str(int(r['cluster'])), fontsize=8,
                 fontweight='bold', ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.75))
    ax3.set_title('Cluster map (label = cluster ID)')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(0)
    for i, c in enumerate(clusters):
        g = rel[rel['cluster'] == c]
        jitter = rng.uniform(-0.25, 0.25, size=len(g))
        ax4.scatter(i + jitter, g['dep'], s=25, alpha=0.75, color=colors[c])
        ax4.text(i, g['dep'].mean(), f"{g['dep'].mean():.1f}", fontsize=8,
                 ha='center', va='bottom')
    ax4.set_xticks(range(len(clusters)))
    ax4.set_xticklabels([str(c) for c in clusters])
    ax4.invert_yaxis()
    ax4.set_title('Depth distribution per cluster')
    ax4.set_xlabel('Cluster ID')
    ax4.set_ylabel('Depth (km)')
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(folders['root'], f"cluster_diagnostics.{cfg['plot_format']}"),
               dpi=cfg['plot_dpi'])
    plt.close(fig)


def plot_rms_before_after(df_cat, folders, cfg):
    """
    NLLoc RMS (from hypocenter_catalog.csv, 'before') vs hypoDD's per-event
    RMS residual (from relocDD-py's own txtfiles/ output, 'after'), matched
    by cuspid -- exact, unlike the reference notebook's nearest-time match
    against a separately re-parsed .hyp file.
    """
    rms_path = os.path.join(folders['relocdd_out'], 'EDD', 'txtfiles',
                            'rms_residual_per_event_all_clusters.csv')
    if not os.path.exists(rms_path):
        print("   ⚠️  rms_residual_per_event_all_clusters.csv not found -- "
              "skipping RMS before/after")
        return

    after = pd.read_csv(rms_path).replace([np.inf, -np.inf], np.nan)
    after = after.dropna(subset=['cuspid', 'nobs', 'rms_res_s'])
    after['cuspid'] = after['cuspid'].astype(int)
    after = (after.sort_values(['cuspid', 'nobs', 'rms_res_s'], ascending=[True, False, True])
              .drop_duplicates(subset=['cuspid'], keep='first')[['cuspid', 'rms_res_s']]
              .rename(columns={'rms_res_s': 'rms_after_s'}))

    before = df_cat[['cuspid', 'rms_s']].dropna().rename(columns={'rms_s': 'rms_before_s'})
    cmp = before.merge(after, on='cuspid', how='inner')
    if cmp.empty:
        print("   ⚠️  No matching events between NLLoc and relocated RMS -- skipping")
        return
    cmp['delta_s'] = cmp['rms_after_s'] - cmp['rms_before_s']

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    mn = min(cmp['rms_before_s'].min(), cmp['rms_after_s'].min())
    mx = max(cmp['rms_before_s'].max(), cmp['rms_after_s'].max())

    ax[0].scatter(cmp['rms_before_s'], cmp['rms_after_s'], s=25, alpha=0.75, color='#1f77b4')
    ax[0].plot([mn, mx], [mn, mx], 'r--', lw=2)
    ax[0].set_xlabel('RMS NLLoc (s)')
    ax[0].set_ylabel('RMS relocated (s)')
    ax[0].set_title('RMS residual: before vs after')
    ax[0].grid(True, alpha=0.3)

    bins = np.linspace(mn, mx, 30)
    ax[1].hist(cmp['rms_before_s'], bins=bins, alpha=0.55, color='gray', label='NLLoc')
    ax[1].hist(cmp['rms_after_s'], bins=bins, alpha=0.55, color='tab:red', label='Relocated')
    ax[1].set_xlabel('RMS (s)')
    ax[1].set_ylabel('Event count')
    ax[1].set_title('RMS distribution')
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.3)

    ax[2].hist(cmp['delta_s'], bins=30, alpha=0.85, color='#4c72b0')
    ax[2].axvline(0, color='red', ls='--', lw=2)
    ax[2].set_xlabel('Delta RMS = after - before (s)')
    ax[2].set_ylabel('Event count')
    ax[2].set_title('Delta RMS distribution')
    ax[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(folders['root'], f"rms_before_after.{cfg['plot_format']}"),
               dpi=cfg['plot_dpi'])
    plt.close(fig)
    print(f"   RMS before/after: {len(cmp)} matched events, "
          f"median delta = {cmp['delta_s'].median():+.4f} s")


def plot_location_errors(folders, cfg):
    """Horizontal/vertical relocation error per event and their distributions,
    from relocDD-py's own relocation_error_summary.csv."""
    err_path = os.path.join(folders['root'], 'relocation_error_summary.csv')
    if not os.path.exists(err_path):
        print("   ⚠️  relocation_error_summary.csv not found -- skipping location-error plot")
        return
    df = pd.read_csv(err_path)
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(df['cuspid'], df['err_h_m'], s=25, color='#1f77b4', alpha=0.8)
    ax.axhline(df['err_h_m'].median(), color='red', ls='--', lw=2,
              label=f"median = {df['err_h_m'].median():.1f} m")
    ax.set_title('Horizontal error per event')
    ax.set_xlabel('Event ID (cuspid)')
    ax.set_ylabel('Horizontal error (m)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(df['cuspid'], df['err_z_m'], s=25, color='#ff7f0e', alpha=0.8)
    ax.axhline(df['err_z_m'].median(), color='red', ls='--', lw=2,
              label=f"median = {df['err_z_m'].median():.1f} m")
    ax.set_title('Vertical error per event')
    ax.set_xlabel('Event ID (cuspid)')
    ax.set_ylabel('Vertical error (m)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(df['err_h_m'].dropna(), bins=20, color='#1f77b4', alpha=0.8, edgecolor='white')
    ax.set_title('Horizontal error distribution')
    ax.set_xlabel('Horizontal error (m)')
    ax.set_ylabel('Event count')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.hist(df['err_z_m'].dropna(), bins=20, color='#ff7f0e', alpha=0.8, edgecolor='white')
    ax.set_title('Vertical error distribution')
    ax.set_xlabel('Vertical error (m)')
    ax.set_ylabel('Event count')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(folders['root'], f"location_errors.{cfg['plot_format']}"),
               dpi=cfg['plot_dpi'])
    plt.close(fig)


# ------------------------------------------------------------
# 3.10 Main
# ------------------------------------------------------------

def main():
    print("=" * 64)
    print("  Double-Difference Relocation Pipeline (hypoDD / relocDD-py)")
    print("=" * 64)

    folders = setup_output_folders(CONFIG['output_folder'])

    print("\n── STAGE 1: Load NLLoc catalogue & stations " + "─" * 19)
    stations = load_stations_dataframe(CONFIG['stations_file'], CONFIG['network'])
    df_cat, df_pha = load_catalog_and_phases(CONFIG)
    print(f"   {len(df_cat)} events, {len(df_pha)} phase picks, "
          f"{stations['station'].nunique()} stations")

    id_map_path = os.path.join(folders['root'], 'event_id_map.csv')
    df_cat[['cuspid', 'event_id', 'origin_time']].to_csv(id_map_path, index=False)

    print("\n── STAGE 2: Derive MAXSEPE " + "─" * 37)
    maxsepe_km = derive_maxsepe(df_cat, CONFIG, folders)

    print("\n── STAGE 3: Write hypoDD-format inputs " + "─" * 25)
    write_stations_dat(stations, os.path.join(folders['root'], 'stations.dat'))
    write_phase_dat(df_cat, df_pha, os.path.join(folders['root'], 'phase.dat'))

    print("\n── STAGE 4: Cross-correlation (dt.cc) " + "─" * 26)
    dt_by_pair = build_dt_cc(df_cat, df_pha, maxsepe_km, CONFIG, folders)
    write_dt_cc(dt_by_pair, os.path.join(folders['root'], 'dt.cc'))

    print("\n── STAGE 5: Write relocDD-py config " + "─" * 28)
    write_ph2dt_inp(os.path.join(folders['root'], 'ph2dt.inp'), maxsepe_km, CONFIG)
    write_hypodd_inp(os.path.join(folders['root'], 'hypoDD.inp'), CONFIG)
    run_inp_path = os.path.join(folders['root'], 'run.inp')
    write_run_inp(run_inp_path, folders, CONFIG)
    print(f"   ph2dt.inp, hypoDD.inp, run.inp written to {folders['root']}")

    print("\n── STAGE 6: Run ph2dt + hypoDD (relocDD-py) " + "─" * 20)
    run_relocdd(run_inp_path, CONFIG)

    print("\n── STAGE 7: Build relocated catalogue " + "─" * 26)
    df_out = build_relocated_catalog(df_cat, folders, CONFIG)
    plot_relocation_shift(df_out, CONFIG, folders)

    print("\n── STAGE 8: Post-relocation diagnostic plots " + "─" * 19)
    if df_out.empty:
        print("   ⚠️  No relocated events -- diagnostic plots skipped")
    else:
        plot_shift_statistics(df_out, CONFIG, folders)
        plot_cluster_diagnostics(folders, CONFIG)
        plot_rms_before_after(df_cat, folders, CONFIG)
        plot_location_errors(folders, CONFIG)
        print(f"   ✅ shift_statistics, cluster_diagnostics, rms_before_after, "
              f"location_errors -> {folders['root']}")

    print("\n" + "=" * 64)
    print("  Done.")
    print(f"  Relocated catalogue : {os.path.join(folders['root'], 'relocated_catalog.csv')}")
    print(f"  hypoDD.reloc (raw)  : {os.path.join(folders['relocdd_out'], 'EDD', 'tradouts', 'hypoDD.reloc')}")
    print("=" * 64)


if __name__ == '__main__':
    main()
