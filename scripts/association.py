# ============================================================
# PyOcto Association + Quality Control Pipeline
#
# Merges three notebooks into a single run:
#   1) association/pyocto.ipynb           -> phase association (PyOcto)
#   2) association/pyocto_qc_v3.ipynb     -> quality control QC-0..QC-4
#   3) association/visual_pyoctorev.ipynb -> Wadati & pseudo-distance plots
#      + per-event waveform display of the associated P and S picks
#
# Input  : EQTransformer picking output (ALL_results_combined.csv)
#          + stations.txt + mseed waveforms in the preprocessing stage's output folder
#
# Output structure :
#   output_folder/csv/pyocto_picks.csv                 (picks in PyOcto format)
#   output_folder/csv/pyocto_events.csv                (raw associator output)
#   output_folder/csv/pyocto_assignments.csv           (pick -> event)
#   output_folder/csv/pyocto_catalog.csv               (pre-QC catalogue)
#   output_folder/csv/pyoctorev_assignments.csv        (picks passing QC)
#   output_folder/csv/pyoctorev_catalog.csv            (catalogue passing QC)
#   output_folder/csv/qc_event_report.csv              (per-event QC metrics)
#   output_folder/events/<EVENT_ID>/                   (events PASSING QC)
#       pseudo_distance_<EVENT_ID>.png
#       wadati_<EVENT_ID>.png
#       waveform_<EVENT_ID>.png
#       picks_<EVENT_ID>.csv
#   output_folder/rejected/<EVENT_ID>/                 (events REJECTED by QC, same layout)
#   output_folder/summary/                             (global summary plots)
#   output_folder/nonlinloc/                           (NonLinLoc phase input)
#   output_folder/checkpoint.json                      (progress tracking)
#
# NOTE: This file is deliberately split into:
#   1) MODULE IMPORTS
#   2) CONFIGURATION (loaded from config/config.yaml -- see CONFIG_YAML_PATH)
#   3) PROCESS (utilities, association, QC, plotting, main pipeline)
# CONFIG is loaded from the 'association' section of the shared
# config/config.yaml (see pipeline.py at the project root, which runs every
# stage from that same file) -- edit the YAML to change parameters, not this
# file; the CONFIG dict keys and every function signature below are
# unchanged, so PROCESS never needs to be touched for a config-only change.
# ============================================================

# ============================================================
# 1) MODULE IMPORTS
# ============================================================

import os
import json
import math
import time
import warnings
from itertools import combinations
from datetime import timedelta

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pyocto
from obspy import read, Stream, UTCDateTime
from geopy.distance import geodesic

warnings.filterwarnings('ignore')

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

CONFIG = _all_cfg['association']
CONFIG['network'] = _all_cfg['network']   # single global network code (see stations.txt)


# ============================================================
# 3) PROCESS
# ============================================================

# ------------------------------------------------------------
# 3.1 General utilities (folders, checkpoint, geometry, fitting)
# ------------------------------------------------------------

def setup_output_folders(output_folder):
    """Create the main output folder structure."""
    folders = {
        'root'     : output_folder,
        'csv'      : os.path.join(output_folder, "csv"),
        'events'   : os.path.join(output_folder, "events"),
        'rejected' : os.path.join(output_folder, "rejected"),
        'summary'  : os.path.join(output_folder, "summary"),
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    return folders


def setup_event_folder(base_folder, event_id):
    """Create a dedicated folder for one event: <base_folder>/<EVENT_ID>/"""
    path = os.path.join(base_folder, event_id)
    os.makedirs(path, exist_ok=True)
    return path


def make_event_id(event_idx):
    """Stable, sortable naming for per-event folders and files."""
    return f"ev{int(event_idx):06d}"


def load_checkpoint(checkpoint_file):
    """
    Load run progress. This is the only progress tracker used by the pipeline.

    Return dict:
      'associated_days' : set of 'YYYY-MM-DD' dates whose association completed
      'plotted_events'  : set of event_id values whose plots were produced

    Missing or corrupt file -> both empty, meaning start from scratch.
    """
    empty = {'associated_days': set(), 'plotted_events': set()}
    if not os.path.exists(checkpoint_file):
        return empty
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        return {
            'associated_days': set(data.get('associated_days', [])),
            'plotted_events' : set(data.get('plotted_events', [])),
        }
    except Exception:
        return empty


def save_checkpoint(checkpoint_file, checkpoint):
    """Save run progress (associated days + plotted events)."""
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'associated_days' : sorted(checkpoint['associated_days']),
            'plotted_events'  : sorted(checkpoint['plotted_events']),
            'n_days'          : len(checkpoint['associated_days']),
            'n_plotted'       : len(checkpoint['plotted_events']),
            'last_update'     : str(pd.Timestamp.now()),
        }, f, indent=2)


def append_csv(path, df):
    """Append rows to a CSV (the header is written only if the file is new)."""
    if df is None or len(df) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def load_stations_dataframe(path, network):
    """
    Load stations.txt (columns: station latitude longitude elevation_m) ->
    DataFrame with columns: id, network, station, latitude, longitude,
    elevation_m. network is a single global constant (config.yaml's
    top-level 'network' key) since stations.txt carries no per-station
    network code -- every station on this array shares one.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"stations.txt not found: {path}")

    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            sta, lat, lon, elev = line.split()
            rows.append({
                'id'          : sta,
                'network'     : network,
                'station'     : sta,
                'latitude'    : float(lat),
                'longitude'   : float(lon),
                'elevation_m' : float(elev),
            })
    return pd.DataFrame(rows)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def max_station_distance_km(stations_df, return_pair=False):
    """
    Largest inter-station distance (km) -- the network 'aperture', used as the
    base quantity for almost every derived geometry parameter.
    """
    max_dist = 0.0
    pair = (None, None)
    for (_, s1), (_, s2) in combinations(stations_df.iterrows(), 2):
        d = geodesic(
            (s1['latitude'], s1['longitude']),
            (s2['latitude'], s2['longitude'])
        ).km
        if d > max_dist:
            max_dist, pair = d, (s1['station'], s2['station'])
    return (max_dist, pair) if return_pair else max_dist


def derive_area_padding_km(stations, cutoff_km, n_required, mean_lat,
                            lat_min, lat_max, lon_min, lon_max,
                            cap_km, step_km=1.0):
    """
    Per-direction (N/S/E/W) padding beyond the station bounding box.

    Walks outward from the midpoint of each box edge until fewer than
    n_required stations remain within cutoff_km of the probe point -- the
    same eligibility rule PyOcto's associator applies when deciding whether a
    candidate event has enough usable picks. This ties the search box exactly
    to the associator's own reach instead of an arbitrary fraction of the
    aperture, and is capped at cap_km so a network with a very loose cutoff
    doesn't produce an impractically large box.

    Returns a dict {'north', 'south', 'east', 'west'} of padding in km.
    """
    mid_lat = (lat_min + lat_max) / 2.0
    mid_lon = (lon_min + lon_max) / 2.0
    lats = stations['latitude'].to_numpy()
    lons = stations['longitude'].to_numpy()

    def count_within(lat, lon):
        d = np.array([haversine_km(lat, lon, la, lo) for la, lo in zip(lats, lons)])
        return int((d <= cutoff_km).sum())

    def walk(lat0, lon0, dlat_unit, dlon_unit):
        cos_lat0 = max(math.cos(math.radians(lat0)), 1e-6)
        d = 0.0
        while d < cap_km:
            d += step_km
            lat = lat0 + dlat_unit * (d / 111.19)
            lon = lon0 + dlon_unit * (d / (111.19 * cos_lat0))
            if count_within(lat, lon) < n_required:
                return min(d - step_km, cap_km)
        return cap_km

    return {
        'south': walk(lat_min, mid_lon, -1, 0),
        'north': walk(lat_max, mid_lon,  1, 0),
        'west' : walk(mid_lat, lon_min, 0, -1),
        'east' : walk(mid_lat, lon_max, 0,  1),
    }


def azimuth_deg(lat1, lon1, lat2, lon2):
    """Azimuth (degrees, 0 = north, clockwise) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = (math.cos(phi1) * math.sin(phi2)
         - math.sin(phi1) * math.cos(phi2) * math.cos(dlam))
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def azimuthal_gap(azimuths):
    """
    Largest azimuthal gap (degrees) between stations as seen from the epicentre.
    A standard location-quality metric: < 180 deg is generally considered
    well surrounded.
    """
    az = sorted(a for a in azimuths if pd.notna(a))
    if len(az) < 2:
        return 360.0
    gaps = [az[i + 1] - az[i] for i in range(len(az) - 1)]
    gaps.append(360.0 - az[-1] + az[0])   # the gap that wraps through north
    return float(max(gaps))


def derive_network_geometry(stations, cfg):
    """
    Derive every geometry parameter from the station layout (stations.txt), so
    the pipeline adapts itself to a different network or dataset.

    Explicit (non-None) values in CONFIG always take precedence.

    Base quantities:
      APERTURE = largest inter-station distance
      CUTOFF   = ceil(APERTURE * cutoff_factor)
      Z_MAX    = clip(zmax_factor * APERTURE, zmax_min_km, zmax_max_km)
      R_MAX    = sqrt(CUTOFF^2 + Z_MAX^2)
      K_PSEUDO = Vp*Vs/(Vp-Vs)
    """
    vp = cfg['vp']
    vs = vp / cfg['vpvs_ratio']
    if vp <= vs:
        raise ValueError("vp must be greater than vs (check vpvs_ratio)")
    k_pseudo = vp * vs / (vp - vs)

    aperture, pair = max_station_distance_km(stations, return_pair=True)

    # --- Association cutoff distance ---
    cutoff = cfg.get('association_cutoff_km')
    auto_cutoff = cutoff is None
    if auto_cutoff:
        cutoff = float(math.ceil(aperture * cfg['cutoff_factor']))

    # --- Maximum search depth ---
    zlim = cfg.get('zlim_km')
    auto_zlim = zlim is None
    if auto_zlim:
        z_max = float(np.clip(round(cfg['zmax_factor'] * aperture, -1),
                              cfg['zmax_min_km'], cfg['zmax_max_km']))
        # The top of the volume must lie above the highest station, otherwise
        # that station sits outside the travel-time grid and NonLinLoc cannot
        # predict arrivals for it -- every event then fails to locate. Depth is
        # positive downwards, so a station above sea level has negative depth.
        d = cfg['locgrid_node_spacing_km']
        max_elev_km = float(stations['elevation_m'].max()) / 1000.0
        z_min = -math.ceil((max_elev_km + cfg['zmin_margin_km']) / d) * d
        zlim = (z_min, z_max)
    zlim = (float(zlim[0]), float(zlim[1]))
    z_max = zlim[1]

    # --- Lat/lon area padding (per direction, PyOcto-eligibility-driven) ---
    mean_lat    = float(stations['latitude'].mean())
    lat_min_sta = float(stations['latitude'].min())
    lat_max_sta = float(stations['latitude'].max())
    lon_min_sta = float(stations['longitude'].min())
    lon_max_sta = float(stations['longitude'].max())
    pad_deg = cfg.get('area_padding_deg')
    auto_pad = pad_deg is None
    if auto_pad:
        pad_cap_km = cfg.get('area_padding_cap_km')
        if pad_cap_km is None:
            pad_cap_km = aperture
        pad_km_dir = derive_area_padding_km(
            stations, cutoff, cfg['n_p_and_s_picks'], mean_lat,
            lat_min_sta, lat_max_sta, lon_min_sta, lon_max_sta,
            pad_cap_km, cfg['area_padding_step_km'])
    else:
        pad_cap_km = None
        cos_lat_override = max(math.cos(math.radians(mean_lat)), 1e-6)
        pad_km_dir = {'south': float(pad_deg) * 111.19, 'north': float(pad_deg) * 111.19,
                      'west' : float(pad_deg) * 111.19 * cos_lat_override,
                      'east' : float(pad_deg) * 111.19 * cos_lat_override}

    cos_lat = max(math.cos(math.radians(mean_lat)), 1e-6)
    pad_lat_s_deg = pad_km_dir['south'] / 111.19
    pad_lat_n_deg = pad_km_dir['north'] / 111.19
    pad_lon_w_deg = pad_km_dir['west']  / (111.19 * cos_lat)
    pad_lon_e_deg = pad_km_dir['east']  / (111.19 * cos_lat)

    # --- Lat/lon search box (station bbox + per-direction padding) ---
    lat_range = [lat_min_sta - pad_lat_s_deg, lat_max_sta + pad_lat_n_deg]
    lon_range = [lon_min_sta - pad_lon_w_deg, lon_max_sta + pad_lon_e_deg]

    # --- Projection origin for the flat-earth SIMPLE transform ---
    # The station centroid keeps the projection most accurate where the events
    # are, and is fully determined by the network -- nothing is chosen by hand.
    trans_lat = float(stations['latitude'].mean())
    trans_lon = float(stations['longitude'].mean())

    # --- Largest hypocentral distance within the search volume ---
    # Computed AFTER grid clipping so that time_before and velocity_tolerance are
    # derived from the volume actually searched rather than the pre-clip volume
    # (otherwise both end up looser than intended).
    r_max = math.sqrt(cutoff ** 2 + z_max ** 2)

    # --- time_before: the origin may precede the first pick by the S travel time ---
    time_before = cfg.get('time_before')
    auto_tb = time_before is None
    if auto_tb:
        time_before = float(math.ceil((r_max / vs) * cfg['time_before_factor'] / 10.0) * 10)

    # --- Velocity model tolerance ---
    tol = cfg.get('velocity_tolerance')
    auto_tol = tol is None
    if auto_tol:
        tol = float(np.clip(cfg['velocity_tolerance_factor'] * (r_max / vp),
                            cfg['velocity_tolerance_min'],
                            cfg['velocity_tolerance_max']))

    # --- QC-1: Ts-Tp bound ---
    max_ts_tp = cfg.get('max_ts_tp_sec')
    auto_ts_tp = max_ts_tp is None
    if auto_ts_tp:
        ref_dist = aperture if cfg['max_ts_tp_reference'] == 'aperture' else r_max
        max_ts_tp = float(ref_dist / k_pseudo)

    # --- Vp/Vs reference for QC-3 and the Wadati plot ---
    vpvs_ref = cfg.get('vpvs_ref') or cfg['vpvs_ratio']

    geom = {
        'trans_lat'     : trans_lat,
        'trans_lon'     : trans_lon,
        'vp'            : vp,
        'vs'            : vs,
        'k_pseudo'      : k_pseudo,
        'aperture_km'   : aperture,
        'aperture_pair' : pair,
        'cutoff_km'     : cutoff,
        'zlim_km'       : zlim,
        'z_max_km'      : z_max,
        'r_max_km'      : r_max,
        'pad_km_dir'    : pad_km_dir,
        'pad_cap_km'    : pad_cap_km,
        'time_before'   : time_before,
        'tolerance'     : tol,
        'max_ts_tp_sec' : max_ts_tp,
        'vpvs_ref'      : vpvs_ref,
        'vpvs_ratio'    : cfg['vpvs_ratio'],
        'lat_range'     : tuple(lat_range),
        'lon_range'     : tuple(lon_range),
    }

    def _tag(is_auto):
        return "auto " if is_auto else "MANUAL"

    print("   ┌─ Network geometry (derived from stations.txt) " + "─" * 15)
    print(f"   │ Stations                   : {len(stations)}")
    print(f"   │ APERTURE (max separation)  : {aperture:8.2f} km  ({pair[0]}–{pair[1]})")
    print(f"   │ Vp / Vs                    : {vp:8.3f} / {vs:.3f} km/s "
          f"(Vp/Vs={cfg['vpvs_ratio']})")
    print(f"   │ K_PSEUDO = Vp*Vs/(Vp-Vs)   : {k_pseudo:8.4f} km/s")
    print("   ├─ Derived parameters " + "─" * 42)
    print(f"   │ {_tag(auto_cutoff)} association_cutoff  : {cutoff:8.1f} km   "
          f"(= APERTURE x {cfg['cutoff_factor']})")
    print(f"   │ {_tag(auto_zlim)} zlim                : {zlim[0]:6.1f} .. {zlim[1]:.1f} km")
    cap_str = f"cap={pad_cap_km:.0f} km" if pad_cap_km is not None else "manual"
    print(f"   │ {_tag(auto_pad)} area padding (km)   : "
          f"N {pad_km_dir['north']:.0f}  S {pad_km_dir['south']:.0f}  "
          f"E {pad_km_dir['east']:.0f}  W {pad_km_dir['west']:.0f}   ({cap_str})")
    print(f"   │        R_MAX (hypocentral) : {r_max:8.1f} km")
    print(f"   │ {_tag(auto_tb)} time_before         : {time_before:8.1f} s    "
          f"(= R_MAX/Vs x {cfg['time_before_factor']})")
    print(f"   │ {_tag(auto_tol)} velocity_tolerance  : {tol:8.2f} s")
    print(f"   │ {_tag(auto_ts_tp)} QC-1 max Ts-Tp      : {max_ts_tp:8.2f} s    "
          f"(= {cfg['max_ts_tp_reference']}/K_PSEUDO)")
    print(f"   │        projection origin   : lat {trans_lat:.4f}, lon {trans_lon:.4f} "
          f"(station centroid)")
    print("   └" + "─" * 62)

    return geom


def fit_with_intercept(x, y):
    """OLS fit y = slope*x + intercept. Return (slope, intercept)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return np.nan, np.nan
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def fit_r2(x, y, slope, intercept):
    """Coefficient of determination for the linear fit above."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or not np.isfinite(slope) or not np.isfinite(intercept):
        return np.nan
    y_hat  = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def pseudo_dist_from_ts_tp(delta_ts_tp, vp, vs):
    """
    Pseudo distance from the Ts-Tp difference:
        d = (Vp*Vs / (Vp-Vs)) * (Ts - Tp)
    Independent of the hypocentre solution.
    """
    if pd.isna(delta_ts_tp) or vp <= vs:
        return np.nan
    return (vp * vs / (vp - vs)) * float(delta_ts_tp)


def pseudo_dist_from_hypocenter(orig_lat, orig_lon, orig_depth, sta_lat, sta_lon):
    """Hypocentral distance from the catalogue: sqrt(epi^2 + depth^2)."""
    if any(pd.isna(v) for v in [orig_lat, orig_lon, orig_depth, sta_lat, sta_lon]):
        return np.nan
    epi = haversine_km(orig_lat, orig_lon, sta_lat, sta_lon)
    return math.sqrt(epi ** 2 + float(orig_depth) ** 2)


def make_logger(cfg, t0):
    """Stage logger with elapsed-time markers (same style as pick.py)."""
    def log_step(msg):
        if cfg.get('verbose_timing'):
            print(f"  [{time.strftime('%H:%M:%S')}] (+{time.time() - t0:6.1f}s) {msg}",
                  flush=True)
    return log_step


# ------------------------------------------------------------
# 3.2 Stage 1 -- Prepare picks and stations
# ------------------------------------------------------------

def build_pyocto_picks(picks_csv, cfg):
    """
    Convert the EQTransformer picking output (STEAD-style, one row per detected
    event containing a P and S pair) into a PyOcto pick table (one row per pick).

    Output columns: pick_id, network, station, phase, peak_time, peak_value, snr, id
    """
    if not os.path.exists(picks_csv):
        raise FileNotFoundError(f"Picks file not found: {picks_csv}")

    eqt = pd.read_csv(picks_csv, low_memory=False)
    print(f"   Picking result rows : {len(eqt):,}")

    frames = []
    for phase, t_col, prob_col, snr_col in [
        ('P', 'p_arrival_time', 'p_probability', 'p_snr'),
        ('S', 's_arrival_time', 's_probability', 's_snr'),
    ]:
        sub = eqt[eqt[t_col].notna()].copy()
        if sub.empty:
            continue
        frame = pd.DataFrame({
            'pick_id'   : sub.index.astype(str) + f"_{phase}",
            'network'   : sub['network'],
            'station'   : sub['station'],
            'phase'     : phase,
            'peak_time' : pd.to_datetime(sub[t_col], errors='coerce'),
            'peak_value': sub[prob_col].fillna(0.0),
            'snr'       : sub[snr_col] if snr_col in sub.columns else np.nan,
        })
        frames.append(frame)

    picks = pd.concat(frames, ignore_index=True)
    picks = picks[picks['peak_time'].notna()]

    n_before = len(picks)

    # No probability filtering -- see the note in the CONFIG block. All picks are
    # passed to the associator; filtering is done by the associator, QC-0..QC-4,
    # and visual inspection of the per-event waveform plots.

    # --- SNR filter (optional, disabled by default) ---
    if cfg.get('min_p_snr') is not None:
        drop = (picks['phase'] == 'P') & (picks['snr'].fillna(np.inf) < cfg['min_p_snr'])
        picks = picks[~drop]
    if cfg.get('min_s_snr') is not None:
        drop = (picks['phase'] == 'S') & (picks['snr'].fillna(np.inf) < cfg['min_s_snr'])
        picks = picks[~drop]

    picks = picks.sort_values('peak_time').reset_index(drop=True)
    picks['id'] = picks['network'].astype(str) + "." + picks['station'].astype(str)

    n_p, n_s = (picks['phase'] == 'P').sum(), (picks['phase'] == 'S').sum()
    if len(picks) < n_before:
        print(f"   Picks after SNR filter   : {len(picks):,} of {n_before:,}")
    print(f"   Picks into associator    : {len(picks):,} "
          f"(P {n_p:,} | S {n_s:,}) -- no probability filter")
    print(f"     probability P : median {picks.loc[picks['phase']=='P','peak_value'].median():.3f}"
          f" | S : median {picks.loc[picks['phase']=='S','peak_value'].median():.3f}"
          f"  (recorded only, not used for filtering)")

    return picks


def prepare_associator_picks(picks):
    """
    Convert the pick table into the format PyOcto expects:
      station, time (unix seconds), probability, phase
    """
    pyocto_picks = picks.rename(columns={
        'peak_time'  : 'time',
        'peak_value' : 'probability',
    }).copy()
    pyocto_picks['time'] = pd.to_datetime(pyocto_picks['time']).astype('int64') / 1e9

    required = ['station', 'time', 'probability', 'phase']
    missing  = [c for c in required if c not in pyocto_picks.columns]
    if missing:
        raise ValueError(f"Required pick columns are missing: {missing}")

    return pyocto_picks.sort_values('time').reset_index(drop=True)


def prepare_associator_stations(stations):
    """
    Format stations for PyOcto: id, station, latitude, longitude,
    elevation (km), p_residual, s_residual.
    """
    pyocto_stations = stations.rename(columns={'elevation_m': 'elevation'})[
        ['station', 'latitude', 'longitude', 'elevation']
    ].copy()
    pyocto_stations['id']         = pyocto_stations['station']
    pyocto_stations['elevation']  = pyocto_stations['elevation'] / 1000.0  # m -> km
    pyocto_stations['p_residual'] = 0.0
    pyocto_stations['s_residual'] = 0.0
    return pyocto_stations


def build_velocity_model(geom):
    """Build the VelocityModel0D from the derived geometry."""
    return pyocto.VelocityModel0D(
        p_velocity=geom['vp'],
        s_velocity=geom['vs'],
        tolerance=geom['tolerance'],
        association_cutoff_distance=geom['cutoff_km'],
    )


def build_associator(stations, velocity_model, geom, cfg):
    """Build the OctoAssociator from the derived search box."""
    print(f"   Area lat  : {geom['lat_range'][0]:.4f} .. {geom['lat_range'][1]:.4f}")
    print(f"   Area lon  : {geom['lon_range'][0]:.4f} .. {geom['lon_range'][1]:.4f}")
    print(f"   Depth     : {geom['zlim_km'][0]:.1f} .. {geom['zlim_km'][1]:.1f} km")

    return pyocto.OctoAssociator.from_area(
        lat=geom['lat_range'],
        lon=geom['lon_range'],
        zlim=geom['zlim_km'],
        time_before=geom['time_before'],
        velocity_model=velocity_model,
        n_picks=cfg['n_picks'],
        n_p_and_s_picks=cfg['n_p_and_s_picks'],
        n_threads=cfg['n_threads'] or os.cpu_count(),
    )


# ------------------------------------------------------------
# 3.3 Stage 2 -- Phase association (PyOcto)
# ------------------------------------------------------------

def split_pick_batches_by_day(pyocto_picks):
    """
    Split picks by UTC date -> sorted list of (day_str, dataframe).
    Day boundaries are used because events last seconds to minutes, so none is
    ever split, and the date doubles as the checkpoint unit.
    """
    day = pd.to_datetime(pyocto_picks['time'], unit='s').dt.floor('D')
    return [(str(d.date()), g) for d, g in pyocto_picks.groupby(day, sort=True)]


def run_association(associator, pyocto_picks, pyocto_stations, cfg,
                    path_events, path_assignments, checkpoint):
    """
    Associate one day at a time, checkpointing per day.

    Each completed day is appended to the CSVs and its date recorded in the
    checkpoint, so an interrupted run resumes from the next day without redoing
    finished work.

    Event numbering stays continuous across days and across runs: the offset
    starts at the largest existing idx in the CSV plus one.
    """
    done = checkpoint['associated_days']

    # --- Restore CSV <-> checkpoint consistency ---
    # Rows are appended to the CSV before the checkpoint is saved, leaving a
    # narrow window in which the process can die after the append but before the
    # date is recorded. Untreated, that day would be re-associated and its rows
    # duplicated. Because every row carries an assoc_day column, rows whose day is
    # absent from the checkpoint can be dropped -- keeping the checkpoint as the
    # single source of truth.
    for path in (path_events, path_assignments):
        if not os.path.exists(path):
            continue
        try:
            df_old = pd.read_csv(path)
        except Exception:
            continue
        if 'assoc_day' not in df_old.columns:
            continue
        keep = df_old['assoc_day'].astype(str).isin(done)
        if not keep.all():
            print(f"   🧹 dropped {(~keep).sum():,} rows from unrecorded days "
                  f"({os.path.basename(path)}) -- those days will be re-associated")
            df_old[keep].to_csv(path, index=False)

    batches = split_pick_batches_by_day(pyocto_picks)
    todo    = [(d, g) for d, g in batches if d not in done]

    print(f"   Days containing picks : {len(batches)}")
    print(f"   Already associated    : {len(batches) - len(todo)} (skipped)")
    print(f"   To be processed       : {len(todo)}")

    # Continue numbering from the previous run
    event_offset = 0
    if os.path.exists(path_events):
        try:
            prev = pd.read_csv(path_events, usecols=['idx'])
            if len(prev):
                event_offset = int(prev['idx'].max()) + 1
        except Exception:
            pass

    n_failed = 0
    for day_str, batch_picks in tqdm(todo, desc="Associating", unit="day"):
        if len(batch_picks) >= cfg['n_picks']:
            try:
                batch_events, batch_assignments = associator.associate(
                    batch_picks.copy(), pyocto_stations
                )
            except Exception as e:
                n_failed += 1
                tqdm.write(f"  ⚠️  {day_str} failed: {e}")
                continue   # date NOT recorded -> retried on the next run

            if len(batch_events) > 0:
                if 'idx' in batch_events.columns:
                    batch_events['idx'] += event_offset
                if 'event_idx' in batch_assignments.columns:
                    batch_assignments['event_idx'] += event_offset
                event_offset += len(batch_events)

                # assoc_day marks the origin of each row -> used to restore
                # consistency if the process dies before the checkpoint is saved
                batch_events['assoc_day']      = day_str
                batch_assignments['assoc_day'] = day_str
                append_csv(path_events, batch_events)
                append_csv(path_assignments, batch_assignments)

        # This day is complete (including when it has too few picks or no events)
        checkpoint['associated_days'].add(day_str)
        save_checkpoint(cfg['checkpoint_file'], checkpoint)

    if n_failed:
        print(f"   ⚠️  {n_failed} days failed to associate "
              f"(not recorded, will be retried on the next run)")

    # --- Read back all accumulated results (including previous runs) ---
    events = (pd.read_csv(path_events) if os.path.exists(path_events)
              else pd.DataFrame())
    assignments = (pd.read_csv(path_assignments) if os.path.exists(path_assignments)
                   else pd.DataFrame())

    # assoc_day is internal to the checkpoint -- do not pass it on to QC
    return (events.drop(columns=['assoc_day'], errors='ignore'),
            assignments.drop(columns=['assoc_day'], errors='ignore'))


def build_catalog(events, assignments, associator, cfg):
    """
    Convert the associator output (local x/y/z km) into a catalogue with
    lat/lon/depth and the pick count per event.
    """
    if len(events) == 0:
        return pd.DataFrame(), events

    if 'time' in events.columns:
        events['time'] = pd.to_datetime(events['time'], unit='s')

    # Project local coordinates back to lat/lon (using the associator's CRS)
    events = associator.transform_events(events)

    # Recompute the pick count per event from the assignments
    events = events.drop(columns=['event_idx', 'picks', 'n_picks'], errors='ignore')
    if len(assignments) > 0:
        pick_counts = (assignments.groupby('event_idx').size()
                       .reset_index(name='n_picks'))
        events = events.merge(pick_counts, left_on='idx', right_on='event_idx',
                              how='left')
        events['n_picks'] = events['n_picks'].fillna(0).astype(int)
        events = events.drop(columns=['event_idx'], errors='ignore')
    else:
        events['n_picks'] = 0

    catalog = pd.DataFrame({
        'event_index' : events['idx'],
        'time'        : events['time'],
        'latitude'    : events['latitude'],
        'longitude'   : events['longitude'],
        'depth_km'    : events['z'],
        'x_km'        : events['x'],
        'y_km'        : events['y'],
        'z_km'        : events['z'],
        'n_picks'     : events['n_picks'],
    })
    for col in ['rms', 'score', 'n_p_and_s_picks']:
        if col in events.columns:
            catalog[col] = events[col]

    return catalog, events


# ------------------------------------------------------------
# 3.4 Stage 3 -- QC-0: dedup double picks and require both P and S
# ------------------------------------------------------------

def drop_single_phase_stations(df_assign):
    """
    QC-0d: drop every pick from a station contributing only one phase (P or S)
    to an event.

    Such stations cannot enter the Wadati diagram, the pseudo-distance plot or
    QC-4, all of which operate on P-S pairs.

    Missing phases are not interpolated from the hypocentral distance: a value
    derived from the location being solved for would propagate into the
    NonLinLoc input and bias the resulting hypocentre.

    Return: (df, number_of_picks_removed)
    """
    n_before = len(df_assign)
    pair_ok = (df_assign.groupby(['event_idx', 'station'])['phase']
                        .transform(lambda x: {'P', 'S'}.issubset(set(x))))
    df = df_assign[pair_ok].reset_index(drop=True)
    return df, n_before - len(df)


def qc0_clean_assignments(df_assign, cfg):
    """
    QC-0d: drop single-phase stations (always applied, even if run_qc0 is off).
    QC-0a: drop double picks (same event, station and phase within
           double_pick_window_sec) -- the highest probability is kept.
    QC-0b: drop events without at least one P AND one S.
    QC-0c: drop events whose P-S pairs come from < min_ps_pairs distinct stations.

    Return: (df_assign_clean, violate_qc0, stats_dict)
    """
    stats = {}
    df = df_assign.copy()
    df['pick_dt'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_localize(None)

    df, stats['n_single_phase_removed'] = drop_single_phase_stations(df)

    if not cfg.get('run_qc0'):
        stats['n_dup_removed'] = 0
        stats['n_qc0b_events'] = 0
        return df.drop(columns=['pick_dt']), set(), stats

    # --- QC-0a: dedup ---
    n_before = len(df)
    df = df.sort_values(['event_idx', 'station', 'phase', 'probability'],
                        ascending=[True, True, True, False])

    window = cfg['double_pick_window_sec']

    def _mark_dup(grp):
        # First row = highest probability; other rows inside the window are dropped.
        grp = grp.reset_index(drop=True)
        if len(grp) == 1:
            return grp
        dt = (grp['pick_dt'] - grp.loc[0, 'pick_dt']).dt.total_seconds().abs()
        keep = dt > window
        keep.iloc[0] = True
        return grp[keep]

    df = (df.groupby(['event_idx', 'station', 'phase'], group_keys=False)
            .apply(_mark_dup)
            .reset_index(drop=True))

    stats['n_dup_removed'] = n_before - len(df)

    # --- QC-0b: an event must contain both P and S ---
    phases_per_event = df.groupby('event_idx')['phase'].apply(set)
    has_p    = phases_per_event.apply(lambda x: 'P' in x)
    has_s    = phases_per_event.apply(lambda x: 'S' in x)
    has_both = has_p & has_s

    violate_qc0 = set(phases_per_event[~has_both].index)
    stats['n_only_p']      = int((has_p & ~has_s).sum())
    stats['n_only_s']      = int((~has_p & has_s).sum())
    stats['n_qc0b_events'] = len(violate_qc0)

    # --- QC-0c: minimum number of distinct stations with a complete P-S pair ---
    # The associator guarantees this via n_p_and_s_picks, but the QC-0a dedup can
    # reduce the count, so the requirement is re-enforced here.
    min_pairs = cfg.get('min_ps_pairs', 0)
    if min_pairs:
        per_sta = (df.groupby(['event_idx', 'station'])['phase']
                     .apply(lambda x: {'P', 'S'}.issubset(set(x))))
        n_pairs = per_sta.groupby('event_idx').sum()
        violate_qc0c = set(n_pairs[n_pairs < min_pairs].index)
        stats['n_qc0c_events'] = len(violate_qc0c - violate_qc0)
        stats['min_ps_pairs']  = min_pairs
        violate_qc0 |= violate_qc0c
    else:
        stats['n_qc0c_events'] = 0

    df_clean = df[~df['event_idx'].isin(violate_qc0)].drop(columns=['pick_dt'])

    return df_clean.reset_index(drop=True), violate_qc0, stats


# ------------------------------------------------------------
# 3.5 Stage 4 -- Per-event table (tp, ts, delta, pseudo_dist)
# ------------------------------------------------------------

def build_event_table(df_assign_clean, df_cat, df_sta, vp, vs, cfg):
    """
    Build a table with one row per (event, station) containing:
      tp, ts, delta_ts_tp, pseudo_dist_km, p_prob, s_prob,
      p_dt, s_dt (absolute times, needed for the waveform plots),
      origin_time, orig_lat, orig_lon, orig_depth, sta_lat, sta_lon
    """
    df = df_assign_clean.rename(columns={'time': 'pick_time_unix'}).copy()

    cat_cols = df_cat[['event_index', 'time', 'latitude', 'longitude', 'depth_km']].rename(
        columns={'time': 'origin_time', 'latitude': 'orig_lat',
                 'longitude': 'orig_lon', 'depth_km': 'orig_depth'}
    )
    df = df.merge(cat_cols, left_on='event_idx', right_on='event_index', how='left')

    df['pick_dt'] = (pd.to_datetime(df['pick_time_unix'], unit='s', utc=True)
                     .dt.tz_localize(None))

    df = df.merge(
        df_sta[['station', 'latitude', 'longitude']].rename(
            columns={'latitude': 'sta_lat', 'longitude': 'sta_lon'}),
        on='station', how='left'
    )
    df['arrival_rel_s'] = (df['pick_dt'] - df['origin_time']).dt.total_seconds()

    # --- Split P and S, then join per (event, station) ---
    p_tbl = (df[df['phase'] == 'P']
             .sort_values('probability', ascending=False)
             .drop_duplicates(subset=['event_idx', 'station'])
             [['event_idx', 'station', 'arrival_rel_s', 'pick_dt', 'probability',
               'sta_lat', 'sta_lon']]
             .rename(columns={'arrival_rel_s': 'tp', 'pick_dt': 'p_dt',
                              'probability': 'p_prob'}))

    s_tbl = (df[df['phase'] == 'S']
             .sort_values('probability', ascending=False)
             .drop_duplicates(subset=['event_idx', 'station'])
             [['event_idx', 'station', 'arrival_rel_s', 'pick_dt', 'probability']]
             .rename(columns={'arrival_rel_s': 'ts', 'pick_dt': 's_dt',
                              'probability': 's_prob'}))

    df_evt = p_tbl.merge(s_tbl, on=['event_idx', 'station'], how='outer')
    df_evt['delta_ts_tp'] = df_evt['ts'] - df_evt['tp']

    # Fill in station coordinates for S-only rows
    df_evt = df_evt.drop(columns=['sta_lat', 'sta_lon'], errors='ignore').merge(
        df_sta[['station', 'latitude', 'longitude']].rename(
            columns={'latitude': 'sta_lat', 'longitude': 'sta_lon'}),
        on='station', how='left'
    )

    # Origin information
    df_evt = df_evt.merge(
        df_cat[['event_index', 'time', 'latitude', 'longitude', 'depth_km']].rename(
            columns={'event_index': 'event_idx', 'time': 'origin_time',
                     'latitude': 'orig_lat', 'longitude': 'orig_lon',
                     'depth_km': 'orig_depth'}),
        on='event_idx', how='left'
    )

    # --- Both distance definitions, always computed ---
    df_evt['dist_ts_tp_km'] = df_evt['delta_ts_tp'].apply(
        lambda d: pseudo_dist_from_ts_tp(d, vp, vs))
    df_evt['dist_hypo_km'] = df_evt.apply(
        lambda r: pseudo_dist_from_hypocenter(
            r['orig_lat'], r['orig_lon'], r['orig_depth'],
            r['sta_lat'], r['sta_lon']),
        axis=1)

    col_of = {'ts_tp': 'dist_ts_tp_km', 'hypocenter': 'dist_hypo_km'}
    # Used by the pseudo-distance plot and the QC-3 fit
    df_evt['pseudo_dist_km'] = df_evt[col_of[cfg['dist_method_plot']]]
    # Used by the QC-2 geometry test
    df_evt['qc2_dist_km']    = df_evt[col_of[cfg['dist_method_qc2']]]

    # True epicentral distance (shown in the waveform plot headers)
    df_evt['epi_dist_km'] = df_evt.apply(
        lambda r: haversine_km(r['orig_lat'], r['orig_lon'], r['sta_lat'], r['sta_lon'])
        if not any(pd.isna(v) for v in
                   [r['orig_lat'], r['orig_lon'], r['sta_lat'], r['sta_lon']])
        else np.nan,
        axis=1)

    return df_evt.sort_values(['event_idx', 'tp']).reset_index(drop=True)


# ------------------------------------------------------------
# 3.6 Stage 5 -- QC-1, QC-2, QC-3, QC-4
# ------------------------------------------------------------

def qc1_ts_tp(df_evt, geom, cfg):
    """
    QC-1: reject events containing a P-S pair whose Ts-Tp lies outside the
    plausible range.

    The upper bound is derived from the network geometry:
        max Ts-Tp = reference_distance / (Vp*Vs/(Vp-Vs))
    so it adapts when the network or the assumed velocities change.
    """
    if not cfg.get('run_qc1'):
        return set()

    max_ts_tp = geom['max_ts_tp_sec']
    ps = df_evt.dropna(subset=['delta_ts_tp'])
    violate = set(ps[
        (ps['delta_ts_tp'] > max_ts_tp) |
        (ps['delta_ts_tp'] <= cfg['min_ts_tp_sec'])
    ]['event_idx'].unique())

    print(f"   QC-1 | Ts-Tp > {max_ts_tp:.2f}s or <= {cfg['min_ts_tp_sec']}s "
          f": {len(violate):,} events rejected")
    return violate


def qc2_pseudo_distance(df_evt, sta_coord, geom, cfg, folders):
    """
    QC-2: inter-station geometry consistency (triangle inequality).

    For every station pair (A,B) that picked the same event, the pseudo distance
    (derived from Ts-Tp and therefore independent of the location) must satisfy:
            |d_A - d_B|  <=  D_AB
    where D_AB is the physical station separation. A violation means the geometry
    is impossible.

    Mode 'physical'   : tested in km with a pick-error propagation tolerance.
    Mode 'percentile' : ratio |d_A-d_B|/D_AB against a percentile threshold.

    Note this is a necessary but not sufficient condition. Two distinct events
    close in time often still satisfy it; that case is handled by QC-4
    (origin-time consistency).

    Return: (violate_set, ev_max_ratio, threshold, all_ratios)
    """
    if not cfg.get('run_qc2'):
        return set(), {}, np.nan, np.array([])

    pseudo_lookup = (df_evt.dropna(subset=['qc2_dist_km'])
                     .groupby(['event_idx', 'station'])['qc2_dist_km']
                     .mean().to_dict())

    # Cache the true inter-station distances
    real_dist_cache = {}
    for sA, sB in combinations(list(sta_coord.keys()), 2):
        real_dist_cache[tuple(sorted([sA, sB]))] = haversine_km(
            sta_coord[sA]['latitude'], sta_coord[sA]['longitude'],
            sta_coord[sB]['latitude'], sta_coord[sB]['longitude'])

    # Physical tolerance: propagation of pick error into the distance difference.
    # d = K_PSEUDO * (Ts-Tp), so var(d_A - d_B) = K^2 * (2 sigma^2 + 2 sigma^2)
    # -> sigma_diff = K_PSEUDO * 2 * sigma_pick
    sigma_diff = geom['k_pseudo'] * 2.0 * cfg['qc2_pick_sigma_sec']
    tol_km     = cfg['qc2_n_sigma'] * sigma_diff

    all_ratios    = []
    all_excess    = []
    ev_max_ratio  = {}
    ev_max_excess = {}

    grouped = df_evt.dropna(subset=['qc2_dist_km']).groupby('event_idx')['station']
    for ev, stas in tqdm(grouped, desc="QC-2: inter-station geometry", unit="event"):
        stations_ev = stas.dropna().unique()
        if len(stations_ev) < 2:
            ev_max_ratio[ev] = np.nan
            ev_max_excess[ev] = np.nan
            continue

        ev_ratios, ev_excess = [], []
        for sA, sB in combinations(stations_ev, 2):
            dA = pseudo_lookup.get((ev, sA), np.nan)
            dB = pseudo_lookup.get((ev, sB), np.nan)
            if pd.isna(dA) or pd.isna(dB):
                continue
            dist_real = real_dist_cache.get(tuple(sorted([sA, sB])), np.nan)
            if pd.isna(dist_real) or dist_real == 0:
                continue
            diff = abs(dA - dB)
            ev_ratios.append(diff / dist_real)
            ev_excess.append(diff - dist_real)     # km beyond the geometric bound
            all_ratios.append(diff / dist_real)
            all_excess.append(diff - dist_real)

        ev_max_ratio[ev]  = max(ev_ratios) if ev_ratios else np.nan
        ev_max_excess[ev] = max(ev_excess) if ev_excess else np.nan

    all_ratios = np.array([r for r in all_ratios if np.isfinite(r)])
    all_excess = np.array([e for e in all_excess if np.isfinite(e)])
    if len(all_ratios) == 0:
        print("   QC-2 | no computable station pairs -> skipped")
        return set(), ev_max_ratio, np.nan, all_ratios

    if cfg['qc2_method'] == 'physical':
        violate = {ev for ev, x in ev_max_excess.items()
                   if pd.notna(x) and x > tol_km}
        threshold = 1.0   # reference line for the distribution plot
        print(f"   QC-2 | method 'physical' | pairs collected : {len(all_ratios):,}")
        print(f"   QC-2 | sigma_pick={cfg['qc2_pick_sigma_sec']}s -> "
              f"sigma_diff={sigma_diff:.2f} km | tolerance "
              f"{cfg['qc2_n_sigma']}sigma = {tol_km:.2f} km")
        print(f"   QC-2 | violated when |dA-dB| - D_AB > {tol_km:.2f} km "
              f"-> {len(violate):,} events rejected")
    else:
        threshold = cfg.get('qc2_threshold_ratio')
        if threshold is None:
            threshold = float(np.percentile(all_ratios, cfg['qc2_percentile']))
        violate = {ev for ev, mr in ev_max_ratio.items()
                   if pd.notna(mr) and mr > threshold}
        print(f"   QC-2 | method 'percentile' | pairs collected : {len(all_ratios):,}")
        print(f"   QC-2 | median={np.median(all_ratios):.3f}  "
              f"P90={np.percentile(all_ratios, 90):.3f}  "
              f"P95={np.percentile(all_ratios, 95):.3f}  "
              f"P99={np.percentile(all_ratios, 99):.3f}")
        print(f"   QC-2 | ratio threshold = {threshold:.3f} "
              f"-> {len(violate):,} events rejected")

    plot_qc2_distribution(all_ratios, all_excess, threshold, tol_km, cfg, folders)

    return violate, ev_max_ratio, threshold, all_ratios


def qc4_origin_time(df_evt, geom, cfg):
    """
    QC-4: inter-station origin-time consistency.

    Every station with a P-S pair can compute its own origin time:

        OT_i = Tp_i - (Ts_i - Tp_i) / (Vp/Vs - 1)

    Absolute Vp does not appear -- only the Vp/Vs RATIO -- so this test is
    insensitive to the crudeness of the homogeneous Vp assumption.

    If all stations recorded the SAME event, every OT_i must agree within pick
    error. A station that actually recorded a DIFFERENT event deviates by roughly
    the origin-time separation of the two events. This is the case QC-2 misses,
    since two distinct events can still yield a plausible distance geometry.

    The centre is estimated with the median and MAD, which are robust to outliers
    unlike mean/std, which would be pulled by the deviating station.

    Return: (violate_events, dropped_pairs, qc4_details)
        dropped_pairs = set of (event_idx, station), populated only when
        qc4_action == 'drop_station'
    """
    details       = {}
    violate       = set()
    dropped_pairs = set()

    if not cfg.get('run_qc4'):
        return violate, dropped_pairs, details

    vpvs = geom['vpvs_ratio']
    if vpvs <= 1.0:
        raise ValueError("vpvs_ratio must be > 1 to compute an origin time")
    denom = vpvs - 1.0

    df = df_evt.dropna(subset=['tp', 'ts']).copy()
    df['ot_station'] = df['tp'] - (df['ts'] - df['tp']) / denom

    for ev, grp in tqdm(df.groupby('event_idx'),
                        desc="QC-4: origin-time consistency", unit="event"):
        ot = grp['ot_station'].values
        if len(ot) < 2:
            details[ev] = {'ot_median': float(ot[0]) if len(ot) else np.nan,
                           'ot_mad': np.nan, 'ot_max_resid': np.nan,
                           'n_outlier': 0}
            continue

        med   = float(np.median(ot))
        mad   = float(np.median(np.abs(ot - med)))
        resid = np.abs(ot - med)

        # Threshold: the looser of the absolute bound and the adaptive MAD bound.
        # The absolute bound is a safety net when few stations are available (MAD
        # is fragile there); the MAD bound accommodates genuinely wider scatter.
        thr = max(cfg['qc4_abs_tol_sec'], cfg['qc4_n_mad'] * 1.4826 * mad)

        outlier_mask = resid > thr
        n_outlier    = int(outlier_mask.sum())

        details[ev] = {
            'ot_median'    : med,
            'ot_mad'       : mad,
            'ot_max_resid' : float(resid.max()),
            'ot_threshold' : thr,
            'n_outlier'    : n_outlier,
        }

        if n_outlier == 0:
            continue

        if cfg['qc4_action'] == 'drop_station':
            for sta in grp.loc[outlier_mask, 'station'].values:
                dropped_pairs.add((ev, sta))
        else:
            violate.add(ev)

    if cfg['qc4_action'] == 'drop_station':
        print(f"   QC-4 | deviating stations dropped : {len(dropped_pairs):,} "
              f"(events kept; min_ps_pairs re-checked afterwards)")
    else:
        print(f"   QC-4 | inconsistent inter-station origin time : "
              f"{len(violate):,} events rejected")

    return violate, dropped_pairs, details


def apply_qc4_station_drop(df_evt, df_assign_clean, dropped_pairs, cfg):
    """
    Remove the QC-4 outlier stations from the event and assignment tables, then
    re-enforce the min_ps_pairs requirement (QC-0c), so that station removal can
    never violate the minimum P-S pair count.

    Return: (df_evt, df_assign_clean, violate_after_drop)
    """
    if not dropped_pairs:
        return df_evt, df_assign_clean, set()

    drop_idx = pd.MultiIndex.from_tuples(sorted(dropped_pairs),
                                         names=['event_idx', 'station'])

    evt_idx = pd.MultiIndex.from_frame(df_evt[['event_idx', 'station']])
    df_evt  = df_evt[~evt_idx.isin(drop_idx)].copy()

    asg_idx = pd.MultiIndex.from_frame(df_assign_clean[['event_idx', 'station']])
    df_assign_clean = df_assign_clean[~asg_idx.isin(drop_idx)].copy()

    # --- Re-check min_ps_pairs ---
    min_pairs = cfg.get('min_ps_pairs', 0)
    violate_after = set()
    if min_pairs:
        n_pairs = (df_evt.dropna(subset=['tp', 'ts'])
                   .groupby('event_idx')['station'].nunique())
        all_ev  = set(df_evt['event_idx'].unique())
        ok_ev   = set(n_pairs[n_pairs >= min_pairs].index)
        violate_after = all_ev - ok_ev
        print(f"   QC-4 | after station removal, < {min_pairs} P-S pairs : "
              f"{len(violate_after):,} events rejected")

    return df_evt, df_assign_clean, violate_after


def qc3_is_bad(info, cfg, which='both'):
    """
    Judge the physical plausibility of one event from its QC-3 fits.

    which = 'ps'     -> travel-time fit only (distance vs tp)
            'wadati' -> Wadati fit only
            'both'   -> combined (used for the accept/reject decision)

    Criteria (those set to None in CONFIG are skipped):
      - slope <= 0                     -> impossible (arrival time decreasing with
                                          distance, or Vs > Vp). Always tested.
      - apparent Vp outside qc3_vp_range     -> implausible velocity
      - Vp/Vs outside qc3_vpvs_range         -> implausible ratio
      - intercept < -qc3_intercept_tol       -> origin time too far off
    """
    tol       = cfg.get('qc3_intercept_tol_sec')
    vp_range  = cfg.get('qc3_vp_range')
    vpvs_range= cfg.get('qc3_vpvs_range')

    slope_ps, intc_ps = info.get('slope_ps'), info.get('intc_ps')
    slope_wd, intc_wd = info.get('slope_wd'), info.get('intc_wd')
    vp_app, vpvs      = info.get('vp_app'),   info.get('vpvs')

    def _out(val, rng):
        return rng is not None and pd.notna(val) and not (rng[0] <= val <= rng[1])

    def _intc_bad(val):
        return tol is not None and pd.notna(val) and val < -tol

    ps_bad = (
        (pd.notna(slope_ps) and slope_ps <= 0) or
        _intc_bad(intc_ps) or _out(vp_app, vp_range)
    )
    wd_bad = (
        (pd.notna(slope_wd) and slope_wd <= 0) or
        _intc_bad(intc_wd) or _out(vpvs, vpvs_range)
    )

    if which == 'ps':
        return bool(ps_bad)
    if which == 'wadati':
        return bool(wd_bad)
    return bool(ps_bad or wd_bad)


def qc3_slope_intercept(df_evt, cfg):
    """
    QC-3: per-event linear fits with intercept
      - Pseudo-distance : tp          = slope * pseudo_dist + intercept
      - Wadati          : delta_ts_tp = slope * tp          + intercept
    Events are rejected according to qc3_is_bad().

    Return: (violate_set, qc3_details_dict) -- the details are computed even when
            QC-3 is disabled, since the per-event plots annotate them.
    """
    violate = set()
    details = {}

    for ev, grp in tqdm(df_evt.groupby('event_idx'),
                        desc="QC-3: fit slope/intercept", unit="event"):
        grp_ps = grp.dropna(subset=['tp', 'pseudo_dist_km'])
        grp_wd = grp.dropna(subset=['tp', 'delta_ts_tp'])

        slope_ps = intc_ps = r2_ps = np.nan
        slope_wd = intc_wd = r2_wd = np.nan

        if len(grp_ps) >= 2:
            slope_ps, intc_ps = fit_with_intercept(
                grp_ps['pseudo_dist_km'].values, grp_ps['tp'].values)
            r2_ps = fit_r2(grp_ps['pseudo_dist_km'].values, grp_ps['tp'].values,
                           slope_ps, intc_ps)
        if len(grp_wd) >= 2:
            slope_wd, intc_wd = fit_with_intercept(
                grp_wd['tp'].values, grp_wd['delta_ts_tp'].values)
            r2_wd = fit_r2(grp_wd['tp'].values, grp_wd['delta_ts_tp'].values,
                           slope_wd, intc_wd)

        details[ev] = {
            'slope_ps' : slope_ps, 'intc_ps' : intc_ps, 'r2_ps' : r2_ps,
            'slope_wd' : slope_wd, 'intc_wd' : intc_wd, 'r2_wd' : r2_wd,
            'vp_app'   : 1.0 / slope_ps if (pd.notna(slope_ps) and slope_ps > 0) else np.nan,
            'vpvs'     : slope_wd + 1.0 if pd.notna(slope_wd) else np.nan,
        }

        if qc3_is_bad(details[ev], cfg):
            violate.add(ev)

    if not cfg.get('run_qc3'):
        print("   QC-3 | disabled (metrics still computed for plot annotation)")
        return set(), details

    print(f"   QC-3 | non-physical slope/intercept : {len(violate):,} events rejected")
    return violate, details


# ------------------------------------------------------------
# 3.7 Stage 6 -- Per-event plots: pseudo-distance and Wadati
# ------------------------------------------------------------

def _station_colors(stations):
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(stations), 1)))
    return {s: cmap[i] for i, s in enumerate(stations)}


def plot_pseudo_distance(ev, grp, info, event_id, out_dir, cfg, status, reason):
    """Pseudo-distance plot: tp (and ts) vs distance, with the fitted line."""
    grp_ps = grp.dropna(subset=['tp', 'pseudo_dist_km'])
    if grp_ps.empty:
        return

    slope_ps = info.get('slope_ps', np.nan)
    intc_ps  = info.get('intc_ps',  np.nan)
    r2_ps    = info.get('r2_ps',    np.nan)
    vp_app   = info.get('vp_app',   np.nan)

    sc = _station_colors(sorted(grp['station'].dropna().unique()))

    fig, ax = plt.subplots(figsize=(10, 8))

    ps_bad = qc3_is_bad(info, cfg, which='ps')

    if pd.notna(slope_ps) and pd.notna(intc_ps):
        md = max(grp_ps['pseudo_dist_km'].max(), 1)
        dr = np.linspace(0, md * 1.3, 100)
        ax.plot(dr, slope_ps * dr + intc_ps, '--',
                color='red' if ps_bad else 'steelblue', lw=2,
                label=f"t = {slope_ps:.4f}·d + {intc_ps:.3f}"
                      f"{'  ⚠' if ps_bad else ''}")

    # Vp reference line from the configuration
    md_ref = max(grp_ps['pseudo_dist_km'].max(), 1) * 1.3
    dr_ref = np.linspace(0, md_ref, 50)
    ax.plot(dr_ref, dr_ref / cfg['vp'], '-', color='gray', lw=1.5, alpha=0.6,
            label=f"Reference Vp = {cfg['vp']:.2f} km/s")

    for _, row in grp_ps.iterrows():
        c = sc.get(row['station'], 'gray')
        ax.scatter(row['pseudo_dist_km'], row['tp'], color=c, marker='o',
                   s=120, edgecolors='black', lw=1, zorder=5)
        if pd.notna(row.get('ts')):
            ax.scatter(row['pseudo_dist_km'], row['ts'], color=c, marker='s',
                       s=80, alpha=0.6, edgecolors='gray', zorder=4)
            ax.plot([row['pseudo_dist_km']] * 2, [row['tp'], row['ts']],
                    color=c, alpha=0.35, lw=1.2)
        pp = row.get('p_prob', np.nan)
        ax.annotate(f"{row['station']}\n(P:{pp:.2f})" if pd.notna(pp) else row['station'],
                    (row['pseudo_dist_km'], row['tp']),
                    xytext=(6, 6), textcoords='offset points', fontsize=8)

    ax.scatter(0, 0, c='red', marker='*', s=350, zorder=10,
               edgecolors='black', lw=1.5, label='Origin (0,0)')

    ax.text(0.03, 0.97,
            f"slope     = {slope_ps:.4f}\n"
            f"intercept = {intc_ps:.3f}\n"
            f"R²        = {r2_ps:.3f}\n"
            f"Vp app.   = {vp_app:.2f} km/s" if pd.notna(slope_ps) else "fit unavailable",
            transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='mistyrose' if ps_bad else 'lightcyan',
                      edgecolor='black', alpha=0.9))

    xlabel = ('Hypocentral distance from catalogue (km)'
              if cfg['dist_method_plot'] == 'hypocenter'
              else 'Pseudo distance from (Ts-Tp) (km)')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('P arrival time relative to origin (s)', fontsize=11)
    ax.set_title('Pseudo Distance Plot', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='lower right')

    _suptitle_event(fig, ev, event_id, grp, status, reason)
    fname = os.path.join(out_dir, f"pseudo_distance_{event_id}.{cfg['plot_format']}")
    plt.tight_layout()
    plt.savefig(fname, dpi=cfg['plot_dpi'], bbox_inches='tight')
    plt.close(fig)


def plot_wadati(ev, grp, info, event_id, out_dir, cfg, status, reason):
    """Wadati diagram: (Ts-Tp) vs Tp with the Vp/Vs reference line."""
    grp_wd = grp.dropna(subset=['tp', 'delta_ts_tp'])
    if grp_wd.empty:
        return

    slope_wd = info.get('slope_wd', np.nan)
    intc_wd  = info.get('intc_wd',  np.nan)
    r2_wd    = info.get('r2_wd',    np.nan)
    vpvs     = info.get('vpvs',     np.nan)
    vpvs_ref = cfg.get('vpvs_ref') or cfg['vpvs_ratio']

    sc = _station_colors(sorted(grp['station'].dropna().unique()))

    fig, ax = plt.subplots(figsize=(10, 8))

    max_tp   = max(grp_wd['tp'].max(), 1)
    tp_range = np.linspace(0, max_tp * 1.3, 100)

    ax.plot(tp_range, (vpvs_ref - 1) * tp_range, '-', color='blue', lw=2.5,
            alpha=0.7, label=f"Reference Vp/Vs = {vpvs_ref} "
                             f"(slope={vpvs_ref - 1:.2f})")

    wd_bad = qc3_is_bad(info, cfg, which='wadati')

    if pd.notna(slope_wd) and pd.notna(intc_wd):
        ax.plot(tp_range, slope_wd * tp_range + intc_wd, '--',
                color='red' if wd_bad else 'green', lw=2,
                label=f"y = {slope_wd:.4f}·x + {intc_wd:.3f}"
                      f"{'  ⚠' if wd_bad else ''}")

    for _, row in grp_wd.iterrows():
        c = sc.get(row['station'], 'gray')
        ax.scatter(row['tp'], row['delta_ts_tp'], color=c, marker='D',
                   s=110, edgecolors='black', lw=1, zorder=5)
        ax.annotate(row['station'], (row['tp'], row['delta_ts_tp']),
                    xytext=(6, 6), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    dev = (vpvs - vpvs_ref) if pd.notna(vpvs) else np.nan
    ax.text(0.97, 0.03,
            f"Vp/Vs (data) = {vpvs:.3f}\n"
            f"Vp/Vs (ref)  = {vpvs_ref:.2f}\n"
            f"Deviation    = {dev:+.3f}\n"
            f"slope        = {slope_wd:.4f}\n"
            f"intercept    = {intc_wd:.3f}\n"
            f"R²           = {r2_wd:.3f}" if pd.notna(slope_wd) else "fit unavailable",
            transform=ax.transAxes, fontsize=9, fontweight='bold',
            va='bottom', ha='right', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5',
                      facecolor='mistyrose' if wd_bad else 'lightgreen',
                      edgecolor='black', alpha=0.9))

    ax.scatter(0, 0, c='red', marker='*', s=350, zorder=10,
               edgecolors='black', lw=1.5)
    ax.set_xlabel('P arrival time relative to origin (s)', fontsize=11)
    ax.set_ylabel('S-P delay (s)', fontsize=11)
    ax.set_title('Wadati Diagram', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')

    _suptitle_event(fig, ev, event_id, grp, status, reason)
    fname = os.path.join(out_dir, f"wadati_{event_id}.{cfg['plot_format']}")
    plt.tight_layout()
    plt.savefig(fname, dpi=cfg['plot_dpi'], bbox_inches='tight')
    plt.close(fig)


def _suptitle_event(fig, ev, event_id, grp, status, reason):
    """Uniform figure title shared by every per-event plot."""
    r0 = grp.iloc[0]
    origin = r0.get('origin_time')
    lat, lon, dep = r0.get('orig_lat'), r0.get('orig_lon'), r0.get('orig_depth')

    color = 'darkgreen' if status == 'PASS' else 'darkred'
    tag   = f"[{status}]" + (f" {reason}" if reason else "")
    loc   = (f"lat {lat:.4f} | lon {lon:.4f} | depth {dep:.1f} km"
             if all(pd.notna(v) for v in [lat, lon, dep]) else "location unavailable")

    fig.suptitle(
        f"{tag}  Event {event_id} (idx {ev})  |  {origin}\n{loc}  |  "
        f"{grp['station'].nunique()} stations",
        fontsize=11, fontweight='bold', color=color, y=1.02
    )


# ------------------------------------------------------------
# 3.8 Stage 7 -- Per-event waveform plot (with P and S picks)
# ------------------------------------------------------------

def _date_range_str(t1, t2):
    """List of 'YYYY-MM-DD' dates spanned by the range t1..t2."""
    d, end = t1.datetime.date(), t2.datetime.date()
    out = []
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def find_waveform_files(station, t1, t2, cfg):
    """
    Find the mseed files for one station over the time range t1..t2.
    Input layout: <waveform_folder>/<YYYY-MM-DD>/<NET>.<STA>..<YYYY-MM-DD>.mseed
    """
    import glob as _glob

    base  = cfg['waveform_folder']
    found, seen = [], set()

    for dstr in _date_range_str(t1, t2):
        day_dir = os.path.join(base, dstr)
        if not os.path.isdir(day_dir):
            continue
        for pat in cfg['wf_file_patterns']:
            pattern = os.path.join(day_dir, pat.format(sta=station, date=dstr))
            for m in sorted(_glob.glob(pattern)):
                if m not in seen:
                    found.append(m)
                    seen.add(m)
            if found:
                break
    return found


def load_z_trace(station, t1, t2, cfg):
    """
    Read the vertical (Z) component of one station over the window t1..t2.
    Reading is restricted to the window (starttime/endtime) so the whole daily
    file never has to be loaded into memory.
    """
    files = find_waveform_files(station, t1, t2, cfg)
    if not files:
        return None, "file not found"

    st = Stream()
    for f in files:
        try:
            st += read(f, starttime=t1, endtime=t2)
        except Exception:
            continue
    if len(st) == 0:
        return None, "failed to read mseed"

    try:
        st.merge(method=1, fill_value='interpolate')
    except Exception:
        return None, "failed to merge stream"

    for ch in cfg['wf_channel_priority']:
        sel = st.select(channel=ch)
        if sel:
            sel[0].trim(t1, t2, pad=True, fill_value=0)
            return sel[0], None

    for tr in st:
        if tr.stats.channel.endswith('Z'):
            tr.trim(t1, t2, pad=True, fill_value=0)
            return tr, None

    return None, f"no Z channel (available: {[tr.stats.channel for tr in st]})"


def plot_event_waveform(ev, grp, event_id, out_dir, cfg, status, reason):
    """
    Plot the Z-component waveform of every station for one event, with the
    associated P and S picks marked.

    Display conventions:
      - The station receiving P FIRST is placed in the BOTTOM panel and the
        latest one at the top, so moveout reads downwards.
      - Amplitude normalisation uses only the samples inside the display window,
        so spikes outside it cannot flatten the main signal.
    """
    grp = grp.copy()
    origin_ts = grp['origin_time'].iloc[0]
    if pd.isna(origin_ts):
        return
    origin = UTCDateTime(pd.Timestamp(origin_ts).to_pydatetime())

    # --- Display window derived from the pick distribution ---
    tp_all = grp['tp'].dropna().tolist()
    ts_all = grp['ts'].dropna().tolist()

    if tp_all or ts_all:
        ref_start = (min(tp_all) if tp_all else min(ts_all)) - cfg['wf_pad_before_p']
        ref_end   = (max(ts_all) if ts_all else max(tp_all)) + cfg['wf_pad_after_s']
    else:
        ref_start = -cfg['wf_fallback_pre']
        ref_end   = cfg['wf_fallback_post']

    # --- Station order: earliest P -> bottom panel ---
    grp['_t_rel'] = grp['tp'].fillna(grp['ts']).fillna(np.inf)
    grp = grp.sort_values('_t_rel', ascending=False).reset_index(drop=True)

    # Limit the number of panels: keep the earliest arrivals (nearest stations)
    max_sta = cfg['wf_max_stations']
    if max_sta and len(grp) > max_sta:
        grp = grp.tail(max_sta).reset_index(drop=True)

    n_sta = len(grp)
    if n_sta == 0:
        return

    t1_read = origin + ref_start - 10
    t2_read = origin + ref_end + 10

    fig, axes = plt.subplots(n_sta, 1,
                             figsize=(14, cfg['wf_panel_height'] * n_sta),
                             squeeze=False, sharex=True)

    for i, r in grp.iterrows():
        ax  = axes[i, 0]
        sta = r['station']
        tr, err = load_z_trace(sta, t1_read, t2_read, cfg)

        if tr is None:
            ax.text(0.5, 0.5, f"{sta}  --  {err}",
                    ha='center', va='center', transform=ax.transAxes,
                    color='gray', fontsize=9, style='italic')
            ax.set_yticks([])
        else:
            t_rel = tr.times(reftime=origin)
            data  = tr.data.astype(float)

            mask_win = (t_rel >= ref_start) & (t_rel <= ref_end)
            win_data = data[mask_win]
            if len(win_data) > 0 and np.any(np.isfinite(win_data) & (win_data != 0)):
                peak = np.nanmax(np.abs(win_data))
            else:
                peak = np.nanmax(np.abs(data)) if len(data) else 0.0
            norm = data / peak if peak > 0 else data

            ax.plot(t_rel[mask_win], norm[mask_win], 'k-', linewidth=0.6)
            vis  = np.abs(norm[mask_win])
            vmax = float(np.nanmax(vis)) if len(vis) > 0 else 1.0
            ax.set_ylim(-vmax * 1.15, vmax * 1.15)
            ax.set_ylabel('norm. amp.', fontsize=8)

            if pd.notna(r['tp']):
                tp = float(r['tp'])
                ax.axvline(tp, color='steelblue', lw=1.8, ls='--', alpha=0.9,
                           label=f"P  +{tp:.2f}s")
                ax.axvspan(tp - 0.5, tp + 0.5, alpha=0.08, color='steelblue')
            if pd.notna(r['ts']):
                ts = float(r['ts'])
                ax.axvline(ts, color='crimson', lw=1.8, ls='--', alpha=0.9,
                           label=f"S  +{ts:.2f}s")
                ax.axvspan(ts - 0.5, ts + 6.0, alpha=0.06, color='crimson')

            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8, loc='upper right', framealpha=0.75)
            ax.axhline(0, color='gray', lw=0.4, ls=':')

        parts = [
            sta,
            f"R={r['epi_dist_km']:.1f} km"   if pd.notna(r.get('epi_dist_km'))   else "",
            f"d_ps={r['pseudo_dist_km']:.1f} km" if pd.notna(r.get('pseudo_dist_km')) else "",
            f"P prob={r['p_prob']:.2f}"      if pd.notna(r.get('p_prob'))        else "",
            f"S prob={r['s_prob']:.2f}"      if pd.notna(r.get('s_prob'))        else "",
            f"Ts-Tp={r['delta_ts_tp']:.2f}s" if pd.notna(r.get('delta_ts_tp'))   else "",
        ]
        ax.set_title("  |  ".join(p for p in parts if p),
                     fontsize=9, loc='left', pad=3)
        ax.grid(True, alpha=0.22)

    axes[-1, 0].set_xlabel('Time relative to origin (s)', fontsize=10)
    for ax_row in axes:
        ax_row[0].set_xlim(ref_start, ref_end)

    _suptitle_event(fig, ev, event_id, grp, status, reason)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"waveform_{event_id}.{cfg['plot_format']}")
    plt.savefig(fname, dpi=cfg['plot_dpi'], bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------
# 3.9 Cross-event summary output and NonLinLoc input
# ------------------------------------------------------------

def build_summary_tables(df_evt, df_cat, stations, qc3_details, ev_max_ratio,
                         all_rejected, reject_reason, geom, cfg):
    """
    Build two cross-event summary tables:

    1. event_summary : one row per event.
       Position (lat/lon/depth), origin time, contributing stations, P-S pair
       count, azimuthal gap, nearest/farthest distance, QC fit metrics.

    2. phase_picks   : one row per pick (P or S).
       Absolute pick time, travel time, distance and azimuth -- a flat format
       ready for relocation or NonLinLoc input.
    """
    sta_meta = stations.set_index('station')[
        ['network', 'latitude', 'longitude', 'elevation_m']].to_dict('index')

    event_rows, pick_rows = [], []

    for ev, grp in df_evt.groupby('event_idx'):
        rejected = ev in all_rejected
        if rejected and not cfg['summary_include_rejected']:
            continue

        r0       = grp.iloc[0]
        event_id = make_event_id(ev)
        origin   = r0['origin_time']
        olat, olon, odep = r0['orig_lat'], r0['orig_lon'], r0['orig_depth']

        has_p = grp['tp'].notna()
        has_s = grp['ts'].notna()
        ps_pair_stations = sorted(grp.loc[has_p & has_s, 'station'].unique())

        azimuths = []
        for _, row in grp.iterrows():
            sta = row['station']
            meta = sta_meta.get(sta)
            if meta is None or pd.isna(olat):
                az = np.nan
            else:
                az = azimuth_deg(olat, olon, meta['latitude'], meta['longitude'])
            azimuths.append(az)

            # --- Pick rows: P and S written separately ---
            for phase, t_abs, t_rel, prob in [
                ('P', row.get('p_dt'), row.get('tp'), row.get('p_prob')),
                ('S', row.get('s_dt'), row.get('ts'), row.get('s_prob')),
            ]:
                if pd.isna(t_abs):
                    continue
                pick_rows.append({
                    'event_id'        : event_id,
                    'event_idx'       : ev,
                    'origin_time'     : origin,
                    'network'         : meta['network']     if meta else '',
                    'station'         : sta,
                    'station_lat'     : meta['latitude']    if meta else np.nan,
                    'station_lon'     : meta['longitude']   if meta else np.nan,
                    'station_elev_m'  : meta['elevation_m'] if meta else np.nan,
                    'phase'           : phase,
                    'pick_time'       : t_abs,
                    'travel_time_s'   : t_rel,
                    'probability'     : prob,
                    'epi_dist_km'     : row.get('epi_dist_km'),
                    'dist_ts_tp_km'   : row.get('dist_ts_tp_km'),
                    'dist_hypo_km'    : row.get('dist_hypo_km'),
                    'azimuth_deg'     : az,
                    'delta_ts_tp_s'   : row.get('delta_ts_tp'),
                    'qc_status'       : 'REJECT' if rejected else 'PASS',
                })

        info    = qc3_details.get(ev, {})
        cat_row = df_cat[df_cat['event_index'] == ev]

        event_rows.append({
            'event_id'       : event_id,
            'event_idx'      : ev,
            'origin_time'    : origin,
            'latitude'       : olat,
            'longitude'      : olon,
            'depth_km'       : odep,
            'n_station'      : grp['station'].nunique(),
            'n_ps_pairs'     : len(ps_pair_stations),
            'n_p_picks'      : int(has_p.sum()),
            'n_s_picks'      : int(has_s.sum()),
            'n_picks'        : int(has_p.sum() + has_s.sum()),
            'stations'       : ','.join(sorted(grp['station'].dropna().unique())),
            'stations_ps'    : ','.join(ps_pair_stations),
            'azimuthal_gap'  : azimuthal_gap(azimuths),
            'dist_min_km'    : grp['epi_dist_km'].min(),
            'dist_max_km'    : grp['epi_dist_km'].max(),
            'tp_min_s'       : grp['tp'].min(),
            'ts_tp_max_s'    : grp['delta_ts_tp'].max(),
            'vp_apparent'    : info.get('vp_app', np.nan),
            'vpvs_data'      : info.get('vpvs', np.nan),
            'r2_pseudo'      : info.get('r2_ps', np.nan),
            'r2_wadati'      : info.get('r2_wd', np.nan),
            'max_ratio_qc2'  : ev_max_ratio.get(ev, np.nan),
            'pyocto_n_picks' : (int(cat_row['n_picks'].iloc[0])
                                if not cat_row.empty and 'n_picks' in cat_row else np.nan),
            'qc_status'      : 'REJECT' if rejected else 'PASS',
            'reject_reason'  : reject_reason.get(ev, ''),
        })

    df_event_summary = pd.DataFrame(event_rows).sort_values('origin_time')
    df_phase_picks   = pd.DataFrame(pick_rows).sort_values(['event_idx', 'pick_time'])

    return (df_event_summary.reset_index(drop=True),
            df_phase_picks.reset_index(drop=True))


def write_search_volume(geom, stations, folders, cfg):
    """
    Hand the association search volume over to the location stage.

    locator.py reads this file and turns it into the NonLinLoc LOCGRID, so both
    stages search exactly the same volume. Every value is derived from the
    station distribution, which means the location grid is fixed by the network
    geometry rather than chosen by hand.

    The km extents use the same flat-earth convention as NonLinLoc TRANS SIMPLE:
        x = (lon - lon0) * 111.111 * cos(lat)
        y = (lat - lat0) * 111.111
    """
    C111 = 111.111
    lat0, lon0 = geom['trans_lat'], geom['trans_lon']
    (lat_min, lat_max) = geom['lat_range']
    (lon_min, lon_max) = geom['lon_range']
    z_min, z_max = geom['zlim_km']

    # Convert the lat/lon box to km, using the largest cos(lat) in range so the
    # grid is guaranteed to enclose the whole box.
    cos_max = max(math.cos(math.radians(lat_min)), math.cos(math.radians(lat_max)))
    x_min = (lon_min - lon0) * C111 * cos_max
    x_max = (lon_max - lon0) * C111 * cos_max
    y_min = (lat_min - lat0) * C111
    y_max = (lat_max - lat0) * C111

    d = cfg['locgrid_node_spacing_km']
    payload = {
        'generated_by'   : 'association.py',
        'generated_at'   : str(pd.Timestamp.now()),
        'n_stations'     : int(len(stations)),
        'aperture_km'    : round(geom['aperture_km'], 3),
        'trans_type'     : 'SIMPLE',
        'trans_lat_orig' : round(lat0, 6),
        'trans_lon_orig' : round(lon0, 6),
        'trans_rot_angle': 0.0,
        'lat_range'      : [round(lat_min, 6), round(lat_max, 6)],
        'lon_range'      : [round(lon_min, 6), round(lon_max, 6)],
        'depth_range_km' : [round(z_min, 3), round(z_max, 3)],
        'node_spacing_km': d,
        # LOCGRID parameters: origin plus node counts covering the box
        'grid_x_orig'    : round(math.floor(x_min / d) * d, 3),
        'grid_y_orig'    : round(math.floor(y_min / d) * d, 3),
        'grid_z_orig'    : round(math.floor(z_min / d) * d, 3),
        'grid_x_num'     : int(math.ceil((x_max - math.floor(x_min / d) * d) / d)) + 1,
        'grid_y_num'     : int(math.ceil((y_max - math.floor(y_min / d) * d) / d)) + 1,
        'grid_z_num'     : int(math.ceil((z_max - math.floor(z_min / d) * d) / d)) + 1,
        'vp'             : geom['vp'],
        'vs'             : geom['vs'],
        'vpvs_ratio'     : geom['vpvs_ratio'],
    }

    nll_dir = os.path.join(folders['root'], 'nonlinloc')
    os.makedirs(nll_dir, exist_ok=True)
    path = os.path.join(nll_dir, 'search_volume.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"   ✅ {path}")
    print(f"      LOCGRID {payload['grid_x_num']} x {payload['grid_y_num']} "
          f"x {payload['grid_z_num']} nodes at {d} km  |  "
          f"X {payload['grid_x_orig']:.1f} Y {payload['grid_y_orig']:.1f} "
          f"Z {payload['grid_z_orig']:.1f} km")
    return path


def _nll_line(row, cfg, with_priorwt, component):
    """
    One NLLOC_OBS observation line:

    STA  INST COMP ONSET PHASE FM  DATE     HRMN SEC     ERRTYPE ERRMAG   CODA AMP  PER  [PRIORWT]
    EJA14 ?   ChZ  ?     P     ?   20151204 1532 23.9300 GAU     1.00e-01 -1.0 -1.0 -1.0 0.47

    The PRIORWT column is written only for the combined .obs archive; .pick files
    follow the layout already used by the existing NonLinLoc input.
    """
    t     = pd.Timestamp(row['pick_time']).to_pydatetime()
    phase = row['phase'].upper()
    err   = cfg['nll_error_p_sec'] if phase == 'P' else cfg['nll_error_s_sec']

    line = (
        f"{row['station']:<6s} ? {component:<4s} ? {phase:<2s} ? "
        f"{t.strftime('%Y%m%d')} {t.strftime('%H%M')} "
        f"{t.second + t.microsecond / 1e6:>8.4f} "
        f"GAU  {err:.2e} -1.00e+00 -1.00e+00 -1.00e+00"
    )
    if with_priorwt and pd.notna(row.get('probability')):
        line += f" {float(np.clip(row['probability'], 0.0, 1.0)):.4f}"
    return line


def _pick_filename(origin_time):
    """.pick filename following the existing convention: YYYYMMDD-HHMM-SS.pick"""
    t = pd.Timestamp(origin_time).to_pydatetime()
    return f"{t.strftime('%Y%m%d-%H%M-%S')}.pick"


def write_nonlinloc_outputs(df_phase_picks, stations, folders, cfg):
    """
    Write the NonLinLoc input:
      nonlinloc/picks/<YYYYMMDD-HHMM-SS>.pick -- one file per event, matching
                                                 LOCFILES <dir>/*.pick NLLOC_OBS
      nonlinloc/all_events.obs                -- all events combined (archive)
      nonlinloc/stations_GTSRCE.txt           -- GTSRCE lines for the control file
    """
    nll_dir  = os.path.join(folders['root'], 'nonlinloc')
    pick_dir = os.path.join(nll_dir, 'picks')
    os.makedirs(pick_dir, exist_ok=True)

    df = df_phase_picks
    if not cfg['nll_include_rejected'] and 'qc_status' in df.columns:
        df = df[df['qc_status'] == 'PASS']

    if df.empty:
        print("   ⚠️  No QC-passing picks -> NonLinLoc files not written")
        return nll_dir, 0

    comp_of = {'P': cfg['nll_pick_component_p'], 'S': cfg['nll_pick_component_s']}

    all_path  = os.path.join(nll_dir, 'all_events.obs')
    n_events  = 0
    n_pick_f  = 0
    used_name = {}

    f_all = open(all_path, 'w') if cfg['nll_write_combined_obs'] else None
    try:
        for event_id, grp in df.groupby('event_id', sort=True):
            grp = grp.sort_values('pick_time')

            if f_all is not None:
                lines = [_nll_line(r, cfg, cfg['nll_obs_priorwt'],
                                   comp_of[r['phase'].upper()])
                         for _, r in grp.iterrows()]
                f_all.write(f"# EVENT {event_id}\n")
                f_all.write("\n".join(lines))
                f_all.write("\n\n")   # blank line separates events for NLLoc

            if cfg['nll_write_pick_files']:
                lines = [_nll_line(r, cfg, False, comp_of[r['phase'].upper()])
                         for _, r in grp.iterrows()]
                fname = _pick_filename(grp['origin_time'].iloc[0])
                # Two events in the same second -> add a suffix to avoid overwriting
                if fname in used_name:
                    used_name[fname] += 1
                    fname = fname.replace('.pick', f"-{used_name[fname]}.pick")
                else:
                    used_name[fname] = 0
                with open(os.path.join(pick_dir, fname), 'w') as f_ev:
                    f_ev.write("\n".join(lines))
                    f_ev.write("\n\n")
                n_pick_f += 1

            n_events += 1
    finally:
        if f_all is not None:
            f_all.close()

    # --- GTSRCE block for the NonLinLoc control file ---
    # Format: GTSRCE <label> LATLON <lat> <lon> <depth_km> <elev_km>
    used = set(df['station'].unique())
    gtsrce_path = os.path.join(nll_dir, 'stations_GTSRCE.txt')
    with open(gtsrce_path, 'w') as f:
        f.write("# Copy this block into the NonLinLoc control file (Vel2Grid/Grid2Time section)\n")
        for _, s in stations.sort_values('station').iterrows():
            if s['station'] not in used:
                continue
            f.write(f"GTSRCE {s['station']:<6s} LATLON "
                    f"{s['latitude']:>10.4f} {s['longitude']:>10.4f} "
                    f"0.0 {s['elevation_m'] / 1000.0:.4f}\n")

    if cfg['nll_write_pick_files']:
        print(f"   ✅ {pick_dir}/<YYYYMMDD-HHMM-SS>.pick : {n_pick_f:,} file")
    if cfg['nll_write_combined_obs']:
        print(f"   ✅ {all_path} : {n_events:,} event")
    print(f"   ✅ {gtsrce_path} : {len(used)} stations")
    print(f"   → point LOCFILES in the control file to: {pick_dir}/*.pick NLLOC_OBS")

    return nll_dir, n_events


# ------------------------------------------------------------
# 3.10 Stage 8 -- Global summary plots
# ------------------------------------------------------------

def plot_qc2_distribution(all_ratios, all_excess, threshold, tol_km, cfg, folders):
    """
    Distribution of the QC-2 test statistic.

    The plotted variable is whichever quantity actually drives the decision:
      method 'physical'   -> excess = |d_A - d_B| - D_AB, in km, with the
                             geometric bound at 0 and the rejection threshold
                             at +tol_km (derived from pick-error propagation)
      method 'percentile' -> the ratio |d_A - d_B| / D_AB, with the percentile
                             threshold and the geometric bound at 1.0
    """
    physical = cfg['qc2_method'] == 'physical'
    data     = all_excess if physical else all_ratios
    if len(data) == 0:
        return

    if physical:
        bound, cut = 0.0, tol_km
        xlabel     = 'Excess  |d_A - d_B| - D_AB  (km)'
        bound_lbl  = 'Geometric bound (excess = 0)'
        cut_lbl    = (f"Rejection threshold = {tol_km:.2f} km "
                      f"({cfg['qc2_n_sigma']:g} sigma)")
        title      = 'QC-2 test statistic: geometric excess (all station pairs)'
    else:
        bound, cut = 1.0, threshold
        xlabel     = 'Ratio  |d_A - d_B| / D_AB'
        bound_lbl  = 'Geometric bound (ratio = 1.0)'
        cut_lbl    = f'Rejection threshold = {threshold:.3f}'
        title      = 'QC-2 test statistic: distance ratio (all station pairs)'

    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

        n_reject = int(np.sum(data > cut))
        frac     = 100.0 * n_reject / len(data)

        axes[0].hist(data, bins=100, color='steelblue', edgecolor='white', alpha=0.85)
        axes[0].axvline(bound, color='black', lw=2, ls='-', label=bound_lbl)
        axes[0].axvline(cut, color='crimson', lw=2, ls='--', label=cut_lbl)
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel('Count')
        axes[0].set_title(title, fontsize=10, fontweight='bold')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        srt = np.sort(data)
        cdf = np.arange(1, len(srt) + 1) / len(srt)
        axes[1].plot(srt, cdf * 100, color='steelblue', lw=1.6)
        axes[1].axvline(bound, color='black', lw=2, ls='-')
        axes[1].axvline(cut, color='crimson', lw=2, ls='--')
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel('Cumulative percentage (%)')
        axes[1].set_title('Cumulative distribution', fontsize=10, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        axes[1].text(0.03, 0.97,
                     f"n pairs      = {len(data):,}\n"
                     f"median       = {np.median(data):.3f}\n"
                     f"beyond bound = {int(np.sum(data > bound)):,} "
                     f"({100.0*np.sum(data > bound)/len(data):.2f}%)\n"
                     f"rejected     = {n_reject:,} ({frac:.2f}%)",
                     transform=axes[1].transAxes, fontsize=9, fontweight='bold',
                     va='top', ha='left', family='monospace',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan',
                               edgecolor='black', alpha=0.9))

        plt.tight_layout()
        plt.savefig(os.path.join(folders['summary'],
                                 f"qc2_test_statistic.{cfg['plot_format']}"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        plt.close('all')
        print(f"   ⚠️  QC-2 distribution plot failed: {e}")


def plot_global_wadati_pseudo(df_evt, all_rejected, geom, cfg, folders):
    """
    Wadati and pseudo-distance diagrams pooled over many events.

    Two versions of each diagram are produced:
      *_all_events.png -> every associated event, before quality control
      *_passed_qc.png  -> only the events that passed quality control

    Every data point uses the same colour so the eye compares the two
    populations directly rather than individual events. A global linear fit is
    drawn on each, next to the reference line implied by the assumed velocity
    model, so the recovered Vp and Vp/Vs can be read off and compared with the
    values that were assumed.
    """
    df = df_evt.copy()
    df['passed'] = ~df['event_idx'].isin(all_rejected)

    diagrams = [
        ('pseudo', 'pseudo_dist_km', 'tp',
         f"Distance ({cfg['dist_method_plot']}) (km)",
         'P arrival time relative to origin (s)',
         'Pseudo Distance'),
        ('wadati', 'tp', 'delta_ts_tp',
         'P arrival time relative to origin (s)',
         'S-P delay (s)',
         'Wadati Diagram'),
    ]
    subsets = [
        ('all_events', 'ALL ASSOCIATED EVENTS', '#2c7fb8'),
        ('passed_qc',  'EVENTS PASSING QC',     '#238b45'),
    ]

    for name, xcol, ycol, xlabel, ylabel, title in diagrams:
        if (name == 'pseudo' and not cfg.get('plot_summary_pseudo')) or \
           (name == 'wadati' and not cfg.get('plot_summary_wadati')):
            continue

        for suffix, subtitle, colour in subsets:
            try:
                sub = df.dropna(subset=[xcol, ycol])
                if suffix == 'passed_qc':
                    sub = sub[sub['passed']]
                if sub.empty:
                    continue

                fig, ax = plt.subplots(figsize=(11, 8.5))

                ax.scatter(sub[xcol], sub[ycol], s=26, c=colour,
                           edgecolors='black', linewidths=0.3, alpha=0.75,
                           zorder=3, label=f'Station observations (n={len(sub):,})')

                # --- Reference line from the assumed velocity model ---
                xmax = float(sub[xcol].max()) * 1.05
                xr   = np.linspace(0, xmax, 100)
                if name == 'pseudo':
                    ax.plot(xr, xr / geom['vp'], '-', color='crimson', lw=2,
                            alpha=0.85,
                            label=f"Assumed Vp = {geom['vp']:.2f} km/s")
                else:
                    vpvs_ref = cfg.get('vpvs_ref') or cfg['vpvs_ratio']
                    ax.plot(xr, (vpvs_ref - 1) * xr, '-', color='crimson', lw=2,
                            alpha=0.85,
                            label=f"Assumed Vp/Vs = {vpvs_ref}")

                # --- Global linear fit over the plotted points ---
                txt = 'fit unavailable'
                if len(sub) >= 2:
                    sl, ic = fit_with_intercept(sub[xcol].values, sub[ycol].values)
                    r2     = fit_r2(sub[xcol].values, sub[ycol].values, sl, ic)
                    if pd.notna(sl):
                        ax.plot(xr, sl * xr + ic, '--', color='navy', lw=2,
                                label=f'Global fit: y = {sl:.4f}x + {ic:.3f}')
                        if name == 'pseudo':
                            derived = (f"Vp fitted    = {1.0/sl:.3f} km/s"
                                       if sl > 0 else "Vp fitted    = n/a")
                        else:
                            derived = f"Vp/Vs fitted = {sl + 1.0:.3f}"
                        txt = (f"events    = {sub['event_idx'].nunique():,}\n"
                               f"n points  = {len(sub):,}\n"
                               f"slope     = {sl:.4f}\n"
                               f"intercept = {ic:.3f}\n"
                               f"R\u00b2        = {r2:.4f}\n"
                               f"{derived}")

                ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=10,
                        fontweight='bold', va='top', ha='left', family='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan',
                                  edgecolor='black', alpha=0.9))

                ax.set_xlabel(xlabel, fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_title(f"{title} -- {subtitle}  |  "
                             f"{sub['event_idx'].nunique():,} events, "
                             f"{len(sub):,} station observations",
                             fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9, loc='lower right')

                plt.tight_layout()
                plt.savefig(os.path.join(folders['summary'],
                                         f"{name}_{suffix}.{cfg['plot_format']}"),
                            dpi=150, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                plt.close('all')
                print(f"   \u26a0\ufe0f  Pooled {name} plot ({suffix}) failed: {e}")


def plot_catalog_summary(df_cat_qc, df_cat_all, stations, cfg, folders):
    """Post-QC catalogue summary: depth, n_picks, epicentre map, timeline."""
    if df_cat_qc.empty:
        print("   ⚠️  Post-QC catalogue is empty, summary plot skipped")
        return

    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].hist(df_cat_qc['depth_km'].dropna(), bins=50,
                        color='steelblue', edgecolor='white')
        axes[0, 0].set_xlabel('Depth (km)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Depth distribution')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].hist(df_cat_qc['n_picks'].dropna(), bins=30,
                        color='darkorange', edgecolor='white')
        axes[0, 1].set_xlabel('Picks per event')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('n_picks distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # --- Epicentre map with stations ---
        ax = axes[1, 0]
        sc_map = ax.scatter(df_cat_qc['longitude'], df_cat_qc['latitude'],
                            s=np.clip(df_cat_qc['n_picks'] * 2, 8, 80),
                            alpha=0.55, c=df_cat_qc['depth_km'], cmap='plasma_r')
        ax.scatter(stations['longitude'], stations['latitude'],
                   marker='^', s=90, c='black', edgecolors='white',
                   lw=0.8, zorder=5, label='Stations')
        for _, s in stations.iterrows():
            ax.annotate(s['station'], (s['longitude'], s['latitude']),
                        xytext=(4, 4), textcoords='offset points', fontsize=6)
        plt.colorbar(sc_map, ax=ax, label='Depth (km)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Post-QC epicentre distribution')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Timeline of events per day ---
        ax = axes[1, 1]
        daily = (df_cat_qc.set_index(pd.to_datetime(df_cat_qc['time']))
                 .resample('D').size())
        ax.bar(daily.index, daily.values, color='seagreen', width=0.9)
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of events')
        ax.set_title('Events per day (post-QC)')
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        fig.suptitle(
            f"PyOcto Association + QC  --  {len(df_cat_qc):,} events passed "
            f"of {len(df_cat_all):,} associated",
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(folders['summary'],
                                 f"catalog_summary.{cfg['plot_format']}"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        plt.close('all')
        print(f"   ⚠️  Catalogue summary plot failed: {e}")


# ------------------------------------------------------------
# 3.11 Stage 9 -- Per-event plotting orchestration
# ------------------------------------------------------------

def render_all_event_plots(df_evt, qc3_details, all_rejected, reject_reason,
                           cfg, folders, checkpoint):
    """
    For each event: create its own folder and save the pseudo-distance plot, the
    Wadati diagram, the waveform figure and the pick table. QC-passing events go
    to events/, rejected ones to rejected/.
    """
    event_ids = sorted(df_evt['event_idx'].unique())

    if not cfg['plot_rejected_events']:
        event_ids = [ev for ev in event_ids if ev not in all_rejected]

    if cfg['max_events_plot']:
        event_ids = event_ids[:cfg['max_events_plot']]

    plotted = checkpoint['plotted_events']
    todo    = [ev for ev in event_ids if make_event_id(ev) not in plotted]
    print(f"   Events to plot : {len(todo):,} "
          f"(skipped via checkpoint: {len(event_ids) - len(todo):,})")

    n_error = 0
    grouped = dict(tuple(df_evt.groupby('event_idx')))

    for ev in tqdm(todo, desc="Per-event plots", unit="event"):
        event_id = make_event_id(ev)
        grp      = grouped[ev]

        rejected = ev in all_rejected
        status   = 'REJECT' if rejected else 'PASS'
        reason   = reject_reason.get(ev, '') if rejected else ''
        base     = folders['rejected'] if rejected else folders['events']
        out_dir  = setup_event_folder(base, event_id)
        info     = qc3_details.get(ev, {})

        try:
            if cfg['save_event_picks_csv']:
                grp.drop(columns=['_t_rel'], errors='ignore').to_csv(
                    os.path.join(out_dir, f"picks_{event_id}.csv"), index=False)

            if cfg['plot_pseudo_distance']:
                plot_pseudo_distance(ev, grp, info, event_id, out_dir, cfg,
                                     status, reason)
            if cfg['plot_wadati']:
                plot_wadati(ev, grp, info, event_id, out_dir, cfg, status, reason)
            if cfg['plot_waveform']:
                plot_event_waveform(ev, grp, event_id, out_dir, cfg, status, reason)

            plotted.add(event_id)
            save_checkpoint(cfg['checkpoint_file'], checkpoint)

        except Exception as e:
            n_error += 1
            plt.close('all')
            tqdm.write(f"  ❌ Event {event_id} failed to plot: {e}")
            continue

    return n_error


# ------------------------------------------------------------
# 3.12 Main pipeline
# ------------------------------------------------------------

def main():
    t0 = time.time()
    log_step = make_logger(CONFIG, t0)

    print("=" * 64)
    print("  PyOcto Association + Quality Control Pipeline")
    print(f"  PyOcto   : {pyocto.__version__ if hasattr(pyocto, '__version__') else '?'}")
    print(f"  Picks    : {CONFIG['picks_csv']}")
    print("=" * 64)

    folders    = setup_output_folders(CONFIG['output_folder'])
    checkpoint = load_checkpoint(CONFIG['checkpoint_file'])
    print(f"🗂️  Checkpoint: {len(checkpoint['associated_days'])} days associated"
          f" | {len(checkpoint['plotted_events'])} events plotted")

    csv_dir = folders['csv']
    path_picks       = os.path.join(csv_dir, "pyocto_picks.csv")
    path_events      = os.path.join(csv_dir, "pyocto_events.csv")
    path_assignments = os.path.join(csv_dir, "pyocto_assignments.csv")
    path_catalog     = os.path.join(csv_dir, "pyocto_catalog.csv")
    path_rev_assign  = os.path.join(csv_dir, "pyoctorev_assignments.csv")
    path_rev_catalog = os.path.join(csv_dir, "pyoctorev_catalog.csv")
    path_qc_report   = os.path.join(csv_dir, "qc_event_report.csv")

    # ========================================================
    # STAGE 1 -- Stations and network geometry
    # ========================================================
    print("\n── STAGE 1: Stations & network geometry " + "─" * 23)
    stations = load_stations_dataframe(CONFIG['stations_file'], CONFIG['network'])
    print(f"   Stations loaded : {len(stations)}")

    geom     = derive_network_geometry(stations, CONFIG)
    VP, VS   = geom['vp'], geom['vs']

    velocity_model = build_velocity_model(geom)
    associator     = build_associator(stations, velocity_model, geom, CONFIG)

    sta_coord = stations.set_index('station')[['latitude', 'longitude']].to_dict('index')

    # transform_stations() also populates the associator CRS attribute, later used
    # by transform_events() to project x/y km back to lat/lon.
    pyocto_stations = associator.transform_stations(
        prepare_associator_stations(stations))

    # ========================================================
    # STAGE 2 -- Phase association
    # ========================================================
    print("\n── STAGE 2: Phase association (PyOcto) " + "─" * 24)

    # Checkpoint empty (new or deleted) while old association CSVs still exist ->
    # remove them, since the checkpoint is the single source of truth for which
    # days have been processed. Otherwise old results would be double-counted.
    if not checkpoint['associated_days']:
        stale = [p for p in (path_events, path_assignments) if os.path.exists(p)]
        if stale:
            print("   🧹 Checkpoint empty -> old association CSVs removed, "
                  "starting association from scratch")
            for p in stale:
                os.remove(p)

    log_step("reading picks...")
    picks = build_pyocto_picks(CONFIG['picks_csv'], CONFIG)
    picks.to_csv(path_picks, index=False)
    log_step(f"PyOcto-format picks saved -> {path_picks}")

    pyocto_picks = prepare_associator_picks(picks)

    log_step("association started (usually the slowest stage)...")
    events, assignments = run_association(
        associator, pyocto_picks, pyocto_stations, CONFIG,
        path_events, path_assignments, checkpoint)
    log_step(f"association done: {len(events):,} events, "
             f"{len(assignments):,} assignments")

    if len(events) == 0:
        print("\n❌ No events associated. "
              "Try increasing velocity_tolerance or lowering n_picks.")
        return

    df_cat, _ = build_catalog(events, assignments, associator, CONFIG)
    df_assign = assignments
    df_cat.to_csv(path_catalog, index=False)
    log_step(f"initial catalogue saved -> {path_catalog}")

    total_orig = df_assign['event_idx'].nunique()
    print(f"   Events associated : {total_orig:,}")
    print(f"   Total picks       : {len(df_assign):,}")

    # ========================================================
    # STAGE 3 -- QC-0 (dedup + require P & S)
    # ========================================================
    print("\n── STAGE 3: QC-0 (dedup & P-S pairing) " + "─" * 24)
    df_assign_clean, violate_qc0, qc0_stats = qc0_clean_assignments(df_assign, CONFIG)
    print(f"   QC-0a | double picks removed : {qc0_stats['n_dup_removed']:,}")
    print(f"   QC-0d | single-phase station picks removed : "
          f"{qc0_stats.get('n_single_phase_removed', 0):,}")
    print(f"   QC-0b | P only (no S)        : {qc0_stats.get('n_only_p', 0):,} events")
    print(f"   QC-0b | S only (no P)        : {qc0_stats.get('n_only_s', 0):,} events")
    print(f"   QC-0b | events rejected      : {qc0_stats['n_qc0b_events']:,}")
    if qc0_stats.get('min_ps_pairs'):
        print(f"   QC-0c | < {qc0_stats['min_ps_pairs']} P-S pairs "
              f"(distinct stations) : {qc0_stats['n_qc0c_events']:,} events rejected")
    print(f"   Clean picks : {len(df_assign_clean):,} | "
          f"Clean events : {df_assign_clean['event_idx'].nunique():,}")

    if df_assign_clean.empty:
        print("\n❌ No picks remain after QC-0.")
        return

    # ========================================================
    # STAGE 4 -- Per-event table
    # ========================================================
    print("\n── STAGE 4: Per-event table " + "─" * 35)
    df_sta = stations[['station', 'network', 'latitude', 'longitude']]
    df_evt = build_event_table(df_assign_clean, df_cat, df_sta, VP, VS, CONFIG)
    print(f"   Rows (event, station) : {len(df_evt):,}")
    print(f"   Unique events         : {df_evt['event_idx'].nunique():,}")
    print(f"   Distance for plot/QC-3: '{CONFIG['dist_method_plot']}'")
    print(f"   Distance for QC-2     : '{CONFIG['dist_method_qc2']}'")

    # ========================================================
    # STAGE 5 -- QC-1, QC-2, QC-3, QC-4
    # ========================================================
    print("\n── STAGE 5: QC-1 / QC-2 / QC-3 / QC-4 " + "─" * 25)
    violate_qc1 = qc1_ts_tp(df_evt, geom, CONFIG)
    violate_qc2, ev_max_ratio, qc2_threshold, all_ratios = qc2_pseudo_distance(
        df_evt, sta_coord, geom, CONFIG, folders)

    violate_qc4, dropped_pairs, qc4_details = qc4_origin_time(df_evt, geom, CONFIG)
    if dropped_pairs:
        # Outlier stations are removed first so the QC-3 fits are not contaminated
        df_evt, df_assign_clean, violate_after = apply_qc4_station_drop(
            df_evt, df_assign_clean, dropped_pairs, CONFIG)
        violate_qc4 |= violate_after

    violate_qc3, qc3_details = qc3_slope_intercept(df_evt, CONFIG)

    all_rejected = (violate_qc0 | violate_qc1 | violate_qc2
                    | violate_qc3 | violate_qc4)

    reject_reason = {}
    for ev in all_rejected:
        reasons = []
        if ev in violate_qc0: reasons.append('QC0:P-S/pairs')
        if ev in violate_qc1: reasons.append('QC1:Ts-Tp')
        if ev in violate_qc2: reasons.append('QC2:geometry')
        if ev in violate_qc3: reasons.append('QC3:slope/intc')
        if ev in violate_qc4: reasons.append('QC4:origin-time')
        reject_reason[ev] = '+'.join(reasons)

    total_rejected = len(all_rejected)
    total_passed   = total_orig - total_rejected

    sets  = [violate_qc0, violate_qc1, violate_qc2, violate_qc3, violate_qc4]
    only  = [s - set().union(*[o for j, o in enumerate(sets) if j != i])
             for i, s in enumerate(sets)]
    multi = all_rejected - set().union(*only)

    print("\n" + "=" * 64)
    print("                  QUALITY CONTROL SUMMARY")
    print("=" * 64)
    print(f"  Events associated                      : {total_orig:>8,}")
    print(f"  QC-0 (dedup / too few P-S pairs)       : {len(violate_qc0):>8,}")
    print(f"  QC-1 (Ts-Tp out of bounds)             : {len(violate_qc1):>8,}")
    print(f"  QC-2 (impossible station geometry)     : {len(violate_qc2):>8,}")
    print(f"  QC-3 (non-physical slope/intercept)    : {len(violate_qc3):>8,}")
    print(f"  QC-4 (inconsistent origin time)        : {len(violate_qc4):>8,}")
    print("  " + "─" * 56)
    print(f"  QC-0/1/2/3/4 only                      : "
          f"{' / '.join(f'{len(o):,}' for o in only)}")
    print(f"  Rejected by >=2 criteria               : {len(multi):>8,}")
    print("  " + "─" * 56)
    print(f"  Total unique rejections                : {total_rejected:>8,}")
    print(f"  ✅ Events PASSING QC                   : {total_passed:>8,}")
    print("=" * 64)

    # ========================================================
    # STAGE 6 -- Save QC results
    # ========================================================
    print("\n── STAGE 6: Save QC results " + "─" * 35)
    df_assign_qc = df_assign_clean[~df_assign_clean['event_idx'].isin(all_rejected)]
    df_cat_qc    = df_cat[~df_cat['event_index'].isin(all_rejected)]

    df_assign_qc.to_csv(path_rev_assign, index=False)
    df_cat_qc.to_csv(path_rev_catalog, index=False)
    print(f"   ✅ {path_rev_assign} : {len(df_assign_qc):,} picks")
    print(f"   ✅ {path_rev_catalog} : {len(df_cat_qc):,} events")

    # --- Per-event QC report (metrics + status) ---
    qc_rows = []
    for ev in sorted(df_evt['event_idx'].unique()):
        info  = qc3_details.get(ev, {})
        info4 = qc4_details.get(ev, {})
        qc_rows.append({
            'event_idx'     : ev,
            'event_id'      : make_event_id(ev),
            'status'        : 'REJECT' if ev in all_rejected else 'PASS',
            'reject_reason' : reject_reason.get(ev, ''),
            'n_station'     : df_evt[df_evt['event_idx'] == ev]['station'].nunique(),
            'max_ratio_qc2' : ev_max_ratio.get(ev, np.nan),
            'slope_ps'      : info.get('slope_ps', np.nan),
            'intercept_ps'  : info.get('intc_ps', np.nan),
            'r2_ps'         : info.get('r2_ps', np.nan),
            'vp_apparent'   : info.get('vp_app', np.nan),
            'slope_wadati'  : info.get('slope_wd', np.nan),
            'intercept_wadati': info.get('intc_wd', np.nan),
            'r2_wadati'     : info.get('r2_wd', np.nan),
            'vpvs_data'     : info.get('vpvs', np.nan),
            'ot_median_s'   : info4.get('ot_median', np.nan),
            'ot_mad_s'      : info4.get('ot_mad', np.nan),
            'ot_max_resid_s': info4.get('ot_max_resid', np.nan),
            'ot_n_outlier'  : info4.get('n_outlier', np.nan),
        })
    pd.DataFrame(qc_rows).to_csv(path_qc_report, index=False)
    print(f"   ✅ {path_qc_report} : {len(qc_rows):,} events")

    # ========================================================
    # STAGE 6b -- Cross-event summary + NonLinLoc input
    # ========================================================
    df_event_summary = pd.DataFrame()
    df_phase_picks   = pd.DataFrame()

    if CONFIG['write_event_summary']:
        print("\n── STAGE 6b: Cross-event summary " + "─" * 30)
        df_event_summary, df_phase_picks = build_summary_tables(
            df_evt, df_cat, stations, qc3_details, ev_max_ratio,
            all_rejected, reject_reason, geom, CONFIG)

        path_event_summary = os.path.join(csv_dir, "event_summary.csv")
        path_phase_picks   = os.path.join(csv_dir, "phase_picks.csv")
        df_event_summary.to_csv(path_event_summary, index=False)
        df_phase_picks.to_csv(path_phase_picks, index=False)

        print(f"   ✅ {path_event_summary} : {len(df_event_summary):,} events")
        print(f"   ✅ {path_phase_picks} : {len(df_phase_picks):,} picks")
        if not df_event_summary.empty:
            print(f"   P-S pairs per event : "
                  f"min {df_event_summary['n_ps_pairs'].min()} | "
                  f"median {df_event_summary['n_ps_pairs'].median():.0f} | "
                  f"max {df_event_summary['n_ps_pairs'].max()}")
            print(f"   Azimuthal gap       : "
                  f"median {df_event_summary['azimuthal_gap'].median():.1f}° | "
                  f"< 180° : {(df_event_summary['azimuthal_gap'] < 180).sum():,} events")

        if CONFIG['write_nonlinloc'] and not df_phase_picks.empty:
            print("\n── STAGE 6c: NonLinLoc input " + "─" * 34)
            write_nonlinloc_outputs(df_phase_picks, stations, folders, CONFIG)
            if CONFIG['write_search_volume']:
                write_search_volume(geom, stations, folders, CONFIG)

    # ========================================================
    # STAGE 7 -- Per-event plots
    # ========================================================
    print("\n── STAGE 7: Per-event plots " + "─" * 35)
    n_error = render_all_event_plots(
        df_evt, qc3_details, all_rejected, reject_reason,
        CONFIG, folders, checkpoint)

    # ========================================================
    # STAGE 8 -- Summary plots
    # ========================================================
    if CONFIG['plot_summary']:
        print("\n── STAGE 8: Summary plots " + "─" * 37)
        plot_catalog_summary(df_cat_qc, df_cat, stations, CONFIG, folders)
        plot_global_wadati_pseudo(df_evt, all_rejected, geom, CONFIG, folders)
        print(f"   ✅ Summary plots written to {folders['summary']}")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================
    print("\n" + "=" * 64)
    print("  FINAL SUMMARY")
    print("=" * 64)
    print(f"  Events associated        : {total_orig:,}")
    print(f"  Events passing QC        : {total_passed:,}")
    print(f"  Events rejected by QC    : {total_rejected:,}")
    print(f"  Events failed to plot    : {n_error:,}")
    print(f"  Passed events folder     : {folders['events']}")
    print(f"  Rejected events folder   : {folders['rejected']}")
    print(f"  Total runtime            : {time.time() - t0:.1f} s")
    print("\n✅ PIPELINE COMPLETE!")


if __name__ == "__main__":
    main()
