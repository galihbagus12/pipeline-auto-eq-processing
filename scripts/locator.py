# ============================================================
# NonLinLoc Hypocentre Location Pipeline
#
# Stage 3 of the workflow:
#   1) pick.py         -> phase picking (EQTransformer / SeisBench)
#   2) association.py  -> phase association + quality control
#   3) locator.py      -> hypocentre location (this file)
#
# Drives the NonLinLoc binaries end to end:
#   Vel2Grid   -> velocity model grids (P and S in a single run)
#   Grid2Time  -> travel-time grids per station (one run per phase)
#   NLLoc      -> probabilistic hypocentre location
#
# Input  : per-event .pick files written by association.py
#          + stations.txt + the 1D layered velocity model in CONFIG
#
# Output structure :
#   output_folder/run/                        control files handed to the binaries
#   output_folder/model/                      Vel2Grid output (velocity grids)
#   output_folder/time/                       Grid2Time output (travel-time grids)
#   output_folder/obs/                        the .pick files actually located
#   output_folder/loc/                        NLLoc output (.hyp / .hdr / .scat)
#   output_folder/csv/hypocenter_catalog.csv  one row per located event
#   output_folder/csv/hypocenter_phases.csv   one row per arrival, with residuals
#   output_folder/summary/                    summary figures
#   output_folder/checkpoint.json             progress tracking
#
# NOTE: This file is deliberately split into:
#   1) MODULE IMPORTS
#   2) CONFIGURATION (loaded from config/config.yaml -- see CONFIG_YAML_PATH)
#   3) PROCESS (utilities, control file builder, binary drivers, parsing, plots)
# CONFIG is loaded from the 'hypocenter_locator' section of the shared
# config/config.yaml (see pipeline.py at the project root, which runs every
# stage from that same file) -- edit the YAML to change parameters, not this
# file; the CONFIG dict keys and every function signature below are
# unchanged, so PROCESS never needs to be touched for a config-only change.
# CONFIG['velocity_model'] is the one exception -- it is not stored directly
# in the YAML but parsed from config/vel_model/jatim_velocity_model.txt (see
# CONFIG['velocity_model_file']) by load_velocity_model_from_layer_file()
# below, so the layered model has one canonical source shared with
# relocation.py.
# ============================================================

# ============================================================
# 1) MODULE IMPORTS
# ============================================================

import os
import json
import math
import glob
import time
import shutil
import subprocess
import warnings

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
    from obspy import read_events
except ImportError:
    raise ImportError("Package 'obspy' is not installed. Run: pip install obspy")

try:
    import pygmt
    HAS_PYGMT = True
except ImportError:
    HAS_PYGMT = False


# ============================================================
# 2) CONFIGURATION
# ============================================================

def load_velocity_model_from_layer_file(path):
    """Parse an NLLoc LAYER-format file (see config/vel_model/) into the list
    of (depth_top_km, Vp_km_s, Vs_km_s, density_g_cm3) tuples CONFIG['velocity_model']
    expects -- same shape this used to be hardcoded as. Velocity gradient
    columns are always 0 in this project's model (homogeneous per layer) and
    are dropped here."""
    layers = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != 'LAYER':
                continue
            # LAYER  depth  Vp  Vp_grad  Vs  Vs_grad  density  density_grad
            depth, vp, vs, density = float(parts[1]), float(parts[2]), float(parts[4]), float(parts[6])
            layers.append((depth, vp, vs, density))
    return layers


CONFIG_YAML_PATH = "/media/galih/MyBackUp 2024/Jatim_new/config/config.yaml"

with open(CONFIG_YAML_PATH, 'r') as _f:
    _all_cfg = yaml.safe_load(_f)

CONFIG = _all_cfg['hypocenter_locator']
CONFIG['network'] = _all_cfg['network']   # single global network code (see stations.txt)
CONFIG['velocity_model'] = load_velocity_model_from_layer_file(CONFIG['velocity_model_file'])


# ============================================================
# 3) PROCESS
# ============================================================

# ------------------------------------------------------------
# 3.1 General utilities
# ------------------------------------------------------------

def setup_output_folders(output_folder):
    """Create the NonLinLoc working tree."""
    folders = {
        'root'    : output_folder,
        'run'     : os.path.join(output_folder, 'run'),
        'model'   : os.path.join(output_folder, 'model'),
        'time'    : os.path.join(output_folder, 'time'),
        'obs'     : os.path.join(output_folder, 'obs'),
        'loc'     : os.path.join(output_folder, 'loc'),
        'csv'     : os.path.join(output_folder, 'csv'),
        'summary' : os.path.join(output_folder, 'summary'),
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    return folders


def load_checkpoint(checkpoint_file):
    """
    Load run progress. This is the only progress tracker used by the pipeline.

    Return dict:
      'grids_built'   : set of phases whose travel-time grids are complete
      'located_runs'  : set of NLLoc run tags already executed

    Missing or corrupt file -> both empty, meaning start from scratch.
    """
    empty = {'grids_built': set(), 'located_runs': set()}
    if not os.path.exists(checkpoint_file):
        return empty
    try:
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        return {
            'grids_built' : set(data.get('grids_built', [])),
            'located_runs': set(data.get('located_runs', [])),
        }
    except Exception:
        return empty


def save_checkpoint(checkpoint_file, checkpoint):
    """Save run progress."""
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'grids_built'  : sorted(checkpoint['grids_built']),
            'located_runs' : sorted(checkpoint['located_runs']),
            'last_update'  : str(pd.Timestamp.now()),
        }, f, indent=2)


def load_stations_dataframe(path, network):
    """
    Load stations.txt (columns: station latitude longitude elevation_m) ->
    DataFrame with columns: station, network, latitude, longitude,
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
                'station'     : sta,
                'network'     : network,
                'latitude'    : float(lat),
                'longitude'   : float(lon),
                'elevation_m' : float(elev),
            })
    return pd.DataFrame(rows).sort_values('station').reset_index(drop=True)


def make_logger(cfg, t0):
    """Stage logger with elapsed-time markers (same style as the other stages)."""
    def log_step(msg):
        if cfg.get('verbose_timing'):
            print(f"  [{time.strftime('%H:%M:%S')}] (+{time.time() - t0:7.1f}s) {msg}",
                  flush=True)
    return log_step


def find_binary(cfg, name):
    """Locate a NonLinLoc executable, preferring nll_bin_dir then PATH."""
    candidate = os.path.join(cfg['nll_bin_dir'], name)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"NonLinLoc executable '{name}' not found in {cfg['nll_bin_dir']} or on PATH")


def run_binary(binary, control_path, cwd, cfg, label):
    """
    Execute one NonLinLoc binary.

    NonLinLoc resolves the relative paths inside a control file against the
    current working directory, so cwd is always the output root.
    """
    cmd = [binary, control_path]
    print(f"   $ {os.path.basename(binary)} {control_path}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    tail = [l for l in (proc.stdout or '').strip().splitlines() if l.strip()]
    for line in tail[-cfg['binary_log_lines']:]:
        print(f"     | {line}", flush=True)

    if proc.returncode != 0:
        err = (proc.stderr or '').strip().splitlines()
        for line in err[-10:]:
            print(f"     ! {line}", flush=True)
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")
    return proc


# ------------------------------------------------------------
# 3.2 Stage 1 -- Geometry derived from stations and search grid
# ------------------------------------------------------------

def load_search_volume(cfg):
    """
    Read the search volume handed over by association.py.

    That file fixes the projection origin and the LOCGRID, both derived from the
    station distribution, so the location stage searches exactly the volume the
    association stage used. Values in grid_override replace individual entries.
    """
    path = os.path.join(cfg['association_nll_dir'], 'search_volume.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"search_volume.json not found: {path}\n"
            "Run association.py first -- it derives the search volume from the "
            "station distribution and writes it there.")

    with open(path, 'r') as f:
        vol = json.load(f)

    for key in ('trans_lat_orig', 'trans_lon_orig', 'trans_rot_angle',
                'grid_x_orig', 'grid_y_orig', 'grid_z_orig',
                'grid_x_num', 'grid_y_num', 'grid_z_num'):
        if key not in vol:
            raise KeyError(f"search_volume.json is missing '{key}': {path}")

    d = vol.get('node_spacing_km', 0.5)
    vol['grid_dx'] = vol['grid_dy'] = vol['grid_dz'] = d
    vol['trans_type'] = vol.get('trans_type', 'SIMPLE')

    if cfg.get('grid_override'):
        vol.update(cfg['grid_override'])
        print("   ⚠️  grid_override applied -- the handed-over grid was replaced")

    print(f"   Search volume from : {path}")
    print(f"   Derived by         : {vol.get('generated_by', '?')} "
          f"on {str(vol.get('generated_at', '?'))[:19]}")
    print(f"   Network aperture   : {vol.get('aperture_km', float('nan')):.2f} km "
          f"({vol.get('n_stations', '?')} stations)")
    return vol


def grid_extent(vol):
    """Return the LOCGRID extent in km as ((x0,x1),(y0,y1),(z0,z1))."""
    x1 = vol['grid_x_orig'] + (vol['grid_x_num'] - 1) * vol['grid_dx']
    y1 = vol['grid_y_orig'] + (vol['grid_y_num'] - 1) * vol['grid_dy']
    z1 = vol['grid_z_orig'] + (vol['grid_z_num'] - 1) * vol['grid_dz']
    return ((vol['grid_x_orig'], x1), (vol['grid_y_orig'], y1),
            (vol['grid_z_orig'], z1))


def station_xy_km(stations, vol):
    """
    Project stations into the NonLinLoc SIMPLE frame:
        x = (lon - lon_orig) * 111.111 * cos(lat)
        y = (lat - lat_orig) * 111.111
    """
    C = 111.111
    lat0, lon0 = vol['trans_lat_orig'], vol['trans_lon_orig']
    xs = ((stations['longitude'] - lon0) * C
          * np.cos(np.radians(stations['latitude'])))
    ys = (stations['latitude'] - lat0) * C
    return xs.values, ys.values


def derive_2d_grid(stations, vol, cfg):
    """
    Size the 2D travel-time grid used by GRID2D.

    Grid2Time needs the grid to reach from every station to the farthest corner
    of the search volume, otherwise arrivals at distant stations cannot be
    predicted. The required distance is therefore the maximum station-to-corner
    separation, plus a margin.

    Return dict with the VGGRID parameters and the derived distance.
    """
    (gx0, gx1), (gy0, gy1), (gz0, gz1) = grid_extent(vol)
    xs, ys = station_xy_km(stations, vol)

    explicit = cfg.get('vg2d_distance_km')
    if explicit is None:
        max_dist = 0.0
        for x, y in zip(xs, ys):
            for cx in (gx0, gx1):
                for cy in (gy0, gy1):
                    max_dist = max(max_dist, math.hypot(cx - x, cy - y))
        max_dist += cfg['vg2d_margin_km']
    else:
        max_dist = float(explicit)

    ny = int(math.ceil(max_dist / vol['grid_dy'])) + 1
    nz = vol['grid_z_num']

    # One float per node; 2 nodes on the degenerate x axis.
    bytes_per_grid = 2 * ny * nz * 4
    n_phase = 2 if cfg['use_s_grids'] else 1
    n_files = 2 if cfg['compute_angles'] else 1
    total_gb = bytes_per_grid * len(stations) * n_phase * n_files / 1e9

    geom = {
        'max_dist_km'    : max_dist,
        'vg_nx'          : 2,
        'vg_ny'          : ny,
        'vg_nz'          : nz,
        'grid_extent'    : ((gx0, gx1), (gy0, gy1), (gz0, gz1)),
        'sta_x_range'    : (float(xs.min()), float(xs.max())),
        'sta_y_range'    : (float(ys.min()), float(ys.max())),
        'bytes_per_grid' : bytes_per_grid,
        'total_gb'       : total_gb,
    }

    print("   ┌─ Grid geometry " + "─" * 47)
    print(f"   │ Stations                : {len(stations)}")
    print(f"   │ Projection origin       : lat {vol['trans_lat_orig']:.4f}, "
          f"lon {vol['trans_lon_orig']:.4f}")
    print(f"   │ Station X range         : {xs.min():8.1f} .. {xs.max():.1f} km")
    print(f"   │ Station Y range         : {ys.min():8.1f} .. {ys.max():.1f} km")
    print(f"   │ LOCGRID X               : {gx0:8.1f} .. {gx1:.1f} km "
          f"({vol['grid_x_num']} nodes)")
    print(f"   │ LOCGRID Y               : {gy0:8.1f} .. {gy1:.1f} km "
          f"({vol['grid_y_num']} nodes)")
    print(f"   │ LOCGRID Z               : {gz0:8.1f} .. {gz1:.1f} km "
          f"({vol['grid_z_num']} nodes)")
    print("   ├─ Travel-time grids (GRID2D) " + "─" * 34)
    print(f"   │ Max station-corner dist : {max_dist:8.1f} km")
    print(f"   │ VGGRID (2D)             : {geom['vg_nx']} x {ny} x {nz}")
    print(f"   │ Size per grid           : {bytes_per_grid/1e6:8.1f} MB")
    print(f"   │ Estimated total on disk : {total_gb:8.2f} GB "
          f"({len(stations)} stations, {n_phase} phase(s))")
    print("   └" + "─" * 62)

    return geom


# ------------------------------------------------------------
# 3.3 Stage 2 -- Control file construction
# ------------------------------------------------------------

def build_control_file(stations, geom, vol, cfg, phase, folders):
    """
    Compose a complete NonLinLoc control file.

    One file serves all three binaries. Only GTFILES carries a phase, and only
    Grid2Time reads it, so a separate file is written per phase; Vel2Grid and
    NLLoc can use either.

    All paths are relative to the output root, which is also the working
    directory the binaries are launched from.
    """
    lat0, lon0 = vol['trans_lat_orig'], vol['trans_lon_orig']
    rot        = vol.get('trans_rot_angle', 0.0)
    trans      = f"TRANS  SIMPLE  {lat0} {lon0} {rot}"

    # --- Vel2Grid ---
    # Both VGTYPE lines are honoured in a single Vel2Grid run, producing the P
    # and S velocity grids together.
    vgtypes = "VGTYPE  P" + ("\nVGTYPE  S" if cfg['use_s_grids'] else "")
    vggrid = (f"VGGRID  {geom['vg_nx']} {geom['vg_ny']} {geom['vg_nz']}  "
              f"0.0 0.0 {vol['grid_z_orig']}  "
              f"{vol['grid_dx']} {vol['grid_dy']} {vol['grid_dz']}  SLOW_LEN")

    layers = "\n".join(
        f"LAYER  {d:7.2f}  {vp:6.3f} 0.00  {vs:6.3f} 0.00  {rho:5.3f} 0.00"
        for d, vp, vs, rho in cfg['velocity_model'])

    # --- Grid2Time ---
    angles = 'ANGLES_YES' if cfg['compute_angles'] else 'ANGLES_NO'
    gtsrce = "\n".join(
        f"GTSRCE  {r.station:<6} LATLON  {r.latitude:10.6f} {r.longitude:11.6f}  "
        f"0.0  {r.elevation_m / 1000.0:.4f}"
        for r in stations.itertuples())

    # --- NLLoc ---
    # A positive VpVsRatio makes NLLoc derive S times from the P grid; -1
    # disables that so the explicit S grids are used instead.
    vpvs = -1.0 if cfg['use_s_grids'] else cfg['loc_vpvs_ratio']
    locmeth = (f"LOCMETH  {cfg['loc_method']} {cfg['loc_max_dist_sta']} "
               f"{cfg['loc_min_phases']} {cfg['loc_max_phases']} "
               f"{cfg['loc_min_s_phases']} {vpvs} {cfg['loc_max_3d_mem']} "
               f"{cfg['loc_min_dist_sta']} {cfg['loc_reject_dup']}")

    locgrid = (f"LOCGRID  {vol['grid_x_num']} {vol['grid_y_num']} {vol['grid_z_num']}  "
               f"{vol['grid_x_orig']} {vol['grid_y_orig']} {vol['grid_z_orig']}  "
               f"{vol['grid_dx']} {vol['grid_dy']} {vol['grid_dz']}  "
               f"PROB_DENSITY  SAVE")

    obs_glob = "./obs/*.pick"

    return f"""# ============================================================
# NonLinLoc control file -- generated by locator.py
# Phase for Grid2Time: {phase}
# ============================================================
CONTROL 1 54321
{trans}

# ---------------- Vel2Grid ----------------
VGOUT   ./model/layer
{vgtypes}
{vggrid}
{layers}

# ---------------- Grid2Time ----------------
GTFILES  ./model/layer  ./time/layer  {phase}
GTMODE   GRID2D {angles}
{gtsrce}
GT_PLFD  1.0e-3  0

# ---------------- NLLoc ----------------
LOCSIG   {cfg['loc_signature']}
LOCCOM   {cfg['loc_comment']}
LOCFILES {obs_glob} NLLOC_OBS  ./time/layer  ./loc/jatim
LOCHYPOUT {cfg['loc_hypout']}
LOCSEARCH {cfg['loc_search']}
{locgrid}
{locmeth}
LOCGAU   {cfg['loc_gau']}
LOCGAU2  {cfg['loc_gau2']}
LOCPHASEID  P   P p G PN PG
LOCPHASEID  S   S s G SN SG
LOCQUAL2ERR {cfg['loc_qual2err']}
LOCANGLES {angles} 5
"""


def write_control_files(stations, geom, vol, cfg, folders):
    """Write one control file per phase; return {phase: path}."""
    phases = ['P', 'S'] if cfg['use_s_grids'] else ['P']
    paths = {}
    for phase in phases:
        text = build_control_file(stations, geom, vol, cfg, phase, folders)
        path = os.path.join(folders['run'], f"nlloc_{phase}.in")
        with open(path, 'w') as f:
            f.write(text)
        paths[phase] = path
        print(f"   ✅ control file ({phase}) -> {path}")
    return paths


# ------------------------------------------------------------
# 3.4 Stage 3 -- Collect the observation files
# ------------------------------------------------------------

def collect_pick_files(cfg, folders):
    """
    Copy the per-event .pick files produced by association.py into the working
    obs/ directory. Stale files from earlier runs are cleared first so the
    LOCFILES glob cannot pick up events that no longer exist.
    """
    src_dir = os.path.join(cfg['association_nll_dir'], 'picks')
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            f"Pick directory not found: {src_dir}\n"
            "Run association.py first (it writes nonlinloc/picks/).")

    sources = sorted(glob.glob(os.path.join(src_dir, "*.pick")))
    if not sources:
        raise FileNotFoundError(f"No .pick files found in {src_dir}")

    for old in glob.glob(os.path.join(folders['obs'], "*.pick")):
        os.remove(old)

    n_phase = 0
    for src in sources:
        shutil.copy2(src, folders['obs'])
        with open(src) as f:
            n_phase += sum(1 for line in f if line.strip())

    print(f"   Events to locate : {len(sources):,}")
    print(f"   Phase readings   : {n_phase:,}")
    return len(sources)


# ------------------------------------------------------------
# 3.5 Stage 4 -- Build velocity and travel-time grids
# ------------------------------------------------------------

def grids_present(folders, cfg, stations):
    """True when a complete set of travel-time grids already exists."""
    phases = ['P', 'S'] if cfg['use_s_grids'] else ['P']
    for phase in phases:
        for sta in stations['station']:
            hdr = os.path.join(folders['time'], f"layer.{phase}.{sta}.time.hdr")
            if not os.path.exists(hdr):
                return False
    return True


def build_grids(control_paths, folders, cfg, stations, checkpoint):
    """
    Run Vel2Grid once (it emits P and S velocity grids together), then
    Grid2Time once per phase.

    Grid2Time is the expensive stage: it solves the eikonal equation for every
    station. Results are cached and reused on later runs unless rebuild_grids
    is set.
    """
    phases = ['P', 'S'] if cfg['use_s_grids'] else ['P']

    if cfg['rebuild_grids']:
        print("   rebuild_grids = True -> clearing existing model and time grids")
        for d in (folders['model'], folders['time']):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        checkpoint['grids_built'].clear()
    elif grids_present(folders, cfg, stations):
        print("   Travel-time grids already present -> skipping Vel2Grid/Grid2Time")
        checkpoint['grids_built'].update(phases)
        save_checkpoint(cfg['checkpoint_file'], checkpoint)
        return

    vel2grid  = find_binary(cfg, 'Vel2Grid')
    grid2time = find_binary(cfg, 'Grid2Time')
    root      = folders['root']

    # A single Vel2Grid run handles every VGTYPE listed in the control file.
    run_binary(vel2grid, os.path.relpath(control_paths[phases[0]], root),
               root, cfg, 'Vel2Grid')

    # Grid2Time reads one GTFILES phase per run, so it is invoked per phase.
    for phase in phases:
        run_binary(grid2time, os.path.relpath(control_paths[phase], root),
                   root, cfg, f'Grid2Time ({phase})')
        checkpoint['grids_built'].add(phase)
        save_checkpoint(cfg['checkpoint_file'], checkpoint)


# ------------------------------------------------------------
# 3.6 Stage 5 -- Locate
# ------------------------------------------------------------

def run_nlloc(control_paths, folders, cfg, checkpoint):
    """Run NLLoc over every .pick file matched by the LOCFILES glob."""
    for pattern in ("*.hyp", "*.hdr", "*.scat"):
        for old in glob.glob(os.path.join(folders['loc'], pattern)):
            os.remove(old)

    nlloc = find_binary(cfg, 'NLLoc')
    root  = folders['root']
    phase = 'P' if 'P' in control_paths else sorted(control_paths)[0]
    run_binary(nlloc, os.path.relpath(control_paths[phase], root),
               root, cfg, 'NLLoc')

    checkpoint['located_runs'].add(str(pd.Timestamp.now()))
    save_checkpoint(cfg['checkpoint_file'], checkpoint)


# ------------------------------------------------------------
# 3.7 Stage 6 -- Parse the NLLoc output
# ------------------------------------------------------------

def _find_value(parts, keyword):
    """Return the number following `keyword` on a whitespace-split NLLoc line."""
    try:
        return float(parts[parts.index(keyword) + 1])
    except (ValueError, IndexError):
        return np.nan


def parse_hyp_metrics(path):
    """
    Read the quality metrics that obspy does not expose directly from a .hyp file.

      rms_s          QUALITY -> RMS            travel-time residual (s)
      azimuthal_gap  QUALITY -> Gap            largest azimuthal gap (deg)
      pdf_volume_km3 SEARCH  -> scatter_volume volume of the location PDF (km^3)
      err_z_km       STATISTICS -> CovZZ       sqrt of the depth covariance (km)
      err_h_km       QML_OriginUncertainty -> maxHorUnc   horizontal error (km)

    err_h and err_z describe the shape of the confidence ellipsoid, while
    pdf_volume measures how tightly the probability density is concentrated --
    a large volume means the solution is poorly constrained even when the
    residual happens to be small.
    """
    m = {'rms_s': np.nan, 'azimuthal_gap': np.nan, 'pdf_volume_km3': np.nan,
         'err_z_km': np.nan, 'err_h_km': np.nan}
    try:
        with open(path, 'r') as f:
            for line in f:
                p = line.split()
                if not p:
                    continue
                if p[0] == 'QUALITY':
                    m['rms_s']         = _find_value(p, 'RMS')
                    m['azimuthal_gap'] = _find_value(p, 'Gap')
                elif p[0] == 'SEARCH' and 'OCTREE' in p:
                    m['pdf_volume_km3'] = _find_value(p, 'scatter_volume')
                elif p[0] == 'STATISTICS':
                    cov_zz = _find_value(p, 'ZZ')
                    if np.isfinite(cov_zz):
                        m['err_z_km'] = math.sqrt(abs(cov_zz))
                elif p[0] == 'QML_OriginUncertainty':
                    m['err_h_km'] = _find_value(p, 'maxHorUnc')
    except Exception:
        pass
    return m


def get_sampling_rate(cfg):
    """
    Read the sampling rate from the waveform data.

    The RMS quality threshold is expressed in samples rather than seconds
    because a pick can never be more precise than the sample interval, so the
    limit follows from the recording rather than from a chosen number.
    """
    if cfg.get('qc_sampling_rate'):
        return float(cfg['qc_sampling_rate']), 'CONFIG'
    try:
        from obspy import read as _read
        pattern = os.path.join(cfg['waveform_folder'], '*', '*.mseed')
        files = sorted(glob.glob(pattern))[:1]
        if files:
            st = _read(files[0], headonly=True)
            if len(st):
                return float(st[0].stats.sampling_rate), os.path.basename(files[0])
    except Exception:
        pass
    return np.nan, 'unavailable'


def apply_location_qc(df_cat, cfg):
    """
    Select the high-quality subset of the located catalogue.

    An event is kept only if it satisfies every criterion at once:
      * rms below an absolute threshold set by the sampling rate
      * err_h, err_z, azimuthal_gap and pdf_volume each at or below the
        catalogue median

    Using the median makes four of the five thresholds adapt to the dataset
    instead of being imposed on it: the question asked of each event is whether
    it is better than typical on every metric simultaneously. Because the
    criteria combine with AND, far fewer than half the events survive.

    Return: (df_cat with qc columns added, thresholds dict, sampling rate info)
    """
    df = df_cat.copy()
    metrics = [m for m in cfg['qc_metrics'] if m in df.columns]

    sr, sr_src = get_sampling_rate(cfg)
    rms_thr = (cfg['qc_rms_n_samples'] / sr) if np.isfinite(sr) else np.nan

    thresholds = {}
    for m in metrics:
        if m in cfg['qc_median_metrics']:
            thresholds[m] = float(df[m].median())
        elif m == 'rms_s':
            thresholds[m] = rms_thr
        else:
            thresholds[m] = float(df[m].median())

    print(f"   Sampling rate      : {sr:g} Hz (from {sr_src})")
    print(f"   RMS threshold      : {cfg['qc_rms_n_samples']} samples / {sr:g} Hz "
          f"= {rms_thr:.4f} s")
    print("   Threshold per metric:")
    for m in metrics:
        kind = 'median' if m in cfg['qc_median_metrics'] else 'absolute'
        print(f"     {m:<16} <= {thresholds[m]:12.4f}   ({kind})")

    mask = pd.Series(True, index=df.index)
    for m in metrics:
        if np.isfinite(thresholds[m]):
            mask &= df[m] <= thresholds[m]
    df['qc_pass'] = mask

    # Which criterion removed each rejected event, for auditing
    reasons = []
    for _, row in df.iterrows():
        if row['qc_pass']:
            reasons.append('')
            continue
        failed = [m for m in metrics
                  if np.isfinite(thresholds[m]) and not (row[m] <= thresholds[m])]
        reasons.append('+'.join(failed))
    df['qc_reject_reason'] = reasons

    n_pass = int(mask.sum())
    print(f"   High quality       : {n_pass:,} of {len(df):,} events "
          f"({100.0 * n_pass / max(len(df), 1):.1f}%)")
    if n_pass:
        counts = (df.loc[~df['qc_pass'], 'qc_reject_reason']
                  .value_counts().head(6))
        for reason, n in counts.items():
            print(f"     rejected by {reason:<40} {n:>5,}")

    return df, thresholds, (sr, sr_src)


def parse_locations(folders, cfg):
    """
    Read every .hyp file into two tables:

      catalog : one row per located event -- origin time, position, quality
                metrics (RMS, azimuthal gap, phase and station counts, distance
                range, uncertainty ellipsoid) and the derived grade
      phases  : one row per arrival -- observed time, residual and weight, which
                is what reveals systematic station problems
    """
    # NLLoc also writes aggregate files alongside the per-event ones:
    #   *.sum.*  -> every event concatenated (from LOCHYPOUT SAVE_NLLOC_SUM)
    #   last.hyp -> a copy of the most recent event
    # Both would duplicate events in the catalogue, so only per-event files are read.
    hyp_files = sorted(
        p for p in glob.glob(os.path.join(folders['loc'], "*.loc.hyp"))
        if '.sum.' not in os.path.basename(p)
        and os.path.basename(p) != 'last.hyp'
    )
    if not hyp_files:
        print("   ⚠️  No .hyp files produced -- nothing was located")
        return pd.DataFrame(), pd.DataFrame()

    cat_rows, pha_rows = [], []
    n_failed = 0

    for path in tqdm(hyp_files, desc="Parsing locations", unit="event"):
        try:
            cat = read_events(path, format='NLLOC_HYP')
        except Exception:
            n_failed += 1
            continue
        if len(cat) == 0:
            n_failed += 1
            continue

        ev = cat[0]
        origin = ev.preferred_origin() or ev.origins[0]
        if origin is None or origin.time is None:
            n_failed += 1
            continue

        q = origin.quality
        nph   = int(q.used_phase_count) if q and q.used_phase_count else len(origin.arrivals)
        nsta  = int(q.used_station_count) if q and q.used_station_count else np.nan
        dmin  = float(q.minimum_distance) if q and q.minimum_distance is not None else np.nan
        dmax  = float(q.maximum_distance) if q and q.maximum_distance is not None else np.nan

        # RMS, gap, PDF volume and the error ellipsoid come from a direct read of
        # the .hyp text: obspy does not expose the octree scatter volume or the
        # depth covariance term.
        metrics = parse_hyp_metrics(path)

        event_id = os.path.basename(path).replace('.loc.hyp', '')
        depth_km = origin.depth / 1000.0 if origin.depth is not None else np.nan

        cat_rows.append({
            'event_id'       : event_id,
            'origin_time'    : origin.time.datetime,
            'latitude'       : origin.latitude,
            'longitude'      : origin.longitude,
            'depth_km'       : depth_km,
            'rms_s'          : metrics['rms_s'],
            'err_h_km'       : metrics['err_h_km'],
            'err_z_km'       : metrics['err_z_km'],
            'azimuthal_gap'  : metrics['azimuthal_gap'],
            'pdf_volume_km3' : metrics['pdf_volume_km3'],
            'n_phase'        : nph,
            'n_station'      : nsta,
            'dist_min_km'    : dmin,
            'dist_max_km'    : dmax,
        })

        for arr in origin.arrivals:
            pick = arr.pick_id.get_referred_object() if arr.pick_id else None
            sta  = (pick.waveform_id.station_code
                    if pick and pick.waveform_id else None)
            pha_rows.append({
                'event_id'   : event_id,
                'station'    : sta,
                'phase'      : arr.phase,
                'pick_time'  : pick.time.datetime if pick and pick.time else None,
                'residual_s' : arr.time_residual,
                'weight'     : arr.time_weight,
                'distance_km': (arr.distance * 111.195
                                if arr.distance is not None else np.nan),
                'azimuth'    : arr.azimuth,
            })

    if n_failed:
        print(f"   ⚠️  {n_failed} .hyp files could not be parsed")

    df_cat = pd.DataFrame(cat_rows).sort_values('origin_time').reset_index(drop=True)
    df_pha = pd.DataFrame(pha_rows)
    return df_cat, df_pha


# ------------------------------------------------------------
# 3.8 Stage 7 -- Summary plots
# ------------------------------------------------------------

# The five metrics the quality control acts on, with the labels and colours used
# consistently across every figure.
QC_METRIC_SPECS = [
    ('rms_s',          'RMS residual (s)',        'skyblue',    False),
    ('err_h_km',       'Horizontal error (km)',   'salmon',     False),
    ('err_z_km',       'Vertical error (km)',     'lightgreen', False),
    ('azimuthal_gap',  'Azimuthal gap (deg)',     'orange',     False),
    ('pdf_volume_km3', 'PDF volume (km$^3$)',     'orchid',     True),
]


def plot_qc_distributions(df_cat, thresholds, cfg, folders):
    """
    Distribution of every quality-control metric, before and after selection.

    Each panel shows the full catalogue in grey behind the retained events in
    colour, with the threshold marked. This makes the effect of the cut visible
    directly: where the threshold sits within the population, and how much of
    the distribution it removes. PDF volume spans several orders of magnitude,
    so it is drawn on a logarithmic axis.
    """
    if df_cat.empty:
        return
    try:
        passed = df_cat[df_cat['qc_pass']] if 'qc_pass' in df_cat else df_cat

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for i, (col, label, colour, log_scale) in enumerate(QC_METRIC_SPECS):
            ax = axes[i // 3, i % 3]
            if col not in df_cat.columns:
                ax.axis('off')
                continue

            allv = df_cat[col].replace([np.inf, -np.inf], np.nan).dropna()
            okv  = passed[col].replace([np.inf, -np.inf], np.nan).dropna()
            if allv.empty:
                ax.axis('off')
                continue

            if log_scale:
                allv = allv[allv > 0]
                okv  = okv[okv > 0]
                if allv.empty:
                    ax.axis('off')
                    continue
                bins = np.logspace(np.log10(allv.min()), np.log10(allv.max()), 50)
                ax.set_xscale('log')
            else:
                bins = np.linspace(allv.min(), allv.max(), 45)

            ax.hist(allv, bins=bins, color='lightgray', edgecolor='white',
                    label=f'All located (n={len(allv):,})')
            if not okv.empty:
                ax.hist(okv, bins=bins, color=colour, edgecolor='white',
                        alpha=0.9, label=f'Passed QC (n={len(okv):,})')

            thr = thresholds.get(col, np.nan)
            if np.isfinite(thr):
                kind = ('sampling rate' if col == 'rms_s'
                        else 'median')
                ax.axvline(thr, color='red', ls='--', lw=2,
                           label=f'Threshold = {thr:.4g} ({kind})')

            ax.set_xlabel(label)
            ax.set_ylabel('Count')
            ax.set_title(label, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # --- Retained versus rejected ---
        ax = axes[1, 2]
        n_all, n_ok = len(df_cat), len(passed)
        bars = ax.bar(['All located', 'Passed QC'], [n_all, n_ok],
                      color=['lightgray', '#238b45'], edgecolor='black')
        for b, v in zip(bars, [n_all, n_ok]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({100*v/n_all:.1f}%)",
                    ha='center', va='bottom', fontweight='bold')
        ax.set_ylabel('Events')
        ax.set_title('Quality control result', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle('NonLinLoc quality-control metrics', fontsize=15,
                     fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(folders['summary'],
                                 f"qc_distributions.{cfg['plot_format']}"),
                    dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        plt.close('all')
        print(f"   ⚠️  QC distribution plot failed: {e}")


def _simple_basemap(ax, lon_range, lat_range, stations, vol=None):
    """
    Draw a minimal geographic frame: correct aspect ratio, station markers, the
    association search area and a kilometre scale bar.

    No coastline dataset is required. The purpose is to judge the spatial extent
    of the catalogue against the network and the searched region, which needs
    true relative distances rather than cartographic detail.
    """
    lat_mid = float(np.mean(lat_range))
    ax.set_aspect(1.0 / max(math.cos(math.radians(lat_mid)), 1e-6))

    if vol is not None:
        (a, b), (c, d) = vol['lon_range'], vol['lat_range']
        ax.add_patch(plt.Rectangle((a, c), b - a, d - c, fill=False,
                                   edgecolor='navy', lw=1.4, ls='--',
                                   label='Association search area', zorder=1))

    ax.scatter(stations['longitude'], stations['latitude'], marker='^', s=70,
               c='black', edgecolors='white', lw=0.7, zorder=6, label='Stations')

    # Scale bar: a round number of km rendered in degrees of longitude
    span_km = (lon_range[1] - lon_range[0]) * 111.111 * math.cos(math.radians(lat_mid))
    for cand in (200, 100, 50, 25, 10):
        if cand < span_km * 0.45:
            bar_km = cand
            break
    else:
        bar_km = max(round(span_km / 4), 1)
    bar_deg = bar_km / (111.111 * math.cos(math.radians(lat_mid)))
    x0 = lon_range[0] + 0.06 * (lon_range[1] - lon_range[0])
    y0 = lat_range[0] + 0.06 * (lat_range[1] - lat_range[0])
    ax.plot([x0, x0 + bar_deg], [y0, y0], color='black', lw=3, zorder=7)
    ax.text(x0 + bar_deg / 2, y0, f"{bar_km} km", ha='center', va='bottom',
            fontsize=9, fontweight='bold', zorder=7)

    ax.set_xlim(*lon_range)
    ax.set_ylim(*lat_range)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3, ls=':')


def plot_qc_maps(df_cat, stations, vol, cfg, folders):
    """
    Spatial distribution of the quality metrics.

    One panel per metric, each coloured by that metric, drawn over the same
    simple basemap so the catalogue extent can be compared against the network
    footprint and the searched area. Spatial structure in these maps is
    diagnostic: uncertainty that grows away from the network is expected
    geometry, whereas a cluster of poor solutions inside the network points to a
    station or velocity problem instead.
    """
    if df_cat.empty:
        return
    try:
        pad = 0.25
        lon_range = (min(df_cat['longitude'].min(), stations['longitude'].min()) - pad,
                     max(df_cat['longitude'].max(), stations['longitude'].max()) + pad)
        lat_range = (min(df_cat['latitude'].min(), stations['latitude'].min()) - pad,
                     max(df_cat['latitude'].max(), stations['latitude'].max()) + pad)

        specs = [(c, l, cm, lg) for c, l, cm, lg in [
            ('rms_s',          'RMS residual (s)',      'hot_r',   False),
            ('err_h_km',       'Horizontal error (km)', 'magma_r', False),
            ('err_z_km',       'Vertical error (km)',   'viridis_r', False),
            ('azimuthal_gap',  'Azimuthal gap (deg)',   'jet',     False),
            ('pdf_volume_km3', 'PDF volume (km$^3$)',   'cool',    True),
            ('depth_km',       'Depth (km)',            'plasma_r', False),
        ] if c in df_cat.columns]

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        for i, (col, label, cmap, log_scale) in enumerate(specs):
            ax = axes[i // 3, i % 3]
            sub = df_cat[np.isfinite(df_cat[col])]
            if sub.empty:
                ax.axis('off')
                continue

            _simple_basemap(ax, lon_range, lat_range, stations, vol)

            values = sub[col]
            norm = None
            if log_scale:
                positive = values[values > 0]
                if not positive.empty:
                    norm = matplotlib.colors.LogNorm(vmin=positive.min(),
                                                     vmax=positive.max())
                    sub, values = sub[values > 0], positive

            sc = ax.scatter(sub['longitude'], sub['latitude'], c=values,
                            cmap=cmap, norm=norm, s=34, alpha=0.85,
                            edgecolors='black', linewidths=0.3, zorder=5)
            plt.colorbar(sc, ax=ax, label=label, shrink=0.85)
            ax.set_title(f"Coloured by {label}", fontweight='bold')
            if i == 0:
                ax.legend(fontsize=8, loc='upper right')

        for j in range(len(specs), 6):
            axes[j // 3, j % 3].axis('off')

        tag = 'high-quality events' if 'qc_pass' in df_cat.columns else 'located events'
        fig.suptitle(f"Spatial distribution of location quality -- "
                     f"{len(df_cat):,} {tag}", fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(folders['summary'],
                                 f"qc_maps.{cfg['plot_format']}"),
                    dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        plt.close('all')
        print(f"   ⚠️  QC map plot failed: {e}")


def plot_location_summary(df_cat, df_pha, stations, vol, cfg, folders):
    """Overview of the located catalogue: map, depth, quality metrics, residuals."""
    if df_cat.empty:
        print("   ⚠️  Catalogue is empty, summary plot skipped")
        return

    try:
        fig, axes = plt.subplots(2, 3, figsize=(19, 11))

        # --- Epicentre map ---
        ax = axes[0, 0]
        pad = 0.25
        lon_range = (min(df_cat['longitude'].min(), stations['longitude'].min()) - pad,
                     max(df_cat['longitude'].max(), stations['longitude'].max()) + pad)
        lat_range = (min(df_cat['latitude'].min(), stations['latitude'].min()) - pad,
                     max(df_cat['latitude'].max(), stations['latitude'].max()) + pad)
        _simple_basemap(ax, lon_range, lat_range, stations, vol)
        sc = ax.scatter(df_cat['longitude'], df_cat['latitude'],
                        c=df_cat['depth_km'], cmap='plasma_r',
                        s=40, alpha=0.75, edgecolors='black', linewidths=0.3,
                        zorder=6, label='Events')
        plt.colorbar(sc, ax=ax, label='Depth (km)')
        ax.set_title('Located epicentres', fontweight='bold')
        ax.legend(fontsize=8)

        # --- Depth distribution ---
        ax = axes[0, 1]
        ax.hist(df_cat['depth_km'].dropna(), bins=40, color='steelblue',
                edgecolor='white')
        ax.set_xlabel('Depth (km)')
        ax.set_ylabel('Count')
        ax.set_title('Depth distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # --- RMS residual ---
        ax = axes[0, 2]
        ax.hist(df_cat['rms_s'].dropna(), bins=40, color='darkorange',
                edgecolor='white')
        ax.axvline(df_cat['rms_s'].median(), color='red', ls='--', lw=1.5,
                   label=f"median = {df_cat['rms_s'].median():.3f} s")
        ax.set_xlabel('RMS travel-time residual (s)')
        ax.set_ylabel('Count')
        ax.set_title('Location residual', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Azimuthal gap ---
        ax = axes[1, 0]
        ax.hist(df_cat['azimuthal_gap'].dropna(), bins=40, color='seagreen',
                edgecolor='white')
        ax.axvline(df_cat['azimuthal_gap'].median(), color='red', ls='--', lw=1.5,
                   label=f"median = {df_cat['azimuthal_gap'].median():.0f}°")
        ax.set_xlabel('Azimuthal gap (deg)')
        ax.set_ylabel('Count')
        ax.set_title('Azimuthal coverage', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Horizontal and vertical uncertainty ---
        ax = axes[1, 1]
        for col, label, colour in [('err_h_km', 'Horizontal', 'salmon'),
                                   ('err_z_km', 'Vertical', 'lightgreen')]:
            if col in df_cat.columns:
                v = df_cat[col].replace([np.inf, -np.inf], np.nan).dropna()
                if not v.empty:
                    ax.hist(v, bins=40, alpha=0.65, color=colour,
                            edgecolor='white',
                            label=f"{label} (median {v.median():.2f} km)")
        ax.set_xlabel('Location uncertainty (km)')
        ax.set_ylabel('Count')
        ax.set_title('Location uncertainty', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Phase residuals ---
        ax = axes[1, 2]
        if not df_pha.empty:
            for phase, colour in [('P', 'steelblue'), ('S', 'crimson')]:
                res = df_pha[df_pha['phase'] == phase]['residual_s'].dropna()
                if not res.empty:
                    ax.hist(res, bins=50, alpha=0.6, color=colour,
                            label=f"{phase} (n={len(res):,}, "
                                  f"median {res.median():+.3f} s)")
            ax.axvline(0, color='black', lw=1.5)
            ax.legend(fontsize=8)
        ax.set_xlabel('Travel-time residual (s)')
        ax.set_ylabel('Count')
        ax.set_title('Phase residuals', fontweight='bold')
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"NonLinLoc hypocentre location -- {len(df_cat):,} events located",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(folders['summary'],
                                 f"location_summary.{cfg['plot_format']}"),
                    dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        plt.close('all')
        print(f"   ⚠️  Location summary plot failed: {e}")


def plot_hypocenter_distribution_map(df_cat, stations, cfg, folders):
    """
    Coastline map of the located catalogue, rendered with PyGMT.

    Style follows the reference map from the earlier jatim_aus project
    (basemap, coastline, station triangles, depth-coloured event circles,
    scale bar and compass rose), minus the relief shading -- that needs
    pygmt.datasets.load_earth_relief() to reach GMT's remote tile servers,
    which are unreachable from this network, so a plain land/water fill is
    used instead. Unlike that reference, there is no magnitude catalogue or
    fault/well overlay for this network, so event circles use a uniform
    size and depth is encoded by colour alone, and the fault/well layers
    are omitted rather than faked.
    """
    if not HAS_PYGMT:
        print("   ⚠️  pygmt is not installed -- distribution_nonlinloc_hypocenter "
              "map skipped (conda install -c conda-forge pygmt)")
        return
    if df_cat.empty:
        return

    try:
        pad = cfg['pygmt_region_padding_deg']
        lon_min = min(df_cat['longitude'].min(), stations['longitude'].min()) - pad
        lon_max = max(df_cat['longitude'].max(), stations['longitude'].max()) + pad
        lat_min = min(df_cat['latitude'].min(), stations['latitude'].min()) - pad
        lat_max = max(df_cat['latitude'].max(), stations['latitude'].max()) + pad
        region = [lon_min, lon_max, lat_min, lat_max]
        projection = f"M{cfg['pygmt_map_width']}"

        event_size_cm = 0.28
        depth_max = float(df_cat['depth_km'].max())

        fig = pygmt.Figure()
        pygmt.config(FONT_TITLE="14p,Helvetica-Bold", MAP_TITLE_OFFSET="6p",
                     FONT_LABEL="9p", FONT_ANNOT_PRIMARY="8p")

        fig.coast(
            region=region, projection=projection,
            land="gray95", water="lightblue", shorelines="0.75p,black",
            frame=["xa", "ya",
                   f"WSne+tDistribution of NonLinLoc Hypocenters ({len(df_cat):,} events)"])

        pygmt.makecpt(cmap="jet", series=[0, depth_max], reverse=True)
        fig.plot(x=stations['longitude'], y=stations['latitude'],
                  style="t0.45c", fill="black", pen="0.5p,white",
                  label="Stations")
        for _, row in stations.iterrows():
            fig.text(x=row['longitude'] + 0.05, y=row['latitude'] + 0.03,
                      text=row['station'], font="7p,Helvetica-Bold,black",
                      justify="LM")

        fig.plot(x=df_cat['longitude'], y=df_cat['latitude'], style="cc",
                  size=[event_size_cm] * len(df_cat), fill=df_cat['depth_km'],
                  cmap=True, pen="0.15p,black", label="Events")

        fig.colorbar(frame=["xa+lDepth (km)"],
                     position="JBC+w10c/0.4c+h+o0c/1.2c")
        fig.basemap(map_scale="jTL+w50k+o0.8c/1.2c+f+lkm")
        fig.basemap(rose="jTR+w1.3c+o0.5c/0.5c+f+l")
        fig.legend(position="JBR+jBR+o0.3c", box="+gwhite+p0.5p")

        out_path = os.path.join(folders['summary'],
                                 f"distribution_nonlinloc_hypocenter.{cfg['plot_format']}")
        fig.savefig(out_path, dpi=cfg['pygmt_dpi'])
        print(f"   ✅ {out_path}")
    except Exception as e:
        print(f"   ⚠️  PyGMT hypocentre distribution map failed: {e}")


def plot_cross_sections(df_cat, stations, cfg, folders):
    """Depth sections along longitude and latitude, plus a depth-time view."""
    if df_cat.empty:
        return
    try:
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

        ax = axes[0]
        ax.scatter(df_cat['longitude'], df_cat['depth_km'], s=18, alpha=0.7,
                   c='steelblue', edgecolors='black', linewidths=0.3)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Depth (km)')
        ax.set_title('E-W cross-section', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.scatter(df_cat['latitude'], df_cat['depth_km'], s=18, alpha=0.7,
                   c='seagreen', edgecolors='black', linewidths=0.3)
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Depth (km)')
        ax.set_title('N-S cross-section', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        t = pd.to_datetime(df_cat['origin_time'])
        ax.scatter(t, df_cat['depth_km'], s=18, alpha=0.7, c='darkorange',
                   edgecolors='black', linewidths=0.3)
        ax.set_xlabel('Origin time')
        ax.set_ylabel('Depth (km)')
        ax.set_title('Depth vs time', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.savefig(os.path.join(folders['summary'],
                                 f"cross_sections.{cfg['plot_format']}"),
                    dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        plt.close('all')
        print(f"   ⚠️  Cross-section plot failed: {e}")


def plot_station_residuals(df_pha, cfg, folders):
    """
    Median residual per station and phase.

    A station whose residuals are consistently offset from zero indicates a
    timing problem, a wrong location in stations.txt, or a local velocity
    structure the 1D model does not capture.
    """
    if df_pha.empty:
        return
    try:
        stats = (df_pha.dropna(subset=['residual_s'])
                 .groupby(['station', 'phase'])['residual_s']
                 .agg(['median', 'std', 'count']).reset_index())
        if stats.empty:
            return

        order = (stats.groupby('station')['count'].sum()
                 .sort_values(ascending=False).index.tolist())
        fig, ax = plt.subplots(figsize=(max(10, len(order) * 0.5), 6))

        width = 0.38
        for i, (phase, colour) in enumerate([('P', 'steelblue'), ('S', 'crimson')]):
            sub = stats[stats['phase'] == phase].set_index('station').reindex(order)
            xs  = np.arange(len(order)) + (i - 0.5) * width
            ax.bar(xs, sub['median'].values, width, color=colour, alpha=0.8,
                   edgecolor='black', linewidth=0.4, label=phase)

        ax.axhline(0, color='black', lw=1.2)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=8)
        ax.set_ylabel('Median travel-time residual (s)')
        ax.set_title('Median residual per station', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(folders['summary'],
                                 f"station_residuals.{cfg['plot_format']}"),
                    dpi=cfg['plot_dpi'], bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        plt.close('all')
        print(f"   ⚠️  Station residual plot failed: {e}")


# ------------------------------------------------------------
# 3.9 Main pipeline
# ------------------------------------------------------------

def main():
    t0 = time.time()
    log_step = make_logger(CONFIG, t0)

    print("=" * 64)
    print("  NonLinLoc Hypocentre Location Pipeline")
    print(f"  Binaries : {CONFIG['nll_bin_dir']}")
    print(f"  Input    : {CONFIG['association_nll_dir']}")
    print("=" * 64)

    folders    = setup_output_folders(CONFIG['output_folder'])
    checkpoint = load_checkpoint(CONFIG['checkpoint_file'])
    print(f"🗂️  Checkpoint: grids built for {sorted(checkpoint['grids_built']) or 'none'}")

    csv_dir      = folders['csv']
    path_catalog = os.path.join(csv_dir, "hypocenter_catalog.csv")
    path_phases  = os.path.join(csv_dir, "hypocenter_phases.csv")
    path_hq      = os.path.join(csv_dir, "hypocenter_catalog_highquality.csv")

    # ========================================================
    # STAGE 1 -- Stations and grid geometry
    # ========================================================
    print("\n── STAGE 1: Stations & grid geometry " + "─" * 26)
    stations = load_stations_dataframe(CONFIG['stations_file'], CONFIG['network'])
    vol      = load_search_volume(CONFIG)
    geom     = derive_2d_grid(stations, vol, CONFIG)

    # ========================================================
    # STAGE 2 -- Observation files
    # ========================================================
    print("\n── STAGE 2: Observation files " + "─" * 33)
    n_events = collect_pick_files(CONFIG, folders)

    # ========================================================
    # STAGE 3 -- Control files
    # ========================================================
    print("\n── STAGE 3: Control files " + "─" * 37)
    control_paths = write_control_files(stations, geom, vol, CONFIG, folders)

    # ========================================================
    # STAGE 4 -- Velocity and travel-time grids
    # ========================================================
    print("\n── STAGE 4: Velocity & travel-time grids " + "─" * 22)
    log_step("building grids (Grid2Time is usually the slowest stage)...")
    build_grids(control_paths, folders, CONFIG, stations, checkpoint)
    log_step("grids ready")

    # ========================================================
    # STAGE 5 -- Location
    # ========================================================
    print("\n── STAGE 5: Hypocentre location " + "─" * 31)
    log_step(f"locating {n_events:,} events with NLLoc...")
    run_nlloc(control_paths, folders, CONFIG, checkpoint)
    log_step("NLLoc finished")

    # ========================================================
    # STAGE 6 -- Parse results
    # ========================================================
    print("\n── STAGE 6: Parse results " + "─" * 37)
    df_cat, df_pha = parse_locations(folders, CONFIG)

    if df_cat.empty:
        print("\n❌ No events were located. Check the NLLoc output above.")
        return

    # ========================================================
    # STAGE 7 -- Quality control
    # ========================================================
    thresholds = {}
    if CONFIG['run_qc']:
        print("\n── STAGE 7: Quality control " + "─" * 35)
        df_cat, thresholds, _ = apply_location_qc(df_cat, CONFIG)
        df_hq = df_cat[df_cat['qc_pass']].drop(
            columns=['qc_pass', 'qc_reject_reason'])
        df_hq.to_csv(path_hq, index=False)
    else:
        print("\n   run_qc = False -> every located event is kept unfiltered")
        df_cat['qc_pass'] = True
        df_hq = df_cat

    df_cat.to_csv(path_catalog, index=False)
    df_pha.to_csv(path_phases, index=False)
    print(f"   ✅ {path_catalog} : {len(df_cat):,} events")
    print(f"   ✅ {path_phases} : {len(df_pha):,} arrivals")
    if CONFIG['run_qc']:
        print(f"   ✅ {path_hq} : {len(df_hq):,} high-quality events")

    # ========================================================
    # STAGE 8 -- Summary plots
    # ========================================================
    if CONFIG['plot_summary']:
        print("\n── STAGE 8: Summary plots " + "─" * 37)
        plot_qc_distributions(df_cat, thresholds, CONFIG, folders)
        plot_qc_maps(df_hq if len(df_hq) else df_cat, stations, vol,
                     CONFIG, folders)
        plot_location_summary(df_cat, df_pha, stations, vol, CONFIG, folders)
        plot_cross_sections(df_hq if len(df_hq) else df_cat, stations,
                            CONFIG, folders)
        plot_station_residuals(df_pha, CONFIG, folders)
        if CONFIG['plot_pygmt_map']:
            plot_hypocenter_distribution_map(
                df_hq if len(df_hq) else df_cat, stations, CONFIG, folders)
        print(f"   ✅ Summary plots written to {folders['summary']}")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================
    print("\n" + "=" * 64)
    print("  FINAL SUMMARY")
    print("=" * 64)
    print(f"  Events submitted         : {n_events:,}")
    print(f"  Events located           : {len(df_cat):,}")
    print(f"  Arrivals used            : {len(df_pha):,}")
    if CONFIG['run_qc']:
        print(f"  High quality (all QC)    : {len(df_hq):,} "
              f"({100.0 * len(df_hq) / max(len(df_cat), 1):.1f}%)")
    print("  " + "-" * 56)
    for col, label, unit in [('rms_s', 'RMS residual', 's'),
                             ('err_h_km', 'Horizontal error', 'km'),
                             ('err_z_km', 'Vertical error', 'km'),
                             ('azimuthal_gap', 'Azimuthal gap', 'deg'),
                             ('depth_km', 'Depth', 'km')]:
        if col in df_cat.columns and df_cat[col].notna().any():
            print(f"  Median {label:<18}: {df_cat[col].median():8.3f} {unit}")
    print("  " + "-" * 56)
    print(f"  Catalogue                : {path_catalog}")
    print(f"  Total runtime            : {time.time() - t0:.1f} s")
    print("\n✅ PIPELINE COMPLETE!")


if __name__ == "__main__":
    main()
