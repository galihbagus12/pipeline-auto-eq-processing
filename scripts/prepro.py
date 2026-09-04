#!/usr/bin/env python3
# ============================================================
# Raw Waveform Preprocessing Pipeline
#
# Stage 0 of the workflow (runs before pick.py):
#   0) prepro.py        -> merge channels, detrend/demean, resample, bandpass (this file)
#   1) pick.py          -> phase picking (EQTransformer / SeisBench)
#   2) association.py   -> phase association + quality control
#   3) locator.py        -> hypocentre location (NonLinLoc)
#   4) magnitud.py       -> local magnitude
#
# Ported from a reference script (prepro.py) that produced the pipeline's
# existing preprocessed data (this project's own preprocessing/output/
# still carries that run's checkpoint_merge_filter.json -- see below).
#
# Per-group pipeline (station + date, run in PARALLEL across groups):
#   1. MERGE      -- read every raw channel file for the group into one
#                     Stream (0-byte / corrupt files are skipped, not fatal)
#   2. DETREND / DEMEAN (pre-resample) -- remove linear trend and DC offset
#   3. ANTI-ALIAS LOWPASS -- low-pass below the target Nyquist before
#                     resampling, otherwise high-frequency content aliases
#                     down into the passband during resample()
#   4. RESAMPLE   -- source_sampling_rate -> target_sampling_rate. The
#                     ratio here (250 -> 100 Hz) is not an integer, so this
#                     uses ObsPy's Fourier-based resample() rather than
#                     decimate() (which only supports integer factors)
#   5. DETREND / DEMEAN (post-resample) + TAPER + BANDPASS -- resampling can
#                     reintroduce a small DC/trend shift, so this repeats
#                     step 2 before the bandpass filter; the taper avoids
#                     edge transients from the filter
#   6. SAVE       -- cast to int32 + STEIM2 encoding (keeps files small;
#                     the filter output is float64 in memory but the counts
#                     values fit safely back into int32) and write to
#                     output_folder/<date>/<NETWORK>.<STATION>..<date>.mseed
#                     -- the same naming convention pick.py expects.
#
# Input  : raw_root_dir/<YYYY-MM-DD>/<YYYY-MM-DD>_<STATION>_<CHANNEL>.mseed
#          (AusPass Jatim raw export: one file per channel per station-day)
# Output : output_folder/<YYYY-MM-DD>/<NETWORK>.<STATION>..<YYYY-MM-DD>.mseed
#          output_folder/checkpoint_merge_filter.json
#
# NOTE: This file is deliberately split into:
#   1) MODULE IMPORTS
#   2) CONFIGURATION (loaded from config/config.yaml -- see CONFIG_YAML_PATH)
#   3) PROCESS (discovery, per-group worker, checkpointing, main pipeline)
# CONFIG is loaded from the 'preprocessing' section of the shared
# config/config.yaml (see pipeline.py at the project root, which runs every
# stage from that same file) -- edit the YAML to change parameters, not this
# file; the CONFIG dict keys and every function signature below are
# unchanged, so PROCESS never needs to be touched for a config-only change.
# ============================================================

# ============================================================
# 1) MODULE IMPORTS
# ============================================================

import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import yaml
from obspy import Stream, read

try:
    from tqdm import tqdm
except ImportError:
    raise ImportError("Package 'tqdm' is not installed. Run: pip install tqdm")


# ============================================================
# 2) CONFIGURATION
# ============================================================

CONFIG_YAML_PATH = "/media/galih/MyBackUp 2024/Jatim_new/config/config.yaml"

with open(CONFIG_YAML_PATH, 'r') as _f:
    CONFIG = yaml.safe_load(_f)['preprocessing']

# ============================================================
# 3) PROCESS
# ============================================================

# ------------------------------------------------------------
# 3.1 Per-group worker
#
# Must stay at module top-level (not nested) so ProcessPoolExecutor can
# pickle it. cfg is passed in explicitly rather than read from a module
# global, since each worker runs in its own process.
# ------------------------------------------------------------

def process_group(task, cfg):
    date_str = task['date_str']
    station = task['station']
    file_list = task['file_list']
    output_file = task['output_file']

    group_key = f"{station}_{date_str}"
    skipped_files = []

    try:
        # 1. Merge every raw file in the group into one Stream, skipping
        #    0-byte or unreadable files instead of failing the whole group.
        st = Stream()
        for fpath in sorted(file_list):
            fname = os.path.basename(fpath)
            try:
                if os.path.getsize(fpath) == 0:
                    skipped_files.append(fname)
                    continue
            except OSError:
                skipped_files.append(fname)
                continue

            try:
                st += read(fpath)
            except Exception:
                skipped_files.append(fname)
                continue

        if len(st) == 0:
            return {
                'group_key': group_key, 'status': 'error',
                'message': 'Empty stream -- every file in this group was unreadable/corrupt.',
                'skipped_files': skipped_files,
            }

        # 2. Detrend/demean before resampling.
        if cfg['apply_detrend']:
            st.detrend('linear')
        if cfg['apply_demean']:
            st.detrend('demean')

        # 3. Anti-alias lowpass, then merge any gaps/overlaps within the group.
        if cfg['apply_antialias_lowpass']:
            st.filter('lowpass', freq=cfg['antialias_lowpass_freq'])
        st.merge(fill_value='interpolate')

        # 4. Resample to the target rate.
        if cfg['apply_resample']:
            rates = {tr.stats.sampling_rate for tr in st}
            if rates != {cfg['target_sampling_rate']}:
                st.resample(cfg['target_sampling_rate'])

        # 5. Detrend/demean again, taper, bandpass.
        if cfg['apply_demean']:
            st.detrend('demean')
        if cfg['apply_detrend']:
            st.detrend('linear')
        if cfg['apply_taper']:
            st.taper(max_percentage=cfg['taper_max_percentage'], type=cfg['taper_type'])
        if cfg['apply_bandpass']:
            st.filter(
                'bandpass',
                freqmin=cfg['bandpass_freqmin'], freqmax=cfg['bandpass_freqmax'],
                corners=cfg['bandpass_corners'], zerophase=cfg['bandpass_zerophase'],
            )

        # 6. Cast back to int32 for compact STEIM2-encoded output.
        for tr in st:
            tr.data = np.round(tr.data).astype(np.int32)
            tr.stats.mseed = {
                'encoding': cfg['output_encoding'],
                'record_length': cfg['output_record_length'],
            }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        st.write(output_file, format='MSEED', reclen=cfg['output_record_length'])

        return {'group_key': group_key, 'status': 'done', 'message': None, 'skipped_files': skipped_files}

    except Exception as e:
        return {'group_key': group_key, 'status': 'error', 'message': str(e), 'skipped_files': skipped_files}


# ------------------------------------------------------------
# 3.2 Checkpointing
# ------------------------------------------------------------

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded checkpoint: {len(data)} entries.")
        return data
    print('No checkpoint found. Starting fresh.')
    return {}


def save_checkpoint(checkpoint_file, checkpoint):
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


# ------------------------------------------------------------
# 3.3 Task discovery
# ------------------------------------------------------------

def discover_date_folders(cfg):
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    all_dates = sorted(
        d for d in os.listdir(cfg['raw_root_dir'])
        if os.path.isdir(os.path.join(cfg['raw_root_dir'], d)) and date_pattern.match(d)
    )
    if not all_dates:
        raise FileNotFoundError(f"No date folders found under {cfg['raw_root_dir']}")

    start_date = cfg['start_date'] or all_dates[0]
    end_date = cfg['end_date'] or all_dates[-1]
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    selected = [d for d in all_dates if start_dt <= datetime.strptime(d, '%Y-%m-%d') <= end_dt]

    print(f"Date folders found  : {len(all_dates)}")
    print(f"Date folders in range: {len(selected)} ({start_date} to {end_date})")
    if not selected:
        raise FileNotFoundError(f"No date folders between {start_date} and {end_date} under {cfg['raw_root_dir']}")
    return selected


def build_task_list(cfg, checkpoint):
    selected_dates = discover_date_folders(cfg)

    if cfg['merge_channels']:
        # <date>_<STATION>_<CHANNEL>.mseed -- group by (date, station), merging channels.
        file_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})_([A-Za-z0-9]+)_([A-Za-z0-9]+)\.mseed$', re.IGNORECASE)
    else:
        # <date>_<STATION>.mseed -- already merged, one file per (date, station).
        file_pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})_([A-Za-z0-9]+)\.mseed$', re.IGNORECASE)

    groups = {}
    for date_str in selected_dates:
        date_dir = os.path.join(cfg['raw_root_dir'], date_str)
        for fpath in sorted(glob.glob(os.path.join(date_dir, '*.mseed'))):
            m = file_pattern.match(os.path.basename(fpath))
            if not m:
                continue
            _, station = m.groups()[0], m.groups()[1]
            groups.setdefault((date_str, station), []).append(fpath)

    print(f"Groups found (station x date): {len(groups)}")

    tasks = []
    skipped_done = 0
    for (date_str, station), file_list in groups.items():
        group_key = f"{station}_{date_str}"
        if checkpoint.get(group_key) == 'done':
            skipped_done += 1
            continue

        output_file = os.path.join(
            cfg['output_folder'], date_str, f"{cfg['network']}.{station}..{date_str}.mseed"
        )
        tasks.append({'date_str': date_str, 'station': station, 'file_list': file_list, 'output_file': output_file})

    print(f"Groups already done (skipped): {skipped_done}")
    print(f"Groups to process now        : {len(tasks)}\n")

    if cfg['debug_max_groups'] is not None:
        tasks = tasks[: cfg['debug_max_groups']]
        print(f"debug_max_groups set -> processing only {len(tasks)} group(s)\n")

    return tasks


# ------------------------------------------------------------
# 3.4 Main pipeline
# ------------------------------------------------------------

def main(cfg):
    os.makedirs(cfg['output_folder'], exist_ok=True)

    checkpoint = load_checkpoint(cfg['checkpoint_file'])
    tasks = build_task_list(cfg, checkpoint)

    if not tasks:
        print('No new groups to process. Everything is already done.')
        return

    max_workers = cfg['max_workers'] or os.cpu_count()
    print(f"Running in parallel with {max_workers} workers...\n")

    n_done_since_save = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_group, task, cfg): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing', unit='group'):
            result = future.result()
            group_key = result['group_key']

            if result['status'] == 'done':
                checkpoint[group_key] = 'done'
            else:
                checkpoint[group_key] = f"error: {result['message']}"
                tqdm.write(f"Error processing {group_key}: {result['message']}")

            if result['skipped_files']:
                checkpoint[f'{group_key}__skipped_files'] = result['skipped_files']
                tqdm.write(f"Skipped files in {group_key}: {result['skipped_files']}")

            n_done_since_save += 1
            if n_done_since_save >= cfg['checkpoint_save_every']:
                save_checkpoint(cfg['checkpoint_file'], checkpoint)
                n_done_since_save = 0

    save_checkpoint(cfg['checkpoint_file'], checkpoint)

    print('\n====================================')
    print('Preprocessing finished.')
    print(f"Output saved to   : {cfg['output_folder']}")
    print(f"Checkpoint        : {cfg['checkpoint_file']}")
    print('====================================')


if __name__ == '__main__':
    main(CONFIG)
