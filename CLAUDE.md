# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**End-to-End Automated Seismic Processing Pipeline for Earthquake Detection, Relocation, and
Magnitude Estimation** (see `pipeline.py`'s and `config/config.yaml`'s `pipeline_name`) — a seismology
data-processing project for the East Java ("Jatim") seismic network (network code `3M`,
27 stations `EJA12`...`EJA4x`, see `stations.txt`). It is not a software package — there is no
`git`, no test suite, no build system, and no dependency manifest (`requirements.txt` / `environment.yml`
do not exist). It is a two-stage batch pipeline over ~months of continuous raw waveform data:

All pipeline entry-point scripts live in `scripts/`; every stage writes to its own subfolder under
the top-level `output/` (e.g. `output/picking/`, `output/association/`) — see "Architecture /
pipeline flow" below for the full layout.

1. **Picking** (`scripts/pick.py`, and its notebook precursor `picking.ipynb`) — runs the EQTransformer
   model (via SeisBench) over daily/per-station miniSEED waveforms to detect P/S phase arrivals.
2. **Association** (`scripts/association.py`; notebook precursors in `association/*.ipynb`) — takes the
   combined picks CSV and uses PyOcto to associate picks into discrete earthquake events, applies
   custom quality-control rules, and renders per-event diagnostic plots.
3. **Location** (`scripts/locator.py`) — drives the NonLinLoc binaries
   (Vel2Grid → Grid2Time → NLLoc) over the per-event `.pick` files, parses the `.hyp` output into
   a catalogue, and grades each solution.

Runs take hours to days over the full dataset, so both stages are checkpoint/resume-based rather than
one-shot scripts.

## Environment / running

`requirements.txt` (repo root) is the canonical dependency manifest — one pinned list, install into a
project-local `.venv/` (see `README.md`'s Install section). It supersedes the two conda environments
below, which were validated first and are kept as this development machine's own working setup, not
because the two dependency stacks actually conflict — pyocto is already installed alongside
seisbench/torch in the `obspy` env below with no issue, so the two-env split was conda tooling caution,
not a real incompatibility. Don't reintroduce a second env/venv for a "conflict" without checking first.

| Env | Use for | Has |
| --- | --- | --- |
| `pyocto` | `association.py`, `locator.py`, `relocation.py`, `magnitud.py`, `prepro.py`, `focmech.py` | pyocto 0.1.10, obspy 1.5.0, geopy, pyproj, pandas 2.3.3, scipy, pygmt, pyrocko |
| `obspy` | `pick.py`, ad-hoc waveform work | seisbench 0.11.6/torch 2.5.1 stack, obspy, pandas 2.3.3 |

Run each stage with an explicit interpreter — do **not** use `python3` from `base` (no pandas there):
```
/home/galih/miniconda3/envs/pyocto/bin/python scripts/prepro.py
/home/galih/miniconda3/envs/obspy/bin/python  scripts/pick.py
/home/galih/miniconda3/envs/pyocto/bin/python scripts/association.py
/home/galih/miniconda3/envs/pyocto/bin/python scripts/locator.py
/home/galih/miniconda3/envs/pyocto/bin/python scripts/relocation.py
/home/galih/miniconda3/envs/pyocto/bin/python scripts/magnitud.py
/home/galih/miniconda3/envs/pyocto/bin/python scripts/focmech.py
```

Or run any subset (in order) through the central control script, which launches each stage as a
subprocess with the right interpreter automatically — `.venv/bin/python` for every step if a
project-local `.venv/` exists, else the two conda envs above (see `pipeline.py`'s
`resolve_interpreter()`):
```
python pipeline.py                    # interactive menu
python pipeline.py --list             # list available steps
python pipeline.py prepro picking     # run these steps, in the order given
python pipeline.py --all              # run every step, in pipeline order
```
`pipeline.py` only decides which stage runs with which interpreter (see its `STEP_MAP`/
`ENV_PYTHON`) — it does not read or alter any stage's parameters.

NonLinLoc binaries (`Vel2Grid`, `Grid2Time`, `NLLoc`) are installed at `/home/galih/NLLoc/src` and
are already on `PATH`.

Neither script has CLI flags. All behavior is controlled by each stage's own section of the shared
`config/config.yaml` — edit that file to change any parameter, no script needs to be touched.
Each script loads its section (by the same key as its stage name, e.g. `association.py` reads
`config.yaml`'s `association:` block) at import time via `CONFIG_YAML_PATH` near the top of its
`2) CONFIGURATION` section; the `CONFIG` dict keys and every function signature below that are
unchanged from before the migration, so `PROSES`/`PROCESS` never needs touching for a config-only
change (see the "NOTE" header comment in each script). The one exception: `locator.py`'s
`CONFIG['velocity_model']` and `relocation.py`'s `CONFIG['layer_top_km']`/`CONFIG['layer_vp_km_s']`
are parsed at load time from `config/vel_model/jatim_velocity_model.txt` (NLLoc LAYER format) rather
than stored directly in the YAML, so both stages share one canonical velocity model file instead of
two independent copies of the same numbers.

**Hardcoded absolute paths**: every script in `scripts/` hardcodes `/media/galih/MyBackUp 2024/Jatim_new/...`.
The `association/*.ipynb` notebooks additionally carry *stale* paths from an earlier machine
(`/media/galih/New Volume/...`, `E:/TUGAS PT 2/...`) — they are superseded by `scripts/association.py`
and should not be run as-is.

## Architecture / pipeline flow

All pipeline scripts live in `scripts/`; every stage's artifacts live under its own subfolder of the
top-level `output/` (named after the stage, not the script filename — e.g. `scripts/pick.py` writes
to `output/picking/`, `scripts/locator.py` writes to `output/hypocenter_locator/`).

```
AusPass_Jatim/<YYYY-MM-DD>/<YYYY-MM-DD>_<STA>_<CHANNEL>.mseed   (raw input, one file per channel)
        │
        ▼  scripts/prepro.py  (merge channels → detrend/demean → resample → bandpass)
output/preprocessing/<YYYY-MM-DD>/<NET>.<STA>..<YYYY-MM-DD>.mseed   (one folder per day)
        │
        ▼  scripts/pick.py  (EQTransformer via SeisBench)
output/picking/<YYYY-MM-DD>/<NET>.<STA>/csv/results_<NET>.<STA>_<date>.csv   (per station-day picks)
output/picking/<YYYY-MM-DD>/<NET>.<STA>/plots/{overview,events,summary}/     (diagnostic PNGs)
output/picking/summary/ALL_results_combined.csv                              (all station-days appended together)
        │
        ▼  scripts/association.py  (PyOcto associate → QC-0..QC-3 → per-event plots)
output/association/csv/     pyocto_{picks,events,assignments,catalog}.csv   (pre-QC)
                            pyoctorev_{assignments,catalog}.csv             (post-QC)
                            qc_event_report.csv                             (per-event QC metrics)
output/association/events/<EVENT_ID>/     pseudo_distance / wadati / waveform PNG + picks CSV
output/association/rejected/<EVENT_ID>/   same, for events dropped by QC
output/association/summary/               qc2_ratio_distribution.png, catalog_summary.png
```

Current picking output covers **2015-12-01 → 2016-05-04** (156 days, 66,317 detection rows,
23 of 27 stations; `EJA51`–`EJA54` never appear). Picking thresholds were deliberately permissive
(`P_threshold`/`S_threshold` = 0.01), so median `p_probability` is only ~0.076. Every row has a
complete P+S pair (`require_p_and_s`), so association ingests 132,634 picks.

**`velocity_tolerance` is the single most consequential parameter** — it is the time budget allowed
between observed and model-predicted arrivals. Picks outside it are ignored by the associator, which
can push an event below `n_p_and_s_picks` and discard it entirely even though it is real. The yield
curve is humped, measured on a 5-day subset (7,314 picks):

| tol | events | runtime | | tol | events | runtime |
| --- | --- | --- | --- | --- | --- | --- |
| 1.98 s | 1 | 0.7 s | | 5.0 s | 2 | 16.4 s |
| 3.0 s | 2 | 2.5 s | | 6.0 s | 2 | 35.4 s |
| 4.0 s | 2 | 7.1 s | | 8.0 s | **1** | 114.5 s |

Too tight → distant stations are dropped (they refract through the mantle at `Vp≈8.04` while the model
assumes 6.0, so their residuals exceed the budget) and the event fails the 3-pair requirement. Too
loose → noise picks look compatible with any trial location, real events lose out to spurious
high-count nodes, and runtime explodes. The curve is humped, so "looser is always better" is false.
Healthy range is **3–6 s**; the auto rule deliberately sits at the top of it (10% effective velocity
error → 5.92 s) because the user's policy is to let the waveform plots, not the associator, be the
final filter. `velocity_tolerance_max` is pinned at 6.0 — the largest value empirically verified not
to reduce yield.

Derived quantities are computed **after** NLLoc grid clipping, so `R_MAX`, `time_before` and
`velocity_tolerance` all reflect the volume actually searched. (An earlier ordering bug derived them
from the pre-clip volume, making both looser than intended.)

**Validated 30-day run** (2015-12-01 → 2015-12-30, 20,871 rows → 41,742 picks): 24 events associated,
14 dropped by QC, **10 passing**; ~3 s per daily batch. Extrapolating: ~2 h association + plotting for
the full 156 days. Note that associator depths cluster on the octree boundaries (three events at
exactly 94.26 km, the grid limit; two at 5.20 km) — depth is essentially unresolved with 3 stations and
a ~280° azimuthal gap. That is expected: PyOcto is a coarse associator, and proper hypocentres come
from NonLinLoc using the generated `.pick` files.

**Caveat on the "missing events" diagnostic.** Grouping picks by the per-station origin time from QC-4
finds 8 candidate clusters (≥3 stations) in 5 days, while the associator yields 2. This does *not*
mean 6 events are being missed: the OT test only checks time agreement — a necessary but not
sufficient condition — whereas PyOcto additionally requires a consistent (x, y, z). Some clusters are
almost certainly coincidental noise. The true count lies between 2 and 8, and the per-event waveform
plot is the intended arbiter.

Keeping the homogeneous `Vp=6.0, Vs=Vp/1.73` model is a deliberate user decision. `pyocto.VelocityModel1D`
could use the layered `LAYER` lines from `jatim.in` instead, but it needs `pyrocko`, which is not
installed — do not switch to it without asking.

### Stage 1 — Picking (`scripts/pick.py`)

- **Input convention**: `input_folder/<YYYY-MM-DD>/<NET>.<STA>..<YYYY-MM-DD>.mseed`, parsed by
  `parse_mseed_filename()`.
- **Station metadata**: `stations.txt` (4 whitespace-delimited columns: station, latitude, longitude,
  elevation_m) is parsed into `{STA: {network, latitude, longitude, elevation_m}}`, with the network
  code injected from config.yaml's top-level `network` key (stations.txt has no per-station network
  column -- every station on this array shares one code). A station missing from this file is skipped
  (and checkpointed as done) — `build_station_info()` returns `None` in that case.
- **Resume/checkpointing**: `output/checkpoint.json` stores a set of `"<day>/<filename>"` keys
  already processed. Every file is checkpointed the moment it finishes — including deliberate skips
  (missing components, unknown station, excessive gaps) — so reruns only redo files that errored or
  were never attempted. Errors are intentionally *not* checkpointed, so they're retried on the next run.
- **Gap guard**: `check_excessive_gaps()` runs before `st.merge(fill_value='interpolate')` and skips
  (not merges) any file with a gap larger than `max_gap_seconds` (default 1h) — this exists specifically
  to prevent ObsPy from attempting a huge array allocation on corrupt timestamps, which otherwise looks
  like a silent hang (0% CPU, swapping).
- **Output CSV schema is STEAD-style**, built by `build_stead_style_csv()`: one row per detection
  (matched to its best P/S pick within the detection window by peak probability), plus extra rows for
  "orphan" P picks that had no matching detection. Columns: `file_name, network, station,
  instrument_type, station_lat/lon/elv, event_start_time, event_end_time, detection_probability,
  detection_uncertainty, p_arrival_time, p_probability, p_uncertainty, p_snr, s_arrival_time,
  s_probability, s_uncertainty, s_snr`.
- **Quality filtering** (`apply_quality_filters`): drops detections below `min_peak_value`, and — when
  `require_p_and_s` is `True` (default) — drops any row that isn't a complete P+S pair.
- **Combined CSV** (`output/summary/ALL_results_combined.csv`) is appended to per-file (not
  rewritten), so it stays consistent with the checkpoint across interrupted runs.
- Three plot types per station-day (all optional via `CONFIG['plot_*']`): `overview` (full-day
  3-component waveform + P/S/detection probability traces), `events` (zoomed per-detection, up to
  `max_event_plots`), `summary` (grid of histograms/scatter over the station-day's detections).

### Stage 2 — Association (`scripts/association.py`)

Single entry point merging what were three notebooks (`pyocto.ipynb` → `pyocto_qc_v3.ipynb` →
`visual_pyoctorev.ipynb`). Runs in 8 stages, all parameters in `CONFIG`:

1. **Stations & network geometry** — `derive_network_geometry()` derives *every* geometric parameter
   from `stations.txt` so the pipeline self-adapts to a different network. Only `vp` and `vpvs_ratio`
   are true free parameters. Any CONFIG key below set to an explicit number overrides its auto value.

   | Derived | Rule | This network |
   | --- | --- | --- |
   | `APERTURE` | max inter-station distance | 284.35 km (EJA15–EJA32) |
   | `association_cutoff_km` | `ceil(APERTURE × cutoff_factor)` | 342 km |
   | `zlim_km` | `clip(zmax_factor × APERTURE, 30, 200)` | 0–200 km |
   | `area_padding_deg` | `area_padding_factor × APERTURE`, km→deg with cos(lat) | 0.384° / 0.387° |
   | `R_MAX` | `sqrt(cutoff² + z_max²)`, **after** grid clipping | 354.9 km |
   | `time_before` | `R_MAX / Vs × time_before_factor` | 160 s |
   | `velocity_tolerance` | `clip(0.10 × R_MAX/Vp, 1, 6)` | 5.92 s |
   | `max_ts_tp_sec` (QC-1) | `reference / K_PSEUDO`, `K_PSEUDO = Vp·Vs/(Vp−Vs)` | 34.60 s |

   `max_ts_tp_reference` picks the QC-1 reference distance: `'aperture'` (default, stricter — event must
   fit inside the network) or `'search_volume'` (uses `R_MAX`). The auto values reproduce the original
   notebook's hand-tuned config almost exactly (342/200/2.0/35), which is what validates the rules.

   **Downstream grid clipping** (`limit_to_nll_grid`, default on): the search volume is then clipped to
   the NonLinLoc `LOCGRID` in `nonlinlocooo/input/jatim.in` (`TRANS SIMPLE -7.4731 112.5883`;
   X −190.7…190.3, Y −152.9…152.6, Z −5…95 km, using NLLoc's own `x=(lon−lon0)·111.111·cos(lat)`
   convention). Without this, association would produce events outside the location grid that NLLoc
   can never locate — the unclipped auto volume overruns it on lat_max, lon_max and especially depth
   (200 km vs the grid's 95 km). Update these CONFIG values if `LOCGRID` changes.
2. **Association** — **no probability filtering** (deliberate: picking ran with SeisBench `stead` at
   threshold 0.01, so EQTransformer probability carries no gempa/non-gempa information. Verified
   empirically — raising a probability cut from 0.05 to 0.30 removes 5× the picks but leaves the
   associated event count *unchanged*). Only optional `min_p_snr`/`min_s_snr` remain, off by default.
   Real filtering is layered: associator → QC-0…QC-4 → **visual check of the per-event waveform plot,
   which is the final arbiter of whether an event is real**. Picks are associated **one UTC day at a
   time** (events last seconds to minutes, so a day boundary never splits one, and the date doubles as
   the checkpoint unit). `batch_mode: 'day'` (default) splits at UTC day boundaries so batch edges
   never cut through an event; `'count'` reproduces the old notebook's index-based batching.
   `transform_events()` projects local x/y km back to lat/lon (replaces the notebooks' manual pyproj UTM
   math).

   **`checkpoint.json` is the single source of truth for progress** — there is no separate resume flag.
   It holds two lists: `associated_days` (dates whose association finished) and `plotted_events`.
   Each day's raw output is appended to `pyocto_events.csv` / `pyocto_assignments.csv` and the date is
   recorded immediately, so an interrupted run resumes at the next day. **Delete the checkpoint (or run
   with none) and everything restarts from zero**: the stale association CSVs are removed first, since
   keeping them would double-count. Two consistency guards make this crash-safe: every appended row
   carries an `assoc_day` column, and on startup any row whose day is absent from `associated_days` is
   dropped — covering the window where the process dies after the CSV append but before the checkpoint
   save. Event numbering continues across runs from `max(idx) + 1` in the existing CSV. A day that
   errors is left unrecorded and retried next run.
3. **QC-0** — (d) `drop_single_phase_stations()` removes every pick from a station contributing only
   one phase to an event. **Mandatory, not configurable, and applied even when `run_qc0` is off** —
   such stations cannot enter the Wadati / pseudo-distance / QC-4 fits (all pair-based) and clutter the
   waveform plots. Never fabricate the missing phase from the hypocentral distance: its value derives
   from the very location being solved for, and feeding it to NonLinLoc makes the location confirm
   itself. (a) dedupe double picks within `double_pick_window_sec` keeping highest probability;
   (b) drop events lacking both a P and an S pick; (c) re-enforce `min_ps_pairs` (default 3) distinct
   stations with a *complete* P+S pair. Step (c) matters because the associator's `n_p_and_s_picks`
   guarantee can be broken by (a) and (d).
4. **Per-event table** — one row per (event, station) with `tp`, `ts`, `delta_ts_tp`, `epi_dist_km`,
   absolute pick datetimes (`p_dt`/`s_dt`, needed for waveform plots), and **both** distance
   definitions: `dist_ts_tp_km` and `dist_hypo_km`.
5. **QC-1/2/3/4** — individually toggleable via `run_qc0`…`run_qc4`. QC-3 metrics are computed even
   when QC-3 is off, because the per-event plots annotate them.
   - **QC-1**: `Ts−Tp` outside `[min_ts_tp_sec, max_ts_tp_sec]` (upper bound auto-derived).
   - **QC-2** (`qc2_method: 'physical'`): triangle inequality in km with pick-error tolerance —
     violation when `|d_A − d_B| − D_AB > qc2_n_sigma × σ`, `σ = K_PSEUDO × 2 × qc2_pick_sigma_sec`
     (≈4.9 km here, tolerance ≈14.8 km). The old `'percentile'` mode is still selectable but is a
     statistical outlier detector, not a physics test: it discards ~5% of events by construction and
     disproportionately penalises well-recorded events (more stations → more pairs → higher chance one
     exceeds the percentile).
   - **QC-3**: **slope only** — `slope ≤ 0` on either fit is genuinely impossible (arrival time
     decreasing with distance, or Vs > Vp). The intercept criterion is **disabled**
     (`qc3_intercept_tol_sec: None`), as are the optional `qc3_vp_range` / `qc3_vpvs_range` bounds; all
     three can be re-enabled from CONFIG. Reason the intercept test was dropped: `tp` is measured
     relative to PyOcto's origin time, so if that origin is δ seconds late the Wadati intercept shifts
     by roughly `−(Vp/Vs − 1)·δ` — a slightly negative intercept measures PyOcto's origin-time
     precision, not pick quality. The original `intercept < 0 → reject` rule discarded 21 of 24 events
     on a 30-day run, including a 14-station event with `Vp=6.09 km/s`, `Vp/Vs=1.759`, `R²=0.997` and
     textbook moveout — thrown out over a Wadati intercept of −0.57 s (an origin time 0.8 s late).
     With slope-only, QC-3 now rejects 0 of 24 (every slope is comfortably positive).
   - **QC-4**: origin-time consistency. Each station with a P-S pair computes
     `OT_i = Tp_i − (Ts_i − Tp_i)/(Vp/Vs − 1)`. Note Vp cancels — only the **ratio** enters, so this
     test is immune to the crude homogeneous-Vp assumption, unlike QC-2's pseudo distance which
     absorbs the full velocity error. Outliers are found with median + MAD, threshold
     `max(qc4_abs_tol_sec, qc4_n_mad × 1.4826 × MAD)`.

   QC-2 is a *necessary but not sufficient* condition: two genuinely different events at nearby times
   usually still satisfy the triangle inequality (verified — two events 20 s apart at 30 km/41 km from
   stations 162 km apart give an excess of −151 km, passing easily), while QC-4 catches them
   immediately (OT of 0 s vs 20 s). That is why both are kept.

   `qc4_action` controls the response: `'drop_event'` (default) rejects the whole event;
   `'drop_station'` removes only the offending station, then re-runs the `min_ps_pairs` check — so
   the 3-pair minimum can never be violated by station removal (an event with 4 pairs losing one
   survives with 3; an event with exactly 3 losing one is dropped).
6. **Save** — post-QC CSVs plus `qc_event_report.csv` (per-event slope/intercept/R²/Vp-apparent/Vp-Vs,
   status, reject reason).
6b. **Cross-event summary** — `event_summary.csv` (one row per event: origin time, lat/lon/depth,
   `stations_ps` = the stations contributing a complete P+S pair, `n_ps_pairs`, azimuthal gap,
   distance range, fit metrics) and `phase_picks.csv` (one row per pick: absolute `pick_time`,
   `travel_time_s`, probability, distances, azimuth).
6c. **NonLinLoc inputs** — `nonlinloc/picks/<YYYYMMDD-HHMM-SS>.pick` (one file per event, matching the
   existing convention in `nonlinlocooo/input/picks/`: components `ChZ` for P, `ChN` for S, no PRIORWT
   column — drops straight into `LOCFILES ./obs/ja/*.pick NLLOC_OBS`), plus `nonlinloc/all_events.obs`
   as a combined archive (this one does carry PRIORWT = pick probability) and
   `nonlinloc/stations_GTSRCE.txt`. Pick uncertainty from `nll_error_p_sec`/`nll_error_s_sec` — the
   existing hand-made `.pick` files use `GAU 0.0`, i.e. zero uncertainty, which makes NLLoc weight all
   picks equally. Only QC-passing events are written unless `nll_include_rejected` is set.
7. **Per-event plots** — each event gets its own folder containing `pseudo_distance_*.png`,
   `wadati_*.png`, `waveform_*.png`, `picks_*.csv`, tracked by `plotted_events` in the checkpoint.
   Passing events → `events/`, rejected → `rejected/` (`plot_rejected_events: False` to skip the latter).
8. **Summary plots** — `qc2_test_statistic.png` plots whichever quantity actually drives the QC-2
   decision (in `physical` mode the excess `|d_A−d_B| − D_AB` in km against the 0 bound and the
   `n·σ` threshold; percentile lines are *not* drawn, since they control nothing in this mode). Plus a
   catalog overview (depth, n_picks, epicenter map, events-per-day), and four pooled diagrams:
   `wadati_all_events.png`, `wadati_passed_qc.png`, `pseudo_all_events.png`, `pseudo_passed_qc.png` —
   all points in one uniform colour so the two populations compare directly. The pooled fits are the
   strongest independent check that the catalogue is real: on the 30-day run they give
   **Vp = 6.096 km/s (R² = 0.9951)** and **Vp/Vs = 1.755 (R² = 0.9683)**, recovering the assumed 6.0
   and 1.73 (and NLLoc's 1.74) without those values ever being imposed on the fit. Noise associations
   would scatter, not line up.

**Two distance definitions, deliberately used for different things** — this is the one intentional
behavior change from `pyocto_qc_v3.ipynb`, which used the hypocentral distance for both:

- `dist_method_qc2: 'ts_tp'` — QC-2 asks whether the distance *difference* between two stations is
  plausible against their physical separation. That test is only meaningful with pick-derived distances
  (`d = (Vp·Vs/(Vp−Vs))·(Ts−Tp)`). Distances derived from a single hypocenter automatically satisfy the
  triangle inequality, so the notebook's QC-2 could essentially never reject anything.
- `dist_method_plot: 'hypocenter'` — the pseudo-distance plot's x-axis and the QC-3 fit. With `'ts_tp'`
  here, that plot is just the Wadati diagram with rescaled/swapped axes (identical R² — verified), making
  QC-3's two checks redundant. With `'hypocenter'` the plot tests picks against the *location* while
  Wadati tests the Vp/Vs ratio — two independent checks, as QC-3 originally intended.

**Waveform plots** read only the needed time window from `output/preprocessing/<date>/` (via obspy
`read(..., starttime=, endtime=)`), pick the Z channel by `wf_channel_priority`, order stations with the
earliest P at the *bottom* panel, and normalize amplitude using only in-window samples so out-of-window
spikes don't flatten the signal. This stage dominates runtime (~1s per station panel) — set
`plot_waveform: False` or `max_events_plot` when iterating on QC parameters.

### Stage 3 — Location (`scripts/locator.py`)

Drives the three NonLinLoc binaries in sequence, all launched with `cwd` = the output root because
NonLinLoc resolves the relative paths inside a control file against the working directory.

1. **Vel2Grid — run once.** Two `VGTYPE` lines (P and S) in a single control file produce both
   velocity grids in one pass; it does not need running twice.
2. **Grid2Time — run once per phase.** `GTFILES` carries exactly one phase, so P and S require
   separate invocations. `locator.py` writes `run/nlloc_P.in` and `run/nlloc_S.in` for this.
3. **NLLoc** — locates every event matched by the `LOCFILES` glob (`./obs/*.pick`).

**`GRID2D` always.** With a 1D layered model travel time depends only on horizontal distance and
depth, never azimuth, so Grid2Time stores a 2D (distance, depth) grid per station. For this network
that is **~0.25 GB versus ~56 GB** for GRID3D — and the existing hand-run setup at `nlloc_sample/`
used GRID3D, consuming 28 GB for the P grids alone. Under GRID2D, `VGGRID` takes `nx = 2` (per the
official NLLoc sample) while `LOCGRID` stays fully 3D. The 2D grid's extent is derived automatically
from the maximum station-to-LOCGRID-corner distance.

**Grid and projection come from `association.py`, not from CONFIG.** That stage writes
`output/association/nonlinloc/search_volume.json` (projection origin = station centroid, LOCGRID
spanning exactly the associated volume, all derived from station geometry); `locator.py` reads it.
The two stages therefore search the same volume by construction and the handoff needs no manual step.
`grid_override` exists only to force a different grid deliberately.

**The search volume must start above the highest station.** Depth is positive downwards, so a station
at +444 m elevation sits at −0.444 km. A grid starting at z = 0 leaves it outside the travel-time
grid and **every event then fails to locate** (NLLoc reports "N events read, 0 events located" with no
error). `derive_network_geometry` therefore sets `z_min` from `max(elevation_m)` plus
`zmin_margin_km`.

`use_s_grids: True` builds explicit S grids and sets `LOCMETH` VpVs to −1, so S times come from the
layered Vs profile rather than a constant ratio — worth having since Vp/Vs varies with depth in the
model (1.729 near surface, 1.688 at 20 km, 1.795 at 35 km). Setting it False falls back to scaling P
by `loc_vpvs_ratio`.

Grids are cached: `build_grids()` skips Vel2Grid/Grid2Time when a complete set of `.time.hdr` files
exists, unless `rebuild_grids` is set.

**Parsing.** `.hyp` files are read with obspy (`format='NLLOC_HYP'`) into `hypocenter_catalog.csv`
(one row per event: position, RMS, azimuthal gap, phase/station counts, uncertainty, A–D grade) and
`hypocenter_phases.csv` (one row per arrival, with residuals). NLLoc also writes `*.sum.*` (all
events concatenated, from `SAVE_NLLOC_SUM`) and `last.hyp`; both are excluded from the glob or the
first event is counted twice.

**Location QC** (`run_qc`, default on) keeps an event only if it passes all five criteria at once:
`rms_s` below an absolute threshold, and `err_h_km`, `err_z_km`, `azimuthal_gap`, `pdf_volume_km3`
each at or below the **catalogue median**. Four medians make the thresholds adapt to the dataset; the
AND combination means far fewer than half survive. The RMS threshold is instead physical —
`qc_rms_n_samples / sampling_rate` = 4 / 100 Hz = 0.04 s — because a pick cannot beat the sample
interval; the sampling rate is read from the waveform data, not configured. `err_z` is
`sqrt(|CovZZ|)` from the `STATISTICS` line and `pdf_volume` is `scatter_volume` from the `SEARCH`
line, both read by a direct text parse (`parse_hyp_metrics`) since obspy does not expose them.
Rejected events keep a `qc_reject_reason` listing which criteria failed.

`ASSOCIATION_QC.md` is a standalone narrative explanation of this whole stage (the association →
QC → catalogue chain, with the reasoning behind each threshold), written for a non-specialist
audience and intended as source material for a visual explainer. Keep it in sync when QC criteria
change.

## Working with this repo

- Treat `scripts/pick.py` and `scripts/association.py` as the source of truth over the notebooks
  (`picking.ipynb`, `association/*.ipynb`) — the notebooks are earlier iterations of the same logic,
  with stale paths.
- Don't add CLI argument parsing or a config file loader speculatively; the header comments already
  document an intended future move to `config.yaml` — if asked to do that migration, keep the `CONFIG`
  dict keys and the surrounding function signatures unchanged so the "PROSES" sections don't need
  touching.
- `output/` contains large volumes of real data across every stage's subfolder (checkpoint files
  reference thousands of station-day/event entries) — avoid bulk-deleting or rewriting files under
  this tree, and be aware that directory listings there will be large.
- Comments and log strings throughout `pick.py` are in Indonesian; match that convention if extending it.
