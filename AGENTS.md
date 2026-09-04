# AGENTS.md

Seismic processing pipeline (East Java / "Jatim" network). No git, no tests, no CI.
`CLAUDE.md` is the authoritative architecture reference — read it before making changes.

## Running

No CLI flags on any script. All behavior is in `config/config.yaml`.

```bash
python pipeline.py                    # interactive menu
python pipeline.py --list             # list steps
python pipeline.py prepro picking     # run specific steps in order
python pipeline.py --all              # run all steps
```

`pipeline.py` uses `.venv/bin/python` for all stages if `.venv/` exists at project root;
otherwise falls back to two conda envs (`obspy` for picking, `pyocto` for everything else).
No `.venv` currently exists — the conda fallback is the active setup.

Direct execution (conda):
```bash
/home/galih/miniconda3/envs/pyocto/bin/python scripts/prepro.py
/home/galih/miniconda3/envs/obspy/bin/python  scripts/pick.py
/home/galih/miniconda3/envs/pyocto/bin/python scripts/association.py
/home/galih/miniconda3/envs/pyocto/bin/python scripts/locator.py
/home/galih/miniconda3/envs/pyocto/bin/python scripts/relocation.py
/home/galih/miniconda3/envs/pyocto/bin/python scripts/magnitud.py
/home/galih/miniconda3/envs/pyocto/bin/python scripts/focmech.py
```

## Key constraints

- **Hardcoded absolute paths**: every script has `CONFIG_YAML_PATH = "/media/galih/MyBackUp 2024/Jatim_new/config/config.yaml"` near the top. All data paths in `config.yaml` also reference `/media/galih/MyBackUp 2024/...`. These must match the actual mount point.
- **No test suite, no linter, no type checker.** Verification is manual (run the stage, inspect output/plots).
- **Runs take hours to days.** Every stage is checkpoint/resume-based — `checkpoint.json` files under each `output/<stage>/` track progress. Deleting a checkpoint resets that stage.
- **`output/` contains large real data** (thousands of files). Avoid bulk-deleting or listing deep trees.

## Config loading pattern

Each script loads one section of `config/config.yaml` by stage key:

| Script | YAML section key |
|--------|-----------------|
| `prepro.py` | `preprocessing` |
| `pick.py` | `picking` |
| `association.py` | `association` |
| `locator.py` | `hypocenter_locator` |
| `relocation.py` | `relocation` |
| `magnitud.py` | `magnitud` |
| `focmech.py` | `focal_mechanism` |

The `CONFIG` dict keys and function signatures below each script's config section are stable — config-only changes should never require touching the processing logic (`PROSES`/`PROCESS` sections).

Velocity model: `config/vel_model/jatim_velocity_model.txt` (NLLoc LAYER format), parsed at load time by both `locator.py` and `relocation.py` — single source of truth for layer velocities.

Station metadata: `stations.txt` (station, lat, lon, elevation_m). Network code comes from `config.yaml`'s top-level `network` key, not from the file.

## Non-pip dependencies

- **NonLinLoc** binaries (`Vel2Grid`, `Grid2Time`, `NLLoc`) must be on `PATH`
- **relocDD-py** — vendored at `relocation.relocdd_py_dir` in config (default: `output/relocation/relocDD-py/`)
- **GMT C library** — required by `pygmt`, not on PyPI (`sudo apt install gmt` or conda)
- **torch CUDA** — `requirements.txt` installs CPU; for GPU: `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121`

## Pipeline stage output paths

```
output/preprocessing/     → merged/filtered waveforms
output/picking/           → per-station CSVs + summary/ALL_results_combined.csv
output/association/       → associated events, QC reports, NonLinLoc .pick files
output/hypocenter_locator/ → NLLoc grids, .hyp files, hypocenter_catalog.csv
output/relocation/        → relocated_catalog.csv
output/magnitud/          → ml_event_magnitudes.csv
output/focal_mechanism/   → focal_mechanisms.csv, kagan_pairs.csv, plots/
```

## Code conventions

- Comments and log strings in `pick.py` are in **Indonesian** — match that if extending it.
- Notebooks (`picking.ipynb`, `association/*.ipynb`) are superseded by their `scripts/` counterparts and have stale paths — do not run them.
- Association geometry (aperture, search radius, velocity tolerance, QC bounds) is auto-derived from `stations.txt` — do not hand-tune these unless overriding deliberately.
- The homogeneous `Vp=6.0, Vs=Vp/1.73` velocity model in association is a deliberate user choice. Do not switch to a layered `pyocto.VelocityModel1D` (needs `pyrocko`, not installed) without asking.
