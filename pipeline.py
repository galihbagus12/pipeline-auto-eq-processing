#!/usr/bin/env python3
# ============================================================
# End-to-End Automated Seismic Processing Pipeline for Earthquake Detection,
# Relocation, and Magnitude Estimation -- Central Control Script
#
# Runs one or more pipeline stages without having to invoke each stage
# script by hand. Every stage reads its own parameters from the shared
# config/config.yaml (see that file's header) -- this script only decides
# WHICH stages to run and WITH WHICH conda environment, it does not read or
# alter any stage's parameters itself. The display name above/in the
# banners is itself read from config.yaml's 'pipeline_name' key -- rename
# the pipeline there, not here.
#
# Pipeline order:
#   1) prepro       -> scripts/prepro.py       (merge/detrend/resample/bandpass)
#   2) picking      -> scripts/pick.py         (EQTransformer phase picking)
#   3) association   -> scripts/association.py  (PyOcto association + QC)
#   4) locator      -> scripts/locator.py      (NonLinLoc hypocentre location)
#   5) relocation   -> scripts/relocation.py   (hypoDD double-difference relocation)
#   6) magnitud     -> scripts/magnitud.py     (local magnitude + event/magnitude map)
#
# Every step is launched as a SEPARATE PROCESS rather than a naive
# "importlib.import_module() and call main()" dispatcher, so a stage's own
# module-level state/imports never leak into another stage's process.
#
# Interpreter resolution (see resolve_interpreter() below): if a project-local
# ".venv/" exists (single venv covering every stage's dependencies -- see
# requirements.txt and README.md's Install section), every step runs with
# that one interpreter. Otherwise this falls back to two conda environments
# (picking needs SeisBench/torch, installed in 'obspy'; every other stage
# needs pyocto/obspy/pygmt in 'pyocto' -- see ENV_PYTHON), which is how this
# pipeline's own development machine still runs it, from before the
# single-venv setup was validated -- exactly the commands documented in
# CLAUDE.md's "Environment / running" section, just automated instead of
# typed by hand.
#
# Usage:
#   python pipeline.py                    interactive menu
#   python pipeline.py --list             list available steps and exit
#   python pipeline.py prepro picking     run these steps, in the order given
#   python pipeline.py --all              run every step, in pipeline order
#
# NOTE: This file is deliberately split into:
#   1) MODULE IMPORTS
#   2) STEP DEFINITIONS (STEP_MAP, ENV_PYTHON) -> the only things to edit if
#      a stage script moves, is renamed, or a new stage is added
#   3) PROCESS (interpreter resolution, step runner, interactive menu, CLI
#      entry point)
# ============================================================

# ============================================================
# 1) MODULE IMPORTS
# ============================================================

import os
import sys
import subprocess
from collections import OrderedDict

import yaml


# ============================================================
# 2) STEP DEFINITIONS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_YAML_PATH = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')

with open(CONFIG_YAML_PATH, 'r') as _f:
    PIPELINE_NAME = yaml.safe_load(_f)['pipeline_name']

# Project-local venv (see requirements.txt / README.md) -- when present,
# resolve_interpreter() uses this for every step instead of ENV_PYTHON below.
def _get_venv_python():
    candidates = [
        os.path.join(PROJECT_ROOT, '.venv', 'Scripts', 'python.exe'),
        os.path.join(PROJECT_ROOT, '.venv', 'bin', 'python'),
        os.path.join(PROJECT_ROOT, '.venv', 'bin', 'python.exe'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(
        PROJECT_ROOT,
        '.venv',
        'Scripts' if sys.platform == 'win32' else 'bin',
        'python.exe' if sys.platform == 'win32' else 'python'
    )

VENV_PYTHON = _get_venv_python()

# Conda environment -> interpreter, matching CLAUDE.md's "Environment / running"
# table exactly. Fallback used only when VENV_PYTHON doesn't exist. Add an
# entry here before adding a step that needs a new conda env.
if sys.platform == 'win32':
    _user_home = os.path.expanduser('~')
    ENV_PYTHON = {
        'pyocto': os.path.join(_user_home, 'miniconda3', 'envs', 'pyocto', 'python.exe'),
        'obspy' : os.path.join(_user_home, 'miniconda3', 'envs', 'obspy', 'python.exe'),
    }
else:
    ENV_PYTHON = {
        'pyocto': "/home/galih/miniconda3/envs/pyocto/bin/python",
        'obspy' : "/home/galih/miniconda3/envs/obspy/bin/python",
    }

# Ordered so "--all" and the interactive menu both run stages in the
# pipeline's natural sequence. Each script is self-contained (reads its own
# config/config.yaml section, resolves its own absolute I/O paths), so
# running a subset out of order is safe as long as the upstream stage's
# output already exists on disk.
STEP_MAP = OrderedDict([
    ('prepro', {
        'script': 'scripts/prepro.py',
        'env': 'pyocto',
        'description': 'Merge channels, detrend/demean, resample, bandpass filter',
    }),
    ('picking', {
        'script': 'scripts/pick.py',
        'env': 'obspy',
        'description': 'EQTransformer (SeisBench) P/S phase picking',
    }),
    ('association', {
        'script': 'scripts/association.py',
        'env': 'pyocto',
        'description': 'PyOcto phase association + QC-0..QC-4 + per-event plots',
    }),
    ('locator', {
        'script': 'scripts/locator.py',
        'env': 'pyocto',
        'description': 'NonLinLoc hypocentre location (Vel2Grid -> Grid2Time -> NLLoc)',
    }),
    ('relocation', {
        'script': 'scripts/relocation.py',
        'env': 'pyocto',
        'description': 'Double-difference relocation (hypoDD, via relocDD-py)',
    }),
    ('magnitud', {
        'script': 'scripts/magnitud.py',
        'env': 'pyocto',
        'description': 'Local magnitude (Wood-Anderson/ML) + event/magnitude map',
    }),
    ('focmech', {
        'script': 'scripts/focmech.py',
        'env': 'obspy',
        'description': 'Focal mechanism determination (FocoNet) + beachball/cross-section/Kagan plots',
    }),
])


# ============================================================
# 3) PROCESS
# ============================================================

def resolve_interpreter(env_name):
    """VENV_PYTHON if a project-local .venv exists (one interpreter for every
    step), else sys.executable if running inside the target conda env, else
    the conda interpreter for env_name (see ENV_PYTHON)."""
    if VENV_PYTHON and os.path.exists(VENV_PYTHON):
        return VENV_PYTHON

    # 1) If current environment matches the requested conda env
    current_conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if current_conda_env.lower() == env_name.lower():
        return sys.executable
    if os.path.basename(sys.prefix).lower() == env_name.lower():
        return sys.executable

    # 2) Sibling conda env in the same conda installation
    parent_envs = os.path.dirname(sys.prefix)
    if os.path.basename(parent_envs).lower() == 'envs':
        if sys.platform == 'win32':
            candidate = os.path.join(parent_envs, env_name, 'python.exe')
        else:
            candidate = os.path.join(parent_envs, env_name, 'bin', 'python')
        if os.path.exists(candidate):
            return candidate

    # 3) Fallback map by platform
    target_exe = ENV_PYTHON.get(env_name)
    if target_exe and os.path.exists(target_exe):
        return target_exe

    return target_exe or sys.executable


def run_step(name):
    """Run one step as a subprocess with its resolved interpreter (see
    resolve_interpreter()). Returns True on a zero exit code, False otherwise
    (the subprocess's own stdout/stderr is inherited, so failures are visible
    without extra logging here)."""
    if name not in STEP_MAP:
        print(f"Unknown step: '{name}'. Run with --list to see available steps.")
        return False

    step = STEP_MAP[name]
    python_exe = resolve_interpreter(step['env'])
    script_path = os.path.join(PROJECT_ROOT, step['script'])

    if not os.path.exists(python_exe):
        print(f"Interpreter not found: {python_exe}")
        return False
    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return False

    print(f"\n>> Running step: {name}  [env: {step['env']}]")
    print(f"   {step['description']}")
    print('=' * 70)

    result = subprocess.run([python_exe, script_path], cwd=PROJECT_ROOT, check=False)

    print('=' * 70)
    if result.returncode == 0:
        print(f"Step '{name}' finished OK.")
        return True
    print(f"Step '{name}' FAILED (exit code {result.returncode}).")
    return False


def list_steps():
    if os.path.exists(VENV_PYTHON):
        print(f"\nInterpreter: {VENV_PYTHON} (.venv, used for every step)")
    else:
        print("\nInterpreter: no .venv/ found, falling back to conda envs per step")
    print("Available steps (pipeline order):")
    for idx, (name, step) in enumerate(STEP_MAP.items(), 1):
        print(f"  {idx}. {name:<12} [env: {step['env']:<6}] {step['description']}")


def show_menu():
    print('\n' + '=' * 70)
    print(PIPELINE_NAME)
    print('Interactive Mode')
    print('=' * 70)
    list_steps()
    print(f"  0. Exit")
    print(f"  A. Run ALL steps in order")
    print('=' * 70)


def get_user_choice(n_steps):
    step_list = list(STEP_MAP.keys())
    while True:
        try:
            raw = input(f"\nChoose a step (0-{n_steps}, or 'A' for all): ").strip()
            if raw.lower() == 'a':
                return step_list
            choice_num = int(raw)
            if choice_num == 0:
                return None
            if 1 <= choice_num <= n_steps:
                return [step_list[choice_num - 1]]
            print(f"Choose a number between 0 and {n_steps}, or 'A'.")
        except ValueError:
            print("Enter a valid number or 'A'.")
        except KeyboardInterrupt:
            print("\n\nExiting pipeline...")
            sys.exit(0)


def interactive_mode():
    step_list = list(STEP_MAP.keys())

    while True:
        show_menu()
        chosen = get_user_choice(len(step_list))

        if chosen is None:
            print("\nPipeline session ended.")
            break

        for step_name in chosen:
            success = run_step(step_name)
            if not success and len(chosen) > 1:
                try:
                    cont = input(f"\nStep '{step_name}' failed. Continue with the "
                                 f"remaining steps? (y/n): ").strip().lower()
                except KeyboardInterrupt:
                    print("\n\nExiting pipeline...")
                    return
                if cont != 'y':
                    break

        try:
            cont = input("\nRun another step? (y/n): ").strip().lower()
            if cont != 'y':
                print("\nPipeline session ended.")
                break
        except KeyboardInterrupt:
            print("\n\nExiting pipeline...")
            break


def main():
    args = sys.argv[1:]

    if not args:
        try:
            interactive_mode()
        except KeyboardInterrupt:
            print("\n\nPipeline cancelled.")
            sys.exit(0)
        return

    if '--list' in args:
        list_steps()
        return

    if '--all' in args:
        steps = list(STEP_MAP.keys())
    else:
        steps = args

    print('=' * 70)
    print(PIPELINE_NAME)
    print('=' * 70)
    print(f"Steps to run: {', '.join(steps)}")
    print('=' * 70)

    success_count = 0
    for step_name in steps:
        if run_step(step_name):
            success_count += 1
        else:
            print(f"\nStopping: step '{step_name}' failed.")
            break

    print('=' * 70)
    print(f"Pipeline finished: {success_count}/{len(steps)} step(s) succeeded")
    print('=' * 70)


if __name__ == '__main__':
    main()
