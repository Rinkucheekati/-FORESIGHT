#!/usr/bin/env python
"""Project FORESIGHT — final reproducibility runner (root level).

Executes the existing project stages, in order, using the entry points that
are already committed in this repository. This runner does NOT re-implement,
redesign, or override any stage: every step below is the exact module/script
as it exists, invoked with its own defaults and CLI contract. Datasets,
models, thresholds, schemas and results are therefore untouched by this file.

Stages (in order)
-----------------
    1. official_selection   python -m src.official_selection
                            Deterministic 25,000-row selection from the
                            official source (seed 42) + SHA-256 manifest.
    2. retail_adapter       python -m src.retail_adapter
                            Canonical four-file official retail extracts in
                            data/raw (deterministic subset, seed 42). NOTE:
                            this stage deliberately refuses to overwrite its
                            own outputs if they already exist (by design in
                            src/retail_adapter.py), so it must run from a
                            clean data/raw state.
    3. D1 pipeline          python -m src.pipeline
                            Ingest, schema-gate, clean, cross-validate and
                            write analysis-ready outputs + quality report.
    4. D2 report            python reports/generate_d2_report.py
                            Data-quality & EDA memo, JSON report, charts.
    5. D3 forecast          python reports/generate_d3_report.py
                            Weekly SKU forecast, rolling-origin backtest,
                            model comparison and forecast report.
    6. D4 report            python reports/generate_d4_report.py
                            Inventory risk scoring, decisions, rupee values,
                            recommendations and risk report.
    7. D7 executive deck    python reports/generate_d7_executive_readout.py
                            Executive PowerPoint readout.

Behaviour
---------
- Stops immediately at the first failing stage and exits with that stage's
  return code (always non-zero on failure).
- Prints a clear banner (stage name + command) before each step.
- Every stage runs in a fresh interpreter with the repository root (derived
  from this file's location, never hard-coded) as the working directory.
- Contains no business results; all numbers are produced by the stages
  themselves from the official data.

Usage
-----
    python run_all.py             # run the full reproducibility pipeline
    python run_all.py --dry-run   # validate the runner + entry points only;
                                  # executes nothing, regenerates nothing
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# (stage id, stage name, description, command tail after the interpreter).
# Entry points are exactly the existing repository modules/scripts.
STAGES = [
    ("1/7", "official_selection",
     "Deterministic 25,000-row official transaction selection (seed 42) "
     "+ SHA-256 selection manifest",
     ["-m", "src.official_selection"]),
    ("2/7", "retail_adapter",
     "Canonical four-file official retail extracts in data/raw "
     "(deterministic subset, seed 42)",
     ["-m", "src.retail_adapter"]),
    ("3/7", "D1 — data pipeline",
     "Ingest, schema gate, clean, cross-validate; write analysis-ready "
     "outputs + data-quality report",
     ["-m", "src.pipeline"]),
    ("4/7", "D2 — EDA report",
     "Data-quality & EDA memo, JSON report and charts from D1 outputs",
     ["reports/generate_d2_report.py"]),
    ("5/7", "D3 — forecast & report",
     "Weekly SKU-level forecast, rolling-origin backtest, model comparison, "
     "forecast report",
     ["reports/generate_d3_report.py"]),
    ("6/7", "D4 — risk report",
     "Inventory risk scoring, decisions, rupee values, recommendations, "
     "risk report",
     ["reports/generate_d4_report.py"]),
    ("7/7", "D7 — executive readout",
     "Executive PowerPoint readout (rupee impact first, honest accuracy)",
     ["reports/generate_d7_executive_readout.py"]),
]

RULE = "=" * 72

def stage_command(tail: list[str]) -> list[str]:
    """Full command for a stage: this interpreter + the existing entry point."""
    return [sys.executable, *tail]


def entry_point_path(tail: list[str]) -> Path:
    """Repository file that must exist for the stage entry point to be valid."""
    if tail[0] == "-m":
        module = tail[1]
        return REPO_ROOT.joinpath(*module.split(".")).with_suffix(".py")
    return REPO_ROOT.joinpath(*Path(tail[0]).parts)


def child_env() -> dict:
    """Environment for stage processes.

    UTF-8 I/O so stage output (e.g. rupee and check-mark glyphs) survives
    redirection; no bytecode files written. These settings do not alter any
    stage's logic, data, or results.
    """
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def print_stage_banner(stage_id: str, name: str, description: str,
                       tail: list[str]) -> None:
    print(RULE)
    print(f"  STAGE {stage_id}: {name}")
    print(f"  {description}")
    print(f"  command: {Path(sys.executable).name} {' '.join(tail)}")
    print(RULE, flush=True)


def run_stage(tail: list[str]) -> int:
    """Run one stage; return its exit code. Output streams live to the console."""
    completed = subprocess.run(
        stage_command(tail),
        cwd=str(REPO_ROOT),
        env=child_env(),
        check=False,
    )
    return completed.returncode


def dry_run() -> int:
    """Validate the runner and every stage entry point WITHOUT executing them."""
    print(RULE)
    print("  RUN_ALL DRY RUN — no stage will be executed or regenerated")
    print(RULE)
    failures = 0
    for stage_id, name, description, tail in STAGES:
        print_stage_banner(stage_id, name, description, tail)
        target = entry_point_path(tail)
        if target.is_file():
            print(f"  entry point OK  : {target.relative_to(REPO_ROOT).as_posix()}")
        else:
            failures += 1
            print(f"  entry point MISSING: {target}")
    print(RULE)
    if failures:
        print(f"  DRY RUN FAILED — {failures} entry point(s) missing")
        print(RULE)
        return 1
    print(f"  DRY RUN PASSED — all {len(STAGES)} stage entry points exist")
    print("  (re-run without --dry-run to execute the full pipeline)")
    print(RULE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python run_all.py",
        description=(
            "Project FORESIGHT reproducibility runner: executes the existing "
            "stages in order (official_selection, retail_adapter, D1, D2, D3, "
            "D4, D7), stopping at the first failure."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the stage plan and verify entry points exist, but run "
             "nothing (nothing is retrained or regenerated)",
    )
    args = parser.parse_args()

    if args.dry_run:
        return dry_run()

    pipeline_start = time.monotonic()
    for stage_id, name, description, tail in STAGES:
        print_stage_banner(stage_id, name, description, tail)
        stage_start = time.monotonic()
        returncode = run_stage(tail)
        elapsed = time.monotonic() - stage_start
        if returncode != 0:
            print("!" * 72)
            print(f"  STAGE {stage_id} ({name}) FAILED — exit code {returncode}")
            print("  Stopping immediately: later stages depend on it.")
            print("!" * 72, flush=True)
            return returncode
        print(f"  STAGE {stage_id} ({name}) OK — {elapsed:.1f}s", flush=True)

    total = time.monotonic() - pipeline_start
    print(RULE)
    print(f"  ALL {len(STAGES)} STAGES COMPLETED SUCCESSFULLY — total {total:.1f}s")
    print(RULE)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nRUN_ALL INTERRUPTED by user — exiting with code 130",
              file=sys.stderr)
        sys.exit(130)
