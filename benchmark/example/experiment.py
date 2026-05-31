#!/usr/bin/env python3
"""Minimal editable experiment launcher.

Run from the repo root:

    python benchmark/example/experiment.py

Change CONFIG below for quick local experiments. For full CLI control, use
``python benchmark/experiment.py --help``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "benchmark"
SOLUTION_PATH = Path(__file__).with_name("solution.py")

if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from core.cli import main


CONFIG = {
    "model": "solution",
    "solution": str(SOLUTION_PATH),
    "levels": [0],
    "episodes": 1,
    "output": str(REPO_ROOT / "benchmark_results"),
    # True: solution.py returns interaction=3/4 itself, like Apex/R2.
    # False: benchmark state machine inserts carry/drop when close enough.
    "passthrough": True,
    # Keep this false for the first smoke test; enable when you want frames.
    "render": True,
}


def _argv_from_config():
    argv = ["benchmark/example/experiment.py"]
    argv += ["--model", CONFIG["model"]]
    argv += ["--solution", CONFIG["solution"]]
    argv += ["--levels", *[str(level) for level in CONFIG["levels"]]]
    argv += ["--episodes", str(CONFIG["episodes"])]
    argv += ["--output", CONFIG["output"]]
    argv.append("--passthrough" if CONFIG.get("passthrough", True) else "--no-passthrough")
    if CONFIG.get("render"):
        argv.append("--render")
    return argv


if __name__ == "__main__":
    extra_args = sys.argv[1:]
    if not extra_args:
        sys.argv = _argv_from_config()
    elif any(arg in ("-h", "--help") for arg in extra_args):
        # Let the unified CLI render normal help instead of starting the env.
        pass
    else:
        # Keep the editable defaults, but allow quick command-line overrides.
        sys.argv = _argv_from_config() + extra_args
    main()
