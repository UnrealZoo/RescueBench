#!/usr/bin/env python3
"""Recommended Rescue benchmark experiment entrypoint.

This file intentionally stays thin. It exists so new users can run experiments
through a clearly named entrypoint while old code can keep importing or running
``rescue_benchmark.py``.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["UnrealEnv"] = "/media/littlecave/T9/UnrealEnv"

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from core.cli import main


if __name__ == "__main__":
    main()
