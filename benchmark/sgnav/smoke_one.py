#!/usr/bin/env python3
"""Run one real SG-Nav RescueBench episode for installation validation."""

import argparse
from dataclasses import asdict
import json
import os
import sys

BENCHMARK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCHMARK_DIR)

from agents.sgnav_rescue_agent import SGNavRescueAgent  # noqa: E402
from rescue_benchmark import RescueBenchmark  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgnav-root", default=os.environ.get("SGNAV_ROOT"))
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--point-id", type=int, default=0)
    parser.add_argument("--output", default="./benchmark_results/sgnav_smoke")
    args = parser.parse_args()
    agent = SGNavRescueAgent(sgnav_root=args.sgnav_root)
    benchmark = RescueBenchmark(
        agent=agent,
        resolution=(640, 480),
        observation_type="Rgbd",
        place_distance=200.0,
        enable_collision_detection=False,
        output_dir=args.output,
    )
    try:
        result = benchmark.run_episode(args.level, args.point_id)
        os.makedirs(args.output, exist_ok=True)
        destination = os.path.join(args.output, f"level_{args.level}_point_{args.point_id}.json")
        with open(destination, "w", encoding="utf-8") as stream:
            json.dump(asdict(result), stream, ensure_ascii=False, indent=2, default=str)
        print(f"[SGNAV_SMOKE_COMPLETE] {destination}")
    finally:
        benchmark._close_env()


if __name__ == "__main__":
    main()
