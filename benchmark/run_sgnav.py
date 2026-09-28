#!/usr/bin/env python3
"""Run the patched SG-Nav baseline with the shared RescueBench evaluator."""

import os
import sys


def _restart_with_conda_cuda_libraries():
    """Load conda's matching CUDA libraries before importing FAISS/torch."""
    prefix = os.environ.get("CONDA_PREFIX")
    if sys.platform != "linux" or not prefix or os.environ.get("_SGNAV_CUDA_LIBS_FIXED"):
        return
    lib = os.path.join(prefix, "lib")
    keep = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":")
            if p and "/cuda" not in p and "cuda-" not in p]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join([lib] + keep)
    preloads = [os.path.join(lib, name) for name in ("libcublas.so.11", "libcublasLt.so.11")
                if os.path.isfile(os.path.join(lib, name))]
    if preloads:
        env["LD_PRELOAD"] = ":".join(preloads + [env.get("LD_PRELOAD", "")]).rstrip(":")
    env["_SGNAV_CUDA_LIBS_FIXED"] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_restart_with_conda_cuda_libraries()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.profiles import apply_model_profile_defaults  # noqa: E402
from core.cli import add_model_args  # noqa: E402
from rescue_benchmark import create_base_parser, run_benchmark_from_args  # noqa: E402


def main():
    parser = create_base_parser(description="SG-Nav RescueBench baseline")
    add_model_args(parser)
    apply_model_profile_defaults(parser, "sgnav")
    args = parser.parse_args()
    if args.observation_type != "Rgbd":
        parser.error("SG-Nav requires --observation-type Rgbd")
    from agents.sgnav_rescue_agent import SGNavRescueAgent

    agent = SGNavRescueAgent(
        sgnav_root=args.sgnav_root,
        config_file=args.sgnav_config,
        openworld=int(args.sgnav_openworld),
        goal_detect_interval=args.goal_detect_interval,
    )
    run_benchmark_from_args(args, agent, model_name="sgnav")


if __name__ == "__main__":
    main()
