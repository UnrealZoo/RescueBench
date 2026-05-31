"""CLI and benchmark launch entrypoints (Step 11 split from rescue_benchmark.py)."""

from __future__ import annotations

import argparse
import json
import os
import signal
from typing import Any, Optional

from agents.agent_base import BaseAgent
from agents.profiles import apply_model_profile_defaults
from utils.human_trajectory_loader import load_reference_trajectories_from_jsonl


def _advanced_help(text: str, expose_advanced: bool) -> str:
    return text if expose_advanced else argparse.SUPPRESS


def create_base_parser(
    description: str = "Rescue Task Benchmark",
    epilog: Optional[str] = None,
    expose_advanced: bool = False,
) -> argparse.ArgumentParser:
    """Create the shared CLI parser used by thin launchers.

    Default help shows everyday experiment options only. Rule thresholds,
    state-machine debug options, and runtime tuning flags remain parseable but
    are hidden unless advanced help is requested.
    """
    env_choices = [
        "UnrealRescue-HongKongStreet",
        "UnrealRescue-SuburbNeighborhood_Day",
        "UnrealRescue-FlexibleRoom",
        "UnrealRescue-Forglar_Map",
        "UnrealRescue-DesertMap",
        "UnrealRescue-DowntownWest",
        "UnrealRescue-Tokyo",
        "UnrealRescue-SuburbNeighborhood_Day_dooropen",
        "UnrealRescueMultiAgent-HongKongStreet",
        "UnrealRescueMultiAgent-SuburbNeighborhood_Day",
        "UnrealRescueMultiAgent-FlexibleRoom",
        "UnrealRescueMultiAgent-Forglar_Map",
        "UnrealRescueMultiAgent-DesertMap",
        "UnrealRescueMultiAgent-DowntownWest",
        "UnrealRescueMultiAgent-Tokyo",
        "UnrealRescueMultiAgent-SuburbNeighborhood_Day_dooropen",
    ]
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog or "",
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="UnrealRescue-FlexibleRoom",
        choices=env_choices,
        help=_advanced_help("Environment id fallback when test_jsonl does not provide env_id", expose_advanced),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Image resolution as WIDTH HEIGHT; usually set by the model profile",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="Difficulty levels to evaluate (0-4)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes per test point",
    )
    parser.add_argument("--no-collision", action="store_true", help="Disable collision detection")
    parser.add_argument("--enable-trajectory", action="store_true", help="Record trajectories")
    parser.add_argument("--enable-similarity", action="store_true", help="Compute trajectory similarity")
    parser.add_argument("--human-traj-dir", type=str, default=None, help="Directory containing human trajectory jsonl files")
    parser.add_argument("--ref-trajectories", type=str, default=None, help="Reference trajectory JSON file or jsonl directory")
    parser.add_argument(
        "--similarity-method",
        type=str,
        default="dtw",
        choices=["dtw", "frechet", "hausdorff"],
        help=_advanced_help("Trajectory similarity method", expose_advanced),
    )
    parser.add_argument("--output", type=str, default="./benchmark_results", help="Output directory")
    parser.add_argument(
        "--resume-jsonl",
        type=str,
        default=None,
        help="Resume from an existing benchmark jsonl and skip completed episodes",
    )
    parser.add_argument(
        "--resume-skip",
        type=str,
        default="all",
        choices=["all", "success-only"],
        help="Resume skip policy: all=all completed episodes, success-only=successful episodes only",
    )
    parser.add_argument(
        "--resume-append",
        action="store_true",
        help="Append directly to --resume-jsonl instead of creating a new jsonl with resumed records",
    )
    parser.add_argument("--render", action="store_true", help="Save render frames")
    parser.add_argument(
        "--save-frame-every",
        type=int,
        default=5,
        help=_advanced_help("Save one historical render frame every N steps when rendering is enabled", expose_advanced),
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Generate videos from saved frames after evaluation (requires ffmpeg)",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help=_advanced_help("Output video FPS", expose_advanced),
    )

    # Benchmark rule / state-machine knobs. Hidden by default; keep parsing for compatibility.
    parser.add_argument(
        "--rescue-distance",
        type=float,
        default=100.0,
        help=_advanced_help("XY distance threshold for carry interaction (cm)", expose_advanced),
    )
    parser.add_argument(
        "--place-distance",
        type=float,
        default=200.0,
        help=_advanced_help("XY distance threshold for drop interaction (cm)", expose_advanced),
    )
    parser.add_argument(
        "--interaction-z-threshold",
        type=float,
        default=220.0,
        help=_advanced_help("Maximum Z gap allowed for interaction (cm)", expose_advanced),
    )
    parser.add_argument(
        "--stage2-success-radius",
        type=float,
        default=200.0,
        help=_advanced_help("Stage-2 success radius (XY, cm), also gated by Z threshold", expose_advanced),
    )
    parser.add_argument(
        "--passthrough",
        action="store_true",
        default=False,
        help=_advanced_help("Passthrough mode: the model controls interaction actions", expose_advanced),
    )
    parser.add_argument(
        "--no-passthrough",
        dest="passthrough",
        action="store_false",
        help=_advanced_help("Active state-machine mode: benchmark inserts carry/drop", expose_advanced),
    )
    parser.set_defaults(passthrough_env_term_geometry_sync=True)
    parser.add_argument(
        "--passthrough-env-term-sync",
        "--passthrough-env-term-geometry-sync",
        dest="passthrough_env_term_geometry_sync",
        action="store_true",
        help=_advanced_help("Use final-position geometry to sync stage 2 when UE terminates first", expose_advanced),
    )
    parser.add_argument(
        "--no-passthrough-env-term-sync",
        dest="passthrough_env_term_geometry_sync",
        action="store_false",
        help=_advanced_help("Disable UE termination geometry sync", expose_advanced),
    )

    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument(
        "--multiagent-env",
        action="store_true",
        help=_advanced_help("Use the multi-agent Rescue environment", expose_advanced),
    )
    parser.add_argument(
        "--solution-path",
        "--solution",
        dest="solution_path",
        type=str,
        default=None,
        help="Path to user solution.py; use with --model solution",
    )
    parser.add_argument(
        "--solution-class",
        type=str,
        default="AlgSolution",
        help="Class name inside solution.py; default is AlgSolution",
    )
    parser.add_argument(
        "--solution-input-format",
        type=str,
        default="base64",
        choices=["base64", "raw"],
        help="Observation format passed to solution.predicts: base64 PNG or raw observation",
    )
    return parser


def add_model_args(parser: argparse.ArgumentParser, expose_advanced: bool = False) -> None:
    """Add model adapter args used by the unified CLI.

    Individual legacy launchers may still define their own focused subset.
    """

    parser.add_argument("--topomap-dir", type=str, default=None, help="ViNT/NOMAD topomap directory")
    parser.add_argument("--waypoint-idx", type=int, default=None, help=_advanced_help("ViNT/NOMAD waypoint index", expose_advanced))
    parser.add_argument("--yolo-weights", type=str, default=None, help="NOMAD-YOLO weights")
    parser.add_argument("--yolo-conf", type=float, default=None, help=_advanced_help("YOLO confidence", expose_advanced))
    parser.add_argument("--no-yolo-correction", action="store_true", help=_advanced_help("Disable YOLO correction", expose_advanced))
    parser.add_argument("--yolo-blend-ratio", type=float, default=None, help=_advanced_help("YOLO correction blend ratio", expose_advanced))

    parser.add_argument("--ckpt-path", type=str, default=None, help="ROCKET / R2ZeroShot ckpt")
    parser.add_argument("--sa2va-path", type=str, default=None, help="Sa2VA path")
    parser.add_argument("--no-sa2va", action="store_true", help=_advanced_help("Disable Sa2VA", expose_advanced))
    parser.add_argument("--cfg-coef", type=float, default=None, help=_advanced_help("ROCKET CFG coef", expose_advanced))
    parser.add_argument("--person-ref-path", type=str, default=None, help="Person reference image")

    parser.add_argument("--workspace", type=str, default=None, help="Apex workspace")
    parser.add_argument("--model-path", type=str, default=None, help="Model weight file or directory")

    parser.add_argument("--config-path", type=str, default=None, help="CityWalker config path")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="CityWalker checkpoint path")
    parser.add_argument("--citywalker-config-path", type=str, default=None, help="SeePointFly: CityWalker config path")
    parser.add_argument("--citywalker-checkpoint-path", type=str, default=None, help="SeePointFly: CityWalker checkpoint path")
    parser.add_argument("--spf-config-path", type=str, default=None, help="SeePointFly config path")
    parser.add_argument("--citywalker-resolution", type=int, nargs=2, default=None, help=_advanced_help("SeePointFly: CityWalker crop resolution", expose_advanced))
    parser.add_argument("--step-scale", type=float, default=None, help=_advanced_help("CityWalker input normalization scale", expose_advanced))
    parser.add_argument("--plane-mode", type=str, default=None, choices=["xy", "xz"], help=_advanced_help("CityWalker ground plane", expose_advanced))
    parser.add_argument("--rgb-input", action="store_true", help=_advanced_help("Input is already RGB; disable default BGR-to-RGB conversion", expose_advanced))
    parser.add_argument("--disable-near-goal-push", action="store_true", help=_advanced_help("Disable CityWalker near-goal push", expose_advanced))
    parser.add_argument("--fine-injured-push-start-cm", type=float, default=None, help=_advanced_help("Near-goal push start threshold for injured target", expose_advanced))
    parser.add_argument("--fine-stretcher-push-start-cm", type=float, default=None, help=_advanced_help("Near-goal push start threshold for stretcher target", expose_advanced))
    parser.add_argument("--near-goal-push-min-speed", type=float, default=None, help=_advanced_help("Minimum speed for near-goal push", expose_advanced))
    parser.add_argument("--near-goal-push-align-angle-deg", type=float, default=None, help=_advanced_help("Alignment angle threshold for near-goal push", expose_advanced))
    parser.add_argument("--near-goal-push-stop-margin-cm", type=float, default=None, help=_advanced_help("Stop near-goal push this close to the interaction threshold", expose_advanced))

    parser.add_argument("--drone-trigger-radius-m", type=float, default=None, help=_advanced_help("SeePointFly handoff radius in meters", expose_advanced))
    parser.add_argument("--drone-trigger-height-m", type=float, default=None, help=_advanced_help("SeePointFly handoff height in meters", expose_advanced))
    parser.add_argument("--drone-height-offset-cm", type=float, default=None, help=_advanced_help("Drone initial height offset in cm", expose_advanced))
    parser.add_argument("--drone-max-speed", type=float, default=None, help=_advanced_help("SeePointFly drone max speed", expose_advanced))
    parser.add_argument("--drone-max-z-speed", type=float, default=None, help=_advanced_help("SeePointFly drone max z speed", expose_advanced))
    parser.add_argument("--drone-max-yaw", type=float, default=None, help=_advanced_help("SeePointFly drone max yaw", expose_advanced))
    parser.add_argument("--drone-search-yaw", type=float, default=None, help=_advanced_help("Fallback drone search yaw", expose_advanced))

    parser.add_argument("--forward-velocity", type=float, default=None, help=_advanced_help("Discrete forward velocity / waypoint speed cap", expose_advanced))
    parser.add_argument("--turn-angle", type=float, default=None, help=_advanced_help("Discrete turn angle", expose_advanced))
    parser.add_argument("--turn-velocity", type=float, default=None, help=_advanced_help("Forward velocity while turning", expose_advanced))
    parser.set_defaults(reset_on_phase_switch=True)
    parser.add_argument("--no-reset-on-phase-switch", dest="reset_on_phase_switch", action="store_false", help=_advanced_help("Keep Uni-NaVid context across phase switches", expose_advanced))
    parser.add_argument("--phase2-instruction", type=str, default=None, help=_advanced_help("Uni-NaVid phase-2 instruction text", expose_advanced))

    parser.add_argument("--inference-mode", choices=["waypoint", "text"], default=None, help=_advanced_help("OmniNav inference mode", expose_advanced))
    parser.add_argument("--attn-implementation", type=str, default=None, help=_advanced_help("OmniNav attention implementation", expose_advanced))
    parser.add_argument("--disable-multi-view", action="store_true", help=_advanced_help("Disable OmniNav multi-view sampling", expose_advanced))
    parser.add_argument("--view-angles", type=float, nargs="+", default=None, help=_advanced_help("OmniNav multi-view angles", expose_advanced))
    parser.add_argument("--view-settle-time", type=float, default=None, help=_advanced_help("OmniNav view-settle time in seconds", expose_advanced))
    parser.add_argument("--waypoint-dist-scale", type=float, default=None, help=_advanced_help("OmniNav waypoint distance scale", expose_advanced))
    parser.add_argument("--waypoint-speed-floor", type=float, default=None, help=_advanced_help("OmniNav waypoint speed floor", expose_advanced))
    parser.add_argument("--max-turn-deg", type=float, default=None, help=_advanced_help("OmniNav waypoint max turn", expose_advanced))
    parser.add_argument("--waypoint-hz-index", type=int, choices=[0, 1, 2, 3, 4], default=None, help=_advanced_help("OmniNav waypoint horizon index", expose_advanced))
    parser.add_argument("--max-new-tokens", type=int, default=None, help=_advanced_help("OmniNav text mode max new tokens", expose_advanced))
    parser.add_argument("--do-sample", action="store_true", help=_advanced_help("OmniNav text mode sampling", expose_advanced))
    parser.add_argument("--temperature", type=float, default=None, help=_advanced_help("OmniNav text mode temperature", expose_advanced))
    parser.add_argument("--quiet", action="store_true", help=_advanced_help("Reduce agent log output", expose_advanced))
    parser.add_argument("--log-waypoint", action="store_true", help=_advanced_help("Print OmniNav waypoint debug logs", expose_advanced))


def run_benchmark_from_args(
    args: Any,
    agent: BaseAgent,
    model_name: Optional[str] = None,
):
    """Shared benchmark entrypoint used by thin launchers."""
    from rescue_benchmark import RescueBenchmark

    np = __import__("numpy")
    model_name = model_name or getattr(args, "model", "unknown")
    benchmark = None

    def _cleanup(signum=None, frame=None):
        try:
            if benchmark and benchmark.env:
                benchmark.env.close()
                benchmark.env = None
        except Exception:
            pass
        if signum is not None:
            os._exit(1)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    ref_trajs = {}
    human_traj_dir = getattr(args, "human_traj_dir", None)
    if human_traj_dir and os.path.isdir(human_traj_dir):
        ref_trajs.update(load_reference_trajectories_from_jsonl(human_traj_dir))

    ref_path = getattr(args, "ref_trajectories", None)
    if ref_path and os.path.exists(ref_path):
        if os.path.isdir(ref_path):
            ref_trajs.update(load_reference_trajectories_from_jsonl(ref_path))
        else:
            with open(ref_path, encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    ref_trajs[k] = np.asarray(v, dtype=np.float32)
    if ref_trajs:
        print(f"[Setup] Loaded {len(ref_trajs)} reference trajectories")

    base_output = os.path.abspath(getattr(args, "output", "./benchmark_results"))
    model_output_dir = os.path.join(base_output, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    benchmark = RescueBenchmark(
        env_id=getattr(args, "env", "UnrealRescue-FlexibleRoom"),
        agent=agent,
        resolution=tuple(getattr(args, "resolution", [640, 640])),
        render=getattr(args, "render", False),
        output_dir=model_output_dir,
        enable_collision_detection=not getattr(args, "no_collision", False),
        enable_trajectory_recording=getattr(args, "enable_trajectory", False),
        enable_path_similarity=getattr(args, "enable_similarity", False),
        reference_trajectories=ref_trajs,
        similarity_method=getattr(args, "similarity_method", "dtw"),
        rescue_distance=getattr(args, "rescue_distance", 100.0),
        place_distance=getattr(args, "place_distance", 200.0),
        interaction_z_threshold=getattr(args, "interaction_z_threshold", 220.0),
        stage2_success_radius=getattr(args, "stage2_success_radius", 200.0),
        passthrough=getattr(args, "passthrough", False),
        save_frame_every=getattr(args, "save_frame_every", 5),
        save_video=getattr(args, "save_video", False),
        video_fps=getattr(args, "video_fps", 10),
        resume_jsonl=getattr(args, "resume_jsonl", None),
        resume_skip=getattr(args, "resume_skip", "all"),
        resume_append=getattr(args, "resume_append", False),
        passthrough_env_term_geometry_sync=getattr(
            args, "passthrough_env_term_geometry_sync", True
        ),
        multiagent_env=getattr(args, "multiagent_env", False),
    )
    try:
        result = benchmark.run_benchmark(
            levels=getattr(args, "levels", [2, 3, 4]),
            episodes_per_point=getattr(args, "episodes", 1),
            model_name=model_name,
        )
        print(f"\n{'='*60}\n BENCHMARK COMPLETED!\n{'='*60}")
        return result
    except KeyboardInterrupt:
        _cleanup()
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback

        traceback.print_exc()
        _cleanup()
    finally:
        if benchmark and benchmark.env:
            try:
                benchmark.env.close()
            except Exception:
                pass


def _preparse_model(default: str = "random") -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", "--profile", dest="model", default=default)
    args, _ = parser.parse_known_args()
    return args.model


def main() -> None:
    """Entrypoint for running ``python rescue_benchmark.py`` directly."""
    from agents.factory import AGENT_REGISTRY, get_agent_from_cli_args

    selected_model = _preparse_model()
    parser = create_base_parser()
    parser.add_argument(
        "--model",
        "--profile",
        dest="model",
        type=str,
        default=selected_model,
        choices=["random"] + sorted(AGENT_REGISTRY),
        help="Model/profile name: " + ", ".join(["random"] + sorted(AGENT_REGISTRY)),
    )
    add_model_args(parser)
    apply_model_profile_defaults(parser, selected_model)
    args = parser.parse_args()

    agent = get_agent_from_cli_args(args.model, args)
    run_benchmark_from_args(args, agent, model_name=args.model)
