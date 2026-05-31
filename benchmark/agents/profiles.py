"""Model profile defaults for benchmark launchers and the unified CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ModelProfile:
    """Defaults that belong to a model family, not to the benchmark rules."""

    model_name: str
    benchmark_defaults: Dict[str, Any] = field(default_factory=dict)
    agent_defaults: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def parser_defaults(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        defaults.update(self.benchmark_defaults)
        defaults.update(self.agent_defaults)
        return defaults


BASE_BENCHMARK_DEFAULTS: Dict[str, Any] = {
    "resolution": [640, 640],
    "passthrough": False,
    "passthrough_env_term_geometry_sync": True,
}

MODEL_PROFILES: Dict[str, ModelProfile] = {
    "random": ModelProfile(
        model_name="random",
        benchmark_defaults=dict(BASE_BENCHMARK_DEFAULTS),
        description="Random action-space baseline.",
    ),
    "vint": ModelProfile(
        model_name="vint",
        benchmark_defaults=dict(BASE_BENCHMARK_DEFAULTS),
        agent_defaults={"waypoint_idx": 5},
        description="ViNT visual navigation; benchmark state machine handles interaction.",
    ),
    "nomad": ModelProfile(
        model_name="nomad",
        benchmark_defaults=dict(BASE_BENCHMARK_DEFAULTS),
        agent_defaults={"waypoint_idx": 5},
        description="NOMAD visual navigation; benchmark state machine handles interaction.",
    ),
    "nomad_yolo": ModelProfile(
        model_name="nomad_yolo",
        benchmark_defaults=dict(BASE_BENCHMARK_DEFAULTS),
        description="NOMAD with YOLO correction; benchmark state machine handles interaction.",
    ),
    "uninavid": ModelProfile(
        model_name="uninavid",
        benchmark_defaults={**BASE_BENCHMARK_DEFAULTS, "resolution": [224, 224]},
        agent_defaults={
            "forward_velocity": 100.0,
            "turn_angle": 30.0,
            "turn_velocity": 0.0,
            "reset_on_phase_switch": True,
        },
        description="Uni-NaVid navigation-only profile.",
    ),
    "omninav": ModelProfile(
        model_name="omninav",
        benchmark_defaults=dict(BASE_BENCHMARK_DEFAULTS),
        agent_defaults={
            "inference_mode": "waypoint",
            "attn_implementation": "flash_attention_2",
            "multi_view": True,
            "view_angles": [-45.0, 0.0, 45.0],
            "view_settle_time": 0.02,
            "forward_velocity": 100.0,
            "turn_angle": 20.0,
            "turn_velocity": 0.0,
            "waypoint_dist_scale": 100.0,
            "waypoint_speed_floor": 70.0,
            "max_turn_deg": 30.0,
            "waypoint_hz_index": 3,
            "max_new_tokens": 8,
            "do_sample": False,
            "temperature": 0.0,
            "quiet": False,
            "log_waypoint": False,
        },
        description="OmniNav navigation-only profile.",
    ),
    "r2zeroshot": ModelProfile(
        model_name="r2zeroshot",
        benchmark_defaults={
            **BASE_BENCHMARK_DEFAULTS,
            "resolution": [640, 360],
            "passthrough": True,
            "passthrough_env_term_geometry_sync": True,
        },
        agent_defaults={"cfg_coef": 2.0, "use_sa2va": True},
        description="R2ZeroShot controls carry/drop, so passthrough is the default.",
    ),
    "apex": ModelProfile(
        model_name="apex",
        benchmark_defaults={**BASE_BENCHMARK_DEFAULTS, "passthrough": True},
        description="Apex controls carry/drop, so passthrough is the default.",
    ),
    "citywalker": ModelProfile(
        model_name="citywalker",
        benchmark_defaults=dict(BASE_BENCHMARK_DEFAULTS),
        agent_defaults={
            "step_scale": -1.0,
            "plane_mode": "xy",
            "bgr_input": True,
            "enable_near_goal_push": True,
            "fine_injured_push_start_cm": 250.0,
            "fine_stretcher_push_start_cm": 350.0,
            "near_goal_push_min_speed": 75.0,
            "near_goal_push_align_angle_deg": 12.0,
            "near_goal_push_stop_margin_cm": 15.0,
        },
        description="CityWalker navigation-only profile; benchmark state machine handles interaction.",
    ),
    "seepointfly": ModelProfile(
        model_name="seepointfly",
        benchmark_defaults={
            **BASE_BENCHMARK_DEFAULTS,
            "resolution": [640, 360],
            "multiagent_env": True,
        },
        agent_defaults={
            "step_scale": -1.0,
            "plane_mode": "xy",
            "citywalker_resolution": [640, 360],
            "bgr_input": True,
            "drone_trigger_radius_m": 4.0,
            "drone_trigger_height_m": 10.0,
            "drone_height_offset_cm": 200.0,
            "drone_max_speed": 0.5,
            "drone_max_z_speed": 0.08,
            "drone_max_yaw": 1.0,
            "drone_search_yaw": 0.25,
            "fine_injured_push_start_cm": 250.0,
            "fine_stretcher_push_start_cm": 350.0,
            "near_goal_push_min_speed": 75.0,
            "near_goal_push_align_angle_deg": 12.0,
            "near_goal_push_stop_margin_cm": 15.0,
            "disable_near_goal_push": False,
        },
        description="SeePointFly uses the multi-agent env and CityWalker ground controller.",
    ),
    "solution": ModelProfile(
        model_name="solution",
        benchmark_defaults={**BASE_BENCHMARK_DEFAULTS, "passthrough": True},
        agent_defaults={"solution_class": "AlgSolution", "solution_input_format": "base64"},
        description="Competition-style user solution.py adapter.",
    ),
}

# Backward-compatible spelling used by the old launcher/output directory.
MODEL_PROFILES["uni_navid"] = MODEL_PROFILES["uninavid"]



def get_model_profile(model_name: Optional[str]) -> ModelProfile:
    return MODEL_PROFILES.get(model_name or "random", MODEL_PROFILES["random"])


def get_profile_defaults(model_name: Optional[str]) -> Dict[str, Any]:
    return dict(get_model_profile(model_name).parser_defaults())


def apply_model_profile_defaults(parser, model_name: Optional[str]) -> None:
    """Apply model-specific defaults to an argparse parser before parse_args()."""

    parser.set_defaults(**get_profile_defaults(model_name))
