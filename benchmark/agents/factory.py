"""Agent registry and factory (Step 12 — decoupled from rescue_benchmark.py)."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from agents.agent_base import BaseAgent, RandomAgent
from agents.profiles import get_model_profile

# model_name -> (module_path, class_name)
AGENT_REGISTRY: Dict[str, Tuple[str, str]] = {
    "vint": ("agents.vint_agent", "VINTAgent"),
    "nomad": ("agents.nomad_agent", "NOMADAgent"),
    "nomad_yolo": ("agents.nomad_yolo_agent", "NOMADYOLOAgent"),
    "r2zeroshot": ("agents.r2zeroshot_agent", "R2ZeroShotAgent"),
    "apex": ("agents.apex_agent", "ApexAgent"),
    "citywalker": ("agents.citywalker_agent", "CityWalkerAgent"),
    "seepointfly": ("agents.seepointfly_agent", "SeePointFlyAgent"),
    "omninav": ("agents.omninav_agent", "OmniNavAgent"),
    "uninavid": ("agents.uninavid_agent", "UniNaVidAgent"),
    "uni_navid": ("agents.uninavid_agent", "UniNaVidAgent"),
    "solution": ("agents.solution_agent", "SolutionAgent"),
}

AGENT_KWARG_KEYS: Dict[str, Iterable[str]] = {
    "vint": ("device", "topomap_dir", "waypoint_idx"),
    "nomad": ("device", "topomap_dir", "waypoint_idx"),
    "nomad_yolo": (
        "device",
        "topomap_dir",
        "yolo_weights",
        "yolo_conf",
        "use_yolo_correction",
        "yolo_blend_ratio",
    ),
    "r2zeroshot": (
        "cfg_coef",
        "ckpt_path",
        "sa2va_path",
        "use_sa2va",
        "person_ref_path",
    ),
    "apex": ("workspace",),
    "citywalker": (
        "config_path",
        "checkpoint_path",
        "device",
        "step_scale",
        "plane_mode",
        "bgr_input",
        "enable_near_goal_push",
        "fine_injured_push_start_cm",
        "fine_stretcher_push_start_cm",
        "near_goal_push_min_speed",
        "near_goal_push_align_angle_deg",
        "near_goal_push_stop_margin_cm",
    ),
    "seepointfly": (
        "citywalker_checkpoint_path",
        "citywalker_config_path",
        "spf_config_path",
        "device",
        "step_scale",
        "plane_mode",
        "bgr_input",
        "citywalker_resolution",
        "drone_trigger_radius_m",
        "drone_trigger_height_m",
        "drone_cruise_altitude_cm",
        "drone_max_speed",
        "drone_max_z_speed",
        "drone_max_yaw",
        "drone_search_yaw",
        "fine_injured_push_start_cm",
        "fine_stretcher_push_start_cm",
        "near_goal_push_min_speed",
        "near_goal_push_align_angle_deg",
        "near_goal_push_stop_margin_cm",
        "disable_near_goal_push",
    ),
    "omninav": (
        "model_path",
        "device",
        "bgr_input",
        "inference_mode",
        "attn_implementation",
        "multi_view",
        "multi_view_angles",
        "multi_view_settle_time",
        "forward_velocity",
        "turn_angle",
        "turn_velocity",
        "waypoint_dist_scale",
        "waypoint_speed_floor",
        "waypoint_max_turn_deg",
        "waypoint_horizon_index",
        "max_new_tokens",
        "do_sample",
        "temperature",
        "verbose",
        "log_waypoint_inference",
    ),
    "uninavid": (
        "model_path",
        "device",
        "forward_velocity",
        "turn_angle",
        "turn_velocity",
        "reset_on_phase_switch",
        "phase2_instruction",
    ),
    "uni_navid": (
        "model_path",
        "device",
        "forward_velocity",
        "turn_angle",
        "turn_velocity",
        "reset_on_phase_switch",
        "phase2_instruction",
    ),
    "solution": ("solution_path", "solution_class", "solution_input_format"),
}


def get_agent(model_name: str, env, **kwargs) -> BaseAgent:
    """Instantiate a registered agent or RandomAgent baseline."""
    if model_name == "random":
        return RandomAgent(env.action_space)

    if model_name in AGENT_REGISTRY:
        mod_path, cls_name = AGENT_REGISTRY[model_name]
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            filtered = {k: v for k, v in kwargs.items() if v is not None}
            return cls(**filtered)
        except Exception as e:
            if model_name == "solution":
                raise
            print(f"[Warning] {cls_name} load failed: {e}, using RandomAgent")
            return RandomAgent(env.action_space)

    print(f"[Warning] Unknown model '{model_name}', using RandomAgent")
    return RandomAgent(env.action_space)


def make_temp_env_for_action_space(
    env_id: str,
    resolution: Union[tuple, list],
):
    """Create a wrapped gym env solely to read ``action_space`` for agent construction.

    Caller must ``close()`` the returned env when done.
    """
    gym = __import__("gym")
    config_ue = __import__(
        "gym_rescue.envs.wrappers.configUE",
        fromlist=["ConfigUEWrapper"],
    )
    temp_env = gym.make(
        env_id,
        action_type="Mixed",
        observation_type="Color",
        reset_type=2,
    )
    temp_env = config_ue.ConfigUEWrapper(
        temp_env,
        offscreen=True,
        resolution=tuple(resolution),
    )
    return temp_env


def _profile_arg(args: Any, name: str, default: Any = None) -> Any:
    profile = get_model_profile(getattr(args, "model", None))
    value = getattr(args, name, None)
    if value is None:
        value = profile.agent_defaults.get(name, default)
    return value


def _rgb_to_bgr_input(args: Any) -> bool:
    if hasattr(args, "rgb_input"):
        return not bool(getattr(args, "rgb_input"))
    return bool(_profile_arg(args, "bgr_input", True))


def _filter_agent_kwargs(model_name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    keys = AGENT_KWARG_KEYS.get(model_name)
    if keys is None:
        return kwargs
    return {key: kwargs.get(key) for key in keys}


def get_agent_from_cli_args(model_name: str, args: Any) -> BaseAgent:
    """Build agent from argparse namespace (shared by ``core.cli.main``)."""
    setattr(args, "model", model_name)
    temp_env = make_temp_env_for_action_space(args.env, args.resolution)
    try:
        reset_on_phase_switch = getattr(args, "reset_on_phase_switch", None)
        if reset_on_phase_switch is None:
            reset_on_phase_switch = not getattr(args, "no_reset_on_phase_switch", False)

        citywalker_config_path = getattr(args, "citywalker_config_path", None) or getattr(args, "config_path", None)
        citywalker_checkpoint_path = getattr(args, "citywalker_checkpoint_path", None) or getattr(
            args, "checkpoint_path", None
        )

        all_kwargs = {
            "device": getattr(args, "device", "cuda"),
            "topomap_dir": _profile_arg(args, "topomap_dir"),
            "waypoint_idx": _profile_arg(args, "waypoint_idx", 5),
            "yolo_weights": _profile_arg(args, "yolo_weights"),
            "yolo_conf": _profile_arg(args, "yolo_conf", 0.5),
            "use_yolo_correction": not getattr(args, "no_yolo_correction", False),
            "yolo_blend_ratio": _profile_arg(args, "yolo_blend_ratio", 0.3),
            "cfg_coef": _profile_arg(args, "cfg_coef", 2.0),
            "ckpt_path": _profile_arg(args, "ckpt_path"),
            "sa2va_path": _profile_arg(args, "sa2va_path"),
            "use_sa2va": not getattr(args, "no_sa2va", False),
            "person_ref_path": _profile_arg(args, "person_ref_path"),
            "workspace": _profile_arg(args, "workspace"),
            "model_path": _profile_arg(args, "model_path"),
            "config_path": getattr(args, "config_path", None),
            "checkpoint_path": getattr(args, "checkpoint_path", None),
            "citywalker_config_path": citywalker_config_path,
            "citywalker_checkpoint_path": citywalker_checkpoint_path,
            "spf_config_path": _profile_arg(args, "spf_config_path"),
            "step_scale": _profile_arg(args, "step_scale", -1.0),
            "plane_mode": _profile_arg(args, "plane_mode", "xy"),
            "bgr_input": _rgb_to_bgr_input(args),
            "enable_near_goal_push": not getattr(args, "disable_near_goal_push", False),
            "disable_near_goal_push": getattr(args, "disable_near_goal_push", False),
            "fine_injured_push_start_cm": _profile_arg(args, "fine_injured_push_start_cm", 250.0),
            "fine_stretcher_push_start_cm": _profile_arg(args, "fine_stretcher_push_start_cm", 350.0),
            "near_goal_push_min_speed": _profile_arg(args, "near_goal_push_min_speed", 75.0),
            "near_goal_push_align_angle_deg": _profile_arg(args, "near_goal_push_align_angle_deg", 12.0),
            "near_goal_push_stop_margin_cm": _profile_arg(args, "near_goal_push_stop_margin_cm", 15.0),
            "citywalker_resolution": _profile_arg(args, "citywalker_resolution", [640, 360]),
            "drone_trigger_radius_m": _profile_arg(args, "drone_trigger_radius_m", 4.0),
            "drone_trigger_height_m": _profile_arg(args, "drone_trigger_height_m", 10.0),
            "drone_cruise_altitude_cm": _profile_arg(args, "drone_height_offset_cm", 200.0),
            "drone_max_speed": _profile_arg(args, "drone_max_speed", 0.5),
            "drone_max_z_speed": _profile_arg(args, "drone_max_z_speed", 0.08),
            "drone_max_yaw": _profile_arg(args, "drone_max_yaw", 1.0),
            "drone_search_yaw": _profile_arg(args, "drone_search_yaw", 0.25),
            "forward_velocity": _profile_arg(args, "forward_velocity", 100.0),
            "turn_angle": _profile_arg(args, "turn_angle", 30.0),
            "turn_velocity": _profile_arg(args, "turn_velocity", 0.0),
            "reset_on_phase_switch": reset_on_phase_switch,
            "phase2_instruction": _profile_arg(args, "phase2_instruction"),
            "inference_mode": _profile_arg(args, "inference_mode", "waypoint"),
            "attn_implementation": _profile_arg(args, "attn_implementation", "flash_attention_2"),
            "multi_view": not getattr(args, "disable_multi_view", False),
            "multi_view_angles": _profile_arg(args, "view_angles", [-45.0, 0.0, 45.0]),
            "multi_view_settle_time": _profile_arg(args, "view_settle_time", 0.02),
            "waypoint_dist_scale": _profile_arg(args, "waypoint_dist_scale", 100.0),
            "waypoint_speed_floor": _profile_arg(args, "waypoint_speed_floor", 70.0),
            "waypoint_max_turn_deg": _profile_arg(args, "max_turn_deg", 30.0),
            "waypoint_horizon_index": _profile_arg(args, "waypoint_hz_index", 3),
            "max_new_tokens": _profile_arg(args, "max_new_tokens", 8),
            "do_sample": getattr(args, "do_sample", False),
            "temperature": _profile_arg(args, "temperature", 0.0),
            "verbose": not getattr(args, "quiet", False),
            "log_waypoint_inference": getattr(args, "log_waypoint", False),
            "solution_path": getattr(args, "solution_path", None),
            "solution_class": _profile_arg(args, "solution_class", "AlgSolution"),
            "solution_input_format": _profile_arg(args, "solution_input_format", "base64"),
        }
        return get_agent(model_name, temp_env, **_filter_agent_kwargs(model_name, all_kwargs))
    finally:
        temp_env.close()


# Backward-compatible alias used by older imports
_AGENT_REGISTRY = AGENT_REGISTRY
