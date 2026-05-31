from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from agents.agent_base import BaseAgent
from core.metrics import EpisodeMetrics
from agents.citywalker_agent import CityWalkerAgent
from gym_rescue.envs.utils import misc


AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(AGENT_DIR)
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)
SEEPOINTFLY_SRC = os.path.join(PROJECT_ROOT, "baseline_model", "see-point-fly", "src")
SEEPOINTFLY_ROOT = os.path.join(PROJECT_ROOT, "baseline_model", "see-point-fly")
DEFAULT_SPF_CONFIG_PATH = os.path.join(SEEPOINTFLY_ROOT, "config_sim.yaml")
CITYWALKER_ROOT = os.path.join(PROJECT_ROOT, "baseline_model", "CityWalker")
DEFAULT_CITYWALKER_CONFIG_PATH = os.path.join(CITYWALKER_ROOT, "config", "citywalk_2000hr.yaml")

if SEEPOINTFLY_SRC not in sys.path:
    sys.path.insert(0, SEEPOINTFLY_SRC)

from spf.sim.drone_space import SimDroneActionSpace  # noqa: E402
from spf.sim.action_projector import SimActionProjector  # noqa: E402


class _FallbackPID:
    def __init__(self, kp: float, ki: float, kd: float, setpoint: float = 1.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.setpoint = float(setpoint)
        self._integral = 0.0
        self._last_error: Optional[float] = None
        self._last_time: Optional[float] = None

    def reset(self) -> None:
        self._integral = 0.0
        self._last_error = None
        self._last_time = None

    def __call__(self, measurement: float) -> float:
        now = time.time()
        error = self.setpoint - float(measurement)
        dt = 0.0 if self._last_time is None else max(now - self._last_time, 1e-3)
        if dt > 0.0:
            self._integral += error * dt
        derivative = 0.0
        if self._last_error is not None and dt > 0.0:
            derivative = (error - self._last_error) / dt
        self._last_error = error
        self._last_time = now
        return self.kp * error + self.ki * self._integral + self.kd * derivative


try:
    from simple_pid import PID as _PIDImpl
except ImportError:
    _PIDImpl = _FallbackPID


class DronePoseTracker3D:
    def __init__(
        self,
        velocity_high: float = 0.3,
        velocity_low: float = -0.3,
        vertical_high: float = 0.3,
        vertical_low: float = -0.3,
        angle_high: float = 0.5,
        angle_low: float = -0.5,
        expected_distance: float = 0.0,
        expected_angle: float = 0.0,
        expected_z_offset: float = 0.0,
        yaw_sign: float = -1.0,
        enable_z: bool = False,
        z_sign: float = 1.0,
    ):
        self.velocity_high = float(velocity_high)
        self.velocity_low = float(velocity_low)
        self.vertical_high = float(vertical_high)
        self.vertical_low = float(vertical_low)
        self.angle_high = float(angle_high)
        self.angle_low = float(angle_low)
        self.expected_distance = float(expected_distance)
        self.expected_angle = float(expected_angle)
        self.expected_z_offset = float(expected_z_offset)
        # Sign correction for simulator yaw convention.
        # -1.0 is safer for current rescue env mapping.
        self.yaw_sign = float(yaw_sign)
        self.enable_z = bool(enable_z)
        self.z_sign = float(z_sign)
        self.angle_pid = _PIDImpl(0.1, 0.01, 0.0, setpoint=1.0)
        self.velocity_pid = _PIDImpl(0.01, 0.0, 0.001, setpoint=1.0)
        self.vertical_pid = _PIDImpl(0.01, 0.0, 0.001, setpoint=1.0)

    def reset(self) -> None:
        for pid in (self.angle_pid, self.velocity_pid, self.vertical_pid):
            reset_fn = getattr(pid, "reset", None)
            if callable(reset_fn):
                reset_fn()

    def act(self, pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
        delt_yaw = misc.get_direction(pose, target_pose)
        # Keep yaw control explicit and monotonic to avoid PID sign ambiguity.
        # delt_yaw is already a relative angle error (degrees) in current env.
        angle = np.clip(
            self.yaw_sign * (float(delt_yaw) - self.expected_angle) / 45.0,
            self.angle_low,
            self.angle_high,
        )
        delt_distance = float(np.linalg.norm(np.array(pose[:2]) - np.array(target_pose[:2]))) - self.expected_distance
        # Use a simple proportional forward speed (cm -> normalized command).
        velocity = np.clip(
            delt_distance / 300.0,
            self.velocity_low,
            self.velocity_high,
        )
        if self.enable_z:
            delta_z = float(target_pose[2] - pose[2]) - self.expected_z_offset
            vertical = np.clip(
                self.z_sign * delta_z / 300.0,
                self.vertical_low,
                self.vertical_high,
            )
        else:
            # Lock altitude: stable default is restricted to the XY plane.
            vertical = 0.0
        return np.array([velocity, 0.0, vertical, angle], dtype=np.float32)


class SeePointFlyAgent(BaseAgent):
    """
    Composite agent:
    - SPF controls the drone during scouting.
    - CityWalker controls the ground rescuer after handoff.
    """

    def __init__(
        self,
        citywalker_checkpoint_path: str,
        citywalker_config_path: Optional[str] = None,
        spf_config_path: Optional[str] = None,
        device: str = "cuda",
        step_scale: float = -1.0,
        plane_mode: str = "xy",
        bgr_input: bool = True,
        drone_trigger_radius_m: float = 4.0,
        drone_trigger_height_m: float = 10.0,
        drone_max_speed: float = 0.5,
        drone_max_z_speed: float = 0.08,
        drone_max_yaw: float = 1.0,
        drone_cruise_altitude_cm: float = 200.0,
        drone_search_yaw: float = 0.25,
        drone_goal_scale_cm: float = 100.0,
        fine_injured_push_start_cm: float = 250.0,
        fine_stretcher_push_start_cm: float = 350.0,
        near_goal_push_min_speed: float = 75.0,
        near_goal_push_align_angle_deg: float = 12.0,
        near_goal_push_stop_margin_cm: float = 15.0,
        disable_near_goal_push: bool = False,
        citywalker_resolution: Tuple[int, int] = (640, 360),
        **kwargs,
    ):
        self.citywalker = CityWalkerAgent(
            config_path=citywalker_config_path or DEFAULT_CITYWALKER_CONFIG_PATH,
            checkpoint_path=citywalker_checkpoint_path,
            device=device,
            step_scale=step_scale,
            plane_mode=plane_mode,
            bgr_input=bgr_input,
            enable_near_goal_push=not disable_near_goal_push,
            fine_injured_push_start_cm=fine_injured_push_start_cm,
            fine_stretcher_push_start_cm=fine_stretcher_push_start_cm,
            near_goal_push_min_speed=near_goal_push_min_speed,
            near_goal_push_align_angle_deg=near_goal_push_align_angle_deg,
            near_goal_push_stop_margin_cm=near_goal_push_stop_margin_cm,
        )
        self.spf_config_path = os.path.abspath(os.path.expanduser(spf_config_path or DEFAULT_SPF_CONFIG_PATH))
        self.projector: Optional[SimActionProjector] = None
        self.spf_config = self._load_spf_config(self.spf_config_path)
        self.drone_trigger_radius_cm = float(drone_trigger_radius_m) * 100.0
        self.drone_trigger_height_cm = float(drone_trigger_height_m) * 100.0
        self.drone_max_speed = float(drone_max_speed)
        self.drone_max_z_speed = float(drone_max_z_speed)
        self.drone_max_yaw = float(drone_max_yaw)
        self.drone_cruise_altitude_cm = float(drone_cruise_altitude_cm)
        self.drone_search_yaw = float(drone_search_yaw)
        self.drone_goal_scale_cm = float(drone_goal_scale_cm)
        try:
            yaw_sign_raw = float(os.environ.get("SEEPOINTFLY_YAW_SIGN", "-1.0"))
        except ValueError:
            yaw_sign_raw = -1.0
        self.drone_yaw_sign = -1.0 if yaw_sign_raw == 0.0 else yaw_sign_raw
        self.enable_z_control = os.environ.get("SEEPOINTFLY_ENABLE_Z", "0").strip().lower() in ("1", "true", "yes")
        try:
            z_sign_raw = float(os.environ.get("SEEPOINTFLY_Z_SIGN", "1.0"))
        except ValueError:
            z_sign_raw = 1.0
        self.drone_z_sign = 1.0 if z_sign_raw == 0.0 else z_sign_raw
        self.command_loop_delay = max(0.0, float(self.spf_config.get("command_loop_delay", 0.0)))
        self.citywalker_resolution = (
            int(citywalker_resolution[0]),
            int(citywalker_resolution[1]),
        )
        self._drone_action_space = SimDroneActionSpace()
        self.debug_drone_control = os.environ.get("SEEPOINTFLY_DEBUG_DRONE", "0").strip().lower() in ("1", "true", "yes")
        try:
            debug_steps_raw = int(os.environ.get("SEEPOINTFLY_DEBUG_STEPS", "12"))
        except ValueError:
            debug_steps_raw = 12
        self.debug_drone_steps = max(1, debug_steps_raw)
        self._debug_step_counter = 0
        self._debug_prev_goal_dist_xy: Optional[float] = None
        self._debug_prev_yaw: Optional[float] = None
        self._control_prev_goal_dist_xy: Optional[float] = None
        self._distance_worsen_guard_cm = 15.0
        self._yaw_slowdown_ratio = 0.85
        self._forward_slowdown_ratio = 0.35
        self._forward_worsen_ratio = 0.20
        self._drone_tracker = DronePoseTracker3D(
            velocity_high=self.drone_max_speed,
            velocity_low=-self.drone_max_speed,
            vertical_high=self.drone_max_z_speed,
            vertical_low=-self.drone_max_z_speed,
            angle_high=self.drone_max_yaw,
            angle_low=-self.drone_max_yaw,
            yaw_sign=self.drone_yaw_sign,
            enable_z=self.enable_z_control,
            z_sign=self.drone_z_sign,
        )
        self._task_context: Dict[str, Any] = {}
        self._active_phase: Optional[str] = None
        self._handoff_ready = False
        self._drone_step = 0
        self._last_spf_label = ""
        self._last_vlm_time = 0.0
        self._next_vlm_ready_time = 0.0
        self._cached_drone_action = self._fallback_drone_action()
        self._cached_drone_goal_pose: Optional[np.ndarray] = None
        self._cached_spf_info = {
            "action_type": "seepointfly",
            "trajectory": [],
            "spf_instruction": "",
            "spf_label": "uninitialized",
            "spf_reused_previous": False,
            "spf_wait_reason": "uninitialized",
        }
        self.requires_multiagent_env = True
        print(
            "[SeePointFly] Initialized | "
            f"spf_config={self.spf_config_path} trigger_xy={self.drone_trigger_radius_cm:.0f}cm "
            f"trigger_z={self.drone_trigger_height_cm:.0f}cm "
            f"drone_max_speed={self.drone_max_speed:.2f} drone_max_z_speed={self.drone_max_z_speed:.2f} "
            f"drone_max_yaw={self.drone_max_yaw:.2f} "
            f"yaw_sign={self.drone_yaw_sign:.1f} enable_z={self.enable_z_control} z_sign={self.drone_z_sign:.1f} "
            f"command_loop_delay={self.command_loop_delay:.2f}s "
            f"citywalker_resolution={self.citywalker_resolution[0]}x{self.citywalker_resolution[1]}"
        )

    def prepare_episode(self, task_context: Dict[str, Any]) -> None:
        self._task_context = dict(task_context)
        self.citywalker.prepare_episode(task_context)

    def reset(self):
        self.citywalker.reset()
        self._active_phase = None
        self._handoff_ready = False
        self._drone_step = 0
        self._last_spf_label = ""
        self._last_vlm_time = 0.0
        self._next_vlm_ready_time = 0.0
        self._cached_drone_action = self._fallback_drone_action()
        self._cached_drone_goal_pose = None
        self._debug_step_counter = 0
        self._debug_prev_goal_dist_xy = None
        self._debug_prev_yaw = None
        self._control_prev_goal_dist_xy = None
        self._drone_tracker.reset()
        self._cached_spf_info = {
            "action_type": "seepointfly",
            "trajectory": [],
            "spf_instruction": "",
            "spf_label": "reset",
            "spf_reused_previous": False,
            "spf_wait_reason": "reset",
        }

    def on_episode_end(self, success: bool, metrics: EpisodeMetrics):
        self.citywalker.on_episode_end(success, metrics)

    def act(self, observation: np.ndarray, info: Dict) -> Tuple[Any, Dict]:
        phase = info.get("task_phase", "find_injured")
        handoff_event_stage1 = False
        handoff_event_stage2 = False
        if phase != self._active_phase:
            self._active_phase = phase
            self._handoff_ready = False
            self._drone_step = 0
            self._last_vlm_time = 0.0
            self._next_vlm_ready_time = 0.0
            self._cached_drone_action = self._fallback_drone_action()
            self._cached_drone_goal_pose = None
            self._debug_step_counter = 0
            self._debug_prev_goal_dist_xy = None
            self._debug_prev_yaw = None
            self._control_prev_goal_dist_xy = None
            self._drone_tracker.reset()
            self._cached_spf_info = {
                "action_type": "seepointfly",
                "trajectory": [],
                "spf_instruction": "",
                "spf_label": f"phase_switch_{phase}",
                "spf_reused_previous": False,
                "spf_wait_reason": "phase_switch",
            }

        target_pose = self._extract_target_pose(info)
        drone_target_pose = self._select_drone_target_pose(phase, target_pose)
        drone_obs = self._extract_drone_observation(info, observation)
        drone_pose = self._extract_drone_pose(info)

        if not self._handoff_ready and drone_target_pose is not None and drone_pose is not None:
            dist_xy = float(np.linalg.norm(drone_pose[:2] - drone_target_pose[:2]))
            dist_z = abs(float(drone_pose[2]) - float(drone_target_pose[2])) if len(drone_pose) >= 3 and len(drone_target_pose) >= 3 else 0.0
            if dist_xy <= self.drone_trigger_radius_cm and dist_z <= self.drone_trigger_height_cm:
                self._handoff_ready = True
                target_name = "injured" if phase == "find_injured" else "ambulance"
                if phase == "find_injured":
                    handoff_event_stage1 = True
                else:
                    handoff_event_stage2 = True
                level = info.get("level")
                point_id = info.get("point_id")
                episode_id = info.get("episode_id")
                print(
                    f"[DRONE_FOUND_{target_name.upper()}] "
                    f"map={info.get('env_id')} level={level} point={point_id} episode={episode_id} "
                    f"dist_xy={dist_xy:.1f}cm dist_z={dist_z:.1f}cm target={drone_target_pose[:3].tolist()}"
                )
                print(
                    f"[HANDOFF_TO_CITYWALKER] "
                    f"target={target_name} map={info.get('env_id')} level={level} "
                    f"point={point_id} episode={episode_id} coords={target_pose[:3].tolist()}"
                )

        if self._handoff_ready:
            citywalker_obs = self._prepare_citywalker_observation(observation)
            player_action, citywalker_info = self.citywalker.act(citywalker_obs, info)
            drone_action = self._hover_action()
            extra_info = dict(citywalker_info)
            extra_info["control_mode"] = "citywalker"
            extra_info["citywalker_input_resolution"] = list(self.citywalker_resolution)
            # During handoff, keep drone hovering and suppress VLM calls until phase switches.
            extra_info["spf_reused_previous"] = True
            extra_info["spf_wait_reason"] = "handoff_hover_no_vlm"
        else:
            spf_action, spf_info = self._drone_search_action(drone_obs, drone_pose, phase, info)
            player_action = self._idle_ground_action()
            drone_action = spf_action
            extra_info = dict(spf_info)
            extra_info["control_mode"] = "drone_search"

        extra_info["joint_action"] = {
            "player": player_action,
            "drone": drone_action,
        }
        extra_info["drone_stage1_handoff"] = bool(extra_info.get("drone_stage1_handoff", False) or handoff_event_stage1)
        extra_info["drone_stage2_handoff"] = bool(extra_info.get("drone_stage2_handoff", False) or handoff_event_stage2)
        extra_info["handoff_ready"] = self._handoff_ready
        extra_info["task_phase"] = phase
        if self.debug_drone_control and not self._handoff_ready:
            ground_pose = self._extract_ground_pose(info)
            self._debug_drone_step(phase, drone_pose, ground_pose, drone_action, extra_info)
        return player_action, extra_info

    def _extract_target_pose(self, info: Dict) -> Optional[np.ndarray]:
        target_pose = info.get("target_pose")
        if target_pose is None:
            return None
        return np.asarray(target_pose, dtype=np.float32).reshape(-1)

    def _select_drone_target_pose(self, phase: str, target_pose: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if phase == "find_injured":
            return target_pose
        ambulance_pose = self._task_context.get("ambulance_pose")
        if ambulance_pose is None:
            return target_pose
        return np.asarray(ambulance_pose, dtype=np.float32).reshape(-1)

    def _extract_drone_observation(self, info: Dict, fallback_obs: np.ndarray) -> np.ndarray:
        drone_obs = info.get("drone_observation")
        if drone_obs is None:
            return fallback_obs
        return np.asarray(drone_obs)

    def _extract_drone_pose(self, info: Dict) -> Optional[np.ndarray]:
        drone_pose = info.get("drone_pose")
        if drone_pose is None:
            return None
        return np.asarray(drone_pose, dtype=np.float32).reshape(-1)

    def _extract_ground_pose(self, info: Dict) -> Optional[np.ndarray]:
        ground_pose = info.get("ground_pose")
        if ground_pose is None:
            return None
        return np.asarray(ground_pose, dtype=np.float32).reshape(-1)

    def _idle_ground_action(self):
        return (np.array([0.0, 0.0], dtype=np.float32), 0, 0)

    def _hover_action(self) -> np.ndarray:
        return np.zeros(4, dtype=np.float32)

    def _drone_search_action(
        self,
        observation: np.ndarray,
        drone_pose: Optional[np.ndarray],
        phase: str,
        info: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        self._drone_step += 1
        instruction = self._build_instruction(phase, info)
        now = time.time()
        if self._last_vlm_time > 0.0 and now < self._next_vlm_ready_time:
            reuse_info = dict(self._cached_spf_info)
            reuse_info["spf_instruction"] = instruction
            reuse_info["spf_reused_previous"] = True
            reuse_info["spf_reuse_remaining_s"] = float(self._next_vlm_ready_time - now)
            reuse_info["spf_wait_reason"] = "queue_or_delay"
            if drone_pose is not None and self._cached_drone_goal_pose is not None:
                tracked_action = self._drone_tracker.act(drone_pose, self._cached_drone_goal_pose)
                self._cached_drone_action = np.array(tracked_action, dtype=np.float32, copy=True)
                reuse_info["spf_goal_pose"] = self._cached_drone_goal_pose[:3].astype(float).tolist()
                return tracked_action, reuse_info
            return np.array(self._cached_drone_action, dtype=np.float32, copy=True), reuse_info

        projector = self._get_projector(observation)
        actions = projector.get_vlm_points(observation, instruction)
        self._last_vlm_time = time.time()
        if not actions or actions[0] is None:
            fallback_action = self._fallback_drone_action()
            fallback_info = {
                "action_type": "seepointfly",
                "trajectory": [],
                "spf_instruction": instruction,
                "spf_label": "fallback_scan",
                "spf_reused_previous": False,
                "spf_wait_reason": "fallback_scan",
            }
            self._next_vlm_ready_time = self._last_vlm_time + self.command_loop_delay
            self._cached_drone_action = np.array(fallback_action, dtype=np.float32, copy=True)
            self._cached_drone_goal_pose = None
            self._cached_spf_info = dict(fallback_info)
            return fallback_action, fallback_info

        action = actions[0]
        if not self._is_target_visible(action):
            scan_action = self._fallback_drone_action()
            no_target_info = {
                "action_type": "seepointfly",
                "trajectory": [],
                "spf_instruction": instruction,
                "spf_label": "no_target_scan",
                "spf_vlm_label": str(getattr(action, "vlm_label", "")).strip(),
                "spf_reused_previous": False,
                "spf_wait_reason": "no_target_hold",
            }
            self._next_vlm_ready_time = self._last_vlm_time + self.command_loop_delay
            self._cached_drone_action = np.array(scan_action, dtype=np.float32, copy=True)
            self._cached_drone_goal_pose = None
            self._cached_spf_info = dict(no_target_info)
            self._control_prev_goal_dist_xy = None
            return scan_action, no_target_info
        goal_pose = self._action_point_to_goal_pose(drone_pose, action)
        if drone_pose is not None and goal_pose is not None:
            drone_action = self._drone_tracker.act(drone_pose, goal_pose)
        else:
            drone_action = self._action_point_to_env_action(action)
        self._last_spf_label = getattr(action, "action_type", "move")
        estimated_queue_wait_s = self._estimate_action_queue_wait(action)
        self._next_vlm_ready_time = self._last_vlm_time + estimated_queue_wait_s + self.command_loop_delay
        extra = {
            "action_type": "seepointfly",
            "trajectory": [[float(action.dy), float(action.dx), float(action.dz)]],
            "spf_instruction": instruction,
            "spf_label": getattr(action, "action_type", "move"),
            "spf_vlm_label": str(getattr(action, "vlm_label", "")).strip(),
            "spf_screen_point": [float(action.screen_x), float(action.screen_y)],
            "spf_vector": [float(action.dx), float(action.dy), float(action.dz)],
            "spf_reused_previous": False,
            "spf_wait_reason": "fresh_vlm_call",
            "spf_estimated_queue_wait_s": float(estimated_queue_wait_s),
        }
        drone_action = self._apply_planar_stability_guard(drone_action, drone_pose, goal_pose, extra)
        self._cached_drone_action = np.array(drone_action, dtype=np.float32, copy=True)
        self._cached_drone_goal_pose = None if goal_pose is None else np.array(goal_pose, dtype=np.float32, copy=True)
        if self._cached_drone_goal_pose is not None:
            extra["spf_goal_pose"] = self._cached_drone_goal_pose[:3].astype(float).tolist()
        self._cached_spf_info = dict(extra)
        return drone_action, extra

    def _load_spf_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                return {}
            return data
        except Exception as e:
            print(f"[SeePointFly] Failed to load SPF config {config_path}: {e}")
            return {}

    def _get_projector(self, observation: np.ndarray) -> SimActionProjector:
        h, w = observation.shape[:2]
        if (
            self.projector is None
            or self.projector.image_width != w
            or self.projector.image_height != h
        ):
            self.projector = SimActionProjector(
                image_width=w,
                image_height=h,
                adaptive_mode=False,
                config_path=self.spf_config_path,
            )
        return self.projector

    def _build_instruction(self, phase: str, info: Dict) -> str:
        reference_text = str(info.get("reference_text", "")).strip()
        if self.enable_z_control:
            planar_note = (
                # " You may adjust altitude if it helps keep the target centered and visible. "
                "Do not fly too high; the target must remain clearly visible from the drone view."
            )
        else:
            planar_note = (
                " Keep the drone at its current altitude. Do not move up or down. "
                "Only adjust horizontal position and viewing direction. "
                "If the target appears above or below in the image, keep altitude unchanged."
            )
        local_search_note = (
            " Search locally around the current area. Do not keep flying far away in one direction. "
            "If the target is not clearly visible, choose a nearby scan point near the center of the image "
            "instead of a far-away landmark."
        )
        if phase == "find_injured":
            prefix = "You need to find a injured person lying on the ground, who is located in"
            fallback = "the current aerial view"
            context = reference_text if reference_text else fallback
            return f"{prefix} {context}.{local_search_note}{planar_note}"
        else:
            return (
                "You need to find the ambulance in the current aerial view. "
                "The rescue stretcher is expected to be near the ambulance. "
                "Prioritize the nearest visible ambulance or ambulance-like vehicle. "
                "If visible, move close enough to observe the ambulance clearly, then keep hovering."
                f"{local_search_note}{planar_note}"
            )

    def _prepare_citywalker_observation(self, observation: np.ndarray) -> np.ndarray:
        frame = np.asarray(observation)
        if frame.ndim == 4:
            frame = frame[0]
        target_w, target_h = self.citywalker_resolution
        h, w = frame.shape[:2]
        if (w, h) == (target_w, target_h):
            return frame
        interpolation = cv2.INTER_AREA if (target_w <= w and target_h <= h) else cv2.INTER_LINEAR
        return cv2.resize(frame, (target_w, target_h), interpolation=interpolation)

    def _estimate_action_queue_wait(self, action_point) -> float:
        commands = self._drone_action_space.action_to_commands(action_point)
        total_ms = 0.0
        for _, duration_ms in commands:
            total_ms += float(duration_ms)
        if commands:
            total_ms += 100.0 * len(commands)
        return total_ms / 1000.0

    def _action_point_to_goal_pose(
        self,
        drone_pose: Optional[np.ndarray],
        action_point,
    ) -> Optional[np.ndarray]:
        if drone_pose is None:
            return None
        pose = np.asarray(drone_pose, dtype=np.float32).reshape(-1).copy()
        yaw_rad = np.deg2rad(float(pose[4]))
        dx = float(action_point.dx) * self.drone_goal_scale_cm
        dy = float(action_point.dy) * self.drone_goal_scale_cm
        forward = np.array([np.cos(yaw_rad), np.sin(yaw_rad)], dtype=np.float32)
        right = np.array([np.sin(yaw_rad), -np.cos(yaw_rad)], dtype=np.float32)
        delta_xy = forward * dy + right * dx
        pose[0] += float(delta_xy[0])
        pose[1] += float(delta_xy[1])
        if self.enable_z_control:
            pose[2] += float(action_point.dz) * self.drone_goal_scale_cm
        else:
            # Keep target altitude unchanged (XY-only tracking).
            pose[2] = float(pose[2])
        return pose

    def _action_point_to_env_action(self, action_point) -> np.ndarray:
        dx = float(action_point.dx)
        dy = float(action_point.dy)
        yaw = float(np.degrees(np.arctan2(dx, dy + 1e-6)))
        yaw_cmd = np.clip(yaw / 45.0, -self.drone_max_yaw, self.drone_max_yaw)
        forward_cmd = np.clip(dy / 2.0, -self.drone_max_speed, self.drone_max_speed)
        lateral_cmd = np.clip(dx / 2.0, -self.drone_max_speed, self.drone_max_speed)
        if self.enable_z_control:
            vertical_cmd = np.clip(
                self.drone_z_sign * float(action_point.dz) / 2.0,
                -self.drone_max_z_speed,
                self.drone_max_z_speed,
            )
        else:
            vertical_cmd = 0.0
        return np.array([forward_cmd, lateral_cmd, vertical_cmd, yaw_cmd], dtype=np.float32)

    def _fallback_drone_action(self) -> np.ndarray:
        # Avoid pure in-place spinning when VLM is temporarily unstable.
        scan_forward = min(self.drone_max_speed, 0.08)
        return np.array([scan_forward, 0.0, 0.0, self.drone_search_yaw], dtype=np.float32)

    def _is_target_visible(self, action_point) -> bool:
        visible = getattr(action_point, "target_visible", None)
        if visible is None:
            return True
        return bool(visible)

    def _apply_planar_stability_guard(
        self,
        drone_action: np.ndarray,
        drone_pose: Optional[np.ndarray],
        goal_pose: Optional[np.ndarray],
        extra_info: Dict[str, Any],
    ) -> np.ndarray:
        action_arr = np.asarray(drone_action, dtype=np.float32).reshape(-1).copy()
        if action_arr.size < 4:
            return action_arr.astype(np.float32)
        forward_cmd = float(action_arr[0])
        yaw_cmd = float(action_arr[3])
        turning_slowdown = False
        worsening_slowdown = False
        if abs(yaw_cmd) >= self.drone_max_yaw * self._yaw_slowdown_ratio and forward_cmd > 0.0:
            forward_limit = self.drone_max_speed * self._forward_slowdown_ratio
            action_arr[0] = min(forward_cmd, forward_limit)
            forward_cmd = float(action_arr[0])
            turning_slowdown = True

        goal_dist_xy = None
        goal_dist_xy_change = None
        if drone_pose is not None and goal_pose is not None and len(drone_pose) >= 2 and len(goal_pose) >= 2:
            goal_dist_xy = float(
                np.linalg.norm(
                    np.asarray(drone_pose[:2], dtype=np.float32) - np.asarray(goal_pose[:2], dtype=np.float32)
                )
            )
            if self._control_prev_goal_dist_xy is not None:
                goal_dist_xy_change = float(goal_dist_xy - self._control_prev_goal_dist_xy)
                if (
                    goal_dist_xy_change > self._distance_worsen_guard_cm
                    and abs(yaw_cmd) >= self.drone_max_yaw * 0.6
                    and forward_cmd > 0.0
                ):
                    worsen_limit = self.drone_max_speed * self._forward_worsen_ratio
                    action_arr[0] = min(float(action_arr[0]), worsen_limit)
                    worsening_slowdown = True
            self._control_prev_goal_dist_xy = goal_dist_xy
        else:
            self._control_prev_goal_dist_xy = None

        extra_info["spf_control_slowdown_turning"] = turning_slowdown
        extra_info["spf_control_slowdown_worsening"] = worsening_slowdown
        extra_info["spf_control_goal_dist_xy_cm"] = goal_dist_xy
        extra_info["spf_control_goal_dist_xy_change_cm"] = goal_dist_xy_change
        return action_arr.astype(np.float32)

    def _debug_drone_step(
        self,
        phase: str,
        drone_pose: Optional[np.ndarray],
        ground_pose: Optional[np.ndarray],
        drone_action: np.ndarray,
        extra_info: Dict[str, Any],
    ) -> None:
        self._debug_step_counter += 1
        if self._debug_step_counter > self.debug_drone_steps:
            return
        drone_z = None if drone_pose is None else float(drone_pose[2])
        ground_z = None if ground_pose is None else float(ground_pose[2])
        dz_rel = None if (drone_z is None or ground_z is None) else (drone_z - ground_z)
        action_arr = np.asarray(drone_action, dtype=np.float32).reshape(-1)
        cmd_forward = float(action_arr[0]) if action_arr.size > 0 else None
        cmd_z = float(action_arr[2]) if action_arr.size > 2 else None
        cmd_yaw = float(action_arr[3]) if action_arr.size > 3 else None
        drone_yaw = None if drone_pose is None or len(drone_pose) < 5 else float(drone_pose[4])
        yaw_change = None
        if drone_yaw is not None and self._debug_prev_yaw is not None:
            yaw_change = float(drone_yaw - self._debug_prev_yaw)
        self._debug_prev_yaw = drone_yaw
        spf_vector = extra_info.get("spf_vector")
        spf_vector_z = None if not spf_vector or len(spf_vector) < 3 else float(spf_vector[2])
        goal_pose = extra_info.get("spf_goal_pose")
        goal_z = None if not goal_pose or len(goal_pose) < 3 else float(goal_pose[2])
        goal_dist_xy = None
        goal_dist_xy_change = None
        if goal_pose is not None and drone_pose is not None and len(goal_pose) >= 2 and len(drone_pose) >= 2:
            goal_dist_xy = float(np.linalg.norm(np.asarray(drone_pose[:2], dtype=np.float32) - np.asarray(goal_pose[:2], dtype=np.float32)))
            if self._debug_prev_goal_dist_xy is not None:
                goal_dist_xy_change = float(goal_dist_xy - self._debug_prev_goal_dist_xy)
            self._debug_prev_goal_dist_xy = goal_dist_xy
        print(
            f"[DRONE_DEBUG] phase={phase} step={self._debug_step_counter}/{self.debug_drone_steps} "
            f"reused={extra_info.get('spf_reused_previous', False)} reason={extra_info.get('spf_wait_reason', '-')}" 
            f" drone_z={drone_z} ground_z={ground_z} rel_z={dz_rel} "
            f"drone_yaw={drone_yaw} yaw_change={yaw_change} "
            f"goal_dist_xy={goal_dist_xy} goal_dist_xy_change={goal_dist_xy_change} "
            f"spf_vec_z={spf_vector_z} goal_z={goal_z} "
            f"cmd_forward={cmd_forward} cmd_yaw={cmd_yaw} cmd_z={cmd_z} "
            f"slow_turn={extra_info.get('spf_control_slowdown_turning', False)} "
            f"slow_worsen={extra_info.get('spf_control_slowdown_worsening', False)} "
            f"vlm_label={extra_info.get('spf_vlm_label', '')}"
        )
