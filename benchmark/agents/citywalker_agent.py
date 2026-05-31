from __future__ import annotations

import os
import sys
import argparse
import __main__
from collections import deque
from contextlib import nullcontext
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from agents.agent_base import BaseAgent


AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(AGENT_DIR)
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)
CITYWALKER_ROOT = os.path.join(PROJECT_ROOT, "baseline_model", "CityWalker")
DEFAULT_CONFIG_PATH = os.path.join(CITYWALKER_ROOT, "config", "citywalk_2000hr.yaml")

if CITYWALKER_ROOT not in sys.path:
    sys.path.insert(0, CITYWALKER_ROOT)

from pl_modules.citywalker_feat_module import CityWalkerFeatModule  # noqa: E402
from pl_modules.citywalker_module import CityWalkerModule  # noqa: E402


class DictNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                setattr(self, key, DictNamespace(**value))
            else:
                setattr(self, key, value)


# CityWalker checkpoints may serialize this symbol as "__main__.DictNamespace".
if not hasattr(__main__, "DictNamespace"):
    setattr(__main__, "DictNamespace", DictNamespace)


def load_config(config_path: str) -> DictNamespace:
    with open(config_path, "r", encoding="utf-8") as f:
        return DictNamespace(**yaml.safe_load(f))


class CityWalkerAgent(BaseAgent):
    """
    CityWalker benchmark adapter.

    The model consumes:
    - 5 recent RGB frames
    - 5 recent agent positions in the current local frame
    - 1 current target position in the current local frame
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        step_scale: float = -1.0,
        plane_mode: str = "xy",
        bgr_input: bool = True,
        waypoint_index: int = 5,#默认取未来预测的第几个点
        max_turn_deg: float = 30.0,
        max_speed: float = 100.0,
        speed_gain: float = 1.8,
        turn_gain: float = 1.25,
        turn_only_threshold_deg: float = 120.0,
        stop_distance_cm: float = 10.0,
        enable_near_goal_push: bool = True,
        fine_injured_push_start_cm: float = 250.0,
        fine_stretcher_push_start_cm: float = 350.0,
        near_goal_push_min_speed: float = 12.0,
        near_goal_push_align_angle_deg: float = 12.0,
        near_goal_push_stop_margin_cm: float = 15.0,
    ):
        self.config_path = os.path.abspath(os.path.expanduser(config_path or DEFAULT_CONFIG_PATH))
        if checkpoint_path is None:
            raise ValueError("CityWalkerAgent requires checkpoint_path")
        self.checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"CityWalker config not found: {self.config_path}")
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"CityWalker checkpoint not found: {self.checkpoint_path}")

        self.cfg = load_config(self.config_path)
        self.context_size = int(self.cfg.model.obs_encoder.context_size)
        self.pred_horizon = int(self.cfg.model.decoder.len_traj_pred)
        self.crop_h, self.crop_w = [int(v) for v in self.cfg.model.obs_encoder.crop]
        self.resize_h, self.resize_w = [int(v) for v in self.cfg.model.obs_encoder.resize]
        self.device = self._resolve_device(device)
        self.plane_mode = plane_mode
        self.plane_indices = (0, 1) if plane_mode == "xy" else (0, 2)
        self.bgr_input = bool(bgr_input)
        self.default_step_scale = 80.0
        self.step_scale = float(step_scale)
        self.waypoint_index = max(0, min(int(waypoint_index), self.pred_horizon - 1))
        self.max_turn_deg = float(max_turn_deg)
        self.max_speed = float(max_speed)
        self.speed_gain = float(speed_gain)
        self.turn_gain = float(turn_gain)
        self.turn_only_threshold_deg = float(turn_only_threshold_deg)
        self.stop_distance_cm = float(stop_distance_cm)
        self.enable_near_goal_push = bool(enable_near_goal_push)
        self.fine_injured_push_start_cm = float(fine_injured_push_start_cm)
        self.fine_stretcher_push_start_cm = float(fine_stretcher_push_start_cm)
        self.near_goal_push_min_speed = float(near_goal_push_min_speed)
        self.near_goal_push_align_angle_deg = float(near_goal_push_align_angle_deg)
        self.near_goal_push_stop_margin_cm = float(near_goal_push_stop_margin_cm)

        self.model = self._load_model()

        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=self.context_size)
        self.pose_buffer: Deque[np.ndarray] = deque(maxlen=self.context_size)
        self.task_context: Dict[str, Any] = {}
        self.current_goal_pose: Optional[np.ndarray] = None
        self._infer_step = 0

        print(
            "[CityWalker] Loaded | "
            f"config={self.config_path} ckpt={self.checkpoint_path} "
            f"device={self.device} crop={self.crop_h}x{self.crop_w} "
            f"resize={self.resize_h}x{self.resize_w} plane={self.plane_mode}"
        )

    def _resolve_device(self, device: str) -> torch.device:
        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[CityWalker] CUDA unavailable, fallback to CPU")
            return torch.device("cpu")
        return torch.device(device)

    def _load_model(self):
        model_type = self.cfg.model.type
        if model_type == "citywalker_feat":
            module = CityWalkerFeatModule.load_from_checkpoint(
                self.checkpoint_path,
                cfg=self.cfg,
                map_location=self.device,
                weights_only=False,
            )
        elif model_type == "citywalker":
            module = CityWalkerModule.load_from_checkpoint(
                self.checkpoint_path,
                cfg=self.cfg,
                map_location=self.device,
                weights_only=False,
            )
        else:
            raise ValueError(f"Unsupported CityWalker model type: {model_type}")

        module.eval()
        module.to(self.device)
        policy = module.model
        policy.eval()
        policy.to(self.device)
        return policy

    def prepare_episode(self, task_context: Dict[str, Any]) -> None:
        self.task_context = dict(task_context)
        self.current_goal_pose = self._coerce_pose(task_context.get("injured_pose"))

    def reset(self):
        self.frame_buffer.clear()
        self.pose_buffer.clear()
        self._infer_step = 0

    def act(self, observation: np.ndarray, info: Dict) -> Tuple[Any, Dict]:
        current_pose = self._extract_current_pose(info)
        target_pose = self._extract_target_pose(info)
        frame = self._prepare_frame(observation)

        self.frame_buffer.append(frame)
        self.pose_buffer.append(current_pose)
        self._warmup_buffers(frame, current_pose)

        obs_tensor = torch.from_numpy(np.stack(list(self.frame_buffer), axis=0)).unsqueeze(0).to(self.device)
        hist_local, target_local = self._build_local_inputs(list(self.pose_buffer), target_pose)
        scale = self._estimate_step_scale(list(self.pose_buffer))
        input_positions = np.concatenate([hist_local, target_local[None, :]], axis=0) / scale
        cord_tensor = torch.from_numpy(input_positions.astype(np.float32)).unsqueeze(0).to(self.device)

        use_amp = self.device.type == "cuda"
        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_amp
            else nullcontext()
        )
        with torch.no_grad():
            with amp_ctx:
                if self.cfg.model.type == "citywalker_feat":
                    wp_pred, arrive_pred, _, _ = self.model(obs_tensor, cord_tensor, None)
                else:
                    output = self.model(obs_tensor, cord_tensor)
                    wp_pred, arrive_pred = output[0], output[1]

        wp_pred = wp_pred[0].detach().cpu().numpy() * scale
        arrive_prob = float(torch.sigmoid(arrive_pred[0, 0]).item())
        chosen_wp = wp_pred[self.waypoint_index]
        action = self._waypoint_to_action(chosen_wp, target_local, arrive_prob)
        action, push_info = self._apply_near_goal_push(action, target_local, info)
        self._infer_step += 1
        move_action = action[0]
        push_text = ""
        if push_info["applied"]:
            push_text = (
                f" push=on phase={push_info['phase']} "
                f"dist={push_info['distance_cm']:.0f} floor={push_info['min_speed']:.1f}"
            )
        print(
            f"[CityWalkerAgent] Step {self._infer_step} "
            f"phase={info.get('task_phase')} "
            f"arrive_prob={arrive_prob:.3f} "
            f"target_local={target_local.tolist()} "
            f"chosen_wp={chosen_wp.tolist()} "
            f"move=[turn={float(move_action[0]):.2f}, speed={float(move_action[1]):.2f}]"
            f"{push_text}"
        )

        extra_info = {
            "action_type": "citywalker",
            "trajectory": wp_pred.tolist(),
            "waypoint": chosen_wp.tolist(),
            "arrived_prob": arrive_prob,
            "step_scale": float(scale),
            "target_local": target_local.tolist(),
            "task_phase": info.get("task_phase"),
            "reference_text": info.get("reference_text", ""),
            "near_goal_push": push_info,
        }
        return action, extra_info

    def _warmup_buffers(self, frame: np.ndarray, current_pose: np.ndarray) -> None:
        while len(self.frame_buffer) < self.context_size:
            self.frame_buffer.appendleft(frame.copy())
        while len(self.pose_buffer) < self.context_size:
            self.pose_buffer.appendleft(current_pose.copy())

    def _prepare_frame(self, observation: np.ndarray) -> np.ndarray:
        frame = np.asarray(observation)
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim != 3 or frame.shape[-1] < 3:
            raise ValueError(f"Unexpected observation shape for CityWalker: {frame.shape}")
        h, w = frame.shape[:2]
        if h < self.crop_h or w < self.crop_w:
            raise ValueError(
                "Observation resolution is smaller than CityWalker crop size: "
                f"got {w}x{h}, need at least {self.crop_w}x{self.crop_h}. "
                "Adjust benchmark --resolution."
            )

        frame = frame[..., :3]
        if self.bgr_input:
            frame = frame[..., ::-1]
        frame = np.ascontiguousarray(frame.transpose(2, 0, 1)).astype(np.float32)
        if frame.max() > 1.0:
            frame /= 255.0
        return frame

    def _extract_current_pose(self, info: Dict[str, Any]) -> np.ndarray:
        # step() info commonly provides "Pose", while reset() info may only have "pose".
        # Also, "Pose" can be an empty placeholder list before the first step.
        candidates = [info.get("Pose"), info.get("pose"), self.task_context.get("agent_pose")]
        for pose in candidates:
            if pose is None:
                continue
            pose_array = np.asarray(pose, dtype=np.float32)
            if pose_array.size == 0:
                continue
            if pose_array.ndim > 1:
                pose_array = pose_array[0]
            if pose_array.size in (6, 7):
                return pose_array

        raise ValueError("CityWalkerAgent could not find current pose in info/task_context")

    def _extract_target_pose(self, info: Dict[str, Any]) -> np.ndarray:
        target_pose = info.get("target_pose")
        if target_pose is not None:
            self.current_goal_pose = self._coerce_pose(target_pose)
            return self.current_goal_pose

        if info.get("picked", False):
            self.current_goal_pose = self._coerce_pose(self.task_context.get("stretcher_pose"))
        else:
            self.current_goal_pose = self._coerce_pose(self.task_context.get("injured_pose"))

        if self.current_goal_pose is None:
            raise ValueError("CityWalkerAgent could not determine target_pose")
        return self.current_goal_pose

    def _build_local_inputs(
        self,
        pose_history: Sequence[np.ndarray],
        target_pose: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if pose_history[-1].size == 6 and target_pose.size == 6:
            current_pose = pose_history[-1]
            hist_positions = np.stack(
                [self._transform_pose_6d_local(p, current_pose) for p in pose_history],
                axis=0,
            ).astype(np.float32)
            target_local = self._transform_pose_6d_local(target_pose, current_pose).astype(np.float32)
            return hist_positions, target_local

        current_pose = pose_history[-1]
        current_inv = np.linalg.inv(self._pose_to_matrix(current_pose))
        pose_mats = np.stack([self._pose_to_matrix(p) for p in pose_history], axis=0)
        transformed = current_inv[np.newaxis, :, :] @ pose_mats
        hist_positions = transformed[:, :3, 3][:, self.plane_indices].astype(np.float32)

        target_mat = self._pose_to_matrix(target_pose)
        target_local = (current_inv @ target_mat)[:3, 3][list(self.plane_indices)].astype(np.float32)
        return hist_positions, target_local

    def _estimate_step_scale(self, pose_history: Sequence[np.ndarray]) -> float:
        if self.step_scale > 0:
            return self.step_scale

        if len(pose_history) < 2:
            return self.default_step_scale

        positions = np.stack([np.asarray(p)[:3] for p in pose_history], axis=0)
        deltas = np.diff(positions[:, self.plane_indices], axis=0)
        if len(deltas) == 0:
            return self.default_step_scale

        norms = np.linalg.norm(deltas, axis=1)
        norms = norms[np.isfinite(norms)]
        norms = norms[norms > 1e-6]
        if norms.size == 0:
            return self.default_step_scale
        return float(np.clip(norms.mean(), 10.0, 200.0))

    def _waypoint_to_action(
        self,
        waypoint_local: np.ndarray,
        target_local: np.ndarray,
        arrive_prob: float,
    ):
        wp_forward = float(waypoint_local[0])
        goal_forward = float(target_local[0])
        goal_lateral = float(target_local[1])
        # Pure navigation mode:
        # - always steer by the actual target direction
        # - linearly map the predicted waypoint forward component to env speed
        angle_deg = float(np.degrees(np.arctan2(goal_lateral, goal_forward + 1e-6)))

        move = np.zeros(2, dtype=np.float32)
        move[0] = float(np.clip(angle_deg * self.turn_gain, -self.max_turn_deg, self.max_turn_deg))
        move[1] = float(np.clip(wp_forward * self.speed_gain, -self.max_speed, self.max_speed))

        head_action = 0
        anim_action = 0
        return (move, head_action, anim_action)

    def _apply_near_goal_push(
        self,
        action: Tuple[np.ndarray, int, int],
        target_local: np.ndarray,
        info: Dict[str, Any],
    ) -> Tuple[Tuple[np.ndarray, int, int], Dict[str, Any]]:
        move_action, head_action, anim_action = action
        push_info = {
            "applied": False,
            "phase": info.get("task_phase"),
            "distance_cm": float(np.linalg.norm(target_local)),
            "min_speed": 0.0,
        }
        if not self.enable_near_goal_push:
            return action, push_info

        state = info.get("state_machine_state")
        if state in {"RESCUE_INJURED", "PLACE_ON_STRETCHER", "WAIT_ENV_CONFIRM", "COMPLETED", "FAILED"}:
            return action, push_info

        phase = info.get("task_phase")
        if phase not in {"find_injured", "find_stretcher"}:
            return action, push_info

        target_forward = float(target_local[0])
        target_lateral = float(target_local[1])
        if target_forward <= 0.0:
            return action, push_info

        angle_deg = abs(float(np.degrees(np.arctan2(target_lateral, target_forward + 1e-6))))
        if angle_deg > self.near_goal_push_align_angle_deg:
            return action, push_info

        if phase == "find_injured":
            trigger_distance = self.fine_injured_push_start_cm
            gate_distance = float(info.get("rescue_distance", 0.0))
        else:
            trigger_distance = self.fine_stretcher_push_start_cm
            gate_distance = float(info.get("place_distance", 0.0))
        if gate_distance <= 0.0:
            return action, push_info

        distance_cm = float(np.linalg.norm(target_local))
        stop_distance = gate_distance + self.near_goal_push_stop_margin_cm
        if distance_cm > trigger_distance or distance_cm <= stop_distance:
            return action, push_info

        window = max(trigger_distance - stop_distance, 1e-6)
        ratio = float(np.clip((distance_cm - stop_distance) / window, 0.0, 1.0))
        min_speed = max(4.0, self.near_goal_push_min_speed * (0.5 + 0.5 * ratio))
        push_info["min_speed"] = min_speed
        if float(move_action[1]) >= min_speed:
            return action, push_info

        move_action = np.array(move_action, dtype=np.float32, copy=True)
        move_action[1] = float(np.clip(min_speed, 0.0, self.max_speed))
        push_info["applied"] = True
        return (move_action, head_action, anim_action), push_info

    def _transform_pose_6d_local(self, pose: np.ndarray, current_pose: np.ndarray) -> np.ndarray:
        pos = np.asarray(pose, dtype=np.float32)[:3]
        cur = np.asarray(current_pose, dtype=np.float32)[:3]
        delta = pos[list(self.plane_indices)] - cur[list(self.plane_indices)]
        yaw_deg = float(np.asarray(current_pose, dtype=np.float32)[4])
        yaw = np.deg2rad(yaw_deg)
        rot = np.array(
            [[np.cos(-yaw), -np.sin(-yaw)], [np.sin(-yaw), np.cos(-yaw)]],
            dtype=np.float32,
        )
        return rot @ delta.astype(np.float32)

    def _pose_to_matrix(self, pose: Sequence[float]) -> np.ndarray:
        pose = np.asarray(pose, dtype=np.float32).reshape(-1)
        if pose.size == 7:
            position = pose[:3]
            rotation = R.from_quat(pose[3:])
        elif pose.size == 6:
            position = pose[:3]
            rotation = R.from_rotvec(pose[3:])
        else:
            raise ValueError(f"Unsupported pose size for CityWalkerAgent: {pose.size}")

        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation.as_matrix().astype(np.float32)
        matrix[:3, 3] = position.astype(np.float32)
        return matrix

    def _coerce_pose(self, pose: Optional[Sequence[float]]) -> Optional[np.ndarray]:
        if pose is None:
            return None
        return np.asarray(pose, dtype=np.float32).reshape(-1)
