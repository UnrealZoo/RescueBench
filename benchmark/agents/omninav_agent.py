from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BENCHMARK_DIR)
DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "baseline_model", "OmniNav", "ckpt", "OmniNav"
)

if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)

from agents.agent_base import BaseAgent
from core.metrics import EpisodeMetrics

try:
    import torch
except Exception as exc:  # pragma: no cover - import error is reported clearly at runtime
    torch = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class OmniNavAgent(BaseAgent):
    """
    OmniNav 适配器。

    - inference_mode='waypoint'（默认）：对齐 infer_ovon/agent/waypoint_agent_ovon.py
      三路 RGB + 历史帧队列 + habitat 格式 pose + model.forward(action_former=True)。
    - inference_mode='text'：旧版文本生成 FORWARD/LEFT/RIGHT/STOP（便于对照实验）。
    """

    DEFAULT_PHASE1_INSTRUCTION = (
        "Find the injured person lying on the ground and move close enough for rescue."
    )
    DEFAULT_PHASE2_INSTRUCTION = (
        "You are carrying an injured person. Find the ambulance stretcher and move close "
        "enough to place the person down."
    )

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        bgr_input: bool = True,
        inference_mode: str = "waypoint",
        attn_implementation: str = "flash_attention_2",
        multi_view: bool = True,
        multi_view_angles: Sequence[float] = (-45.0, 0.0, 45.0),
        multi_view_settle_time: float = 0.02,
        forward_velocity: float = 100.0,
        turn_angle: float = 20.0,
        turn_velocity: float = 0.0,
        waypoint_dist_scale: float = 100.0,
        waypoint_speed_floor: float = 70.0,
        waypoint_max_turn_deg: float = 30.0,
        waypoint_horizon_index: int = 3,
        max_new_tokens: int = 8,
        do_sample: bool = False,
        temperature: float = 0.0,
        verbose: bool = True,
        log_waypoint_inference: bool = False,
    ):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "OmniNavAgent requires PyTorch (torch) in the current environment. "
                f"Original import error: {_IMPORT_ERROR}"
            )

        self.model_path = os.path.abspath(os.path.expanduser(model_path or DEFAULT_MODEL_PATH))
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"OmniNav checkpoint directory not found: {self.model_path}")

        self.device = self._resolve_device(device)
        self.bgr_input = bool(bgr_input)
        self.inference_mode = (inference_mode or "waypoint").strip().lower()
        if self.inference_mode not in ("waypoint", "text"):
            raise ValueError("inference_mode must be 'waypoint' or 'text'")

        self.multi_view = bool(multi_view)
        self.multi_view_angles = tuple(float(angle) for angle in multi_view_angles)
        if len(self.multi_view_angles) == 0:
            raise ValueError("multi_view_angles must not be empty")
        self.multi_view_settle_time = max(0.0, float(multi_view_settle_time))
        self.forward_velocity = float(forward_velocity)
        self.turn_angle = float(turn_angle)
        self.turn_velocity = float(turn_velocity)
        self.waypoint_dist_scale = float(waypoint_dist_scale)
        self.waypoint_speed_floor = float(waypoint_speed_floor)
        self.waypoint_max_turn_deg = float(abs(waypoint_max_turn_deg))
        if self.waypoint_speed_floor > self.forward_velocity:
            raise ValueError("waypoint_speed_floor must be <= forward_velocity (速度上限)")
        whi = int(waypoint_horizon_index)
        if whi < 0 or whi > 4:
            raise ValueError("waypoint_horizon_index must be in [0, 4] (模型共 5 个 horizon 点)")
        self.waypoint_horizon_index = whi
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.verbose = bool(verbose)
        self.log_waypoint_inference = bool(log_waypoint_inference)

        self.processor = None
        self.model = None
        self._ovon_model = None
        self._ovon_history = None

        if self.inference_mode == "waypoint":
            from .omninav_waypoint_model import OmniNavOVONHistory, QwenOVONModel

            self._ovon_history = OmniNavOVONHistory()
            try:
                self._ovon_model = QwenOVONModel(
                    self.model_path, self.device, attn_impl=attn_implementation
                )
            except Exception as first_exc:
                if attn_implementation == "flash_attention_2":
                    self._log(f"flash_attention_2 不可用，回退 sdpa: {first_exc}")
                    self._ovon_model = QwenOVONModel(
                        self.model_path, self.device, attn_impl="sdpa"
                    )
                else:
                    raise
            self._log(
                f"Loaded OmniNav OVON waypoint mode (infer_ovon 对齐): "
                f"path={self.model_path}, device={self.device}, multi_view={self.multi_view}"
            )
        else:
            try:
                from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            except ImportError as exc:
                raise ImportError(
                    "text 模式需要 transformers。Original import error: " + str(exc)
                ) from exc

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype=dtype,
                device_map=self.device.type,
            )
            self.model.eval()
            self._log(
                "Loaded OmniNav text-action mode (generate) — 与官方 waypoint 推理不同；"
                f"multi_view={self.multi_view}"
            )

        self.task_context: Dict[str, Any] = {}
        self.step_count = 0
        self.total_inferences = 0
        self.total_inference_time = 0.0
        self.last_action_name = "STOP"
        self.last_response_text = ""
        self._warned_multi_view_fallback = False
        self._warned_single_view_waypoint = False

    def _log(self, msg: str):
        if self.verbose:
            print(f"[OmniNavAgent] {msg}")

    def _log_waypoint_line(self, msg: str) -> None:
        if self.log_waypoint_inference:
            print(f"[OmniNavAgent:waypoint] {msg}")

    def _resolve_device(self, device: str):
        if device.startswith("cuda") and not torch.cuda.is_available():
            self._log("CUDA unavailable, fallback to CPU")
            return torch.device("cpu")
        return torch.device(device)

    def _waypoint_map_turn_speed(self, raw_turn_deg: float, dist: float) -> Tuple[float, float, Dict[str, float]]:
        """转角裁剪到 ±max_turn；速度 = dist×scale 后夹在 [floor, cap]（推理慢、wp 近时宜保持高巡航）。"""
        mx = self.waypoint_max_turn_deg
        turn = float(np.clip(raw_turn_deg, -mx, mx))
        cap = float(min(self.forward_velocity, 100.0))
        floor = self.waypoint_speed_floor
        v_dist = float(dist * self.waypoint_dist_scale)
        v_final = float(np.clip(max(floor, min(cap, v_dist)), 0.0, cap))
        detail = {
            "raw_turn_deg": float(raw_turn_deg),
            "turn_clipped_deg": turn,
            "v_from_dist": v_dist,
            "v_final": v_final,
        }
        return turn, v_final, detail

    def prepare_episode(self, task_context: Dict[str, Any]) -> None:
        self.task_context = dict(task_context)

    def reset(self):
        self.step_count = 0
        self.total_inferences = 0
        self.total_inference_time = 0.0
        self.last_action_name = "STOP"
        self.last_response_text = ""
        self._warned_multi_view_fallback = False
        self._warned_single_view_waypoint = False
        if self._ovon_history is not None:
            self._ovon_history.reset()

    def prepare_step_inputs(self, env, observation: np.ndarray, info: Dict) -> Tuple[np.ndarray, Dict]:
        enriched = dict(info)
        try:
            env_u = env.unwrapped
            enriched["protagonist_id"] = int(getattr(env_u, "protagonist_id", 0))
            pose_arr = enriched.get("Pose") or enriched.get("pose")
            if pose_arr is not None and len(pose_arr) > enriched["protagonist_id"]:
                from .omninav_waypoint_model import unreal_obj_pose_to_habitat_dict

                enriched["pose_habitat_dict"] = unreal_obj_pose_to_habitat_dict(
                    pose_arr[enriched["protagonist_id"]]
                )
        except Exception as exc:
            self._log(f"prepare_step_inputs: pose 缓存跳过: {exc}")

        if not self.multi_view:
            return observation, enriched

        sampling = self._capture_multi_view_images(env)
        if sampling is None:
            return observation, enriched

        enriched["current_images"] = sampling["images"]
        enriched["current_view_labels"] = sampling["labels"]
        enriched["current_view_angles"] = sampling["angles"]
        enriched["multi_view_sampling_time"] = sampling["elapsed"]
        return observation, enriched

    def act(self, observation: np.ndarray, info: Dict) -> Tuple[Any, Dict]:
        self.step_count += 1
        phase = info.get("task_phase", "find_injured")

        if self.inference_mode == "waypoint":
            move_action, extra = self._act_waypoint(observation, info, phase)
        else:
            move_action, extra = self._act_text(observation, info, phase)

        self.total_inferences += 1
        extra.setdefault("step", self.step_count)
        extra.setdefault("task_phase", phase)
        extra.setdefault("total_inferences", self.total_inferences)
        extra.setdefault(
            "avg_inference_time",
            self.total_inference_time / self.total_inferences if self.total_inferences > 0 else 0.0,
        )
        return (move_action, 0), extra

    def _mission_instruction(self, info: Dict[str, Any]) -> str:
        phase = info.get("task_phase", "find_injured")
        reference_text = (info.get("reference_text") or "").strip()
        if phase == "find_injured":
            return reference_text or self.DEFAULT_PHASE1_INSTRUCTION
        return reference_text or self.DEFAULT_PHASE2_INSTRUCTION

    def _act_waypoint(self, observation: np.ndarray, info: Dict, phase: str) -> Tuple[np.ndarray, Dict]:
        assert self._ovon_model is not None and self._ovon_history is not None

        imgs_bgr = info.get("current_images")
        if not imgs_bgr or len(imgs_bgr) < 3:
            if not self._warned_single_view_waypoint:
                self._warned_single_view_waypoint = True
                self._log("waypoint 模式需要三路 BGR（启用 multi_view）；本步用单帧复制三份降级。")
            base = observation
            if base.ndim == 4:
                base = base[0]
            imgs_bgr = [base.copy(), base.copy(), base.copy()]

        # 官方 add_frame 顺序 [left, right, front]；当前连拍为 [left, front, right]
        left, front, right = imgs_bgr[0], imgs_bgr[1], imgs_bgr[2]
        reorder_lr_f = [left, right, front]

        pose = info.get("pose_habitat_dict")
        if pose is None:
            from .omninav_waypoint_model import unreal_obj_pose_to_habitat_dict

            pose_arr = info.get("Pose") or info.get("pose")
            pid = int(info.get("protagonist_id", 0))
            if pose_arr is not None and len(pose_arr) > pid:
                pose = unreal_obj_pose_to_habitat_dict(pose_arr[pid])
            else:
                pose = unreal_obj_pose_to_habitat_dict([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self._ovon_history.add_frame(reorder_lr_f, pose)
        instruction = self._mission_instruction(info)

        t0 = time.time()
        messages = self._ovon_history.generate_infer_prompt(instruction, self._ovon_model)
        wp_pred, arrive_pred, sin_angle, cos_angle = self._ovon_model.qwen_infer(messages)
        elapsed = time.time() - t0
        self.total_inference_time += elapsed

        cnt = 0
        for cur_arrive in arrive_pred.squeeze().reshape(-1):
            if cur_arrive.item() >= 0:
                cnt += 1
        arrive = 1 if cnt == 5 else 0

        wp_np = wp_pred.detach().cpu().float().numpy().squeeze()
        recover = torch.atan2(sin_angle, cos_angle).detach().cpu().float().numpy().squeeze()
        arrive_np = arrive_pred.detach().cpu().float().numpy().squeeze()

        wp_2d = np.asarray(wp_np, dtype=np.float64)
        if wp_2d.ndim == 1:
            wp_2d = wp_2d.reshape(-1, 2)
        n_wp = wp_2d.shape[0]
        if n_wp == 0:
            raise ValueError("OmniNav waypoint 预测为空")
        idx_used = min(self.waypoint_horizon_index, n_wp - 1)

        theta_deg: Optional[float] = None
        dist: Optional[float] = None
        raw_turn_deg: Optional[float] = None
        wp_map_detail: Optional[Dict[str, float]] = None

        if arrive == 1:
            move_action = np.array([0.0, 0.0], dtype=np.float32)
            action_name = "STOP"
        else:
            wp_sel = wp_2d[idx_used]
            recover_flat = np.atleast_1d(recover)
            if recover_flat.size <= idx_used:
                raise ValueError(
                    f"recover 长度 {recover_flat.size} 不足以匹配 waypoint_horizon_index={self.waypoint_horizon_index}"
                )
            theta_rad = float(recover_flat[idx_used])
            dist = float(np.linalg.norm(wp_sel))
            raw_turn_deg = float(np.degrees(theta_rad))
            turn_cmd, speed_cmd, wp_map_detail = self._waypoint_map_turn_speed(raw_turn_deg, dist)
            move_action = np.array([turn_cmd, speed_cmd], dtype=np.float32)
            theta_deg = turn_cmd
            action_name = "WAYPOINT"

        extra_info: Dict[str, Any] = {
            "action_type": "omninav_waypoint",
            "action_name": action_name,
            "task_phase": phase,
            "arrive_pred": int(arrive),
            "arrive_cnt": int(cnt),
            "waypoint_infer_s": elapsed,
            "reference_text": info.get("reference_text", ""),
            "num_current_views": len(imgs_bgr),
            "multi_view_sampling_time": float(info.get("multi_view_sampling_time", 0.0)),
            "waypoint_horizon_index": int(idx_used),
        }

        if self.log_waypoint_inference:
            dbg = {
                "wp_np": np.asarray(wp_np).tolist(),
                "waypoint_horizon_index": int(idx_used),
                "arrive_branch": np.asarray(arrive_np).tolist(),
                "recover_rad": np.asarray(np.atleast_1d(recover)).tolist(),
                "dist": dist,
                "raw_turn_deg": raw_turn_deg,
                "move_action": move_action.tolist(),
                "waypoint_map": wp_map_detail,
            }
            extra_info["omninav_wp_debug"] = dbg
            wm = wp_map_detail or {}
            self._log_waypoint_line(
                f"step={self.step_count} phase={phase} {elapsed:.3f}s | "
                f"hz={idx_used} | arrive_cnt={cnt} STOP={arrive} | "
                f"wp_np={dbg['wp_np']} | "
                f"raw_turn={raw_turn_deg} clip_turn={wm.get('turn_clipped_deg')} | "
                f"dist={dist} v_final={wm.get('v_final')} move={move_action.tolist()}"
            )

        return move_action, extra_info

    def _act_text(self, observation: np.ndarray, info: Dict, phase: str) -> Tuple[np.ndarray, Dict]:
        assert self.model is not None and self.processor is not None

        prompt = self._build_prompt(info)
        current_images = self._get_current_images(observation, info)
        reference_image = info.get("reference_image")
        ref_pil = self._to_pil_rgb(reference_image) if reference_image is not None else None

        action_name, response_text, elapsed = self._predict_action(
            current_images=current_images,
            reference_image=ref_pil,
            prompt=prompt,
            view_labels=info.get("current_view_labels"),
        )
        self.total_inference_time += elapsed
        self.last_action_name = action_name
        self.last_response_text = response_text

        move_action = self._action_name_to_move(action_name)
        extra_info = {
            "action_type": "omninav_text_action",
            "action_name": action_name,
            "task_phase": phase,
            "reference_text": info.get("reference_text", ""),
            "model_response": response_text,
            "step": self.step_count,
            "num_current_views": len(current_images),
            "current_view_labels": list(info.get("current_view_labels", [])),
            "multi_view_sampling_time": float(info.get("multi_view_sampling_time", 0.0)),
            "total_inferences": self.total_inferences,
            "avg_inference_time": (
                self.total_inference_time / self.total_inferences
                if self.total_inferences > 0
                else 0.0
            ),
        }
        return move_action, extra_info

    def _build_prompt(self, info: Dict[str, Any]) -> str:
        phase = info.get("task_phase", "find_injured")
        reference_text = (info.get("reference_text") or "").strip()
        if phase == "find_injured":
            goal_text = reference_text or self.DEFAULT_PHASE1_INSTRUCTION
        else:
            goal_text = self.DEFAULT_PHASE2_INSTRUCTION

        if phase == "find_injured":
            phase_hint = (
                "Current stage: find the injured person. "
                "Move toward the injured person and stop only when close enough for carry."
            )
        else:
            phase_hint = (
                "Current stage: find the stretcher while carrying the injured person. "
                "Move toward the stretcher and stop only when close enough for drop."
            )

        return (
            "You are a navigation policy controlling a rescue robot from RGB observations. "
            "If multiple current views are provided, they are ordered from left to right across the robot's heading. "
            "A reference image may show the goal or related target clue. "
            f"{phase_hint} "
            f"Task instruction: {goal_text} "
            "Reply with exactly one word from this set only: FORWARD, LEFT, RIGHT, STOP."
        )

    def _to_pil_rgb(self, image: Optional[np.ndarray]) -> Optional[Image.Image]:
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise ValueError(f"Unexpected image shape for OmniNavAgent: {arr.shape}")
        arr = arr[..., :3]
        if self.bgr_input:
            arr = arr[..., ::-1]
        arr = np.ascontiguousarray(arr).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def _get_current_images(self, observation: np.ndarray, info: Dict[str, Any]) -> List[Image.Image]:
        images = info.get("current_images")
        if images:
            pil_images = [self._to_pil_rgb(image) for image in images if image is not None]
            if pil_images:
                return pil_images
        return [self._to_pil_rgb(observation)]

    def _capture_multi_view_images(self, env) -> Optional[Dict[str, Any]]:
        try:
            env_unwrapped = env.unwrapped
            player = env_unwrapped.player_list[env_unwrapped.protagonist_id]
            cam_id = env_unwrapped.cam_list[env_unwrapped.protagonist_id]
            agent_cfg = env_unwrapped.agents[player]
            base_loc = list(agent_cfg["relative_location"])
            base_rot = list(agent_cfg.get("relative_rotation", [0.0, 0.0, 0.0]))
            view_images = []
            view_labels = []
            start = time.time()
            try:
                for angle in self.multi_view_angles:
                    rot = list(base_rot)
                    if len(rot) < 3:
                        raise ValueError(f"Unexpected camera rotation config: {rot}")
                    rot[1] = float(base_rot[1]) + float(angle)
                    env_unwrapped.unrealcv.set_cam(player, base_loc, rot)
                    if self.multi_view_settle_time > 0:
                        time.sleep(self.multi_view_settle_time)
                    # Character_API / unrealcv：与 gym_rescue 其它处一致使用 get_image，勿用 read_image（部分环境无该方法）
                    view_images.append(env_unwrapped.unrealcv.get_image(cam_id, "lit", "bmp"))
                    view_labels.append(self._view_label_from_angle(angle))
            finally:
                env_unwrapped.unrealcv.set_cam(player, base_loc, base_rot)
                if self.multi_view_settle_time > 0:
                    time.sleep(self.multi_view_settle_time)
            return {
                "images": view_images,
                "labels": view_labels,
                "angles": list(self.multi_view_angles),
                "elapsed": time.time() - start,
            }
        except Exception as exc:
            if not self._warned_multi_view_fallback:
                self._warned_multi_view_fallback = True
                self._log(f"Multi-view capture unavailable, fallback to single view: {exc}")
            return None

    def _view_label_from_angle(self, angle: float) -> str:
        if angle < -1e-6:
            return f"LEFT_{abs(int(round(angle)))}"
        if angle > 1e-6:
            return f"RIGHT_{int(round(angle))}"
        return "FRONT"

    def _predict_action(
        self,
        current_images: Sequence[Image.Image],
        reference_image: Optional[Image.Image],
        prompt: str,
        view_labels: Optional[Sequence[str]] = None,
    ) -> Tuple[str, str, float]:
        content = []
        images = []
        if reference_image is not None:
            content.append({"type": "text", "text": "Reference image of the target or clue:"})
            content.append({"type": "image"})
            images.append(reference_image)

        view_labels = list(view_labels or [])
        if len(current_images) == 1:
            content.append({"type": "text", "text": "Current first-person observation:"})
            content.append({"type": "image"})
            images.append(current_images[0])
        else:
            for idx, current_image in enumerate(current_images):
                label = view_labels[idx] if idx < len(view_labels) else f"VIEW_{idx}"
                content.append({"type": "text", "text": f"Current {label} observation:"})
                content.append({"type": "image"})
                images.append(current_image)
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[chat_text],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample and self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        start = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - start

        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, input_len:]
        response_text = self.processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        action_name = self._parse_action(response_text)

        self._log(
            f"Step {self.step_count}: phase inference -> {action_name} "
            f"({elapsed:.2f}s) raw={response_text!r}"
        )
        return action_name, response_text, elapsed

    def _parse_action(self, text: str) -> str:
        upper = (text or "").strip().upper()
        aliases = [
            ("FORWARD", ["FORWARD", "AHEAD", "STRAIGHT", "GO FORWARD", "MOVE FORWARD"]),
            ("LEFT", ["LEFT", "TURN LEFT", "GO LEFT"]),
            ("RIGHT", ["RIGHT", "TURN RIGHT", "GO RIGHT"]),
            ("STOP", ["STOP", "WAIT", "HOLD", "STAY"]),
        ]
        for canonical, words in aliases:
            for word in words:
                if word in upper:
                    return canonical
        return self.last_action_name or "STOP"

    def _action_name_to_move(self, action_name: str) -> np.ndarray:
        if action_name == "FORWARD":
            return np.array([0.0, self.forward_velocity], dtype=np.float32)
        if action_name == "LEFT":
            return np.array([-self.turn_angle, self.turn_velocity], dtype=np.float32)
        if action_name == "RIGHT":
            return np.array([self.turn_angle, self.turn_velocity], dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)

    def on_episode_end(self, success: bool, metrics: EpisodeMetrics):
        status = "SUCCESS" if success else "FAILED"
        reason = f" ({metrics.failure_reason})" if metrics.failure_reason else ""
        avg_time = (
            self.total_inference_time / self.total_inferences
            if self.total_inferences > 0
            else 0.0
        )
        self._log(
            f"{status}{reason} | steps={metrics.steps}, time={metrics.time_cost:.1f}s | "
            f"phase1={'OK' if metrics.phase1_success else 'FAIL'}, "
            f"phase2={'OK' if metrics.phase2_success else 'FAIL'} | "
            f"model_calls={self.total_inferences}, avg_inference={avg_time:.2f}s"
        )
