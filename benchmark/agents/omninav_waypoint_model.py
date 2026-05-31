"""
OmniNav OVON 官方推理核心（从 infer_ovon/agent/waypoint_agent_ovon.py 抽取，供 benchmark 使用）。

依赖：transformers、safetensors、scipy、qwen_vl_utils（与 Qwen2.5-VL 配套）。
"""

from __future__ import annotations

import os
import sys

# OmniNav 训练仓库内打补丁的 transformers（action_former 返回四元组），必须在首次 import transformers 之前
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_OMNI_TRANS = os.path.join(_PROJECT_ROOT, "baseline_model", "OmniNav", "train_code", "transformers-main", "src")
if os.path.isdir(_OMNI_TRANS) and _OMNI_TRANS not in sys.path:
    sys.path.insert(0, _OMNI_TRANS)

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from scipy.spatial.transform import Rotation as R
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError as e:  # pragma: no cover
    process_vision_info = None
    _QWEN_VL_UTILS_ERROR = e
else:
    _QWEN_VL_UTILS_ERROR = None

# 与官方 waypoint_agent_ovon.py 保持一致
PREDICT_SCALE = 0.3
MAX_HISTORY_FRAMES = 20
NUM_CURRENT_IMAGE = 3
INPUT_IMG_SIZE = (640, 569)
HISTORY_RESIZE_RATIO = 1 / 4
NUM_ACTION_TRUNK = 5
FLOW_MATCH = False


def unreal_obj_pose_to_habitat_dict(pose: Sequence[float]) -> Dict[str, Any]:
    """
    gym_rescue obj_pose: location(3) + rotation(3)，与角色 set_cam 一致为 roll, pitch, yaw（度）。
    转为官方 pose_to_matrix 使用的 habitat 格式：position + quaternion [w,x,y,z]。
    """
    if len(pose) < 6:
        raise ValueError(f"Expected pose length >= 6, got {len(pose)}")
    x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
    roll, pitch, yaw = float(pose[3]), float(pose[4]), float(pose[5])
    q_xyzw = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_quat()
    return {
        "position": [x, y, z],
        "rotation": [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])],
    }


class QwenOVONModel:
    """对应官方 QwenModel：load 权重 + qwen_data_pack + qwen_infer (action_former)。"""

    def __init__(self, model_path: str, device: torch.device, attn_impl: str = "flash_attention_2"):
        if _QWEN_VL_UTILS_ERROR is not None:
            raise ImportError(
                "需要安装 qwen_vl_utils（通常随 Qwen2-VL 生态提供）。"
                f"原始错误: {_QWEN_VL_UTILS_ERROR}"
            )

        self.model_path = os.path.abspath(model_path)
        self.device = device
        self.nav_version = "special_token"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True)

        if config.model_type != "qwen2_5_vl":
            raise ValueError(f"当前 OmniNav ckpt 期望 qwen2_5_vl，得到 {config.model_type}")

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype="auto",
                device_map=None,
                attn_implementation=attn_impl,
            )
        except Exception:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
                torch_dtype="auto",
                device_map=None,
            )
        self.model = self.model.to(device)
        self.model.eval()

        for name in os.listdir(self.model_path):
            if not name.endswith("safetensors"):
                continue
            safe_model_path = os.path.join(self.model_path, name)
            state_dict = load_file(safe_model_path)
            self.model.load_state_dict(state_dict, strict=False)

    def qwen_data_pack(self, images: List[Image.Image], user_content: str):
        content = []
        for idx, image in enumerate(images):
            if idx >= len(images) - NUM_CURRENT_IMAGE:
                cur_json = {
                    "type": "image",
                    "image": image,
                    "resized_height": INPUT_IMG_SIZE[1],
                    "resized_width": INPUT_IMG_SIZE[0],
                }
            else:
                cur_json = {
                    "type": "image",
                    "image": image,
                    "resized_height": int(INPUT_IMG_SIZE[1] * HISTORY_RESIZE_RATIO),
                    "resized_width": int(INPUT_IMG_SIZE[0] * HISTORY_RESIZE_RATIO),
                }
            content.append(cur_json)
        content.append({"type": "text", "text": user_content})
        messages = [{"role": "user", "content": content}]
        return messages

    def qwen_infer(self, messages):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + "<|im_end|>"
        if self.nav_version == "special_token":
            text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", "")
            num_image = len(messages[0]["content"]) - 1
            num_current_image = NUM_CURRENT_IMAGE
            num_history_image = num_image - num_current_image
            history_img_str = "".join(["<|vision_start|><|image_pad|><|vision_end|>"] * num_history_image)
            history_str_pos = text.rfind("Your historical pictures are: ") + len("Your historical pictures are: ")
            text = text[:history_str_pos] + history_img_str + text[history_str_pos:]
            text = text.replace("leftside: ", "leftside: <|vision_start|><|image_pad|><|vision_end|>")
            text = text.replace("rightside: ", "rightside: <|vision_start|><|image_pad|><|vision_end|>")
            text = text.replace("frontside: ", "frontside: <|vision_start|><|image_pad|><|vision_end|>")

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if FLOW_MATCH:
                norm = [{"min": [], "max": []}]
                wp_pred, arrive_pred, sin_angle, cos_angle = self.model.forward(
                    **inputs,
                    norm=norm,
                    action_former=True,
                    gt_waypoints=0,
                    train=False,
                    train_branch=["continue"],
                )
            else:
                wp_pred, arrive_pred, sin_angle, cos_angle = self.model.forward(
                    **inputs,
                    action_former=True,
                    gt_waypoints=0,
                    train=False,
                    train_branch=["continue"],
                )
        return wp_pred * PREDICT_SCALE, arrive_pred, sin_angle, cos_angle


class OmniNavOVONHistory:
    """对应官方 Waypoint_Agent 中 rgb_list / pose_list / add_frame / generate_infer_prompt。"""

    promt_template = """You are an autonomous navigation robot. You will get a task with historical pictures and current pictures you see.
            Based on these information, you need to decide your next {num_action_trunck} actions, which could involve <|left|>,<|right|>,<|forward|>. If you finish your mission, output <|stop|>. Here are some examples: <|left|><|forward|><|forward|><|stop|>, <|forward|><|forward|><|forward|><|left|><|forward|> or <|stop|>
            # Your historical pictures are: {history_img_string}
            # {current_img_string}
            # Your mission is: {instruction}<|NAV|>\nOutput the waypoint"""

    def __init__(self):
        self.rgb_list: List[Image.Image] = []
        self.pose_list: List[Dict[str, Any]] = []
        self.image_indices: List[int] = []
        self.total_frame_count = 0

    def reset(self):
        self.rgb_list = []
        self.pose_list = []
        self.image_indices = []
        self.total_frame_count = 0

    @staticmethod
    def pose_to_matrix(pose: Dict[str, Any]) -> np.ndarray:
        position = np.array(pose["position"])
        rotation = np.array(pose["rotation"])
        rotation_matrix = R.from_quat(rotation[[1, 2, 3, 0]]).as_matrix()
        rot_normal_raw = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rotation_matrix = rotation_matrix @ rot_normal_raw
        mat = np.eye(4)
        mat[:3, :3] = rotation_matrix
        mat[:3, 3] = position
        return mat

    def transform_poses_to_local(self, current_pose: Dict[str, Any], input_poses: List[Dict[str, Any]]):
        current_pose_m = self.pose_to_matrix(current_pose)
        current_pose_inv = np.linalg.inv(current_pose_m)
        return [current_pose_inv @ self.pose_to_matrix(p) for p in input_poses]

    def add_frame(self, rgbs_bgr: Sequence[np.ndarray], pose: Dict[str, Any]):
        """rgbs_bgr: 顺序为 [left, right, front]（与官方 observations 一致），BGR uint8。"""
        rgbs_new = []
        for rgb in rgbs_bgr:
            if isinstance(rgb, np.ndarray):
                arr = rgb[..., :3] if rgb.ndim == 3 else rgb
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                rgb_img = Image.fromarray(arr)
                rgb_pil = rgb_img.resize((INPUT_IMG_SIZE[0], INPUT_IMG_SIZE[1]))
            else:
                rgb_pil = rgb.resize((INPUT_IMG_SIZE[0], INPUT_IMG_SIZE[1]))
            rgbs_new.append(rgb_pil)

        rgbs_new = [rgbs_new[idx] for idx in [0, 2, 1]]

        if len(self.rgb_list) >= NUM_CURRENT_IMAGE:
            pop_idx = [-1, -2]
            for idx in pop_idx:
                self.rgb_list.pop(idx)
                self.pose_list.pop(idx)
                self.image_indices.pop(idx)

        self.rgb_list.extend(rgbs_new)
        self.pose_list.extend([pose] * len(rgbs_new))
        self.image_indices.extend([self.total_frame_count] * len(rgbs_new))
        self.total_frame_count += 1

        if len(self.rgb_list) > NUM_CURRENT_IMAGE:
            self.rgb_list[-1 - NUM_CURRENT_IMAGE] = self.rgb_list[-1 - NUM_CURRENT_IMAGE].resize(
                (
                    int(INPUT_IMG_SIZE[0] * HISTORY_RESIZE_RATIO),
                    int(INPUT_IMG_SIZE[1] * HISTORY_RESIZE_RATIO),
                )
            )

        if len(self.rgb_list) > MAX_HISTORY_FRAMES + NUM_CURRENT_IMAGE:
            min_interval_idx = int(np.argmin(np.diff(self.image_indices[:-NUM_CURRENT_IMAGE])))
            self.rgb_list.pop(min_interval_idx + 1)
            self.pose_list.pop(min_interval_idx + 1)
            self.image_indices.pop(min_interval_idx + 1)

    def generate_infer_prompt(self, instruction: str, model: QwenOVONModel):
        cur_prompt = deepcopy(self.promt_template)
        input_poses = deepcopy(self.pose_list)
        local_poses = self.transform_poses_to_local(self.pose_list[-1], input_poses)
        _ = [[pose[0, 3], pose[2, 3]] for pose in local_poses]

        history_img_string = ""
        current_img_string = "Your current observations is leftside: , frontside: , rightside: "

        cur_prompt = cur_prompt.format(
            instruction=instruction,
            num_action_trunck=NUM_ACTION_TRUNK,
            current_img_string=current_img_string,
            history_img_string=history_img_string,
        )
        return model.qwen_data_pack(self.rgb_list, cur_prompt)
