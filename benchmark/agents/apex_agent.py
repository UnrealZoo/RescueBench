"""
Apex Agent - 薄包装器
直接调用赛队原始 AlgSolution (solution.py)，仅做 benchmark 接口格式转换。

官方模型特点:
    - 输入: base64 编码后的 PNG 图像
    - 内部: YOLO + ILAgent，两帧堆叠后输出离散动作
    - 输出: {angular, velocity, viewport, interaction}

本文件只负责:
    1. numpy BGR -> base64
    2. Apex action dict -> benchmark 的 (move_action, head_action, anim_action)
    3. 处理 workspace / sys.path / 相对 checkpoint 路径
"""

import base64
import importlib.util
import os
import sys
from typing import Any, Dict, Tuple

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(BENCHMARK_DIR)
APEX_WORKSPACE = os.environ.get(
    "APEX_WORKSPACE",
    os.path.join(REPO_ROOT, "baseline_model", "Apexcode", "apex_code"),
)
APEX_GEA_ROOT = os.path.join(APEX_WORKSPACE, "gea")

if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)
if APEX_WORKSPACE not in sys.path:
    sys.path.insert(0, APEX_WORKSPACE)
if APEX_GEA_ROOT not in sys.path:
    sys.path.insert(0, APEX_GEA_ROOT)

from agents.agent_base import BaseAgent
from core.metrics import EpisodeMetrics


def _numpy_to_base64(image: np.ndarray) -> str:
    """numpy BGR 图像 -> base64 PNG 字符串"""
    _, buf = cv2.imencode(".png", image)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _load_apex_solution_class():
    """
    用唯一模块名加载 Apex solution.py，避免与其它赛队同名模块冲突。
    """
    solution_path = os.path.join(APEX_WORKSPACE, "solution.py")
    if not os.path.isfile(solution_path):
        raise FileNotFoundError(f"Apex solution.py not found: {solution_path}")

    module_name = "apex_solution_module"
    spec = importlib.util.spec_from_file_location(module_name, solution_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Apex solution module from {solution_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AlgSolution


class ApexAgent(BaseAgent):
    """
    Apex benchmark adapter.

    数据流:
        benchmark obs (numpy BGR)
            -> base64 PNG
            -> AlgSolution.predicts(ob_b64, picked)
            -> {angular, velocity, viewport, interaction}
            -> (np.array([angle, velocity]), head_action, anim_action)

    注意:
        - 官方模型自己决定 carry/drop，因此默认应配合 passthrough=True 使用
        - viewport 是上下视角，benchmark 的 head_action 是左右转头，无法直接映射
    """

    def __init__(self, workspace: str = None, **kwargs):
        del kwargs
        self.workspace = os.path.abspath(os.path.expanduser(workspace or APEX_WORKSPACE))
        self._need_reset = True

        saved_cwd = os.getcwd()
        os.chdir(self.workspace)
        try:
            AlgSolution = _load_apex_solution_class()
            self.alg = AlgSolution()
        finally:
            os.chdir(saved_cwd)

        print(f"[Apex] Initialized with workspace={self.workspace}")

    def reset(self):
        self._need_reset = True

    def act(self, observation: np.ndarray, info: Dict) -> Tuple[Any, Dict]:
        if self._need_reset:
            saved_cwd = os.getcwd()
            os.chdir(self.workspace)
            try:
                self.alg.reset(
                    reference_text=info.get("reference_text"),
                    reference_image=info.get("reference_image"),
                )
            finally:
                os.chdir(saved_cwd)
            self._need_reset = False

        ob_b64 = _numpy_to_base64(observation)
        unreal_action = self.alg.predicts(ob_b64, info.get("picked", False))

        move_action = np.array(
            [
                float(unreal_action.get("angular", 0.0)),
                float(unreal_action.get("velocity", 0.0)),
            ],
            dtype=np.float32,
        )

        interaction = int(unreal_action.get("interaction", 0))
        anim_action = 0
        if interaction == 3:
            anim_action = 3
        elif interaction == 4:
            anim_action = 4

        head_action = 0

        return (move_action, head_action, anim_action), {
            "source": "apex",
            "picked_internal": bool(getattr(self.alg, "if_pick", False)),
            "interaction": interaction,
            "viewport": unreal_action.get("viewport", 0),
        }

    def on_episode_end(self, success: bool, metrics: EpisodeMetrics):
        status = "SUCCESS" if success else "FAILED"
        reason = f" ({metrics.failure_reason})" if metrics.failure_reason else ""
        print(
            f"[Apex] {status}{reason} | "
            f"steps={metrics.steps}, time={metrics.time_cost:.1f}s | "
            f"phase1={'OK' if metrics.phase1_success else 'FAIL'}, "
            f"phase2={'OK' if metrics.phase2_success else 'FAIL'}"
        )
