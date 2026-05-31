"""
R2ZeroShot Agent - 薄包装器
直接调用原始 AlgSolution (solution.py)，仅做 benchmark 接口格式转换。

原始 AlgSolution 已包含完整的 R2ZeroShot 决策逻辑:
    - CrossViewRocket + CFGWrapper 导航
    - Sa2VA 视觉检测 (carry/drop)
    - 两阶段策略 (找人 → 找担架)
    - 动作映射 (Minecraft → Unreal)

本文件只负责:
    1. numpy BGR → base64 (AlgSolution 期望的输入格式)
    2. AlgSolution 返回的 {angular, velocity, viewport, interaction}
       → benchmark 的 (move_action, head_action, anim_action) 3-tuple

作者: Beiyu GUO
日期: 2026-02
"""

import cv2
import base64
import numpy as np
import os
import sys
from typing import Dict, Any, Tuple

# === 路径设置 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(BENCHMARK_DIR)
# 默认：仓库内 baseline_model/R2ZeroShot/workspace；可用环境变量覆盖
R2ZEROSHOT_WORKSPACE = os.environ.get(
    "R2ZEROSHOT_WORKSPACE",
    os.path.join(_REPO_ROOT, "baseline_model", "R2ZeroShot", "workspace"),
)
# 确保能 import rescue_benchmark 和原始 solution.py
if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)
if R2ZEROSHOT_WORKSPACE not in sys.path:
    sys.path.insert(0, R2ZEROSHOT_WORKSPACE)

from agents.agent_base import BaseAgent
from core.metrics import EpisodeMetrics


# ============================================================================
# 工具函数
# ============================================================================

def _numpy_to_base64(image: np.ndarray) -> str:
    """numpy BGR 图像 → base64 字符串 (原始 AlgSolution 期望的输入格式)"""
    _, buf = cv2.imencode('.png', image)
    return base64.b64encode(buf.tobytes()).decode('utf-8')


# ============================================================================
# R2ZeroShot Agent (薄包装器)
# ============================================================================

class R2ZeroShotAgent(BaseAgent):
    """
    R2ZeroShot Agent - 直接包装 AlgSolution

    数据流:
        benchmark obs (numpy BGR)
            → base64 编码
            → AlgSolution.predicts(ob_b64, success)
            → {angular, velocity, viewport, interaction}
            → (np.array([angle, velocity]), head_action, anim_action)
            → 返回给 benchmark
    """

    def __init__(
        self,
        cfg_coef: float = 2.0,
        mem_freq: int = 20,
        clr_freq: int = 100,
        ckpt_path: str = None,
        sa2va_path: str = None,
        use_sa2va: bool = True,
        person_ref_path: str = None,
        **kwargs,
    ):
        """
        Args:
            cfg_coef: Classifier-Free Guidance 系数 (越大越依赖参考图, 推荐 2.0~3.0)
            mem_freq: 每 N 步用 Sa2VA 更新参考图
            clr_freq: 每 N 步清除 ROCKET RNN 状态
        """
        # 原始代码使用相对路径加载模型 (./ckpts/...)，需切换到 workspace 目录
        saved_cwd = os.getcwd()
        os.chdir(R2ZEROSHOT_WORKSPACE)
        try:
            from solution import AlgSolution
            self.alg = AlgSolution(
                cfg_coef=cfg_coef,
                mem_freq=mem_freq,
                clr_freq=clr_freq,
                ckpt_path=ckpt_path,
                sa2va_path=sa2va_path,
                use_sa2va=use_sa2va,
                person_ref_path=person_ref_path,
            )
        finally:
            os.chdir(saved_cwd)

        self._need_reset = True
        print(
            f"[R2ZeroShot] Initialized (cfg_coef={cfg_coef}, "
            f"mem_freq={mem_freq}, clr_freq={clr_freq}, use_sa2va={use_sa2va})"
        )

    # ====================================================================
    # BaseAgent 接口
    # ====================================================================

    def reset(self):
        """
        标记需要在首次 act() 时重置。

        原因: benchmark 在 reset() 时还没有 reference_image (它在 info 中传入),
        所以延迟到第一次 act() 调用时才执行 AlgSolution.reset()。
        """
        self._need_reset = True

    def act(self, observation: np.ndarray, info: Dict) -> Tuple[Any, Dict]:
        """
        核心接口: 观测 → 动作

        Args:
            observation: BGR 图像 numpy array [H, W, 3], dtype uint8
            info: 环境信息字典, 包含:
                - 'picked': bool, 是否已抱起伤员
                - 'reference_image': numpy array 或 None, 参考图 (来自 JSON 配置)
                - 'reference_text': str, 任务描述文本

        Returns:
            action: (np.array([angle, velocity]), head_action, anim_action)
            extra_info: 调试信息字典
        """
        # --- 首次调用: 用参考图 reset 原始 AlgSolution ---
        if self._need_reset:
            ref_img = info.get('reference_image')
            ref_path = info.get('reference_image_path')
            if ref_path:
                print(f"[R2ZeroShot] reset() reference_image_path: {ref_path}")
            else:
                print("[R2ZeroShot] reset() reference_image_path: None")
            if ref_img is None:
                # fallback: 用当前观测帧作为参考
                print("[R2ZeroShot] WARNING: No reference_image in info, using current observation")
                ref_img = observation
            ref_b64 = _numpy_to_base64(ref_img)

            # reset 中会读 ./truck3.png，需要在 workspace 目录下
            saved_cwd = os.getcwd()
            os.chdir(R2ZEROSHOT_WORKSPACE)
            try:
                self.alg.reset(
                    reference_text=info.get('reference_text', ''),
                    reference_image=ref_b64,
                )
            finally:
                os.chdir(saved_cwd)
            self._need_reset = False

        # --- 调用原始 predicts() ---
        ob_b64 = _numpy_to_base64(observation)
        success = info.get('picked', False)
        unreal_action = self.alg.predicts(ob_b64, success)
        print(f"[DEBUG] unreal_action: angular={unreal_action.get('angular')}, velocity={unreal_action.get('velocity')}, interaction={unreal_action.get('interaction')}")
        # --- 动作格式转换: unreal dict → benchmark 3-tuple ---
        move_action = np.array([
            float(unreal_action.get('angular', 0)),
            float(unreal_action.get('velocity', 0)),
        ])

        interaction = unreal_action.get('interaction', 0)
        anim_action = 0
        if interaction == 3:
            anim_action = 3   # carry
        elif interaction == 4:
            anim_action = 4   # drop

        head_action = 0  # viewport (上下视角) 无法映射到 head_action (左右)

        return (move_action, head_action, anim_action), {
            'source': 'r2zeroshot',
            'goal': getattr(self.alg, 'present_goal', 'unknown'),
            'step': getattr(self.alg, 'num_steps', 0),
        }

    def on_episode_end(self, success: bool, metrics: EpisodeMetrics):
        """Episode 结束回调"""
        status = "SUCCESS" if success else "FAILED"
        reason = f" ({metrics.failure_reason})" if metrics.failure_reason else ""
        print(
            f"[R2ZeroShot] {status}{reason} | "
            f"steps={metrics.steps}, time={metrics.time_cost:.1f}s | "
            f"phase1={'OK' if metrics.phase1_success else 'FAIL'}, "
            f"phase2={'OK' if metrics.phase2_success else 'FAIL'}"
        )


# ============================================================================
# 独立测试
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("R2ZeroShotAgent - Thin Wrapper around AlgSolution")
    print("=" * 60)
    print(f"\nWorkspace: {R2ZEROSHOT_WORKSPACE}")
    print("\nTo run:")
    print("  python rescue_benchmark.py --model r2zeroshot --levels 1 --episodes 1")
    print("\nDone.")
