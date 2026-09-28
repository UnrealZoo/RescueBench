"""
SG-Nav Agent adapter for rescue benchmark.

This adapter reuses SG_Nav_Agent from a patched SG-Nav checkout and maps
benchmark observations/actions to the expected SG-Nav interface.
"""

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
if BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, BENCHMARK_DIR)

from agents.agent_base import BaseAgent  # noqa: E402
from core.metrics import EpisodeMetrics  # noqa: E402


def ensure_conda_cuda_libs_first() -> None:
    """
    conda-forge 的 faiss-gpu 会加载 conda 里的 libcublas.so.11；若 LD_LIBRARY_PATH
    或系统默认搜索路径里先有 /usr/local/cuda-*/lib64，则可能载入旧版 libcublasLt，
    与新版 libcublas 不匹配，触发::

        undefined symbol: cublasLtHSHMatmulAlgoInit, version libcublasLt.so.11

    在导入依赖 faiss 的 SG-Nav 代码之前，将 CONDA_PREFIX/lib 置于最前。
    """
    if sys.platform != "linux":
        return
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return
    lib = os.path.join(prefix, "lib")
    prev = os.environ.get("LD_LIBRARY_PATH", "")
    # 同样移除系统 CUDA 路径，避免 libcublas / libcublasLt 混用。
    keep = []
    for p in prev.split(":"):
        if not p:
            continue
        if "/cuda" in p or "cuda-" in p:
            continue
        keep.append(p)
    os.environ["LD_LIBRARY_PATH"] = ":".join([lib] + keep) if keep else lib

    # 强制预加载 conda 版本的 cublas / cublasLt。
    preload = []
    for name in ("libcublas.so.11", "libcublasLt.so.11"):
        cand = os.path.join(lib, name)
        if os.path.exists(cand):
            preload.append(cand)
    if preload:
        prev_preload = os.environ.get("LD_PRELOAD", "")
        prev_list = [x for x in prev_preload.split(":") if x]
        for x in reversed(preload):
            if x not in prev_list:
                prev_list.insert(0, x)
        os.environ["LD_PRELOAD"] = ":".join(prev_list)


def prepare_sgnav_sys_path(sgnav_root: str) -> str:
    """
    将 SG-Nav 仓库根目录加入 sys.path，并保证 Meta SAM 可导入。

    sgnav_agent 下的 ``segment_anything/`` 是 SAM 克隆仓库外层目录（无顶层
    ``__init__.py``）。若仅把仓库根放在 path 最前，``import segment_anything``
    会误解析为该外层目录。须把 ``.../segment_anything``（SAM 的 setup 根）插在
    sgnav 根目录之前。

    先移除再插入，保证 [SAM 根, sgnav 根] 始终位于 sys.path 最前（即使二者此前
    已在 path 中但顺序靠后）。
    """
    root = os.path.abspath(os.path.expanduser(sgnav_root))
    sam_repo = os.path.join(root, "segment_anything")
    for p in (sam_repo, root):
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, root)
    if os.path.isdir(sam_repo):
        sys.path.insert(0, sam_repo)
    return root


class SGNavRescueAgent(BaseAgent):
    """
    Migrate SG_Nav_Agent into rescue benchmark with phase-aware targets.

    - Phase1 (find_injured): target = person
    - Phase2 (find_stretcher): target = car (ambulance proxy)
    """

    def __init__(
        self,
        sgnav_root: Optional[str] = None,
        config_file: Optional[str] = None,
        visualize: bool = False,
        experiment_name: str = "sgnav_rescue",
        goal_detect_interval: int = 5,
        print_glip_labels: bool = True,
        openworld: int = 0,
        rescue_visual_align: bool = True,
        rescue_pick_approach: bool = True,
        rescue_depth_forward_priority: bool = True,
        rescue_forward_priority_depth_m: float = 2.0,
        verbose: bool = True,
        **kwargs,
    ):
        self.verbose = bool(verbose)

        ensure_conda_cuda_libs_first()
        sgnav_root = sgnav_root or os.environ.get("SGNAV_ROOT") or os.path.join(
            os.path.dirname(BENCHMARK_DIR), "baseline_model", "SG-Nav"
        )
        sgnav_root = os.path.abspath(os.path.expanduser(sgnav_root))
        if not os.path.isfile(os.path.join(sgnav_root, "SG_Nav_unreal.py")):
            raise FileNotFoundError(
                f"Patched SG-Nav not found at {sgnav_root}. See benchmark/sgnav/README.md"
            )
        self.sgnav_root = prepare_sgnav_sys_path(sgnav_root)
        # Both projects define a top-level ``utils`` package. The shared
        # benchmark imports its own package first; extend that package's
        # search path so SG-Nav's utils_scenegraph/utils_glip remain visible.
        import utils  # noqa: WPS433

        sgnav_utils = os.path.join(self.sgnav_root, "utils")
        if sgnav_utils not in utils.__path__:
            utils.__path__.insert(0, sgnav_utils)
        self.config_file = config_file or os.path.join(
            self.sgnav_root, "configs", "challenge_objectnav2021.local.rgbd.yaml"
        )

        import habitat  # noqa: WPS433
        from SG_Nav_unreal import SG_Nav_Agent  # noqa: WPS433

        cfg = habitat.get_config(self.config_file)
        # openworld=1：GLIP 用 object_captions_openworld，且跳过 scenegraph.update_scenegraph。
        # openworld=0（默认）：室内 GLIP 词表 + 场景图；person/car 已加入 utils_glip.categories_21，救援目标走零共现分支。
        _ow = 1 if int(openworld) else 0
        sgnav_args = SimpleNamespace(
            visualize=bool(visualize),
            experiment_name=experiment_name,
            split_l=-1,
            split_r=11,
            openworld=_ow,
            goal_detect_interval=max(1, int(goal_detect_interval)),
            print_glip_labels=bool(print_glip_labels),
            rescue_visual_align=bool(rescue_visual_align),
            rescue_pick_approach=bool(rescue_pick_approach),
            rescue_depth_forward_priority=bool(rescue_depth_forward_priority),
            rescue_forward_priority_depth_m=float(rescue_forward_priority_depth_m),
        )

        self.agent = SG_Nav_Agent(agent_id=0, task_config=cfg, args=sgnav_args)

        self._phase = None
        self._goal_name = None
        self._last_obs_shape = None

        # Runtime context injected from benchmark loop
        self._env = None

    def bind_env(self, env):
        """Bind benchmark env to read pose/depth if needed."""
        self._env = env

    def _log(self, msg: str):
        if self.verbose:
            print(f"[SGNavRescueAgent] {msg}")

    def _phase_to_goal(self, task_phase: str) -> str:
        if task_phase == "find_stretcher":
            return "car"
        return "person"

    def _apply_phase_goal(self, task_phase: str):
        goal = self._phase_to_goal(task_phase)
        if self._goal_name == goal:
            return

        self._phase = task_phase
        self._goal_name = goal

        # Switch target online and clear goal-specific transients.
        self.agent.obj_goal = goal
        self.agent.obj_goal_sg = goal
        self.agent.found_goal = False
        self.agent.found_goal_times = 0
        self.agent.found_possible_goal = False
        self.agent.goal_map = np.zeros(self.agent.full_map.shape[-2:])
        self.agent.goal_gps_map = np.zeros_like(self.agent.goal_gps_map)
        if hasattr(self.agent, "_rescue_turn_streak"):
            self.agent._rescue_turn_streak = 0
        self._log(f"phase={task_phase}, sg_goal={goal}")

    def _extract_rgb(self, observation: np.ndarray) -> np.ndarray:
        obs = observation[0] if isinstance(observation, tuple) else observation
        if obs.ndim != 3:
            raise ValueError(f"Unexpected observation shape: {getattr(obs, 'shape', None)}")
        if obs.shape[2] >= 3:
            rgb = obs[..., :3]
        else:
            raise ValueError(f"Need 3 channels RGB, got shape={obs.shape}")
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _extract_depth(self, observation: np.ndarray, info: Dict) -> np.ndarray:
        obs = observation[0] if isinstance(observation, tuple) else observation
        if obs.ndim == 3 and obs.shape[2] >= 4:
            # Keep the same convention as my_NavigationMultiAgent_sgnav.py
            return obs[..., 3][..., None].astype(np.float32) / 100.0

        d = info.get("depth")
        if isinstance(d, np.ndarray):
            if d.ndim == 2:
                return d[..., None].astype(np.float32)
            if d.ndim == 3 and d.shape[2] == 1:
                return d.astype(np.float32)

        raise ValueError("SG-Nav requires real RGB-D observations; use --observation-type Rgbd")

    def _get_pose_from_env(self):
        if self._env is None:
            raise RuntimeError("SG-Nav requires the benchmark environment for pose input")
        try:
            env_u = self._env.unwrapped
            pose = list(env_u.obj_poses[env_u.protagonist_id])
            gps = pose[:3]
            compass = pose[3:] if len(pose) > 3 else [0.0, 0.0, 0.0]
            return gps, compass
        except Exception as exc:
            raise RuntimeError("Cannot read SG-Nav pose from benchmark environment") from exc

    def prepare_step_inputs(self, env, observation, info):
        self.bind_env(env)
        return observation, info

    def reset(self):
        self.agent.reset()
        self._phase = None
        self._goal_name = None
        self._last_obs_shape = None

    def act(self, observation: np.ndarray, info: Dict) -> Tuple[Any, Dict]:
        task_phase = info.get("task_phase", "find_injured")
        self._apply_phase_goal(task_phase)

        rgb = self._extract_rgb(observation)
        depth = self._extract_depth(observation, info)
        gps, compass = self._get_pose_from_env()

        sg_obs = {
            "rgb": rgb[..., ::-1],  # SG_Nav expects BGR-style input then flips internally.
            "depth": depth,
            "gps": [x / 100.0 for x in gps],
            "compass": compass,
        }

        action_list, number_action = self.agent.act(sg_obs)
        final_action = action_list[0] if isinstance(action_list, (list, tuple)) else action_list

        if len(final_action) == 3:
            move_action, head_action, _ = final_action
        else:
            move_action, head_action = final_action, 0

        extra = {
            "source": "sg_nav",
            "task_phase": task_phase,
            "sg_goal": self._goal_name,
            "number_action": int(number_action),
        }
        return (np.array(move_action, dtype=np.float32), int(head_action)), extra

    def on_episode_end(self, success: bool, metrics: EpisodeMetrics):
        state = "SUCCESS" if success else "FAILED"
        self._log(
            f"{state} steps={metrics.steps} "
            f"p1={int(metrics.phase1_success)} p2={int(metrics.phase2_success)} "
            f"reason={metrics.failure_reason or 'NONE'}"
        )
