"""
救援任务通用评估脚本 (Rescue Task Benchmark)
用法: python rescue_benchmark.py --model random --levels 2 3 4 --episodes 2
详见 README.md
"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple, Set

# --- 环境配置 ---
np = __import__("numpy")
np.bool8 = np.bool_
# 设置 UnrealEnv 路径 (根据你的实际路径修改)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GYM_RESCUE_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_UNREAL_ENV = '/media/littlecave/T9/UnrealEnv'

if 'UnrealEnv' not in os.environ:
    os.environ['UnrealEnv'] = DEFAULT_UNREAL_ENV

# 添加gym-rescue到path
if GYM_RESCUE_ROOT not in sys.path:
    sys.path.insert(0, GYM_RESCUE_ROOT)

from agents.agent_base import BaseAgent
from agents.factory import AGENT_REGISTRY, get_agent
from core.benchmark_runner import BenchmarkRunner
from core.env_manager import EnvManager
from core.episode_runner import EpisodeRunner
from core.metrics import BenchmarkResult, EpisodeMetrics, LevelMetrics
from core.result_writer import ResultWriter
from core.resume import ResumeManager
from core.task_loader import TaskLoader


# ---------------------------------------------------------------------------
# 1. 数据结构
# ---------------------------------------------------------------------------

# Re-exported for backward compatibility:
# existing agents can still use ``from rescue_benchmark import EpisodeMetrics``.


# ---------------------------------------------------------------------------
# 2. Agent 基类
# ---------------------------------------------------------------------------

# Re-exported for backward compatibility:
# existing agents can still use ``from rescue_benchmark import BaseAgent``.


# ---------------------------------------------------------------------------
# 4. 可选模块 (从 utils/ 导入)
# ---------------------------------------------------------------------------

from utils.path_similarity import PathSimilarityCalculator  # noqa: E402
from utils.collision_detector import CollisionDetector       # noqa: E402


# ---------------------------------------------------------------------------
# 5. 核心评估器
# ---------------------------------------------------------------------------

class RescueBenchmark:
    """救援任务 Benchmark 评估器"""
    
    # 时间限制配置 (秒)，与 gym_rescue/envs/setting/test_jsonl/level_*.jsonl 对齐
    TIME_LIMITS = {
        0: 180,  # L0: 3分钟
        1: 180,  # L1: 3分钟
        2: 240,  # L2: 4分钟
        3: 300,  # L3: 5分钟
        4: 300,  # L4: 5分钟
    }

    def __init__(
        self,
        env_id: str = 'UnrealRescue-HongKongStreet',
        agent: BaseAgent = None,
        resolution: Tuple[int, int] = (320, 320),
        render: bool = False,
        output_dir: str = './benchmark_results',
        # 可选功能开关
        enable_collision_detection: bool = True,
        enable_trajectory_recording: bool = False,
        enable_path_similarity: bool = False,
        reference_trajectories: Optional[Dict[str, Any]] = None,
        collision_method: str = 'api',
        similarity_method: str = 'dtw',
        # 状态机配置
        rescue_distance: float = 120.0,
        place_distance: float = 100.0,
        interaction_z_threshold: float = 220.0,
        stage2_success_radius: float = 200.0,
        passthrough: bool = False,
        # Passthrough：UE 先 termination 时是否用终点位姿补阶段二门限（见 run_episode 末段 sync）
        passthrough_env_term_geometry_sync: bool = True,
        # 高级配置
        render_quality: int = 2,
        offscreen: bool = True,
        # 渲染帧/视频配置
        save_frame_every: int = 5,
        save_video: bool = False,
        video_fps: int = 10,
        # 断点续跑配置
        resume_jsonl: Optional[str] = None,
        resume_skip: str = 'all',
        resume_append: bool = False,
        multiagent_env: bool = False,
    ):
        self.env_id = env_id
        self.agent = agent
        self.resolution = resolution
        self.render = render
        self.output_dir = output_dir
        
        # 功能开关
        self.enable_collision_detection = enable_collision_detection
        self.enable_trajectory_recording = enable_trajectory_recording
        self.enable_path_similarity = enable_path_similarity
        self.reference_trajectories = reference_trajectories or {}
        self.collision_method = collision_method
        self.similarity_method = similarity_method
        
        # 状态机配置
        self.rescue_distance = rescue_distance
        self.place_distance = place_distance
        self.interaction_z_threshold = interaction_z_threshold
        self.stage2_success_radius = stage2_success_radius
        self.passthrough = passthrough
        self.passthrough_env_term_geometry_sync = passthrough_env_term_geometry_sync
        
        # 高级配置
        self.render_quality = render_quality
        self.offscreen = offscreen if not render else False
        self.save_frame_every = max(1, int(save_frame_every))
        self.save_video = bool(save_video)
        self.video_fps = int(video_fps)
        self.resume_jsonl = os.path.abspath(resume_jsonl) if resume_jsonl else None
        self.resume_skip = resume_skip
        self.resume_append = bool(resume_append)
        self.multiagent_env = bool(multiagent_env)
        
        self.env = None
        self.current_env_id = None
        self.current_level = None
        self.collision_detector = None
        self.path_calculator = PathSimilarityCalculator()
        self.env_manager = EnvManager(
            resolution=self.resolution,
            render_quality=self.render_quality,
            offscreen=self.offscreen,
            multiagent_env=self.multiagent_env,
        )
        self.task_loader = TaskLoader(
            gym_rescue_root=os.path.join(GYM_RESCUE_ROOT, "gym_rescue"),
            fallback_env_id=self.env_id,
            time_limits=self.TIME_LIMITS,
            multiagent_env=self.multiagent_env,
        )
        self.episode_runner = EpisodeRunner(self)
        self.resume_manager = ResumeManager(self.resume_jsonl, self.resume_skip)
        self.result_writer = ResultWriter(self)
        self.current_run_timestamp: Optional[str] = None
        self.current_model_name: Optional[str] = None
        self.incremental_result_file: Optional[str] = None
        self.incremental_trajectory_file: Optional[str] = None
        self.resume_episode_records: Dict[Tuple[int, int, int], EpisodeMetrics] = {}
        self.resume_episode_keys: Set[Tuple[int, int, int]] = set()
        self._runner = BenchmarkRunner(self)

        os.makedirs(output_dir, exist_ok=True)
        self.resume_episode_records, self.resume_episode_keys = self.resume_manager.load()
        
        # 打印配置
        self._print_config()
    
    def _print_config(self):
        mode = "Passthrough" if self.passthrough else "Active"
        sim = f", Similarity={self.similarity_method}({len(self.reference_trajectories)} refs)" if self.enable_path_similarity else ""
        print(f"\n{'='*60}\n BENCHMARK CONFIG\n{'='*60}")
        print(f" Env=Auto(from test_jsonl, fallback={self.env_id})  Res={self.resolution}  Render={self.render}")
        if self.multiagent_env:
            print(" MultiAgentEnv: ON (自动将 UnrealRescue-* 映射到 UnrealRescueMultiAgent-*)")
        print(f" Collision={self.enable_collision_detection}  Trajectory={self.enable_trajectory_recording}{sim}")
        print(
            f" StateMachine: {mode}  RescueXY={self.rescue_distance}cm  PlaceXY={self.place_distance}cm  "
            f"ZGate={self.interaction_z_threshold}cm  SuccessRadius={self.stage2_success_radius}cm(XY+Z)"
        )
        if self.passthrough and self.passthrough_env_term_geometry_sync:
            print(
                " PassthroughEnvTermSync: ON (UE 先终止时用终点几何补阶段二判定，减轻 ENV_TERM_INCOMPLETE 误报)"
            )
        if self.resume_jsonl:
            print(
                f" Resume: {self.resume_jsonl}  Skip={self.resume_skip}  "
                f"Append={self.resume_append}  Loaded={len(self.resume_episode_keys)}"
            )
        if self.render:
            print(f" RenderFrames: every {self.save_frame_every} steps  SaveVideo={self.save_video}  VideoFPS={self.video_fps}")
        print(f" Output: {self.output_dir}\n{'='*60}\n")
    
    def _sync_env_state_from_manager(self):
        self.env = self.env_manager.env
        self.current_env_id = self.env_manager.current_env_id
        self.current_level = self.env_manager.current_level

    def _close_env(self):
        """关闭当前环境并同步 ``self.env`` / 碰撞检测器状态（编排入口，非纯转发）。"""
        self.env_manager.close_env()
        self._sync_env_state_from_manager()
        self.collision_detector = None

    def _ensure_env(self, env_id: str, level: int):
        """创建或复用环境并按需挂载碰撞检测（编排入口，非纯转发）。"""
        created = self.env_manager.ensure_env(env_id, level)
        self._sync_env_state_from_manager()
        if created:
            self.collision_detector = None
        if created and self.enable_collision_detection:
            self.collision_detector = CollisionDetector(self.env, self.collision_method)

    def run_episode(self, level: int, point_id: int, episode_id: int = 0) -> EpisodeMetrics:
        return self.episode_runner.run_episode(level, point_id, episode_id)

    def evaluate_level(
        self,
        level: int,
        episodes_per_point: int = 1,
        point_ids: Optional[List[int]] = None,
        close_env: bool = True,
        label: Optional[str] = None,
    ) -> Tuple[Optional[LevelMetrics], List[EpisodeMetrics]]:
        return self._runner.evaluate_level(
            level,
            episodes_per_point,
            point_ids=point_ids,
            close_env=close_env,
            label=label,
        )

    def run_benchmark(
        self,
        levels: List[int] = [2, 3, 4],
        episodes_per_point: int = 1,
        model_name: str = "unknown",
    ) -> BenchmarkResult:
        return self._runner.run_benchmark(levels, episodes_per_point, model_name)

# ---------------------------------------------------------------------------
# 6. Agent 注册与工厂 — 实现位于 agents.factory（向后兼容 re-export）
# ---------------------------------------------------------------------------

_AGENT_REGISTRY = AGENT_REGISTRY
# get_agent 由文件顶部 ``from agents.factory import ... get_agent`` 注入命名空间

# ---------------------------------------------------------------------------
# 7. CLI / 启动接口 — 实现位于 core.cli（供薄启动器与 __main__ 复用）
# ---------------------------------------------------------------------------

from core.cli import create_base_parser, main, run_benchmark_from_args  # noqa: E402

if __name__ == "__main__":
    main()
