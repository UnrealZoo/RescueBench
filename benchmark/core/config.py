"""Benchmark configuration data structures.

These dataclasses are introduced without changing the existing runtime path.
Later phases can use them to replace the long argument lists in
``rescue_benchmark.py``.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class BenchmarkConfig:
    """集中描述 RescueBenchmark 自身拥有的配置。"""

    env_id: str = "UnrealRescue-HongKongStreet"
    resolution: Tuple[int, int] = (320, 320)
    render: bool = False
    output_dir: str = "./benchmark_results"

    enable_collision_detection: bool = True
    enable_trajectory_recording: bool = False
    enable_path_similarity: bool = False
    collision_method: str = "api"
    similarity_method: str = "dtw"

    rescue_distance: float = 120.0
    place_distance: float = 100.0
    interaction_z_threshold: float = 220.0
    stage2_success_radius: float = 200.0
    passthrough: bool = False
    passthrough_env_term_geometry_sync: bool = True

    render_quality: int = 2
    offscreen: bool = True
    save_frame_every: int = 5
    save_video: bool = False
    video_fps: int = 10

    resume_jsonl: Optional[str] = None
    resume_skip: str = "all"
    resume_append: bool = False
    multiagent_env: bool = False


@dataclass
class TaskContext:
    """单个 benchmark 测试点的任务上下文。"""

    env_id: str
    injured_pose: Any
    stretcher_pose: Any
    agent_pose: Any
    ambulance_pose: Any
    timeout: int
    level: int
    point_id: int
    injured_agent_id: Optional[int] = None
    reference_text: str = ""
    reference_image_path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a dict compatible with the current benchmark code."""

        return {
            "env_id": self.env_id,
            "injured_pose": self.injured_pose,
            "injured_agent_id": self.injured_agent_id,
            "stretcher_pose": self.stretcher_pose,
            "agent_pose": self.agent_pose,
            "ambulance_pose": self.ambulance_pose,
            "reference_text": self.reference_text,
            "reference_image_path": self.reference_image_path,
            "timeout": self.timeout,
            "level": self.level,
            "point_id": self.point_id,
        }
