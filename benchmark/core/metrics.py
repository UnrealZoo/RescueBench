"""Benchmark result data structures."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeMetrics:
    """单个episode的评估指标"""

    episode_id: int
    level: int
    point_id: int
    success: bool
    time_cost: float
    steps: int
    collision_count: int
    trajectory: List[List[float]] = field(default_factory=list)
    drone_trajectory: List[List[float]] = field(default_factory=list)
    path_similarity: Optional[float] = None

    # 阶段性指标
    phase1_success: bool = False
    phase1_time: float = 0.0
    phase1_steps: int = 0
    phase1_collisions: int = 0

    phase2_success: bool = False
    phase2_time: float = 0.0
    phase2_steps: int = 0
    phase2_collisions: int = 0

    # 新评估指标
    task_completion: bool = False
    stage1_initial_distance: float = 0.0
    stage1_best_distance: float = 0.0
    stage1_final_distance: float = 0.0
    stage2_initial_distance: float = 0.0
    stage2_best_distance: float = 0.0
    stage2_final_distance: float = 0.0
    s1_score: float = 0.0
    s2_score: float = 0.0
    s3_score: float = 0.0
    s4_score: float = 0.0
    stage1_score: float = 0.0
    stage2_score: float = 0.0
    task_score: float = 0.0
    movement_effectiveness: float = 0.0
    control_loop_fps: float = 0.0
    path_length: float = 0.0
    env_terminated: bool = False
    env_truncated: bool = False

    carry_success: bool = False
    drop_success: bool = False

    # 失败原因
    failure_reason: str = ""
    final_state: str = ""

    # 任务上下文
    reference_text: str = ""


@dataclass
class LevelMetrics:
    """某个难度等级的汇总指标"""

    level: int
    num_episodes: int
    success_rate: float
    avg_time_cost: float
    avg_steps: float
    avg_collision_count: float
    avg_task_score: float = 0.0
    avg_s1_score: float = 0.0
    avg_s2_score: float = 0.0
    avg_s3_score: float = 0.0
    avg_s4_score: float = 0.0
    avg_movement_effectiveness: float = 0.0
    avg_control_loop_fps: float = 0.0
    avg_path_length: float = 0.0
    avg_path_similarity: Optional[float] = None
    std_time_cost: float = 0.0
    std_steps: float = 0.0
    std_collision_count: float = 0.0
    std_task_score: float = 0.0
    std_s1_score: float = 0.0
    std_s2_score: float = 0.0
    std_s3_score: float = 0.0
    std_s4_score: float = 0.0
    std_movement_effectiveness: float = 0.0
    std_control_loop_fps: float = 0.0
    std_path_length: float = 0.0


@dataclass
class BenchmarkResult:
    """完整的benchmark结果"""

    model_name: str
    env_name: str
    timestamp: str
    level_metrics: Dict[int, LevelMetrics]
    episode_details: List[EpisodeMetrics]
    config: Dict[str, Any] = field(default_factory=dict)
