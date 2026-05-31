"""Per-episode path similarity vs reference trajectories (Step 14)."""

from __future__ import annotations

from typing import Any, List, Optional

from utils.human_trajectory_loader import build_reference_trajectory_key


def episode_path_similarity(benchmark: Any, trajectory: List, level: int, point_id: int) -> Optional[float]:
    """Normalized similarity for the current episode trajectory; ``None`` if disabled or no ref."""
    if not benchmark.enable_path_similarity:
        return None
    if len(trajectory) < 2:
        return None
    np = __import__("numpy")
    env_id = benchmark.current_env_id or benchmark.env_id
    traj_key = build_reference_trajectory_key(env_id, level, point_id)
    legacy_key = f"level_{level}_point_{point_id}"
    ref_traj = benchmark.reference_trajectories.get(traj_key)
    if ref_traj is None:
        ref_traj = benchmark.reference_trajectories.get(legacy_key)
    if ref_traj is None:
        return None
    agent_traj = np.array(trajectory)
    calc = benchmark.path_calculator
    method_map = {
        "dtw": calc.dtw_distance,
        "frechet": calc.frechet_distance,
        "hausdorff": calc.hausdorff_distance,
    }
    dist_fn = method_map.get(benchmark.similarity_method, calc.dtw_distance)
    return calc.normalized_similarity(dist_fn(agent_traj, ref_traj))
