"""Load human trajectories from jsonl files for benchmark similarity."""

import json
import os
from typing import Dict, List

import numpy as np


def _extract_xyz_points(trajectory: List) -> List[List[float]]:
    xyz_points: List[List[float]] = []
    for p in trajectory:
        if isinstance(p, dict):
            if all(k in p for k in ("x", "y", "z")):
                xyz_points.append([float(p["x"]), float(p["y"]), float(p["z"])])
        elif isinstance(p, (list, tuple)) and len(p) >= 3:
            xyz_points.append([float(p[0]), float(p[1]), float(p[2])])
    return xyz_points


def build_reference_trajectory_key(env_id: str, level: int, point_id: int) -> str:
    """Build a unique reference-trajectory key.

    Including env_id avoids collisions when multiple maps share the same
    level/point_id namespace.
    """
    return f"{env_id}_level_{level}_point_{point_id}"


def load_reference_trajectories_from_jsonl(human_traj_dir: str) -> Dict[str, np.ndarray]:
    """Load jsonl human trajectories and return key->xyz array map.

    Key format: {env_id}_level_{level}_point_{point_id}
    point_id priority:
      1) record["episode_id"]
      2) line index in the file
    """
    refs: Dict[str, np.ndarray] = {}
    if not human_traj_dir or not os.path.isdir(human_traj_dir):
        return refs

    total_loaded = 0
    total_skipped = 0

    for env_name in sorted(os.listdir(human_traj_dir)):
        env_path = os.path.join(human_traj_dir, env_name)
        if not os.path.isdir(env_path):
            continue

        for fname in sorted(os.listdir(env_path)):
            if not (fname.startswith("level_") and fname.endswith(".jsonl")):
                continue
            level_path = os.path.join(env_path, fname)
            try:
                level_from_name = int(fname[len("level_"):-len(".jsonl")])
            except Exception:
                level_from_name = None

            with open(level_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        total_skipped += 1
                        continue

                    env_id = rec.get("env_id", env_name)
                    level = int(rec.get("level", level_from_name if level_from_name is not None else -1))
                    point_id = int(rec.get("episode_id", line_idx))
                    traj = rec.get("trajectory", [])
                    xyz_points = _extract_xyz_points(traj)
                    if len(xyz_points) < 2:
                        total_skipped += 1
                        continue

                    key = build_reference_trajectory_key(env_id, level, point_id)
                    refs[key] = np.asarray(xyz_points, dtype=np.float32)
                    total_loaded += 1

    print(f"[HumanTrajLoader] Loaded {total_loaded} trajectories, skipped {total_skipped}")
    return refs

