"""Load benchmark test points and build task contexts."""

import json
import os
from typing import Any, Dict, List, Optional

from .config import TaskContext


class TaskLoader:
    """读取 test jsonl，并保持与旧 benchmark 逻辑一致的 task context 输出。"""

    def __init__(
        self,
        gym_rescue_root: str,
        fallback_env_id: str,
        time_limits: Dict[int, int],
        multiagent_env: bool = False,
    ):
        self.gym_rescue_root = gym_rescue_root
        self.fallback_env_id = fallback_env_id
        self.time_limits = time_limits
        self.multiagent_env = bool(multiagent_env)
        self._test_points_cache: Dict[int, List[Dict[str, Any]]] = {}

    def resolve_env_id(self, env_id: str) -> str:
        if not self.multiagent_env:
            return env_id
        if env_id.startswith("UnrealRescue-"):
            return env_id.replace("UnrealRescue-", "UnrealRescueMultiAgent-", 1)
        return env_id

    def load_level_test_points(self, level: int) -> List[Dict[str, Any]]:
        if level in self._test_points_cache:
            return self._test_points_cache[level]

        json_file = os.path.join(
            self.gym_rescue_root,
            "envs",
            "setting",
            "test_jsonl",
            f"level_{level}.jsonl",
        )
        if not os.path.exists(json_file):
            self._test_points_cache[level] = []
            return []

        points: List[Dict[str, Any]] = []
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                points.append(json.loads(line))

        self._test_points_cache[level] = points
        return points

    def get_point_count(self, level: int) -> int:
        return len(self.load_level_test_points(level))

    def get_level_time_limit_text(self, level: int, point_ids: Optional[List[int]] = None) -> str:
        points = self.load_level_test_points(level)
        if not points:
            return f"{self.time_limits.get(level, 300)}s"

        if point_ids is None:
            selected_points = points
        else:
            selected_points = [points[i] for i in point_ids if 0 <= i < len(points)]

        if not selected_points:
            return f"{self.time_limits.get(level, 300)}s"

        time_limits = sorted({
            int(point.get("timeout", self.time_limits.get(level, 300)))
            for point in selected_points
        })
        if len(time_limits) == 1:
            return f"{time_limits[0]}s"
        return f"{time_limits[0]}-{time_limits[-1]}s ({time_limits})"

    def build_task_context(self, level: int, point_id: int) -> Dict[str, Any]:
        points = self.load_level_test_points(level)
        if point_id >= len(points):
            raise ValueError(f"Point {point_id} not found for level {level}")
        test_point = points[point_id]

        ref_text = ""
        if test_point.get("reference_text"):
            rt = test_point["reference_text"]
            ref_text = rt[0] if isinstance(rt, list) else rt

        ref_image_full_path = None
        if test_point.get("reference_image_path"):
            rp = test_point["reference_image_path"]
            ref_image_name = rp[0] if isinstance(rp, list) else rp
            ref_image_full_path = os.path.join(
                self.gym_rescue_root,
                "envs",
                "setting",
                "ref_image",
                ref_image_name,
            )

        injured_agent_id = test_point.get("injured_agent_id")
        if isinstance(injured_agent_id, list):
            injured_agent_id = injured_agent_id[0] if injured_agent_id else None
        if injured_agent_id is not None:
            try:
                injured_agent_id = int(injured_agent_id)
            except (TypeError, ValueError):
                injured_agent_id = None

        return TaskContext(
            env_id=self.resolve_env_id(test_point.get("env_id", self.fallback_env_id)),
            injured_pose=test_point["injured_player_loc"],
            injured_agent_id=injured_agent_id,
            stretcher_pose=test_point["stretcher_loc"],
            agent_pose=test_point["agent_loc"],
            ambulance_pose=test_point["ambulance_loc"],
            reference_text=ref_text,
            reference_image_path=ref_image_full_path,
            timeout=int(test_point.get("timeout", self.time_limits.get(level, 300))),
            level=level,
            point_id=point_id,
        ).as_dict()
