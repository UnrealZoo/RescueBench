"""Environment lifecycle helpers for the Rescue benchmark."""

from typing import Any, Dict, Tuple


class EnvManager:
    """Create, reuse, close, and configure benchmark environments."""

    def __init__(
        self,
        resolution: Tuple[int, int],
        render_quality: int = 2,
        offscreen: bool = True,
        multiagent_env: bool = False,
    ):
        self.resolution = resolution
        self.render_quality = render_quality
        self.offscreen = offscreen
        self.multiagent_env = bool(multiagent_env)

        self.env = None
        self.current_env_id = None
        self.current_level = None

    def create_env(self, level: int, env_id: str):
        gym = __import__("gym")
        config_ue = __import__("gym_rescue.envs.wrappers.configUE", fromlist=["ConfigUEWrapper"])
        env = gym.make(
            env_id,
            action_type="Mixed",
            observation_type="Color",
            reset_type=level,
        )
        env = config_ue.ConfigUEWrapper(
            env,
            offscreen=self.offscreen,
            resolution=self.resolution,
            render_quality=self.render_quality,
        )
        if level > 0 and not self.multiagent_env:
            population = __import__(
                "gym_rescue.envs.wrappers.population",
                fromlist=["RandomPopulationWrapper"],
            )
            env = population.RandomPopulationWrapper(env, 1, 1, random_target=False)

        return env

    def close_env(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = None
        self.current_env_id = None
        self.current_level = None

    def ensure_env(self, env_id: str, level: int) -> bool:
        """Ensure the requested env is active. Returns True when a new env was created."""

        if self.env is not None and self.current_env_id == env_id and self.current_level == level:
            return False

        self.close_env()
        print(f"[Setup] Creating environment: {env_id} (L{level})")
        self.env = self.create_env(level, env_id)
        self.current_env_id = env_id
        self.current_level = level
        return True

    def apply_task_context(self, task_context: Dict[str, Any]) -> None:
        env = self.env.unwrapped
        env.injured_player_pose = task_context["injured_pose"]
        env.injured_agent_appid = task_context.get("injured_agent_id")
        env.rescue_pose = task_context["stretcher_pose"]
        env.agent_pose = task_context["agent_pose"]
        env.ambulance_pose = task_context["ambulance_pose"]

    def compose_env_action(self, final_action: Tuple, extra_info: Dict[str, Any]):
        if not self.multiagent_env:
            return [final_action]

        joint_action = extra_info.get("joint_action")
        if isinstance(joint_action, dict):
            env_action = dict(joint_action)
            env_action["player"] = final_action
            return env_action

        if isinstance(joint_action, (list, tuple)):
            env_action = list(joint_action)
            if env_action:
                env_action[0] = final_action
            else:
                env_action = [final_action]
            return env_action

        return {"player": final_action}

    def get_object_pose(self, object_id: int):
        try:
            return list(self.env.unwrapped.obj_poses[object_id])
        except Exception:
            return None

    def get_current_pose(self):
        try:
            protagonist_id = self.env.unwrapped.protagonist_id
            return self.get_object_pose(protagonist_id) or [0.0] * 6
        except Exception:
            return [0.0] * 6

    def get_current_drone_pose(self):
        try:
            env = self.env.unwrapped
            drone_id = getattr(env, "drone_id", None)
            if drone_id is None or drone_id < 0 or drone_id >= len(env.obj_poses):
                return None
            return self.get_object_pose(drone_id)
        except Exception:
            return None

    def get_current_metric_position(self):
        return self.get_current_pose()[:3]

    def get_current_interaction_position(self):
        try:
            env = self.env.unwrapped
            protagonist_id = getattr(env, "protagonist_id", 0)
            cam_id = env.cam_list[protagonist_id]
            return list(env.unrealcv.get_cam_pose(cam_id)[:3])
        except Exception:
            return self.get_current_metric_position()
