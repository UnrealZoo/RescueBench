"""Shared benchmark agent interfaces."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseAgent(ABC):
    """Agent 基类。子类必须实现 act()，可选重写 reset() / on_episode_end()。"""

    @abstractmethod
    def act(self, observation: Any, info: Dict) -> Tuple[Any, Dict]:
        """返回 (action, extra_info)。action: (move, head[, anim])"""
        pass

    def reset(self):
        """Episode 开始时重置内部状态"""
        pass

    def prepare_episode(self, task_context: Dict[str, Any]) -> None:
        """在 env.reset() 之前调用；子类可在此绑定本局 task_context（通用扩展点，默认无操作）。"""
        pass

    def prepare_step_inputs(self, env, observation: Any, info: Dict) -> Tuple[Any, Dict]:
        """在 act() 之前调用；可选扩展当前步的观测/附加信息。"""
        return observation, info

    def on_episode_end(self, success: bool, metrics):
        """Episode 结束时回调"""
        pass


class RandomAgent(BaseAgent):
    """随机智能体 - 用于基线测试"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation: Any, info: Dict) -> Tuple[Any, Dict]:
        action = self.action_space.sample()
        return action, {"action_type": "random"}
