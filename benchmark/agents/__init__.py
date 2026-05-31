"""Benchmark agent adapters."""
"""
Agent模块
=========

这里存放各种模型的Agent实现。
每个Agent需要继承 BaseAgent 类并实现 act() 方法。

使用方法:
    from agents.vint_agent import VINTAgent
    agent = VINTAgent(device='cuda', topomap_dir='./topomap')
"""

from .agent_base import BaseAgent, RandomAgent
from .factory import AGENT_REGISTRY, get_agent, get_agent_from_cli_args
from .profiles import MODEL_PROFILES, ModelProfile, apply_model_profile_defaults, get_model_profile

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "AGENT_REGISTRY",
    "get_agent",
    "get_agent_from_cli_args",
    "MODEL_PROFILES",
    "ModelProfile",
    "apply_model_profile_defaults",
    "get_model_profile",
]
