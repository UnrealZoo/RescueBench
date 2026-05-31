"""
NOMAD Agent
===========

NOMAD (Navigation with Diffusion) Agent实现。
继承自VINTAgent，使用扩散模型进行轨迹生成；交互由 benchmark 状态机负责。

使用方法:
    from agents.nomad_agent import NOMADAgent
    agent = NOMADAgent(device='cuda', topomap_dir='./topomap')
"""

# 直接从vint_agent导入NOMADAgent
# NOMADAgent继承自VINTAgent，只是model_name不同
from agents.vint_agent import NOMADAgent

__all__ = ['NOMADAgent']
