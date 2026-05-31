"""
Agent模板文件 (新版接口)
========================

复制这个文件并修改来创建你自己的模型Agent。

重要变化:
- Agent只需要返回导航动作 (move_action, head_action)
- 不需要管理任务状态（carry/drop由状态机决定）
- info中包含任务上下文（task_phase, target_pose, reference_text）

使用步骤:
1. 复制此文件并重命名 (如 my_model_agent.py)
2. 修改类名和实现
3. 在 agents/factory.py 的 AGENT_REGISTRY 中注册一行（见 get_agent）
"""

import numpy as np
from typing import Dict, Any, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.agent_base import BaseAgent
from core.metrics import EpisodeMetrics


class TemplateAgent(BaseAgent):
    """
    模板Agent（新版接口）
    
    你需要实现的方法:
        - __init__(): 加载你的模型
        - act(): 根据观测返回导航动作
        
    可选重写的方法:
        - reset(): 每个episode开始时重置状态
        - on_episode_end(): episode结束时的回调
    """
    
    def __init__(self, model_path: str = None, device: str = 'cuda', **kwargs):
        """
        初始化你的模型
        
        Args:
            model_path: 模型权重路径
            device: 运行设备 ('cuda' 或 'cpu')
            **kwargs: 其他参数
        """
        self.device = device
        self.model_path = model_path
        
        # 在这里加载你的模型
        # self.model = load_your_model(model_path)
        
        self.context_queue = []
        self.context_size = 5
        
        print(f"[TemplateAgent] Initialized with device={device}")
    
    def act(self, observation: np.ndarray, info: Dict) -> Tuple[Any, Dict]:
        """
        根据观测返回导航动作
        
        Args:
            observation: RGB图像, numpy array, shape [H, W, 3], dtype uint8
            info: 环境信息字典，包含:
                - 'Pose': 当前位姿 [x, y, z, roll, yaw, pitch]
                - 'picked': bool, 是否已经抱起伤员
                
                【新增任务上下文】:
                - 'task_phase': str, 当前任务阶段
                    - 'find_injured': 正在寻找伤员
                    - 'find_stretcher': 正在寻找担架
                - 'target_pose': list, 当前目标位置 [x, y, z, ...]
                - 'reference_text': str, 任务描述文本（来自JSON配置）
                - 'state_machine_state': str, 状态机状态
        
        Returns:
            action: 导航动作，支持两种格式:
            
                格式1（推荐）: (move_action, head_action)
                    - 只返回导航动作，carry/drop由状态机决定
                    
                格式2（兼容）: (move_action, head_action, animation_action)
                    - 完全控制模式，适合需要精细控制的模型
                
                move_action: np.array([angle, velocity])
                    - angle: 转向角度 (-30 到 30)
                    - velocity: 速度 (-100 到 100, 正为前进)
                
                head_action: int (0, 1, 2)
                    - 0: 不动
                    - 1: 向右看30度
                    - 2: 向左看30度
            
            extra_info: 额外信息字典(用于调试和可视化)
                - 'trajectory': 预测的轨迹点（可选，用于可视化）
        """
        # 获取任务上下文
        task_phase = info.get('task_phase', 'find_injured')
        target_pose = info.get('target_pose')
        reference_text = info.get('reference_text', '')
        is_picked = info.get('picked', False)
        
        # 你的导航逻辑
        # 示例：根据目标位置计算导航方向
        move_action = np.array([0.0, 50.0])  # 直走
        head_action = 0  # 不转头
        
        # 只返回导航动作，不需要返回animation_action
        # 状态机会根据距离自动触发carry/drop
        action = (move_action, head_action)
        
        extra_info = {
            'task_phase': task_phase,
            'is_picked': is_picked,
            'action_type': 'template'
        }
        
        return action, extra_info
    
    def reset(self):
        """Episode开始时调用，重置内部状态"""
        self.context_queue = []
    
    def on_episode_end(self, success: bool, metrics: EpisodeMetrics):
        """
        Episode结束时调用
        
        Args:
            success: 是否成功完成任务
            metrics: Episode的评估指标（现在包含阶段性指标）
                - metrics.phase1_success: 阶段1是否成功
                - metrics.phase2_success: 阶段2是否成功
                - metrics.failure_reason: 失败原因
        """
        status = "SUCCESS" if success else "FAILED"
        reason = f" ({metrics.failure_reason})" if metrics.failure_reason else ""
        print(f"[TemplateAgent] Episode {status}{reason}, "
              f"steps={metrics.steps}, time={metrics.time_cost:.2f}s")


if __name__ == '__main__':
    agent = TemplateAgent(device='cpu')
    
    fake_obs = np.zeros((320, 320, 3), dtype=np.uint8)
    fake_info = {
        'Pose': [0, 0, 0, 0, 0, 0],
        'picked': False,
        'task_phase': 'find_injured',
        'target_pose': [100, 200, 50, 0, 0, 0],
        'reference_text': 'Find the injured person near the car.'
    }
    
    action, extra = agent.act(fake_obs, fake_info)
    print(f"Action: {action}")
    print(f"Extra info: {extra}")
