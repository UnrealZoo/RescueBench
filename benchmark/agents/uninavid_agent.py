"""
Uni-NaVid Agent for Rescue Benchmark
=====================================

将 Uni-NaVid 的 VLN 导航能力适配到救援任务评测框架。

核心适配逻辑:
    1. 动作队列: Uni-NaVid 一次输出4个离散动作，用队列逐步执行
    2. 动作映射: forward/left/right/stop -> np.array([angle, velocity])
    3. 阶段切换: Phase1(找伤员) -> Phase2(找担架) 时切换指令并重置模型上下文
    4. 输入重组: benchmark的 (observation, info) -> Uni-NaVid的 {"observations", "instruction"}

使用方法:
    python run_uni_navid.py --levels 1 2 --episodes 1 --resolution 224 224

作者: Auto-generated
日期: 2026-02
"""

import numpy as np
import sys
import os
import time
from typing import Dict, Any, Tuple, Optional, List

# === 路径设置 ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BENCHMARK_DIR)

from agents.agent_base import BaseAgent
from core.metrics import EpisodeMetrics

# Uni-NaVid 仓库根目录
UNINAVID_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(BENCHMARK_DIR)),  # -> Offline_RL_Active_Tracking/
    'Uni-NaVid'
)
if UNINAVID_ROOT not in sys.path:
    sys.path.insert(0, UNINAVID_ROOT)


class UniNaVidAgent(BaseAgent):
    """
    Uni-NaVid Agent 适配器
    
    ┌──────────────────────────────────────────────────────────┐
    │  Benchmark 主循环 (每步)                                  │
    │                                                          │
    │  obs, info ──► UniNaVidAgent.act(obs, info)              │
    │                    │                                     │
    │                    ├─ 队列非空? → 取出1个动作 → 返回      │
    │                    │                                     │
    │                    └─ 队列空?   → 调用 Uni-NaVid 模型     │
    │                                    ↓                     │
    │                              得到4个离散动作              │
    │                              ["forward","left",...]       │
    │                                    ↓                     │
    │                              入队列 → 取出第1个 → 返回    │
    │                                                          │
    │  返回: (np.array([angle, vel]), head_action), extra_info │
    └──────────────────────────────────────────────────────────┘
    
    阶段切换策略 (可配置):
        - reset_on_phase_switch=True:  清除历史帧缓存 + KV cache (默认)
        - reset_on_phase_switch=False: 仅切换指令文本，保留空间记忆
    """
    
    # Phase2 的默认指令 (伤员已抱起, 需要找担架)
    DEFAULT_PHASE2_INSTRUCTION = (
        "You are carrying an injured person. Navigate to the yellow ambulance stretcher to place the person down. The stretcher is near the ambulance parked on the street. Find the stretcher and move towards it."
    )
    
    # Phase1 的兜底指令 (如果 JSON 中没有 reference_text)
    DEFAULT_PHASE1_INSTRUCTION = (
        "Find the injured person lying on the ground and navigate towards them."
    )
    
    def __init__(
        self,
        model_path: str = None,
        device: str = 'cuda',
        # --- 动作映射参数 (可调) ---
        forward_velocity: float = 80.0,
        turn_angle: float = 30.0,
        turn_velocity: float = 0.0,
        # --- 阶段切换策略 ---
        reset_on_phase_switch: bool = True,
        phase2_instruction: str = None,
        # --- 调试选项 ---
        verbose: bool = True,
        **kwargs
    ):
        """
        Args:
            model_path: Uni-NaVid 模型权重路径
                默认: Uni-NaVid/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2
            device: 推理设备 ('cuda' 或 'cpu')
            
            forward_velocity: forward 动作映射的速度值 (0~100), 越大走得越快
                → 调大: 每步走得远, 到达快但可能冲过头
                → 调小: 每步走得近, 精确但慢
            turn_angle: left/right 动作映射的转向角度 (0~30)
                → 调大: 转弯幅度大, 灵活但可能过度转向
                → 调小: 转弯幅度小, 平滑但转弯慢
            turn_velocity: 转向时是否同时前进 (0=原地转, >0=边走边转)
                → 0: 原地转(Habitat风格), 动作精确
                → >0: 边走边转, 更自然但轨迹不同
            
            reset_on_phase_switch: Phase切换时是否重置模型上下文
                → True:  清除所有历史, 干净但失去空间记忆
                → False: 保留历史, 可能帮助定位但也可能干扰
            phase2_instruction: 自定义Phase2指令文本
            verbose: 是否打印详细日志
        """
        self.device = device
        self.verbose = verbose
        self.reset_on_phase_switch = reset_on_phase_switch
        self.phase2_instruction = phase2_instruction or self.DEFAULT_PHASE2_INSTRUCTION
        
        # 动作映射表 (离散 -> 连续)
        self.action_map = {
            'forward': np.array([0.0, forward_velocity]),
            'left':    np.array([-turn_angle, turn_velocity]),
            'right':   np.array([turn_angle, turn_velocity]),
            'stop':    np.array([0.0, 0.0]),
        }
        
        # 记录配置参数供调试
        self.config = {
            'forward_velocity': forward_velocity,
            'turn_angle': turn_angle,
            'turn_velocity': turn_velocity,
            'reset_on_phase_switch': reset_on_phase_switch,
        }
        
        # === 加载 Uni-NaVid 模型 ===
        if model_path is None:
            model_path = os.path.join(
                UNINAVID_ROOT, 'model_zoo',
                'uninavid-7b-full-224-video-fps-1-grid-2'
            )
        
        self._log(f"Loading model from: {model_path}")
        # 临时切换工作目录到 Uni-NaVid 根目录
        # 因为模型 config.json 中使用了相对路径（如 ./model_zoo/eva_vit_g.pth）
        original_cwd = os.getcwd()
        os.chdir(UNINAVID_ROOT)
        try:
            from offline_eval_uninavid import UniNaVid_Agent
            self.navid_agent = UniNaVid_Agent(model_path)
        finally:
            os.chdir(original_cwd)
        self._log(f"Model loaded successfully on {device}")
        self._log(f"Config: {self.config}")
        
        # === 运行时状态 ===
        self.action_queue: List[str] = []        # 待执行的离散动作队列
        self.step_count: int = 0                 # 当前 episode 步数
        self.current_phase: Optional[str] = None # 当前阶段
        self.current_instruction: str = ""       # 当前使用的指令
        self.last_prediction: List[str] = []     # 最近一次模型预测
        self.total_inferences: int = 0           # 模型调用总次数
        self.total_inference_time: float = 0.0   # 模型推理总耗时
    
    def _log(self, msg: str):
        """条件日志输出"""
        if self.verbose:
            print(f"[UniNaVidAgent] {msg}")
    
    def reset(self):
        """Episode 开始时重置所有状态"""
        self.action_queue = []
        self.step_count = 0
        self.current_phase = None
        self.current_instruction = ""
        self.last_prediction = []
        self.total_inferences = 0
        self.total_inference_time = 0.0
        
        # 重置 Uni-NaVid 内部状态 (清除历史帧 + KV cache)
        self.navid_agent.reset(task_type='vln')
        self._log("Reset for new episode")
    
    def _get_instruction(self, info: Dict) -> str:
        """
        根据当前阶段获取指令文本
        
        Phase1: 优先使用 JSON 中的 reference_text (描述伤员位置)
        Phase2: 使用担架寻找指令
        """
        task_phase = info.get('task_phase', 'find_injured')
        reference_text = info.get('reference_text', '')
        
        if task_phase == 'find_injured':
            if reference_text:
                return reference_text
            else:
                return self.DEFAULT_PHASE1_INSTRUCTION
        else:
            # Phase2: 找担架
            return self.phase2_instruction
    
    def _predict(self, observation: np.ndarray, instruction: str) -> List[str]:
        """
        调用 Uni-NaVid 模型预测动作序列
        
        Args:
            observation: RGB 图像 (numpy, [H, W, 3])
            instruction: 自然语言指令
            
        Returns:
            有效离散动作列表 (已过滤无效动作)
        """
        data = {
            "observations": observation,
            "instruction": instruction
        }
        
        t_start = time.time()
        result = self.navid_agent.act(data)
        t_elapsed = time.time() - t_start
        
        self.total_inferences += 1
        self.total_inference_time += t_elapsed
        
        actions = result.get('actions', [])
        
        # 过滤无效动作 (模型可能输出非标准词汇)
        valid_actions = [a for a in actions if a in self.action_map]
        
        if not valid_actions:
            self._log(f"WARNING: No valid actions in prediction: {actions}, using 'forward' as fallback")
            valid_actions = ['forward']
        
        self._log(f"Step {self.step_count}: Model inference #{self.total_inferences} "
                   f"({t_elapsed:.2f}s) -> {valid_actions}")
        
        return valid_actions
    
    def act(self, observation: np.ndarray, info: Dict) -> Tuple[Any, Dict]:
        """
        主决策函数
        
        流程:
        1. 检测阶段切换 → 可选重置模型上下文
        2. 动作队列为空 → 调用模型预测4个新动作
        3. 从队列头部取出1个动作 → 映射为连续值 → 返回
        
        Args:
            observation: RGB图像 numpy array, shape [H, W, 3], dtype uint8
            info: 环境信息字典, 包含 task_phase, reference_text 等
            
        Returns:
            (move_action, head_action): 导航动作
            extra_info: 调试信息字典
        """
        self.step_count += 1
        task_phase = info.get('task_phase', 'find_injured')
        
        # === 1. 检测阶段切换 ===
        if self.current_phase is not None and task_phase != self.current_phase:
            self._log(f"Phase switch: {self.current_phase} -> {task_phase}")
            
            if self.reset_on_phase_switch:
                # 清除历史帧缓存和 KV cache
                self.navid_agent.reset(task_type='vln')
                self._log("  -> Model context reset (reset_on_phase_switch=True)")
            else:
                self._log("  -> Keeping model context (reset_on_phase_switch=False)")
            
            # 无论是否 reset 模型, 都清空动作队列 (旧动作已不适用)
            self.action_queue = []
        
        self.current_phase = task_phase
        
        # === 2. 获取当前指令 ===
        instruction = self._get_instruction(info)
        self.current_instruction = instruction
        
        # === 3. 队列为空时调用模型 ===
        if len(self.action_queue) == 0:
            actions = self._predict(observation, instruction)
            self.action_queue = actions
            self.last_prediction = actions.copy()
        
        # === 4. 从队列取出动作并映射 ===
        action_name = self.action_queue.pop(0)
        move_action = self.action_map.get(action_name, np.array([0.0, 0.0])).copy()
        head_action = 0  # Uni-NaVid 不控制头部转动
        
        # 构建调试信息
        extra_info = {
            'action_name': action_name,
            'task_phase': task_phase,
            'queue_remaining': len(self.action_queue),
            'instruction_preview': instruction[:80] + '...' if len(instruction) > 80 else instruction,
            'last_prediction': self.last_prediction,
            'total_inferences': self.total_inferences,
            'avg_inference_time': (self.total_inference_time / self.total_inferences
                                   if self.total_inferences > 0 else 0),
        }
        
        return (move_action, head_action), extra_info
    
    def on_episode_end(self, success: bool, metrics: EpisodeMetrics):
        """Episode 结束回调"""
        status = "SUCCESS" if success else "FAILED"
        reason = f" ({metrics.failure_reason})" if metrics.failure_reason else ""
        avg_time = (self.total_inference_time / self.total_inferences
                    if self.total_inferences > 0 else 0)
        
        self._log(
            f"{status}{reason} | "
            f"steps={metrics.steps}, time={metrics.time_cost:.1f}s | "
            f"phase1={'OK' if metrics.phase1_success else 'FAIL'}, "
            f"phase2={'OK' if metrics.phase2_success else 'FAIL'} | "
            f"model_calls={self.total_inferences}, "
            f"avg_inference={avg_time:.2f}s"
        )


# ============================================================================
# 独立测试
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("UniNaVidAgent - Standalone Test (CPU, no model)")
    print("=" * 60)
    
    # 仅测试接口兼容性 (不加载模型)
    print("\n[Test] Checking interface compatibility...")
    
    import inspect
    sig = inspect.signature(UniNaVidAgent.act)
    print(f"  act() signature: {sig}")
    
    sig_reset = inspect.signature(UniNaVidAgent.reset)
    print(f"  reset() signature: {sig_reset}")
    
    print("\n[Test] Action mapping:")
    agent_config = {
        'forward_velocity': 80.0,
        'turn_angle': 30.0,
        'turn_velocity': 0.0,
    }
    action_map = {
        'forward': np.array([0.0, agent_config['forward_velocity']]),
        'left':    np.array([-agent_config['turn_angle'], agent_config['turn_velocity']]),
        'right':   np.array([agent_config['turn_angle'], agent_config['turn_velocity']]),
        'stop':    np.array([0.0, 0.0]),
    }
    for name, action in action_map.items():
        print(f"  '{name}' -> angle={action[0]:+.0f}, velocity={action[1]:.0f}")
    
    print("\n[Test] To run with model:")
    print("  python run_uni_navid.py --levels 1 --episodes 1 --resolution 224 224")
    print("\nDone.")
