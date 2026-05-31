#!/usr/bin/env python3
"""
Uni-NaVid 薄启动器
==================
使用 rescue_benchmark.py 基座 + Uni-NaVid Agent。

用法:
    # 基础运行 (默认 224x224, 状态机主动模式, 100cm 阈值)
    python run_uni_navid.py --levels 1 --episodes 1 --render

    # 调整动作参数
    python run_uni_navid.py --forward-velocity 60 --turn-angle 20 --levels 2 3

    # 查看渲染帧 (另一终端):
    feh --reload 0.5 --auto-zoom benchmark_results/_render_frames/latest_frame.jpg
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rescue_benchmark import create_base_parser, run_benchmark_from_args
from agents.profiles import apply_model_profile_defaults


def main():
    parser = create_base_parser(
        description='Uni-NaVid 救援任务评估',
        epilog="""
示例:
    python run_uni_navid.py --levels 1 --episodes 1 --render
    python run_uni_navid.py --levels 2 3 4 --episodes 3
    python run_uni_navid.py --forward-velocity 60 --turn-angle 20 --levels 1
        """
    )
    
    # Uni-NaVid 特定参数
    parser.add_argument('--model-path', type=str, default=None,
                        help='Uni-NaVid 模型权重路径 (None 表示使用默认路径)')
    parser.add_argument('--forward-velocity', type=float, default=100.0,
                        help='前进动作速度 (0~100)')
    parser.add_argument('--turn-angle', type=float, default=30.0,
                        help='左/右转动作角度 (0~30)')
    parser.add_argument('--turn-velocity', type=float, default=0.0,
                        help='转向时前进速度 (0=原地转向)')
    parser.add_argument('--no-reset-on-phase-switch', action='store_true',
                        help='阶段切换时保留模型上下文（默认会重置）')
    parser.add_argument('--phase2-instruction', type=str, default=None,
                        help='自定义 Phase2 指令文本')
    
    # Uni-NaVid 默认 224x224 分辨率
    apply_model_profile_defaults(parser, "uni_navid")
    
    args = parser.parse_args()
    
    # 创建 Uni-NaVid Agent
    from agents.uninavid_agent import UniNaVidAgent
    agent = UniNaVidAgent(
        model_path=args.model_path,
        device=args.device,
        forward_velocity=args.forward_velocity,
        turn_angle=args.turn_angle,
        turn_velocity=args.turn_velocity,
        reset_on_phase_switch=not args.no_reset_on_phase_switch,
        phase2_instruction=args.phase2_instruction,
    )
    
    run_benchmark_from_args(args, agent, model_name='uni_navid')


if __name__ == '__main__':
    main()
