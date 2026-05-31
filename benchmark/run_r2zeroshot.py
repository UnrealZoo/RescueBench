#!/usr/bin/env python3
"""
R2ZeroShot 薄启动器
===================
使用 rescue_benchmark.py 基座 + R2ZeroShot Agent。

用法:
    # 默认: passthrough 模式, R2ZeroShot 完全自主 carry/drop
    python run_r2zeroshot.py --levels 1 --episodes 1 --passthrough --render

    # 恢复状态机距离判断 (100cm 默认阈值)
    python run_r2zeroshot.py --levels 2 3 --episodes 1 --no-passthrough

    # 关闭「UE 先终止时的终点几何对齐」（默认开启，一般勿关）
    python run_r2zeroshot.py --no-passthrough-env-term-sync ...

    # 查看渲染帧 (另一终端):
    feh --reload 0.5 --auto-zoom benchmark_results/_render_frames/latest_frame.jpg
"""

import sys
import os

# 确保 benchmark 目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rescue_benchmark import create_base_parser, run_benchmark_from_args
from agents.profiles import apply_model_profile_defaults


def main():
    parser = create_base_parser(
        description='R2ZeroShot 救援任务评估',
        epilog="""
示例:
    python run_r2zeroshot.py --levels 1 --episodes 1 --passthrough --render
    python run_r2zeroshot.py --levels 2 3 4 --episodes 3
    python run_r2zeroshot.py --no-passthrough --rescue-distance 200 --levels 1
        """
    )
    
    # R2ZeroShot 特定参数
    parser.add_argument('--cfg-coef', type=float, default=2.0,
                        help='ROCKET 的 CFG 引导系数')
    parser.add_argument('--ckpt-path', type=str, default=None,
                        help='ROCKET 权重路径 (None 表示使用默认路径)')
    parser.add_argument('--sa2va-path', type=str, default=None,
                        help='Sa2VA 模型路径 (None 表示使用默认路径)')
    parser.add_argument('--no-sa2va', action='store_true',
                        help='禁用 Sa2VA（仅导航）')
    parser.add_argument('--person-ref-path', type=str, default=None,
                        help='人物参考图像路径')
    parser.add_argument(
        '--no-passthrough-env-term-sync',
        dest='passthrough_env_term_geometry_sync',
        action='store_false',
        help='关闭：UE 先返回 termination 时用终点位姿补阶段二门限（默认开启，减轻 ENV_TERM_INCOMPLETE 与几何不一致；与 ckpt 无关）',
    )
    
    # R2ZeroShot 默认参数:
    # - passthrough=True: Agent 自主 carry/drop
    # - resolution=640x360: 对齐 ROCKET 16:9 输入，避免比例失真
    # - passthrough_env_term_geometry_sync=True: 与上项配套的终点几何对齐（换 ckpt 仍默认生效）
    apply_model_profile_defaults(parser, "r2zeroshot")
    
    args = parser.parse_args()
    
    # 创建 R2ZeroShot Agent
    from agents.r2zeroshot_agent import R2ZeroShotAgent
    agent = R2ZeroShotAgent(
        cfg_coef=args.cfg_coef,
        ckpt_path=args.ckpt_path,
        sa2va_path=args.sa2va_path,
        use_sa2va=not args.no_sa2va,
        person_ref_path=args.person_ref_path,
    )
    
    run_benchmark_from_args(args, agent, model_name='r2zeroshot')


if __name__ == '__main__':
    main()
