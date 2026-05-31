#!/usr/bin/env python3
"""
Apex 薄启动器
=============
使用 rescue_benchmark.py 基座 + Apex Agent。

官方模型特点:
    - 内部自带 carry/drop 决策
    - 依赖相对路径 checkpoints/
    - 输入为 base64 图像，输出为 Unreal 风格动作字典

因此默认:
    - passthrough=True
    - 分辨率保持 640x640（与官方 YOLO imgsz=640 对齐）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rescue_benchmark import create_base_parser, run_benchmark_from_args
from agents.profiles import apply_model_profile_defaults


def main():
    parser = create_base_parser(
        description="Apex 救援任务评估",
        epilog="""
示例:
    python run_apex.py --levels 1 --episodes 1 --render
    python run_apex.py --levels 2 3 4 --episodes 1 --enable-trajectory
    python run_apex.py --workspace ../baseline_model/Apexcode/apex_code --device cuda
        """,
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Apex 赛队代码根目录（默认使用仓库内 baseline_model/Apexcode/apex_code）",
    )

    apply_model_profile_defaults(parser, "apex")

    args = parser.parse_args()

    from agents.apex_agent import ApexAgent

    agent = ApexAgent(workspace=args.workspace)
    run_benchmark_from_args(args, agent, model_name="apex")


if __name__ == "__main__":
    main()
