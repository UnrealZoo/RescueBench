#!/usr/bin/env python3
"""
ViNT / NOMAD 统一启动器（visualnav-transformer）
===============================================
同一作者、同一部署管线，仅 `model_name` 与 checkpoint 不同；共享拓扑图与分辨率等参数。

用法:
    python run_visualnav.py --model nomad --topomap-dir ./rescue_topomaps --levels 2 --episodes 1 --render
    python run_visualnav.py --model vint --topomap-dir ./my_topomap --levels 1 --episodes 1

多地图目录约定见 agents/topomap_utils.py 与 example/VINT/collect_rescue_topomap.py --topomap-root。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rescue_benchmark import create_base_parser, run_benchmark_from_args
from agents.profiles import apply_model_profile_defaults


def main():
    parser = create_base_parser(
        description='ViNT / NOMAD 救援任务评估（visualnav-transformer）',
        epilog="""
示例:
  python run_visualnav.py --model nomad --topomap-dir ./rescue_topomaps --levels 2 --episodes 1 --render
  python run_visualnav.py -m vint --topomap-dir ./single_map_topomap --levels 1 --episodes 1 --device cuda

拓扑:
  单地图: --topomap-dir 指向该地图的 topomap 目录（含 images/0.png …）
  多地图: --topomap-dir 指向根目录，其下为 UnrealRescue-* 子目录（与采集 --topomap-root 一致）
        """,
    )
    parser.add_argument(
        '--model', '-m',
        choices=['vint', 'nomad'],
        default='nomad',
        help='视觉导航模型：vint 或 nomad（默认 nomad）',
    )
    parser.add_argument(
        '--topomap-dir',
        type=str,
        default=None,
        help='拓扑目录：单地图=该地图 topomap 文件夹；多地图=含 UnrealRescue-* 子目录的根',
    )
    parser.add_argument(
        '--waypoint-idx',
        type=int,
        default=5,
        help='从模型输出的 waypoint 序列中取第几个点作为当前动作',
    )
    apply_model_profile_defaults(parser, "nomad")

    args = parser.parse_args()
    model_name = args.model

    if model_name == 'nomad':
        from agents.nomad_agent import NOMADAgent
        agent = NOMADAgent(
            device=args.device,
            topomap_dir=args.topomap_dir,
            waypoint_idx=args.waypoint_idx,
        )
    else:
        from agents.vint_agent import VINTAgent
        agent = VINTAgent(
            device=args.device,
            topomap_dir=args.topomap_dir,
            waypoint_idx=args.waypoint_idx,
        )

    run_benchmark_from_args(args, agent, model_name=model_name)


if __name__ == '__main__':
    main()
