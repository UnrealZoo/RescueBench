#!/usr/bin/env python3
"""
OmniNav 薄启动器
================
使用 rescue_benchmark.py 基座 + OmniNav Agent。

默认对齐 infer_ovon（waypoint + action_former + 历史帧 + 三路图）。
可选 --inference-mode text 使用旧版 generate 文本动作。

示例:
    conda activate omninav
    python benchmark/run_omninav.py --levels 1 --episodes 1 --render

    conda activate omninav
    python benchmark/run_omninav.py --levels 2 3 --episodes 1 --model-path baseline_model/OmniNav/ckpt/OmniNav
"""

import os
import sys

_BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_BENCH_DIR)
sys.path.insert(0, _BENCH_DIR)
# 确保 waypoint 模式在 import 任何 transformers 子模块前能优先使用 OmniNav 补丁包（双保险，主逻辑在 omninav_waypoint_model）
_OMNI_TRANS = os.path.join(_ROOT, "baseline_model", "OmniNav", "train_code", "transformers-main", "src")
if os.path.isdir(_OMNI_TRANS) and _OMNI_TRANS not in sys.path:
    sys.path.insert(0, _OMNI_TRANS)

from rescue_benchmark import create_base_parser, run_benchmark_from_args
from agents.profiles import apply_model_profile_defaults


def main():
    parser = create_base_parser(
        description="OmniNav 救援任务评估",
        epilog="""
示例:
    python benchmark/run_omninav.py --levels 1 --episodes 1 --render
    python benchmark/run_omninav.py --levels 2 3 4 --episodes 1
    python benchmark/run_omninav.py --model-path baseline_model/OmniNav/ckpt/OmniNav --device cuda
        """,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="OmniNav 权重目录 (默认使用 baseline_model/OmniNav/ckpt/OmniNav)",
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        choices=["waypoint", "text"],
        default="waypoint",
        help="waypoint=官方 OVON forward(action_former)；text=旧版 generate 四向词",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        help="仅 waypoint 模式：注意力实现；失败时自动尝试 sdpa",
    )
    parser.add_argument(
        "--waypoint-dist-scale",
        type=float,
        default=100.0,
        help="waypoint：dist×该系数为速度中间量，再经下限/上限夹紧",
    )
    parser.add_argument(
        "--waypoint-speed-floor",
        type=float,
        default=70.0,
        help="waypoint：线速度下限（避免 dist 过小时几乎不走）",
    )
    parser.add_argument(
        "--max-turn-deg",
        type=float,
        default=30.0,
        help="waypoint：转向角裁剪到 [-该值, +该值]（与 Mixed 连续动作常见 ±30° 对齐）",
    )
    parser.add_argument(
        "--waypoint-hz-index",
        type=int,
        default=3,
        choices=[0, 1, 2, 3, 4],
        help="仅 waypoint 模式：使用第几个 horizon 点与对应 recover（0-based，默认 3=第4个点，4=第5个点-官方默认第0个点）",
    )
    parser.add_argument(
        "--rgb-input",
        action="store_true",
        help="声明 benchmark 输入已经是 RGB；默认按 BGR 处理并在 agent 内转换",
    )
    parser.add_argument(
        "--disable-multi-view",
        action="store_true",
        help="禁用 OmniNav 的同相机三视角采样，退回单视角输入",
    )
    parser.add_argument(
        "--view-angles",
        type=float,
        nargs="+",
        default=[-45.0, 0.0, 45.0],
        help="多视角采样角度（相对默认相机朝向，默认 -45 0 45）",
    )
    parser.add_argument(
        "--view-settle-time",
        type=float,
        default=0.02,
        help="每次切换相机朝向后的等待时间（秒）",
    )
    parser.add_argument(
        "--forward-velocity",
        type=float,
        default=100.0,
        help="waypoint：线速度上限（且不超过环境 100）；text：FORWARD 线速度",
    )
    parser.add_argument(
        "--turn-angle",
        type=float,
        default=20.0,
        help="仅 text：LEFT/RIGHT 转角 (°)",
    )
    parser.add_argument(
        "--turn-velocity",
        type=float,
        default=0.0,
        help="仅 text：LEFT/RIGHT 时前进速度",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="每次生成的最大 token 数",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="启用采样生成（默认贪心）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度，仅在 --do-sample 时生效",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="减少 agent 日志输出",
    )
    parser.add_argument(
        "--log-waypoint",
        action="store_true",
        help="waypoint 模式：每步打印模型原始输出与映射后的 move_action（调速度/转角映射用）",
    )

    apply_model_profile_defaults(parser, "omninav")
    args = parser.parse_args()

    from agents.omninav_agent import OmniNavAgent

    agent = OmniNavAgent(
        model_path=args.model_path,
        device=args.device,
        bgr_input=not args.rgb_input,
        inference_mode=args.inference_mode,
        attn_implementation=args.attn_implementation,
        multi_view=not args.disable_multi_view,
        multi_view_angles=args.view_angles,
        multi_view_settle_time=args.view_settle_time,
        forward_velocity=args.forward_velocity,
        turn_angle=args.turn_angle,
        turn_velocity=args.turn_velocity,
        waypoint_dist_scale=args.waypoint_dist_scale,
        waypoint_speed_floor=args.waypoint_speed_floor,
        waypoint_max_turn_deg=args.max_turn_deg,
        waypoint_horizon_index=args.waypoint_hz_index,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        verbose=not args.quiet,
        log_waypoint_inference=args.log_waypoint,
    )

    run_benchmark_from_args(args, agent, model_name="omninav")


if __name__ == "__main__":
    main()
