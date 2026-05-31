#!/usr/bin/env python3
"""
SeePointFly + CityWalker 薄启动器
================================
无人机负责侦察，CityWalker 负责 carry/drop 前后的地面导航。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rescue_benchmark import create_base_parser, run_benchmark_from_args
from agents.profiles import apply_model_profile_defaults


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CITYWALKER_ROOT = os.path.join(PROJECT_ROOT, "baseline_model", "CityWalker")
SEEPOINTFLY_ROOT = os.path.join(PROJECT_ROOT, "baseline_model", "see-point-fly")

DEFAULT_CITYWALKER_CONFIG = os.path.join(CITYWALKER_ROOT, "config", "citywalk_2000hr.yaml")
DEFAULT_SPF_CONFIG = os.path.join(SEEPOINTFLY_ROOT, "config_sim.yaml")


def main():
    parser = create_base_parser(
        description="SeePointFly + CityWalker 协同救援评估",
    )
    parser.add_argument(
        "--citywalker-config-path",
        type=str,
        default=DEFAULT_CITYWALKER_CONFIG,
        help="CityWalker 配置文件路径",
    )
    parser.add_argument(
        "--citywalker-checkpoint-path",
        type=str,
        required=True,
        help="CityWalker checkpoint 路径",
    )
    parser.add_argument(
        "--spf-config-path",
        type=str,
        default=DEFAULT_SPF_CONFIG,
        help="SeePointFly 配置文件路径",
    )
    parser.add_argument(
        "--drone-trigger-radius-m",
        type=float,
        default=4.0,
        help="无人机判定找到目标的 2D 距离阈值 (m)",
    )
    parser.add_argument(
        "--drone-trigger-height-m",
        type=float,
        default=15.0,
        help="无人机判定找到目标的 Z 高度差阈值 (m)",
    )
    parser.add_argument(
        "--drone-height-offset-cm",
        type=float,
        default=250.0,
        help="无人机相对地面 agent 的初始高度偏移 (cm)",
    )
    parser.add_argument(
        "--drone-max-speed",
        type=float,
        default=0.2,
        help="无人机连续动作最大平移幅度",
    )
    parser.add_argument(
        "--drone-max-z-speed",
        type=float,
        default=0.08,
        help="无人机连续动作最大 Z 轴升降幅度",
    )
    parser.add_argument(
        "--drone-max-yaw",
        type=float,
        default=1.0,
        help="无人机连续动作最大 yaw 幅度",
    )
    parser.add_argument(
        "--drone-search-yaw",
        type=float,
        default=0.25,
        help="SPF 解析失败时的无人机默认搜索 yaw",
    )
    parser.add_argument(
        "--step-scale",
        type=float,
        default=-1.0,
        help="CityWalker 坐标归一化尺度。<=0 表示自动估计",
    )
    parser.add_argument(
        "--plane-mode",
        type=str,
        default="xy",
        choices=["xy", "xz"],
        help="CityWalker ground plane selection",
    )
    parser.add_argument(
        "--citywalker-resolution",
        type=int,
        nargs=2,
        default=[640, 360],
        help="CityWalker 单独输入分辨率 (宽 高)，默认 640 360",
    )
    parser.add_argument(
        "--rgb-input",
        action="store_true",
        help="若 observation 已经是 RGB，则关闭默认 BGR->RGB 转换",
    )
    parser.add_argument(
        "--disable-near-goal-push",
        action="store_true",
        help="关闭 CityWalker 近场保底前推",
    )
    parser.add_argument(
        "--fine-injured-push-start-cm",
        type=float,
        default=300.0,
        help="找伤员阶段开始近场前推的距离阈值 (cm)",
    )
    parser.add_argument(
        "--fine-stretcher-push-start-cm",
        type=float,
        default=350.0,
        help="找担架阶段开始近场前推的距离阈值 (cm)",
    )
    parser.add_argument(
        "--near-goal-push-min-speed",
        type=float,
        default=80.0,
        help="近场前推的最小保底速度",
    )
    parser.add_argument(
        "--near-goal-push-align-angle-deg",
        type=float,
        default=12.0,
        help="仅在目标基本位于前方时启用前推的角度阈值",
    )
    parser.add_argument(
        "--near-goal-push-stop-margin-cm",
        type=float,
        default=15.0,
        help="距离交互门槛多近时停止前推 (cm)",
    )
    apply_model_profile_defaults(parser, "seepointfly")

    args = parser.parse_args()

    from agents.seepointfly_agent import SeePointFlyAgent

    agent = SeePointFlyAgent(
        citywalker_checkpoint_path=args.citywalker_checkpoint_path,
        citywalker_config_path=args.citywalker_config_path,
        spf_config_path=args.spf_config_path,
        device=args.device,
        step_scale=args.step_scale,
        plane_mode=args.plane_mode,
        citywalker_resolution=tuple(args.citywalker_resolution),
        bgr_input=not args.rgb_input,
        drone_trigger_radius_m=args.drone_trigger_radius_m,
        drone_trigger_height_m=args.drone_trigger_height_m,
        drone_max_speed=args.drone_max_speed,
        drone_max_z_speed=args.drone_max_z_speed,
        drone_max_yaw=args.drone_max_yaw,
        drone_cruise_altitude_cm=args.drone_height_offset_cm,
        drone_search_yaw=args.drone_search_yaw,
        fine_injured_push_start_cm=args.fine_injured_push_start_cm,
        fine_stretcher_push_start_cm=args.fine_stretcher_push_start_cm,
        near_goal_push_min_speed=args.near_goal_push_min_speed,
        near_goal_push_align_angle_deg=args.near_goal_push_align_angle_deg,
        near_goal_push_stop_margin_cm=args.near_goal_push_stop_margin_cm,
        disable_near_goal_push=args.disable_near_goal_push,
    )

    run_benchmark_from_args(args, agent, model_name="seepointfly")


if __name__ == "__main__":
    main()
