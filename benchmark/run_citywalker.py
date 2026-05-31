#!/usr/bin/env python3
"""
CityWalker 薄启动器
=================
使用 rescue_benchmark.py 基座 + CityWalker Agent。
"""

import os
import sys


# 确保 benchmark 目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rescue_benchmark import create_base_parser, run_benchmark_from_args
from agents.profiles import apply_model_profile_defaults


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CITYWALKER_ROOT = os.path.join(PROJECT_ROOT, "baseline_model", "CityWalker")
DEFAULT_CONFIG_PATH = os.path.join(CITYWALKER_ROOT, "config", "citywalk_2000hr.yaml")


def _recommend_resolution(config_path: str):
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    crop_h, crop_w = [int(v) for v in cfg["model"]["obs_encoder"]["crop"]]
    if crop_w >= 600:
        return [max(640, crop_w), max(360, crop_h)]
    return [max(400, crop_w), max(400, crop_h)]


def main():
    parser = create_base_parser(
        description="CityWalker 救援任务评估",
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="CityWalker 配置文件路径",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="CityWalker checkpoint 路径",
    )
    parser.add_argument(
        "--step-scale",
        type=float,
        default=-1.0,
        help="坐标归一化尺度。<=0 表示根据最近轨迹自动估计",
    )
    parser.add_argument(
        "--plane-mode",
        type=str,
        default="xy",
        choices=["xy", "xz"],
        help="从 3D pose 中提取哪两个平面坐标给 CityWalker。Rescue 推荐 xy",
    )
    parser.add_argument(
        "--rgb-input",
        action="store_true",
        help="若 observation 已经是 RGB，则关闭默认的 BGR->RGB 转换",
    )
    parser.add_argument(
        "--disable-near-goal-push",
        action="store_true",
        help="关闭 CityWalker 近场保底前推",
    )
    parser.add_argument(
        "--fine-injured-push-start-cm",
        type=float,
        default=250.0,
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
        default=75.0,
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
    apply_model_profile_defaults(parser, "citywalker")

    args = parser.parse_args()
    args.config_path = os.path.abspath(os.path.expanduser(args.config_path))
    args.checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path))

    if args.resolution == [640, 640]:
        args.resolution = _recommend_resolution(args.config_path)
        print(f"[CityWalker] Auto resolution from config: {args.resolution[0]}x{args.resolution[1]}")

    from agents.citywalker_agent import CityWalkerAgent

    agent = CityWalkerAgent(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        step_scale=args.step_scale,
        plane_mode=args.plane_mode,
        bgr_input=not args.rgb_input,
        enable_near_goal_push=not args.disable_near_goal_push,
        fine_injured_push_start_cm=args.fine_injured_push_start_cm,
        fine_stretcher_push_start_cm=args.fine_stretcher_push_start_cm,
        near_goal_push_min_speed=args.near_goal_push_min_speed,
        near_goal_push_align_angle_deg=args.near_goal_push_align_angle_deg,
        near_goal_push_stop_margin_cm=args.near_goal_push_stop_margin_cm,
    )

    run_benchmark_from_args(args, agent, model_name="citywalker")


if __name__ == "__main__":
    main()
