import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
os.environ['UnrealEnv']='/media/littlecave/T9/UnrealEnv'

# 允许从任意工作目录启动脚本时也能找到 gym_rescue 包
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import cv2
import gym
import gym_rescue
import numpy as np
import torch

from gym_rescue.envs.utils import misc
from gym_rescue.envs.wrappers import (
    configUE,
    early_done,
    monitor,
    task_cue,
    time_dilation,
)


def pose_to_record(pose, step, action=None, picked=None, reward=None, timestamp=None, obs=None):
    record = {
        "step": int(step),
        "x": float(pose[0]),
        "y": float(pose[1]),
        "z": float(pose[2]),
        "roll": float(pose[3]),
        "yaw": float(pose[4]),
        "pitch": float(pose[5]),
    }
    if action is not None:
        record["action"] = {
            "move": [float(action[0][0]), float(action[0][1])],
            "head": int(action[1]),
            "anim": int(action[2]),
        }
    if picked is not None:
        record["picked"] = bool(picked)
    if reward is not None:
        record["reward"] = float(reward)
    if timestamp is not None:
        record["timestamp"] = float(timestamp)
    if obs is not None:
        record["obs"] = obs
    return record


def build_output_path(output_dir, env_id, level, episode_id):
    return Path(output_dir) / env_id / f"level_{level}_episode_{episode_id}.pt"


def build_video_path(video_dir, env_id, level, episode_id):
    return Path(video_dir) / env_id / f"level_{level}_episode_{episode_id}.mp4"


def save_episode_to_pt(output_path, episode_record):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(episode_record, output_path)


def get_current_agent_pose(env):
    # 优先使用 wrapper 中的 test_point 信息（如果存在）
    if hasattr(env, "test_point") and "agent_loc" in env.test_point:
        return list(env.test_point["agent_loc"])
    # 否则回退到 env.unwrapped.agent_pose
    return list(env.unwrapped.agent_pose)


def read_test_point(args, point_id):
    """读取当前 level 对应 test_jsonl 中第 point_id 条测试点配置。"""
    file_name = "level_" + str(args.level) + ".jsonl"
    gympath = os.path.dirname(gym_rescue.__file__)
    json_file = os.path.join(gympath, "envs/setting/test_jsonl", file_name)
    test_json = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            test_json.append(json.loads(line))
    return test_json[point_id]


def node_name_from_test_point(test_point):
    """测试点节点名：优先用 reference_image_path 首项（与 env_config 中 reference 一致）。"""
    rip = test_point.get("reference_image_path")
    if isinstance(rip, list) and len(rip) > 0 and rip[0]:
        return str(rip[0])
    return f"level_{test_point.get('level', '?')}_episode_unknown"


def level2PointNum(args):
    file_name = "level_" + str(args.level) + ".jsonl"
    gympath = os.path.dirname(gym_rescue.__file__)
    json_file = os.path.join(gympath, "envs/setting/test_jsonl", file_name)
    test_json = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            test_json.append(json.loads(line))
    return len(test_json)


def load_level_test_points(level):
    file_name = f"level_{level}.jsonl"
    gympath = os.path.dirname(gym_rescue.__file__)
    json_file = os.path.join(gympath, "envs/setting/test_jsonl", file_name)
    test_json = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_json.append(json.loads(line))
    return test_json


def build_tasks(levels, start_episode, trajectory_dir, resume_missing=False):
    tasks = []
    for level in levels:
        points = load_level_test_points(level)
        for episode_id, test_point in enumerate(points):
            if episode_id < start_episode:
                continue
            env_id = test_point["env_id"]
            output_path = build_output_path(trajectory_dir, env_id, int(level), episode_id)
            if resume_missing and output_path.exists():
                continue
            tasks.append(
                {
                    "level": int(level),
                    "episode_id": int(episode_id),
                    "test_point": test_point,
                    "env_id": env_id,
                    "node_name": node_name_from_test_point(test_point),
                    "output_path": output_path,
                }
            )
    return tasks


def xy_dist_xy(pose6d, target_xyz):
    # 统一用 x-y 平面距离
    return float(np.linalg.norm(np.array(pose6d[:2]) - np.array(target_xyz[:2])))

class Nav2EnemyAgent:
    """
    基于 UE 返回 waypoints 的最简导航 agent。

    目标：
    1) 输出每一步移动控制 [angle, velocity]
    2) 使用阈值 100 判断 waypoint 是否“到达”（与全局“两个都用100”的约定一致）
    """

    def __init__(
        self,
        action_space,
        way_points,
        fix_point=False,
        random_th=0,
        max_len=200,
        reach_th=50.0,
    ):
        self.step_counter = 0
        self.keep_steps = 0
        self.random_th = random_th
        self.max_len = max_len
        self.fix = fix_point
        self.reach_th = float(reach_th)

        # Mixed action space
        self.discrete = False
        self.velocity_high = action_space.spaces[0].high[1]
        self.velocity_low = action_space.spaces[0].low[1]
        self.angle_high = action_space.spaces[0].high[0]
        self.angle_low = action_space.spaces[0].low[0]


        self.waypoints = way_points
        self.goal_idx = 0
        self.goal = self.waypoints[self.goal_idx]

        self.reset()

    def act(self, pose, ref_goal=None):
        self.step_counter += 1
        if self.pose_last is None or self.fix:
            self.pose_last = pose
            d_moved = 10
        else:
            d_moved = np.linalg.norm(np.array(self.pose_last) - np.array(pose))
            self.pose_last = pose
        self.d_move_ave = self.d_move_ave * 0.7 + d_moved * 0.3

        if self.check_reach(self.goal, pose):
            self.goal_idx += 1
            if self.goal_idx >= len(self.waypoints):
                self.goal_idx = len(self.waypoints) - 1
            self.goal = self.waypoints[self.goal_idx]

        delt_yaw = misc.get_direction(pose, self.goal)  # x-y 平面转角
        angle = np.clip(delt_yaw, self.angle_low, self.angle_high)
        velocity = self.velocity * (1 + 0.2 * np.random.random())
        return [angle, velocity]

    def reset(self):
        self.step_counter = 0
        self.keep_steps = 0
        self.angle_noise_step = 0
        self.goal_id = 0
        self.d_move_ave = 5
        self.velocity = np.random.randint(self.velocity_high * 0.5, self.velocity_high)
        self.pose_last = None

    def check_reach(self, goal, now):
        error = np.array(now[:2]) - np.array(goal[:2])
        distance = np.linalg.norm(error)
        return distance < self.reach_th


class RescueAutoNavController:
    """
    状态机：
      go_to_injured: stand, 目标=injured_player_pose
      carry_or_pick: carry, 目标仍=injured_player_pose；当 picked=True 切到 carry_to_stretcher
      carry_to_stretcher: carry, 目标=sretcher_pose；当 picked=True 且 dist<100 切 drop
      drop_on_stretcher: drop, move=[0,0]
    你要求的逻辑：drop 必须同时满足 picked=True 与 dist_to_stretcher < 100
    """

    def __init__(
        self,
        env,
        distance_threshold=200.0,
        carry_drop_threshold=150.0,
        head_id=0,
        debug_context=None,
    ):
        self.env = env
        self.threshold = float(distance_threshold)
        self.carry_drop_threshold = float(carry_drop_threshold)
        self.head_id = int(head_id)
        self.debug_context = debug_context or {}

        # animation_action 在 env_config 里的顺序通常是：
        # ["stand","jump","crouch","carry","drop","open_door"]
        self.anim_stand = 0
        self.anim_carry = 3
        self.anim_drop = 4
        self.anim_open_door = 5

        self.player_obj = env.unwrapped.player_list[env.unwrapped.protagonist_id]
        self.nav_agent = None
        self.injured_xyz = None
        self.stretcher_xyz = None
        self.stage = "go_to_injured"

        # 门交互参数
        self.door_distance_th = 200.0  # 与门的平面距离阈值（UE单位）；小于该值才考虑开门
        self.door_heading_th = 35.0  # 朝向门的角度阈值（度）；绝对偏角小于该值视为“正朝门前进”
        self.open_door_cooldown = 12  # 开门动作冷却步数；防止连续多帧重复触发 open_door
        self.waypoint_refresh_interval = 50  # 固定每 N 步重新规划一次 waypoints
        self.step_idx = 0  # 控制器内部步计数（每次 act 调用 +1）
        self.last_open_step = -10**9  # 最近一次触发 open_door 的步号（用于冷却判定）
        self.door_positions = {}  # 缓存门名到门位置的映射：{door_name: [x, y, z]}
        self.interactive_doors = []  # 当前地图可交互门名列表（从 env_config["interactive_door"] 读取）

    @staticmethod
    def _fmt_xyz(xyz):
        arr = list(map(float, xyz[:3]))
        return f"[{arr[0]:.2f}, {arr[1]:.2f}, {arr[2]:.2f}]"

    def _make_nav_agent(self, target_xyz):
        # 通过你写的 UE path 接口获取 waypoints
        waypoints = self.env.unwrapped.unrealcv.find_path(self.player_obj, target_xyz)

        if waypoints is None or len(waypoints) == 0:
            current_pose = getattr(self.env.unwrapped, "player_pose", None)
            start_xyz = (
                list(map(float, current_pose[:3]))
                if current_pose is not None and len(current_pose) >= 3
                else None
            )
            meta = " ".join(
                [
                    f"env={self.debug_context.get('env_id', 'unknown')}",
                    f"level={self.debug_context.get('level', 'unknown')}",
                    f"episode={self.debug_context.get('episode_id', 'unknown')}",
                    f"node={self.debug_context.get('node_name', 'unknown')}",
                    f"stage={self.stage}",
                    f"step={self.step_idx}",
                ]
            )
            start_info = (
                f"start={self._fmt_xyz(start_xyz)}"
                if start_xyz is not None
                else "start=[unknown]"
            )
            raise AssertionError(
                "No waypoints found, check navpoint location"
                + f" | {meta}"
                + f" | {start_info}"
                + f" target={self._fmt_xyz(target_xyz)}"
            )

        if waypoints is None or len(waypoints) == 0:
            waypoints = [list(map(float, target_xyz[:3]))]

        # 把每个 waypoint 统一整理成至少 [x,y,z]
        target_xyz = list(map(float, target_xyz[:3]))

        def normalize_wp(wp):
            wp_list = list(map(float, wp))
            if len(wp_list) >= 3:
                return wp_list[:3]
            if len(wp_list) == 2:
                return [wp_list[0], wp_list[1], target_xyz[2]]
            if len(wp_list) == 1:
                return [wp_list[0], target_xyz[1], target_xyz[2]]
            return [target_xyz[0], target_xyz[1], target_xyz[2]]

        # 防止 Nav2EnemyAgent 到最后一个 waypoint 后 goal_idx 越界
        waypoints = [normalize_wp(wp) for wp in waypoints]
        last_wp = waypoints[-1]
        # 额外堆几个终点 waypoint，提供“缓冲区”
        waypoints.extend([last_wp] * 5)
        return Nav2EnemyAgent(
            action_space=self.env.action_space,
            way_points=waypoints,
            fix_point=False,
            random_th=0,
            max_len=200,
            reach_th=self.threshold,
        )

    def reset(self):
        self.stage = "go_to_injured"
        self.pick_sent = False
        self.step_idx = 0
        self.last_open_step = -10**9
        # 1) 你的要求：伤者目标点使用 env.unwrapped.injured_player_pose
        self.injured_xyz = list(self.env.unwrapped.injured_player_pose[:3])
        self.stretcher_xyz = list(self.env.unwrapped.rescue_pose[:3])
        self.nav_agent = self._make_nav_agent(self.injured_xyz)
        self._init_interactive_doors()

    def _init_interactive_doors(self):
        self.interactive_doors = list(self.env.unwrapped.env_configs.get("interactive_door", []))
        self.door_positions = {}
        for door_name in self.interactive_doors:
            try:
                loc = self.env.unwrapped.unrealcv.get_obj_location(door_name)
                if loc is not None and len(loc) >= 3:
                    self.door_positions[door_name] = [float(loc[0]), float(loc[1]), float(loc[2])]
            except Exception:
                continue

    def _should_open_door(self, agent_pose6d, planned_move):
        if self.step_idx - self.last_open_step < self.open_door_cooldown:
            return False
        if not self.door_positions:
            return False
        # 没有向前走，不触发开门
        if float(planned_move[1]) <= 1e-3:
            return False

        nearest_name = None
        nearest_dist = float("inf")
        nearest_loc = None
        for name, loc in self.door_positions.items():
            d = xy_dist_xy(agent_pose6d, loc)
            if d < nearest_dist:
                nearest_dist = d
                nearest_name = name
                nearest_loc = loc

        if nearest_loc is None or nearest_dist > self.door_distance_th:
            return False

        # 仍朝着门方向前进才开门
        heading_to_door = abs(misc.get_direction(agent_pose6d, nearest_loc))
        if heading_to_door > self.door_heading_th:
            return False

        self.last_open_step = self.step_idx
        return True

    def _refresh_nav_agent_if_needed(self, target_xyz):
        if self.waypoint_refresh_interval > 0 and self.step_idx % self.waypoint_refresh_interval == 0:
            self.nav_agent = self._make_nav_agent(target_xyz)

    def act(self, agent_pose6d, picked):
        self.step_idx += 1
        picked = bool(picked)

        if self.stage == "go_to_injured":
            dist = xy_dist_xy(agent_pose6d, self.injured_xyz)
            z_gap = abs(float(agent_pose6d[2]) - float(self.injured_xyz[2]))
            if dist < self.carry_drop_threshold and z_gap <= 100.0:
                self.stage = "carry_or_pick"
                # 进入拾取：只在切入的那一帧触发一次 carry
                self.pick_sent = True
                return ([0.0, 0.0], self.head_id, self.anim_carry)

            # 仍在接近伤者：导航移动 + stand
            self._refresh_nav_agent_if_needed(self.injured_xyz)
            move = self.nav_agent.act(agent_pose6d)
            if self._should_open_door(agent_pose6d, move):
                return ([0.0, 0.0], self.head_id, self.anim_open_door)
            return ([float(move[0]), float(move[1])], self.head_id, self.anim_stand)

        if self.stage == "carry_or_pick":
            if picked:
                self.stage = "carry_to_stretcher"
                self.nav_agent = self._make_nav_agent(self.stretcher_xyz)
                move = self.nav_agent.act(agent_pose6d)
                # 已经拿起后：避免 carry 连续触发
                return ([float(move[0]), float(move[1])], self.head_id, self.anim_stand)

            # picked 还没变 True：carry 只触发一次；之后保持 stand + 不移动
            if not self.pick_sent:
                self.pick_sent = True
                return ([0.0, 0.0], self.head_id, self.anim_carry)

            return ([0.0, 0.0], self.head_id, self.anim_stand)

        if self.stage == "carry_to_stretcher":
            dist = xy_dist_xy(agent_pose6d, self.stretcher_xyz)
            # 你的要求：drop 触发同时满足条件1和2
            if picked and dist < self.carry_drop_threshold:
                self.stage = "drop_on_stretcher"
                return ([0.0, 0.0], self.head_id, self.anim_drop)

            self._refresh_nav_agent_if_needed(self.stretcher_xyz)
            move = self.nav_agent.act(agent_pose6d)
            if self._should_open_door(agent_pose6d, move):
                return ([0.0, 0.0], self.head_id, self.anim_open_door)
            # 拿起后的移动阶段：固定 stand
            return ([float(move[0]), float(move[1])], self.head_id, self.anim_stand)

        # drop_on_stretcher
        return ([0.0, 0.0], self.head_id, self.anim_drop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto navigation data collection for Rescue task")
    parser.add_argument("-r", "--render", dest="render", action="store_true", help="show env using cv2")
    parser.add_argument("-s", "--seed", dest="seed", default=10, help="random seed")
    parser.add_argument("-t", "--time-dilation", dest="time_dilation", default=-1, help="time_dilation to keep fps in simulator")
    parser.add_argument("-n", "--nav-agent", dest="nav_agent", action="store_true", help="(kept for compatibility; always uses auto nav)")
    parser.add_argument("-d", "--early-done", dest="early_done", default=-1, help="early_done when lost in n steps")
    parser.add_argument("-m", "--monitor", dest="monitor", action="store_true", help="auto_monitor")
    parser.add_argument("-l", "--level", dest="level", default=3, help="Difficulty level for rescue task(0-4), kept for compatibility")
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 3, 4], help="Levels to run, e.g. --levels 0 1 2 3 4")
    parser.add_argument("--trajectory-dir", default="/media/littlecave/T9/Offline_RL_Active_Tracking/gym-rescue/benchmark/aoe_eval/auto_trajectories", help="Directory used to save trajectories in pt format")
    parser.add_argument("--start-episode", default=1, type=int, help="start episode index")
    parser.add_argument("--continue-on-nav-error", action="store_true", help="continue to next episode if find_path returns empty")
    parser.add_argument("--resume-missing", action="store_true", help="only run episodes whose pt file is missing under --trajectory-dir")
    parser.add_argument("--record-video", action="store_true", help="record mp4 for each episode")
    parser.add_argument("--video-dir", default="/media/littlecave/T9/Offline_RL_Active_Tracking/gym-rescue/benchmark/aoe_eval/auto_videos", help="Directory used to save episode videos")
    parser.add_argument("--video-fps", type=int, default=10, help="fps for saved mp4")
    parser.add_argument("--keep-success-video", action="store_true", help="keep success mp4; default is deleting success video and keeping only failed/timeout/nav_error")
    args = parser.parse_args()

    timeout_nodes = []
    failed_nodes = []
    nav_error_nodes = []
    tasks = build_tasks(
        levels=args.levels,
        start_episode=args.start_episode,
        trajectory_dir=args.trajectory_dir,
        resume_missing=args.resume_missing,
    )
    print(
        f"[TASKS] total={len(tasks)} "
        f"mode={'resume_missing' if args.resume_missing else 'full_scan'} "
        f"levels={args.levels} start_episode={args.start_episode}"
    )

    for task in tasks:
        level = int(task["level"])
        e = int(task["episode_id"])
        test_point = task["test_point"]
        env_id = task["env_id"]
        node_name = task["node_name"]
        output_path = Path(task["output_path"])
        video_path = build_video_path(args.video_dir, env_id, level, e)
        video_writer = None
        keep_video = False
        run_result = "unknown"
        elapsed_sec = 0.0
        fps = 0.0

        print(
            "[EP_START] "
            f"env={env_id} level={level} episode={e} node={node_name} "
            f"agent={test_point['agent_loc'][:3]} injured={test_point['injured_player_loc'][:3]} "
            f"stretcher={test_point['stretcher_loc'][:3]}"
        )

        env = gym.make(env_id, action_type="Mixed", observation_type="Rgbd", reset_type=level)
        env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(240, 240), use_lumen=False)
        if int(args.time_dilation) > 0:
            env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
        if int(args.early_done) > 0:
            env = early_done.EarlyDoneWrapper(env, int(args.early_done))
        if args.monitor:
            env = monitor.DisplayWrapper(env)

        # 设置 injured/rescue/agent 等测试点初始状态
        env = task_cue.TaskCueWrapper(env, level, e)
        obs, info = env.reset()

        if args.record_video:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            h, w = obs.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(str(video_path), fourcc, max(1, int(args.video_fps)), (w, h))
            video_writer.write(obs[:, :, :3].astype(np.uint8))

        t0 = time.time()
        episode_start_time = time.time()
        timeout_sec = int(env.test_point.get("timeout", 60 if level <= 2 else 90))

        trajectory = []
        init_pose = get_current_agent_pose(env)
        trajectory.append(
            pose_to_record(
                init_pose,
                step=0,
                picked=info.get("picked", False),
                reward=0.0,
                timestamp=0.0,
                obs=np.concatenate([obs[:, :, :3].astype(np.uint8), obs[:, :, 3:4]], axis=-1),
            )
        )

        controller = RescueAutoNavController(
            env,
            distance_threshold=20.0,
            carry_drop_threshold=150.0,
            head_id=0,
            debug_context={
                "env_id": env_id,
                "level": level,
                "episode_id": e,
                "node_name": node_name,
            },
        )
        try:
            controller.reset()
        except AssertionError as nav_err:
            run_result = "nav_error"
            nav_error_nodes.append(
                {
                    "node_name": node_name,
                    "env_id": env_id,
                    "level": level,
                    "episode_id": e,
                    "reason": str(nav_err),
                }
            )
            print(f"[NAV_ERROR][reset] {nav_err}")
            keep_video = args.record_video
            env.close()
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            if args.continue_on_nav_error:
                continue
            raise

        # 用 reset 时的 agent pose 做第一步动作决策
        agent_pose6d = init_pose
        picked = info.get("picked", False)

        count_step = 0
        while True:
            try:
                action = controller.act(agent_pose6d, picked=picked)
            except AssertionError as nav_err:
                run_result = "nav_error"
                nav_error_nodes.append(
                    {
                        "node_name": node_name,
                        "env_id": env_id,
                        "level": level,
                        "episode_id": e,
                        "reason": str(nav_err),
                    }
                )
                print(f"[NAV_ERROR][act] {nav_err}")
                keep_video = args.record_video
                if args.continue_on_nav_error:
                    break
                raise

            obs, rewards, termination, truncation, info = env.step([action])
            elapsed_sec = time.time() - episode_start_time
            agent_pose6d = info["Pose"][0]
            picked = info.get("picked", False)

            trajectory.append(
                pose_to_record(
                    agent_pose6d,
                    step=count_step + 1,
                    action=action,
                    picked=picked,
                    reward=rewards,
                    timestamp=elapsed_sec,
                    obs=np.concatenate([obs[:, :, :3].astype(np.uint8), obs[:, :, 3:4]], axis=-1),
                )
            )

            if video_writer is not None:
                video_writer.write(obs[:, :, :3].astype(np.uint8))

            if args.render:
                cv2.imshow("agent obs", obs[:, :, :3].astype(np.uint8))
                cv2.imshow("agent depth", 255 / obs[:, :, 3])
                if hasattr(env.unwrapped, "goal_show"):
                    cv2.imshow("goal", env.unwrapped.goal_show)
                cv2.waitKey(1)

            count_step += 1
            fps = count_step / max(1e-6, (time.time() - t0))

            if termination:
                run_result = "success"
                episode_record = {
                    "env_id": env_id,
                    "level": level,
                    "episode_id": e,
                    "goal_image": info["reference_image"],
                    "reference_text": info["reference_text"],
                    "seed": int(args.seed),
                    "result": "success",
                    "terminated": True,
                    "truncated": False,
                    "timeout_sec": timeout_sec,
                    "elapsed_sec": elapsed_sec,
                    "steps": count_step,
                    "fps": fps,
                    "trajectory": trajectory,
                }
                save_episode_to_pt(output_path, episode_record)
                print("Success")
                print(f"Trajectory saved for episode {e} -> {output_path}")
                keep_video = bool(args.keep_success_video)
                break

            if elapsed_sec >= timeout_sec:
                run_result = "timeout"
                timeout_nodes.append(
                    {
                        "node_name": node_name,
                        "env_id": env_id,
                        "level": level,
                        "episode_id": e,
                    }
                )
                print(f"Timeout after {elapsed_sec:.2f}s")
                keep_video = args.record_video
                break

            if truncation:
                run_result = "failed"
                failed_nodes.append(
                    {
                        "node_name": node_name,
                        "env_id": env_id,
                        "level": level,
                        "episode_id": e,
                    }
                )
                print("Failed")
                print(f"Fps: {fps}")
                keep_video = args.record_video
                break

        env.close()
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        if args.record_video and video_path.exists() and not keep_video:
            video_path.unlink(missing_ok=True)
            print(f"[VIDEO] success episode video removed: {video_path}")
        elif args.record_video and video_path.exists():
            print(f"[VIDEO] kept: {video_path} ({run_result})")

    print("\n========== 采集结束统计（失败 / 超时 节点） ==========")
    print(f"Timeout 数量: {len(timeout_nodes)}")
    for item in timeout_nodes:
        print(
            f"  [timeout] node={item['node_name']} | env_id={item['env_id']} | "
            f"level={item['level']} | episode_id={item['episode_id']}"
        )
    print(f"Failed 数量: {len(failed_nodes)}")
    for item in failed_nodes:
        print(
            f"  [failed] node={item['node_name']} | env_id={item['env_id']} | "
            f"level={item['level']} | episode_id={item['episode_id']}"
        )
    print(f"NAV_ERROR 数量: {len(nav_error_nodes)}")
    for item in nav_error_nodes:
        print(
            f"  [nav_error] node={item['node_name']} | env_id={item['env_id']} | "
            f"level={item['level']} | episode_id={item['episode_id']} | reason={item['reason']}"
        )
    print("====================================================\n")
