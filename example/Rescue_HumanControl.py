import argparse
import random
from pathlib import Path

import gym_rescue
import gym
from gym import wrappers
import cv2
import time
import numpy as np
from gym_rescue.envs.wrappers import time_dilation, early_done, monitor, population, configUE,task_cue
from pynput import keyboard
import os,json
os.environ['UnrealEnv']='/media/littlecave/T9/UnrealEnv'
key_state = {
    'i': False,
    'j': False,
    'k': False,
    'l': False,
    'space': False,
    'ctrl':False,
    '1': False,
    '2': False,
    '3': False,
    'head_up': False,
    'head_down': False
}

def on_press(key):
    try:
        if key.char in key_state:
            key_state[key.char] = True
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = True
        if key == keyboard.Key.up:
            key_state['head_up'] = True
        if key == keyboard.Key.down:
            key_state['head_down'] = True
        if key ==keyboard.Key.ctrl_l:
            key_state['ctrl'] = True


def on_release(key):
    try:
        if key.char in key_state:
            key_state[key.char] = False
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = False
        if key == keyboard.Key.up:
            key_state['head_up'] = False
        if key == keyboard.Key.down:
            key_state['head_down'] = False
        if key ==keyboard.Key.ctrl_l:
            key_state['ctrl'] = False
#
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def get_key_action():
    action = ([0, 0], 0, 0)
    action = list(action)  # Convert tuple to list for modification
    action[0] = list(action[0])  # Convert inner tuple to list for modification

    if key_state['i']:
        action[0][1] = 200
    if key_state['k']:
        action[0][1] = -200
    if key_state['j']:
        action[0][0] = -30
    if key_state['l']:
        action[0][0] = 30
    if key_state['space']:
        action[2] = 1
    if key_state['ctrl']:
        action[2] = 2
    if key_state['1']:
        action[2] = 3
    if key_state['2']:
        action[2] = 4
    if key_state['3']:
        action[2] = 5
    if key_state['head_up']:
        action[1] = 1
    if key_state['head_down']:
        action[1] = 2

    action[0] = tuple(action[0])  # Convert inner list back to tuple
    action = tuple(action)  # Convert list back to tuple
    return action


def pose_to_record(pose, step, action=None, picked=None, reward=None, timestamp=None):
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
    return record


def build_output_path(output_dir, env_id, level):
    return Path(output_dir) / env_id / f"level_{level}.jsonl"


def append_episode_to_jsonl(output_path, episode_record):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(episode_record, ensure_ascii=False) + "\n")


def build_rgbd_output_dir(base_dir, env_id, level, episode_id):
    return Path(base_dir) / env_id / f"level_{level}" / f"episode_{episode_id}"


def extract_rgb_frame(obs):
    frame = np.asarray(obs)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim == 3 and frame.shape[-1] >= 3:
        return frame[:, :, :3].astype(np.uint8)
    return None


def extract_depth_from_obs(obs):
    frame = np.asarray(obs)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim == 3 and frame.shape[-1] >= 4:
        return frame[:, :, 3:4].astype(np.float32)
    return None


def get_current_depth_frame(env):
    cam_id = env.unwrapped.cam_list[env.unwrapped.protagonist_id]
    depth = env.unwrapped.unrealcv.get_depth(cam_id)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 2:
        depth = np.expand_dims(depth, axis=-1)
    return depth


def depth_to_uint8(depth):
    depth2d = np.squeeze(np.asarray(depth, dtype=np.float32))
    depth2d = np.nan_to_num(depth2d, nan=0.0, posinf=0.0, neginf=0.0)
    if depth2d.size == 0:
        return None
    d_min = float(np.min(depth2d))
    d_max = float(np.max(depth2d))
    if d_max <= d_min:
        return np.zeros_like(depth2d, dtype=np.uint8)
    depth_norm = (depth2d - d_min) / (d_max - d_min)
    return (depth_norm * 255.0).astype(np.uint8)


def save_rgbd_frame(output_dir, step_idx, obs, depth):
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = extract_rgb_frame(obs)
    if rgb is not None:
        cv2.imwrite(str(output_dir / f"rgb_{step_idx:06d}.png"), rgb)

    if depth is not None:
        np.save(output_dir / f"depth_{step_idx:06d}.npy", depth.astype(np.float32))
        depth_vis = depth_to_uint8(depth)
        if depth_vis is not None:
            cv2.imwrite(str(output_dir / f"depth_{step_idx:06d}.png"), depth_vis)


def get_current_agent_pose(env):
    if hasattr(env, "test_point") and "agent_loc" in env.test_point:
        return list(env.test_point["agent_loc"])
    return list(env.unwrapped.agent_pose)


def choose_map(args,point_id):
    file_name = 'level_' + str(args.level) + '.jsonl'

    gympath = os.path.dirname(gym_rescue.__file__)
    json_file = os.path.join(gympath, 'envs/setting/test_jsonl', file_name)
    test_json = []
    with open(json_file, 'r') as f:
        for line in f:
            test_json.append(json.loads(line))
    test_point = test_json[point_id]
    env_id = test_point['env_id']
    return env_id
def level2PointNum(args):
    file_name = 'level_' + str(args.level) + '.jsonl'

    gympath = os.path.dirname(gym_rescue.__file__)
    json_file = os.path.join(gympath, 'envs/setting/test_jsonl', file_name)
    test_json = []
    with open(json_file, 'r') as f:
        for line in f:
            test_json.append(json.loads(line))
    return len(test_json)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument("-e", "--env_id", nargs='?', default='UnrealRescue-SuburbNeighborhood_Day', help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=10, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("-l", '--level', dest='level', default=1, help='Difficulty level for rescue task(0-4) ')
    parser.add_argument("--trajectory-dir", default="new/human_trajectories",
                        help='Directory used to save human rescue trajectories in jsonl format')
    parser.add_argument("--save-rgbd", action='store_true',
                        help='Save current RGB observation and depth map for each step')
    parser.add_argument("--rgbd-dir", default=None,
                        help='Directory used to save RGB-D frames. Defaults to <trajectory-dir>/rgbd_frames')

    args = parser.parse_args()
    test_points_num=level2PointNum(args) #determin how many test points in the level, and each test point corresponds to a specific map and initial setting
    # for e in range(1,test_points_num):
    for e in range(37,59):
        env_id = choose_map(args,e)
        output_path = build_output_path(args.trajectory_dir, env_id, int(args.level))
        env = gym.make(env_id, action_type='Mixed', observation_type='Color', reset_type=args.level)
        env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(480, 480),use_lumen=True)
        if int(args.time_dilation) > 0:  # -1 means no time_dilation
            env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
        if int(args.early_done) > 0:  # -1 means no early_done
            env = early_done.EarlyDoneWrapper(env, int(args.early_done))
        if args.monitor:
            env = monitor.DisplayWrapper(env)
        # if args.level>0:
        #     env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
        # agent = RandomAgent(env.action_space[0])
        rewards = 0
        done = False
        Total_rewards = 0
        count_step = 0
        # env.seed(int(args.seed))
        s = 0


        env = task_cue.TaskCueWrapper(env, args.level, e)
        obs,info = env.reset()
        rgbd_base_dir = args.rgbd_dir if args.rgbd_dir else str(Path(args.trajectory_dir) / "rgbd_frames")
        rgbd_output_dir = build_rgbd_output_dir(rgbd_base_dir, env_id, int(args.level), e)
        if args.save_rgbd:
            depth0 = extract_depth_from_obs(obs)
            if depth0 is None:
                depth0 = get_current_depth_frame(env)
            save_rgbd_frame(rgbd_output_dir, 0, obs, depth0)
        t0=time.time()
        episode_start_time = time.time()
        timeout_sec = int(env.test_point.get("timeout", 180 if int(args.level) <= 2 else 300))
        trajectory = []
        init_pose = get_current_agent_pose(env)
        trajectory.append(
            pose_to_record(
                init_pose,
                step=0,
                picked=info.get('picked', False),
                reward=0.0,
                timestamp=0.0,
            )
        )
        while True:
            action = get_key_action()
            s=s+1
            obs, rewards, termination,truncation, info= env.step([action])
            if args.save_rgbd:
                depth = extract_depth_from_obs(obs)
                if depth is None:
                    depth = get_current_depth_frame(env)
                save_rgbd_frame(rgbd_output_dir, count_step + 1, obs, depth)
            elapsed_sec = time.time() - episode_start_time
            pose = info['Pose'][0]
            trajectory.append(
                pose_to_record(
                    pose,
                    step=count_step + 1,
                    action=action,
                    picked=info.get('picked', False),
                    reward=rewards,
                    timestamp=elapsed_sec,
                )
            )
            cv2.imwrite('test.png',obs)
            cv2.imshow('agent obs',obs)
            cv2.imshow('goal',env.unwrapped.goal_show)
            cv2.waitKey(30)
            count_step+=1
            if termination:
                fps = count_step / (time.time() - t0)
                episode_record = {
                    "env_id": env_id,
                    "level": int(args.level),
                    "episode_id": e,
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
                append_episode_to_jsonl(output_path, episode_record)
                print('Success')
                print(f'Trajectory saved for episode {e} -> {output_path}')
                break
            if elapsed_sec >= timeout_sec:
                fps = count_step / (time.time() - t0)
                episode_record = {
                    "env_id": env_id,
                    "level": int(args.level),
                    "episode_id": e,
                    "seed": int(args.seed),
                    "result": "timeout",
                    "terminated": False,
                    "truncated": True,
                    "timeout_sec": timeout_sec,
                    "elapsed_sec": elapsed_sec,
                    "steps": count_step,
                    "fps": fps,
                    "trajectory": trajectory,
                }
                append_episode_to_jsonl(output_path, episode_record)
                print(f'Timeout after {elapsed_sec:.2f}s')
                print(f'Trajectory saved for episode {e} -> {output_path}')
                break
            if truncation:
                fps = count_step / (time.time() - t0)
                episode_record = {
                    "env_id": env_id,
                    "level": int(args.level),
                    "episode_id": e,
                    "seed": int(args.seed),
                    "result": "failed",
                    "terminated": False,
                    "truncated": True,
                    "timeout_sec": timeout_sec,
                    "elapsed_sec": elapsed_sec,
                    "steps": count_step,
                    "fps": fps,
                    "trajectory": trajectory,
                }
                append_episode_to_jsonl(output_path, episode_record)
                print('Failed')
                print('Fps:' + str(fps))
                print(f'Trajectory saved for episode {e} -> {output_path}')
                break
        env.close()