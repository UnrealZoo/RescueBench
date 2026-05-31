"""
Collect topological map images for rescue task in gym-rescue environment

输出 <输出目录>/images/ 与 topomap.json，供 benchmark ViNT/NOMAD 的 --topomap-dir 使用。
images/*.png 存为 RGB。UnrealCV（bmp/png 解码）观测为 BGR，若不经 cv2.COLOR_BGR2RGB 就 PIL 保存，会把 BGR 当 RGB 写入，画面会整体发蓝/发冷（红蓝通道对调）。
评测中的伤员参考图等由 test_jsonl + ref_image 提供，与本脚本无关。

参考格式：
- collect_topomap.py (unrealzoo-gym): 拓扑地图收集实现
- record_bag.sh (visualnav-transformer): ROS环境图像收集格式思路
"""

import argparse
import os
import sys
import time
import json
import re
import cv2
import numpy as np
from pynput import keyboard
from PIL import Image as PILImage

# Fix NumPy version compatibility
np.bool8 = np.bool_
os.environ['UnrealEnv'] = '/media/littlecave/T9/UnrealEnv'

# Import Gym and Dependencies
import gym
import gym_rescue
from gym_rescue.envs.wrappers import configUE, population

# Keyboard state
key_state = {
    'i': False, 'k': False, 'j': False, 'l': False,
    's': False,  # Save image manually
    'b': False,  # Begin automatic collection
    'e': False,  # End automatic collection
    'r': False,  # Reset environment
    '0': False, '1': False, '2': False, '3': False, '4': False,
    '5': False, '6': False, '7': False, '8': False, '9': False  # Switch spawn / rescue point
}

def _set_key_state(key, pressed):
    try:
        if key.char in key_state:
            key_state[key.char] = pressed
    except AttributeError:
        pass

def on_press(key):
    _set_key_state(key, True)

def on_release(key):
    _set_key_state(key, False)

# Topomap helper class
class TopoMap:
    def __init__(self, save_dir, resume=True):
        self.save_dir = save_dir
        self.images_dir = os.path.join(save_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
        self.nodes = []
        self.edges = []
        self.manifest_path = os.path.join(self.save_dir, 'topomap.json')
        if resume:
            self._load_existing()

    def add_node(self, pil_img):
        idx = len(self.nodes)
        fname = self._next_node_filename(start_idx=idx)
        path = os.path.join(self.images_dir, fname)
        pil_img.save(path)
        self.nodes.append(fname)
        if idx > 0:
            self.edges.append([idx - 1, idx])
        self._save_manifest()
        return idx

    def _next_node_filename(self, start_idx):
        idx = start_idx
        while True:
            fname = f"{idx}.png"
            path = os.path.join(self.images_dir, fname)
            if not os.path.exists(path):
                return fname
            idx += 1

    def _load_existing(self):
        """Load existing nodes/edges so collection can continue from prior run."""
        loaded_from_manifest = False
        if os.path.isfile(self.manifest_path):
            try:
                with open(self.manifest_path, 'r') as f:
                    manifest = json.load(f)
                nodes = manifest.get('nodes', [])
                edges = manifest.get('edges', [])
                if isinstance(nodes, list) and isinstance(edges, list):
                    valid_nodes = []
                    missing_count = 0
                    for node in nodes:
                        if not isinstance(node, str):
                            continue
                        p = os.path.join(self.images_dir, node)
                        if os.path.isfile(p):
                            valid_nodes.append(node)
                        else:
                            missing_count += 1
                    self.nodes = valid_nodes
                    self.edges = edges
                    loaded_from_manifest = True
                    if missing_count > 0:
                        print(f"[Topomap] Warning: manifest has {missing_count} missing image files; skipped.")
            except Exception as e:
                print(f"[Topomap] Warning: failed to load existing manifest, fallback to images dir. err={e}")

        if not loaded_from_manifest:
            # Fallback: infer from existing numeric image filenames.
            numbered = []
            pat = re.compile(r'^(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)
            for fname in os.listdir(self.images_dir):
                m = pat.match(fname)
                if m:
                    numbered.append((int(m.group(1)), fname))
            numbered.sort(key=lambda x: x[0])
            self.nodes = [x[1] for x in numbered]
            self.edges = [[i - 1, i] for i in range(1, len(self.nodes))]

        if self.nodes:
            print(f"[Topomap] Resume enabled: loaded {len(self.nodes)} existing nodes from {self.save_dir}")
        else:
            print(f"[Topomap] Resume enabled: no existing nodes found in {self.save_dir}, start fresh")

    def _save_manifest(self):
        manifest = {'nodes': self.nodes, 'edges': self.edges}
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def num_nodes(self):
        return len(self.nodes)

def get_keyboard_action():
    """Get action from keyboard input for Mixed action space"""
    action = [[0, 0], 0, 0]
    if key_state['i']:
        action[0][1] = 200  # Forward
    if key_state['k']:
        action[0][1] = -50  # Backward
    if key_state['j']:
        action[0][0] = -30  # Turn left
    if key_state['l']:
        action[0][0] = 30   # Turn right
    return tuple(action[0]), action[1], action[2]

def calculate_yaw_to_target(agent_pos, target_pos):
    """水平面内指向伤员的 yaw（度）；写入 agent_pose[4]（与 set_obj_rotation 的 [roll,yaw,pitch] 一致）。"""
    y_delta = target_pos[1] - agent_pos[1]
    x_delta = target_pos[0] - agent_pos[0]
    return 0.0 if (x_delta == 0 and y_delta == 0) else np.degrees(np.arctan2(y_delta, x_delta))

def ensure_agent_stand(env):
    """Rescue.reset 会 drop_body；补站立，减轻躺地/相机过低。"""
    try:
        ue = getattr(env.unwrapped, 'unrealcv', None)
        if ue is None or not hasattr(ue, 'set_standup'):
            return
        pl = getattr(env.unwrapped, 'player_list', None)
        if not pl:
            return
        protagonist_id = getattr(env.unwrapped, 'protagonist_id', 0)
        if protagonist_id >= len(pl):
            return
        player = pl[protagonist_id]
        ue.set_standup(player)
        time.sleep(0.5)
        ue.set_standup(player)
        time.sleep(0.2)
    except Exception as e:
        print(f"[Warning] Could not set agent to stand: {e}")

def reset_episode(env):
    obs, _ = env.reset()
    ensure_agent_stand(env)
    return obs

def load_layouts_from_test_jsonl(env_id, level):
    """Load layout points from test_jsonl/level_<level>.jsonl for the specified env_id."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    jsonl_path = os.path.join(
        project_root,
        "gym_rescue",
        "envs",
        "setting",
        "test_jsonl",
        f"level_{level}.jsonl",
    )
    if not os.path.isfile(jsonl_path):
        return []

    env_short = env_id.replace("UnrealRescue-", "") if env_id else ""
    layouts = []
    try:
        with open(jsonl_path, "r") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[Warning] Skip invalid jsonl line {line_idx} in {jsonl_path}: {e}")
                    continue

                row_env = row.get("env_id", "")
                row_env_short = row_env.replace("UnrealRescue-", "").replace("UnrealRescueMultiAgent-", "")
                if row_env != env_id and row_env_short != env_short:
                    continue

                if not all(k in row for k in ("agent_loc", "injured_player_loc", "stretcher_loc", "ambulance_loc")):
                    continue
                injured_agent_id = row.get("injured_agent_id")
                if isinstance(injured_agent_id, list):
                    injured_agent_id = injured_agent_id[0] if injured_agent_id else None
                if injured_agent_id is not None:
                    try:
                        injured_agent_id = int(injured_agent_id)
                    except (TypeError, ValueError):
                        injured_agent_id = None
                layouts.append(
                    {
                        "agent_loc": row["agent_loc"],
                        "injured_player_loc": row["injured_player_loc"],
                        "stretcher_loc": row["stretcher_loc"],
                        "ambulance_loc": row["ambulance_loc"],
                        "injured_agent_id": injured_agent_id,
                    }
                )
    except Exception as e:
        print(f"[Warning] Failed to read test_jsonl layouts from {jsonl_path}: {e}")
        return []

    return layouts

def main():
    parser = argparse.ArgumentParser(
        description='Collect topological map images (environment graph only)',
        epilog="""
多地图：--topomap-root <根> 与 -e 联用时保存到 <根>/<env_id>/，与 benchmark --topomap-dir 一致。
例如:
  --topomap-root ./rescue_topomaps -e UnrealRescue-HongKongStreet
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-e", "--env_id", default='UnrealRescue-DesertMap', help='Environment ID')
    parser.add_argument(
        "-l", '--level', type=int, default=1,
        help='与 env_config 中 level_* 一致（如 FlexibleRoom 仅有 level_0，须用 -l 0）',
    )
    parser.add_argument("-r", '--render', action='store_true', help='Show OpenCV window')
    parser.add_argument("-o", '--offscreen', action='store_true', help='UnrealCV offscreen mode')
    parser.add_argument(
        "--topomap-root",
        type=str,
        default=None,
        help='多地图根目录；与 -e 联用时保存到 <root>/<env_id>/（供 benchmark --topomap-dir 指向同一根目录）',
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help='显式输出目录（与 --topomap-root 二选一或同时指定时优先本项）',
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=('W', 'H'),
        help='相机分辨率 宽 高（与 benchmark 默认一致便于分布对齐）',
    )
    parser.add_argument("--interval", "-i", type=float, default=2.0, help='Time interval between auto-saves (seconds)')
    parser.add_argument("-s", '--seed', type=int, default=1, help='Random seed')
    parser.add_argument("-p", '--point', dest='point', type=int, default=-1, help='Initial rescue layout index (0-9)')
    parser.add_argument(
        '--layout-source',
        choices=['auto', 'jsonl', 'env_config'],
        default='auto',
        help='Layout source: auto(优先 test_jsonl，失败回退 env_config), jsonl(仅 test_jsonl), env_config(仅 env_config)',
    )
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume collection from existing images/topomap.json if present')
    parser.add_argument('--no-resume', dest='resume', action='store_false', help='Do not load existing topomap data; start a new manifest')
    parser.set_defaults(resume=True)
    
    args = parser.parse_args()

    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
        if args.topomap_root:
            print("[System] 同时指定 --output_dir 与 --topomap-root：使用显式 --output_dir")
    elif args.topomap_root:
        output_dir = os.path.abspath(os.path.join(args.topomap_root, args.env_id))
    else:
        parser.error('请指定 --output_dir 或 --topomap-root（推荐多地图时使用 --topomap-root）')

    os.makedirs(output_dir, exist_ok=True)
    print(f"[System] UnrealEnv={os.environ.get('UnrealEnv', '')}")
    print(f"[System] Resolution={tuple(args.resolution)}")
    print(f"[System] Topomap output: {output_dir}")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print(f"[System] Creating environment: {args.env_id}")
    env = gym.make(args.env_id, action_type='Mixed', observation_type='Color', reset_type=args.level)
    res = tuple(args.resolution)
    env = configUE.ConfigUEWrapper(env, offscreen=args.offscreen, resolution=res)
    
    if args.level > 0:
        env = population.RandomPopulationWrapper(env, args.seed, args.seed, random_target=False)

    type_key = f'level_{args.level}'
    layouts = []
    layout_source_name = None

    if args.layout_source in ('auto', 'jsonl'):
        jsonl_layouts = load_layouts_from_test_jsonl(args.env_id, args.level)
        if jsonl_layouts:
            layouts = jsonl_layouts
            layout_source_name = 'test_jsonl'
        elif args.layout_source == 'jsonl':
            print(f"[Error] 按 --layout-source=jsonl 未在 test_jsonl 中找到点位：env={args.env_id}, level={args.level}")
            sys.exit(1)

    if not layouts and args.layout_source in ('auto', 'env_config'):
        try:
            configs = env.unwrapped.env_configs[type_key]
            num_points_cfg = len(configs['injured_player_loc'])
            for i in range(num_points_cfg):
                layouts.append(
                    {
                        "agent_loc": configs['agent_loc'][i],
                        "injured_player_loc": configs['injured_player_loc'][i],
                        "stretcher_loc": configs['stretcher_loc'][i],
                        "ambulance_loc": configs['ambulance_loc'][i],
                        "injured_agent_id": (
                            int(configs['injured_agent_id'][i])
                            if 'injured_agent_id' in configs and i < len(configs['injured_agent_id'])
                            else None
                        ),
                    }
                )
            layout_source_name = 'env_config'
        except (KeyError, AttributeError, IndexError) as e:
            if args.layout_source == 'env_config':
                print(f"[Error] 无法从 env_config 加载 '{type_key}'：{e}")
                print(f"  请确认该地图的 env_config 是否包含 {type_key}（例如 UnrealRescue-FlexibleRoom 只有 level_0，请使用 -l 0）。")
                sys.exit(1)

    if not layouts:
        print(f"[Error] 无可用点位：env={args.env_id}, level={args.level}, source={args.layout_source}")
        sys.exit(1)

    num_points = len(layouts)
    print(f"[System] Found {num_points} layout points from {layout_source_name}")

    def set_spawn_layout(point_idx):
        """agent_pose[3:6] 为 [roll, yaw, pitch]；朝向伤员只改 yaw（索引 4），勿写入 [5] 会当 pitch 导致姿态/视角异常。"""
        if point_idx < 0 or point_idx >= num_points:
            return False
        
        try:
            point = layouts[point_idx]
            injured_pos = list(point['injured_player_loc'])
            agent_pos = list(point['agent_loc'])
            
            yaw_to_target = calculate_yaw_to_target(agent_pos[:3], injured_pos[:3])
            while len(agent_pos) < 6:
                agent_pos.append(0.0)
            agent_pos[4] = yaw_to_target

            env.unwrapped.injured_player_pose = injured_pos
            env.unwrapped.injured_agent_appid = point.get('injured_agent_id')
            env.unwrapped.rescue_pose = list(point['stretcher_loc'])
            env.unwrapped.agent_pose = agent_pos
            env.unwrapped.ambulance_pose = list(point['ambulance_loc'])
            
            return True
        except (KeyError, AttributeError, IndexError, TypeError) as e:
            print(f"[Warning] Could not set layout for point {point_idx}: {e}")
            return False

    topo_map = TopoMap(output_dir, resume=args.resume)
    
    point_idx = args.point if args.point >= 0 else 0
    if set_spawn_layout(point_idx):
        print(f"[System] Starting with layout point {point_idx}")
    if getattr(env.unwrapped, 'agent_pose', None) is None:
        print("[Error] agent_pose 未设置：请检查 --level / --point 是否与 env_config 一致。")
        sys.exit(1)

    obs = reset_episode(env)
    is_collecting = False
    last_save_time = time.time()
    
    print("\n" + "="*60)
    print("Topological map collection")
    print("="*60)
    print("Controls:")
    print("  I/K: Forward / backward")
    print("  J/L: Turn left / right")
    print("  S: Save current view as node")
    print("  B: Start auto-save every --interval")
    print("  E: Stop auto-save")
    print("  R: Reset environment")
    print("  0-9: Switch layout point (respawn)")
    print(f"  Auto-save interval: {args.interval:.1f}s when collecting")
    print("  Q in window or Ctrl+C to quit")
    print("="*60 + "\n")
    
    try:
        while True:
            switch_point = None
            for digit in '0123456789':
                if key_state[digit]:
                    switch_point = int(digit)
                    key_state[digit] = False
                    break
            
            if switch_point is not None:
                if switch_point < num_points:
                    print(f"\n[Layout] Switching to point {switch_point}...")
                    if set_spawn_layout(switch_point):
                        obs = reset_episode(env)
                        is_collecting = False
                        last_save_time = time.time()
                        print(f"  ✓ Switched to layout {switch_point}")
                        time.sleep(0.3)
                        continue
                else:
                    print(f"[Warning] Point {switch_point} exceeds available layouts ({num_points})")
            
            if key_state['r']:
                print("\n[Reset] Environment reset...")
                obs = reset_episode(env)
                key_state['r'] = False
                time.sleep(0.5)
                continue
            
            action = get_keyboard_action()
            
            if key_state['b'] and not is_collecting:
                is_collecting = True
                print("\n[Collection] Auto-save started")
                last_save_time = time.time()
                key_state['b'] = False
            
            if key_state['e'] and is_collecting:
                is_collecting = False
                print("\n[Collection] Auto-save stopped")
                key_state['e'] = False
            
            obs, reward, done, trunc, info = env.step([action])
            
            if isinstance(obs, tuple):
                img = obs[0]
            else:
                img = obs
            # 观测为 BGR；PIL/PNG 需 RGB，否则保存后整体偏蓝（与 Forglar 等夜间冷光叠加更明显）
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_obs = PILImage.fromarray(img_rgb)
            
            if key_state['s']:
                topo_map.add_node(pil_obs)
                print(f"Saved node {topo_map.num_nodes()}")
                key_state['s'] = False
                last_save_time = time.time()
                time.sleep(0.2)
            
            if is_collecting:
                current_time = time.time()
                if current_time - last_save_time >= args.interval:
                    topo_map.add_node(pil_obs)
                    print(f"Auto-saved node {topo_map.num_nodes()}")
                    last_save_time = current_time
            
            vis = img.copy()
            status_text = f"Nodes: {topo_map.num_nodes()} | Auto: {is_collecting}"
            cv2.putText(vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Topomap collection (B/E/S, IJKL move, R reset, Q quit)', vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            if done or trunc:
                print(f"[Episode] done. Nodes: {topo_map.num_nodes()}")
                time.sleep(0.5)
                obs = reset_episode(env)
    
    except KeyboardInterrupt:
        print(f"\n\nStopped. Total nodes: {topo_map.num_nodes()}")
        print(f"Saved under: {output_dir}")
        print("Benchmark:  cd gym-rescue/benchmark && python run_visualnav.py --model nomad --topomap-dir <topomap_root> ...")
    
    finally:
        env.close()
        cv2.destroyAllWindows()
        listener.stop()

if __name__ == '__main__':
    main()




