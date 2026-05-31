"""
test_pose_sampler.py

Smoke-test for RescuePoseSamplerWrapper.  Exercises the full wrapper stack,
runs a short step loop per episode, and verifies that poses were actually
written to the underlying env after each reset().

Run from the project root:
    python example/test_pose_sampler.py \\
        --map SuburbNeighborhood_Day \\
        --graph-path example/agent_configs_sampler/points_graph/SuburbNeighborhood_Day/environment_graph_1.gpickle \\
        --episodes 2 --max-steps 5
"""

import os
import sys

# Insert project root BEFORE any gym_rescue import so the local copy
# (containing rescue_pose_sampler.py) takes priority over any installed version.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import time

import numpy as np
np.bool8 = np.bool_

import gym
from gym_rescue.envs.wrappers import configUE, time_dilation, population, rescue_pose_sampler

# ---------------------------------------------------------------------------
# Maps registered in gym_rescue/__init__.py
# ---------------------------------------------------------------------------
MAPS = [
    'FlexibleRoom',
    'SuburbNeighborhood_Day',
    'Forglar_Map',
    'HongKongStreet',
    'track_train',
    'SuburbNeighborhood_Night',
]

DEFAULT_GRAPH_PATH = os.path.join(
    _SCRIPT_DIR,
    'agent_configs_sampler', 'points_graph',
    'SuburbNeighborhood_Day', 'environment_graph.gpickle',
)
os.environ["UnrealEnv"] = r"E:\UnrealEnv"
# ---------------------------------------------------------------------------
# Pose verification
# ---------------------------------------------------------------------------
POSE_ATTRS = {
    'rescuer':   'agent_pose',
    'injured':   'injured_player_pose',
    'stretcher': 'rescue_pose',
    'ambulance': 'ambulance_pose',
}

# set vehicle zones for ambulance sampling
vehicle_zones = {
    "SuburbNeighborhood_Day": {
        "ambulance": [
            [[-3500, -170], [-3500, 170], [1600, 170], [1600, -170]]
        ]
        },
}

# hide trees for better visibility
obj_2_hide = {
    "SuburbNeighborhood_Day": ["BP_Tree_Skinned_LargeSplit2", "BP_Tree_Skinned_LargeSplit3", "BP_Tree_Skinned_LargeSplit4", "BP_Tree_Skinned_LargeSplit5", "BP_Tree_Skinned_LargeSplit6",
                            "BP_Tree_Skinned_LargeSplit7", "BP_Tree_Skinned_LargeSplit_6", "BP_Tree_Skinned_Large2_2", "BP_Tree_Skinned_Large3", "BP_Tree_Skinned_Large4",
                            "BP_Tree_Skinned_Large5", "BP_Tree_Skinned_Large6", "BP_Tree_Skinned_Large_9"],
}


def verify_poses(unwrapped_env):
    all_ok = True
    for agent_type, attr in POSE_ATTRS.items():
        pose = getattr(unwrapped_env, attr, None)
        if pose is None:
            print(f'  [FAIL] {agent_type}: {attr} is None')
            all_ok = False
        elif all(v == 0 for v in pose):
            print(f'  [WARN] {agent_type}: {attr} = {pose}  (all zeros)')
        else:
            print(f'  [OK]   {agent_type}: {attr} = {[round(v, 1) for v in pose]}')
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Smoke-test for RescuePoseSamplerWrapper'
    )
    parser.add_argument('--map', default='SuburbNeighborhood_Day',
                        choices=MAPS, help='Environment map name')
    parser.add_argument('--graph-path', default=DEFAULT_GRAPH_PATH,
                        help='Path to .gpickle nav graph')
    parser.add_argument('--cam-id', type=int, default=1,
                        help='Overhead camera ID')
    parser.add_argument('--cam-height', type=float, default=800.0,
                        help='Overhead camera height (cm)')
    parser.add_argument('--model', default='gemini-2.5-flash',
                        help='VLM model name (ignored in fast-test mode)')
    parser.add_argument('--config-path', default='.',
                        help='Directory containing model_config.json')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=10,
                        help='Maximum steps per episode')
    parser.add_argument('--level', type=int, default=0,
                        help='Environment difficulty level (0-4)')
    parser.add_argument('--offscreen', action='store_true',
                        help='Enable offscreen rendering')
    parser.add_argument('--time-dilation', type=int, default=-1,
                        dest='time_dilation',
                        help='Time-dilation value; -1 = disabled')

    # VLM toggle: --fast-test (default on) vs --vlm
    vlm_group = parser.add_mutually_exclusive_group()
    vlm_group.add_argument('--fast-test', dest='vlm', action='store_false',
                           help='Use random sampling instead of VLM (default)')
    vlm_group.add_argument('--vlm', dest='vlm', action='store_true',
                           help='Enable VLM-based pose sampling (requires API key)')
    parser.set_defaults(vlm=True)  # Default to VLM-enabled mode; override with --fast-test

    args = parser.parse_args()



    env = None
    try:
        # ------------------------------------------------------------------
        # Build wrapper stack (innermost → outermost)
        # ------------------------------------------------------------------
        env = gym.make(
            f'UnrealRescue-{args.map}',
            action_type='Mixed',
            observation_type='Color',
            reset_type=args.level,
            disable_env_checker=True,
        )

        env = configUE.ConfigUEWrapper(
            env,
            offscreen=args.offscreen,
            resolution=(1000, 1000),
        )

        if args.time_dilation > 0:
            env = time_dilation.TimeDilationWrapper(env, args.time_dilation)

        if args.level > 0:
            env = population.RandomPopulationWrapper(env, 1, 1)

        ## preprare for sampling
        map_vehicle_zones = vehicle_zones.get(args.map, None)
        obj_2_hide_list = obj_2_hide.get(args.map, None)


        env = rescue_pose_sampler.RescuePoseSamplerWrapper(
            env,
            graph_path=args.graph_path,
            cam_id=args.cam_id,
            cam_height=args.cam_height,
            vehicle_zones=map_vehicle_zones,
            obj_2_hide=obj_2_hide_list,
            model=args.model,
            config_path=args.config_path,
            fast_test_mode=not args.vlm,
        )

        print(f'Map:        {args.map}')
        print(f'Graph:      {args.graph_path}')
        print(f'Fast-test:  {not args.vlm}')
        print(f'Episodes:   {args.episodes}  |  Max steps: {args.max_steps}')

        # ------------------------------------------------------------------
        # Episode loop
        # ------------------------------------------------------------------
        all_checks_passed = True

        for ep in range(1, args.episodes + 1):
            print(f'\n--- Episode {ep}/{args.episodes} ---')

            obs, info = env.reset()

            print('Pose check after reset:')
            ep_ok = verify_poses(env.unwrapped)
            if not ep_ok:
                all_checks_passed = False

            count_step = 0
            t0 = time.time()
            done = False
            termination = False

            while not done and count_step < args.max_steps:
                action = env.action_space.sample()
                obs, reward, termination, truncation, info = env.step([action])
                count_step += 1
                done = termination or truncation

            elapsed = time.time() - t0
            fps = count_step / elapsed if elapsed > 0 else float('nan')

            if not done:
                status = 'max_steps'
            elif termination:
                status = 'success'
            else:
                status = 'truncated'

            print(f'Steps: {count_step}  FPS: {fps:.1f}  {status}')

        print()
        if all_checks_passed:
            print('All pose checks passed.')
        else:
            print('Some pose checks failed or returned warnings.')
            sys.exit(1)

    finally:
        if env is not None:
            env.close()
