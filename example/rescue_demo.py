import argparse
import gym_rescue
import gym
from gym import wrappers
import cv2
import time
from gym_rescue.envs.wrappers import time_dilation, early_done, monitor, population, configUE
from pynput import keyboard
from gym_rescue.envs.utils.keyboard_util import get_key_action,on_press,on_release
import numpy as np
np.bool8 = np.bool_
import os
# os.environ['UnrealEnv']='E:\\gym-rescue\\gym-rescue\\gym_rescue\\envs\\UnrealEnv'
os.environ['UnrealEnv']='/media/littlecave/T9/Offline_RL_Active_Tracking/gym-rescue/gym_rescue/UnrealEnv'


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()
#SuburbNeighborhood_Day，SuburbNeighborhood_Night，track_train，FlexibleRoom，Forglar_Map，HongKongStreet
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealRescue-HongKongStreet', help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', type=int, default=1, help='number of population')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', type=int, default=-1, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', type=int, default=10, help='early_done when reach n seconds')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("-k", '--keyboard', dest='keyboard', action='store_true', help='use keyboard to control the agents')
    parser.add_argument("-l", '--level', dest='level', type=int, default=3, help='Difficulty level for rescue task(0-4) ')
    parser.add_argument("-p", '--point', dest='point', type=int, default=-1, help='The point id to reach')
    parser.add_argument("-ep", '--episode', dest='episode', type=int, default=20, help='The num of episode to run')
    parser.add_argument("-o", '--offscreen', dest='offscreen', action='store_true', help='offscreen rendering')
    parser.add_argument("-q", '--quality', dest='quality', type=int, default=3, help='render quality')
    parser.add_argument("-u", '--use_lumen', dest='use_lumen', action='store_true', help='use lumen')

    try:
        InterruptedException = False
        args = parser.parse_args()
        type = f'level_{args.level}'
        resolution = (320, 320)
        env = gym.make(args.env_id, action_type='Mixed', observation_type='Color', reset_type=args.level)
        # create the environment
        env = configUE.ConfigUEWrapper(env, offscreen=args.offscreen, resolution=resolution, use_lumen=args.use_lumen, render_quality=args.quality)

        if int(args.time_dilation) > 0:  # -1 means no time_dilation
            env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
        if int(args.early_done) > 0:  # -1 means no early_done, determined by test.jsonl
            env = early_done.EarlyDoneWrapper(env, int(args.early_done))
        if args.monitor:
            env = monitor.DisplayWrapper(env, dynamic_top_down=False, fix_camera=True)
        if args.level>0:
            env = population.RandomPopulationWrapper(env, args.seed, args.seed, random_target=False)
        agent = RandomAgent(env.action_space)
        reward = 0
        done = False
        Total_rewards = 0
        count_step = 0
        for i in range(args.episode):
            if args.point > 0:
                env.unwrapped.injured_player_pose = env.unwrapped.env_configs[type]['injured_player_loc'][args.point]
                env.unwrapped.rescue_pose = env.unwrapped.env_configs[type]['stretcher_loc'][args.point]
                env.unwrapped.agent_pose = env.unwrapped.env_configs[type]['agent_loc'][args.point]
                env.unwrapped.ambulance_pose = env.unwrapped.env_configs[type]['ambulance_loc'][args.point]
            else:
                env.unwrapped.injured_player_pose = env.unwrapped.env_configs[type]['injured_player_loc'][i]
                env.unwrapped.rescue_pose = env.unwrapped.env_configs[type]['stretcher_loc'][i]
                env.unwrapped.agent_pose = env.unwrapped.env_configs[type]['agent_loc'][i]
                env.unwrapped.ambulance_pose = env.unwrapped.env_configs[type]['ambulance_loc'][i]

            obs,info = env.reset()
            count_step = 0
            t0=time.time()
            while True:
                try:
                    # print(env.unwrapped.env_configs[type]['reference_text'][args.level])
                    if args.keyboard:
                        action = get_key_action()
                    else:
                        action = agent.act(obs)
                    obs, reward, termination, truncation, info= env.step([action])
                    if reward != 0:
                        print('Reward:', reward)
                    if args.render:
                        frame,_=env.render()
                    count_step+=1
                    # print(count_step)

                    if termination:
                        fps = count_step / (time.time() - t0)
                        print('Success')
                        print('Fps:' + str(fps))
                        break
                    if truncation:
                        fps = count_step / (time.time() - t0)
                        print('Failed')
                        print('Fps:' + str(fps))
                        break
                except KeyboardInterrupt:
                    print('\nReceived CTRL+C. Cleaning up...')
                    InterruptedException = True
                    break
                except Exception as e:
                    print(e)
                    InterruptedException = True
                    break
            if InterruptedException:
                break
        env.close()

    except Exception as e:
        print(e)
        env.close()
        exit(0)