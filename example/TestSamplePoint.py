import argparse
import gym_rescue
import gym
from gym import wrappers
import cv2
import time
import numpy as np
from gym_rescue.envs.wrappers import time_dilation, early_done, monitor, population, configUE
import os
os.environ['UnrealEnv']='E:\\gym-rescue\\gym-rescue\\gym_rescue\\envs\\UnrealEnv'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-track_train-ContinuousMask-v4',
    #                     help='Select the environment to run')

    # parser.add_argument("-e", "--env_id", nargs='?', default='UnrealRescue-AbandonedIndustry', help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=10, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("-l", '--level', dest='level', default=0, help='Difficulty level for rescue task(0-4) ')


    args = parser.parse_args()

    Maps =['AbandonedIndustry','SuburbNeighborhood_Day','track_train']

    # type = 'level_medium'
    for map in Maps:
        env_id = 'UnrealRescue-'+map
        env = gym.make(env_id, action_type='Mixed', observation_type='Color',reset_type=args.level)
        type_list =[key for key in env.unwrapped.env_configs.keys() if key.startswith('level_')]
        for type in type_list:
            sample_point_num = len(env.unwrapped.env_configs[type]['agent_loc'])
            for i in range(0,sample_point_num):
                env.unwrapped.injured_player_pose = env.unwrapped.env_configs[type]['injured_player_loc'][i]
                env.unwrapped.rescue_pose = env.unwrapped.env_configs[type]['stretcher_loc'][i]
                env.unwrapped.agent_pose = env.unwrapped.env_configs[type]['agent_loc'][i]
                env.unwrapped.ambulance_pose = env.unwrapped.env_configs[type]['ambulance_loc'][i]
                env = configUE.ConfigUEWrapper(env, offscreen=False,resolution=(640,480))

                rewards = 0
                done = False
                Total_rewards = 0
                count_step=0
                env.seed(int(args.seed))
                obs,info = env.reset()
                print('Test point {} in {}-{}'.format(i,map,type))

                actions = [([0, 100], 0, 0),([20, 100], 0, 0)]
                for action in actions:
                    obs, rewards, termination,truncation, info = env.step([action])
                    cv2.imshow('ageng',obs[0])
                    cv2.waitKey(1)
                    time.sleep(1)

                current_agent_pose = env.unwrapped.unrealcv.get_obj_pose(env.unwrapped.player_list[env.unwrapped.protagonist_id])
                current_injured_agent_pose = env.unwrapped.unrealcv.get_obj_pose(env.unwrapped.injured_agent)
                current_stretcher_pose = env.unwrapped.unrealcv.get_obj_pose(env.unwrapped.stretcher)
                current_ambulance_pose = env.unwrapped.unrealcv.get_obj_pose(env.unwrapped.ambulance)

                #check agent movement and interaction
                distance_tmp = np.linalg.norm(np.array(current_agent_pose[:2]) - np.array(env.unwrapped.agent_pose[:2]))
                rotation_tmp = abs(current_agent_pose[4] - env.unwrapped.agent_pose[4])
                assert distance_tmp>50, "Agent not moving"
                assert rotation_tmp>19 , "Agent not rotating"

                #check injured player start location
                distance_tmp = np.linalg.norm(np.array(current_injured_agent_pose[:2]) - np.array(env.unwrapped.injured_player_pose[:2]))
                assert distance_tmp<50, "Injured player not settled correctly"

                distance_tmp = np.linalg.norm(np.array(current_stretcher_pose[:2]) - np.array(env.unwrapped.rescue_pose[:2]))
                assert distance_tmp<50, "Stretcher not settled correctly"

                distance_tmp = np.linalg.norm(np.array(current_ambulance_pose[:2]) - env.unwrapped.ambulance_pose[:2])
                assert distance_tmp<50, "Ambulance not settled correctly"
        env.close()
        time.sleep(3)



