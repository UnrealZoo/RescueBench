import argparse
import gym_rescue
import gym
from gym import wrappers
import cv2
import time
import numpy as np
from gym_rescue.envs.wrappers import time_dilation, early_done, monitor, population, configUE
import os
from pynput import keyboard

os.environ['UnrealEnv'] = 'C:\\AllProject\\gym-rescue\\gym_rescue\\envs\\UnrealEnv\\'

key_state = {
    'i': False,
    'j': False,
    'k': False,
    'l': False,
    'space': False,
    'ctrl':False,
    '1': False,
    '2': False,
    'head_up': False,
    'head_down': False,
    'e': False
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
    if key_state['head_up']:
        action[1] = 1
    if key_state['head_down']:
        action[1] = 2
    if key_state['e']:
        action[2] = 5

    action[0] = tuple(action[0])  # Convert inner list back to tuple
    action = tuple(action)  # Convert list back to tuple
    return action


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

    Maps =['FlexibleRoom','SuburbNeighborhood_Day','HongKongStreet','Forglar_Map']

    # type = 'level_medium'
    for map in Maps:
        env_id = 'UnrealRescue-'+map
        print(env_id)
        env = gym.make(env_id, action_type='Mixed', observation_type='Color',reset_type=args.level)
        type_list =[key for key in env.unwrapped.env_configs.keys() if key.startswith('level_')]
        for type in type_list: #遍历难度级别
            sample_point_num = len(env.unwrapped.env_configs[type]['agent_loc'])
            for i in range(0,sample_point_num): #遍历每个难度的所有采样点

                #设置初始的agent， injured agent， stretcher， ambulance位置 （重点检查reset后，是否存在设置位置失败的情况）
                env.unwrapped.injured_player_pose = env.unwrapped.env_configs[type]['injured_player_loc'][i]
                env.unwrapped.rescue_pose = env.unwrapped.env_configs[type]['stretcher_loc'][i]
                env.unwrapped.agent_pose = env.unwrapped.env_configs[type]['agent_loc'][i]
                env.unwrapped.ambulance_pose = env.unwrapped.env_configs[type]['ambulance_loc'][i]
                env = configUE.ConfigUEWrapper(env, offscreen=False,resolution=(640,480))

                app_id = env.unwrapped.env_configs[type]['injured_agent_id'][i]# 

                reference_text=env.unwrapped.env_configs[type]['reference_text'][i][0]
                goal_frame_path = os.path.join('C:\\AllProject\\gym-rescue\\gym_rescue\\envs\\setting\\ref_image',env.unwrapped.env_configs[type]['referemce_image_path'][i][0]) #自己用os.join 设置一下正确的image路径
                goal_frame=cv2.imread(goal_frame_path) #
                goal_frame = cv2.resize(goal_frame, (640, 480)) #
                for idx, line in enumerate([reference_text[i:i + 50] for i in range(0, len(reference_text), 50)]):
                    cv2.putText(goal_frame, line, (10, 30 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                1, cv2.LINE_AA) #
                rewards = 0
                done = False
                Total_rewards = 0
                count_step=0
                env.seed(int(args.seed))
                obs,info = env.reset()
                print('Test point {} in {}-{}'.format(i,map,type))

                env.unwrapped.unrealcv.set_appearance(env.unwrapped.injured_agent,app_id)

                t0=time.time()

                # 
                video_path = f'D:\\rescue\\test_point_{i}_{map}_{type}.avi'
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(video_path, fourcc, 20.0, (1280, 480))

                while True:
                    #
                    action = get_key_action()
                    obs, rewards, termination,truncation, info = env.step([action])
                    frame = cv2.hconcat((goal_frame,obs[0]))
                    cv2.imshow('ageng',frame)
                    cv2.waitKey(1)

                    # 
                    out.write(frame)

                    if termination:
                        fps = count_step / (time.time() - t0)
                        print('Success')
                        break
                    if truncation:
                        fps = count_step / (time.time() - t0)
                        print('Failed')
                        print('Fps:' + str(fps))
                        break
                # 
                out.release()
                print(f'Video saved to {video_path}')

        env.close()
        time.sleep(3)



