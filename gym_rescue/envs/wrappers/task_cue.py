import os.path
import cv2
import gym
import gym_rescue
from gym import Wrapper
from unrealcv.util import get_path2UnrealEnv
import json
class TaskCueWrapper(Wrapper):
    def __init__(self, env, level, point_id):
        super().__init__(env)
        self.level = level
        self.point_id = point_id
        # file_name = 'user_L'+str(self.level)+'.jsonl'
        file_name = 'level_'+str(self.level)+'.jsonl'

        gympath = os.path.dirname(gym_rescue.__file__)
        json_file=os.path.join(gympath, 'envs/setting/test_jsonl', file_name)
        self.test_json = []
        with open(json_file, 'r') as f:
            for line in f:
                self.test_json.append(json.loads(line))
        self.test_point = self.test_json[self.point_id]
        env.unwrapped.injured_player_pose = self.test_point['injured_player_loc']
        env.unwrapped.rescue_pose = self.test_point['stretcher_loc']
        env.unwrapped.agent_pose = self.test_point['agent_loc']
        env.unwrapped.ambulance_pose = self.test_point['ambulance_loc']

    def step(self, action):
        obs, reward, termination,truncation, info = self.env.step(action)
        return obs, reward, termination,truncation, info

    def reset(self, **kwargs):
        states,info = self.env.reset(**kwargs)
        injured_agent_appid = self.test_point['injured_agent_id']
        self.env.unwrapped.unrealcv.set_appearance(self.env.unwrapped.injured_agent, injured_agent_appid)

        gympath = os.path.dirname(gym_rescue.__file__)
        image_path = os.path.join(gympath, 'envs/setting/ref_image', self.test_point['reference_image_path'][0])

        image = cv2.imread(image_path)
        text = self.test_point['reference_text'][0]
        info['reference_image'] = image
        info['reference_text']=text

        # Display the reference image and text
        goal_show = cv2.resize(info['reference_image'], (480, 320))
        reference_text = info['reference_text']
        for idx, line in enumerate([reference_text[i:i + 50] for i in range(0, len(reference_text), 50)]):
            y = 30 + idx * 20
            # Create rectangle background
            overlay = goal_show.copy()
            cv2.rectangle(overlay, (5, y - 15), (len(line) * 10 + 15, y + 5), (128, 128, 128), -1)
            # Apply semi-transparent overlay
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, goal_show, 1 - alpha, 0, goal_show)
            # Draw text on top of background
            cv2.putText(goal_show, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
        self.env.unwrapped.goal_show = goal_show
        return states,info