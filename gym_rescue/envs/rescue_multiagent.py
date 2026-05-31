import time
import numpy as np

from gym_rescue.envs.base_env import UnrealCv_base


class RescueMultiAgent(UnrealCv_base):
    """
    Multi-agent rescue environment that keeps both the ground rescuer and drone.

    Compatibility:
    - Returns the ground agent observation as the main observation.
    - Exposes drone observation / pose through info for downstream wrappers.
    - Accepts either:
      1) a single ground-agent action (legacy behavior), or
      2) a dict/list of joint actions for player + drone.
    """

    def __init__(
        self,
        env_file,
        task_file=None,
        action_type='Discrete',
        observation_type='Color',
        resolution=(160, 160),
        reset_type=0,
        drone_height_offset=300.0,
    ):
        super(RescueMultiAgent, self).__init__(
            setting_file=env_file,
            action_type=action_type,
            observation_type=observation_type,
            resolution=resolution,
            reset_type=reset_type,
        )
        self.count_reach = 0
        self.max_reach_steps = 5
        self.distance_threshold = 120
        self.agents_category = ['player', 'drone']
        self.injured_agent = self.env_configs['injured_player'][0]
        self.stretcher = self.env_configs['stretcher'][0]
        self.ambulance = self.env_configs['ambulance'][0]
        self.injured_player_pose = None
        self.injured_agent_appid = None
        self.rescue_pose = None
        self.agent_pose = None
        self.ambulance_pose = None
        self.drone_height_offset = float(drone_height_offset)
        self.ground_agent_name = None
        self.drone_agent_name = None
        self.drone_id = None
        self._refresh_agent_handles()

    def _refresh_agent_handles(self):
        player_names = [name for name in self.player_list if self.agents[name]['agent_type'] == 'player']
        drone_names = [name for name in self.player_list if self.agents[name]['agent_type'] == 'drone']
        if not player_names:
            raise RuntimeError('RescueMultiAgent requires at least one player agent in env_config')

        self.ground_agent_name = player_names[0]
        self.protagonist_id = self.player_list.index(self.ground_agent_name)
        self.drone_agent_name = drone_names[0] if drone_names else None
        self.drone_id = self.player_list.index(self.drone_agent_name) if self.drone_agent_name else None

        protagonist_name = self.player_list[self.protagonist_id]
        self.action_space = self.define_action_space(self.action_type, self.agents[protagonist_name])
        self.observation_space = self.define_observation_space(
            self.cam_list[self.protagonist_id], self.observation_type, self.resolution
        )

    def _make_drone_pose(self, base_pose):
        pose = list(base_pose[:6])
        if self.env_name in ('Forglar_Map', 'HongKongStreet') and self.injured_player_pose is not None:
            pose[2] = float(self.injured_player_pose[2]) + self.drone_height_offset
        else:
            pose[2] += self.drone_height_offset
        return pose

    def _build_joint_actions(self, action):
        joint_actions = [None] * len(self.player_list)

        # Legacy benchmark call style: env.step([ground_action])
        if isinstance(action, list) and len(action) == 1 and len(self.player_list) > 1:
            action = action[0]

        # Explicit multi-agent dict.
        if isinstance(action, dict):
            for idx, name in enumerate(self.player_list):
                if name in action:
                    joint_actions[idx] = action[name]
                    continue
                agent_type = self.agents[name]['agent_type']
                if agent_type in action:
                    joint_actions[idx] = action[agent_type]
            return joint_actions

        # Explicit list/tuple of per-agent actions.
        if isinstance(action, list) and len(action) == len(self.player_list):
            return list(action)

        # Default: only control the ground rescuer.
        joint_actions[self.protagonist_id] = action
        return joint_actions

    def _augment_info_with_multiagent_views(self, observations, obj_poses):
        self.info['player_list'] = list(self.player_list)
        self.info['ground_agent_name'] = self.ground_agent_name
        self.info['drone_agent_name'] = self.drone_agent_name
        self.info['ground_agent_id'] = self.protagonist_id
        self.info['drone_agent_id'] = self.drone_id
        self.info['AllPose'] = obj_poses
        self.info['AllObservations'] = observations
        self.info['ground_observation'] = observations[self.protagonist_id]
        self.info['ground_pose'] = obj_poses[self.protagonist_id]
        if self.drone_id is not None:
            self.info['drone_observation'] = observations[self.drone_id]
            self.info['drone_pose'] = obj_poses[self.drone_id]
        else:
            self.info['drone_observation'] = None
            self.info['drone_pose'] = None

    def step(self, action):
        joint_actions = self._build_joint_actions(action)
        actions2move, actions2turn, actions2animate = self.action_mapping(joint_actions, self.player_list)
        if self.info['picked'] and actions2animate[self.protagonist_id] == 'jump':
            actions2animate[self.protagonist_id] = 'stand'

        move_cmds = [
            self.unrealcv.set_move_bp(obj, actions2move[i], return_cmd=True)
            for i, obj in enumerate(self.player_list)
            if actions2move[i] is not None
        ]
        head_cmds = [
            self.unrealcv.set_cam(obj, self.agents[obj]['relative_location'], actions2turn[i], return_cmd=True)
            for i, obj in enumerate(self.player_list)
            if actions2turn[i] is not None
        ]
        anim_cmds = [
            self.unrealcv.set_animation(obj, actions2animate[i], return_cmd=True)
            for i, obj in enumerate(self.player_list)
            if actions2animate[i] is not None
        ]
        self.unrealcv.batch_cmd(move_cmds + head_cmds + anim_cmds, None)
        self.count_steps += 1

        obj_poses, cam_poses, imgs, masks, depths = self.unrealcv.get_pose_img_batch(
            self.player_list, self.cam_list, self.cam_flag
        )
        self.obj_poses = obj_poses
        observations = self.prepare_observation(self.observation_type, imgs, masks, depths, obj_poses)
        self.img_show = self.prepare_img2show(self.protagonist_id, observations)
        pose_obs, relative_pose = self.get_pose_states(obj_poses)

        self.info['Pose'] = obj_poses
        self.info['Relative_Pose'] = relative_pose
        self.info['Pose_Obs'] = pose_obs
        self.info['Reward'] = 0
        self._augment_info_with_multiagent_views(observations, obj_poses)

        reward = 0.0
        if self.info['picked']:
            self.target_pose = self.rescue_pose
            if self.first_picked is False:
                reward = 0.5
                self.first_picked = True

        current_pose = [self.unrealcv.get_cam_pose(self.cam_list[self.protagonist_id])]
        metrics = self.rescue_metrics(current_pose, self.target_pose)
        self.info['picked'] = metrics['picked']

        current_injured_player_pose = (
            self.unrealcv.get_obj_location(self.injured_agent)
            + self.unrealcv.get_obj_rotation(self.injured_agent)
        )
        _, dis_tmp, _ = self.get_relative(current_injured_player_pose, self.rescue_pose)

        if not metrics['picked'] and dis_tmp < 200:
            self.info['termination'] = True
            self.info['truncation'] = False
            reward = 0.5
        else:
            self.count_reach = 0
        self.info['Reward'] = reward

        return observations[self.protagonist_id], reward, self.info['termination'], self.info['truncation'], self.info

    def reset(self):
        _, info = super(RescueMultiAgent, self).reset()
        self._refresh_agent_handles()

        self.target_pose = self.injured_player_pose
        self.count_reach = 0
        self.first_picked = False

        self.unrealcv.drop_body(self.ground_agent_name)
        self.unrealcv.set_obj_rotation(self.ground_agent_name, self.agent_pose[3:])
        self.unrealcv.set_obj_location(self.ground_agent_name, self.agent_pose[:3])
        self.unrealcv.set_cam_fov(self.cam_list[self.protagonist_id], 110)
        time.sleep(1)

        if self.drone_agent_name is not None:
            drone_pose = self._make_drone_pose(self.agent_pose)
            self.unrealcv.set_obj_rotation(self.drone_agent_name, drone_pose[3:])
            self.unrealcv.set_obj_location(self.drone_agent_name, drone_pose[:3])
            if self.drone_id is not None and self.cam_list[self.drone_id] >= 0:
                self.unrealcv.set_cam_fov(self.cam_list[self.drone_id], 110)
            time.sleep(1)

        self.unrealcv.set_obj_rotation(self.injured_agent, self.injured_player_pose[3:])
        self.unrealcv.set_obj_location(self.injured_agent, self.injured_player_pose[:3])
        self.injured_player_pose = self.unrealcv.get_obj_location(self.injured_agent)+self.unrealcv.get_obj_rotation(self.injured_agent)
        if self.injured_agent_appid is not None:
            self.unrealcv.set_appearance(self.injured_agent, int(self.injured_agent_appid))
            print(f"[TASK_CUE] injured_agent appearance id={int(self.injured_agent_appid)}")
        random_color = np.random.randint(100, 255, 3)
        self.unrealcv.set_obj_color(self.injured_agent, random_color)
        time.sleep(1)

        self.unrealcv.set_obj_rotation(self.stretcher, self.rescue_pose[3:])
        self.unrealcv.set_obj_location(self.stretcher, self.rescue_pose[:3])
        self.unrealcv.set_obj_scale(self.stretcher, [1.0, 1.0, 0.5])
        self.rescue_pose = self.unrealcv.get_obj_location(self.stretcher)+self.unrealcv.get_obj_rotation(self.stretcher)
        time.sleep(1)

        self.unrealcv.set_obj_rotation(self.ambulance, self.ambulance_pose[3:])
        self.unrealcv.set_obj_location(self.ambulance, self.ambulance_pose[:3])
        time.sleep(1)
        self.unrealcv.set_phy(self.ambulance, 0)
        time.sleep(1)

        observations, self.obj_poses, self.img_show = self.update_observation(
            self.player_list, self.cam_list, self.cam_flag, self.observation_type
        )
        self.info['pose'] = self.obj_poses
        self.info['Pose'] = self.obj_poses
        self.info['Steps'] = self.count_steps
        self.info['termination'] = False
        self.info['truncation'] = False
        self.info['picked'] = False
        self._augment_info_with_multiagent_views(observations, self.obj_poses)
        return observations[self.protagonist_id], self.info

    def reward(self, metrics):
        if 'individual' in self.reward_type:
            if 'sparse' in self.reward_type:
                rewards = metrics['reach']
            else:
                rewards = 1 - metrics['dis_each'] / self.distance_threshold - np.fabs(metrics['ori_each']) / 180 + metrics['reach']
        elif 'shared' in self.reward_type:
            if 'sparse' in self.reward_type:
                rewards = metrics['reach'].max()
            else:
                rewards = 1 - metrics['dis_min'] / self.distance_threshold + metrics['reach'].max()
        else:
            raise ValueError('reward type is not defined')
        return rewards

    def rescue_metrics(self, objs_pose, target_loc):
        info = dict()
        relative_pose = []
        for obj_pos in objs_pose:
            _, distance, direction = self.get_relative(obj_pos, target_loc)
            relative_pose.append(np.array([distance, direction]))
        relative_pose = np.array(relative_pose)
        relative_dis = relative_pose[:, 0]
        relative_ori = relative_pose[:, 1]
        reach_mat = np.zeros_like(relative_dis)
        reach_mat[np.where(relative_dis < self.distance_threshold)] = 1
        reach_mat[np.where(np.fabs(relative_ori) > 45)] = 0

        info['reach'] = reach_mat
        info['dis_min'] = relative_dis.min(-1)
        info['dis_each'] = relative_dis
        info['ori_each'] = relative_ori
        info['picked'] = self.unrealcv.Is_picked(self.injured_agent)
        return info
