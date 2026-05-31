import gym
import numpy as np
from gym import Wrapper

DEFAULT_Z_BIAS = {
    'rescuer':   0,
    'injured':   0,
    'stretcher': 0,
    'ambulance': 0,
}


class RescuePoseSamplerWrapper(Wrapper):
    """
    Reset 时用 AgentBasedSamplerboost 为救援任务的四个固定对象
    （rescuer / injured / stretcher / ambulance）采样世界坐标，
    并将结果写入底层 env 的四个 pose 属性，再调用下层 reset。

    四个对象的身份（name / app_id / animation / feature_caption）
    直接从已初始化的 env 中读取，无需额外配置。
    """

    def __init__(
        self,
        env,
        graph_path,
        cam_id,
        cam_height,
        model,
        config_path,
        vehicle_zones=None,
        obj_2_hide=None,
        save_dir=None,
        fast_test_mode=False,
        experiment_config=None,
        z_bias=None,
        normal_variance_threshold=0.1,
        slope_threshold=0.5,
        safety_margin_cm=30,
        gaussian_kernel_size=5,
        gaussian_sigma=1.0,
    ):
        """
        Args:
            env: 底层 Gym 环境
            graph_path: nav 图 .gpickle 文件路径
            cam_id: 俯视采样使用的相机 ID
            cam_height: 俯视相机离地高度 (cm)
            model: VLM 模型名称（如 'gpt-4o'）
            config_path: model_config.json 所在目录路径
            vehicle_zones: 车辆可用区域定义，无车辆时传 {} 或 None
            save_dir: 调试图像保存目录
            fast_test_mode: True 时跳过 VLM，直接随机选点（用于流程验证）
            experiment_config: 传给 sample_single_object 的实验配置字典
            z_bias: {agent_type: z_offset} 按类型修正采样 z 坐标；
                    默认值见 DEFAULT_Z_BIAS
            normal_variance_threshold: 法线方差阈值
            slope_threshold: 坡度阈值
            safety_margin_cm: 安全边距 (cm)
            gaussian_kernel_size: 高斯核大小
            gaussian_sigma: 高斯 sigma
        """
        super().__init__(env)

        self.cam_id = cam_id
        self.cam_height = cam_height
        self.vehicle_zones = vehicle_zones or {}
        self.save_dir = save_dir or './test_results/'
        self.obj_2_hide = obj_2_hide or None
        self.fast_test_mode = fast_test_mode
        self.experiment_config = experiment_config or {}
        self.z_bias = DEFAULT_Z_BIAS.copy()
        if z_bias:
            self.z_bias.update(z_bias)

        self.normal_variance_threshold = normal_variance_threshold
        self.slope_threshold = slope_threshold
        self.safety_margin_cm = safety_margin_cm
        self.gaussian_kernel_size = gaussian_kernel_size
        self.gaussian_sigma = gaussian_sigma

        from example.agent_configs_sampler.agent_sample_agent_advanced import AgentBasedSamplerboost
        self.position_sampler = AgentBasedSamplerboost(
            graph_pickle_file=graph_path,
            model=model,
            config_path=config_path,
        )

    # ------------------------------------------------------------------
    # 私有方法：从 env 读取对象标识，构建 agent_configs
    # ------------------------------------------------------------------

    def _build_rescue_agent_configs(self, env):
        return {
            'rescuer': {
                'name':            [env.player_list[env.protagonist_id]],
                'app_id':          [0],
                'animation':       ['stand'],
                'feature_caption': ['rescue worker searching for injured person'],
            },
            'injured': {
                'name':            [env.injured_agent],
                'app_id':          [0],
                'animation':       ['lie_down'],
                'feature_caption': ['injured person lying on the ground'],
            },
            'stretcher': {
                'name':            [env.stretcher],
                'app_id':          [0],
                'animation':       ['None'],
                'feature_caption': ['medical stretcher for patient evacuation'],
            },
            'ambulance': {
                'name':            [env.ambulance],
                'app_id':          [0],
                'animation':       ['None'],
                'feature_caption': ['ambulance vehicle parked near rescue zone'],
                'type':            ['ambulance'],
            },
        }

    # ------------------------------------------------------------------
    # reset：采样四个对象的 pose，再调用下层 reset
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        env = self.env.unwrapped

        # 1. Lazy-launch UE（首次 reset 时启动）
        if not env.launched:
            env.launched = env.launch_ue_env()
            env.init_agents()
            env.init_objects()

        if self.obj_2_hide is not None:
            env.unrealcv.set_hide_objects(self.obj_2_hide)



        # 2. 构建四个救援对象的 agent_configs
        agent_configs = self._build_rescue_agent_configs(env)

        # 3. Phase 1：建立采样 session
        sampling_kwargs = dict(
            fast_test_mode=self.fast_test_mode,
            save_dir=self.save_dir,
            normal_variance_threshold=self.normal_variance_threshold,
            slope_threshold=self.slope_threshold,
            safety_margin_cm=self.safety_margin_cm,
            gaussian_kernel_size=self.gaussian_kernel_size,
            gaussian_sigma=self.gaussian_sigma,
        )

        # Step 3a: pure graph computation (no UE)
        object_list, agent_sampling_center_pos = \
            self.position_sampler._compute_scene_layout(agent_configs, self.vehicle_zones)

        # Step 3b: all unrealcv camera operations
        ue = env.unrealcv
        orginal_cam_pose = ue.get_cam_location(self.cam_id) + ue.get_cam_rotation(self.cam_id)
        if agent_sampling_center_pos is not None:
            ue.set_cam_location(
                self.cam_id,
                np.append(agent_sampling_center_pos[:2],
                          agent_sampling_center_pos[2] + self.cam_height)
            )
            ue.set_cam_rotation(self.cam_id, [-90, 0, 0])
        obs_bgr    = ue.get_image(self.cam_id, 'lit')
        cam_pose   = ue.get_cam_location(self.cam_id) + ue.get_cam_rotation(self.cam_id)
        fov_deg    = float(ue.get_cam_fov(self.cam_id))
        normal_bgr = ue.get_image(self.cam_id, 'normal')

        # Step 3c: build session state (pure computation, no UE)
        session_state = self.position_sampler.prepare_sampling_session(
            obs_bgr=obs_bgr,
            normal_bgr=normal_bgr,
            cam_pose=cam_pose,
            fov_deg=fov_deg,
            orginal_cam_pose=orginal_cam_pose,
            agent_sampling_center_pos=agent_sampling_center_pos,
            object_list=object_list,
            cam_id=self.cam_id,
            height=self.cam_height,
            **sampling_kwargs
        )

        # 4. Phase 2：逐对象采样位置
        for step_index, obj_info in enumerate(object_list):
            sampled_data = self.position_sampler.sample_single_object(
                session_state, obj_info, self.experiment_config, step_index
            )
            # set agent pose
            agent_name = sampled_data['name']
            agent_loc = sampled_data["position"]
            agent_rot = sampled_data["rotation"]
            env.unrealcv.set_obj_location(agent_name, agent_loc)
            env.unrealcv.set_obj_rotation(agent_name, agent_rot)
        # 5. 还原相机
        env.unrealcv.set_cam_pose(self.cam_id, session_state['orginal_cam_pose'])

        # 6. 将采样结果写入 env 的四个 pose 属性
        for obj in session_state['sampled_objects']:
            pose6d = list(obj['position']) + list(obj['rotation'])  # [x,y,z,roll,yaw,pitch]
            pose6d[2] += self.z_bias.get(obj['agent_type'], 0)      # 按类型修正 z

            t = obj['agent_type']
            if   t == 'rescuer':   env.agent_pose          = pose6d
            elif t == 'injured':   env.injured_player_pose = pose6d
            elif t == 'stretcher': env.rescue_pose         = pose6d
            elif t == 'ambulance': env.ambulance_pose      = pose6d

        if self.obj_2_hide is not None:
            env.unrealcv.set_show_objects(self.obj_2_hide) 

        # 7. 调用下层 reset（Rescue.reset() 会用四个 pose 属性传送对象）
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
