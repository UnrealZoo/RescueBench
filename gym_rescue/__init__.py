from gym.envs.registration import register
import logging
from gym_rescue.envs.utils.misc import load_env_setting
logger = logging.getLogger(__name__)
use_docker = False  # True: use nvidia docker   False: do not use nvidia-docker



maps = ['FlexibleRoom','SuburbNeighborhood_Day',
'Forglar_Map','HongKongStreet','track_train','SuburbNeighborhood_Night',
'SuburbNeighborhood_Day_dooropen',
'DesertMap','DowntownWest','JapanTrainStation_Optimised','Tokyo']

Tasks = ['Rescue', 'RescueMultiAgent', 'UnrealCv_base']
Observations = ['Color', 'Depth', 'Rgbd', 'Mask', 'Pose','MaskDepth','ColorMask']
Actions = ['Discrete', 'Continuous', 'Mixed']


# Task-oriented envs
for env in maps:
                for task in Tasks:
                        name = f'Unreal{task}-{env}'
                        setting_env = env
                        if task == 'RescueMultiAgent' and env in {'Forglar_Map', 'HongKongStreet'}:
                                setting_env = f'{env}_drone'
                        setting_file = 'env_config/{env}.json'.format(env=setting_env)
                        register(
                            id=name,
                            entry_point=f'gym_rescue.envs:{task}',
                            kwargs={'env_file': setting_file,},
                            max_episode_steps=10000
                            )

# Clean alias for base env (used by build_nav_graph.py)
for env in maps:
        register(
            id=f'UnrealBase-{env}',
            entry_point='gym_rescue.envs:UnrealCv_base',
            kwargs={'setting_file': f'env_config/{env}.json'},
            max_episode_steps=10000
        )