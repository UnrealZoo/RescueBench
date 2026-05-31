import json
import os
import random
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate test_jsonl JSON files for gym_rescue environment.')
parser.add_argument('--output_level', type=int, default=3, help='Output level for the test_jsonl JSON files.')
parser.add_argument('--path', type=str, default='../gym_rescue/envs/setting/env_config', help='Path to the environment configuration files.')
parser.add_argument('--ref_img_path', type=str, default='/home/admin/gym-rescue/gym_rescue/envs/setting/ref_image', help='Path to the reference images.')
# parser.add_argument('--output_path', type=str, default='/media/littlecave/T9/Offline_RL_Active_Tracking/gym-rescue/gym_rescue/envs/setting/env_config', help='Path to save the generated test_jsonl JSON files.')
parser.add_argument('--output_path', type=str, default='../gym_rescue/envs/setting/test_jsonl', help='Path to save the generated test_jsonl JSON files.')
parser.add_argument('--episode_count', type=int, default=5, help='Number of episodes to generate.')

args = parser.parse_args()

output_level = args.output_level
path = args.path
ref_img_path = args.ref_img_path
output_path = args.output_path
episode_count = args.episode_count
# path = '../gym_rescue/envs/setting/env_config'
# ref_img_path = '../gym_rescue/envs/setting/ref_image'
# output_path = '../gym_rescue/envs/setting/test_jsonl'
# episode_count=5

map_list = os.listdir(path)

template = {
    "env_id": None,
    "level": output_level,
    "agent_loc": [],
    "injured_player_loc": [],
    "injured_agent_id": [],
    "stretcher_loc": [],
    "ambulance_loc": [],
    "reference_text":None,
    "reference_image_path":None,
    "timeout":None
}
# map_name = random.choice(map_list)
for map_name in map_list:
    with open(os.path.join(path, map_name), 'r') as f:
        config = json.load(f)
        if 'level_'+str(output_level) in config["env"]:
            for i in range(0,len(config["env"]['level_'+str(output_level)]['agent_loc'])):
                json_template = template.copy()
                json_template["env_id"] ="UnrealRescue-"+ config["env_name"]
                json_template["agent_loc"]=config["env"]['level_'+str(output_level)]['agent_loc'][i]
                json_template["injured_player_loc"]=config["env"]['level_'+str(output_level)]['injured_player_loc'][i]
                json_template["injured_agent_id"]=config["env"]['level_'+str(output_level)]['injured_agent_id'][i]
                json_template["stretcher_loc"]=config["env"]['level_'+str(output_level)]['stretcher_loc'][i]
                json_template["ambulance_loc"]=config["env"]['level_'+str(output_level)]['ambulance_loc'][i]
                json_template["reference_text"]=config["env"]['level_'+str(output_level)]['reference_text'][i]
                # img_name = config["env_name"]+'_simple_'+str(i)+'.png'
                # json_template["reference_image_path"]=os.path.join(ref_img_path,img_name)
                json_template["reference_image_path"] = config["env"]['level_'+str(output_level)]['reference_image_path'][i]
                if output_level<2:
                    json_template["timeout"] = 180
                elif output_level==2:
                    json_template["timeout"] = 240
                else:
                    json_template["timeout"] = 300
                # output_file_name = 'test_L'+str(output_level)+'.jsonl'
                output_file_name = 'level_'+str(output_level)+'.jsonl'

                with open(os.path.join(output_path, output_file_name), 'a') as f:
                    json.dump(json_template, f)
                    f.write('\n')
