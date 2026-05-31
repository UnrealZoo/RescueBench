import argparse
import gym
import cv2
import time
import numpy as np
from gym_rescue.envs.wrappers import time_dilation, early_done, monitor, population, configUE
import base64
from PIL import Image
import io
from openai import OpenAI

system_prompt="""
Objective:
You are an intelligent rescue agent designed to perform a rescue operation. Your task is to find and help a person in need, navigating through the environment based on visual input. The agent will be provided with first-person visual observations (RGB images) of the environment. The agent needs to first turn around to find the target then infer the relative position of the target person (a person lying on the ground), estimate the distance to the target, and decide on a single action to either move closer to the target, perform the rescue, or place the person on the stretcher, depending on the current goal.

Representation Details:
1. **Visual Input**: The input consists of a first-person RGB image showing the environment, including the location of obstacles, the target person (lying on the ground), and the stretcher.
2. **Target Detection**: The agent will identify the target person in the image. The target is explicitly a person lying on the ground, which can be detected by visual cues such as body posture.
3. **Distance Estimation**: The agent should estimate the distance to the target based on the visual input. The estimated distance helps determine whether the agent needs to continue moving closer to the target or the stretcher, or whether it can perform the "Rescue" or "Place" action.
4. **Current Goal**: The agent is aware of its current goal, which can be either:
   - **Toward the Target**: If the agent is approaching the person to perform the rescue.
   - **Toward the Stretcher**: After the rescue, if the agent is navigating toward the stretcher to place the person on it.

5. **Action Space**: The available actions the agent can perform are as follows:
   - **Move Forward**: Move forward by 1 meter (100 cm).
   - **Move Backward**: Move backward by 50 cm.
   - **Turn Left**: Rotate 90 degrees to the left.
   - **Turn Right**: Rotate 90 degrees to the right.
   - **Look Up**: Tilt the camera up.
   - **Look Down**: Tilt the camera down.
   - **Stay**: Do not move, stay in the current position.
   - **Rescue**: Perform the rescue operation while the agent arrived near the either side of the target's upper arm, the distance should be less than 0.5m(help the person, such as stabilizing or lifting them).
   - **Place**: Perform the Place operation while the agent arrived near the either side of stretcher, distance should be less than 0.5m.

Input Understanding:
1. **Image**: The input is a first-person view RGB image showing the environment, obstacles, the target person (lying on the ground), and the stretcher.
2. **Past Memories**: A list of recent state-action pairs in the format (Visible, Action) that describes the visibility state and action sequence leading to the current step. Example: [(True, Move Forward), (False, Turn Left), (True, Look Down)].
3. **Current Goal**: The agent is aware of its current goal, which is either to move toward the target (the person) or toward the stretcher after the rescue.

Output Understanding:
1. **Output Format**: The output should contain two clearly separated elements:
   - [chain-of-thought]: A brief explanation of how the agent inferred the correct action, including reasoning based on the visual input, the estimated distance to the target/stretcher, and the current goal.
   - [Visible]: A Boolean value indicating whether the current goal (target or stretcher) is visible in the current observation.
   - [Action]: A single action chosen from the available action space, based on the reasoning in the chain-of-thought.
   Each element should be clearly labeled to ensure readability and consistency.

Strategy Considerations:
1. **Goal-Oriented Action**: Depending on the current goal, the agent should predict actions that move it toward the target or stretcher. For example:
   - If the agent is moving toward the target, actions should help it approach the person.
   - If the agent is moving toward the stretcher, actions should help it navigate toward the stretcher.
2. **Target Detection and Distance Estimation**: First, detect the target person in the imag. Once identified, inference the next reasonable direction to get closer to the target.
3.**Past Memories for Decision-Making**:
    -If the target is not visible from begining, continuous turning around in a consistent direction to search the target
    -Use recent (Visible, Action) pairs to identify patterns (e.g., repeated "Look Down" actions indicating proximity to the target).
    -Adjust predictions accordingly (e.g., moving forward if the target is assumed to be directly below).
4. **Maintain Visual Contact with the Target**:
    -If the agent is close to the target and the target is no longer visible in the center or upper half of the visual field, the agent should predict the Look Down action to re-center the target in the field of view.
    -If the agent has already looked down and the target is still not visible, the agent should determine the most likely position of the target (e.g., immediately below) and attempt to re-establish visual contact by slightly adjusting position.
5. **Rescue Operation**: If the agent is close enough to the either side of the target and its goal is "Toward the Target," the agent should predict the "Rescue" action.
6. **Placing the Target**: After the rescue, if the agent’s goal is "Toward the Stretcher," it should move toward the stretcher and predict the "Place" action once it is in range.
7. **Action Prediction**: The agent should predict actions based on the target’s relative position and the distance to the target/stretcher. The prediction must be coherent with the current goal (whether it's rescuing the person or placing them on the stretcher).

Instructions:
1. For each step, provide the output in the following format:
   - [chain-of-thought]: ***. 
   - [Visible]: ***.
   - [Action]: ***.
2. Ensure the reasoning and action align with the current goal and visual input.
3. The rescue action will be execute successfully only when the agent is close enough to either side of the target(face to the target left body side or rigth body side)

Example Inputs and Outputs:

[input:]
Image: 
Past Memories: [(True, Move Forward), (True, Turn Left), (False, Look Down)].
Current Goal: Toward the Target

[output:]
[chain-of-thought]: The target is visible on the left side of the frame. Based on past actions, I have already adjusted the view downward, and the target is now in view. The estimated distance is medium. Moving forward will bring me closer.  
[Visible]: True.  
[Action]: Move Forward.  



[input:]
Image: 
Past Memories: [(True, Move Forward), (True, Look Down), (True, Move Forward)].
Current Goal: Toward the Target

[output:]
[chain-of-thought]: The target is no longer visible but was visible in the last step. Based on my proximity, it is likely directly below. Looking down further will help regain visibility.  
[Visible]: False.  
[Action]: Look Down.  


[input:]
Image:
Past Memories: [(True, Rescue), (False, Turn Left), (True, Move Forward)].
Current Goal: Toward the Stretcher

[output:]
[chain-of-thought]: The stretcher is visible and centered in the frame. The estimated distance is around 2 meters. I will move forward to reduce the distance.  
[Visible]: True.  
[Action]: Move Forward.  


"""
prompt_template = """
Image:
Current Goal: {CURRENT_GOAL}
Past Memories: {PAST_MEMORIES}
"""
def encode_image_array(image_array):
    # Convert the image array to a PIL Image object
    image = Image.fromarray(np.uint8(image_array))

    # Save the PIL Image object to a bytes buffer
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")  # You can change JPEG to PNG or other formats depending on your needs

    # Encode the bytes buffer to Base64
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str
def call_gpt_api( current_goal, past_memories,base64_image=None):

    User_prompt = prompt_template.format(
        CURRENT_GOAL=current_goal,
        PAST_MEMORIES = past_memories
    )



    client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        # sk-xxx替换为自己的key
        api_key=''
    )

    response = client.chat.completions.create(
        model='gpt-4o',
        max_tokens=300,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": User_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }]
             },
        ],

    )

    return response.choices[0].message.content

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-track_train-ContinuousMask-v4',
    #                     help='Select the environment to run')

    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealRescue-track_train',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=10, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1,
                        help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true',
                        help='use nav agent to control the agents')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("-l", '--level', dest='level', default=0, help='Difficulty level for rescue task(0-4) ')

    args = parser.parse_args()
    env = gym.make(args.env_id, action_type='Mixed', observation_type='Color', reset_type=args.level)
    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)
    if args.level > 0:
        env = augmentation.RandomPopulationWrapper(env, 4, 10, random_target=False)
    env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(1280, 960))
    rewards = 0
    done = False
    Total_rewards = 0
    count_step = 0
    env.seed(int(args.seed))
    obs, info = env.reset()
    t0 = time.time()
    action=([0, 0], 0, 0)
    try:
        for i in range(10):
            env.seed(i)
            ob,info = env.reset()
            count_step = 0
            t0 = time.time()
            action_idx = 0
            past_memories = []
            current_goal = "Toward the Target"
            while True:
                obs, rewards, termination, truncation, info = env.step([action])
                response = call_gpt_api(current_goal,  past_memories, encode_image_array(obs))
                print(response)
                action = ([0, 0], 0, 0)
                action = list(action)  # Convert tuple to list for modification
                action[0] = list(action[0])
                action_pred = response.split('[Action]: ')[1].strip()
                visible = response.split('[Visible]: ')[1].split('[Action]: ')[0].strip()
                if env.unwrapped.unrealcv.is_carrying(env.unwrapped.player_list[env.unwrapped.protagonist_id]):
                    current_goal = "Toward the Stretcher"
                    past_memories=[]
                past_memories.append([visible, action_pred])
                if 'move forward' in action_pred.lower():
                    action[0][1] = 100
                elif 'move backward' in action_pred.lower():
                    action[0][1] = -50
                elif 'turn right' in action_pred.lower():
                    action[0][0] = 30
                elif 'turn left' in action_pred.lower():
                    action[0][0] = -30
                elif 'stay' in action_pred.lower():
                    pass
                elif 'rescue' in action_pred.lower():
                    action[2] = 3
                elif 'place' in action_pred.lower():
                    action[2] = 4
                elif 'look up' in action_pred.lower():
                    action[1] = 1
                elif 'look down' in action_pred.lower():
                    action[1] = 2
                action[0] = tuple(action[0])  # Convert inner list back to tuple
                action = tuple(action)
                print(action)
                cv2.imshow('mask', obs)
                cv2.waitKey(1)

                count_step += 1
                if termination:
                    fps = count_step / (time.time() - t0)
                    print('Success')
                    break
                if truncation:
                    fps = count_step / (time.time() - t0)
                    print('Failed')
                    print('Fps:' + str(fps))
                    break

        # Close the env and write monitor result info to disk
        env.close()
    except KeyboardInterrupt:
        print('exiting')
        env.close()



