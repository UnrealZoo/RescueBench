import io
import sys
# sys.path.append('/home/wuk/gym-rescue')
import base64
from PIL import Image
import numpy as np
from gym_rescue.envs.wrappers import time_dilation, early_done, monitor, population, configUE
import argparse
import gym
import time
import torch
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, predict
import grounding_dino.groundingdino.datasets.transforms as T
import matplotlib.pyplot as plt
import cv2

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
system_prompt_rescue = """
Objective:
You are a rescue agent navigating a first-person environment to locate and assist a person lying on the ground in front of you. Using first-person RGB images and detection results from a Grounding DINO model, predict the best action to achieve the rescue task.

**Detection Input**:
The Grounding DINO model provides the following structured information:
1. **Person Bounding Box**: 
   - Coordinates: [x_min, y_min, x_max, y_max].
   - Normalized Area: A float value (0 to 1) indicating the person's relative size in the image.

**Action Space**:
You can choose one of the following actions:
- Move Forward, Move Backward, Turn Left, Turn Right, Rescue.

**Decision Strategy**:
1. **Target Visibility**:
   - If no bounding box is detected, assume the person is not visible and take exploratory actions, based on past memories (e.g., Turn Left, Right, move forward, etc).
   - If a bounding box is detected, analyze its position and size for decision-making.
2. **Target Position Analysis**:
   - **Horizontal Alignment**:
     - Left: x_max < 0.3.
     - Centered: 0.3 ≤ x_min and x_max ≤ 0.7.
     - Right: x_min > 0.7.
   - **Vertical Proximity**:
     - Near the bottom edge: y_max > 0.8.
   - **Size**:
     - Small: Normalized Area < 0.02.
     - Medium: Normalized Area 0.02–0.045.
     - Large: Normalized Area > 0.045
3. **Rescue Condition**:
   - The person is considered ready for rescue if:
     - The bounding box is centered horizontally.
     - The vertical position is near the bottom edge.
     - The normalized area is large.
4. **Action Recommendation**:
   - "Move Forward": If the target size is small .
   - "Move Backward": If the target size is small and near the bottom edge(y_max ≥ 0.8)
   - "Turn Left" : If the Target Position is in the Left.
   - "Turn Right": If the Target Position is in the Right.
   - "Rescue": If the target is centered, near the bottom edge (y_max ≥ 0.8), and the bounding box size is Large.
5. **Failure Recovery**:
   - If "Rescue" fails (person remains visible), adjust position by centering the bounding box and ensuring sufficient proximity.

**Output Requirements**:
Your output must strictly follow this structured format:
[Target Visibility]: Is a bounding box detected? (True/False).
[Bounding Box Details]: 
   - Position: Left/Centered/Right.
   - Size: Small (<0.02), Medium (0.02–0.045), Large (>0.045).
[Recommended Action]: Select a single action from the action space based on the analysis.

**Output Format Reminder**:
- Always output the structured fields exactly as defined.
- Avoid including any extra text or explanations.

**Example Inputs and Outputs**:
[input:]
Detection:
- Bounding Box: [0.4, 0.6, 0.6, 0.88].
- Area: 0.06.
Past Memories: [(True, Move Forward)].

[output:]
[Target Visibility]: True.
[Bounding Box Details]: Centered, Large.
[Recommended Action]: Rescue.

[input:]
Detection:
- Bounding Box: None.
Past Memories: [(False, Turn Right), (False, Turn Left)].

[output:]
[Target Visibility]: False.
[Bounding Box Details]: N/A.
[Recommended Action]: Turn Left.

"""
system_prompt_place = """
Objective:
You are a rescue agent navigating a first-person environment to find the yellow stretcher near the ambulance cat and put the injured person on the stretcher. Using first-person RGB images and detection results from a Grounding DINO model, predict the best action to achieve the task.

**Detection Input**:
The Grounding DINO model provides the following structured information:
1. **Stretcher Bounding Box**: 
   - Coordinates: [x_min, y_min, x_max, y_max].
   - Normalized Area: A float value (0 to 1) indicating the Stretcher's relative size in the image.

**Action Space**:
You can choose one of the following actions:
- Move Forward, Move Backward, Turn Left, Turn Right, Place.

**Decision Strategy**:
1. **Target Visibility**:
   - If no bounding box is detected, assume the Stretcher is not visible and take exploratory actions, based on past memories (e.g., Turn Left, Right, move forward, etc).
   - If a bounding box is detected, analyze its position and size for decision-making.
2. **Target Position Analysis**:
   - **Horizontal Alignment**:
     - Left: x_max < 0.3.
     - Centered: 0.3 ≤ x_min and x_max ≤ 0.7.
     - Right: x_min > 0.7.
   - **Vertical Proximity**:
     - Near the bottom edge: y_max > 0.8.
   - **Size**:
     - Small: Normalized Area < 0.02.
     - Medium: Normalized Area 0.02–0.045.
     - Large: Normalized Area > 0.045
3. **Place Condition**:
   - The injured person is considered ready for be placed if:
     - The bounding box is centered horizontally.
     - The vertical position is near the bottom edge.
     - The normalized area is large.
4. **Action Recommendation**:
   - "Move Forward": If the Stretcher size is small .
   - "Turn Left" : If the Stretcher is in the Left.
   - "Turn Right": If the Stretcher is in the Right.
   - "Place": If the Stretcher is centered, near the bottom edge (y_max ≥ 0.8), and the bounding box size is Large.
5.**Past Memories for Decision-Making**:
    -If the target is not visible for many steps, continuous turning around in a consistent direction to search the target
    -Use recent (Visible, Action) pairs to identify patterns, avoid invalid actions(e,g. same actions for many step while the target is still not detected).

**Output Requirements**:
Your output must strictly follow this structured format:
[Target Visibility]: Is a bounding box detected? (True/False).
[Bounding Box Details]: 
   - Position: Left/Centered/Right.
   - Size: Small (<0.02), Medium (0.02–0.045), Large (>0.045).
[Recommended Action]: Select a single action from the action space based on the analysis.

**Output Format Reminder**:
- Always output the structured fields exactly as defined.
- Avoid including any extra text or explanations.

**Example Inputs and Outputs**:
[input:]
Detection:
- Bounding Box: [0.4, 0.6, 0.6, 0.88].
- Area: 0.06.
Past Memories: [(True, Move Forward)].

[output:]
[Target Visibility]: True.
[Bounding Box Details]: Centered, Large.
[Recommended Action]: Place.

[input:]
Detection:
- Bounding Box: None.
Past Memories: [(False, Turn Right), (False, Turn Left)].

[output:]
[Target Visibility]: False.
[Bounding Box Details]: N/A.
[Recommended Action]: Turn Left.

"""

prompt_template = """
[input:]
Detection:
- Bounding Box: {BOUNDING_BOX}.
- Area: {AREA}.
Past Memories: {PAST_MEMORIES}

[output:]
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

def call_huggingface_api( current_goal, bounding_box, area, past_memories,gemma2_model,tokenizer):

    if current_goal=="Toward the Target":
        system_prompt=system_prompt_rescue
    else:
        system_prompt=system_prompt_place

    User_prompt = prompt_template.format(
        BOUNDING_BOX=bounding_box,
        AREA=area,
        PAST_MEMORIES=past_memories
    )

    messages = [
        {"role": "user", "content": User_prompt}
    ]
    # input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    outputs = gemma2_model.generate(input_ids=inputs.to(gemma2_model.device), max_new_tokens=250)

    return tokenizer.decode(outputs[0])

def normalize_bbox_and_area(bbox, image_shape):
    """
    Normalize bounding box coordinates and area.

    Parameters:
    bbox (list or tuple): Bounding box coordinates [x_min, y_min, x_max, y_max].
    image_shape (tuple): Shape of the image (height, width).

    Returns:
    tuple: Normalized bounding box coordinates and area.
    """
    height, width = image_shape[:2]
    x_min, y_min, x_max, y_max = bbox

    # Normalize coordinates
    x_min_norm = x_min / width
    y_min_norm = y_min / height
    x_max_norm = x_max / width
    y_max_norm = y_max / height

    # Calculate and normalize area
    area = (x_max - x_min) * (y_max - y_min)
    normalized_area = area / (width * height)

    normalized_bbox = [x_min_norm, y_min_norm, x_max_norm, y_max_norm]
    return normalized_bbox, normalized_area
def init_groundsam():
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )
    return grounding_model,sam2_predictor
def load_image(image):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.fromarray(image).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed
def groundingsam_generate(grounding_model,sam2_predictor,image, text_prompt):
    text = text_prompt
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    if "Stretcher.Ambulance car"in text:
        BOX_THRESHOLD = 0.5
        TEXT_THRESHOLD = 0.5
    raw_image = image.copy()
    image_source, image = load_image(image)

    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # FIXME: figure how does this influence the G-DINO model
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    """
    Post-process the output of the model to get the masks, scores, and logits for visualization
    """
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    confidences = confidences.numpy().tolist()
    class_names = labels

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    """
    Visualize image with supervision useful API
    """
    img = raw_image
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=class_ids
    )
    normalized_box, normalized_box_area = normalize_bbox_and_area(detections.xyxy[0], image_source.shape)
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)
    # mask_annotator = sv.MaskAnnotator()
    # annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    # cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)


    return normalized_box, normalized_box_area,annotated_frame

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
    parser.add_argument("-l", '--level', dest='level', default=2, help='Difficulty level for rescue task(0-4) ')

    args = parser.parse_args()
    env = gym.make(args.env_id, action_type='Mixed', observation_type='Color', reset_type=args.level)
    env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(480, 480))
    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)
    # if args.level > 0:
    #     env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    rewards = 0
    done = False
    Total_rewards = 0
    count_step = 0
    env.seed(int(args.seed))
    obs, info = env.reset()
    env.unwrapped.unrealcv.set_obj_location(env.unwrapped.stretcher, env.unwrapped.rescue_pose[:3])
    env.unwrapped.unrealcv.set_obj_rotation(env.unwrapped.stretcher, env.unwrapped.rescue_pose[3:])
    env.unwrapped.unrealcv.set_obj_location(env.unwrapped.ambulance,  env.unwrapped.env_configs['level_simple']['ambulance_loc'][args.level][:3])
    env.unwrapped.unrealcv.set_obj_rotation(env.unwrapped.ambulance,  env.unwrapped.env_configs['level_simple']['ambulance_loc'][args.level][3:])

    t0 = time.time()
    action = ([0, 0], 0, 0)
    
    grounding_model,sam2_predictor=init_groundsam()
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
    gemma2_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-27b-it",
        device_map="auto",
        torch_dtype= torch.bfloat16
    )
    messages = [
        {"role": "user", "content": system_prompt_rescue}
    ]
    # input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_specialtokens=False, return_tensors="pt")
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    outputs = gemma2_model.generate(input_ids=inputs.to(gemma2_model.device), max_new_tokens=250)

    try:
        for i in range(1):
            env.seed(i)
            ob, info = env.reset()

            #cv2.imwrite('first_obs.png',ob)
            #import pdb
            #pdb.set_trace()
            count_step = 0
            t0 = time.time()
            action_idx = 0
            past_memories = []
            current_goal = "Toward the Target"
            text_prompt = "person."

            while True:
                obs, rewards, termination, truncation, info = env.step([action])
                try:
                    normalized_box, normalized_box_area,annotated_frame=groundingsam_generate(grounding_model, sam2_predictor, obs, text_prompt)
                    plt.imshow(annotated_frame)
                    plt.show()
                except:
                    normalized_box=None
                    normalized_box_area=0
                response = call_huggingface_api(current_goal, normalized_box, normalized_box_area,past_memories,gemma2_model,tokenizer)
                response = response.lower()
                print(response)
                action = ([0, 0], 0, 0)
                action = list(action)  # Convert tuple to list for modification
                action[0] = list(action[0])
                try:
                    action_pred = response.split('[recommended action]:')[1].split('.')[0].strip()
                    visible = response.split('[target visibility]:')[1].split('.')[0].strip()
                    if "true" in visible.lower():
                        visible=True
                    elif "false" in visible.lower():
                        visible=False
                    else:
                        visible=None
                    # cv2.imwrite('/home/admin/rank_project/obs.png',obs)
                    # import pdb
                    # pdb.set_trace()
                except:
                    action_pred=''
                    visible = False
                    pass
                if info['picked']:
                    if current_goal == "Toward the Target":
                        visible=None
                        action_pred=''
                        past_memories = []
                    current_goal = "Toward the Stretcher"
                    text_prompt="Stretcher.Ambulance car."
                past_memories.append([visible, action_pred])
                if 'move forward' in action_pred.lower():
                    action[0][1] = 100
                if 'move backward' in action_pred.lower():
                    action[0][1] = -50
                if 'turn right' in action_pred.lower():
                    action[0][0] = 25
                if 'turn left' in action_pred.lower():
                    action[0][0] = -25
                if 'stay' in action_pred.lower():
                    pass
                if 'rescue' in action_pred.lower():
                    action[2] = 3
                elif 'place' in action_pred.lower():
                    action[2] = 4
                if 'look up' in action_pred.lower():
                    action[1] = 1
                elif 'look down' in action_pred.lower():
                    action[1] = 2
                action[0] = tuple(action[0])  # Convert inner list back to tuple
                action = tuple(action)
                print(action)

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
