import argparse
import copy
import math
import os
from matplotlib import colors
import cv2
import numpy as np
import pandas
import skimage
import torch
import habitat
import sys
import pdb

# 使用本仓库根目录；SAM 内层包优先。
_sgnav_root = os.path.dirname(os.path.abspath(__file__))
_sam_repo = os.path.join(_sgnav_root, "segment_anything")
for _p in (_sam_repo, _sgnav_root):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, _sgnav_root)
if os.path.isdir(_sam_repo):
    sys.path.insert(0, _sam_repo)

from pathlib import Path
cwd = Path(__file__).resolve().parent  # sgnav_agent 根目录，勿依赖启动 cwd

from GLIP.maskrcnn_benchmark.config import cfg as glip_cfg
from GLIP.maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

from pslpython.model import Model as PSLModel
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

from scenegraph import SceneGraph

import utils.utils_fmm.control_helper as CH
import utils.utils_fmm.pose_utils as pu
from utils.utils_fmm.fmm_planner import FMMPlanner    
from utils.utils_fmm.mapping import Semantic_Mapping

from utils.utils_glip import *
from utils.utils_glip_openworld import *

from utils.image_process import (
    add_resized_image,
    add_rectangle,
    add_text,
    add_text_list,
    crop_around_point,
    draw_agent,
    draw_goal,
    line_list
)


fov = 90
# fov = 79


def print_scenegraph(scenegraph):
    # 打印结点
    print("===== Nodes =====")
    for i, node in enumerate(scenegraph.get_nodes()):
        print(f"[Node {i}]")
        print(f"  caption : {node.caption}")
        print(f"  center  : {node.center}")
        print(f"  is_goal : {node.is_goal_node}")
        print(f"  room    : {node.room_node.caption if node.room_node is not None else 'None'}")
        print()

    # 打印边
    print("===== Edges =====")
    edges = scenegraph.get_edges()
    for i, edge in enumerate(edges):
        print(f"[Edge {i}]")
        print(f"  {edge.node1.caption} --{edge.relation}--> {edge.node2.caption}")
    # pdb.set_trace()
    print(f"Total nodes: {len(scenegraph.get_nodes())}, total edges: {len(edges)}")


def habitat_to_unreal(action_id):
    """
    输入: Habitat 的动作 id
    输出: [Unreal 动作 ...] ，每个 Unreal 动作为 ((move_x, move_y), look_action, misc_action)
    move_x: -30/30 左右转；move_y: -100/100 后/前
    look_action: 0无, 1抬头, 2低头
    misc_action: 这里保持 0（未用）
    """
    actions = []
    if action_id == 0:  # STOP
        actions.append(((0, 0), 0, 0))
    elif action_id == 1:  # MOVE_FORWARD
        actions.append(((0, 50), 0, 0))
    elif action_id == 2:  # TURN_LEFT
        actions.append(((-30, 0), 0, 0))
    elif action_id == 3:  # TURN_RIGHT
        actions.append(((30, 0), 0, 0))
    elif action_id == 4:  # LOOK_DOWN_60
        actions.append(((0, 0), 1, 0))
    elif action_id == 5:  # LOOK_DOWN_30
        actions.append(((0, 0), 2, 0))
    elif action_id == 6:
        actions.append(((60, 0), 0, 0))
    elif action_id == 7:  # LOOK_DOWN_60 and TURN_RIGHT
        actions.append(((60, 0), 1, 0))
    elif action_id == 8:  # LOOK_DOWN and TURN_RIGHT
        actions.append(((60, 0), 2, 0))
    elif action_id == 9:  # LOOK LEVEL
        actions.append(((0, 0), 0, 0))
    else:
        raise ValueError(f"未知 Habitat 动作 id: {action_id}")
    return actions


def deg_to_rad(deg):
    """
    将角度从 [-180, 180] 映射到 [-π, π]
    """
    return -deg * math.pi / 180.0


def unreal_to_habitat_obs(sg_obs, yaw_index_in_compass=1):
    # print('before: gps:', sg_obs['gps'], 'compass:', sg_obs['compass'])
    pos3 = np.asarray(sg_obs['gps'], dtype=np.float32)
    gps_xy = pos3[:2].copy()

    angles3 = np.asarray(sg_obs['compass'], dtype=np.float32)
    yaw_deg = float(angles3[yaw_index_in_compass])
    compass = deg_to_rad(yaw_deg)
    compass = np.asarray([compass],  dtype=np.float32)

    return {
        'rgb': sg_obs['rgb'],
        'depth': sg_obs['depth'],
        'gps': gps_xy,
        'compass': compass,
    }


class SG_Nav_Agent():
    def __init__(self, agent_id, task_config, args=None):
        self._POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS
        self.config = task_config
        self.args = args
        self.panoramic = []
        self.panoramic_depth = []
        self.turn_angles = 0
        self.device = (
            torch.device("cuda:{}".format(0))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.prev_action = 0
        self.navigate_steps = 0
        self.move_steps = 0
        self.total_steps = 0
        self.found_goal = False
        self.found_goal_times = 0
        # self.distance_threshold = 5
        self.distance_threshold = 20
        self.correct_room = False
        self.changing_room = False
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.force_change_room = False
        self.current_room_search_step = 0
        self.target_room = ''
        self.current_rooms = []
        self.nav_without_goal_step = 0
        self.former_collide = 0
        self.history_pose = []
        self.visualize_image_list = []
        self.count_episodes = -1
        self.loop_time = 0
        self.last_segment_num = 0
        self.goal_merge_threshold = 0.8
        self.rooms = rooms
        self.rooms_captions = rooms_captions
        self.split = (self.args.split_l >= 0)
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}

        ### ------ init glip model ------ ###
        # config_file = "GLIP/configs/pretrain/glip_Swin_L.yaml" 
        # weight_file = "GLIP/MODEL/glip_large_model.pth"
        # config_file = f"{cwd}/sgnav_agent/GLIP/configs/pretrain/glip_Swin_L.yaml"
        # weight_file = f"{cwd}/sgnav_agent/GLIP/MODEL/glip_large_model.pth"
        config_file = f"{cwd}/GLIP/configs/pretrain/glip_Swin_L.yaml"
        weight_file = f"{cwd}/GLIP/MODEL/glip_large_model.pth"
        glip_cfg.local_rank = 0
        glip_cfg.num_gpus = 1
        glip_cfg.merge_from_file(config_file) 
        glip_cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
        glip_cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
        self.glip_demo = GLIPDemo(
            glip_cfg,
            min_image_size=800,
            # confidence_threshold=0.61,
            confidence_threshold=0.4,

            show_mask_heatmaps=False
        )

        # self.map_size_cm = 4000
        self.map_size_cm = 12000
        self.map_resolution = int(getattr(self.args, "map_resolution_cm", 5))
        if self.map_resolution <= 0:
            self.map_resolution = 5
        self.resolution = self.map_resolution
        self.camera_horizon = 0
        self.dilation_deg = 0
        self.collision_threshold = 0.08
        self.selem = skimage.morphology.square(1)
        self.explanation = ''
        
        self.init_map()
        self.sem_map_module = Semantic_Mapping(self).to(self.device) 
        self.free_map_module = Semantic_Mapping(self, max_height=10,min_height=-150).to(self.device)
        self.room_map_module = Semantic_Mapping(self, max_height=200,min_height=-10, num_cats=9).to(self.device)
        
        self.free_map_module.eval()
        self.free_map_module.set_view_angles(self.camera_horizon)
        self.sem_map_module.eval()
        self.sem_map_module.set_view_angles(self.camera_horizon)
        self.room_map_module.eval()
        self.room_map_module.set_view_angles(self.camera_horizon)

        self.camera_matrix = self.free_map_module.camera_matrix
        
        self.openworld = self.args.openworld
        
        self.goal_idx = {}
        if self.openworld:
            for key in projection_openworld:
                self.goal_idx[projection_openworld[key]] = categories_21_openworld.index(projection_openworld[key])
        else:
            for key in projection:
                self.goal_idx[projection[key]] = categories_21.index(projection[key])
        
        # self.co_occur_mtx = np.load('tools/obj.npy')
        # self.co_occur_mtx = np.load(f"{cwd}/sgnav_agent/tools/obj.npy")
        self.co_occur_mtx = np.load(f"{cwd}/tools/obj.npy")

        self.co_occur_mtx -= self.co_occur_mtx.min()
        self.co_occur_mtx /= self.co_occur_mtx.max() 
        
        # self.co_occur_room_mtx = np.load('tools/room.npy')
        # self.co_occur_room_mtx = np.load(f"{cwd}/sgnav_agent/tools/room.npy")
        self.co_occur_room_mtx = np.load(f"{cwd}/tools/room.npy")
        self.co_occur_room_mtx -= self.co_occur_room_mtx.min()
        self.co_occur_room_mtx /= self.co_occur_room_mtx.max()
        
        self.scenegraph = SceneGraph(map_resolution=self.map_resolution, map_size_cm=self.map_size_cm, map_size=self.map_size, camera_matrix=self.camera_matrix, agent=self)

        self.experiment_name = self.args.experiment_name
        self.agent_id = agent_id
        # 多机协同（可选）：由 multi.MultiSGNavCoordinator 写入；单机默认 False，不影响原逻辑
        self.coop_use_external_frontier = False
        self.coop_frontier_cell = None

        if self.split:
            self.experiment_name = self.experiment_name + f'/{self.args.split_l}_{self.args.split_r}'

        self.visualization_dir = f'data/visualization/{self.experiment_name}/{self.agent_id}'
        self._rescue_last_seen_cx = None
        self._rescue_lost_view_steps = 0
        self._rescue_scan_dir = 2

        print('scene graph module init finish!!!')

    def add_predicates(self, model):
        predicate = Predicate('IsNearObj', closed = True, size = 2)
        model.add_predicate(predicate)
        predicate = Predicate('ObjCooccur', closed = True, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('IsNearRoom', closed = True, size = 2)
        model.add_predicate(predicate)
        predicate = Predicate('RoomCooccur', closed = True, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('Choose', closed = False, size = 1)
        model.add_predicate(predicate)
        predicate = Predicate('ShortDist', closed = True, size = 1)
        model.add_predicate(predicate)
        
    def add_rules(self, model):
        model.add_rule(Rule('2: ObjCooccur(O) & IsNearObj(O,F)  -> Choose(F)^2'))
        model.add_rule(Rule('2: !ObjCooccur(O) & IsNearObj(O,F) -> !Choose(F)^2'))
        model.add_rule(Rule('2: RoomCooccur(R) & IsNearRoom(R,F) -> Choose(F)^2'))
        model.add_rule(Rule('2: !RoomCooccur(R) & IsNearRoom(R,F) -> !Choose(F)^2'))
        model.add_rule(Rule('2: ShortDist(F) -> Choose(F)^2'))
        model.add_rule(Rule('Choose(+F) = 1 .'))
    
    def reset(self):
        self.navigate_steps = 0
        self.turn_angles = 0
        self.move_steps = 0
        self.total_steps = 0
        self.current_room_search_step = 0
        self.found_goal = False
        self.found_goal_times = 0
        self.correct_room = False
        self.changing_room = False
        self.goal_loc = None
        self.changing_room_steps = 0
        self.move_after_new_goal = False
        self.former_check_step = -10
        self.goal_disappear_step = 100
        self.prev_action = 0
        self.former_collide = 0
        self.goal_gps = np.array([0.,0.])
        self.possible_goal_temp_gps = np.array([0.,0.])
        self.last_gps = np.array([11100.,11100.])
        self.has_panarama = False
        self.init_map()
        self.last_loc = self.full_pose
        self.panoramic = []
        self.panoramic_depth = []
        self.current_rooms = []
        self.dist_to_frontier_goal = 10
        self.first_fbe = True
        self.goal_map = np.zeros(self.full_map.shape[-2:])
        self.found_possible_goal = False
        self.history_pose = []
        self.visualize_image_list = []
        self.count_episodes = self.count_episodes + 1
        self.loop_time = 0
        self.last_segment_num = 0
        self.metrics = {'distance_to_goal': 0., 'spl': 0., 'softspl': 0.}
        # self.obj_goal = self.simulator._env.current_episode.object_category
        # self.obj_goal_sg = self.simulator._env.current_episode.object_category
        
        self.obj_goal = 'tv_monitor'
        if self.openworld:
            self.obj_goal = 'car'
            # self.obj_goal = 'swimming_pool'
        self.obj_goal_sg = self.obj_goal 
        print('obj_goal:', self.obj_goal, 'obj_goal_sg:', self.obj_goal_sg)
        
        if self.obj_goal == 'gym_equipment':
            self.obj_goal_sg = 'treadmill. fitness equipment.'
        elif self.obj_goal == 'chest_of_drawers':
            self.obj_goal_sg = 'drawers'
        elif self.obj_goal == 'tv_monitor':
            self.obj_goal_sg = 'tv'
        self.current_obj_predictions = []
        _n_slots = len(categories_21_openworld) if self.openworld else len(categories_21_origin)
        self.obj_locations = [[] for _ in range(_n_slots)]
        self.not_move_steps = 0
        self.move_since_random = 0
        self.using_random_goal = False
        self.fronter_this_ex = 0
        self.random_this_ex = 0
        self.last_location = np.array([0.,0.])
        self.current_stuck_steps = 0
        self.total_stuck_steps = 0
        self.explanation = ''
        self.text_node = ''
        self.text_edge = ''
        self._rescue_align_cx = None
        self._rescue_align_cy = None
        self._rescue_align_bw = None
        self._rescue_align_bh = None
        self._rescue_align_depth_m = None
        self._rescue_align_score = -1.0
        self._rescue_turn_streak = 0
        self._rescue_last_seen_cx = None
        self._rescue_lost_view_steps = 0
        self._rescue_scan_dir = 2
        self.coop_use_external_frontier = False
        self.coop_frontier_cell = None

        self.scenegraph.reset()
        self.gps0 = None

    def _coop_maybe_apply_frontier_goal(self):
        """多机分支：用协调器下发的栅格前沿替代本机 FBE 目标；仅当未锁定语义目标时生效。"""
        if self.found_goal or self.found_possible_goal:
            return
        if not getattr(self, "coop_use_external_frontier", False):
            return
        cell = getattr(self, "coop_frontier_cell", None)
        if cell is None:
            return
        r, c = int(cell[0]), int(cell[1])
        r = max(0, min(self.map_size - 1, r))
        c = max(0, min(self.map_size - 1, c))
        self.not_use_random_goal()
        self.using_random_goal = False
        self.goal_map = np.zeros(self.full_map.shape[-2:])
        self.goal_map[r, c] = 1
        self.goal_map = self.goal_map[::-1]
        
    def detect_objects(self, observations):
        if self.openworld:
            self.current_obj_predictions = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], object_captions_openworld) # GLIP object detection, time cosuming
        else:    
            self.current_obj_predictions = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], object_captions) # GLIP object detection, time cosuming
        new_labels = self.get_glip_real_label(self.current_obj_predictions) # transfer int labels to string labels
        self.current_obj_predictions.add_field("labels", new_labels)
        if getattr(self.args, "print_glip_labels", True):
            _scores = self.current_obj_predictions.get_field("scores")
            _s = [round(float(x), 3) for x in _scores.tolist()]
            print(f"[GLIP] labels={new_labels}  scores={_s}  goal={self.obj_goal}")

        shortest_distance = 120
        shortest_distance_angle = 0
        goal_prediction = copy.deepcopy(self.current_obj_predictions)
        obj_labels = self.current_obj_predictions.get_field("labels")
        goal_bbox = []
        if self.obj_goal in ('person', 'car'):
            self._rescue_align_cx = None
            self._rescue_align_cy = None
            self._rescue_align_bw = None
            self._rescue_align_bh = None
            self._rescue_align_depth_m = None
            self._rescue_align_score = -1.0
        for j, label in enumerate(obj_labels):
            if self.obj_goal in label:
                goal_bbox.append(self.current_obj_predictions.bbox[j])
                if self.obj_goal in ('person', 'car'):
                    sc = float(self.current_obj_predictions.get_field("scores")[j].item())
                    bb = self.current_obj_predictions.bbox[j]
                    cx = (bb[0] + bb[2]) * 0.5
                    cy = (bb[1] + bb[3]) * 0.5
                    cx = float(cx.item() if hasattr(cx, "item") else cx)
                    cy = float(cy.item() if hasattr(cy, "item") else cy)
                    if sc > self._rescue_align_score:
                        self._rescue_align_score = sc
                        self._rescue_align_cx = cx
                        self._rescue_align_cy = cy
                        bb_i = bb.to(torch.int64)
                        self._rescue_align_bw = float(bb_i[2] - bb_i[0])
                        self._rescue_align_bh = float(bb_i[3] - bb_i[1])
                        self._rescue_align_depth_m = self._robust_depth_for_bbox(bb_i)
            elif self.obj_goal == 'gym_equipment' and (label in ['treadmill', 'exercise machine']):
                goal_bbox.append(self.current_obj_predictions.bbox[j])
        
        for j, label in enumerate(obj_labels):
            if self.openworld:
                if label in categories_21_openworld:
                    confidence = self.current_obj_predictions.get_field("scores")[j]
                    bbox = self.current_obj_predictions.bbox[j].to(torch.int64)
                    center_point = (bbox[:2] + bbox[2:]) // 2
                    temp_direction = (center_point[0] - 320) * fov / 640
                    temp_distance = self.depth[center_point[1],center_point[0],0]
                    if temp_distance >= self.distance_threshold:
                        continue
                    obj_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                    # obj_gps = obj_gps - self.gps0
                    x = int(self.map_center_cells - obj_gps[1]*100/self.resolution)
                    y = int(self.map_center_cells + obj_gps[0]*100/self.resolution)
                    self.obj_locations[categories_21_openworld.index(label)].append([confidence, x, y])
            else:
                if label in categories_21_origin:
                    confidence = self.current_obj_predictions.get_field("scores")[j]
                    bbox = self.current_obj_predictions.bbox[j].to(torch.int64)
                    center_point = (bbox[:2] + bbox[2:]) // 2
                    temp_direction = (center_point[0] - 320) * fov / 640
                    temp_distance = self.depth[center_point[1],center_point[0],0]
                    if temp_distance >= self.distance_threshold:
                        continue
                    obj_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                    # obj_gps = obj_gps - self.gps0
                    x = int(self.map_center_cells - obj_gps[1]*100/self.resolution)
                    y = int(self.map_center_cells + obj_gps[0]*100/self.resolution)
                    self.obj_locations[categories_21_origin.index(label)].append([confidence, x, y])
        
        if self.scenegraph.obj_goal in self.scenegraph.small_objects:
            self.segment_num = len(self.scenegraph.segment2d_results)
            goal_mask = []
            if self.segment_num > self.last_segment_num:
                self.last_segment_num = self.segment_num
                segment2d_result = self.scenegraph.segment2d_results[-1]
                indices = []
                for index, element in enumerate(segment2d_result['caption']):
                    if self.obj_goal_sg in element.split(' '):
                        for node in self.scenegraph.nodes:
                            if node.is_goal_node and node.object['image_idx'][-1] == len(self.scenegraph.segment2d_results) - 1 and node.object['mask_idx'][-1] == index:
                                indices.append(index)
                goal_mask = [segment2d_result['mask'][index] for index in indices]
            if len(goal_mask) > 0:
                possible_goal_detected_before = copy.deepcopy(self.found_possible_goal)
                for mask in goal_mask:
                    center_point = torch.tensor(np.argwhere(mask).mean(axis=0).astype(int))
                    center_point = torch.tensor([center_point[1], center_point[0]])
                    temp_direction = (center_point[0] - 320) * fov / 640
                    temp_distance = self.depth[center_point[1],center_point[0],0]
                    k = 0
                    pos_neg = 1
                    while temp_distance >= 100 and 0<center_point[1]+int(pos_neg*k)<479 and 0<center_point[0]+int(pos_neg*k)<639:
                        pos_neg *= -1
                        k += 0.5
                        temp_distance = max(self.depth[center_point[1]+int(pos_neg*k),center_point[0],0],
                        self.depth[center_point[1],center_point[0]+int(pos_neg*k),0])
                        
                    if temp_distance >= self.distance_threshold:
                        self.found_possible_goal = True
                    else:
                        if self.found_goal:
                            if temp_distance < self.distance_threshold:
                                self.found_goal_times = self.found_goal_times + 1
                        self.found_goal = True
                        self.found_possible_goal = False
                    
                    ## select the closest goal
                    direction = temp_direction
                    distance = temp_distance
                    if distance < shortest_distance:
                        shortest_distance = distance
                        shortest_distance_angle = direction
                
                if self.found_goal:
                    self.goal_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
                elif not possible_goal_detected_before:
                    # if detected a long goal before, then don't change it until see a goal within 5 meters
                    self.possible_goal_temp_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)
            else:
                if self.found_goal:
                    self.found_goal = False
                    self.found_goal_times = 0
            return
        else:
            if len(goal_bbox) > 0:
                possible_goal_detected_before = copy.deepcopy(self.found_possible_goal)
                goal_prediction.bbox = torch.stack(goal_bbox)
                for box in goal_prediction.bbox:
                    box = box.to(torch.int64)
                    center_point = (box[:2] + box[2:]) // 2
                    temp_direction = (center_point[0] - 320) * fov / 640
                    if self.obj_goal in ('person', 'car'):
                        temp_distance = self._robust_depth_for_bbox(box)
                    else:
                        temp_distance = float(self.depth[center_point[1], center_point[0], 0])
                    goal_gps = self.get_goal_gps(observations, temp_direction, temp_distance)
                    k = 0
                    pos_neg = 1
                    while temp_distance >= 100 and 0<center_point[1]+int(pos_neg*k)<479 and 0<center_point[0]+int(pos_neg*k)<639:
                        pos_neg *= -1
                        k += 0.5
                        temp_distance = max(self.depth[center_point[1]+int(pos_neg*k),center_point[0],0],
                        self.depth[center_point[1],center_point[0]+int(pos_neg*k),0])
                        
                    if temp_distance >= self.distance_threshold:
                        self.found_possible_goal = True
                    else:
                        thres = int(self.goal_merge_threshold * 100 / self.map_resolution)
                        if 0 <= int(self.map_center_cells + goal_gps[0]*100/self.resolution) < self.map_size and 0 <= int(self.map_center_cells + goal_gps[1]*100/self.resolution) < self.map_size:
                            goal_gps_map_local = self.goal_gps_map[max(int(self.map_center_cells + goal_gps[1]*100/self.resolution) - thres, 0):min(int(self.map_center_cells + goal_gps[1]*100/self.resolution) + thres, self.map_size - 1), max(int(self.map_center_cells + goal_gps[0]*100/self.resolution) - thres, 0):min(int(self.map_center_cells + goal_gps[0]*100/self.resolution) + thres, self.map_size - 1)]
                            if goal_gps_map_local.max() > 0:
                                goal_gps_map_local[np.where(goal_gps_map_local == goal_gps_map_local.max())[0][0], np.where(goal_gps_map_local == goal_gps_map_local.max())[1][0]] = goal_gps_map_local[np.where(goal_gps_map_local == goal_gps_map_local.max())[0][0], np.where(goal_gps_map_local == goal_gps_map_local.max())[1][0]] + 1
                            else:
                                self.goal_gps_map[
                                    min(max(int(self.map_center_cells + goal_gps[1]*100/self.resolution), 0), self.map_size - 1),
                                    min(max(int(self.map_center_cells + goal_gps[0]*100/self.resolution), 0), self.map_size - 1),
                                ] = 1
                        self.found_possible_goal = False
                    
                    direction = temp_direction
                    distance = temp_distance
                    if distance < shortest_distance:
                        shortest_distance = distance
                        shortest_distance_angle = direction
                
                self.found_goal_times = self.goal_gps_map.max()
                if self.found_goal_times >= self.scenegraph.cfg.obj_min_detections:
                    self.found_goal = True

                if self.found_goal:
                    self.goal_gps = np.flip(np.array(np.where(self.goal_gps_map == self.goal_gps_map.max()))[:, 0])
                    self.goal_gps = (self.goal_gps - self.map_center_cells) / 100 * self.resolution
                elif not possible_goal_detected_before:
                    self.possible_goal_temp_gps = self.get_goal_gps(observations, shortest_distance_angle, shortest_distance)

                # 救援 person/car：地图投票易因投影抖动或单格越界长期达不到 obj_min_detections；
                # 在稳健深度已给出合理距离时直接锁定目标，避免「GLIP 一直有 car 但 found_goal 长期 False」。
                _rescue_dmax = max(float(self.distance_threshold), 80.0)
                if (
                    self.obj_goal in ('person', 'car')
                    and len(goal_bbox) > 0
                    and shortest_distance < _rescue_dmax
                ):
                    self.found_goal = True
                    self.found_possible_goal = False
                    self.goal_gps = self.get_goal_gps(
                        observations, shortest_distance_angle, shortest_distance
                    )
            return
                        
    def act(self, unreal_observations):
        observations = unreal_to_habitat_obs(unreal_observations)
        # if self.total_steps >= 500:
        if self.total_steps >= 999:
            return habitat_to_unreal(0), 0
        
        if self.navigate_steps == 0:
            self.gps0 = observations['gps']
        observations['gps'] = observations['gps'] - self.gps0
        
        self.total_steps += 1
        if self.navigate_steps == 0:
            _n_obj = len(categories_21_openworld) if self.openworld else len(categories_21_origin)
            if self.openworld:
                self.prob_array_room = np.zeros(9)
                self.prob_array_obj = np.zeros(_n_obj, dtype=float)
            elif self.obj_goal in ('person', 'car'):
                # 救援目标不在 Matterport 共现矩阵中，行为与 openworld 类似：不用 co_occur 行
                self.prob_array_room = np.zeros(9)
                self.prob_array_obj = np.zeros(_n_obj, dtype=float)
            else:
                self.prob_array_room = self.co_occur_room_mtx[self.goal_idx[self.obj_goal]]
                _row = self.co_occur_mtx[self.goal_idx[self.obj_goal]]
                self.prob_array_obj = np.zeros(_n_obj, dtype=float)
                _take = min(_row.shape[0], _n_obj)
                self.prob_array_obj[:_take] = _row[:_take]

        observations["depth"][observations["depth"]==0.5] = 100 # don't construct unprecise map with distance less than 0.5 m
        self.depth = observations["depth"]
        self.rgb = observations["rgb"][:,:,[2,1,0]]
        self.rgb_visualization = observations["rgb"]

        self.scenegraph.set_agent(self)
        self.scenegraph.set_navigate_steps(self.navigate_steps)
        self.scenegraph.set_obj_goal(self.obj_goal, self.obj_goal_sg)
        self.scenegraph.set_room_map(self.room_map)
        self.scenegraph.set_fbe_free_map(self.fbe_free_map)
        self.scenegraph.set_observations(observations)
        self.scenegraph.set_full_map(self.full_map)
        self.scenegraph.set_full_pose(self.full_pose)
        if self.openworld:
            pass
        else:
            self.scenegraph.update_scenegraph()
        
        self.update_map(observations)
        self.update_free_map(observations)
        
        # 启动阶段扫描从 22 步压缩到 5 步，减少长时间原地旋转。
        if self.total_steps == 1:
            self.sem_map_module.set_view_angles(30)
            self.free_map_module.set_view_angles(30)
            return habitat_to_unreal(5), 5
        elif self.total_steps <= 3:
            return habitat_to_unreal(8), 8
        elif self.total_steps == 4:
            self.sem_map_module.set_view_angles(60)
            self.free_map_module.set_view_angles(60)
            return habitat_to_unreal(4), 4
        elif self.total_steps == 5:
            self.sem_map_module.set_view_angles(0)
            self.free_map_module.set_view_angles(0)
        if self.total_steps <= 5 and not self.found_goal:
            self.panoramic.append(observations["rgb"][:,:,[2,1,0]])
            self.panoramic_depth.append(observations["depth"])
            self.detect_objects(observations)
            room_detection_result = self.glip_demo.inference(observations["rgb"][:,:,[2,1,0]], rooms_captions)
            self.update_room_map(observations, room_detection_result)
            if not self.found_goal: # if found a goal, directly go to it
                return habitat_to_unreal(6), 6
        elif (
            self.total_steps <= 5
            and self.found_goal
            and self.obj_goal in ("person", "car")
            and getattr(self.args, "rescue_visual_align", True)
        ):
            # 已 found_goal 时原分支不再进，若不补跑 GLIP，_rescue_align_cx 会永远停在「锁定目标」那一帧，
            # 视觉对齐会一直同一转向（例如恒为 3）而目标在画面中的位置已随视角改变。
            self.detect_objects(observations)

        # 仅 total_steps<=5 时上面分支会调用 detect_objects；之后 GLIP 不再运行，晚进入视野的目标永远检不出。
        # 这里按间隔补跑检测（代价：GLIP 耗时），避免一直 frontier 导航+原地转向却“看不见”目标。
        # found_goal 后救援对齐仍依赖最新框中心，须按同一间隔刷新，否则会 Phase1 原地打转超时。
        _interval = int(getattr(self.args, "goal_detect_interval", 5))
        _interval = max(1, _interval)
        _rescue_align_refresh = (
            self.found_goal
            and self.obj_goal in ("person", "car")
            and getattr(self.args, "rescue_visual_align", True)
        )
        if self.total_steps > 5 and (not self.found_goal or _rescue_align_refresh):
            if (self.total_steps - 6) % _interval == 0:
                self.detect_objects(observations)

        # if (self.total_steps - 6) % 4 == 1:
        #     self.sem_map_module.set_view_angles(60)
        #     self.free_map_module.set_view_angles(60)
        #     return habitat_to_unreal(4), 4
        # elif (self.total_steps - 6) % 4 == 2:
        #     self.sem_map_module.set_view_angles(30)
        #     self.free_map_module.set_view_angles(30)
        #     return habitat_to_unreal(5), 5   
        # elif (self.total_steps - 6) % 4 == 3:
        #     self.sem_map_module.set_view_angles(0)
        #     self.free_map_module.set_view_angles(0)
        #     return habitat_to_unreal(9), 9
        

        if np.linalg.norm(observations["gps"] - self.last_gps) >= 0.05:
            self.move_steps += 1
            self.not_move_steps = 0
            if self.using_random_goal:
                self.move_since_random += 1
        else:
            self.not_move_steps += 1
            
        self.last_gps = observations["gps"]
        
        self.scenegraph.perception()
          
        self.history_pose.append(self.full_pose.cpu().detach().clone())
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        traversible, cur_start, cur_start_o = self.get_traversible(self.full_map.cpu().numpy()[0,0,::-1], input_pose)
        
        if self.found_goal: 
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.goal_map[max(0,min(self.map_size - 1,int(self.map_center_cells + self.goal_gps[1]*100/self.resolution))), max(0,min(self.map_size - 1,int(self.map_center_cells + self.goal_gps[0]*100/self.resolution)))] = 1
        elif self.found_possible_goal: 
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            self.goal_map[max(0,min(self.map_size - 1,int(self.map_center_cells + self.possible_goal_temp_gps[1]*100/self.resolution))), max(0,min(self.map_size - 1,int(self.map_center_cells + self.possible_goal_temp_gps[0]*100/self.resolution)))] = 1
        elif self.first_fbe:
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.first_fbe = False
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None:
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
            else:
                self.fronter_this_ex += 1
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1
                self.goal_map = self.goal_map[::-1]
        
        self._coop_maybe_apply_frontier_goal()
        # local policy
        stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        if self.found_possible_goal and number_action == 0:
            self.found_possible_goal = False
        
        # reach long-term goal and fbe
        if (not self.found_goal and not self.found_possible_goal and number_action == 0) or (self.using_random_goal and self.move_since_random > 20): 
            if (self.using_random_goal and self.move_since_random > 20):
                goal_x, goal_y = np.where(self.goal_map == 1)
                x_0 = max(goal_x[0] - 8, 0)
                y_0 = max(goal_y[0] - 8, 0)
                x_1 = min(goal_x[0] + 8, self.map_size)
                y_1 = min(goal_y[0] + 8, self.map_size)
                self.fbe_free_map[x_0:x_1, y_0:y_1] = 0
            self.goal_loc = self.fbe(traversible, cur_start)
            self.not_use_random_goal()
            self.goal_map = np.zeros(self.full_map.shape[-2:])
            if self.goal_loc is None:
                self.random_this_ex += 1
                self.goal_map = self.set_random_goal()
                self.using_random_goal = True
            else:
                self.fronter_this_ex += 1
                self.goal_map[self.goal_loc[0], self.goal_loc[1]] = 1
                self.goal_map = self.goal_map[::-1]
            self._coop_maybe_apply_frontier_goal()
            stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        
        self.loop_time = 0
        # 原条件里 not_move_steps>=7 会在「已 found_goal」时仍进入循环并清空目标。
        # 常见情况：FMM 判定已到达(stop→action0)或原地转向对齐航向时 GPS 位移<阈值，
        # not_move_steps 连涨 7 步后误删 person/car 目标，表现为「检测到目标却不持续朝目标走」。
        while (not self.found_goal and number_action == 0) or (
            self.not_move_steps >= 7 and not self.found_goal
        ):
            if self.not_move_steps >= 7:
                self.found_goal = False
                self.found_possible_goal = False
            self.loop_time += 1
            self.random_this_ex += 1
            if self.loop_time > 20:
                return habitat_to_unreal(0), 0
            self.not_move_steps = 0
            self.goal_map = self.set_random_goal()
            self.using_random_goal = True
            self._coop_maybe_apply_frontier_goal()
            stg_y, stg_x, replan, number_action = self._plan(traversible, self.goal_map, self.full_pose, cur_start, cur_start_o, self.found_goal)
        
        # 救援 person/car：FMM 短期航点在噪声地图上易「原地打转」。在 GLIP 仍给出目标框时用画面中心对齐 + 直行。
        if (
            getattr(self.args, "rescue_visual_align", True)
            and self.found_goal
            and self.obj_goal in ('person', 'car')
            and self.rgb_visualization is not None
            and (self._rescue_align_cx is not None or self._has_goal_node_memory())
        ):
            w = int(self.rgb_visualization.shape[1])
            h = int(self.rgb_visualization.shape[0])
            mid = w * 0.5
            band = max(28.0, w * 0.055)
            if self.obj_goal == 'car':
                band = max(band, w * 0.11)
            cx = self._rescue_align_cx
            bb_w = self._rescue_align_bw
            bb_h = self._rescue_align_bh
            dm = self._rescue_align_depth_m
            # 按框的像素宽度 + 框内稳健深度微调带宽：大目标/近距在画面上更宽，应对齐更松；远距略放宽减少左右抖
            if getattr(self.args, "rescue_adaptive_band", True) and bb_w is not None and bb_w > 2.0:
                beta = float(getattr(self.args, "rescue_band_beta_w", 0.22))
                if self.obj_goal == 'car':
                    beta = float(getattr(self.args, "rescue_band_beta_w_car", max(beta, 0.26)))
                band = max(band, beta * bb_w)
            if (
                getattr(self.args, "rescue_adaptive_band", True)
                and dm is not None
                and dm > 0.5
                and dm < 95.0
            ):
                d_ref = float(getattr(self.args, "rescue_band_depth_ref_m", 4.0))
                scale = (dm / d_ref) ** 0.28
                scale = float(min(1.22, max(0.86, scale)))
                band *= scale
            band = float(min(band, 0.48 * w))
            # person/car 在画面下方时常表示还需靠近才能触发 pick/place；优先前进。car 车身宽，仅横向对准易长期 2/3 导致 GPS 不动。
            y_thr = float(getattr(self.args, "rescue_pick_approach_y_frac", 0.56))
            y_thr = min(0.92, max(0.35, y_thr))
            edge = float(getattr(self.args, "rescue_pick_lateral_edge_frac", 0.18))
            edge = min(0.28, max(0.08, edge))
            if getattr(self.args, "rescue_adaptive_band", True) and bb_w is not None and bb_w > 2.0:
                edge = max(edge, min(0.28, 0.10 + 0.40 * (bb_w / float(w))))
            # 目标仍较远时少打转、多逼近：放宽「对准」带、收窄左右必须转向的窄条，提高前进优先级
            d_pri = float(getattr(self.args, "rescue_forward_priority_depth_m", 2.0))
            if (
                getattr(self.args, "rescue_depth_forward_priority", True)
                and dm is not None
                and dm > d_pri
                and dm < 95.0
            ):
                band *= float(getattr(self.args, "rescue_far_band_mult", 1.55))
                band = float(min(band, 0.48 * w))
                edge = min(edge, float(getattr(self.args, "rescue_far_edge_max", 0.10)))
            lower_approach = (
                self._rescue_align_cy is not None
                and getattr(self.args, "rescue_pick_approach", True)
                and self._rescue_align_cy >= h * y_thr
            )
            # 仅在 SGNav 本轮局部规划返回 stop 后，才启用「2m 外不允许停」约束。
            stop_then_constrain = (number_action == 0 and self.obj_goal == 'car')
            keep_stop = False
            est_dist_m = None
            near_m = float(getattr(self.args, "rescue_car_stop_near_m", 2.0))
            if stop_then_constrain:
                if dm is not None and 0.05 < float(dm) < 95.0:
                    est_dist_m = float(dm)
                else:
                    try:
                        est_dist_m = float(np.linalg.norm(self.goal_gps - observations["gps"]))
                    except Exception:
                        est_dist_m = None
                if est_dist_m is not None and est_dist_m <= near_m:
                    keep_stop = True
            goal_in_graph = self._has_goal_node_memory()
            aggressive_push = bool(getattr(self.args, "rescue_aggressive_push", True)) and (
                self._rescue_align_cx is not None or goal_in_graph
            )
            if keep_stop:
                number_action = 0
                self._rescue_turn_streak = 0
            elif aggressive_push or stop_then_constrain:
                if self._rescue_align_cx is None:
                    # 朝目标前进时若目标短暂丢失，先左右扫视几步再继续前进。
                    lost_after_forward = self.prev_action == 1
                    if lost_after_forward or self._rescue_lost_view_steps > 0:
                        self._rescue_lost_view_steps += 1
                        max_scan_steps = int(getattr(self.args, "rescue_lost_scan_turn_steps", 2))
                        max_scan_steps = max(1, min(6, max_scan_steps))
                        if self._rescue_lost_view_steps <= max_scan_steps:
                            if self._rescue_last_seen_cx is not None:
                                number_action = 2 if self._rescue_last_seen_cx < mid else 3
                            else:
                                number_action = self._rescue_scan_dir
                                self._rescue_scan_dir = 3 if self._rescue_scan_dir == 2 else 2
                        else:
                            number_action = 1
                    else:
                        number_action = 1
                else:
                    self._rescue_last_seen_cx = float(cx)
                    self._rescue_lost_view_steps = 0
                    aggressive_edge = float(getattr(self.args, "rescue_aggressive_edge_frac", 0.10))
                    aggressive_edge = min(0.20, max(0.03, aggressive_edge))
                    if cx < w * aggressive_edge:
                        number_action = 2
                    elif cx > w * (1.0 - aggressive_edge):
                        number_action = 3
                    else:
                        number_action = 1
                self._rescue_turn_streak = 0
            else:
                if lower_approach:
                    if cx < w * edge:
                        number_action = 2
                    elif cx > w * (1.0 - edge):
                        number_action = 3
                    else:
                        number_action = 1
                elif cx < mid - band:
                    number_action = 2
                elif cx > mid + band:
                    number_action = 3
                else:
                    number_action = 1
                _mx = int(getattr(self.args, "rescue_align_max_turn_streak", 5))
                if _mx > 0:
                    if number_action in (2, 3):
                        self._rescue_turn_streak += 1
                        if self._rescue_turn_streak >= _mx:
                            number_action = 1
                            self._rescue_turn_streak = 0
                    else:
                        self._rescue_turn_streak = 0
            if stop_then_constrain:
                print(
                    "[RescueStopConstraint] "
                    f"step={self.total_steps} "
                    f"dm={None if dm is None else round(float(dm), 3)} "
                    f"est_dist_m={None if est_dist_m is None else round(float(est_dist_m), 3)} "
                    f"near_m={round(float(near_m), 3)} "
                    f"keep_stop={int(keep_stop)} "
                    f"final_action={number_action} "
                    f"align_cx={None if self._rescue_align_cx is None else round(float(self._rescue_align_cx), 1)}"
                )

        if self.args.visualize:
            self.visualize(traversible, observations, number_action)

        observations["pointgoal_with_gps_compass"] = self.get_relative_goal_gps(observations)

        self.last_loc = copy.deepcopy(self.full_pose)
        self.prev_action = number_action
        self.navigate_steps += 1
        torch.cuda.empty_cache()
        
        print('&'*10, self.total_steps, self.navigate_steps, observations["gps"], self.found_goal, self.found_possible_goal, self.using_random_goal, number_action)
        if self.navigate_steps % 10 == 0:
            print_scenegraph(self.scenegraph)
        
        return habitat_to_unreal(number_action), number_action
    
    def not_use_random_goal(self):
        self.move_since_random = 0
        self.using_random_goal = False

    def _has_goal_node_memory(self) -> bool:
        """场景图中已存在目标节点（即便当前帧框不稳定）."""
        try:
            for node in getattr(self.scenegraph, "nodes", []):
                if getattr(node, "is_goal_node", False):
                    return True
        except Exception:
            return False
        return False
        
    def get_glip_real_label(self, prediction):
        labels = prediction.get_field("labels").tolist()
        new_labels = []
        if self.glip_demo.entities and self.glip_demo.plus:
            for i in labels:
                if i <= len(self.glip_demo.entities):
                    new_labels.append(self.glip_demo.entities[i - self.glip_demo.plus])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for i in labels]
        return new_labels
    
    def fbe(self, traversible, start):
        fbe_map = torch.zeros_like(self.full_map[0,0])
        fbe_map[self.fbe_free_map[0,0]>0] = 1 # first free 
        fbe_map[skimage.morphology.binary_dilation(self.full_map[0,0].cpu().numpy(), skimage.morphology.disk(4))] = 3 # then dialte obstacle

        fbe_cp = copy.deepcopy(fbe_map)
        fbe_cpp = copy.deepcopy(fbe_map)
        fbe_cp[fbe_cp==0] = 4 # don't know space is 4
        fbe_cp[fbe_cp<4] = 0 # free and obstacle
        selem = skimage.morphology.disk(1)
        fbe_cpp[skimage.morphology.binary_dilation(fbe_cp.cpu().numpy(), selem)] = 0 # don't know space is 0 dialate unknown space
        
        diff = fbe_map - fbe_cpp # intersection between unknown area and free area 
        frontier_map = diff == 1
        frontier_locations = torch.stack([torch.where(frontier_map)[0], torch.where(frontier_map)[1]]).T
        num_frontiers = len(torch.where(frontier_map)[0])
        if num_frontiers == 0:
            return None
        
        # for each frontier, calculate the inverse of distance
        planner = FMMPlanner(traversible, None)
        state = [start[0] + 1, start[1] + 1]
        planner.set_goal(state)
        fmm_dist = planner.fmm_dist[::-1]
        frontier_locations += 1
        frontier_locations = frontier_locations.cpu().numpy()
        distances = fmm_dist[frontier_locations[:,0],frontier_locations[:,1]] / 20
        
        ## use the threshold of 1.6 to filter close frontiers to encourage exploration
        idx_16 = np.where(distances>=1.6)
        distances_16 = distances[idx_16]
        distances_16_inverse = 1 - (np.clip(distances_16,0,11.6)-1.6) / (11.6-1.6)
        frontier_locations_16 = frontier_locations[idx_16]
        self.frontier_locations = frontier_locations
        self.frontier_locations_16 = frontier_locations_16
        if len(distances_16) == 0:
            return None
        num_16_frontiers = len(idx_16[0])  # 175

        if self.openworld:
            scores = np.zeros((num_16_frontiers))
        else:
            scores = self.scenegraph.score(frontier_locations_16, num_16_frontiers)
        print('scores:', len(np.unique(scores)))
        scores += 2 * distances_16_inverse
        idx_16_max = idx_16[0][np.argmax(scores)]
        goal = frontier_locations[idx_16_max] - 1
        self.scores = scores
        return goal

    def _robust_depth_for_bbox(self, bbox_tlbr):
        """框内最小有效深度，排除中心点在玻璃/天空上的伪远距（如 0.5→100 的填充值）。"""
        box = bbox_tlbr
        if hasattr(box, "detach"):
            box = box.detach().cpu().numpy()
        box = np.asarray(box, dtype=np.int64).reshape(-1)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        d = self.depth
        if d.ndim == 3:
            d = np.asarray(d[:, :, 0], dtype=np.float64)
        else:
            d = np.asarray(d, dtype=np.float64)
        h, w = d.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        patch = d[y1 : y2 + 1, x1 : x2 + 1].ravel()
        if patch.size == 0:
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
            return float(d[cy, cx])
        # 与 act 里 depth==0.5→100 一致：>90 视为无效/天空
        valid = patch[(patch > 1e-3) & (patch < 90.0)]
        if valid.size == 0:
            return float(np.min(patch))
        return float(np.min(valid))
        
    def get_goal_gps(self, observations, angle, distance):
        if type(angle) is torch.Tensor:
            angle = angle.cpu().numpy()
        agent_gps = observations['gps']
        agent_compass = observations['compass']
        goal_direction = agent_compass - angle/180*np.pi
        goal_gps = np.array([(agent_gps[0]+np.cos(goal_direction)*distance).item(),
         (agent_gps[1]-np.sin(goal_direction)*distance).item()])
        return goal_gps

    def get_relative_goal_gps(self, observations, goal_gps=None):
        if goal_gps is None:
            goal_gps = self.goal_gps
        direction_vector = goal_gps - np.array([observations['gps'][0].item(),observations['gps'][1].item()])
        rho = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
        phi_world = np.arctan2(direction_vector[1], direction_vector[0])
        agent_compass = observations['compass']
        phi = phi_world - agent_compass
        return np.array([rho, phi.item()], dtype=np.float32)
   
    def init_map(self):
        self.map_size = self.map_size_cm // self.map_resolution
        self.map_center_cells = self.map_size / 2.0
        self.map_center_idx = int(round(self.map_center_cells))
        full_w, full_h = self.map_size, self.map_size
        self.full_map = torch.zeros(1,1 ,full_w, full_h).float().to(self.device)
        self.room_map = torch.zeros(1,9 ,full_w, full_h).float().to(self.device)
        self.visited = self.full_map[0,0].cpu().numpy()
        self.collision_map = self.full_map[0,0].cpu().numpy()
        self.fbe_free_map = copy.deepcopy(self.full_map).to(self.device) # 0 is unknown, 1 is free
        self.full_pose = torch.zeros(3).float().to(self.device)
        self.goal_gps_map = self.full_map[0,0].cpu().numpy()
        self.origins = np.zeros((2))
        
        def init_map_and_pose():
            self.full_map.fill_(0.)
            self.full_pose.fill_(0.)
            self.full_pose[:2] = self.map_size_cm / 100.0 / 2.0  # put the agent in the middle of the map

        init_map_and_pose()

    def update_map(self, observations):
        gps_np = observations['gps']
        # rel_gps = gps_np - self.gps0

        # print(observations['gps'], rel_gps, observations['compass'])
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(gps_np).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(gps_np).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.full_map = self.sem_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.full_map)
    
    def update_free_map(self, observations):
        gps_np = observations['gps']
        # rel_gps = gps_np - self.gps0
        
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+torch.from_numpy(gps_np).to(self.device)[0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-torch.from_numpy(gps_np).to(self.device)[1]
        self.full_pose[2:] = torch.from_numpy(observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.fbe_free_map = self.free_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.fbe_free_map)
        self.fbe_free_map[self.map_center_idx - 3:self.map_center_idx + 4, self.map_center_idx - 3:self.map_center_idx + 4] = 1
    
    def update_room_map(self, observations, room_prediction_result):
        new_room_labels = self.get_glip_real_label(room_prediction_result)
        type_mask = np.zeros((9,self.config.SIMULATOR.DEPTH_SENSOR.HEIGHT, self.config.SIMULATOR.DEPTH_SENSOR.WIDTH))
        bboxs = room_prediction_result.bbox
        score_vec = torch.zeros((9)).to(self.device)
        for i, box in enumerate(bboxs):
            box = box.to(torch.int64)
            idx = rooms.index(new_room_labels[i])
            type_mask[idx,box[1]:box[3],box[0]:box[2]] = 1
            score_vec[idx] = room_prediction_result.get_field("scores")[i]
        self.room_map = self.room_map_module(torch.squeeze(torch.from_numpy(observations['depth']), dim=-1).to(self.device), self.full_pose, self.room_map, torch.from_numpy(type_mask).to(self.device).type(torch.float32), score_vec)
    
    def get_traversible(self, map_pred, pose_pred):
        grid = np.rint(map_pred)
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]
        r, c = start_y, start_x
        start = [int(r*100/self.map_resolution - gy1),
                 int(c*100/self.map_resolution - gx1)]
        start = pu.threshold_poses(start, grid.shape)
        self.visited[gy1:gy2, gx1:gx2][start[0]-2:start[0]+3,
                                       start[1]-2:start[1]+3] = 1
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat
        
        [gx1, gx2, gy1, gy2] = planning_window
        x1, y1, = 0, 0
        x2, y2 = grid.shape

        traversible = skimage.morphology.binary_dilation(
                    grid[y1:y2, x1:x2],
                    self.selem) != True

        if not(traversible[start[0], start[1]]):
            print("Not traversible, step is  ", self.navigate_steps)

        traversible = 1 - traversible
        selem = skimage.morphology.disk(2)
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem)
        traversible[self.collision_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible = skimage.morphology.binary_dilation(
                        traversible, selem) != True
        
        _pad = 1
        if self.found_goal and self.obj_goal in ("person", "car"):
            _pad = 3
        sy, sx = int(start[0] - y1), int(start[1] - x1)
        traversible[
            max(sy - _pad, 0) : min(sy + _pad + 1, traversible.shape[0]),
            max(sx - _pad, 0) : min(sx + _pad + 1, traversible.shape[1]),
        ] = 1
        traversible = traversible * 1.
        
        traversible[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible = add_boundary(traversible)
        return traversible, start, start_o
    
    def _plan(self, traversible, goal_map, agent_pose, start, start_o, goal_found):
        if self.prev_action == 1:
            x1, y1, t1 = self.last_loc.cpu().numpy()
            x2, y2, t2 = self.full_pose.cpu()
            y1 = self.map_size_cm/100 - y1
            y2 = self.map_size_cm/100 - y2
            t1 = -t1
            t2 = -t2
            buf = 4
            length = 5

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            col_threshold = self.collision_threshold

            if dist < col_threshold: # Collision
                self.former_collide += 1
                for i in range(length):
                    wx = x1 + 0.05 * ((i + buf) * np.cos(np.deg2rad(t1)))
                    wy = y1 + 0.05 * ((i + buf) * np.sin(np.deg2rad(t1)))
                    r, c = wy, wx
                    r = int(round(r * 100 / self.map_resolution))
                    c = int(round(c * 100 / self.map_resolution))
                    [r, c] = pu.threshold_poses([r, c], self.collision_map.shape)
                    self.collision_map[r,c] = 1
            else:
                self.former_collide = 0

        stg, replan, stop, = self._get_stg(traversible, start, np.copy(goal_map), goal_found)

        # Deterministic Local Policy
        if stop:
            action = 0
            (stg_y, stg_x) = stg

        else:
            (stg_y, stg_x) = stg
            angle_st_goal = math.degrees(math.atan2(stg_y - start[0],
                                                stg_x - start[1]))
            angle_agent = (start_o)%360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_st_goal- angle_agent)%360.0
            if relative_angle > 180:
                relative_angle -= 360
            if self.former_collide < 10:
                if relative_angle > 16:
                # if relative_angle > 30:
                    action = 3 # Right
                elif relative_angle < -16:
                # elif relative_angle < -30:
                    action = 2 # Left
                else:
                    action = 1
            elif self.prev_action == 1:
                if relative_angle > 0:
                    action = 3 # Right
                else:
                    action = 2 # Left
            else:
                action = 1
            if self.former_collide >= 10 and self.prev_action != 1:
                self.former_collide  = 0
            if stg_y == start[0] and stg_x == start[1]:
                action = 1
            # print('relative_angle:', relative_angle, angle_st_goal, angle_agent, 'att:', (stg_y, stg_x), start, action)
        return stg_y, stg_x, replan, action
    
    def _get_stg(self, traversible, start, goal, goal_found):
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat

        def _fallback_goal_from_traversible(traversible_map, state_rc):
            """当 goal 退化为空/不可用时，回退到离当前位置最近的可通行栅格。"""
            g = np.zeros_like(goal)
            try:
                free = np.argwhere(np.asarray(traversible_map) > 0.5)
                if free.size > 0:
                    s = np.array(state_rc, dtype=np.int64).reshape(1, 2)
                    d2 = np.sum((free - s) ** 2, axis=1)
                    rr, cc = free[int(np.argmin(d2))]
                    g[int(rr), int(cc)] = 1
                    return g
            except Exception:
                pass
            rr = int(np.clip(state_rc[0], 0, g.shape[0] - 1))
            cc = int(np.clip(state_rc[1], 0, g.shape[1] - 1))
            g[rr, cc] = 1
            return g
        
        goal = add_boundary(goal, value=0)
        original_goal = copy.deepcopy(goal)
        
        centers = []
        if len(np.where(goal !=0)[0]) > 1:
            goal, centers = CH._get_center_goal(goal)
        state = [start[0] + 1, start[1] + 1]
        self.planner = FMMPlanner(traversible, None)
            
        if self.dilation_deg!=0: 
            goal = CH._add_cross_dilation(goal, self.dilation_deg, 3)
            
        if goal_found:
            try:
                goal = CH._block_goal(centers, goal, original_goal, goal_found)
            except:
                goal = _fallback_goal_from_traversible(traversible, state)

        if np.count_nonzero(goal == 1) == 0:
            goal = _fallback_goal_from_traversible(traversible, state)

        try:
            self.planner.set_multi_goal(goal, state) # time cosuming
        except ValueError as e:
            if "no zero contour" in str(e):
                print(
                    f"[FMMFallback] no zero contour at step={self.total_steps}, "
                    "fallback to nearest traversible goal."
                )
                goal = _fallback_goal_from_traversible(traversible, state)
                self.planner.set_multi_goal(goal, state)
            else:
                raise

        decrease_stop_cond =0
        if self.dilation_deg >= 6:
            decrease_stop_cond = 0.2 #decrease to 0.2 (7 grids until closest goal)
        stg_y, stg_x, replan, stop = self.planner.get_short_term_goal(state, found_goal = goal_found, decrease_stop_cond=decrease_stop_cond)
        stg_x, stg_y = stg_x - 1, stg_y - 1
        
        return (stg_y, stg_x), replan, stop
    
    def set_random_goal(self):
        obstacle_map = self.full_map.cpu().numpy()[0,0,::-1]
        goal = np.zeros_like(obstacle_map)
        goal_index = np.where((obstacle_map<1))
        np.random.seed(self.total_steps)
        if len(goal_index[0]) != 0:
            i = np.random.choice(len(goal_index[0]), 1)[0]
            h_goal = goal_index[0][i]
            w_goal = goal_index[1][i]
        else:
            h_goal = np.random.choice(goal.shape[0], 1)[0]
            w_goal = np.random.choice(goal.shape[1], 1)[0]
        goal[h_goal, w_goal] = 1
        return goal
    
    def update_metrics(self, metrics):
        self.metrics['distance_to_goal'] = metrics['distance_to_goal']
        self.metrics['spl'] = metrics['spl']
        self.metrics['softspl'] = metrics['softspl']
        if self.args.visualize:
            # if self.simulator._env.episode_over or self.total_steps == 500:
            if self.simulator._env.episode_over or self.total_steps == 999:
                self.save_video()

    def visualize(self, traversible, observations, number_action):
        if self.args.visualize:
            self.visualization_dir_unreal = self.visualization_dir.replace('data', 'data_unreal')
            
            save_map = copy.deepcopy(torch.from_numpy(traversible))
            gray_map = torch.stack((save_map, save_map, save_map))
            paper_obstacle_map = copy.deepcopy(gray_map)[:,1:-1,1:-1]
            paper_map = torch.zeros_like(paper_obstacle_map)
            paper_map_trans = paper_map.permute(1,2,0)
            unknown_rgb = colors.to_rgb('#FFFFFF')
            paper_map_trans[:,:,:] = torch.tensor( unknown_rgb)
            free_rgb = colors.to_rgb('#E7E7E7')
            paper_map_trans[self.fbe_free_map.cpu().numpy()[0,0,::-1]>0.5,:] = torch.tensor( free_rgb).double()
            obstacle_rgb = colors.to_rgb('#A2A2A2')
            paper_map_trans[skimage.morphology.binary_dilation(self.full_map.cpu().numpy()[0,0,::-1]>0.5,skimage.morphology.disk(1)),:] = torch.tensor(obstacle_rgb).double()
            paper_map_trans = paper_map_trans.permute(2,0,1)
            self.visualize_agent_and_goal(paper_map_trans)
            agent_coordinate = (int(self.history_pose[-1][0]*100/self.resolution), int((self.map_size_cm/100-self.history_pose[-1][1])*100/self.resolution))
            occupancy_map = crop_around_point((paper_map_trans.permute(1, 2, 0) * 255).numpy().astype(np.uint8), agent_coordinate, (150, 200))

            
            os.makedirs(f'{self.visualization_dir_unreal}/debug', exist_ok=True)
            # Build full canvas RGB from paper_map_trans (C,H,W -> H,W,C)
            full_canvas_rgb = paper_map_trans.permute(1, 2, 0).detach().cpu().numpy()
            if full_canvas_rgb.dtype != np.uint8:
                full_canvas_rgb = np.clip(full_canvas_rgb * 255.0, 0, 255).astype(np.uint8)
            full_canvas_rgb = np.ascontiguousarray(full_canvas_rgb)
            
            # Draw agent trajectory points: past (radius=3), latest (radius=6), color red
            H_fc, W_fc = full_canvas_rgb.shape[:2]
            n_hist = len(self.history_pose)
            for idx, pose in enumerate(self.history_pose):
                # Convert pose to pixel coords: x from pose[0], y from flipped pose[1]
                px = int((float(pose[0].item() if torch.is_tensor(pose[0]) else pose[0])) * 100 / self.resolution)
                py = int((self.map_size_cm / 100 - float(pose[1].item() if torch.is_tensor(pose[1]) else pose[1])) * 100 / self.resolution)
                if 0 <= px < W_fc and 0 <= py < H_fc:
                    radius = 6 if idx == n_hist - 1 else 3
                    cv2.circle(full_canvas_rgb, (px, py), radius, (0, 0, 255), -1)

                # Recompute current frontier map from free/unknown and obstacle
                free_np = (self.fbe_free_map.cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
                occ_dil = skimage.morphology.binary_dilation(
                    self.full_map.cpu().numpy()[0, 0], skimage.morphology.disk(4)
                )
                fbe_map_np = np.zeros_like(free_np, dtype=np.uint8)
                fbe_map_np[free_np > 0] = 1  # free=1
                fbe_map_np[occ_dil] = 3      # obstacle (dilated)=3

                fbe_cp = fbe_map_np.copy()
                fbe_cpp = fbe_map_np.copy()
                fbe_cp[fbe_cp == 0] = 4
                fbe_cp[fbe_cp < 4] = 0
                selem = skimage.morphology.disk(1)
                dil_unknown = skimage.morphology.binary_dilation(fbe_cp, selem)
                fbe_cpp[dil_unknown] = 0

                diff = fbe_map_np - fbe_cpp
                frontier_map_np = (diff == 1)
                ys, xs = np.where(frontier_map_np)
                # Note: full_canvas is vertically flipped vs raw map
                for yy, xx in zip(ys, xs):
                    py = H_fc - 1 - int(yy)
                    px = int(xx)
                    if 0 <= px < W_fc and 0 <= py < H_fc:
                        cv2.circle(full_canvas_rgb, (px, py), 2, (255, 0, 255), -1)

                # Optional: label and scale bar
                sb_len_px = 100  # 10 m at 10 cm/px
                sb_th = 3
                sb_margin = 10
                sb_y = H_fc - sb_margin
                sb_x1 = sb_margin
                sb_x2 = sb_margin + sb_len_px
                cv2.line(full_canvas_rgb, (sb_x1, sb_y), (sb_x2, sb_y), (0, 0, 0), sb_th)
                cv2.putText(full_canvas_rgb, '10 m', (sb_x1, sb_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

                out_path = f"{self.visualization_dir_unreal}/debug/full_canvas_{self.total_steps:04d}.png"
                cv2.imwrite(out_path, full_canvas_rgb)

            
            visualize_image = np.full((450, 800, 3), 255, dtype=np.uint8)
            visualize_image = add_resized_image(visualize_image, self.rgb_visualization, (10, 60), (320, 240))
            visualize_image = add_resized_image(visualize_image, occupancy_map, (340, 60), (180, 240))
            visualize_image = add_rectangle(visualize_image, (10, 60), (330, 300), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (340, 60), (520, 300), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (540, 60), (790, 165), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (540, 195), (790, 300), (128, 128, 128), thickness=1)
            visualize_image = add_rectangle(visualize_image, (10, 350), (790, 400), (128, 128, 128), thickness=1)
            visualize_image = add_text(visualize_image, "Observation (Goal: {})".format(self.obj_goal), (70, 50), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "Occupancy Map", (370, 50), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "Scene Graph Nodes", (580, 50), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "Scene Graph Edges", (580, 185), font_scale=0.5, thickness=1)
            visualize_image = add_text(visualize_image, "LLM Explanation", (330, 340), font_scale=0.5, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.text_node, 40), (550, 80), font_scale=0.3, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.text_edge, 40), (550, 215), font_scale=0.3, thickness=1)
            visualize_image = add_text_list(visualize_image, line_list(self.explanation, 150), (20, 370), font_scale=0.3, thickness=1)
            visualize_image = visualize_image[:, :, ::-1]
            
            # 存单帧图像
            # self.visualization_dir_unreal = self.visualization_dir.replace('data', 'data_unreal')
            # save_image_dir = os.path.join(self.visualization_dir_unreal, 'images', f'ep_{self.count_episodes:06d}')
            save_image_dir = os.path.join(self.visualization_dir_unreal, 'images')
            os.makedirs(save_image_dir, exist_ok=True)
            img_path = os.path.join(save_image_dir, f'{self.navigate_steps:06d}.png')
            cv2.imwrite(img_path, visualize_image)

            self.visualize_image_list.append(visualize_image)


    def save_video(self):
        self.visualization_dir_unreal = self.visualization_dir.replace('data', 'data_unreal')
        save_video_dir = os.path.join(self.visualization_dir_unreal, 'video')
        save_video_path = f'{save_video_dir}/vid_{self.count_episodes:06d}.mp4'
        if not os.path.exists(save_video_dir):
            os.makedirs(save_video_dir)
        height, width, layers = self.visualize_image_list[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(save_video_path, fourcc, 4.0, (width, height))

        for idx, visualize_image in enumerate(self.visualize_image_list):  
            video.write(visualize_image)
        video.release()
        
    def visualize_agent_and_goal(self, map):
        for idx, pose in enumerate(self.history_pose):
            draw_step_num = 30
            alpha = max(0, 1 - (len(self.history_pose) - idx) / draw_step_num)
            agent_size = 1
            if idx == len(self.history_pose) - 1:
                agent_size = 2
            draw_agent(agent=self, map=map, pose=pose, agent_size=agent_size, color_index=0, alpha=alpha)
        draw_goal(agent=self, map=map, goal_size=2, color_index=1)
        return map


def main():
    from pathlib import Path
    cwd = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize", action='store_true'
    )
    parser.add_argument(
        "--split_l", default=0, type=int
    )
    parser.add_argument(
        "--split_r", default=11, type=int
    )
    args = parser.parse_args()
    os.environ["CHALLENGE_CONFIG_FILE"] = str(cwd / "configs" / "challenge_objectnav2021.local.rgbd.yaml")
    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = habitat.get_config(config_paths)
    agent = SG_Nav_Agent(task_config=config, args=args)

    challenge = habitat.Challenge(eval_remote=False, split_l=args.split_l, split_r=args.split_r)

    challenge.submit(agent)


if __name__ == "__main__":
    main()