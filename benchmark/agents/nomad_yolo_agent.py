"""
NOMAD + YOLO Agent (简化版)
============================

结合NOMAD导航和YOLO目标检测的Agent。
只返回导航动作，任务状态由benchmark状态机管理。

使用方法:
    from agents.nomad_yolo_agent import NOMADYOLOAgent
    agent = NOMADYOLOAgent(
        device='cuda',
        topomap_dir='./topomap',
        yolo_weights='./weights/best.pt'
    )
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BENCHMARK_DIR)

from agents.agent_base import BaseAgent
from core.metrics import EpisodeMetrics
from .topomap_utils import is_topomap_multimap_root, resolve_topomap_dir_for_env

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[Warning] ultralytics not installed. YOLO detection disabled.")


class FallenPersonDetector:
    """YOLO检测器"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 conf_threshold: float = 0.5, device: str = 'cuda'):
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = None
        self.last_detection = None
        
        if not YOLO_AVAILABLE or not model_path:
            return
        
        if os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        try:
            import torch
            self.model = YOLO(model_path)
            actual_device = self.device if torch.cuda.is_available() else 'cpu'
            self.model.to(actual_device)
            print(f"[YOLO] Loaded: {model_path}, device: {actual_device}")
        except Exception as e:
            print(f"[Error] YOLO load failed: {e}")
            self.model = None
    
    def detect(self, image: np.ndarray, target_classes: Optional[List[int]] = None):
        if self.model is None:
            return []
        
        try:
            results = self.model(image, conf=self.conf_threshold, 
                               device=self.device, verbose=False, 
                               classes=target_classes)
            
            detections = []
            if results and results[0].boxes:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = self.model.names.get(cls_id, f"class_{cls_id}")
                    detections.append({
                        'box': (box[0], box[1], box[2], box[3]),
                        'confidence': conf,
                        'class': cls_name,
                        'class_id': cls_id
                    })
                detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.last_detection = detections[0] if detections else None
            return detections
        except Exception as e:
            print(f"[YOLO Error] {e}")
            return []


class NOMADYOLOAgent(BaseAgent):
    """
    NOMAD + YOLO Agent (简化版)
    
    只返回导航动作，任务状态由状态机管理。
    """
    
    def __init__(self, device: str = 'cuda', topomap_dir: Optional[str] = None,
                 yolo_weights: Optional[str] = None, yolo_conf: float = 0.5,
                 use_yolo_correction: bool = True, yolo_blend_ratio: float = 0.3,
                 **kwargs):
        
        self.device = device
        self.use_yolo_correction = use_yolo_correction
        self.yolo_blend_ratio = yolo_blend_ratio
        
        # YOLO检测器
        self.yolo_detector = None
        if YOLO_AVAILABLE and yolo_weights:
            self.yolo_detector = FallenPersonDetector(yolo_weights, yolo_conf, device)
            if self.yolo_detector.model is None:
                self.use_yolo_correction = False
        else:
            self.use_yolo_correction = False
        
        # 拓扑地图
        self.topomap_phase1 = []
        self.topomap_phase2 = []
        self.current_topomap = []
        self.current_phase = 'find_injured'
        self._topomap_multimap_root: Optional[str] = None
        
        if topomap_dir and os.path.exists(topomap_dir):
            abs_dir = os.path.abspath(topomap_dir)
            if is_topomap_multimap_root(abs_dir):
                self._topomap_multimap_root = abs_dir
                print("[NOMADYOLOAgent] topomap_dir 为多地图根目录，将在 prepare_episode 按 env_id 加载")
            else:
                self._load_topomaps(abs_dir)
        
        # 导航状态
        self.chosen_trajectory = None
        self.context_queue = []
        
        print(f"[NOMADYOLOAgent] Init: device={device}, yolo_correction={self.use_yolo_correction}")
    
    def prepare_episode(self, task_context: Dict[str, Any]) -> None:
        if not self._topomap_multimap_root:
            return
        env_id = task_context.get('env_id') or ''
        path = resolve_topomap_dir_for_env(self._topomap_multimap_root, env_id)
        if path:
            self._load_topomaps(path)
        else:
            self.topomap_phase1 = []
            self.topomap_phase2 = []
            self.current_topomap = self.topomap_phase1

    def _load_topomaps(self, topomap_dir: str):
        from PIL import Image as PILImage

        self.topomap_phase1 = []
        self.topomap_phase2 = []
        
        for path in [os.path.join(topomap_dir, 'images'),
                     os.path.join(topomap_dir, 'to_injured', 'images'), 
                     topomap_dir]:
            if os.path.exists(path):
                pngs = sorted([f for f in os.listdir(path) if f.endswith('.png')],
                             key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)
                if pngs:
                    for f in pngs:
                        self.topomap_phase1.append(
                            PILImage.open(os.path.join(path, f)).convert('RGB'))
                    print(f"[NOMADYOLOAgent] Phase1 topomap: {len(self.topomap_phase1)} nodes")
                    break
        
        for path in [os.path.join(topomap_dir, 'to_stretcher', 'images'),
                     os.path.join(topomap_dir, 'to_stretcher')]:
            if os.path.exists(path):
                pngs = sorted([f for f in os.listdir(path) if f.endswith('.png')],
                             key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)
                if pngs:
                    for f in pngs:
                        self.topomap_phase2.append(
                            PILImage.open(os.path.join(path, f)).convert('RGB'))
                    print(f"[NOMADYOLOAgent] Phase2 topomap: {len(self.topomap_phase2)} nodes")
                    break
        
        self.current_topomap = self.topomap_phase1
    
    def _detect_target(self, obs: np.ndarray, phase: str) -> Optional[Dict]:
        if self.yolo_detector is None:
            return None
        
        target_classes = [0] if phase == 'find_injured' else [2]
        detections = self.yolo_detector.detect(obs, target_classes)
        
        if detections:
            det = detections[0]
            h, w = obs.shape[:2]
            x1, y1, x2, y2 = det['box']
            return {
                'center_x': (x1 + x2) / 2 / w,
                'center_y': (y1 + y2) / 2 / h,
                'area': (x2 - x1) * (y2 - y1) / (w * h),
                'confidence': det['confidence'],
                'box': det['box']
            }
        return None
    
    def _get_yolo_nav_action(self, det: Dict) -> np.ndarray:
        angle = (det['center_x'] - 0.5) * 60
        area = det['area']
        velocity = 20 if area > 0.15 else (40 if area > 0.08 else 60)
        return np.array([np.clip(angle, -30, 30), velocity])
    
    def act(self, observation: np.ndarray, info: Dict) -> Tuple[Any, Dict]:
        """返回导航动作"""
        task_phase = info.get('task_phase', 'find_injured')
        target_pose = info.get('target_pose')
        
        if task_phase != self.current_phase:
            self.current_phase = task_phase
            if task_phase == 'find_stretcher':
                self.current_topomap = self.topomap_phase2
            else:
                self.current_topomap = self.topomap_phase1
        
        # 默认导航
        move_action = np.array([0.0, 50.0])
        head_action = 0
        
        # YOLO检测和修正
        detection = None
        if self.use_yolo_correction:
            detection = self._detect_target(observation, task_phase)
            if detection:
                yolo_move = self._get_yolo_nav_action(detection)
                move_action = (1 - self.yolo_blend_ratio) * move_action + self.yolo_blend_ratio * yolo_move
        
        extra_info = {
            'task_phase': task_phase,
            'yolo_detected': detection is not None,
            'trajectory': self.chosen_trajectory
        }
        
        return (move_action, head_action), extra_info
    
    def reset(self):
        self.current_phase = 'find_injured'
        self.current_topomap = self.topomap_phase1
        self.chosen_trajectory = None
        self.context_queue = []
    
    def on_episode_end(self, success: bool, metrics: EpisodeMetrics):
        status = "SUCCESS" if success else "FAILED"
        reason = f" ({metrics.failure_reason})" if metrics.failure_reason else ""
        print(f"[NOMADYOLOAgent] {status}{reason}, steps={metrics.steps}")


if __name__ == '__main__':
    print("Testing NOMADYOLOAgent...")
    agent = NOMADYOLOAgent(device='cpu')
    
    fake_obs = np.zeros((320, 320, 3), dtype=np.uint8)
    fake_info = {'task_phase': 'find_injured', 'picked': False}
    
    action, extra = agent.act(fake_obs, fake_info)
    print(f"Action: {action}")
    print(f"Extra: {extra}")
