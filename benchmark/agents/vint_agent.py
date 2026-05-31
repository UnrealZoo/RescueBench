"""
VINT/NOMAD Agent示例
====================

这是一个VINT/NOMAD模型的Agent实现示例。
仅输出导航动作；carry/drop 由 rescue_benchmark 状态机处理。

目标图（仅语义，不与拓扑图拼接）：
  - 一阶段 find_injured：test_jsonl 对应 ref_image（benchmark 注入 reference_image，BGR→RGB）
  - 二阶段 find_stretcher：benchmark/rescue_topmaps/stretcher/ 下与 point 对齐的担架/救护车视角图

拓扑图 images/ 仍加载；**进入二阶段会再次 _load_topomap**（优先 to_stretcher/，否则与一阶段同一套整图采集）。**送入模型的 goal_img 仍为上述语义图**（ref / stretcher），不与拓扑节点拼接。

依赖:
    - visualnav-transformer 项目
    - PyTorch
    - diffusers（NoMaD：DDPMScheduler + vision_encoder / dist_pred_net / noise_pred_net，与官方 navigate 一致）
"""

import numpy as np
import cv2
import traceback
import time
from typing import Dict, Any, Tuple, Optional, List
import sys
import os

# 添加父目录到path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.agent_base import BaseAgent
from core.metrics import EpisodeMetrics
from .topomap_utils import is_topomap_multimap_root, resolve_topomap_dir_for_env

# 尝试导入VINT相关依赖
# 默认路径：gym-rescue 的父目录下的 visualnav-transformer（如 .../Offline_RL_Active_Tracking/visualnav-transformer）。
# 可设环境变量 VISUALNAV_PATH 覆盖。
VINT_AVAILABLE = False
VISUALNAV_PATH = ''
try:
    import torch
    import yaml
    from PIL import Image as PILImage

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    # agents -> benchmark -> gym-rescue（两级）；再上一级为与 visualnav-transformer 同级的工作区根
    _gym_rescue_root = os.path.dirname(os.path.dirname(_script_dir))
    _workspace_root = os.path.dirname(_gym_rescue_root)
    _default_vn = os.path.join(_workspace_root, 'visualnav-transformer')
    VISUALNAV_PATH = os.path.abspath(
        os.environ.get('VISUALNAV_PATH', '').strip() or _default_vn
    )

    if os.path.isdir(VISUALNAV_PATH):
        sys.path.insert(0, VISUALNAV_PATH)
        sys.path.insert(0, os.path.join(VISUALNAV_PATH, 'train'))

        from deployment.src.utils import load_model, transform_images, to_numpy
        VINT_AVAILABLE = True
        print(f"[VINTAgent] VINT dependencies loaded from {VISUALNAV_PATH}")
    else:
        print(
            f"[VINTAgent] 未找到 visualnav-transformer 目录: {VISUALNAV_PATH}\n"
            f"  请克隆到与 gym-rescue 同级，或设置环境变量 VISUALNAV_PATH=/path/to/visualnav-transformer"
        )
except ImportError as e:
    print(f"[VINTAgent] Failed to import VINT dependencies: {e}")
    VINT_AVAILABLE = False


class VINTAgent(BaseAgent):
    """
    VINT (Visual Navigation Transformer) Agent
    
    使用VINT模型进行视觉导航，支持:
    - 单目标图像导航
    - 拓扑地图导航
    """
    
    def __init__(
        self,
        model_name: str = 'vint',
        device: str = 'cuda',
        topomap_dir: Optional[str] = None,
        waypoint_idx: int = 2,
        **kwargs
    ):
        """
        Args:
            model_name: 模型名称 ('vint', 'nomad', 'gnm')
            device: 运行设备
            topomap_dir: 拓扑地图目录
            waypoint_idx: 使用的轨迹点索引（与官方 navigate.py --waypoint 一致，默认 2；需 < len_traj_pred）

        仅负责导航（waypoint → 位移）；carry/drop 由 benchmark 状态机判定，不再用模型预测距离触发交互。
        """
        if not VINT_AVAILABLE:
            raise ImportError("VINT dependencies not available. "
                            "Please install visualnav-transformer.")
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.waypoint_idx = waypoint_idx
        
        # 加载模型
        self._load_model()
        
        # 加载拓扑地图（可为单地图目录，或含 UnrealRescue-* 子目录的多地图根目录）
        self.topomap_images = None
        self._topomap_multimap_root: Optional[str] = None
        self._resolved_topomap_dir: Optional[str] = None
        if topomap_dir:
            abs_dir = os.path.abspath(topomap_dir)
            if is_topomap_multimap_root(abs_dir):
                self._topomap_multimap_root = abs_dir
                print("[VINTAgent] topomap_dir 为多地图根目录，将在每局 prepare_episode 按 env_id 加载")
            else:
                self._resolved_topomap_dir = abs_dir
                self._load_topomap(abs_dir)
        
        # 状态变量（context_size / _obs_frame_count 已在 _load_model 中按各模型 yaml 设置，勿在此覆盖，否则 NoMaD 会堆错帧数导致 6ch/12ch 维度错误）
        self.context_queue = []
        self.current_phase = 'find_injured'  # 与 info['task_phase'] 对齐
        self.closest_node = 0
        self.goal_node = 0
        self._episode_point_id = 0
        self._episode_level = 0
        self._episode_env_id = ''
        self._stretcher_pil: Optional['PILImage.Image'] = None
        _bd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._benchmark_rescue_topmaps_stretcher = os.path.join(_bd, 'rescue_topmaps', 'stretcher')
        # 与 deployment/config/robot.yaml 一致，用于 normalize=True 时对 waypoint 去归一化（见官方 navigate.py）
        self._deploy_max_v = 0.2
        self._deploy_rate_hz = 4.0
        # 每局 env step 计数，与 rescue_benchmark 中每步一次 act 对齐（Step 1 = 首次推理）
        self._infer_step = 0

        print(f"[VINTAgent] Initialized: model={model_name}, device={self.device}")
    
    def _load_model(self):
        """加载VINT/NOMAD模型"""
        model_config_path = os.path.join(VISUALNAV_PATH, 'deployment/config/models.yaml')
        
        with open(model_config_path, 'r') as f:
            model_paths = yaml.safe_load(f)
        
        if self.model_name not in model_paths:
            raise ValueError(f"Model {self.model_name} not found in models.yaml")
        
        # 加载配置
        cfg_rel = model_paths[self.model_name]['config_path'].replace("../../", "").replace("../", "")
        model_config_file = os.path.join(VISUALNAV_PATH, cfg_rel)
        
        with open(model_config_file, 'r') as f:
            self.model_params = yaml.safe_load(f)
        
        # 加载模型权重
        ckpt_rel = model_paths[self.model_name]['ckpt_path'].replace("../", "")
        ckpt_path = os.path.join(VISUALNAV_PATH, 'deployment', ckpt_rel)
        
        print(f"[VINTAgent] Loading model from {ckpt_path}")
        self.model = load_model(ckpt_path, self.model_params, self.device)
        self.model = self.model.to(self.device).eval()
        
        self.context_size = self.model_params['context_size']
        # visualnav ViNT / NoMaD_ViNT: forward 里 obs_img[:, 3*context_size:,] 取「当前帧」3 通道再与 goal 拼成 6 通道；
        # 故 transform_images 需要 (context_size+1) 张图 → 3*(context_size+1) 通道，仅 context_size 张会少 3 通道而报错。
        self._obs_frame_count = self.context_size + 1

        self._noise_scheduler = None
        if str(self.model_params.get('model_type', '')).lower() == 'nomad':
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

            self._noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.model_params['num_diffusion_iters'],
                beta_schedule='squaredcos_cap_v2',
                clip_sample=True,
                prediction_type='epsilon',
            )
    
    def _infer_nomad(self, goal_image: 'PILImage.Image') -> Tuple[float, np.ndarray]:
        """单语义目标 NoMaD：vision_encoder → dist_pred_net → 扩散 noise_pred_net → get_action（对齐官方 demo）。"""
        from vint_train.training.train_utils import get_action

        pil_list = self._build_obs_pil_list_for_model()
        obs_tensor = transform_images(
            pil_list,
            self.model_params['image_size'],
            center_crop=False,
        ).to(self.device)
        goal_t = transform_images(
            [goal_image],
            self.model_params['image_size'],
            center_crop=False,
        ).to(self.device)
        mask = torch.zeros(1, dtype=torch.long, device=self.device)
        obsgoal_cond = self.model(
            'vision_encoder',
            obs_img=obs_tensor,
            goal_img=goal_t,
            input_goal_mask=mask,
        )
        dist_raw = self.model('dist_pred_net', obsgoal_cond=obsgoal_cond)
        distance = float(to_numpy(dist_raw).ravel()[0])
        obs_cond = obsgoal_cond[0].unsqueeze(0)
        n_it = int(self.model_params['len_traj_pred'])
        noisy_action = torch.randn((1, n_it, 2), device=self.device)
        naction = noisy_action
        assert self._noise_scheduler is not None
        self._noise_scheduler.set_timesteps(self.model_params['num_diffusion_iters'])
        for k in self._noise_scheduler.timesteps:
            noise_pred = self.model(
                'noise_pred_net',
                sample=naction,
                timestep=k,
                global_cond=obs_cond,
            )
            naction = self._noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction,
            ).prev_sample
        waypoints = to_numpy(get_action(naction))[0]
        return distance, waypoints

    def _build_obs_pil_list_for_model(self) -> List:
        """供 transform_images：共 context_size+1 帧；不足时在首部重复最早帧（与 deployment 热启动一致）。"""
        need = self._obs_frame_count
        q = list(self.context_queue)
        if not q:
            raise RuntimeError('context_queue is empty')
        if len(q) >= need:
            return q[-need:]
        pad_n = need - len(q)
        return [q[0]] * pad_n + q
    
    def _topomap_image_sort_key(self, fname: str):
        stem, _ = os.path.splitext(fname)
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    def _load_topomap(self, topomap_dir: str, *, phase2: bool = False):
        """加载拓扑地图图像。

        phase2=False：images/ → to_injured/images/ → 目录本身。
        phase2=True：二阶段重新加载时优先 to_stretcher/images（若你单独采了去担架段），否则仍用与一阶段同一套 images/（整图走一遍采集）。
        """
        if not os.path.exists(topomap_dir):
            print(f"[VINTAgent] Topomap directory not found: {topomap_dir}")
            self.topomap_images = None
            self.goal_node = 0
            return

        candidates: List[str] = []
        if phase2:
            candidates.extend(
                [
                    os.path.join(topomap_dir, 'to_stretcher', 'images'),
                    os.path.join(topomap_dir, 'to_stretcher'),
                ]
            )
        candidates.extend(
            [
                os.path.join(topomap_dir, 'images'),
                os.path.join(topomap_dir, 'to_injured', 'images'),
                topomap_dir,
            ]
        )
        search_dir = None
        image_files: List[str] = []
        for d in candidates:
            if not os.path.isdir(d):
                continue
            files = [
                f for f in os.listdir(d)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            if files:
                search_dir = d
                image_files = sorted(files, key=self._topomap_image_sort_key)
                break

        if not image_files:
            print(f"[VINTAgent] No images found under {topomap_dir}")
            self.topomap_images = None
            self.goal_node = 0
            return

        self.topomap_images = []
        for img_file in image_files:
            img_path = os.path.join(search_dir, img_file)
            img = PILImage.open(img_path).convert('RGB')
            self.topomap_images.append(img)

        self.goal_node = max(0, len(self.topomap_images) - 1)
        tag = '（二阶段）' if phase2 else ''
        print(f"[VINTAgent] Loaded {len(self.topomap_images)} topomap images{tag} from {search_dir}")

    def prepare_episode(self, task_context: Dict[str, Any]) -> None:
        """按 task_context 切换拓扑图（多地图），并记录 point/level 以加载 stretcher 目标图。"""
        self._episode_point_id = int(task_context.get('point_id', 0))
        self._episode_level = int(task_context.get('level', 0))
        self._episode_env_id = task_context.get('env_id') or ''

        if self._topomap_multimap_root:
            path = resolve_topomap_dir_for_env(self._topomap_multimap_root, self._episode_env_id)
            if path:
                self._resolved_topomap_dir = path
                self._load_topomap(path)
            else:
                self.topomap_images = None
                self.goal_node = 0
                self._resolved_topomap_dir = None
        else:
            # 单地图：__init__ 已 _load_topomap；目录即当前 topomap
            pass

        self._try_load_stretcher_image()

    def _try_load_stretcher_image(self) -> None:
        """二阶段：从 benchmark/rescue_topmaps/stretcher/ 或与 topomap 同级的 stretcher/ 加载（救护车/担架视角）。"""
        self._stretcher_pil = None
        pid, lv = self._episode_point_id, self._episode_level
        env_short = self._episode_env_id.replace('UnrealRescue-', '') if self._episode_env_id else ''
        env_short_variants = []
        if env_short:
            env_short_variants.append(env_short)
            if env_short.endswith('_dooropen'):
                env_short_variants.append(env_short[:-len('_dooropen')])
        candidates = [
            f'{pid}.png', f'{pid}.jpg',
            f'level_{lv}_{pid}.png', f'level_{lv}_{pid}.jpg',
        ]
        for env_name in env_short_variants:
            candidates.extend(
                [
                    f'{env_name}_level_{lv}_{pid}.png',
                    f'{env_name}_level_{lv}_{pid}.jpg',
                ]
            )
        # 每地图仅一张救护车/担架图时：{env_short} = env_id 去掉 UnrealRescue- 前缀
        for env_name in env_short_variants:
            candidates.extend([f'{env_name}.png', f'{env_name}.jpg'])
        search_dirs: List[str] = []
        if getattr(self, '_topomap_multimap_root', None):
            search_dirs.append(os.path.join(self._topomap_multimap_root, 'stretcher'))
        if self._resolved_topomap_dir:
            search_dirs.append(os.path.join(self._resolved_topomap_dir, 'stretcher'))
        search_dirs.append(self._benchmark_rescue_topmaps_stretcher)
        seen = set()
        for sd in search_dirs:
            sd = os.path.abspath(sd)
            if sd in seen or not os.path.isdir(sd):
                continue
            seen.add(sd)
            for name in candidates:
                p = os.path.join(sd, name)
                if os.path.isfile(p):
                    self._stretcher_pil = PILImage.open(p).convert('RGB')
                    print(f"[VINTAgent] 二阶段 stretcher 目标图: {p}")
                    return
        print(
            f"[VINTAgent] 未找到二阶段 stretcher 图 point={pid}（已搜: {search_dirs}），"
            f"请将图放入 benchmark/rescue_topmaps/stretcher/ 或各地图 topomap/stretcher/"
        )

    def _reference_array_to_pil(self, ref: np.ndarray) -> Optional['PILImage.Image']:
        """benchmark 用 cv2.imread → BGR uint8，模型需 RGB PIL。"""
        if ref is None or ref.size == 0:
            return None
        if ref.ndim != 3 or ref.shape[2] < 3:
            return None
        rgb = cv2.cvtColor(ref[:, :, :3], cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb.astype(np.uint8)).convert('RGB')

    def _get_semantic_goal_pil(self, info: Dict) -> Optional['PILImage.Image']:
        task_phase = info.get('task_phase', 'find_injured')
        if task_phase == 'find_injured':
            return self._reference_array_to_pil(info.get('reference_image'))
        return self._stretcher_pil
    
    def _preprocess_observation(self, observation: np.ndarray) -> PILImage.Image:
        """UnrealCV 观测为 BGR uint8，统一转 RGB PIL（勿用均值启发式，否则易错通道导致原地转圈）。"""
        if observation.ndim == 3 and observation.shape[2] >= 3:
            rgb = cv2.cvtColor(observation[:, :, :3], cv2.COLOR_BGR2RGB)
            return PILImage.fromarray(rgb.astype(np.uint8)).convert('RGB')
        return PILImage.fromarray(observation.astype(np.uint8))
    
    def _get_action_from_waypoint(self, waypoint: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        将 ViNT/NoMaD 输出的 waypoint 转为 Rescue 的 move_action [角度°, 速度]。

        与官方 deployment/src/pd_controller.py 一致：机体坐标系 **x 前向、y 侧向**，
        角速度用 atan2(dy, dx)（不要用 arctan2(dx, dy)，否则正前方会被当成 90° 而原地打转）。
        learn_angle=True 时为 4 维 [dx, dy, cos, sin]，在 |dx|+|dy| 极小时用 (cos,sin) 朝向（同官方）。
        """
        wp = np.asarray(waypoint, dtype=np.float64).reshape(-1)
        dx, dy = float(wp[0]), float(wp[1])
        hx = hy = None
        if wp.size >= 4:
            hx, hy = float(wp[2]), float(wp[3])

        if self.model_params.get('normalize', False):
            s = self._deploy_max_v / self._deploy_rate_hz
            dx *= s
            dy *= s

        eps = 1e-8
        dt = 1.0 / self._deploy_rate_hz
        # 官方 pd_controller：纯转向时用 arctan2(sin, cos) 即 hy, hx
        if hx is not None and abs(dx) < eps and abs(dy) < eps:
            angle_rad = np.arctan2(hy, hx)
        else:
            angle_rad = np.arctan2(dy, dx)

        angle = float(np.clip(np.degrees(angle_rad), -30.0, 30.0))

        # 线速度：与官方 pd_controller 一致 v_cmd = dx/dt（带符号，倒车为负）。
        # Mixed 连续动作第二维为 [-100,100]，与 FlexibleRoom move_action_continuous 一致。
        if abs(dx) < eps and abs(dy) < eps and hx is not None:
            v_cmd = 0.0
        else:
            v_cmd = dx / dt
        velocity = float(np.clip(v_cmd / self._deploy_max_v * 100.0, -100.0, 100.0))

        move_action = np.array([angle, velocity], dtype=np.float32)
        head_action = 0
        animation_action = 0

        return move_action, head_action, animation_action
    
    def act(self, observation: np.ndarray, info: Dict) -> Tuple[Any, Dict]:
        """goal_img：一阶段仅 ref_image；二阶段仅 stretcher 目录预载的图。不与拓扑节点拼接。"""
        task_phase = info.get('task_phase', 'find_injured')
        if task_phase != self.current_phase:
            if task_phase == 'find_stretcher':
                self.closest_node = 0
                # 抱起人后相机视角与任务目标与一阶段差异大；保留 phase1 的时序帧会严重干扰 ViNT 预测（易输出近零速度）。
                self.context_queue = []
                # 二阶段重新加载拓扑：与采集时同一地图走一遍则与一阶段同一 images/；若另有 to_stretcher/ 则优先加载。
                if self._resolved_topomap_dir:
                    self._load_topomap(self._resolved_topomap_dir, phase2=True)
                print(
                    '[VINTAgent] 进入二阶段 find_stretcher：已清空观测上下文，并已重新加载拓扑图'
                )
            self.current_phase = task_phase

        # 预处理观测
        obs_image = self._preprocess_observation(observation)
        
        # 更新 context：需保留 (context_size+1) 帧，与 visualnav ViNT/NoMaD 的 obs 通道数一致
        self.context_queue.append(obs_image)
        if len(self.context_queue) > self._obs_frame_count:
            self.context_queue.pop(0)

        semantic_goal = self._get_semantic_goal_pil(info)
        goal_image = semantic_goal if semantic_goal is not None else obs_image
        if semantic_goal is None:
            goal_src = 'fallback_obs'
        elif task_phase == 'find_injured':
            goal_src = 'ref_image'
        else:
            goal_src = 'stretcher'
        
        # 模型推理（每步打印一行，与 UniNaVid 的 Model inference 日志风格对齐）
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                if str(self.model_params.get('model_type', '')).lower() == 'nomad':
                    distance, waypoints = self._infer_nomad(goal_image)
                else:
                    # ViNT/GNM：必须 (context_size+1) 帧，否则 goal 分支只有 3 通道而 EfficientNet 要 6）
                    obs_images = transform_images(
                        self._build_obs_pil_list_for_model(),
                        self.model_params['image_size'],
                    ).to(self.device)
                    goal_tensor = transform_images(
                        [goal_image],
                        self.model_params['image_size'],
                    ).to(self.device)
                    distance_pred, waypoints = self.model(obs_images, goal_tensor)
                    distance = to_numpy(distance_pred)[0]
                    waypoints = to_numpy(waypoints)

                if len(waypoints.shape) == 3:
                    waypoints = waypoints[0]
                wp_idx = min(self.waypoint_idx, len(waypoints) - 1)
                chosen_wp = waypoints[wp_idx]

            dt = time.perf_counter() - t0
            self._infer_step += 1
            d0 = float(np.asarray(distance).ravel()[0])
            print(
                f"[VINTAgent] Step {self._infer_step}: Model inference #{self._infer_step} "
                f"({dt:.2f}s) phase={task_phase} distance={d0:.4f} "
                f"waypoint_idx={wp_idx} chosen_wp={np.asarray(chosen_wp).tolist()}"
            )

        except Exception as e:
            print(f"[VINTAgent] Model inference failed: {e!r}")
            traceback.print_exc()
            chosen_wp = np.array([0.0, 1.0])
            distance = 1.0
        
        # 仅导航；交互动画由 RescueTaskStateMachine 注入
        move_action, head_action, animation_action = self._get_action_from_waypoint(chosen_wp)
        action = (move_action, head_action, animation_action)
        
        extra_info = {
            'phase': self.current_phase,
            'task_phase': task_phase,
            'distance': float(np.asarray(distance).ravel()[0]),
            'waypoint': chosen_wp.tolist(),
            'closest_node': self.closest_node,
            'goal_source': goal_src,
        }
        
        return action, extra_info
    
    def reset(self):
        """重置Agent状态"""
        self.context_queue = []
        self.current_phase = 'find_injured'
        self.closest_node = 0
        self._infer_step = 0
    
    def on_episode_end(self, success: bool, metrics: EpisodeMetrics):
        """Episode结束回调"""
        status = "SUCCESS" if success else "FAILED"
        print(f"[VINTAgent] Episode {status}: "
              f"steps={metrics.steps}, time={metrics.time_cost:.2f}s")


# NOMAD Agent (继承VINT Agent)
class NOMADAgent(VINTAgent):
    """NOMAD Agent - 基于扩散模型的导航"""
    
    def __init__(self, **kwargs):
        kwargs['model_name'] = 'nomad'
        super().__init__(**kwargs)


if __name__ == '__main__':
    print("Testing VINTAgent initialization...")
    
    if VINT_AVAILABLE:
        try:
            agent = VINTAgent(device='cpu')
            print("VINTAgent initialized successfully!")
        except Exception as e:
            print(f"Failed to initialize VINTAgent: {e}")
    else:
        print("VINT dependencies not available")
