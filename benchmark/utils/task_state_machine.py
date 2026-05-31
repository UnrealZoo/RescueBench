from typing import Dict, List, Optional, Tuple

import numpy as np


class RescueTaskStateMachine:
    """救援任务状态机: NAV_INJURED → RESCUE → NAV_STRETCHER → PLACE → COMPLETED"""

    STATES = {
        'NAVIGATE_TO_INJURED': 'Phase1: Navigating to injured',
        'RESCUE_INJURED': 'Phase1: Executing carry',
        'NAVIGATE_TO_STRETCHER': 'Phase2: Navigating to stretcher',
        'PLACE_ON_STRETCHER': 'Phase2: Executing drop',
        'WAIT_ENV_CONFIRM': 'Phase2: Drop done, waiting env termination',
        'COMPLETED': 'Task completed',
        'FAILED': 'Task failed'
    }

    FAILURE_REASONS = {
        'TIMEOUT_PHASE1': 'Timeout during navigation to injured',
        'TIMEOUT_PHASE2': 'Timeout during navigation to stretcher',
        'CARRY_FAILED': 'Failed to carry injured person',
        'DROP_FAILED': 'Failed to drop on stretcher',
        'DROP_OUT_OF_RANGE': 'Dropped injured person outside stage2 success radius',
        'TRUNCATED': 'Episode truncated by environment',
        # UE 已结束 episode，但 EpisodeProgressTracker 未满足 task_completion（常见于 passthrough 自主 drop）
        'ENV_TERM_INCOMPLETE': 'Env ended before benchmark completion (UE termination, tracker S1∧S2 not met)',
    }

    def __init__(self, env, rescue_distance: float = 120.0, place_distance: float = 100.0,
                 max_rescue_attempts: int = 50, max_place_attempts: int = 50,
                 action_wait_steps: int = 30, passthrough: bool = False,
                 interaction_z_threshold: float = 220.0):
        self.env = env
        self.passthrough = passthrough
        self.rescue_distance = rescue_distance
        self.place_distance = place_distance
        self.interaction_z_threshold = interaction_z_threshold
        self.max_rescue_attempts = max_rescue_attempts
        self.max_place_attempts = max_place_attempts
        self.action_wait_max = action_wait_steps

        self.state = 'NAVIGATE_TO_INJURED'
        self.injured_pose = None
        self.stretcher_pose = None

        self.current_step = 0
        self.phase1_start_step = 0
        self.phase2_start_step = 0
        self.phase1_start_time = 0
        self.phase2_start_time = 0

        self.rescue_attempts = 0
        self.place_attempts = 0
        self.action_wait_steps = 0

        self.phase1_collisions = 0
        self.phase2_collisions = 0

        self._carry_success = False
        self._drop_success = False
        self.failure_reason = ""

        self.open_door_distance = 200.0
        self.open_door_heading_deg = 35.0
        self.open_door_cooldown_steps = 12
        self._last_open_door_step = -10**9
        self._door_positions = None

    def set_task(self, injured_pose: List[float], stretcher_pose: List[float]):
        self.injured_pose = injured_pose
        self.stretcher_pose = stretcher_pose
        self.state = 'NAVIGATE_TO_INJURED'
        self.failure_reason = ""

    def reset(self, start_time: float = 0):
        self.state = 'NAVIGATE_TO_INJURED'
        self.current_step = 0
        self.phase1_start_step = 0
        self.phase2_start_step = 0
        self.phase1_start_time = start_time
        self.phase2_start_time = 0
        self.rescue_attempts = 0
        self.place_attempts = 0
        self.action_wait_steps = 0
        self.phase1_collisions = 0
        self.phase2_collisions = 0
        self._carry_success = False
        self._drop_success = False
        self.failure_reason = ""
        self._last_open_door_step = -10**9

    def get_current_phase(self) -> str:
        if self.state in ['NAVIGATE_TO_INJURED', 'RESCUE_INJURED']:
            return 'find_injured'
        return 'find_stretcher'

    def get_current_target(self) -> Optional[List[float]]:
        if self.state in ['NAVIGATE_TO_INJURED', 'RESCUE_INJURED']:
            return self.injured_pose
        if self.state in ['NAVIGATE_TO_STRETCHER', 'PLACE_ON_STRETCHER']:
            return self.stretcher_pose
        return None

    def _get_agent_position(self) -> Optional[np.ndarray]:
        try:
            env = self.env.unwrapped
            protagonist_id = getattr(env, 'protagonist_id', 0)
            cam_id = env.cam_list[protagonist_id]
            pos = env.unrealcv.get_cam_pose(cam_id)[:3]
            return np.array(pos)
        except Exception:
            return None

    def _get_agent_pose(self) -> Optional[np.ndarray]:
        try:
            env = self.env.unwrapped
            protagonist_id = getattr(env, 'protagonist_id', 0)
            cam_id = env.cam_list[protagonist_id]
            pose = env.unrealcv.get_cam_pose(cam_id)
            if pose is not None and len(pose) >= 5:
                return np.array(pose, dtype=float)
        except Exception:
            pass
        return None

    def _distance_to(self, target: List[float]) -> float:
        """2D 水平距离 (XY, 忽略 Z 高度差)"""
        if target is None:
            return float('inf')
        agent_pos = self._get_agent_position()
        if agent_pos is None:
            return float('inf')
        target_xy = np.array(target[:2])
        agent_xy = agent_pos[:2]
        return np.linalg.norm(agent_xy - target_xy)

    def _z_diff_to(self, target: List[float]) -> float:
        """垂直高度差 (Z)"""
        if target is None:
            return float('inf')
        agent_pos = self._get_agent_position()
        if agent_pos is None:
            return float('inf')
        target_z = target[2] if len(target) > 2 else 0.0
        return abs(agent_pos[2] - target_z)

    def _within_interaction_gate(self, target: List[float], xy_threshold: float) -> Tuple[bool, float, float]:
        dist_xy = self._distance_to(target)
        z_diff = self._z_diff_to(target)
        return dist_xy < xy_threshold and z_diff < self.interaction_z_threshold, dist_xy, z_diff

    @staticmethod
    def _relative_heading_deg(agent_pose: np.ndarray, target_pose: List[float]) -> float:
        y_delta = float(target_pose[1]) - float(agent_pose[1])
        x_delta = float(target_pose[0]) - float(agent_pose[0])
        if x_delta == 0 and y_delta == 0:
            return 0.0
        angle_abs = np.degrees(np.arctan2(y_delta, x_delta))
        angle_relative = angle_abs - float(agent_pose[4])
        if angle_relative > 180:
            angle_relative -= 360
        if angle_relative < -180:
            angle_relative += 360
        return float(angle_relative)

    def _get_door_positions(self) -> List[List[float]]:
        if self._door_positions is not None:
            return self._door_positions
        self._door_positions = []
        try:
            env = self.env.unwrapped
            door_names = list(getattr(env, 'env_configs', {}).get('interactive_door', []))
            for door_name in door_names:
                try:
                    loc = env.unrealcv.get_obj_location(door_name)
                    if loc is not None and len(loc) >= 3:
                        self._door_positions.append([float(loc[0]), float(loc[1]), float(loc[2])])
                except Exception:
                    continue
        except Exception:
            pass
        return self._door_positions

    def _should_open_door(self, move_action) -> bool:
        if self.current_step - self._last_open_door_step < self.open_door_cooldown_steps:
            return False
        try:
            if float(move_action[1]) <= 1e-3:
                return False
        except Exception:
            return False

        agent_pose = self._get_agent_pose()
        if agent_pose is None:
            return False
        door_positions = self._get_door_positions()
        if not door_positions:
            return False

        nearest = min(
            door_positions,
            key=lambda loc: float(np.linalg.norm(agent_pose[:2] - np.array(loc[:2], dtype=float))),
        )
        nearest_dist = float(np.linalg.norm(agent_pose[:2] - np.array(nearest[:2], dtype=float)))
        if nearest_dist > self.open_door_distance:
            return False

        heading = abs(self._relative_heading_deg(agent_pose, nearest))
        if heading > self.open_door_heading_deg:
            return False

        self._last_open_door_step = self.current_step
        return True

    def _is_carrying(self, info: Dict) -> bool:
        is_picked = info.get('picked', False)
        if is_picked:
            return True
        try:
            env = self.env.unwrapped
            protagonist_id = getattr(env, 'protagonist_id', 0)
            player = env.player_list[protagonist_id]
            return env.unrealcv.is_carrying(player)
        except Exception:
            return False

    def add_collision(self):
        if self.state in ['NAVIGATE_TO_INJURED', 'RESCUE_INJURED']:
            self.phase1_collisions += 1
        else:
            self.phase2_collisions += 1

    def update(self, nav_action, info: Dict, current_time: float = 0) -> Tuple[Tuple, Dict, bool]:
        """更新状态机 → (final_action, phase_info, should_continue)"""
        self.current_step += 1
        is_carrying = self._is_carrying(info)

        if len(nav_action) == 2:
            move_action, head_action = nav_action
            anim_action = 0
        else:
            move_action, head_action, anim_action = nav_action

        should_continue = True

        if self.passthrough:
            if self.state == 'NAVIGATE_TO_INJURED' and is_carrying:
                self.state = 'NAVIGATE_TO_STRETCHER'
                self._carry_success = True
                self.phase2_start_step = self.current_step
                self.phase2_start_time = current_time
                print(f"  [State] Step {self.current_step}: ✓ Carry detected -> NAVIGATE_TO_STRETCHER")
            elif self.state == 'NAVIGATE_TO_STRETCHER' and not is_carrying:
                self.state = 'WAIT_ENV_CONFIRM'
                self._drop_success = True
                print(f"  [State] Step {self.current_step}: ✓ Drop detected -> WAIT_ENV_CONFIRM")
            elif self.state == 'WAIT_ENV_CONFIRM' and is_carrying:
                self.state = 'NAVIGATE_TO_STRETCHER'
                print(f"  [State] Step {self.current_step}: Carry re-detected -> NAVIGATE_TO_STRETCHER")

            final_action = (move_action, head_action, anim_action)
            phase_info = {
                'state': self.state,
                'phase': self.get_current_phase(),
                'step': self.current_step,
                'target': self.get_current_target()
            }
            return final_action, phase_info, should_continue

        if self.state == 'NAVIGATE_TO_INJURED':
            anim_action = 0
            if is_carrying:
                self.state = 'NAVIGATE_TO_STRETCHER'
                self._carry_success = True
                self.phase2_start_step = self.current_step
                self.phase2_start_time = current_time
                print(f"  [State] Step {self.current_step}: ✓ Agent-triggered carry -> NAVIGATE_TO_STRETCHER")
            else:
                reachable, dist, z_diff = self._within_interaction_gate(self.injured_pose, self.rescue_distance)
                if reachable:
                    self.state = 'RESCUE_INJURED'
                    self.rescue_attempts = 0
                    self.action_wait_steps = self.action_wait_max
                    anim_action = 3
                    move_action = np.array([0.0, 0.0])
                    print(f"  [State] Step {self.current_step}: -> RESCUE_INJURED (xy={dist:.0f}cm, z={z_diff:.0f}cm)")
                elif self._should_open_door(move_action):
                    anim_action = 5
                    move_action = np.array([0.0, 0.0])
                    print(f"  [State] Step {self.current_step}: open_door")

        elif self.state == 'RESCUE_INJURED':
            if self.action_wait_steps > 0:
                self.action_wait_steps -= 1
                return (np.array([0.0, 0.0]), 0, 3), {'state': self.state}, True

            if is_carrying:
                self.state = 'NAVIGATE_TO_STRETCHER'
                self._carry_success = True
                self.phase2_start_step = self.current_step
                self.phase2_start_time = current_time
                print(f"  [State] Step {self.current_step}: ✓ Carry success -> NAVIGATE_TO_STRETCHER")
            else:
                self.rescue_attempts += 1
                if self.rescue_attempts >= self.max_rescue_attempts:
                    self.state = 'FAILED'
                    self.failure_reason = 'CARRY_FAILED'
                    should_continue = False
                    print(f"  [State] FAILED: Carry failed after {self.max_rescue_attempts} attempts")
                else:
                    if self.rescue_attempts % 10 == 0:
                        move_action = np.array([30.0 if self.rescue_attempts % 20 < 10 else -30.0, 0.0])
                    else:
                        move_action = np.array([0.0, 0.0])
                    anim_action = 3

        elif self.state == 'NAVIGATE_TO_STRETCHER':
            anim_action = 0
            reachable, dist, z_diff = self._within_interaction_gate(self.stretcher_pose, self.place_distance)
            if reachable:
                self.state = 'PLACE_ON_STRETCHER'
                self.place_attempts = 0
                self.action_wait_steps = self.action_wait_max
                anim_action = 4
                move_action = np.array([0.0, 0.0])
                print(f"  [State] Step {self.current_step}: -> PLACE_ON_STRETCHER (xy={dist:.0f}cm, z={z_diff:.0f}cm)")
            elif self._should_open_door(move_action):
                anim_action = 5
                move_action = np.array([0.0, 0.0])
                print(f"  [State] Step {self.current_step}: open_door")

        elif self.state == 'PLACE_ON_STRETCHER':
            if self.action_wait_steps > 0:
                self.action_wait_steps -= 1
                return (np.array([0.0, 0.0]), 0, 4), {'state': self.state}, True

            if not is_carrying:
                self.state = 'WAIT_ENV_CONFIRM'
                self._drop_success = True
                print(f"  [State] Step {self.current_step}: ✓ Drop success -> WAIT_ENV_CONFIRM")
            else:
                self.place_attempts += 1
                if self.place_attempts >= self.max_place_attempts:
                    self.state = 'FAILED'
                    self.failure_reason = 'DROP_FAILED'
                    should_continue = False
                    print(f"  [State] FAILED: Drop failed after {self.max_place_attempts} attempts")
                else:
                    if self.place_attempts % 10 == 0:
                        move_action = np.array([30.0 if self.place_attempts % 20 < 10 else -30.0, 0.0])
                    else:
                        move_action = np.array([0.0, 0.0])
                    anim_action = 4

        elif self.state == 'WAIT_ENV_CONFIRM':
            move_action = np.array([0.0, 0.0])
            anim_action = 0

        elif self.state == 'COMPLETED':
            move_action = np.array([0.0, 0.0])
            anim_action = 0
            should_continue = False

        elif self.state == 'FAILED':
            move_action = np.array([0.0, 0.0])
            anim_action = 0
            should_continue = False

        final_action = (move_action, head_action, anim_action)
        phase_info = {
            'state': self.state,
            'phase': self.get_current_phase(),
            'step': self.current_step,
            'target': self.get_current_target()
        }
        return final_action, phase_info, should_continue

    def get_metrics(self, total_time: float) -> Dict:
        p1s = self.phase2_start_step if self.phase2_start_step > 0 else self.current_step
        p1t = (self.phase2_start_time - self.phase1_start_time) if self.phase2_start_time > 0 else total_time
        return {
            'phase1_success': self._carry_success,
            'phase1_time': p1t,
            'phase1_steps': p1s,
            'phase1_collisions': self.phase1_collisions,
            'phase2_success': self._drop_success,
            'phase2_time': (total_time - p1t) if self.phase2_start_time > 0 else 0,
            'phase2_steps': (self.current_step - self.phase2_start_step) if self.phase2_start_step > 0 else 0,
            'phase2_collisions': self.phase2_collisions,
            'carry_success': self._carry_success,
            'drop_success': self._drop_success,
            'failure_reason': self.failure_reason,
            'final_state': self.state,
        }
