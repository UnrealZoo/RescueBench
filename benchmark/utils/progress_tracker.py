import numpy as np


class EpisodeProgressTracker:
    """基于状态机同款 XY + Z gate 的独立评分器。"""

    STAGE_NEAR_RADIUS = 750.0

    def __init__(
        self,
        injured_pose,
        stretcher_pose,
        rescue_distance: float = 120.0,
        place_distance: float = 100.0,
        stage2_success_radius: float = 200.0,
        interaction_z_threshold: float = 220.0,
        eps: float = 1e-6,
    ):
        self.injured_pos = np.array(injured_pose[:3], dtype=float)
        self.stretcher_pos = np.array(stretcher_pose[:3], dtype=float)
        self.rescue_distance = float(rescue_distance)
        self.place_distance = float(place_distance)
        self.stage2_success_radius = float(stage2_success_radius)
        self.interaction_z_threshold = float(interaction_z_threshold)
        self.eps = float(eps)

        self.stage1_success = False
        self.stage2_success = False
        self.task_completion = False
        self.stage2_drop_zone_entered = False

        self.stage1_initial_distance = 0.0
        self.stage1_best_distance = float("inf")
        self.stage1_final_distance = float("inf")
        self.stage2_initial_distance = 0.0
        self.stage2_best_distance = float("inf")
        self.stage2_final_distance = float("inf")

        self.path_length = 0.0
        self.prev_position = None
        self.prev_goal_distance = None
        self.effective_steps = 0
        self.movement_steps = 0

    def _current_target_pos(self):
        return self.stretcher_pos if self.stage1_success else self.injured_pos

    def _xy_distance(self, pos, target_pos) -> float:
        pos_xy = np.asarray(pos[:2], dtype=float)
        target_xy = np.asarray(target_pos[:2], dtype=float)
        return float(np.linalg.norm(pos_xy - target_xy))

    def _z_diff(self, pos, target_pos) -> float:
        pos_z = float(pos[2]) if len(pos) > 2 else 0.0
        target_z = float(target_pos[2]) if len(target_pos) > 2 else 0.0
        return abs(pos_z - target_z)

    def _within_gate(self, pos, target_pos, xy_threshold: float) -> bool:
        return (
            self._xy_distance(pos, target_pos) <= float(xy_threshold)
            and self._z_diff(pos, target_pos) <= self.interaction_z_threshold
        )

    def reset(self, initial_position):
        pos = np.array(initial_position[:3], dtype=float)
        self.prev_position = pos
        self.stage1_initial_distance = self._xy_distance(pos, self.injured_pos)
        self.stage1_best_distance = self.stage1_initial_distance
        self.stage1_final_distance = self.stage1_initial_distance
        self.stage2_initial_distance = self._xy_distance(pos, self.stretcher_pos)
        self.stage2_best_distance = self.stage2_initial_distance
        self.stage2_final_distance = self.stage2_initial_distance
        self.prev_goal_distance = self.stage1_initial_distance
        self.effective_steps = 0
        self.movement_steps = 0

    def update(self, current_position, carrying_now: bool):
        pos = np.array(current_position[:3], dtype=float)
        if self.prev_position is not None:
            self.path_length += float(np.linalg.norm(pos - self.prev_position))
        self.prev_position = pos

        current_goal_distance = self._xy_distance(pos, self._current_target_pos())
        current_stretcher_distance = self._xy_distance(pos, self.stretcher_pos)
        self.stage2_final_distance = current_stretcher_distance
        if self.prev_goal_distance is not None:
            self.movement_steps += 1
            if current_goal_distance < self.prev_goal_distance:
                self.effective_steps += 1

        if not self.stage1_success:
            self.stage1_best_distance = min(self.stage1_best_distance, current_goal_distance)
            self.stage1_final_distance = current_goal_distance

        if carrying_now and not self.stage1_success:
            self.stage1_success = True
            d2 = current_stretcher_distance
            self.stage2_initial_distance = d2
            self.stage2_best_distance = d2
            self.stage2_final_distance = d2

        if self.stage1_success:
            d2 = current_stretcher_distance
            self.stage2_best_distance = min(self.stage2_best_distance, d2)
            self.stage2_final_distance = d2

        self.task_completion = self.stage1_success and self.stage2_success
        self.prev_goal_distance = self._xy_distance(pos, self._current_target_pos())

    def mark_stage2_drop_zone_entered(self, current_position=None):
        if current_position is not None:
            pos = np.array(current_position[:3], dtype=float)
            d2 = self._xy_distance(pos, self.stretcher_pos)
            self.stage2_best_distance = min(self.stage2_best_distance, d2)
            if self.stage2_initial_distance == 0.0:
                self.stage2_initial_distance = d2
            self.stage2_drop_zone_entered = self._within_gate(pos, self.stretcher_pos, self.stage2_success_radius)
        return self.stage2_drop_zone_entered

    def confirm_stage2_completion(self, current_position=None):
        if current_position is not None:
            pos = np.array(current_position[:3], dtype=float)
            d2 = self._xy_distance(pos, self.stretcher_pos)
            self.stage2_best_distance = min(self.stage2_best_distance, d2)
            self.stage2_final_distance = d2
            if self.stage2_initial_distance == 0.0:
                self.stage2_initial_distance = d2
        if self.stage1_success and self.stage2_drop_zone_entered:
            self.stage2_success = True
            self.task_completion = True
        return self.task_completion

    def sync_passthrough_drop_at_wait_confirm(self, current_position) -> bool:
        """Passthrough 对齐：状态机进入 WAIT_ENV_CONFIRM 时从未经过 PLACE_ON_STRETCHER，需补登阶段二成功区。

        先按 ``stage2_success_radius`` 尝试 ``mark_stage2_drop_zone_entered``；若仍未进入（极少数位姿），
        再用 ``place_distance`` 作为与主动模式一致的交互门限兜底，然后 ``confirm_stage2_completion``。
        """
        if not self.stage1_success:
            return False
        pos = np.array(current_position[:3], dtype=float)
        self.mark_stage2_drop_zone_entered(current_position)
        if not self.stage2_drop_zone_entered and self._within_gate(
            pos, self.stretcher_pos, self.place_distance
        ):
            d2 = self._xy_distance(pos, self.stretcher_pos)
            self.stage2_best_distance = min(self.stage2_best_distance, d2)
            self.stage2_final_distance = d2
            self.stage2_drop_zone_entered = True
        return self.confirm_stage2_completion(current_position)

    def _linear_score(self, distance: float, max_score: float, near_radius: float, far_radius: float = 0.0) -> float:
        if distance <= far_radius:
            return float(max_score)
        if distance >= near_radius:
            return 0.0
        return float(max_score) * float(
            np.clip((near_radius - distance) / max(near_radius - far_radius, self.eps), 0.0, 1.0)
        )

    def finalize(self):
        if self.stage1_success:
            s1_score = 25.0
            s2_score = 25.0
        else:
            s1_score = 25.0 if self.stage1_final_distance <= self.STAGE_NEAR_RADIUS else 0.0
            s2_score = self._linear_score(
                self.stage1_final_distance,
                max_score=25.0,
                near_radius=self.STAGE_NEAR_RADIUS,
                far_radius=0.0,
            )

        if not self.stage1_success:
            s3_score = 0.0
            s4_score = 0.0
        else:
            s3_score = 25.0 if self.stage2_final_distance <= self.STAGE_NEAR_RADIUS else 0.0
            if self.stage2_success:
                s4_score = 25.0
            else:
                s4_score = self._linear_score(
                    self.stage2_final_distance,
                    max_score=25.0,
                    near_radius=self.STAGE_NEAR_RADIUS,
                    far_radius=self.stage2_success_radius,
                )

        stage1_score = s1_score + s2_score
        stage2_score = s3_score + s4_score
        movement_effectiveness = (
            float(self.effective_steps) / float(self.movement_steps) if self.movement_steps > 0 else 0.0
        )

        return {
            "stage1_success": self.stage1_success,
            "stage2_success": self.stage2_success,
            "task_completion": self.task_completion,
            "stage1_initial_distance": self.stage1_initial_distance,
            "stage1_best_distance": self.stage1_best_distance if np.isfinite(self.stage1_best_distance) else 0.0,
            "stage1_final_distance": self.stage1_final_distance if np.isfinite(self.stage1_final_distance) else 0.0,
            "stage2_initial_distance": self.stage2_initial_distance,
            "stage2_best_distance": self.stage2_best_distance if np.isfinite(self.stage2_best_distance) else 0.0,
            "stage2_final_distance": self.stage2_final_distance if np.isfinite(self.stage2_final_distance) else 0.0,
            "s1_score": s1_score,
            "s2_score": s2_score,
            "s3_score": s3_score,
            "s4_score": s4_score,
            "stage1_score": stage1_score,
            "stage2_score": stage2_score,
            "task_score": stage1_score + stage2_score,
            "movement_effectiveness": movement_effectiveness,
            "path_length": self.path_length,
        }
