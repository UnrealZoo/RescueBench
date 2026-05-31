"""State-machine controller for Rescue benchmark episodes."""

from utils.task_state_machine import RescueTaskStateMachine


class RescueStateMachineController:
    """Wrap task-control logic for Rescue benchmark episodes.

    This preserves both modes:
    - passthrough=False: state machine inserts carry/drop/open_door actions.
    - passthrough=True: model controls interaction actions, while the controller observes picked state.
    """

    def __init__(self, benchmark, task_context, progress_controller):
        self.benchmark = benchmark
        self.task_context = task_context
        self.progress = progress_controller
        self.state_machine = RescueTaskStateMachine(
            benchmark.env,
            rescue_distance=benchmark.rescue_distance,
            place_distance=benchmark.place_distance,
            passthrough=benchmark.passthrough,
            interaction_z_threshold=benchmark.interaction_z_threshold,
        )
        self.state_machine.set_task(task_context["injured_pose"], task_context["stretcher_pose"])

    def reset(self, start_time):
        self.state_machine.reset(start_time)

    def prepare_info(
        self,
        info,
        reference_image,
        reference_image_path,
        level,
        point_id,
        episode_id,
    ):
        info["task_phase"] = self.state_machine.get_current_phase()
        info["target_pose"] = self.state_machine.get_current_target()
        info["reference_text"] = self.task_context.get("reference_text", "")
        info["reference_image"] = reference_image
        info["reference_image_path"] = reference_image_path
        info["state_machine_state"] = self.state_machine.state
        info["rescue_distance"] = self.benchmark.rescue_distance
        info["place_distance"] = self.benchmark.place_distance
        info["interaction_z_threshold"] = self.benchmark.interaction_z_threshold
        info["env_id"] = self.task_context["env_id"]
        info["level"] = level
        info["point_id"] = point_id
        info["episode_id"] = episode_id
        return info

    def update_action(self, nav_action, info, current_time, interaction_pos_before_step):
        prev_state = self.state_machine.state
        final_action, phase_info, should_continue = self.state_machine.update(
            nav_action, info, current_time
        )
        # Passthrough：模型自主 drop 时不经过 PLACE_ON_STRETCHER，需将阶段二成功区与评分器对齐。
        if (
            self.benchmark.passthrough
            and prev_state == "NAVIGATE_TO_STRETCHER"
            and self.state_machine.state == "WAIT_ENV_CONFIRM"
        ):
            self.progress.sync_passthrough_drop_at_wait_confirm(interaction_pos_before_step)
        if prev_state != "PLACE_ON_STRETCHER" and self.state_machine.state == "PLACE_ON_STRETCHER":
            self.progress.mark_stage2_drop_zone_entered(interaction_pos_before_step)
        return prev_state, final_action, phase_info, should_continue

    def log_wait_confirm_transition(self, prev_state, level, point_id, episode_id):
        if prev_state != "WAIT_ENV_CONFIRM" and self.state_machine.state == "WAIT_ENV_CONFIRM":
            print(
                f"[CITYWALKER_DROP_DONE] map={self.task_context['env_id']} "
                f"level={level} point={point_id} episode={episode_id}"
            )

    def is_carrying(self, info):
        return self.state_machine._is_carrying(info)

    def validate_wait_confirm(self, interaction_pos, elapsed_time, steps):
        if self.state_machine.state == "WAIT_ENV_CONFIRM":
            drop_completed = self.progress.confirm_stage2_completion(interaction_pos)
            if not drop_completed:
                self.state_machine.failure_reason = "DROP_OUT_OF_RANGE"
                self.state_machine.state = "FAILED"
                print(
                    "  [DropOutOfRange] drop succeeded but outside stage2 "
                    f"success radius at {elapsed_time:.2f}s, {steps} steps"
                )
                return False
        return True

    def mark_success_if_completed(self, elapsed_time, steps):
        if self.progress.task_completion:
            self.state_machine.state = "COMPLETED"
            print(f"  [Success] custom task completion at {elapsed_time:.2f}s, {steps} steps")
            return True
        return False

    def mark_timeout_if_needed(self, elapsed_time, time_limit):
        if elapsed_time <= time_limit:
            return False
        self.state_machine.failure_reason = (
            "TIMEOUT_PHASE1"
            if self.state_machine.state in ("NAVIGATE_TO_INJURED", "RESCUE_INJURED")
            else "TIMEOUT_PHASE2"
        )
        self.state_machine.state = "FAILED"
        print(f"  [Timeout] {time_limit}s at {self.state_machine.state}")
        return True

    def handle_env_termination(self, elapsed_time, steps):
        if not self.progress.task_completion:
            if not self.state_machine.failure_reason:
                self.state_machine.failure_reason = "ENV_TERM_INCOMPLETE"
            print(
                f"  [EnvTermination] {elapsed_time:.2f}s, {steps} steps "
                f"(failure_reason={self.state_machine.failure_reason})"
            )

    def handle_truncation(self):
        self.state_machine.failure_reason = "TRUNCATED"
        self.state_machine.state = "FAILED"
        print("  [Truncated]")

    def finalize_after_loop(self, env_terminated):
        benchmark = self.benchmark
        # 统一收尾补判：无论主动/透传模式，只要 UE 先 termination 且最终已完成 drop，
        # 都尝试用终点几何补记阶段二成功，避免 ENV_TERM_INCOMPLETE 误判。
        recovered_on_termination = False
        if benchmark.passthrough_env_term_geometry_sync:
            recovered_on_termination = self._try_finalize_on_env_termination(env_terminated)
        if recovered_on_termination:
            print("  [EnvTerminationRecovery] terminal state reconciled to COMPLETED")
        elif (
            benchmark.passthrough_env_term_geometry_sync
            and benchmark.passthrough
            and self.progress.stage1_success
            and self.state_machine.state == "WAIT_ENV_CONFIRM"
        ):
            ip = benchmark.env_manager.get_current_interaction_position()
            self.progress.sync_passthrough_drop_at_wait_confirm(ip)
            if (
                self.progress.task_completion
                and self.state_machine.failure_reason == "ENV_TERM_INCOMPLETE"
            ):
                self.state_machine.failure_reason = ""
                self.state_machine.state = "COMPLETED"
                self.state_machine._drop_success = True

        if (
            env_terminated
            and not self.progress.task_completion
            and not self.state_machine.failure_reason
        ):
            self.state_machine.failure_reason = "ENV_TERM_INCOMPLETE"

    def get_state_metrics(self, elapsed_time):
        return self.state_machine.get_metrics(elapsed_time)

    def _try_finalize_on_env_termination(self, env_terminated):
        """Reconcile UE termination with the benchmark completion geometry."""

        if not env_terminated or not self.progress.stage1_success:
            return False

        # UE 终止与 picked 状态刷新存在一帧竞态：即使 drop 已发生，终止帧仍可能显示 carrying。
        # 这里不再用 carrying 作为一票否决，改为仅在阶段二后半段状态下尝试终止补判。
        if self.state_machine.state not in ("PLACE_ON_STRETCHER", "WAIT_ENV_CONFIRM"):
            return False

        ip = self.benchmark.env_manager.get_current_interaction_position()
        self.progress.sync_passthrough_drop_at_wait_confirm(ip)
        if not self.progress.task_completion:
            return False

        self.state_machine._drop_success = True
        self.state_machine.failure_reason = ""
        self.state_machine.state = "COMPLETED"
        return True
