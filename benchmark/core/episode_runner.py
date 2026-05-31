"""Single-episode execution for the Rescue benchmark."""

import os
import time

from core.metrics import EpisodeMetrics
from core.path_similarity_runtime import episode_path_similarity
from wrappers.progress_tracking_wrapper import ProgressTrackingController
from wrappers.state_machine_wrapper import RescueStateMachineController


class EpisodeRunner:
    """Run one benchmark episode without changing the existing behavior."""

    def __init__(self, benchmark):
        self.benchmark = benchmark

    def run_episode(self, level: int, point_id: int, episode_id: int = 0) -> EpisodeMetrics:
        benchmark = self.benchmark

        task_context = benchmark.task_loader.build_task_context(level, point_id)
        benchmark._ensure_env(task_context["env_id"], level)
        benchmark.env_manager.apply_task_context(task_context)
        time_limit = int(task_context.get("timeout", benchmark.TIME_LIMITS.get(level, 300)))

        reference_image = None
        ref_img_path = task_context.get("reference_image_path")
        if ref_img_path:
            if os.path.exists(ref_img_path):
                cv2 = __import__("cv2")
                reference_image = cv2.imread(ref_img_path)
                if reference_image is not None:
                    print(f"[RefImage] L{level} P{point_id} using: {ref_img_path}")
                else:
                    print(f"[RefImage] L{level} P{point_id} read failed: {ref_img_path}")
            else:
                print(f"[RefImage] L{level} P{point_id} missing: {ref_img_path}")
        else:
            print(f"[RefImage] L{level} P{point_id} no reference_image_path in env config")

        benchmark.agent.prepare_episode(task_context)
        obs, info = benchmark.env.reset()
        benchmark.agent.reset()

        progress_controller = ProgressTrackingController(benchmark, task_context)
        state_controller = RescueStateMachineController(benchmark, task_context, progress_controller)
        state_machine = state_controller.state_machine

        start_time = time.time()
        initial_interaction_pos = benchmark.env_manager.get_current_interaction_position()
        state_controller.reset(start_time)
        progress_controller.reset(initial_interaction_pos)
        if benchmark.enable_collision_detection and benchmark.collision_detector:
            benchmark.collision_detector.reset()

        trajectory = []
        drone_trajectory = []
        if benchmark.enable_trajectory_recording or benchmark.enable_path_similarity:
            trajectory.append(benchmark.env_manager.get_current_pose())
            drone_pose0 = benchmark.env_manager.get_current_drone_pose()
            if drone_pose0 is not None:
                drone_trajectory.append(drone_pose0)
        collision_count, steps, success = 0, 0, False
        env_terminated, env_truncated = False, False
        drone_stage1_handoff = False
        drone_stage2_handoff = False

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            interaction_pos_before_step = benchmark.env_manager.get_current_interaction_position()

            state_controller.prepare_info(
                info,
                reference_image,
                ref_img_path,
                level,
                point_id,
                episode_id,
            )
            step_obs, step_info = benchmark.agent.prepare_step_inputs(benchmark.env, obs, info)

            nav_action, extra_info = benchmark.agent.act(step_obs, step_info)
            drone_stage1_handoff = drone_stage1_handoff or bool(
                extra_info.get("drone_stage1_handoff", False)
            )
            drone_stage2_handoff = drone_stage2_handoff or bool(
                extra_info.get("drone_stage2_handoff", False)
            )
            prev_state, final_action, phase_info, should_continue = state_controller.update_action(
                nav_action,
                info,
                current_time,
                interaction_pos_before_step,
            )
            env_action = benchmark.env_manager.compose_env_action(final_action, extra_info)
            obs, reward, termination, truncation, info = benchmark.env.step(env_action)
            state_controller.log_wait_confirm_transition(prev_state, level, point_id, episode_id)
            steps += 1

            if benchmark.enable_trajectory_recording or benchmark.enable_path_similarity:
                trajectory.append(benchmark.env_manager.get_current_pose())
                drone_pose = benchmark.env_manager.get_current_drone_pose()
                if drone_pose is not None:
                    drone_trajectory.append(drone_pose)
            interaction_pos = benchmark.env_manager.get_current_interaction_position()
            progress_controller.update_after_env_step(
                interaction_pos,
                state_controller.is_carrying(info),
            )
            if not state_controller.validate_wait_confirm(interaction_pos, elapsed_time, steps):
                break
            if benchmark.enable_collision_detection and benchmark.collision_detector:
                if benchmark.collision_detector.check():
                    collision_count += 1
                    state_machine.add_collision()
            if benchmark.render:
                benchmark.result_writer.render_with_info(
                    obs,
                    state_machine,
                    extra_info,
                    steps,
                    elapsed_time,
                    level=level,
                    point_id=point_id,
                    episode_id=episode_id,
                    info=info,
                )

            if state_controller.mark_success_if_completed(elapsed_time, steps):
                success = True
                break
            if state_controller.mark_timeout_if_needed(elapsed_time, time_limit):
                break
            if termination:
                env_terminated = True
                state_controller.handle_env_termination(elapsed_time, steps)
                break
            if truncation:
                env_truncated = True
                state_controller.handle_truncation()
                break
            if not should_continue:
                break

        state_controller.finalize_after_loop(env_terminated)

        elapsed_time = time.time() - start_time
        sm_metrics = state_controller.get_state_metrics(elapsed_time)
        progress_metrics = progress_controller.finalize()
        if benchmark.agent.__class__.__name__ == "SeePointFlyAgent":
            progress_metrics = dict(progress_metrics)
            progress_metrics["s1_score"] = 25.0 if drone_stage1_handoff else 0.0
            progress_metrics["s3_score"] = 25.0 if drone_stage2_handoff else 0.0
            progress_metrics["stage1_score"] = progress_metrics["s1_score"] + progress_metrics["s2_score"]
            progress_metrics["stage2_score"] = progress_metrics["s3_score"] + progress_metrics["s4_score"]
            progress_metrics["task_score"] = (
                progress_metrics["stage1_score"] + progress_metrics["stage2_score"]
            )
        success = progress_metrics["task_completion"]
        control_loop_fps = float(steps / max(elapsed_time, 1e-6))
        path_similarity = episode_path_similarity(benchmark, trajectory, level, point_id)

        metrics = EpisodeMetrics(
            episode_id=episode_id,
            level=level,
            point_id=point_id,
            success=success,
            time_cost=elapsed_time,
            steps=steps,
            collision_count=collision_count,
            trajectory=trajectory if benchmark.enable_trajectory_recording else [],
            drone_trajectory=drone_trajectory if benchmark.enable_trajectory_recording else [],
            path_similarity=path_similarity,
            phase1_success=progress_metrics["stage1_success"],
            phase1_time=sm_metrics["phase1_time"],
            phase1_steps=sm_metrics["phase1_steps"],
            phase1_collisions=sm_metrics["phase1_collisions"],
            phase2_success=progress_metrics["stage2_success"],
            phase2_time=sm_metrics["phase2_time"],
            phase2_steps=sm_metrics["phase2_steps"],
            phase2_collisions=sm_metrics["phase2_collisions"],
            task_completion=progress_metrics["task_completion"],
            stage1_initial_distance=progress_metrics["stage1_initial_distance"],
            stage1_best_distance=progress_metrics["stage1_best_distance"],
            stage1_final_distance=progress_metrics["stage1_final_distance"],
            stage2_initial_distance=progress_metrics["stage2_initial_distance"],
            stage2_best_distance=progress_metrics["stage2_best_distance"],
            stage2_final_distance=progress_metrics["stage2_final_distance"],
            s1_score=progress_metrics["s1_score"],
            s2_score=progress_metrics["s2_score"],
            s3_score=progress_metrics["s3_score"],
            s4_score=progress_metrics["s4_score"],
            stage1_score=progress_metrics["stage1_score"],
            stage2_score=progress_metrics["stage2_score"],
            task_score=progress_metrics["task_score"],
            movement_effectiveness=progress_metrics["movement_effectiveness"],
            control_loop_fps=control_loop_fps,
            path_length=progress_metrics["path_length"],
            env_terminated=env_terminated,
            env_truncated=env_truncated,
            carry_success=sm_metrics["carry_success"],
            drop_success=sm_metrics["drop_success"],
            failure_reason=sm_metrics["failure_reason"],
            final_state=sm_metrics["final_state"],
            reference_text=task_context.get("reference_text", ""),
        )
        benchmark.agent.on_episode_end(success, metrics)
        return metrics
