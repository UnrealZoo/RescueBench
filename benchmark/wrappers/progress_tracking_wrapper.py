"""Progress tracking controller for Rescue benchmark episodes.

This wrapper owns the benchmark scoring flow while keeping the scoring formula
in ``utils.progress_tracker.EpisodeProgressTracker`` unchanged.
"""

from utils.progress_tracker import EpisodeProgressTracker


class ProgressTrackingController:
    """Wrap per-step score updates and task-completion checks."""

    def __init__(self, benchmark, task_context):
        self.benchmark = benchmark
        self.tracker = EpisodeProgressTracker(
            task_context["injured_pose"],
            task_context["stretcher_pose"],
            rescue_distance=benchmark.rescue_distance,
            place_distance=benchmark.place_distance,
            stage2_success_radius=benchmark.stage2_success_radius,
            interaction_z_threshold=benchmark.interaction_z_threshold,
        )

    @property
    def stage1_success(self):
        return self.tracker.stage1_success

    @property
    def task_completion(self):
        return self.tracker.task_completion

    def reset(self, initial_position):
        self.tracker.reset(initial_position)

    def update_after_env_step(self, interaction_pos, carrying_now: bool):
        self.tracker.update(interaction_pos, carrying_now)

    def mark_stage2_drop_zone_entered(self, interaction_pos):
        return self.tracker.mark_stage2_drop_zone_entered(interaction_pos)

    def confirm_stage2_completion(self, interaction_pos):
        return self.tracker.confirm_stage2_completion(interaction_pos)

    def sync_passthrough_drop_at_wait_confirm(self, interaction_pos):
        return self.tracker.sync_passthrough_drop_at_wait_confirm(interaction_pos)

    def finalize(self):
        return self.tracker.finalize()
