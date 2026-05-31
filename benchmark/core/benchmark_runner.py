"""Multi-map scheduling and level evaluation loop (Step 13)."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from core.metrics import BenchmarkResult, EpisodeMetrics, LevelMetrics


class BenchmarkRunner:
    """Orchestrates map-first dispatch and per-level evaluation."""

    def __init__(self, benchmark):
        self.benchmark = benchmark

    def build_map_level_schedule(self, levels: List[int]) -> Dict[str, Dict[int, List[int]]]:
        b = self.benchmark
        schedule: Dict[str, Dict[int, List[int]]] = {}
        for level in levels:
            points = b.task_loader.load_level_test_points(level)
            if not points:
                continue
            for point_id, point in enumerate(points):
                env_id = b.task_loader.resolve_env_id(point.get("env_id", b.env_id))
                env_schedule = schedule.setdefault(env_id, {})
                env_schedule.setdefault(level, []).append(point_id)
        return schedule

    def evaluate_level(
        self,
        level: int,
        episodes_per_point: int = 1,
        point_ids: Optional[List[int]] = None,
        close_env: bool = True,
        label: Optional[str] = None,
    ) -> Tuple[Optional[LevelMetrics], List[EpisodeMetrics]]:
        b = self.benchmark
        num_points = b.task_loader.get_point_count(level)
        if num_points == 0:
            print(f"[Warning] No test points for Level {level}")
            return None, []

        point_ids = list(range(num_points)) if point_ids is None else [p for p in point_ids if p < num_points]
        episode_metrics_list = []
        episode_id = 0
        title = label or f"LEVEL {level}"

        print(f"\n{'='*60}")
        print(f"EVALUATING {title}")
        print(f"{'='*60}")
        print(f"Test Points: {len(point_ids)}")
        print(f"Episodes per Point: {episodes_per_point}")
        print(f"Time Limit: {b.task_loader.get_level_time_limit_text(level, point_ids)}")
        print(f"{'='*60}")

        for point_id in point_ids:
            for ep in range(episodes_per_point):
                print(f"\n[L{level}] Point {point_id}/{num_points-1}, Episode {ep+1}/{episodes_per_point}")
                resumed_metrics = b.resume_manager.get(level, point_id, episode_id)
                if resumed_metrics is not None:
                    episode_metrics_list.append(resumed_metrics)
                    status = "OK" if resumed_metrics.success else "FAIL"
                    print(
                        f"  [SKIP_RESUME] L{resumed_metrics.level} P{resumed_metrics.point_id} "
                        f"E{ep+1}/{episodes_per_point} {status} "
                        f"t={resumed_metrics.time_cost:.2f}s steps={resumed_metrics.steps}"
                    )
                    episode_id += 1
                    continue

                try:
                    metrics = b.run_episode(level, point_id, episode_id)
                    episode_metrics_list.append(metrics)
                    b.result_writer.append_episode_result(metrics)
                    reason = metrics.failure_reason if metrics.failure_reason else "NONE"
                    final_state = metrics.final_state if metrics.final_state else "UNKNOWN"
                    status = "OK" if metrics.success else "FAIL"
                    print(
                        f"  [EP_RESULT] L{metrics.level} P{metrics.point_id} E{ep+1}/{episodes_per_point} "
                        f"{status} t={metrics.time_cost:.2f}s steps={metrics.steps} coll={metrics.collision_count} "
                        f"eff={metrics.movement_effectiveness:.0%} fps={metrics.control_loop_fps:.1f} "
                        f"p1={int(metrics.phase1_success)} p2={int(metrics.phase2_success)} "
                        f"s1={metrics.s1_score:.1f} s2={metrics.s2_score:.1f} "
                        f"s3={metrics.s3_score:.1f} s4={metrics.s4_score:.1f} "
                        f"d_inj_final={metrics.stage1_final_distance:.0f}cm "
                        f"d_str_final={metrics.stage2_final_distance:.0f}cm "
                        f"carry={int(metrics.carry_success)} drop={int(metrics.drop_success)} "
                        f"reason={reason} state={final_state}"
                    )
                except KeyboardInterrupt:
                    print(f"\n  [Interrupt] Ctrl+C in episode, stopping...")
                    raise
                except Exception as e:
                    print(f"  [Error] Episode failed: {e}")
                    import traceback

                    traceback.print_exc()
                    metrics = EpisodeMetrics(
                        episode_id=episode_id,
                        level=level,
                        point_id=point_id,
                        success=False,
                        time_cost=0,
                        steps=0,
                        collision_count=0,
                    )
                    episode_metrics_list.append(metrics)
                    b.result_writer.append_episode_result(metrics)
                    print(
                        f"  [EP_RESULT] L{level} P{point_id} E{ep+1}/{episodes_per_point} "
                        f"FAIL t=0.00s steps=0 coll=0 eff=0% fps=0.0 p1=0 p2=0 "
                        f"s1=0.0 s2=0.0 s3=0.0 s4=0.0 d_inj_final=0cm d_str_final=0cm "
                        f"carry=0 drop=0 reason=EXCEPTION state=FAILED"
                    )

                episode_id += 1

        if close_env:
            b._close_env()
        if not episode_metrics_list:
            return None, []

        eps = episode_metrics_list
        level_metrics = b.result_writer.build_level_metrics(level, eps)
        b.result_writer.print_level_summary(level, eps, label=label)
        return level_metrics, eps

    def run_benchmark(
        self,
        levels: List[int],
        episodes_per_point: int = 1,
        model_name: str = "unknown",
    ) -> BenchmarkResult:
        b = self.benchmark
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        b.result_writer.prepare_incremental_result_files(
            model_name,
            timestamp,
            b.resume_episode_records,
        )
        level_metrics_dict: Dict[int, LevelMetrics] = {}
        all_episodes: List[EpisodeMetrics] = []
        episodes_by_level: Dict[int, List[EpisodeMetrics]] = {level: [] for level in levels}
        schedule = self.build_map_level_schedule(levels)

        for level in levels:
            if not b.task_loader.get_point_count(level):
                print(f"[Warning] No test points for Level {level}")

        print(f"\n{'='*60}")
        print(" MAP-FIRST DISPATCH")
        print(f"{'='*60}")
        print(f"Requested Levels: {levels}")
        print(f"Maps Scheduled: {len(schedule)}")
        for env_id, level_points in schedule.items():
            scheduled_levels = [level for level in levels if level in level_points]
            num_points = sum(len(level_points[level]) for level in scheduled_levels)
            print(f"  - {env_id}: levels={scheduled_levels}, points={num_points}")
        print(f"{'='*60}")

        for env_id, level_points in schedule.items():
            scheduled_levels = [level for level in levels if level in level_points]
            num_points = sum(len(level_points[level]) for level in scheduled_levels)
            print(f"[MAP_START] {env_id} Levels={scheduled_levels} Points={num_points}")
            map_episode_count = 0
            try:
                for level in scheduled_levels:
                    point_ids = level_points[level]
                    if not point_ids:
                        continue
                    level_label = f"{env_id} L{level}"
                    lm, eps = self.evaluate_level(
                        level,
                        episodes_per_point,
                        point_ids=point_ids,
                        close_env=False,
                        label=level_label,
                    )
                    if lm:
                        episodes_by_level[level].extend(eps)
                        all_episodes.extend(eps)
                        map_episode_count += len(eps)
            except Exception as e:
                print(f"[Error] Map {env_id}: {e}")
                import traceback

                traceback.print_exc()
            finally:
                b._close_env()
                print(f"[MAP_DONE] {env_id} Episodes={map_episode_count}")

        for level in levels:
            eps = episodes_by_level.get(level, [])
            if eps:
                level_metrics_dict[level] = b.result_writer.build_level_metrics(level, eps)

        config = {
            k: getattr(b, k)
            for k in (
                "env_id",
                "resolution",
                "enable_collision_detection",
                "enable_trajectory_recording",
                "enable_path_similarity",
                "collision_method",
                "similarity_method",
                "rescue_distance",
                "place_distance",
                "interaction_z_threshold",
                "stage2_success_radius",
                "passthrough",
                "resume_jsonl",
                "resume_skip",
                "resume_append",
            )
        }
        config.update(
            levels=levels,
            episodes_per_point=episodes_per_point,
            dispatch_order="map->level->point",
            scheduled_maps=list(schedule.keys()),
        )

        result = BenchmarkResult(
            model_name=model_name,
            env_name=b.env_id,
            timestamp=timestamp,
            level_metrics=level_metrics_dict,
            episode_details=all_episodes,
            config=config,
        )
        b.result_writer.print_summary(result)
        b.result_writer.save_results(result, model_name, timestamp)
        return result
