"""Result printing and persistence for Rescue benchmark runs."""

import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from typing import Dict, List, Optional

from core.metrics import BenchmarkResult, EpisodeMetrics, LevelMetrics
from utils.task_state_machine import RescueTaskStateMachine


def _np():
    return __import__("numpy")


class ResultWriter:
    """Write incremental and final benchmark outputs with the existing schema."""

    def __init__(self, benchmark):
        self.benchmark = benchmark

    @staticmethod
    def metrics_to_episode_record(metrics: EpisodeMetrics) -> Dict:
        return {
            k: v
            for k, v in asdict(metrics).items()
            if k not in ("trajectory", "drone_trajectory")
        }

    def prepare_incremental_result_files(
        self,
        model_name: str,
        timestamp: str,
        resume_episode_records: Dict,
    ) -> None:
        benchmark = self.benchmark
        benchmark.current_model_name = model_name
        benchmark.current_run_timestamp = timestamp
        if benchmark.resume_jsonl and benchmark.resume_append:
            benchmark.incremental_result_file = benchmark.resume_jsonl
        else:
            benchmark.incremental_result_file = os.path.join(
                benchmark.output_dir, f"benchmark_{model_name}_{timestamp}.jsonl"
            )
            with open(benchmark.incremental_result_file, "w", encoding="utf-8") as f:
                for metrics in resume_episode_records.values():
                    f.write(json.dumps(self.metrics_to_episode_record(metrics), ensure_ascii=False) + "\n")
        print(f"Episode JSONL will be appended to: {benchmark.incremental_result_file}")

        benchmark.incremental_trajectory_file = None
        if benchmark.enable_trajectory_recording:
            if benchmark.resume_jsonl and benchmark.resume_append:
                resume_dir = os.path.dirname(benchmark.resume_jsonl)
                resume_name = os.path.basename(benchmark.resume_jsonl)
                if resume_name.startswith(f"benchmark_{model_name}_"):
                    traj_name = resume_name.replace(
                        f"benchmark_{model_name}_",
                        f"trajectories_{model_name}_",
                        1,
                    )
                else:
                    traj_name = f"trajectories_{model_name}_{timestamp}.jsonl"
                benchmark.incremental_trajectory_file = os.path.join(resume_dir, traj_name)
                open(benchmark.incremental_trajectory_file, "a", encoding="utf-8").close()
            else:
                benchmark.incremental_trajectory_file = os.path.join(
                    benchmark.output_dir, f"trajectories_{model_name}_{timestamp}.jsonl"
                )
                with open(benchmark.incremental_trajectory_file, "w", encoding="utf-8"):
                    pass
            print(f"Trajectory JSONL will be appended to: {benchmark.incremental_trajectory_file}")

    def append_episode_result(self, metrics: EpisodeMetrics) -> None:
        benchmark = self.benchmark
        if not benchmark.incremental_result_file:
            return

        episode_record = self.metrics_to_episode_record(metrics)
        with open(benchmark.incremental_result_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode_record, ensure_ascii=False) + "\n")

        if (
            benchmark.enable_trajectory_recording
            and benchmark.incremental_trajectory_file
            and metrics.trajectory
        ):
            trajectory_record = {
                "episode_id": metrics.episode_id,
                "level": metrics.level,
                "point_id": metrics.point_id,
                "trajectory": metrics.trajectory,
                "drone_trajectory": metrics.drone_trajectory,
            }
            with open(benchmark.incremental_trajectory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trajectory_record, ensure_ascii=False) + "\n")

    def build_level_metrics(self, level: int, eps: List[EpisodeMetrics]) -> LevelMetrics:
        np = _np()
        sims = [m.path_similarity for m in eps if m.path_similarity is not None]
        return LevelMetrics(
            level=level,
            num_episodes=len(eps),
            success_rate=np.mean([m.success for m in eps]),
            avg_time_cost=np.mean([m.time_cost for m in eps]),
            avg_steps=np.mean([m.steps for m in eps]),
            avg_collision_count=np.mean([m.collision_count for m in eps]),
            avg_task_score=np.mean([m.task_score for m in eps]),
            avg_s1_score=np.mean([m.s1_score for m in eps]),
            avg_s2_score=np.mean([m.s2_score for m in eps]),
            avg_s3_score=np.mean([m.s3_score for m in eps]),
            avg_s4_score=np.mean([m.s4_score for m in eps]),
            avg_movement_effectiveness=np.mean([m.movement_effectiveness for m in eps]),
            avg_control_loop_fps=np.mean([m.control_loop_fps for m in eps]),
            avg_path_length=np.mean([m.path_length for m in eps]),
            avg_path_similarity=np.mean(sims) if sims else None,
            std_time_cost=np.std([m.time_cost for m in eps]),
            std_steps=np.std([m.steps for m in eps]),
            std_collision_count=np.std([m.collision_count for m in eps]),
            std_task_score=np.std([m.task_score for m in eps]),
            std_s1_score=np.std([m.s1_score for m in eps]),
            std_s2_score=np.std([m.s2_score for m in eps]),
            std_s3_score=np.std([m.s3_score for m in eps]),
            std_s4_score=np.std([m.s4_score for m in eps]),
            std_movement_effectiveness=np.std([m.movement_effectiveness for m in eps]),
            std_control_loop_fps=np.std([m.control_loop_fps for m in eps]),
            std_path_length=np.std([m.path_length for m in eps]),
        )

    def print_level_summary(
        self,
        level: int,
        eps: List[EpisodeMetrics],
        label: Optional[str] = None,
    ) -> None:
        if not eps:
            return
        level_metrics = self.build_level_metrics(level, eps)
        ln = len(eps) or 1
        p1 = sum(e.phase1_success for e in eps) / ln
        p2 = sum(e.phase2_success for e in eps) / ln
        level_tag = label or f"L{level}"
        print(
            f"[LEVEL_SUMMARY] {level_tag} N={len(eps)} SR={level_metrics.success_rate:.0%} "
            f"P1={p1:.0%} P2={p2:.0%} Score={level_metrics.avg_task_score:.1f} "
            f"S1={level_metrics.avg_s1_score:.1f} S2={level_metrics.avg_s2_score:.1f} "
            f"S3={level_metrics.avg_s3_score:.1f} S4={level_metrics.avg_s4_score:.1f} "
            f"Eff={level_metrics.avg_movement_effectiveness:.0%} FPS={level_metrics.avg_control_loop_fps:.1f} "
            f"Time={level_metrics.avg_time_cost:.1f}s Steps={level_metrics.avg_steps:.0f} "
            f"Coll={level_metrics.avg_collision_count:.1f}"
        )
        level_fails = [e for e in eps if not e.success]
        if level_fails:
            reasons = {}
            for e in level_fails:
                key = e.failure_reason or "UNKNOWN"
                reasons[key] = reasons.get(key, 0) + 1
            parts = [
                f"{RescueTaskStateMachine.FAILURE_REASONS.get(r, r)}:{c}"
                for r, c in sorted(reasons.items(), key=lambda x: -x[1])
            ]
            print(f"[LEVEL_FAILS] {level_tag} Failures({len(level_fails)}): {', '.join(parts)}")

    def render_with_info(
        self,
        obs,
        state_machine,
        extra_info,
        steps,
        elapsed_time,
        level: int,
        point_id: int,
        episode_id: int,
        info: Optional[Dict] = None,
    ) -> None:
        """Write non-blocking render frames for live monitoring and later video export."""

        benchmark = self.benchmark
        np = _np()
        try:
            cv2 = __import__("cv2")
            vis = (obs[0] if isinstance(obs, tuple) else obs).copy()
            if vis.dtype != np.uint8:
                vis = (vis * 255).astype(np.uint8) if vis.max() <= 1.0 else vis.astype(np.uint8)
            h, w = vis.shape[:2]
            overlay = vis.copy()
            cv2.rectangle(overlay, (5, 5), (400, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
            for i, txt in enumerate(
                [
                    f"State: {state_machine.STATES.get(state_machine.state, state_machine.state)}",
                    f"Step: {steps} | Time: {elapsed_time:.1f}s",
                    f"Phase: {state_machine.get_current_phase()}",
                ]
            ):
                cv2.putText(
                    vis,
                    txt,
                    (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            traj = extra_info.get("trajectory")
            if traj is not None:
                center = (w // 2, h)
                prev = center
                for wp in traj:
                    px = max(0, min(w - 1, int(center[0] + wp[1] * 20)))
                    py = max(0, min(h - 1, int(center[1] + wp[0] * -20)))
                    cv2.line(vis, prev, (px, py), (0, 255, 0), 2)
                    cv2.circle(vis, (px, py), 2, (0, 255, 255), -1)
                    prev = (px, py)

            render_root = os.path.join(benchmark.output_dir, "_render_frames")
            render_dir = os.path.join(
                render_root,
                f"level_{level}",
                f"point_{point_id}",
                f"episode_{episode_id:04d}",
            )
            os.makedirs(render_dir, exist_ok=True)
            # 全局 latest_frame 便于实时监看；每个 episode 也保留独立 latest_frame 便于回放。
            self.atomic_imwrite(os.path.join(render_root, "latest_frame.jpg"), vis)
            self.atomic_imwrite(os.path.join(render_dir, "latest_frame.jpg"), vis)
            if steps % benchmark.save_frame_every == 0:
                self.atomic_imwrite(os.path.join(render_dir, f"frame_{steps:06d}.jpg"), vis)

            drone_vis = info.get("drone_observation") if info is not None else None
            if drone_vis is not None:
                drone_vis = np.asarray(drone_vis)
                if drone_vis.ndim == 4:
                    drone_vis = drone_vis[0]
                if drone_vis.ndim == 3 and drone_vis.shape[-1] >= 3:
                    drone_vis = drone_vis[..., :3]
                    if drone_vis.dtype != np.uint8:
                        drone_vis = (
                            (drone_vis * 255).astype(np.uint8)
                            if drone_vis.max() <= 1.0
                            else drone_vis.astype(np.uint8)
                        )
                    self.atomic_imwrite(os.path.join(render_root, "drone_latest_frame.jpg"), drone_vis)
                    self.atomic_imwrite(os.path.join(render_dir, "drone_latest_frame.jpg"), drone_vis)
                    if steps % benchmark.save_frame_every == 0:
                        frame_name = f"drone_frame_{steps:06d}.jpg"
                        self.atomic_imwrite(os.path.join(render_dir, frame_name), drone_vis)
        except Exception as e:
            print(f"[Render] step {steps}: {e}")

    @staticmethod
    def atomic_imwrite(target_path: str, image, jpeg_quality: int = 90) -> None:
        """先写临时文件再原子替换，避免 viewer 读到半写入帧。"""

        cv2 = __import__("cv2")
        parent = os.path.dirname(target_path) or "."
        base, ext = os.path.splitext(os.path.basename(target_path))
        fd, tmp_path = tempfile.mkstemp(prefix=f"{base}_", suffix=ext, dir=parent)
        os.close(fd)
        try:
            ok = cv2.imwrite(tmp_path, image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if ok:
                os.replace(tmp_path, target_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def print_summary(self, result: BenchmarkResult) -> None:
        np = _np()
        benchmark = self.benchmark
        eps = result.episode_details
        n = len(eps) or 1
        print(f"\n{'='*90}\n SUMMARY - {result.model_name}\n{'='*90}")
        hdr = (
            f"{'Level':<7} {'N':>4} {'SR':>6} {'P1':>6} {'P2':>6} "
            f"{'S1':>6} {'S2':>6} {'S3':>6} {'S4':>6} {'Score':>7} {'Eff':>6} {'FPS':>6} {'Time':>7} {'Steps':>7}"
        )
        if benchmark.enable_collision_detection:
            hdr += f" {'Coll':>5}"
        print(hdr)
        print("-" * 90)

        for level, m in sorted(result.level_metrics.items()):
            le = [e for e in eps if e.level == level]
            ln = len(le) or 1
            p1 = sum(e.phase1_success for e in le) / ln
            p2 = sum(e.phase2_success for e in le) / ln
            row = (
                f"L{level:<6} {m.num_episodes:>4} {m.success_rate:>5.0%} {p1:>5.0%} {p2:>5.0%} "
                f"{m.avg_s1_score:>5.1f} {m.avg_s2_score:>5.1f} {m.avg_s3_score:>5.1f} {m.avg_s4_score:>5.1f} "
                f"{m.avg_task_score:>6.1f} {m.avg_movement_effectiveness:>5.0%} {m.avg_control_loop_fps:>6.1f} "
                f"{m.avg_time_cost:>6.1f}s {m.avg_steps:>7.0f}"
            )
            if benchmark.enable_collision_detection:
                row += f" {m.avg_collision_count:>5.1f}"
            print(row)

        print(f"{'='*90}")
        sr = sum(e.success for e in eps) / n
        p1 = sum(e.phase1_success for e in eps) / n
        p2 = sum(e.phase2_success for e in eps) / n
        avg_score = float(np.mean([e.task_score for e in eps])) if eps else 0.0
        avg_s1 = float(np.mean([e.s1_score for e in eps])) if eps else 0.0
        avg_s2 = float(np.mean([e.s2_score for e in eps])) if eps else 0.0
        avg_s3 = float(np.mean([e.s3_score for e in eps])) if eps else 0.0
        avg_s4 = float(np.mean([e.s4_score for e in eps])) if eps else 0.0
        avg_eff = float(np.mean([e.movement_effectiveness for e in eps])) if eps else 0.0
        avg_fps = float(np.mean([e.control_loop_fps for e in eps])) if eps else 0.0
        avg_time = float(np.mean([e.time_cost for e in eps])) if eps else 0.0
        print(
            f" Overall({n}): SR={sr:.0%}  P1={p1:.0%}  P2={p2:.0%}  "
            f"S1={avg_s1:.1f} S2={avg_s2:.1f} S3={avg_s3:.1f} S4={avg_s4:.1f}  "
            f"Score={avg_score:.1f}  Eff={avg_eff:.0%}  FPS={avg_fps:.1f}  Time={avg_time:.1f}s"
        )

        fails = [e for e in eps if not e.success]
        if fails:
            reasons = {}
            for e in fails:
                reasons[e.failure_reason or "UNKNOWN"] = reasons.get(e.failure_reason or "UNKNOWN", 0) + 1
            parts = [
                f"{RescueTaskStateMachine.FAILURE_REASONS.get(r, r)}:{c}"
                for r, c in sorted(reasons.items(), key=lambda x: -x[1])
            ]
            print(f" Failures({len(fails)}): {', '.join(parts)}")

    def save_results(self, result: BenchmarkResult, model_name: str, timestamp: str) -> None:
        np = _np()
        benchmark = self.benchmark
        eps = result.episode_details
        n = len(eps) or 1
        result_dict = {
            "model_name": result.model_name,
            "env_name": result.env_name,
            "timestamp": result.timestamp,
            "config": result.config,
            "level_metrics": {str(k): asdict(v) for k, v in result.level_metrics.items()},
            "episode_summary": {
                "total": len(eps),
                "successes": sum(e.success for e in eps),
                "success_rate": sum(e.success for e in eps) / n,
            },
            "task_score_summary": {
                "avg_task_score": float(np.mean([e.task_score for e in eps])) if eps else 0.0,
                "avg_s1_score": float(np.mean([e.s1_score for e in eps])) if eps else 0.0,
                "avg_s2_score": float(np.mean([e.s2_score for e in eps])) if eps else 0.0,
                "avg_s3_score": float(np.mean([e.s3_score for e in eps])) if eps else 0.0,
                "avg_s4_score": float(np.mean([e.s4_score for e in eps])) if eps else 0.0,
                "avg_movement_effectiveness": float(np.mean([e.movement_effectiveness for e in eps])) if eps else 0.0,
            },
            "distance_summary": {
                "avg_stage1_final_distance": float(np.mean([e.stage1_final_distance for e in eps])) if eps else 0.0,
                "avg_stage2_final_distance": float(np.mean([e.stage2_final_distance for e in eps])) if eps else 0.0,
            },
            "efficiency_summary": {
                "avg_time_cost": float(np.mean([e.time_cost for e in eps])) if eps else 0.0,
                "avg_control_loop_fps": float(np.mean([e.control_loop_fps for e in eps])) if eps else 0.0,
                "avg_path_length": float(np.mean([e.path_length for e in eps])) if eps else 0.0,
            },
            "episodes": [{k: v for k, v in asdict(e).items() if k != "trajectory"} for e in eps],
            "phase_summary": {
                "phase1_success_rate": sum(e.phase1_success for e in eps) / n,
                "phase2_success_rate": sum(e.phase2_success for e in eps) / n,
                "task_completion_rate": sum(e.task_completion for e in eps) / n,
                "carry_success_rate": sum(e.carry_success for e in eps) / n,
                "drop_success_rate": sum(e.drop_success for e in eps) / n,
            },
        }

        if benchmark.enable_trajectory_recording:
            traj_file = os.path.join(benchmark.output_dir, f"trajectories_{model_name}_{timestamp}.json")
            trajs = {f"ep_{e.episode_id}_L{e.level}_P{e.point_id}": e.trajectory for e in eps if e.trajectory}
            with open(traj_file, "w") as f:
                json.dump(trajs, f)
            print(f"Trajectories saved to: {traj_file}")

        result_file = os.path.join(benchmark.output_dir, f"benchmark_{model_name}_{timestamp}.json")
        with open(result_file, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"Results saved to: {result_file}")
        if benchmark.render and benchmark.save_video:
            self.generate_video_from_frames(model_name, timestamp)

    def generate_video_from_frames(self, model_name: str, timestamp: str) -> None:
        benchmark = self.benchmark
        render_root = os.path.join(benchmark.output_dir, "_render_frames")
        if not os.path.exists(render_root):
            return

        episode_dirs = []
        for root, _, files in os.walk(render_root):
            has_ground_frames = any(f.startswith("frame_") and f.endswith(".jpg") for f in files)
            has_drone_frames = any(f.startswith("drone_frame_") and f.endswith(".jpg") for f in files)
            if has_ground_frames or has_drone_frames:
                episode_dirs.append(root)

        if not episode_dirs:
            return

        generated = 0
        for episode_dir in sorted(episode_dirs):
            rel_parts = os.path.relpath(episode_dir, render_root).split(os.sep)
            level_part = next((p for p in rel_parts if p.startswith("level_")), "level_unknown")
            point_part = next((p for p in rel_parts if p.startswith("point_")), "point_unknown")
            episode_part = next((p for p in rel_parts if p.startswith("episode_")), "episode_unknown")
            for frame_prefix, video_suffix in (("frame_", ""), ("drone_frame_", "_drone")):
                frame_files = sorted(
                    os.path.join(episode_dir, f)
                    for f in os.listdir(episode_dir)
                    if f.startswith(frame_prefix) and f.endswith(".jpg")
                )
                if not frame_files:
                    continue

                video_name = f"{level_part}_{point_part}_{episode_part}{video_suffix}.mp4"
                video_path = os.path.join(render_root, video_name)
                list_file = None
                try:
                    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
                        list_file = tmp.name
                        for frame in frame_files:
                            tmp.write(f"file '{frame}'\n")

                    cmd = [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-r",
                        str(benchmark.video_fps),
                        "-i",
                        list_file,
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        video_path,
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    generated += 1
                except (FileNotFoundError, subprocess.CalledProcessError):
                    pass
                finally:
                    if list_file and os.path.exists(list_file):
                        try:
                            os.remove(list_file)
                        except Exception:
                            pass

        print(f"[Video] Generated {generated} episode videos in: {render_root}")
