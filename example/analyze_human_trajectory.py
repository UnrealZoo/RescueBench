import argparse
import csv
import json
import math
from pathlib import Path


def load_jsonl(jsonl_path):
    episodes = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            episode["_line_id"] = line_id
            episodes.append(episode)
    return episodes


def distance_2d(point_a, point_b):
    return math.hypot(point_b["x"] - point_a["x"], point_b["y"] - point_a["y"])


def distance_3d(point_a, point_b):
    dx = point_b["x"] - point_a["x"]
    dy = point_b["y"] - point_a["y"]
    dz = point_b["z"] - point_a["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def sanitize_trajectory(trajectory, initial_jump_threshold):
    if initial_jump_threshold <= 0 or len(trajectory) < 2:
        return trajectory, False

    first_jump = distance_3d(trajectory[0], trajectory[1])
    if first_jump > initial_jump_threshold:
        return trajectory[1:], True
    return trajectory, False


def compute_path_lengths(trajectory):
    total_2d = 0.0
    total_3d = 0.0
    moving_steps = 0

    for prev_point, curr_point in zip(trajectory[:-1], trajectory[1:]):
        seg_2d = distance_2d(prev_point, curr_point)
        seg_3d = distance_3d(prev_point, curr_point)
        total_2d += seg_2d
        total_3d += seg_3d
        if seg_2d > 1e-6 or seg_3d > 1e-6:
            moving_steps += 1

    return total_2d, total_3d, moving_steps


def compute_episode_stats(episode, initial_jump_threshold):
    raw_trajectory = episode.get("trajectory", [])
    trajectory, dropped_initial_jump = sanitize_trajectory(raw_trajectory, initial_jump_threshold)
    if not trajectory:
        raise ValueError(f"episode_id={episode.get('episode_id')} has empty trajectory")

    path_length_2d, path_length_3d, moving_steps = compute_path_lengths(trajectory)

    pick_index = None
    for idx, point in enumerate(trajectory):
        if point.get("picked", False):
            pick_index = idx
            break

    if pick_index is None:
        search_2d = path_length_2d
        carry_2d = 0.0
        pick_step = None
        pick_time = None
    else:
        search_2d, _, _ = compute_path_lengths(trajectory[:pick_index + 1])
        carry_2d, _, _ = compute_path_lengths(trajectory[pick_index:])
        pick_step = trajectory[pick_index]["step"]
        pick_time = trajectory[pick_index].get("timestamp")

    start_point = trajectory[0]
    end_point = trajectory[-1]
    duration = episode.get("elapsed_sec", end_point.get("timestamp", 0.0))
    timeout_sec = episode.get("timeout_sec")

    return {
        "env_id": episode.get("env_id"),
        "level": episode.get("level"),
        "episode_id": episode.get("episode_id"),
        "seed": episode.get("seed"),
        "result": episode.get("result"),
        "terminated": episode.get("terminated"),
        "truncated": episode.get("truncated"),
        "timeout_sec": timeout_sec,
        "steps": episode.get("steps"),
        "fps": episode.get("fps"),
        "elapsed_sec": duration,
        "duration_sec": duration,
        "path_length_2d": path_length_2d,
        "path_length_3d": path_length_3d,
        "moving_steps": moving_steps,
        "pick_step": pick_step,
        "pick_time_sec": pick_time,
        "search_path_length_2d": search_2d,
        "carry_path_length_2d": carry_2d,
        "start_x": start_point["x"],
        "start_y": start_point["y"],
        "start_z": start_point["z"],
        "end_x": end_point["x"],
        "end_y": end_point["y"],
        "end_z": end_point["z"],
        "trajectory_points": len(trajectory),
        "dropped_initial_jump": dropped_initial_jump,
        "_trajectory": trajectory,
    }


def save_summary_csv(stats_list, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "env_id",
        "level",
        "episode_id",
        "seed",
        "result",
        "terminated",
        "truncated",
        "timeout_sec",
        "elapsed_sec",
        "steps",
        "fps",
        "duration_sec",
        "path_length_2d",
        "path_length_3d",
        "moving_steps",
        "pick_step",
        "pick_time_sec",
        "search_path_length_2d",
        "carry_path_length_2d",
        "start_x",
        "start_y",
        "start_z",
        "end_x",
        "end_y",
        "end_z",
        "trajectory_points",
        "dropped_initial_jump",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stats in stats_list:
            row = {key: stats.get(key) for key in fieldnames}
            writer.writerow(row)


def print_summary(stats_list):
    success_num = sum(1 for stats in stats_list if stats["result"] == "success")
    timeout_num = sum(1 for stats in stats_list if stats["result"] == "timeout")
    failed_num = sum(1 for stats in stats_list if stats["result"] == "failed")
    total_num = len(stats_list)
    avg_steps = sum(stats["steps"] for stats in stats_list) / total_num
    avg_duration = sum(stats["elapsed_sec"] for stats in stats_list) / total_num
    avg_path_2d = sum(stats["path_length_2d"] for stats in stats_list) / total_num

    print(f"Episodes: {total_num}")
    print(f"Success rate: {success_num}/{total_num} = {success_num / total_num:.2%}")
    print(f"Timeout count: {timeout_num}")
    print(f"Failed count: {failed_num}")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Average duration (sec): {avg_duration:.2f}")
    print(f"Average 2D path length: {avg_path_2d:.2f}")
    print("")

    for stats in stats_list:
        timeout_info = (
            f"{stats['elapsed_sec']:.2f}/{stats['timeout_sec']}s"
            if stats.get("timeout_sec") is not None
            else f"{stats['elapsed_sec']:.2f}s"
        )
        print(
            f"Episode {stats['episode_id']}: "
            f"result={stats['result']}, "
            f"steps={stats['steps']}, "
            f"time={timeout_info}, "
            f"path2d={stats['path_length_2d']:.2f}, "
            f"pick_step={stats['pick_step']}, "
            f"drop_initial_jump={stats['dropped_initial_jump']}"
        )


def plot_episode(stats, output_dir, show=False):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    trajectory = stats["_trajectory"]
    xs = [point["x"] for point in trajectory]
    ys = [point["y"] for point in trajectory]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xs, ys, linewidth=1.6, color="tab:blue", label="trajectory")
    ax.scatter(xs[0], ys[0], color="tab:green", s=60, label="start", zorder=3)
    ax.scatter(xs[-1], ys[-1], color="tab:red", s=60, label="end", zorder=3)

    pick_points = [point for point in trajectory if point.get("picked", False)]
    if pick_points:
        first_pick = pick_points[0]
        ax.scatter(first_pick["x"], first_pick["y"], color="tab:orange", s=60, label="first picked", zorder=3)

    ax.set_title(
        f"Episode {stats['episode_id']} | {stats['result']} | "
        f"time={stats['elapsed_sec']:.1f}s | "
        f"path2d={stats['path_length_2d']:.1f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"episode_{stats['episode_id']}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Analyze human rescue trajectory jsonl files.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a trajectory jsonl file generated by Rescue_HumanControl.py",
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        default=None,
        help="Only analyze one episode_id. Default: analyze all episodes in the jsonl file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory used to save plots and summary csv. Default: <jsonl_parent>/analysis/<jsonl_stem>",
    )
    parser.add_argument(
        "--initial-jump-threshold",
        type=float,
        default=500.0,
        help="If the first two trajectory points differ by more than this 3D distance, drop the first point.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show matplotlib figures in addition to saving them.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    episodes = load_jsonl(input_path)
    if args.episode_id is not None:
        episodes = [episode for episode in episodes if episode.get("episode_id") == args.episode_id]

    if not episodes:
        raise ValueError("No episodes found for the given filter.")

    if args.output_dir is None:
        output_dir = input_path.parent / "analysis" / input_path.stem
    else:
        output_dir = Path(args.output_dir)

    stats_list = [compute_episode_stats(episode, args.initial_jump_threshold) for episode in episodes]
    save_summary_csv(stats_list, output_dir / "summary.csv")

    plot_dir = output_dir / "plots"
    for stats in stats_list:
        plot_episode(stats, plot_dir, show=args.show)

    print_summary(stats_list)
    print("")
    print(f"Summary csv saved to: {output_dir / 'summary.csv'}")
    print(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
