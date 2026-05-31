"""
build_nav_graph.py

End-to-end pipeline for building a navigation graph of a rescue environment.

  Phase 1 — Point collection  (from point_picker.py)
    Deploys multiple random agents to explore the UE5 environment and records
    all visited grid cells as reachable points, saving them to a .pkl file.

  Phase 2 — Graph construction (from make_graph.py)
    Loads the .pkl, filters nearby duplicate nodes, connects them via agent
    trajectories and spatial proximity, and saves a NetworkX .gpickle that
    can be used by the agent placement sampler (points_sampler.py).

Usage:
    # Full pipeline
    python build_nav_graph.py -e FlexibleRoom --max_step 3000

    # Skip collection, rebuild graph from an existing pkl
    python build_nav_graph.py -e FlexibleRoom --skip_collection \\
        --pkl_path path/to/FlexibleRoom_reachable_points.pkl
"""

import sys
import os
import math
import time
import random
import pickle
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import tqdm

# Try to load CJK font on Windows; fall back to system default otherwise
try:
    matplotlib.font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
except Exception:
    pass
os.environ['UnrealEnv'] = r"E:\Code_Space"
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# import gymnasium as gym
import gym
from gym_rescue.envs.wrappers import time_dilation, configUE, population


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Reachable-point collection  (originally point_picker.py)
# ─────────────────────────────────────────────────────────────────────────────

class RandomAgent:
    """Holds the same action for *keep_steps* steps, then resamples."""

    def __init__(self, action_space):
        self.action_space = action_space
        self.count_steps = 0
        self.action = self.action_space.sample()

    def act(self, keep_steps=10):
        self.count_steps += 1
        if self.count_steps > keep_steps:
            self.action = self.action_space.sample()
            self.count_steps = 0
        return self.action

    def reset(self):
        self.action = self.action_space.sample()
        self.count_steps = 0


class RegionConstrainedRandomAgent:
    """Random agent that biases actions toward keeping agents inside a bounding region."""

    def __init__(self, action_space, env, sample_region=None):
        self.action_space = action_space
        self.env = env
        self.count_steps = 0
        self.sample_region = sample_region  # [[x_min, x_max], [y_min, y_max]]

        try:
            self.move_actions = self.env.unwrapped.player_move_action
            print(f"Loaded {len(self.move_actions)} move actions from environment")
        except Exception:
            self.move_actions = [
                [0, 100], [0, -100], [15, 50], [-15, 50],
                [30, 0], [-30, 0], [0, 0],
            ]
            print("Using default move actions")

        if isinstance(self.action_space, list):
            self.action = [self._sample_valid_action(i) for i in range(len(self.action_space))]
        else:
            self.action = self.action_space.sample()

    def _predict_next_position(self, current_pos, current_rot, action_idx):
        if action_idx >= len(self.move_actions):
            return current_pos
        action = self.move_actions[action_idx]
        if len(action) == 2:
            angle_change, distance = action
            new_angle_rad = math.radians(current_rot[1] + angle_change)
            return [
                current_pos[0] + distance * math.cos(new_angle_rad),
                current_pos[1] + distance * math.sin(new_angle_rad),
                current_pos[2],
            ]
        elif len(action) == 4:
            forward, right, up, _ = action
            angle_rad = math.radians(current_rot[1])
            return [
                current_pos[0] + forward * 100 * math.cos(angle_rad) - right * 100 * math.sin(angle_rad),
                current_pos[1] + forward * 100 * math.sin(angle_rad) + right * 100 * math.cos(angle_rad),
                current_pos[2] + up * 100,
            ]
        return current_pos

    def _is_in_region(self, position):
        if self.sample_region is None:
            return True
        x, y = position[0], position[1]
        (x0, x1), (y0, y1) = self.sample_region
        return x0 <= x <= x1 and y0 <= y <= y1

    def _get_agent_pose(self, agent_idx):
        try:
            name = self.env.unwrapped.player_list[agent_idx]
            loc = self.env.unwrapped.unrealcv.get_obj_location(name)
            rot = self.env.unwrapped.unrealcv.get_obj_rotation(name)
            return loc, rot
        except Exception:
            return [0, 0, 0], [0, 0, 0]

    def _sample_valid_action(self, agent_idx):
        if self.sample_region is None:
            return random.randint(0, len(self.move_actions) - 1)

        current_pos, current_rot = self._get_agent_pose(agent_idx)

        if not self._is_in_region(current_pos):
            x_c = (self.sample_region[0][0] + self.sample_region[0][1]) / 2
            y_c = (self.sample_region[1][0] + self.sample_region[1][1]) / 2
            target_deg = math.degrees(math.atan2(y_c - current_pos[1], x_c - current_pos[0]))
            angle_diff = (target_deg - current_rot[1] + 180) % 360 - 180
            if abs(angle_diff) < 30:
                cands = [i for i, a in enumerate(self.move_actions) if len(a) == 2 and a[1] > 0]
            elif angle_diff > 0:
                cands = [i for i, a in enumerate(self.move_actions) if len(a) == 2 and a[0] > 0]
            else:
                cands = [i for i, a in enumerate(self.move_actions) if len(a) == 2 and a[0] < 0]
            if cands:
                return random.choice(cands)
            return random.randint(0, len(self.move_actions) - 1)

        valid = [
            i for i in range(len(self.move_actions))
            if self._is_in_region(self._predict_next_position(current_pos, current_rot, i))
        ]
        return random.choice(valid) if valid else random.randint(0, len(self.move_actions) - 1)

    def act(self, keep_steps=10):
        self.count_steps += 1
        if self.count_steps > keep_steps:
            if len(self.action_space) > 1:
                self.action = tuple(self._sample_valid_action(i) for i in range(len(self.action_space)))
            else:
                self.action = self._sample_valid_action(0)
            self.count_steps = 0
        return self.action

    def reset(self):
        if isinstance(self.action_space, list):
            self.action = [self._sample_valid_action(i) for i in range(len(self.action_space))]
        else:
            self.action = self._sample_valid_action(0)
        self.count_steps = 0


class ReachablePointsCollector:
    """Explores the environment with random agents and records all visited positions."""

    def __init__(self, env, num_agents=5, grid_resolution=100.0, sample_region=None):
        self.num_agents = num_agents
        self.resolution = grid_resolution
        self.sample_region = sample_region

        self.reachable_grid = {}
        self.agent_paths = [[] for _ in range(num_agents)]

        self.env = population.RandomPopulationWrapper(env, num_agents, num_agents)
        self.env = configUE.ConfigUEWrapper(self.env, offscreen=False)
        self.env.reset()

        if sample_region:
            print(f"Region constraint enabled: X={sample_region[0]}, Y={sample_region[1]}")
            self.agents = RegionConstrainedRandomAgent(self.env.action_space, self.env, sample_region)
        else:
            self.agents = RandomAgent(self.env.action_space)

        # Per-agent random action speed scaling
        self.action_clip = {
            obj: [random.choice([0.5, 1.0]) for _ in range(8)]
            for obj in self.env.unwrapped.player_list
        }

    def _discretize_position(self, position):
        x, y, z = position
        return (round(x / self.resolution), round(y / self.resolution), round(z / self.resolution))

    def _is_in_sample_region(self, position):
        if self.sample_region is None:
            return True
        x, y = position[0], position[1]
        (x0, x1), (y0, y1) = self.sample_region
        return x0 <= x <= x1 and y0 <= y <= y1

    def get_all_agent_poses(self):
        return [
            self.env.unwrapped.unrealcv.get_obj_location(name)
            for name in self.env.unwrapped.player_list
        ]

    def collect(self, env_name, steps=5000, render=True, save_interval=500, save_path="."):
        """Run exploration loop, periodically save progress, and return the final .pkl path."""
        initial_poses = self.get_all_agent_poses()
        for i, pose in enumerate(initial_poses):
            self.agent_paths[i].append(pose)
            if self._is_in_sample_region(pose):
                self.reachable_grid[self._discretize_position(pose)] = pose

        if self.sample_region and not any(self._is_in_sample_region(p) for p in initial_poses):
            print("Warning: no agents inside the specified sampling region at start.")

        pbar = tqdm.tqdm(range(steps), desc="Collecting reachable points", unit="step", ncols=100)
        for step in pbar:
            action = self.agents.act(keep_steps=1)
            n_agents = len(self.env.unwrapped.player_list)
            if not isinstance(action, (list, tuple, np.ndarray)):
                action = [action] * n_agents
            self.env.step(action)
            time.sleep(0.1)

            for i, pose in enumerate(self.get_all_agent_poses()):
                self.agent_paths[i].append(pose)
                if self._is_in_sample_region(pose):
                    self.reachable_grid[self._discretize_position(pose)] = pose

            if step % 100 == 0:
                print(f"Step {step}/{steps} — {len(self.reachable_grid)} reachable points")
                if render:
                    self.visualize(save=True, env_name=env_name,
                                   save_path=save_path, filename=f"coverage_map_{step}.png")

            if step % save_interval == 0 and step > 0:
                os.makedirs(save_path, exist_ok=True)
                self.save(os.path.join(save_path, f"{env_name}_{step}.pkl"))

        os.makedirs(save_path, exist_ok=True)
        pkl_path = os.path.join(save_path, f"{env_name}_reachable_points.pkl")
        self.save(pkl_path)
        return pkl_path

    def visualize(self, save=True, env_name="unknown", filename="coverage_map.png", save_path="."):
        if not self.reachable_grid:
            print("Warning: no reachable points to visualize yet")
            return

        plt.figure(figsize=(10, 10))

        if self.sample_region is not None:
            (x0, x1), (y0, y1) = self.sample_region
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  fill=False, edgecolor='red', linestyle='--',
                                  linewidth=2, label='Sampling region')
            plt.gca().add_patch(rect)
            inside = np.array([p for p in self.reachable_grid.values() if self._is_in_sample_region(p)])
            outside = np.array([p for p in self.reachable_grid.values() if not self._is_in_sample_region(p)])
            if len(inside):
                plt.scatter(inside[:, 0], inside[:, 1], c='green', s=2, alpha=0.7, label='Inside region')
            if len(outside):
                plt.scatter(outside[:, 0], outside[:, 1], c='blue', s=1, alpha=0.3, label='Outside region')
        else:
            pts = np.array(list(self.reachable_grid.values()))
            plt.scatter(pts[:, 0], pts[:, 1], c='blue', s=1, alpha=0.5)

        colors = ['r', 'g', 'm', 'y', 'c', 'b', 'k', 'orange']
        for i, path in enumerate(self.agent_paths):
            if path:
                arr = np.array(path)
                plt.plot(arr[:, 0], arr[:, 1],
                         color=colors[i % len(colors)], linewidth=0.5, alpha=0.3,
                         label=f"Agent {i}")

        plt.title(f"{len(self.reachable_grid)} reachable points — {env_name}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()

        if save:
            vis_path = os.path.join(save_path, "visualization")
            os.makedirs(vis_path, exist_ok=True)
            out = os.path.join(vis_path, filename)
            plt.savefig(out, dpi=300)
            print(f"Coverage map saved to {out}")
        else:
            plt.tight_layout()
            plt.draw()
            plt.waitforbuttonpress()
        plt.close()

    def save(self, filename):
        data = {
            'reachable_points': self.reachable_grid,
            'agent_paths': self.agent_paths,
            'resolution': self.resolution,
            'sample_region': self.sample_region,
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Reachable points saved to {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Connectivity graph construction  (originally make_graph.py)
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryGraphBuilder:
    """Loads a saved .pkl of reachable points and builds a NetworkX connectivity graph."""

    def __init__(self, pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)

        self.reachable_grid = data["reachable_points"]
        self.agent_paths = data["agent_paths"]
        self.resolution = data["resolution"]
        self.graph = None

        print(f"Loaded {len(self.reachable_grid)} reachable points "
              f"from {len(self.agent_paths)} agent trajectories")

    def _discretize_position(self, position):
        x, y, z = position
        return (round(x / self.resolution), round(y / self.resolution), round(z / self.resolution))

    def build_graph(self, distance_threshold=150.0, min_node_distance=100.0):
        """
        Build a connectivity graph from the loaded reachable points.

        Args:
            distance_threshold:  Max distance (UE units) for an edge to be added.
            min_node_distance:   Points closer than this are deduplicated (keep first).
        """
        G = nx.Graph()
        points_list = list(self.reachable_grid.items())
        print(f"Starting with {len(points_list)} candidate nodes...")

        # Deduplicate nodes that are too close to each other
        filtered_points = {}
        if points_list:
            g0, r0 = points_list[0]
            filtered_points[g0] = r0
        for grid_pos, real_pos in points_list[1:]:
            if not any(
                np.linalg.norm(np.array(real_pos) - np.array(ep)) < min_node_distance
                for ep in filtered_points.values()
            ):
                filtered_points[grid_pos] = real_pos

        print(f"Kept {len(filtered_points)} nodes after deduplication "
              f"(removed {len(points_list) - len(filtered_points)})")

        for grid_pos, real_pos in filtered_points.items():
            G.add_node(grid_pos, pos=real_pos)

        edges_added = set()

        # Edges from consecutive steps in agent trajectories
        for path in self.agent_paths:
            for i in range(len(path) - 1):
                p1 = self._discretize_position(path[i])
                p2 = self._discretize_position(path[i + 1])
                if p1 not in filtered_points or p2 not in filtered_points or p1 == p2:
                    continue
                edge_key = tuple(sorted([p1, p2]))
                if edge_key in edges_added:
                    continue
                dist = np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))
                if dist <= distance_threshold:
                    G.add_edge(p1, p2, weight=dist)
                    edges_added.add(edge_key)

        # Additional edges from spatial proximity
        nodes = list(filtered_points.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                edge_key = tuple(sorted([nodes[i], nodes[j]]))
                if edge_key in edges_added:
                    continue
                dist = np.linalg.norm(
                    np.array(filtered_points[nodes[i]]) - np.array(filtered_points[nodes[j]])
                )
                if dist <= distance_threshold:
                    G.add_edge(nodes[i], nodes[j], weight=dist)
                    edges_added.add(edge_key)

        self.graph = G
        print(f"Built graph: {len(G.nodes())} nodes, {len(G.edges())} edges")

        components = list(nx.connected_components(G))
        print(f"Connected components: {len(components)}")
        for i, comp in enumerate(sorted(components, key=len, reverse=True)[:5]):
            print(f"  Component {i + 1}: {len(comp)} nodes "
                  f"({100 * len(comp) / len(G.nodes()):.1f}%)")

        return G

    def visualize_graph(self, output_file="connectivity_graph.png", show_largest_component=True):
        if self.graph is None:
            print("Call build_graph() first")
            return

        plt.figure(figsize=(12, 12))
        pos = {n: (self.reachable_grid[n][0], self.reachable_grid[n][1]) for n in self.graph.nodes()}
        node_sizes = [1 + 3 * self.graph.degree(n) for n in self.graph.nodes()]

        nx.draw_networkx_nodes(self.graph, pos, node_size=node_sizes, node_color='skyblue', alpha=0.6)
        nx.draw_networkx_edges(self.graph, pos, width=0.3, edge_color='gray', alpha=0.3)

        if show_largest_component:
            largest_cc = max(nx.connected_components(self.graph), key=len)
            sub = self.graph.subgraph(largest_cc)
            sub_sizes = [1 + 3 * self.graph.degree(n) for n in sub.nodes()]
            nx.draw_networkx_nodes(sub, pos, node_size=sub_sizes, node_color='red', alpha=0.4)

        plt.title(f"Connectivity graph — {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to {output_file}")
        plt.close()

    def find_path(self, start_pos, end_pos):
        """Return the shortest path (as real-world coordinates) between two positions."""
        if self.graph is None:
            print("Call build_graph() first")
            return []

        start_grid = self._discretize_position(start_pos)
        end_grid = self._discretize_position(end_pos)

        if start_grid not in self.graph:
            start_grid = min(self.graph.nodes(),
                             key=lambda n: np.linalg.norm(np.array(self.reachable_grid[n]) - np.array(start_pos)))
            print(f"Start snapped to nearest node: {self.reachable_grid[start_grid]}")

        if end_grid not in self.graph:
            end_grid = min(self.graph.nodes(),
                           key=lambda n: np.linalg.norm(np.array(self.reachable_grid[n]) - np.array(end_pos)))
            print(f"End snapped to nearest node: {self.reachable_grid[end_grid]}")

        if not nx.has_path(self.graph, start_grid, end_grid):
            print("No path: start and end are in different connected components")
            return []

        path = nx.shortest_path(self.graph, start_grid, end_grid, weight='weight')
        total = sum(self.graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
        print(f"Path found: {len(path)} nodes, total length {total:.2f}")
        return [self.reachable_grid[p] for p in path]

    def analyze_graph(self, env_name, save_path="."):
        """Print graph statistics and save the graph as a .gpickle file."""
        if self.graph is None:
            print("Call build_graph() first")
            return

        print("\n===== Graph Analysis =====")
        print(f"Nodes: {len(self.graph.nodes())},  Edges: {len(self.graph.edges())}")

        components = list(nx.connected_components(self.graph))
        largest_cc = max(components, key=len)
        print(f"Connected components: {len(components)}")
        print(f"Largest component: {len(largest_cc)} nodes "
              f"({100 * len(largest_cc) / len(self.graph.nodes()):.1f}%)")

        diameter = nx.diameter(self.graph.subgraph(largest_cc))
        print(f"Diameter of largest component: {diameter}")

        print("\nTop-5 nodes by degree centrality:")
        for rank, (node, c) in enumerate(
            sorted(nx.degree_centrality(self.graph).items(), key=lambda x: x[1], reverse=True)[:5], 1
        ):
            print(f"  {rank}. {node}  centrality={c:.4f}  coords={self.reachable_grid[node]}")

        os.makedirs(save_path, exist_ok=True)
        gpickle_path = os.path.join(save_path, f"{env_name}_graph.gpickle")
        with open(gpickle_path, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"\nGraph saved to {gpickle_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: collect reachable points then build a nav graph."
    )

    # Environment
    parser.add_argument("-e", "--env_name", default="SuburbNeighborhood_Day",
                        choices=["FlexibleRoom", "SuburbNeighborhood_Day",
                                 "Forglar_Map", "HongKongStreet"],
                        help="Rescue environment map name")

    # Phase 1 — collection
    parser.add_argument("--max_step", type=int, default=3000,
                        help="Exploration steps for point collection")
    parser.add_argument("-n", "--num_agents", type=int, default=5,
                        help="Number of parallel exploration agents")
    parser.add_argument("--grid_resolution", type=float, default=100.0,
                        help="Grid cell size (UE units) for position discretisation")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="Save intermediate .pkl every N steps")
    parser.add_argument("-t", "--time_dilation", type=int, default=-1,
                        help="UE time-dilation factor to boost FPS (disabled if -1)")
    parser.add_argument("--no_render", action="store_true",
                        help="Disable per-step coverage map rendering during collection")

    # Skip phase 1
    parser.add_argument("--skip_collection", action="store_true",
                        help="Skip collection and load an existing .pkl instead")
    parser.add_argument("--pkl_path", default=None,
                        help="Path to existing .pkl (required with --skip_collection)")

    # Phase 2 — graph construction
    parser.add_argument("--distance_threshold", type=float, default=200.0,
                        help="Max edge distance for graph connectivity (UE units)")
    parser.add_argument("--min_node_distance", type=float, default=50.0,
                        help="Min distance between graph nodes; closer nodes are merged")

    # Output
    parser.add_argument("--save_path", default=None,
                        help="Root output directory "
                             "(default: example/agent_configs_sampler/points_graph/<env_name>)")

    args = parser.parse_args()
    env_name = args.env_name

    if args.save_path is None:
        args.save_path = os.path.join(
            os.path.dirname(__file__), "agent_configs_sampler", "points_graph", env_name
        )

    # ── Phase 1: collect reachable points ──────────────────────────────────
    if args.skip_collection:
        if not args.pkl_path:
            parser.error("--pkl_path is required when using --skip_collection")
        pkl_path = args.pkl_path
        print(f"Skipping collection — loading {pkl_path}")
    else:
        print(f"\n{'=' * 60}")
        print(f"Phase 1: Collecting reachable points for '{env_name}'")
        print(f"{'=' * 60}")

        env = gym.make(f"UnrealBase-{env_name}", disable_env_checker=True)
        if args.time_dilation > 0:
            env = time_dilation.TimeDilationWrapper(env, args.time_dilation)

        sample_region = getattr(env.unwrapped, 'sample_region', None)
        collector = ReachablePointsCollector(
            env,
            num_agents=args.num_agents,
            grid_resolution=args.grid_resolution,
            sample_region=sample_region,
        )

        try:
            pkl_path = collector.collect(
                env_name=env_name,
                steps=args.max_step,
                render=not args.no_render,
                save_interval=args.save_interval,
                save_path=args.save_path,
            )
            collector.visualize(
                save=True, env_name=env_name, save_path=args.save_path,
                filename="coverage_map_final.png",
            )
        except KeyboardInterrupt:
            print("Interrupted — saving current state...")
            pkl_path = os.path.join(args.save_path, f"{env_name}_reachable_points_interrupted.pkl")
            collector.save(pkl_path)
        finally:
            env.close()

    # ── Phase 2: build connectivity graph ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Phase 2: Building connectivity graph for '{env_name}'")
    print(f"{'=' * 60}")

    builder = TrajectoryGraphBuilder(pkl_path)
    builder.build_graph(
        distance_threshold=args.distance_threshold,
        min_node_distance=args.min_node_distance,
    )
    builder.visualize_graph(
        output_file=os.path.join(args.save_path, f"{env_name}_connectivity_graph.png")
    )
    builder.analyze_graph(env_name=env_name, save_path=args.save_path)

    print(f"\nDone. All outputs written to: {args.save_path}")
