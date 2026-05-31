#!/usr/bin/env python3
r"""用法:
cd /media/littlecave/T9/Offline_RL_Active_Tracking/gym-rescue && \
for L in  1 2 3 4; do \
  case "$L" in \
    1) E="UnrealRescue-SuburbNeighborhood_Day" ;; \
    2|3|4) E="UnrealRescue-Forglar_Map" ;; \
  esac; \
  echo "=== level ${L} env ${E} ==="; \
  python3 benchmark/check_train_test_injured_separation.py \
    --train-dir /media/littlecave/T9/UnrealEnv/AutoNav_RescueTrajectories \
    --glob '**/*.pt' \
    --test-jsonl "/media/littlecave/T9/Offline_RL_Active_Tracking/gym-rescue/gym_rescue/envs/setting/test_jsonl/level_${L}.jsonl" \
    --level "${L}" \
    --env-id "${E}"; \
done
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_test_positions(
    path: Path, env_id: str | None, level: int | None
) -> tuple[list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]]]:
    injured: list[tuple[int, np.ndarray]] = []
    agents: list[tuple[int, np.ndarray]] = []
    for n, line in enumerate(path.open(encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        if env_id and o.get("env_id") != env_id:
            continue
        if level is not None and o.get("level") != level:
            continue
        il = o.get("injured_player_loc") or []
        if len(il) >= 3:
            injured.append((n, np.asarray(il[:3], dtype=np.float64)))
        al = o.get("agent_loc") or []
        if len(al) >= 3:
            agents.append((n, np.asarray(al[:3], dtype=np.float64)))
    return injured, agents


def pickup_points(traj: list[dict[str, Any]]) -> list[tuple[int, np.ndarray]]:
    out: list[tuple[int, np.ndarray]] = []
    prev = False
    for r in traj:
        p = bool(r.get("picked"))
        if p and not prev:
            out.append(
                (
                    int(r["step"]),
                    np.array([float(r["x"]), float(r["y"]), float(r["z"])], dtype=np.float64),
                )
            )
        prev = p
    return out


def traj_start_xyz(traj: list[dict[str, Any]]) -> tuple[int, np.ndarray]:
    r0 = traj[0]
    return int(r0["step"]), np.array([float(r0["x"]), float(r0["y"]), float(r0["z"])], dtype=np.float64)


def nearest_violation(
    pos: np.ndarray, xyz: np.ndarray, lines: np.ndarray, radius: float
) -> tuple[int, np.ndarray, float] | None:
    dist = np.linalg.norm(xyz - pos, axis=1)
    j = int(np.argmin(dist))
    if dist[j] < radius:
        return int(lines[j]), xyz[j], float(dist[j])
    return None


def best_pickup_violation(
    traj: list[dict[str, Any]],
    inj_xyz: np.ndarray,
    inj_line: np.ndarray,
    radius: float,
) -> tuple[int, np.ndarray, int, np.ndarray, float] | None:
    """在「距测试伤者 < 半径」的抱起帧里取距离最小的一条; 无则 None。"""
    best: tuple[float, int, np.ndarray, int, np.ndarray] | None = None
    for step, pos in pickup_points(traj):
        nv = nearest_violation(pos, inj_xyz, inj_line, radius)
        if nv is None:
            continue
        ln, ixyz, d = nv
        if best is None or d < best[0]:
            best = (d, step, pos, ln, ixyz)
    if best is None:
        return None
    _, step, pos, ln, ixyz = best
    return step, pos, ln, ixyz, best[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-jsonl", type=Path, default=_root() / "gym_rescue/envs/setting/test_jsonl/level_1.jsonl")
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--glob", default="*.pt")
    ap.add_argument("--radius", type=float, default=100.0)
    ap.add_argument("--env-id", default="UnrealRescue-SuburbNeighborhood_Day")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    env = a.env_id.strip() or None
    level = None if a.level < 0 else a.level
    inj_rows, ag_rows = load_test_positions(a.test_jsonl, env, level)
    if not inj_rows or not ag_rows:
        print(
            f"组合检查需要 JSONL 中同时存在 injured_player_loc 与 agent_loc: {a.test_jsonl}",
            file=sys.stderr,
        )
        return 2
    inj_line = np.array([r[0] for r in inj_rows], dtype=np.int64)
    inj_xyz = np.stack([r[1] for r in inj_rows])
    ag_line = np.array([r[0] for r in ag_rows], dtype=np.int64)
    ag_xyz = np.stack([r[1] for r in ag_rows])
    files = sorted(a.train_dir.glob(a.glob))
    if a.max_files > 0:
        files = files[: a.max_files]
    if not files:
        print(f"无匹配文件: {a.train_dir} {a.glob!r}", file=sys.stderr)
        return 2
    viol: list[
        tuple[Path, int | None, int, np.ndarray, int, np.ndarray, float, int, np.ndarray, int, np.ndarray, float]
    ] = []
    for i, pt in enumerate(files):
        if not a.quiet and (i == 0 or (i + 1) % 10 == 0):
            print(f"  [{i + 1}/{len(files)}] {pt.name}", flush=True)
        d = torch.load(pt, map_location="cpu", weights_only=False)
        traj = d["trajectory"]
        eid = d.get("episode_id")
        eid = int(eid) if eid is not None else None
        pv = best_pickup_violation(traj, inj_xyz, inj_line, a.radius)
        step0, start = traj_start_xyz(traj)
        sv = nearest_violation(start, ag_xyz, ag_line, a.radius)
        if pv is None or sv is None:
            continue
        ag_ln, ag_xyz1, ag_d = sv
        pk_step, pk_pos, inj_ln, inj_xyz1, inj_d = pv
        viol.append((pt, eid, step0, start, ag_ln, ag_xyz1, ag_d, pk_step, pk_pos, inj_ln, inj_xyz1, inj_d))
    if viol:
        print(f"\n{len(viol)} 处违规（首帧与抱起瞬间**同时**距测试点 < {a.radius}）:\n")
        for pt, eid, s0, st, aln, axy, ad, pk, pp, iln, ixy, id_ in viol:
            print(
                f"  {pt.name}  ep={eid}\n"
                f"    首帧 step={s0} agent=({st[0]:.4f}, {st[1]:.4f}, {st[2]:.4f})\n"
                f"      → test 行 {aln} agent_loc=({axy[0]:.4f}, {axy[1]:.4f}, {axy[2]:.4f})  dist={ad:.4f}\n"
                f"    抱起 step={pk} agent=({pp[0]:.4f}, {pp[1]:.4f}, {pp[2]:.4f})\n"
                f"      → test 行 {iln} injured=({ixy[0]:.4f}, {ixy[1]:.4f}, {ixy[2]:.4f})  dist={id_:.4f}\n"
            )
        return 1
    if not a.quiet:
        print(f"\n无违规（{len(files)} 个 .pt）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
