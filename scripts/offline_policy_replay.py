#!/usr/bin/env python3
"""Replay real-robot log observations through a merged DreamWaQ policy_full.pt.

The log motor lines contain both the physical motor index and the mapped policy
index. This script trusts "映射索引" and writes q/dq/tau_est into policy order.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import yaml


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
QUAT_RE = re.compile(
    rf"\bw:\s*({FLOAT})\s+x:\s*({FLOAT})\s+y:\s*({FLOAT})\s+z:\s*({FLOAT})"
)
XYZ_RE = re.compile(rf"\bx:\s*({FLOAT})\s+y:\s*({FLOAT})\s+z:\s*({FLOAT})")
MOTOR_RE = re.compile(
    rf"电机\[(\d+)\]\s+映射索引:\s*(\d+)\s+位置q:\s*({FLOAT})\s+速度dq:\s*({FLOAT})\s+扭矩tauEst:\s*({FLOAT})"
)
RL_CMD_RE = re.compile(rf"RL Controller x:\s*({FLOAT})\s+y:\s*({FLOAT})\s+yaw:\s*({FLOAT})")
RL_TARGET_RE = re.compile(r"RL ControllerResult \| 12个电机位置\(q\)：(.+)")


def quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 0.0:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_log(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    xyz_seen = 0
    cmd = np.zeros(3, dtype=np.float32)
    pending_target: np.ndarray | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            cmd_match = RL_CMD_RE.search(line)
            if cmd_match:
                cmd = np.array([float(cmd_match.group(i)) for i in (1, 2, 3)], dtype=np.float32)

            target_match = RL_TARGET_RE.search(line)
            if target_match:
                vals = [float(x) for x in re.findall(FLOAT, target_match.group(1))]
                if len(vals) == 12:
                    pending_target = np.asarray(vals, dtype=np.float32)

            quat_match = QUAT_RE.search(line)
            if quat_match:
                current = {
                    "line": lineno,
                    "quat_wxyz": np.array([float(quat_match.group(i)) for i in range(1, 5)], dtype=np.float32),
                    "gyro": None,
                    "acc": None,
                    "q": np.full(12, np.nan, dtype=np.float32),
                    "dq": np.full(12, np.nan, dtype=np.float32),
                    "tau_est": np.full(12, np.nan, dtype=np.float32),
                    "target_q": pending_target.copy() if pending_target is not None else None,
                    "motor_to_policy": np.full(12, -1, dtype=np.int32),
                    "cmd": cmd.copy(),
                }
                pending_target = None
                xyz_seen = 0
                continue

            if current is None:
                continue

            motor_match = MOTOR_RE.search(line)
            if motor_match:
                motor_idx = int(motor_match.group(1))
                policy_idx = int(motor_match.group(2))
                if 0 <= motor_idx < 12 and 0 <= policy_idx < 12:
                    current["motor_to_policy"][motor_idx] = policy_idx
                    current["q"][policy_idx] = float(motor_match.group(3))
                    current["dq"][policy_idx] = float(motor_match.group(4))
                    current["tau_est"][policy_idx] = float(motor_match.group(5))
                if np.all(np.isfinite(current["q"])) and np.all(np.isfinite(current["dq"])):
                    frames.append(current)
                    current = None
                continue

            xyz_match = XYZ_RE.search(line)
            if xyz_match and not QUAT_RE.search(line):
                vec = np.array([float(xyz_match.group(i)) for i in range(1, 4)], dtype=np.float32)
                if xyz_seen == 0:
                    current["gyro"] = vec
                    xyz_seen += 1
                elif xyz_seen == 1:
                    current["acc"] = vec
                    xyz_seen += 1

    return [f for f in frames if f["gyro"] is not None]


def build_frame(
    parsed: dict[str, Any],
    q_default: np.ndarray,
    prev_action: np.ndarray,
    command_override: np.ndarray | None,
    gravity_source: str,
) -> dict[str, np.ndarray]:
    gyro = parsed["gyro"].astype(np.float32) * np.float32(0.25)
    if gravity_source == "quat":
        rot_bw = quat_to_rotmat_wxyz(parsed["quat_wxyz"]).T
        projected_gravity = rot_bw @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        acc = parsed["acc"].astype(np.float32)
        norm = float(np.linalg.norm(acc))
        projected_gravity = -acc / norm if norm > 1e-6 else np.array([0.0, 0.0, -1.0], dtype=np.float32)
    cmd = command_override if command_override is not None else parsed["cmd"].astype(np.float32)
    return {
        "base_ang_vel": gyro.astype(np.float32),
        "projected_gravity": projected_gravity.astype(np.float32),
        "velocity_commands": cmd.astype(np.float32),
        "joint_pos": (parsed["q"] - q_default).astype(np.float32),
        "joint_vel": (parsed["dq"] * np.float32(0.05)).astype(np.float32),
        "actions": prev_action.astype(np.float32),
    }


def flatten_history(history: dict[str, deque[np.ndarray]]) -> np.ndarray:
    return np.concatenate(
        [
            np.concatenate(list(history["base_ang_vel"]), axis=0),
            np.concatenate(list(history["projected_gravity"]), axis=0),
            np.concatenate(list(history["velocity_commands"]), axis=0),
            np.concatenate(list(history["joint_pos"]), axis=0),
            np.concatenate(list(history["joint_vel"]), axis=0),
            np.concatenate(list(history["actions"]), axis=0),
        ],
        axis=0,
    ).astype(np.float32)


def flatten_current(frame: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            frame["base_ang_vel"],
            frame["projected_gravity"],
            frame["velocity_commands"],
            frame["joint_pos"],
            frame["joint_vel"],
            frame["actions"],
        ],
        axis=0,
    ).astype(np.float32)


def build_row(
    *,
    frame_idx: int,
    parsed: dict[str, Any],
    obs_frame: dict[str, np.ndarray],
    joint_order: list[str],
    action: np.ndarray,
    q_target: np.ndarray,
    tau_policy: np.ndarray,
    target_q: np.ndarray | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame": frame_idx,
        "line": parsed["line"],
        "cmd_x": float(obs_frame["velocity_commands"][0]),
        "cmd_y": float(obs_frame["velocity_commands"][1]),
        "cmd_yaw": float(obs_frame["velocity_commands"][2]),
    }
    for i, joint_name in enumerate(joint_order):
        row[f"q/{joint_name}"] = float(parsed["q"][i])
        row[f"dq/{joint_name}"] = float(parsed["dq"][i])
        row[f"tau_est/{joint_name}"] = float(parsed["tau_est"][i])
        row[f"action/{joint_name}"] = float(action[i])
        row[f"q_target/{joint_name}"] = float(q_target[i])
        row[f"tau_policy/{joint_name}"] = float(tau_policy[i])
        row[f"target_log/{joint_name}"] = float(target_q[i]) if target_q is not None else ""
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("sim2real_config.yaml"))
    parser.add_argument("--policy", type=Path, default=Path("policy_full.pt"))
    parser.add_argument("--log", type=Path, default=Path("Rl-log.txt"))
    parser.add_argument("--out", type=Path, default=Path("offline_policy_runing.csv"))
    parser.add_argument("--start-line", type=int, default=214805, help="Only replay frames whose first line is >= this.")
    parser.add_argument("--stop-line", type=int, default=None, help="Only replay frames whose first line is <= this.")
    parser.add_argument("--start-frame", type=int, default=0, help="Skip this many parsed frames.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--cmd", nargs=3, type=float, default=None, metavar=("VX", "VY", "WZ"))
    parser.add_argument("--calf-default", type=float, default=None, help="Override all four calf q_default values.")
    parser.add_argument(
        "--target-order",
        choices=("policy", "motor"),
        default="policy",
        help="Order of RL ControllerResult target positions. Use motor if printed in real motor order.",
    )
    parser.add_argument("--gravity-source", choices=("quat", "acc"), default="quat")
    parser.add_argument("--dry-parse", action="store_true", help="Only parse log and print mapping/ranges.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    control = cfg["control_contract"]
    obs = cfg["observation_contract"]
    q_default = np.asarray(control["q_default"], dtype=np.float32)
    if args.calf_default is not None:
        q_default = q_default.copy()
        q_default[[2, 5, 8, 11]] = np.float32(args.calf_default)
    action_scale = np.asarray(control["action_scale"], dtype=np.float32)
    kp = np.asarray(control["kp"], dtype=np.float32)
    kd = np.asarray(control["kd"], dtype=np.float32)
    tau_limit = np.asarray(control["tau_limit"], dtype=np.float32)
    history_len = int(obs["history_len"])
    action_clip = control.get("action_clip_before_pd", [-10.0, 10.0])

    all_frames = parse_log(args.log)
    if not all_frames:
        raise RuntimeError(f"No complete frames parsed from {args.log}")
    frames = all_frames
    if args.start_line is not None:
        frames = [f for f in frames if int(f["line"]) >= args.start_line]
    if args.stop_line is not None:
        frames = [f for f in frames if int(f["line"]) <= args.stop_line]
    frames = frames[args.start_frame :]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    print(f"[OK] parsed_frames_total={len(all_frames)} selected_frames={len(frames)}")
    print(f"[OK] first_line={frames[0]['line']} first_motor_to_policy={frames[0]['motor_to_policy'].tolist()}")
    print(f"[OK] q_default={q_default.tolist()}")
    if args.dry_parse:
        return 0

    import torch

    command_override = np.asarray(args.cmd, dtype=np.float32) if args.cmd is not None else None
    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()

    history: dict[str, deque[np.ndarray]] = {
        name: deque(maxlen=history_len)
        for name in ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")
    }
    prev_action = np.zeros(12, dtype=np.float32)
    rows: list[dict[str, Any]] = []

    with torch.inference_mode():
        for frame_idx, parsed in enumerate(frames):
            obs_frame = build_frame(parsed, q_default, prev_action, command_override, args.gravity_source)
            for name, value in obs_frame.items():
                while len(history[name]) < history_len:
                    history[name].append(value.copy())
                history[name].append(value.copy())

            hist_np = flatten_history(history)
            cur_np = flatten_current(obs_frame)
            action = policy(torch.from_numpy(hist_np[None, :]), torch.from_numpy(cur_np[None, :])).cpu().numpy()[0]
            action = np.clip(action.astype(np.float32), float(action_clip[0]), float(action_clip[1]))
            q_target = q_default + action_scale * action
            tau_raw = kp * (q_target - parsed["q"]) + kd * (0.0 - parsed["dq"])
            tau_policy = np.clip(tau_raw, -tau_limit, tau_limit)
            target_q = parsed.get("target_q")
            if target_q is not None:
                target_q = target_q.astype(np.float32)
                if args.target_order == "motor":
                    target_q_policy = np.full(12, np.nan, dtype=np.float32)
                    for motor_idx, policy_idx in enumerate(parsed["motor_to_policy"]):
                        if 0 <= int(policy_idx) < 12:
                            target_q_policy[int(policy_idx)] = target_q[motor_idx]
                    target_q = target_q_policy
            else:
                prev_action = action.copy()
                continue

            row = build_row(
                frame_idx=frame_idx,
                parsed=parsed,
                obs_frame=obs_frame,
                joint_order=control["joint_order"],
                action=action,
                q_target=q_target,
                tau_policy=tau_policy,
                target_q=target_q,
            )
            rows.append(row)
            prev_action = action.copy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No frames with clean RL ControllerResult target were found.")
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
