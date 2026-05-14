#!/usr/bin/env python3
"""DreamWaQ sim2sim viewer: CENet checkpoint + actor ONNX (see deploy_mujoco.sim2sim_core_dreamwaq)."""
from __future__ import annotations

import argparse

import mujoco.viewer
import numpy as np

from deploy_mujoco.sim2sim_core_dreamwaq import DreamWaQSim2SimCfg, DreamWaQSim2SimRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dolanga1 DreamWaQ sim2sim (MuJoCo viewer).")
    parser.add_argument("--load_model", type=str, required=True, help="Path to MuJoCo scene xml.")
    parser.add_argument("--policy", type=str, required=True, help="Path to actor policy.onnx (MLP tail).")
    parser.add_argument(
        "--cenet",
        type=str,
        required=True,
        help="exported/cenet.pt (from play.py) or model_*.pt (full RSL-RL ckpt).",
    )
    parser.add_argument("--sim_duration", type=float, default=120.0)
    parser.add_argument("--cmd_x", type=float, default=1.5, help="Commanded forward velocity.")
    parser.add_argument("--cmd_y", type=float, default=0.0, help="Commanded lateral velocity.")
    parser.add_argument("--cmd_yaw", type=float, default=0.0, help="Commanded yaw velocity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DreamWaQSim2SimCfg(
        mujoco_model_path=args.load_model,
        actor_onnx_path=args.policy,
        cenet_ckpt_path=args.cenet,
        sim_duration=args.sim_duration,
        cmd=np.array([args.cmd_x, args.cmd_y, args.cmd_yaw], dtype=np.float32),
    )
    runner = DreamWaQSim2SimRunner(cfg)
    sim_steps = int(cfg.sim_duration / cfg.dt)

    with mujoco.viewer.launch_passive(runner.model, runner.data) as viewer:
        viewer.cam.distance = 5.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 120
        viewer.cam.lookat[:] = runner.data.qpos[:3]
        for step in range(sim_steps):
            runner.step(step)
            viewer.sync()


if __name__ == "__main__":
    main()
