#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy_mujoco.sim2sim_core import Sim2SimCfg, Sim2SimRunner
from deploy_mujoco.sim2sim_obs_corruption import add_sim2sim_perturb_cli, sim2sim_perturb_cfg_from_ns
from deploy_mujoco_viewer.keyboard_control import KeyboardCommandController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dolanga1 sim2sim runner (MuJoCo viewer).")
    parser.add_argument("--load_model", type=str, required=True, help="Path to MuJoCo scene xml.")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy.onnx.")
    parser.add_argument("--sim_duration", type=float, default=120.0)
    parser.add_argument("--cmd_x", type=float, default=2.5, help="Commanded forward velocity.")
    parser.add_argument("--cmd_y", type=float, default=0.0, help="Commanded lateral velocity.")
    parser.add_argument("--cmd_yaw", type=float, default=0.4, help="Commanded yaw velocity.")
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Start from zero command and drive cmd_x/cmd_y/cmd_yaw with keyboard.",
    )
    add_sim2sim_perturb_cli(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    startup_cmd = np.zeros(3, dtype=np.float32) if args.keyboard else np.array(
        [args.cmd_x, args.cmd_y, args.cmd_yaw], dtype=np.float32
    )
    cfg = Sim2SimCfg(
        mujoco_model_path=args.load_model,
        onnx_path=args.policy,
        sim_duration=args.sim_duration,
        cmd=startup_cmd,
        perturb=sim2sim_perturb_cfg_from_ns(args),
    )
    runner = Sim2SimRunner(cfg)
    sim_steps = int(cfg.sim_duration / cfg.dt)
    keyboard = KeyboardCommandController(cfg.cmd) if args.keyboard else None

    launch_kwargs = {"key_callback": keyboard.key_callback} if keyboard is not None else {}
    with mujoco.viewer.launch_passive(runner.model, runner.data, **launch_kwargs) as viewer:
        if keyboard is not None:
            keyboard.print_help()
        viewer.cam.distance = 5.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 120
        viewer.cam.lookat[:] = runner.data.qpos[:3]
        for step in range(sim_steps):
            if not viewer.is_running():
                break
            if keyboard is not None:
                runner.cmd[:] = keyboard.current_cmd()
            runner.step(step)
            viewer.cam.lookat[:] = runner.data.qpos[:3]
            if keyboard is not None:
                keyboard.set_overlay(viewer)
            viewer.sync()


if __name__ == "__main__":
    main()
