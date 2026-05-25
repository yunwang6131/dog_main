#!/usr/bin/env python3
from __future__ import annotations

import argparse

import mujoco
import mujoco.viewer
import numpy as np

from deploy_mujoco.sim2sim_core import Sim2SimCfg, Sim2SimRunner, quat_to_rotmat_wxyz
from mujoco.glfw import glfw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dolanga1 sim2sim runner (MuJoCo viewer).")
    parser.add_argument("--load_model", type=str, required=True, help="Path to MuJoCo scene xml.")
    parser.add_argument("--policy", type=str, required=True, help="Path to exported policy_full.pt.")
    parser.add_argument("--sim_duration", type=float, default=120.0)
    parser.add_argument("--cmd_x", type=float, default=1.0, help="Commanded forward velocity.")
    parser.add_argument("--cmd_y", type=float, default=0.0, help="Commanded lateral velocity.")
    parser.add_argument("--cmd_yaw", type=float, default=0.0, help="Commanded yaw velocity.")
    parser.add_argument("--hide_velocity_vis", action="store_true", help="Hide command/actual velocity arrows.")
    parser.add_argument("--velocity_arrow_scale", type=float, default=0.6, help="Arrow length scale for m/s.")
    return parser.parse_args()


def _add_arrow(scene: mujoco.MjvScene, start: np.ndarray, vec: np.ndarray, rgba: np.ndarray, scale: float) -> None:
    if scene.ngeom >= scene.maxgeom:
        return

    length = float(np.linalg.norm(vec[:2]))
    if length < 1.0e-4:
        return

    geom = scene.geoms[scene.ngeom]
    end = start + scale * vec
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba.astype(np.float32),
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.035, start.astype(np.float64), end.astype(np.float64))
    scene.ngeom += 1


def _draw_velocity_arrows(viewer, runner: Sim2SimRunner, scale: float) -> None:
    scene = viewer.user_scn
    scene.ngeom = 0

    base_pos_w = runner.data.qpos[:3].astype(np.float32)
    arrow_origin = base_pos_w + np.array([0.0, 0.0, 0.35], dtype=np.float32)

    base_quat_wxyz = runner.data.qpos[3:7].astype(np.float32)
    rot_wb = quat_to_rotmat_wxyz(base_quat_wxyz)
    cmd_lin_b = np.array([runner.cmd[0], runner.cmd[1], 0.0], dtype=np.float32)
    cmd_lin_w = rot_wb @ cmd_lin_b
    actual_lin_w = runner.data.qvel[:3].astype(np.float32)
    actual_lin_w[2] = 0.0

    _add_arrow(
        scene,
        arrow_origin + np.array([0.0, 0.0, 0.06], dtype=np.float32),
        cmd_lin_w,
        np.array([0.1, 0.9, 0.2, 0.85], dtype=np.float32),
        scale,
    )
    _add_arrow(
        scene,
        arrow_origin,
        actual_lin_w,
        np.array([0.1, 0.35, 1.0, 0.85], dtype=np.float32),
        scale,
    )


def main() -> None:
    args = parse_args()
    cfg = Sim2SimCfg(
        mujoco_model_path=args.load_model,
        onnx_path=args.policy,
        sim_duration=args.sim_duration,
        cmd=np.array([args.cmd_x, args.cmd_y, args.cmd_yaw], dtype=np.float32),
    )
    runner = Sim2SimRunner(cfg)
    sim_steps = int(cfg.sim_duration / cfg.dt)

    with mujoco.viewer.launch_passive(runner.model, runner.data) as viewer:
        viewer.cam.distance = 5.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 120
        for step in range(sim_steps):
            runner.step(step)
            if not args.hide_velocity_vis:
                _draw_velocity_arrows(viewer, runner, args.velocity_arrow_scale)
            viewer.cam.lookat[:] = runner.data.qpos[:3]
            viewer.sync()


if __name__ == "__main__":
    main()