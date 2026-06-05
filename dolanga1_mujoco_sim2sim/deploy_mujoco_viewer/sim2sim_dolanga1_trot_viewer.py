#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np
from mujoco.glfw import glfw

from deploy_mujoco.sim2sim_core import Sim2SimRunner, quat_to_rotmat_wxyz, resolve_sim2sim_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dolanga1 sim2sim runner with GLFW mouse force.")
    parser.add_argument("--config", type=str, help="Path to sim2sim yaml config.")
    parser.add_argument("--load_model", type=str)
    parser.add_argument("--policy", type=str)
    parser.add_argument("--sim_duration", type=float)

    parser.add_argument("--cmd_x", type=float)
    parser.add_argument("--cmd_y", type=float)
    parser.add_argument("--cmd_yaw", type=float)
    parser.add_argument("--hide_velocity_vis", action="store_true")
    parser.add_argument("--velocity_arrow_scale", type=float, default=0.6)

    parser.add_argument("--mouse_force_body", type=str, default="base")
    parser.add_argument("--mouse_force_gain", type=float, default=40.0)
    parser.add_argument("--mouse_force_max", type=float, default=200.0)
    parser.add_argument("--mouse_force_decay", type=float, default=0.95)

    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def _add_arrow(scene, start, vec, rgba, scale, width=0.035) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    if float(np.linalg.norm(vec)) < 1e-5:
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

    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        width,
        start.astype(np.float64),
        end.astype(np.float64),
    )
    scene.ngeom += 1


def _draw_velocity_arrows(scene, runner: Sim2SimRunner, scale: float) -> None:
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


class MouseForce:
    def __init__(
        self,
        runner: Sim2SimRunner,
        body_name: str,
        gain: float,
        max_force: float,
        decay: float,
        cam: mujoco.MjvCamera,
    ):
        self.runner = runner
        self.gain = gain
        self.max_force = max_force
        self.decay = decay
        self.cam = cam

        self.body_id = mujoco.mj_name2id(
            runner.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        if self.body_id < 0:
            raise RuntimeError(f"Body not found: {body_name}")

        self.dragging_force = False
        self.force_w = np.zeros(3, dtype=np.float64)

    def _camera_screen_axes(self) -> tuple[np.ndarray, np.ndarray]:
        az = np.deg2rad(self.cam.azimuth)
        el = np.deg2rad(self.cam.elevation)

        forward = np.array(
            [
                np.cos(el) * np.cos(az),
                np.cos(el) * np.sin(az),
                np.sin(el),
            ],
            dtype=np.float64,
        )
        forward /= np.linalg.norm(forward)

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            right /= np.linalg.norm(right)

        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        return right, up

    def add_from_mouse_delta(self, dx: float, dy: float) -> None:
        right, up = self._camera_screen_axes()
        self.force_w += self.gain * (dx * right - dy * up)

        norm = float(np.linalg.norm(self.force_w))
        if norm > self.max_force:
            self.force_w *= self.max_force / norm

    def clear(self) -> None:
        self.force_w[:] = 0.0

    def apply(self) -> None:
        self.runner.data.xfrc_applied[:, :] = 0.0
        self.runner.data.xfrc_applied[self.body_id, :3] = self.force_w
        self.force_w *= self.decay

    def draw(self, scene: mujoco.MjvScene) -> None:
        if float(np.linalg.norm(self.force_w)) < 1e-3:
            return

        base_pos = self.runner.data.xpos[self.body_id].astype(np.float32)
        start = base_pos + np.array([0.0, 0.0, 0.50], dtype=np.float32)
        vec = (self.force_w / 80.0).astype(np.float32)

        _add_arrow(
            scene,
            start,
            vec,
            np.array([1.0, 0.15, 0.05, 0.9], dtype=np.float32),
            scale=1.0,
            width=0.045,
        )


class MouseCameraController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        cam: mujoco.MjvCamera,
        scene: mujoco.MjvScene,
        mouse_force: MouseForce,
    ):
        self.model = model
        self.data = data
        self.cam = cam
        self.scene = scene
        self.mouse_force = mouse_force

        self.button_left = False
        self.button_middle = False
        self.button_right = False

        self.last_x = 0.0
        self.last_y = 0.0

    def mouse_button_callback(self, window, button, action, mods) -> None:
        self.button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

        self.last_x, self.last_y = glfw.get_cursor_pos(window)

        shift_pressed = bool(mods & glfw.MOD_SHIFT)

        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS and shift_pressed:
            self.mouse_force.dragging_force = True

        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
            self.mouse_force.dragging_force = False
            self.mouse_force.clear()

    def cursor_pos_callback(self, window, xpos, ypos) -> None:
        dx = xpos - self.last_x
        dy = ypos - self.last_y
        self.last_x = xpos
        self.last_y = ypos

        width, height = glfw.get_window_size(window)
        if width <= 0 or height <= 0:
            return

        if self.mouse_force.dragging_force:
            self.mouse_force.add_from_mouse_delta(dx, dy)
            return

        if not (self.button_left or self.button_middle or self.button_right):
            return

        if self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H
        elif self.button_middle:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H

        mujoco.mjv_moveCamera(
            self.model,
            action,
            dx / height,
            dy / height,
            self.scene,
            self.cam,
        )

    def scroll_callback(self, window, xoffset, yoffset) -> None:
        mujoco.mjv_moveCamera(
            self.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.scene,
            self.cam,
        )


def main() -> None:
    args = parse_args()

    cfg = resolve_sim2sim_cfg(
        args.config,
        args.load_model,
        args.policy,
        args.sim_duration,
        args.cmd_x,
        args.cmd_y,
        args.cmd_yaw,
    )

    runner = Sim2SimRunner(cfg)
    sim_steps = int(cfg.sim_duration / cfg.dt)

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW.")

    window = glfw.create_window(
        args.width,
        args.height,
        "Dolanga1 MuJoCo Mouse Force Viewer",
        None,
        None,
    )
    if window is None:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(runner.model, maxgeom=10000)
    context = mujoco.MjrContext(runner.model, mujoco.mjtFontScale.mjFONTSCALE_150)

    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)

    cam.distance = 5.0
    cam.elevation = -20
    cam.azimuth = 120

    mouse_force = MouseForce(
        runner=runner,
        body_name=args.mouse_force_body,
        gain=args.mouse_force_gain,
        max_force=args.mouse_force_max,
        decay=args.mouse_force_decay,
        cam=cam,
    )

    mouse_controller = MouseCameraController(
        model=runner.model,
        data=runner.data,
        cam=cam,
        scene=scene,
        mouse_force=mouse_force,
    )

    glfw.set_mouse_button_callback(window, mouse_controller.mouse_button_callback)
    glfw.set_cursor_pos_callback(window, mouse_controller.cursor_pos_callback)
    glfw.set_scroll_callback(window, mouse_controller.scroll_callback)

    step = 0

    try:
        while not glfw.window_should_close(window) and step < sim_steps:
            loop_start = time.time()

            mouse_force.apply()
            runner.step(step)

            cam.lookat[:] = runner.data.qpos[:3]

            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)

            mujoco.mjv_updateScene(
                runner.model,
                runner.data,
                opt,
                None,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )

            if not args.hide_velocity_vis:
                _draw_velocity_arrows(scene, runner, args.velocity_arrow_scale)

            mouse_force.draw(scene)

            mujoco.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
            glfw.poll_events()

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, cfg.dt - elapsed))

            step += 1

    finally:
        runner.data.xfrc_applied[:, :] = 0.0
        glfw.terminate()


if __name__ == "__main__":
    main()
