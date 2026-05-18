from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import mujoco
import numpy as np
from mujoco.glfw import glfw


@dataclass(frozen=True)
class KeyboardControlCfg:
    x_step: float = 0.2
    y_step: float = 0.2
    yaw_step: float = 0.2
    x_limit: float = 3.0
    y_limit: float = 1.5
    yaw_limit: float = 1.5


class KeyboardCommandController:
    def __init__(self, initial_cmd: np.ndarray, cfg: KeyboardControlCfg | None = None):
        self.cfg = cfg or KeyboardControlCfg()
        self._lock = Lock()
        self._initial_cmd = np.asarray(initial_cmd, dtype=np.float32).copy()
        if self._initial_cmd.shape != (3,):
            raise ValueError(f"initial_cmd must have shape (3,), got {self._initial_cmd.shape}")
        self._cmd = self._initial_cmd.copy()

    def current_cmd(self) -> np.ndarray:
        with self._lock:
            return self._cmd.copy()

    def key_callback(self, keycode: int) -> None:
        changed = False
        show_help = False
        with self._lock:
            if keycode == glfw.KEY_W:
                self._cmd[0] = np.clip(self._cmd[0] + self.cfg.x_step, -self.cfg.x_limit, self.cfg.x_limit)
                changed = True
            elif keycode == glfw.KEY_S:
                self._cmd[0] = np.clip(self._cmd[0] - self.cfg.x_step, -self.cfg.x_limit, self.cfg.x_limit)
                changed = True
            elif keycode == glfw.KEY_A:
                self._cmd[1] = np.clip(self._cmd[1] + self.cfg.y_step, -self.cfg.y_limit, self.cfg.y_limit)
                changed = True
            elif keycode == glfw.KEY_D:
                self._cmd[1] = np.clip(self._cmd[1] - self.cfg.y_step, -self.cfg.y_limit, self.cfg.y_limit)
                changed = True
            elif keycode == glfw.KEY_Q:
                self._cmd[2] = np.clip(self._cmd[2] + self.cfg.yaw_step, -self.cfg.yaw_limit, self.cfg.yaw_limit)
                changed = True
            elif keycode == glfw.KEY_E:
                self._cmd[2] = np.clip(self._cmd[2] - self.cfg.yaw_step, -self.cfg.yaw_limit, self.cfg.yaw_limit)
                changed = True
            elif keycode == glfw.KEY_SPACE:
                self._cmd[:] = 0.0
                changed = True
            elif keycode == glfw.KEY_R:
                self._cmd[:] = self._initial_cmd
                changed = True
            elif keycode == glfw.KEY_H:
                show_help = True

            cmd = self._cmd.copy()

        if changed:
            print(f"[keyboard] cmd_x={cmd[0]:+.2f}, cmd_y={cmd[1]:+.2f}, cmd_yaw={cmd[2]:+.2f}")
        elif show_help:
            self.print_help()

    def print_help(self) -> None:
        print(
            "[keyboard] W/S: +/- cmd_x, "
            "A/D: +/- cmd_y, "
            "Q/E: +/- cmd_yaw, "
            "SPACE: zero cmd, "
            "R: reset startup cmd, "
            "H: print help"
        )

    def set_overlay(self, viewer: mujoco.viewer.Handle) -> None:
        cmd = self.current_cmd()
        viewer.set_texts(
            (
                int(mujoco.mjtFontScale.mjFONTSCALE_150),
                int(mujoco.mjtGridPos.mjGRID_TOPLEFT),
                "Keyboard Control\nW/S: +/- x\nA/D: +/- y\nQ/E: +/- yaw\nSpace: stop\nR: reset\nH: help",
                f"cmd_x  {cmd[0]:+5.2f}\ncmd_y  {cmd[1]:+5.2f}\ncmd_yaw {cmd[2]:+5.2f}",
            )
        )
