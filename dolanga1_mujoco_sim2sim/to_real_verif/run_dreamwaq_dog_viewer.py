#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import mujoco.viewer
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from deploy_mujoco_viewer.keyboard_control import KeyboardCommandController
from sim2sim_policy_full_runner import PolicyFullSim2SimCfg, PolicyFullSim2SimRunner


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "dreamwaq_dog_viewer_2026-05-13_17-33-33_initial.yaml"
CALF_INDICES = (2, 5, 8, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuJoCo policy_full viewer for the 2026-05-13_17-33-33 bundle.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the local MuJoCo viewer yaml.")
    parser.add_argument("--load_model", type=str, default=None, help="Override scene xml path.")
    parser.add_argument("--policy_full", type=str, default=None, help="Override policy_full.pt path.")
    parser.add_argument("--sim_duration", type=float, default=None)
    parser.add_argument("--cmd_x", type=float, default=None)
    parser.add_argument("--cmd_y", type=float, default=None)
    parser.add_argument("--cmd_yaw", type=float, default=None)
    parser.add_argument("--init_base_height", type=float, default=None, help="Override base init z.")
    parser.add_argument("--calf_default", type=float, default=None, help="Override all four calf q_default values.")
    parser.add_argument("--keyboard", action="store_true", help="Start from zero command and use keyboard control.")
    parser.add_argument("--no_viewer", action="store_true", help="Run a short smoke test without opening the GUI.")
    parser.add_argument("--smoke_steps", type=int, default=8, help="Number of steps for --no_viewer.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} does not contain a YAML mapping.")
    return data


def build_cfg_from_yaml(data: dict[str, Any], args: argparse.Namespace) -> PolicyFullSim2SimCfg:
    runtime = data["runtime"]
    control = data["control"]
    commands = data["commands"]

    q_default = np.asarray(control["q_default"], dtype=np.float32)
    if args.calf_default is not None:
        q_default = q_default.copy()
        for idx in CALF_INDICES:
            q_default[idx] = np.float32(args.calf_default)

    if args.keyboard:
        startup_cmd = np.zeros(3, dtype=np.float32)
    else:
        startup_cmd = np.asarray(commands["startup_cmd"], dtype=np.float32).copy()
        if args.cmd_x is not None:
            startup_cmd[0] = np.float32(args.cmd_x)
        if args.cmd_y is not None:
            startup_cmd[1] = np.float32(args.cmd_y)
        if args.cmd_yaw is not None:
            startup_cmd[2] = np.float32(args.cmd_yaw)

    return PolicyFullSim2SimCfg(
        sim_duration=float(args.sim_duration if args.sim_duration is not None else runtime["sim_duration"]),
        dt=float(runtime["dt"]),
        decimation=int(runtime["decimation"]),
        history_len=int(runtime["history_len"]),
        warmup_seconds=float(runtime["warmup_seconds"]),
        init_base_height=float(args.init_base_height if args.init_base_height is not None else runtime["init_base_height"]),
        mujoco_model_path=str(args.load_model or data["scene_xml"]),
        policy_full_path=str(args.policy_full or data["policy_full"]),
        q_default=q_default,
        action_scale=np.asarray(control["action_scale"], dtype=np.float32),
        kp=np.asarray(control["kp"], dtype=np.float32),
        kd=np.asarray(control["kd"], dtype=np.float32),
        tau_limit=np.asarray(control["tau_limit"], dtype=np.float32),
        cmd=startup_cmd,
    )


def validate_paths(cfg: PolicyFullSim2SimCfg) -> None:
    for label, path_str in (
        ("scene_xml", cfg.mujoco_model_path),
        ("policy_full", cfg.policy_full_path),
    ):
        path = Path(path_str)
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")


def run_no_viewer(runner: PolicyFullSim2SimRunner, smoke_steps: int) -> None:
    steps = max(1, int(smoke_steps))
    for step in range(steps):
        runner.step(step)
    print(
        "[OK] Smoke run finished: "
        f"steps={steps}, "
        f"base_pos={runner.data.qpos[:3].tolist()}, "
        f"cmd={runner.cmd.tolist()}, "
        f"calf_q_default={runner.q_default[list(CALF_INDICES)].tolist()}"
    )


def run_viewer(runner: PolicyFullSim2SimRunner, cfg: PolicyFullSim2SimCfg, keyboard_enabled: bool) -> None:
    sim_steps = int(cfg.sim_duration / cfg.dt)
    keyboard = KeyboardCommandController(cfg.cmd) if keyboard_enabled else None
    launch_kwargs = {"key_callback": keyboard.key_callback} if keyboard is not None else {}

    with mujoco.viewer.launch_passive(runner.model, runner.data, **launch_kwargs) as viewer:
        if keyboard is not None:
            keyboard.print_help()
        viewer.cam.distance = 10.0
        viewer.cam.elevation = -10
        viewer.cam.azimuth = 90
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


def main() -> None:
    args = parse_args()
    data = load_yaml(args.config.resolve())
    cfg = build_cfg_from_yaml(data, args)
    validate_paths(cfg)

    print(
        "[INFO] Launch config: "
        f"scene={cfg.mujoco_model_path}, "
        f"policy_full={cfg.policy_full_path}, "
        f"init_base_height={cfg.init_base_height}, "
        f"calf_q_default={cfg.q_default[list(CALF_INDICES)].tolist()}, "
        f"cmd={cfg.cmd.tolist()}, "
        f"keyboard={args.keyboard}, "
        f"viewer={not args.no_viewer}"
    )

    runner = PolicyFullSim2SimRunner(cfg)
    if args.no_viewer:
        run_no_viewer(runner, args.smoke_steps)
    else:
        run_viewer(runner, cfg, keyboard_enabled=args.keyboard)


if __name__ == "__main__":
    main()
