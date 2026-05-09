#!/usr/bin/env python3
from __future__ import annotations

import argparse

from deploy_mujoco.sim2sim_core import Sim2SimCfg, Sim2SimRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dolanga1 sim2sim runner (headless MuJoCo).")
    parser.add_argument("--load_model", type=str, required=True, help="Path to MuJoCo scene xml.")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy.onnx.")
    parser.add_argument("--sim_duration", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Sim2SimCfg(
        mujoco_model_path=args.load_model,
        onnx_path=args.policy,
        sim_duration=args.sim_duration,
    )
    runner = Sim2SimRunner(cfg)
    sim_steps = int(cfg.sim_duration / cfg.dt)
    for step in range(sim_steps):
        runner.step(step)
    print("[INFO] sim2sim finished.")


if __name__ == "__main__":
    main()

