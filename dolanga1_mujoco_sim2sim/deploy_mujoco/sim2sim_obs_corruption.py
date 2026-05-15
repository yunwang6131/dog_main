"""Test-time observation / control perturbations for MuJoCo sim2sim (no Isaac training).

Noise is applied in *physical* units before ``build_actor_obs_frame`` scaling:
  - gyro: rad/s
  - projected gravity: unit vector component noise (same as training Unoise on gravity)
  - joint_pos_rel: rad
  - joint_vel_rel: rad/s

Friction / lumped mass / contact dynamics are *not* handled here: change MJCF or call
``mujoco`` APIs on ``MjModel`` in your own script if you need those.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sim2SimPerturbCfg:
    """Perturbations while *testing* an exported policy in MuJoCo."""

    seed: int | None = None

    # --- IMU / proprio uniform noise (half-width of U[-w, w]; 0 = disabled) ---
    noise_gyro: float = 0.0
    noise_projected_gravity: float = 0.0
    noise_joint_pos: float = 0.0
    noise_joint_vel: float = 0.0

    # --- Encoder-style bias, sampled once when the runner is constructed (rad / rad/s) ---
    encoder_bias_pos_range: tuple[float, float] | None = None
    encoder_bias_vel_range: tuple[float, float] | None = None

    # --- Delays in *control* ticks (every ``decimation`` physics steps one tick) ---
    obs_delay_control_steps: int = 0
    action_delay_control_steps: int = 0

    # --- PD gain scale sampled once at init (proxy for motor strength / gear mismatch) ---
    motor_gain_scale_range: tuple[float, float] | None = None


def _u(rng: np.random.Generator, w: float, size) -> np.ndarray:
    if w <= 0.0:
        return np.zeros(size, dtype=np.float32)
    return rng.uniform(-w, w, size=size).astype(np.float32)


def sample_encoder_biases(
    rng: np.random.Generator,
    n_joints: int,
    pos_range: tuple[float, float] | None,
    vel_range: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    pos_bias = np.zeros(n_joints, dtype=np.float32)
    vel_bias = np.zeros(n_joints, dtype=np.float32)
    if pos_range is not None:
        lo, hi = pos_range
        pos_bias = rng.uniform(lo, hi, size=n_joints).astype(np.float32)
    if vel_range is not None:
        lo, hi = vel_range
        vel_bias = rng.uniform(lo, hi, size=n_joints).astype(np.float32)
    return pos_bias, vel_bias


def corrupt_raw_proprio(
    rng: np.random.Generator,
    cfg: Sim2SimPerturbCfg,
    base_ang_vel_body: np.ndarray,
    projected_gravity_body: np.ndarray,
    joint_pos_rel: np.ndarray,
    joint_vel_rel: np.ndarray,
    encoder_pos_bias: np.ndarray,
    encoder_vel_bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return corrupted raw vectors (gyro, gravity, q_rel, dq_rel)."""
    gyro = base_ang_vel_body.astype(np.float32).copy()
    grav = projected_gravity_body.astype(np.float32).copy()
    qrel = joint_pos_rel.astype(np.float32).copy() + encoder_pos_bias
    dqrel = joint_vel_rel.astype(np.float32).copy() + encoder_vel_bias

    gyro += _u(rng, cfg.noise_gyro, 3)
    grav += _u(rng, cfg.noise_projected_gravity, 3)
    qrel += _u(rng, cfg.noise_joint_pos, qrel.shape)
    dqrel += _u(rng, cfg.noise_joint_vel, dqrel.shape)
    return gyro, grav, qrel, dqrel


def apply_motor_gain_scale(
    kp: np.ndarray, kd: np.ndarray, rng: np.random.Generator, scale_range: tuple[float, float] | None
) -> tuple[np.ndarray, np.ndarray, float]:
    if scale_range is None:
        return kp, kd, 1.0
    lo, hi = scale_range
    s = float(rng.uniform(lo, hi))
    return (kp * s).astype(np.float32), (kd * s).astype(np.float32), s


def _dict_to_perturb_cfg(d: dict) -> Sim2SimPerturbCfg:
    if "noise_gravity" in d and "noise_projected_gravity" not in d:
        d = {**d, "noise_projected_gravity": d["noise_gravity"]}

    def opt_float(key: str, default: float = 0.0) -> float:
        v = d.get(key)
        if v is None:
            return default
        return float(v)

    def opt_tup2(key: str) -> tuple[float, float] | None:
        v = d.get(key)
        if v is None:
            return None
        if not isinstance(v, (list, tuple)) or len(v) != 2:
            raise ValueError(f"{key} must be [lo, hi] or null, got {v!r}")
        return (float(v[0]), float(v[1]))

    def opt_int(key: str, default: int = 0) -> int:
        v = d.get(key)
        if v is None:
            return default
        return int(v)

    seed = d.get("seed")
    if seed is not None:
        seed = int(seed)

    return Sim2SimPerturbCfg(
        seed=seed,
        noise_gyro=opt_float("noise_gyro", 0.0),
        noise_projected_gravity=opt_float("noise_projected_gravity", 0.0),
        noise_joint_pos=opt_float("noise_joint_pos", 0.0),
        noise_joint_vel=opt_float("noise_joint_vel", 0.0),
        encoder_bias_pos_range=opt_tup2("encoder_bias_pos_range"),
        encoder_bias_vel_range=opt_tup2("encoder_bias_vel_range"),
        obs_delay_control_steps=opt_int("obs_delay_control_steps", 0),
        action_delay_control_steps=opt_int("action_delay_control_steps", 0),
        motor_gain_scale_range=opt_tup2("motor_gain_scale_range"),
    )


def load_sim2sim_perturb_cfg_from_yaml(path: str | Path) -> Sim2SimPerturbCfg:
    """Load :class:`Sim2SimPerturbCfg` from YAML. Requires ``pip install pyyaml``."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError("YAML perturb config requires PyYAML: pip install pyyaml") from e

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    with p.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return Sim2SimPerturbCfg()
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a mapping, got {type(raw).__name__}")
    return _dict_to_perturb_cfg(raw)


def add_sim2sim_perturb_cli(parser: argparse.ArgumentParser) -> None:
    """Attach optional test perturbation flags (use with :func:`build_sim2sim_perturb_cfg`)."""
    g = parser.add_argument_group("sim2sim test perturbations (no training)")
    g.add_argument(
        "--perturb_yaml",
        type=str,
        default=None,
        help="YAML file for perturbations (see deploy_mujoco/configs/sim2sim_perturb.yaml). "
        "CLI options below override YAML when provided.",
    )
    g.add_argument("--s2s_seed", type=int, default=None, help="RNG seed (overrides YAML if set).")
    g.add_argument(
        "--noise_gyro",
        type=float,
        default=None,
        help="Override: U[-w,w] rad/s on gyro. Omit to use YAML / default 0.",
    )
    g.add_argument("--noise_gravity", type=float, default=None, help="Override: gravity component noise half-width.")
    g.add_argument("--noise_joint_pos", type=float, default=None, help="Override: q_rel noise half-width (rad).")
    g.add_argument("--noise_joint_vel", type=float, default=None, help="Override: dq noise half-width (rad/s).")
    g.add_argument(
        "--enc_bias_pos",
        type=float,
        default=None,
        help="Override: if >0, joint pos bias U[-w,w] rad once per run; if 0, disable bias.",
    )
    g.add_argument("--enc_bias_vel", type=float, default=None, help="Override: joint vel bias U[-w,w] rad/s.")
    g.add_argument("--obs_delay", type=int, default=None, help="Override: obs delay in policy steps.")
    g.add_argument("--action_delay", type=int, default=None, help="Override: action delay in policy steps.")
    g.add_argument("--motor_gain_min", type=float, default=None, help="Override: kp/kd scale range (use with max).")
    g.add_argument("--motor_gain_max", type=float, default=None)


def build_sim2sim_perturb_cfg(ns: argparse.Namespace) -> Sim2SimPerturbCfg:
    """Load YAML if ``--perturb_yaml`` set, then apply non-None CLI overrides."""
    if getattr(ns, "perturb_yaml", None):
        cfg = load_sim2sim_perturb_cfg_from_yaml(ns.perturb_yaml)
    else:
        cfg = Sim2SimPerturbCfg()

    if ns.s2s_seed is not None:
        cfg.seed = ns.s2s_seed
    if ns.noise_gyro is not None:
        cfg.noise_gyro = float(ns.noise_gyro)
    if ns.noise_gravity is not None:
        cfg.noise_projected_gravity = float(ns.noise_gravity)
    if ns.noise_joint_pos is not None:
        cfg.noise_joint_pos = float(ns.noise_joint_pos)
    if ns.noise_joint_vel is not None:
        cfg.noise_joint_vel = float(ns.noise_joint_vel)

    if ns.enc_bias_pos is not None:
        cfg.encoder_bias_pos_range = (
            (-ns.enc_bias_pos, ns.enc_bias_pos) if ns.enc_bias_pos > 0 else None
        )
    if ns.enc_bias_vel is not None:
        cfg.encoder_bias_vel_range = (
            (-ns.enc_bias_vel, ns.enc_bias_vel) if ns.enc_bias_vel > 0 else None
        )

    if ns.obs_delay is not None:
        cfg.obs_delay_control_steps = int(ns.obs_delay)
    if ns.action_delay is not None:
        cfg.action_delay_control_steps = int(ns.action_delay)

    if ns.motor_gain_min is not None and ns.motor_gain_max is not None:
        cfg.motor_gain_scale_range = (float(ns.motor_gain_min), float(ns.motor_gain_max))
    elif ns.motor_gain_min is not None or ns.motor_gain_max is not None:
        raise ValueError("Set both --motor_gain_min and --motor_gain_max, or neither.")

    return cfg


def sim2sim_perturb_cfg_from_ns(ns: argparse.Namespace) -> Sim2SimPerturbCfg:
    """Backward-compatible alias for :func:`build_sim2sim_perturb_cfg`."""
    return build_sim2sim_perturb_cfg(ns)
