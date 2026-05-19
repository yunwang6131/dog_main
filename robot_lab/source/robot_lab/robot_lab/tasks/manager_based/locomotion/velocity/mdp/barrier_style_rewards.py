# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Relaxed log-barrier style rewards (Kim et al., barrier-based legged locomotion)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Default barrier weight alpha_k in Eq. (1).
_DEFAULT_ALPHA = 0.1


def relaxed_log_barrier(z: torch.Tensor, delta: float) -> torch.Tensor:
    """Relaxed logarithmic barrier B(z; delta) for constraint z >= 0 (Kim et al., Eq. in Sec. III-A)."""
    delta_t = torch.tensor(delta, device=z.device, dtype=z.dtype)
    log_delta = torch.log(delta_t)
    quad = log_delta - 0.5 * torch.square((z - 2.0 * delta_t) / delta_t) + 0.5
    return torch.where(z > delta_t, torch.log(torch.clamp(z, min=1e-8)), quad)


def barrier_soft_interval(
    constraint: torch.Tensor,
    *,
    d_lower: float | None = None,
    d_upper: float | None = None,
    delta: float,
    alpha: float = _DEFAULT_ALPHA,
) -> torch.Tensor:
    """Sum of barrier terms for d_lower <= C <= d_upper (Eq. 1)."""
    reward = torch.zeros(constraint.shape[0], device=constraint.device, dtype=constraint.dtype)
    if d_lower is not None:
        reward = reward + alpha * relaxed_log_barrier(constraint - d_lower, delta)
    if d_upper is not None:
        reward = reward + alpha * relaxed_log_barrier(d_upper - constraint, delta)
    return reward


def _upright_scale(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7


def _command_active(env: "ManagerBasedRLEnv", command_name: str, threshold: float = 0.1) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.linalg.norm(cmd, dim=1) > threshold).float()


def _stand_mode(env: "ManagerBasedRLEnv", command_name: str, threshold: float = 0.2) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.linalg.norm(cmd, dim=1) < threshold).float()


def _mean_terrain_height(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    fallback: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits = sensor.data.ray_hits_w[..., 2]
    finite_mask = torch.isfinite(ray_hits) & (torch.abs(ray_hits) < 1e6)
    if reduction == "mean":
        safe_hits = torch.where(finite_mask, ray_hits, torch.zeros_like(ray_hits))
        valid_counts = finite_mask.sum(dim=1).clamp(min=1)
        reduced_hits = safe_hits.sum(dim=1) / valid_counts
    elif reduction == "max":
        neg_inf = torch.full_like(ray_hits, float("-inf"))
        safe_hits = torch.where(finite_mask, ray_hits, neg_inf)
        reduced_hits = safe_hits.max(dim=1).values
    else:
        raise ValueError(f"Unsupported terrain reduction '{reduction}'. Expected 'mean' or 'max'.")
    has_valid_hits = finite_mask.any(dim=1)
    return torch.where(has_valid_hits, reduced_hits, fallback)


def _terrain_heights_from_sensors(
    env: "ManagerBasedRLEnv",
    sensor_cfgs: list[SceneEntityCfg] | None,
    fallback_height: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    if sensor_cfgs is None or len(sensor_cfgs) == 0:
        return torch.full((env.num_envs, 0), fallback_height, device=env.device)
    fallback = torch.full((env.num_envs,), fallback_height, device=env.device)
    heights = [_mean_terrain_height(env, sensor_cfg, fallback, reduction=reduction) for sensor_cfg in sensor_cfgs]
    return torch.stack(heights, dim=1)


def _get_gait_state(
    env: "ManagerBasedRLEnv",
    *,
    period: float,
    phase_offsets: list[float],
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gait cycle g_i in [-1, 1], constraint variable f_i, and contact flags."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    is_contact = forces > contact_threshold

    t = env.episode_length_buf.float() * env.step_dt
    num_feet = len(sensor_cfg.body_ids)
    g = torch.zeros(env.num_envs, num_feet, device=env.device)
    for i, phi in enumerate(phase_offsets):
        phase = (t / period + phi) * (2.0 * math.pi)
        g[:, i] = torch.sin(phase)

    # Eq. (2): f_i = g_i if contacting else -g_i
    f = torch.where(is_contact, g, -g)
    return g, f, is_contact


def barrier_style_gait(
    env: "ManagerBasedRLEnv",
    period: float,
    phase_offsets: list[float],
    sensor_cfg: SceneEntityCfg,
    d_lower: float = -0.6,
    d_upper: float = 2.0,
    delta: float = 0.1,
    alpha: float = _DEFAULT_ALPHA,
    command_name: str = "base_velocity",
    command_threshold: float = 0.2,
    stand_threshold: float = 0.2,
) -> torch.Tensor:
    """Preferred gait via relaxed barrier on f_i (Sec. III-B, Table I)."""
    _, f, is_contact = _get_gait_state(env, period=period, phase_offsets=phase_offsets, sensor_cfg=sensor_cfg)
    moving_mask = _command_active(env, command_name, command_threshold)
    stand_mask = _stand_mode(env, command_name, stand_threshold)
    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(f.shape[1]):
        reward = reward + moving_mask * barrier_soft_interval(
            f[:, i],
            d_lower=d_lower,
            d_upper=d_upper,
            delta=delta,
            alpha=alpha,
        )
        stand_contact = torch.where(is_contact[:, i], torch.ones_like(f[:, i]), -torch.ones_like(f[:, i]))
        reward = reward + stand_mask * barrier_soft_interval(
            stand_contact,
            d_lower=d_lower,
            d_upper=d_upper,
            delta=delta,
            alpha=alpha,
        )
    reward *= _upright_scale(env)
    return reward


def barrier_style_foot_clearance(
    env: "ManagerBasedRLEnv",
    period: float,
    phase_offsets: list[float],
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    terrain_sensor_cfgs: list[SceneEntityCfg] | None = None,
    p_des: float = 0.15,
    d_lower_gait: float = -0.6,
    d_lower_clearance: float = -0.08,
    d_upper_clearance: float = 1.0,
    delta: float = 0.01,
    alpha: float = _DEFAULT_ALPHA,
    command_name: str = "base_velocity",
    command_threshold: float = 0.2,
    stand_threshold: float = 0.2,
    terrain_height: float = 0.0,
) -> torch.Tensor:
    """Foot clearance barrier l_i (Eq. 3, Table I). Uses flat terrain_height when no height scan."""
    g, _, _ = _get_gait_state(env, period=period, phase_offsets=phase_offsets, sensor_cfg=sensor_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if terrain_sensor_cfgs is None or len(terrain_sensor_cfgs) != foot_z.shape[1]:
        terrain_heights = torch.full_like(foot_z, terrain_height)
    else:
        # Eq. (3) uses the highest sampled terrain around each foot within the local neighborhood.
        terrain_heights = _terrain_heights_from_sensors(env, terrain_sensor_cfgs, terrain_height, reduction="max")
    moving_mask = _command_active(env, command_name, command_threshold)
    stand_mask = _stand_mode(env, command_name, stand_threshold)

    reward = torch.zeros(env.num_envs, device=env.device)
    swing_mask = (g <= d_lower_gait) & moving_mask[:, None].bool()
    for i in range(foot_z.shape[1]):
        # l_i = p_i - (max terrain sample + p_des); flat terrain => max sample = terrain_height.
        l_i = foot_z[:, i] - (terrain_heights[:, i] + p_des)
        l_i = torch.where(swing_mask[:, i], l_i, torch.zeros_like(l_i))
        reward = reward + barrier_soft_interval(
            l_i, d_lower=d_lower_clearance, d_upper=d_upper_clearance, delta=delta, alpha=alpha
        )
        stand_l_i = -(foot_z[:, i] - terrain_heights[:, i])
        reward = reward + stand_mask * barrier_soft_interval(
            stand_l_i, d_lower=d_lower_clearance, d_upper=d_upper_clearance, delta=delta, alpha=alpha
        )
    reward *= _upright_scale(env)
    return reward


def barrier_style_joint_position(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    hip_joint_names: list[str],
    thigh_joint_names: list[str],
    calf_joint_names: list[str],
    roll_bounds: tuple[float, float] = (-math.pi / 6.0, math.pi / 6.0),
    thigh_bounds: tuple[float, float] = (-math.pi / 4.0, math.pi / 4.0),
    calf_bounds: tuple[float, float] = (-2.0 * math.pi / 5.0, math.pi / 4.0),
    delta: float = 0.08,
    alpha: float = _DEFAULT_ALPHA,
) -> torch.Tensor:
    """Joint position barriers on deviation from nominal posture (Table I, quadruped)."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.joint_pos
    q_nom = asset.data.default_joint_pos

    def _joint_barrier(names: list[str], bounds: tuple[float, float]) -> torch.Tensor:
        ids = asset.find_joints(names, preserve_order=True)[0]
        if len(ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)
        dev = q[:, ids] - q_nom[:, ids]
        term = torch.zeros(env.num_envs, device=env.device)
        for j in range(len(ids)):
            term = term + barrier_soft_interval(
                dev[:, j], d_lower=bounds[0], d_upper=bounds[1], delta=delta, alpha=alpha
            )
        return term

    reward = (
        _joint_barrier(hip_joint_names, roll_bounds)
        + _joint_barrier(thigh_joint_names, thigh_bounds)
        + _joint_barrier(calf_joint_names, calf_bounds)
    )
    reward *= _upright_scale(env)
    return reward


def barrier_style_body_height(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    front_body_names: list[str],
    hind_body_names: list[str],
    front_terrain_sensor_cfgs: list[SceneEntityCfg] | None = None,
    hind_terrain_sensor_cfgs: list[SceneEntityCfg] | None = None,
    front_bounds: tuple[float, float] = (0.52, 0.66),
    hind_bounds: tuple[float, float] = (1.05, 1.55),
    front_delta: float = 0.04,
    hind_delta: float = 0.04,
    alpha: float = _DEFAULT_ALPHA,
) -> torch.Tensor:
    """Front / hind body height barriers b_hF, b_hH (Table I).

    Dolanga1 does not provide separate roll-joint bodies, so the hip links are used as the
    closest proxy to the roll-joint heights described in the paper.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain_fallback = asset.data.root_pos_w[:, 2] * 0.0

    def _reference_height(body_names: list[str]) -> torch.Tensor:
        ids = asset.find_bodies(body_names, preserve_order=True)[0]
        if len(ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)
        return asset.data.body_pos_w[:, ids, 2].mean(dim=1)

    bh_f = _reference_height(front_body_names)
    bh_h = _reference_height(hind_body_names)
    if front_terrain_sensor_cfgs:
        front_terrain = torch.stack(
            [
                _mean_terrain_height(env, sensor_cfg, terrain_fallback, reduction="max")
                for sensor_cfg in front_terrain_sensor_cfgs
            ],
            dim=1,
        ).mean(dim=1)
        bh_f = bh_f - front_terrain
    if hind_terrain_sensor_cfgs:
        hind_terrain = torch.stack(
            [
                _mean_terrain_height(env, sensor_cfg, terrain_fallback, reduction="max")
                for sensor_cfg in hind_terrain_sensor_cfgs
            ],
            dim=1,
        ).mean(dim=1)
        bh_h = bh_h - hind_terrain
    reward = barrier_soft_interval(bh_f, d_lower=front_bounds[0], d_upper=front_bounds[1], delta=front_delta, alpha=alpha)
    reward = reward + barrier_soft_interval(
        bh_h, d_lower=hind_bounds[0], d_upper=hind_bounds[1], delta=hind_delta, alpha=alpha
    )
    reward *= _upright_scale(env)
    return reward


def barrier_style_velocity_tracking(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vel_bounds: tuple[float, float] = (-0.4, 0.4),
    ang_bounds: tuple[float, float] = (-0.4, 0.4),
    delta: float = 0.2,
    alpha: float = _DEFAULT_ALPHA,
) -> torch.Tensor:
    """Target velocity tracking via barriers on command errors (Table I)."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    err_xy = cmd[:, :2] - asset.data.root_lin_vel_b[:, :2]
    err_yaw = cmd[:, 2] - asset.data.root_ang_vel_b[:, 2]

    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(2):
        reward = reward + barrier_soft_interval(err_xy[:, i], d_lower=vel_bounds[0], d_upper=vel_bounds[1], delta=delta, alpha=alpha)
    reward = reward + barrier_soft_interval(
        err_yaw, d_lower=ang_bounds[0], d_upper=ang_bounds[1], delta=delta, alpha=alpha
    )
    reward *= _upright_scale(env)
    return reward


def barrier_style_base_motion(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    omega_xy_bounds: tuple[float, float] = (-0.3, 0.3),
    vz_bounds: tuple[float, float] = (-0.2, 0.2),
    omega_delta: float = 0.3,
    vz_delta: float = 0.2,
    alpha: float = _DEFAULT_ALPHA,
) -> torch.Tensor:
    """Base roll/pitch rate and vertical velocity barriers (Table I)."""
    asset: Articulation = env.scene[asset_cfg.name]
    omega_xy = asset.data.root_ang_vel_b[:, :2]
    vz = asset.data.root_lin_vel_b[:, 2]

    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(2):
        reward = reward + barrier_soft_interval(
            omega_xy[:, i], d_lower=omega_xy_bounds[0], d_upper=omega_xy_bounds[1], delta=omega_delta, alpha=alpha
        )
    reward = reward + barrier_soft_interval(
        vz, d_lower=vz_bounds[0], d_upper=vz_bounds[1], delta=vz_delta, alpha=alpha
    )
    reward *= _upright_scale(env)
    return reward


def barrier_style_joint_velocity(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    vel_bounds: tuple[float, float] = (-8.0, 8.0),
    delta: float = 2.0,
    alpha: float = _DEFAULT_ALPHA,
) -> torch.Tensor:
    """Joint velocity barriers (Table I)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or len(joint_ids) == 0:
        joint_ids = list(range(asset.num_joints))
    qd = asset.data.joint_vel[:, joint_ids]
    reward = torch.zeros(env.num_envs, device=env.device)
    for j in range(qd.shape[1]):
        reward = reward + barrier_soft_interval(
            qd[:, j], d_lower=vel_bounds[0], d_upper=vel_bounds[1], delta=delta, alpha=alpha
        )
    reward *= _upright_scale(env)
    return reward


def barrier_style_quadruped_trot(
    env: "ManagerBasedRLEnv",
    period: float = 0.72,
    phase_offsets: list[float] | None = None,
    sensor_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Aggregate quadruped trot barrier stream (Table I defaults for rough terrain)."""
    if phase_offsets is None:
        # LF, RF, LH, RH — diagonal trot (phi=0 on LF/RH, 0.5 on RF/LH).
        phase_offsets = [0.0, 0.5, 0.5, 0.0]
    if sensor_cfg is None:
        sensor_cfg = SceneEntityCfg("contact_forces", body_names=[".*_foot_link"])

    return (
        barrier_style_gait(
            env,
            period=period,
            phase_offsets=phase_offsets,
            sensor_cfg=sensor_cfg,
            command_name=command_name,
        )
        + barrier_style_foot_clearance(
            env,
            period=period,
            phase_offsets=phase_offsets,
            sensor_cfg=sensor_cfg,
            asset_cfg=asset_cfg,
            command_name=command_name,
        )
        + barrier_style_joint_position(
            env,
            asset_cfg=asset_cfg,
            hip_joint_names=[".*_hip_joint"],
            thigh_joint_names=[".*_thigh_joint"],
            calf_joint_names=[".*_calf_joint"],
        )
        + barrier_style_body_height(
            env,
            asset_cfg=asset_cfg,
            front_body_names=["LF_hip_link", "RF_hip_link"],
            hind_body_names=["LH_hip_link", "RH_hip_link"],
        )
        + barrier_style_velocity_tracking(env, command_name=command_name, asset_cfg=asset_cfg)
        + barrier_style_base_motion(env, asset_cfg=asset_cfg)
        + barrier_style_joint_velocity(env, asset_cfg=SceneEntityCfg(asset_cfg.name, joint_names=[".*"]))
    )
