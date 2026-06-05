# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from robot_lab.tasks.manager_based.locomotion.velocity.mdp.barrier_style_rewards import (
    command_uses_trot_gait,
)


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def phase_with_command(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    stand_threshold: float = 0.2,
    cycle_time: float | None = None,
    walk_cycle_time: float | None = None,
    trot_cycle_time: float | None = None,
    gait_velocity_threshold: float | None = None,
) -> torch.Tensor:
    """Return gait phase features and zero them in stand-mode, matching the paper description."""
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    t = env.episode_length_buf.float() * env.step_dt
    if gait_velocity_threshold is not None:
        if walk_cycle_time is None or trot_cycle_time is None:
            raise ValueError("Velocity-selective phase requires walk_cycle_time and trot_cycle_time.")
        use_trot = command_uses_trot_gait(env, command_name, gait_velocity_threshold)
        cycle_time_per_env = torch.where(
            use_trot,
            torch.full_like(t, trot_cycle_time),
            torch.full_like(t, walk_cycle_time),
        )
        phase_norm = t / cycle_time_per_env
    else:
        if cycle_time is None:
            raise ValueError("Fixed phase observation requires cycle_time.")
        phase_norm = t / cycle_time

    phase_tensor = torch.stack(
        (torch.sin(2.0 * torch.pi * phase_norm), torch.cos(2.0 * torch.pi * phase_norm)),
        dim=-1,
    )
    command = env.command_manager.get_command(command_name)
    moving_mask = (torch.linalg.norm(command, dim=1, keepdim=True) >= stand_threshold).float()
    return phase_tensor * moving_mask


def command_stand_mode(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    threshold: float = 0.2,
) -> torch.Tensor:
    """Binary stand-mode indicator used by the actor/critic observations."""
    command = env.command_manager.get_command(command_name)
    return (torch.linalg.norm(command, dim=1, keepdim=True) < threshold).float()


def foot_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Binary contact state for the selected feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return (forces > contact_threshold).float()


def feet_positions_body(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Feet positions in the body frame, flattened as [x1, y1, z1, ..., xn, yn, zn]."""
    asset: Articulation = env.scene[asset_cfg.name]
    translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[:, None, :]
    feet_pos_body = torch.zeros_like(translated)
    for i in range(len(asset_cfg.body_ids)):
        feet_pos_body[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), translated[:, i, :]
        )
    return feet_pos_body.view(env.num_envs, -1)
