# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel_ratio(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    move_up_fraction: float = 0.60,
    move_down_fraction: float = 0.40,
    stand_command_threshold: float = 0.08,
) -> torch.Tensor:
    """Terrain curriculum keyed to commanded-distance completion ratio (slow-speed friendly).

    Expected distance per episode is ``||cmd_xy|| * episode_length_s``. Upgrade when the robot
    walks farther than ``move_up_fraction`` of that expectation; downgrade when it walks less
    than ``move_down_fraction``. Stand/near-zero commands skip both checks.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    cmd_speed = torch.norm(command[env_ids, :2], dim=1)
    expected_distance = cmd_speed * env.max_episode_length_s
    moving = cmd_speed > stand_command_threshold

    move_up = moving & (distance > expected_distance * move_up_fraction)
    move_down = moving & (distance < expected_distance * move_down_fraction)
    move_down &= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_vel_tracking_stability(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    move_up_fraction: float = 0.60,
    move_down_fraction: float = 0.40,
    stand_command_threshold: float = 0.08,
    lin_vel_error_threshold: float = 0.35,
    ang_vel_error_threshold: float = 0.50,
    gravity_xy_threshold: float = 0.45,
) -> torch.Tensor:
    """Terrain curriculum that gates terrain progression on:
    1. Distance traveled (original criterion)
    2. Velocity tracking accuracy (lin_vel + ang_vel errors)
    3. Postural stability (projected_gravity xy component)

    Upgrade requires: good distance + good tracking + stable posture.
    Downgrade triggers on: bad distance OR bad tracking OR unstable posture.
    Stand/near-zero commands skip all checks.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    # 1. Distance completion (original logic)
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    cmd_xy = command[env_ids, :2]
    cmd_speed = torch.norm(cmd_xy, dim=1)
    expected_distance = cmd_speed * env.max_episode_length_s
    moving = cmd_speed > stand_command_threshold

    distance_good = distance > expected_distance * move_up_fraction
    distance_bad = distance < expected_distance * move_down_fraction

    # 2. Velocity tracking error
    root_lin_vel_b = asset.data.root_lin_vel_b[env_ids, :2]
    root_ang_vel_b = asset.data.root_ang_vel_b[env_ids, 2]

    lin_vel_error = torch.norm(root_lin_vel_b - command[env_ids, :2], dim=1)
    ang_vel_error = torch.abs(root_ang_vel_b - command[env_ids, 2])

    tracking_good = (
        (lin_vel_error < lin_vel_error_threshold)
        & (ang_vel_error < ang_vel_error_threshold)
    )
    tracking_bad = (
        (lin_vel_error > lin_vel_error_threshold * 1.8)
        | (ang_vel_error > ang_vel_error_threshold * 1.8)
    )

    # 3. Postural stability via projected_gravity xy magnitude
    gravity_xy = torch.norm(asset.data.projected_gravity_b[env_ids, :2], dim=1)
    stable = gravity_xy < gravity_xy_threshold
    unstable = gravity_xy > gravity_xy_threshold * 1.4

    # 4. Move up: all criteria must be met
    move_up = moving & distance_good & tracking_good & stable

    # 5. Move down: any criterion fails badly
    move_down = moving & (distance_bad | tracking_bad | unstable)
    move_down &= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)
