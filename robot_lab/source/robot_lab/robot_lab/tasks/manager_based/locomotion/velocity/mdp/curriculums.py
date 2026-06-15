# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _init_sim2sim_slide_column_indices(env: ManagerBasedRLEnv) -> None:
    """Cache which terrain columns were generated as sim2sim_slide."""
    terrain: TerrainImporter = env.scene.terrain
    gen_cfg = terrain.cfg.terrain_generator
    num_cols = gen_cfg.num_cols
    sub_names = list(gen_cfg.sub_terrains.keys())
    proportions = np.array([sub_cfg.proportion for sub_cfg in gen_cfg.sub_terrains.values()])
    proportions /= np.sum(proportions)
    cumsum = np.cumsum(proportions)

    slide_cols: list[int] = []
    non_slide_cols: list[int] = []
    for col in range(num_cols):
        sub_index = int(np.min(np.where(col / num_cols + 0.001 < cumsum)[0]))
        if sub_names[sub_index] == "sim2sim_slide":
            slide_cols.append(col)
        else:
            non_slide_cols.append(col)

    env._sim2sim_slide_col_indices = torch.tensor(slide_cols, device=env.device, dtype=torch.long)
    env._sim2sim_non_slide_col_indices = torch.tensor(non_slide_cols, device=env.device, dtype=torch.long)
    env._sim2sim_terrain_cols_ready = True


def _reassign_sim2sim_slide_column_mix(env: ManagerBasedRLEnv, slide_fraction: float) -> None:
    """Assign envs across terrain columns so ``slide_fraction`` train on sim2sim_slide patches."""
    terrain: TerrainImporter = env.scene.terrain
    num_envs = env.num_envs
    slide_cols = env._sim2sim_slide_col_indices
    non_slide_cols = env._sim2sim_non_slide_col_indices

    slide_fraction = float(np.clip(slide_fraction, 0.0, 1.0))
    n_slide = int(round(num_envs * slide_fraction))
    n_non = num_envs - n_slide

    col_types = torch.empty(num_envs, device=env.device, dtype=torch.long)
    if n_slide > 0:
        col_types[:n_slide] = slide_cols[torch.randint(len(slide_cols), (n_slide,), device=env.device)]
    if n_non > 0:
        col_types[n_slide:] = non_slide_cols[torch.randint(len(non_slide_cols), (n_non,), device=env.device)]

    perm = torch.randperm(num_envs, device=env.device)
    terrain.terrain_types[:] = col_types[perm]
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]


def terrain_levels_vel_slide_mix(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    switch_level: float = 5.8,
    switch_down_level: float = 5.3,
    phase_a_slide_fraction: float = 0.20,
    phase_b_slide_fraction: float = 0.60,
) -> torch.Tensor:
    """Terrain row curriculum + automatic sim2sim slide column mix keyed to mean terrain level.

    Phase A (mean level < ``switch_level``): ``phase_a_slide_fraction`` of envs on slide columns.
    Phase B (mean level >= ``switch_level``): ``phase_b_slide_fraction`` on slide columns.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    if not getattr(env, "_sim2sim_terrain_cols_ready", False):
        _init_sim2sim_slide_column_indices(env)
        _reassign_sim2sim_slide_column_mix(env, phase_a_slide_fraction)
        env._sim2sim_slide_fraction_applied = phase_a_slide_fraction
        env._sim2sim_slide_phase_b = False

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down &= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    mean_level = torch.mean(terrain.terrain_levels.float())

    if not env._sim2sim_slide_phase_b and mean_level >= switch_level:
        env._sim2sim_slide_phase_b = True
    elif env._sim2sim_slide_phase_b and mean_level < switch_down_level:
        env._sim2sim_slide_phase_b = False

    target_fraction = phase_b_slide_fraction if env._sim2sim_slide_phase_b else phase_a_slide_fraction
    if env._sim2sim_slide_fraction_applied != target_fraction:
        _reassign_sim2sim_slide_column_mix(env, target_fraction)
        env._sim2sim_slide_fraction_applied = target_fraction

    return mean_level


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
