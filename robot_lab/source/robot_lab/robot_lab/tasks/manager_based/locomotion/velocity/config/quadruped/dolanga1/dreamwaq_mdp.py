from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def disturbance_force_torque(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the privileged disturbance wrench applied to the selected body.

    DreamWaQ critic receives d_t as privileged state. IsaacLab stores external
    reset forces/torques in the robot's permanent wrench composer, so we expose
    that buffer as [force_b, torque_b].
    """
    asset = env.scene[asset_cfg.name]
    forces = asset.permanent_wrench_composer.composed_force_as_torch[:, asset_cfg.body_ids, :]
    torques = asset.permanent_wrench_composer.composed_torque_as_torch[:, asset_cfg.body_ids, :]
    return torch.cat([forces.flatten(start_dim=1), torques.flatten(start_dim=1)], dim=-1)

