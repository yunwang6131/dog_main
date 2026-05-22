# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP


@dataclass
class BarrierEstimatorLosses:
    total: torch.Tensor
    velocity: torch.Tensor
    foot_contact: torch.Tensor
    terrain: torch.Tensor


class _BarrierEstimatorActorExport(nn.Module):
    is_recurrent: bool = False

    def __init__(self, model: "BarrierEstimatorActor") -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.estimator_encoder = copy.deepcopy(model.estimator_encoder)
        self.velocity_head = copy.deepcopy(model.velocity_head)
        self.foot_contact_head = copy.deepcopy(model.foot_contact_head)
        self.terrain_head = copy.deepcopy(model.terrain_head)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = int(model.obs_dim)

    def _estimate_privileged(self, raw_obs: torch.Tensor) -> torch.Tensor:
        features = self.estimator_encoder(raw_obs)
        velocity = self.velocity_head(features)
        foot_contact = torch.sigmoid(self.foot_contact_head(features))
        terrain = self.terrain_head(features)
        return torch.cat([velocity, foot_contact, terrain], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        estimated = self._estimate_privileged(x)
        latent = torch.cat([x, estimated], dim=-1)
        latent = self.obs_normalizer(latent)
        out = self.mlp(latent)
        return self.deterministic_output(out)


class _TorchBarrierEstimatorActor(_BarrierEstimatorActorExport):
    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxBarrierEstimatorActor(_BarrierEstimatorActorExport):
    def __init__(self, model: "BarrierEstimatorActor", verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class BarrierEstimatorActor(MLPModel):
    """Actor that predicts privileged locomotion state from deployable observations."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        estimator_input_groups: list[str] | None = None,
        velocity_target_group: str = "base_lin_vel",
        foot_contact_target_group: str = "foot_contact_state",
        terrain_target_group: str = "height_scan_feet",
        estimator_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        estimator_activation: str = "elu",
        velocity_loss_weight: float = 1.0,
        foot_contact_loss_weight: float = 1.0,
        terrain_loss_weight: float = 1.0,
        hidden_dims: tuple[int, ...] | list[int] = (256, 128, 64),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        self.estimator_input_groups = list(estimator_input_groups or obs_groups[obs_set])
        self.velocity_target_group = velocity_target_group
        self.foot_contact_target_group = foot_contact_target_group
        self.terrain_target_group = terrain_target_group
        self.velocity_loss_weight = float(velocity_loss_weight)
        self.foot_contact_loss_weight = float(foot_contact_loss_weight)
        self.terrain_loss_weight = float(terrain_loss_weight)

        self.estimator_input_dim = self._sum_obs_dim(obs, self.estimator_input_groups)
        self.velocity_dim = self._sum_obs_dim(obs, [self.velocity_target_group])
        self.foot_contact_dim = self._sum_obs_dim(obs, [self.foot_contact_target_group])
        self.terrain_dim = self._sum_obs_dim(obs, [self.terrain_target_group])
        self.estimated_privileged_dim = self.velocity_dim + self.foot_contact_dim + self.terrain_dim

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        self.estimator_encoder = MLP(
            self.estimator_input_dim,
            int(estimator_hidden_dims[-1]),
            estimator_hidden_dims,
            estimator_activation,
        )
        estimator_output_dim = int(estimator_hidden_dims[-1])
        self.velocity_head = nn.Linear(estimator_output_dim, self.velocity_dim)
        self.foot_contact_head = nn.Linear(estimator_output_dim, self.foot_contact_dim)
        self.terrain_head = nn.Linear(estimator_output_dim, self.terrain_dim)

    @staticmethod
    def _sum_obs_dim(obs: TensorDict, groups: list[str]) -> int:
        dim = 0
        for group in groups:
            if group not in obs:
                raise KeyError(f"Observation group '{group}' is required by BarrierEstimatorActor but is missing.")
            if len(obs[group].shape) != 2:
                raise ValueError(
                    f"BarrierEstimatorActor only supports 1D observations, got {obs[group].shape} for {group}."
                )
            dim += int(obs[group].shape[-1])
        return dim

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.estimated_privileged_dim

    def _cat_obs(self, obs: TensorDict, groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[group] for group in groups], dim=-1)

    def _estimate_privileged_from_obs(
        self, obs: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        estimator_obs = self._cat_obs(obs, self.estimator_input_groups)
        estimator_features = self.estimator_encoder(estimator_obs)
        velocity = self.velocity_head(estimator_features)
        foot_contact_logits = self.foot_contact_head(estimator_features)
        foot_contact = torch.sigmoid(foot_contact_logits)
        terrain = self.terrain_head(estimator_features)
        estimated = torch.cat([velocity, foot_contact, terrain], dim=-1)
        return estimated, velocity, foot_contact_logits, terrain

    def _build_actor_latent(self, obs: TensorDict, normalize: bool) -> torch.Tensor:
        actor_obs = self._cat_obs(obs, self.obs_groups)
        estimated, _, _, _ = self._estimate_privileged_from_obs(obs)
        latent = torch.cat([actor_obs, estimated], dim=-1)
        if normalize:
            return self.obs_normalizer(latent)
        return latent

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
    ) -> torch.Tensor:
        return self._build_actor_latent(obs, normalize=True)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            latent = self._build_actor_latent(obs, normalize=False)
            self.obs_normalizer.update(latent)  # type: ignore[arg-type]

    def compute_estimator_losses(self, obs: TensorDict) -> BarrierEstimatorLosses:
        _, velocity, foot_contact_logits, terrain = self._estimate_privileged_from_obs(obs)
        velocity_target = obs[self.velocity_target_group]
        foot_contact_target = obs[self.foot_contact_target_group]
        terrain_target = obs[self.terrain_target_group]

        velocity_loss = F.mse_loss(velocity, velocity_target)
        foot_contact_loss = F.binary_cross_entropy_with_logits(foot_contact_logits, foot_contact_target)
        terrain_loss = F.mse_loss(terrain, terrain_target)
        total = (
            self.velocity_loss_weight * velocity_loss
            + self.foot_contact_loss_weight * foot_contact_loss
            + self.terrain_loss_weight * terrain_loss
        )
        return BarrierEstimatorLosses(
            total=total,
            velocity=velocity_loss,
            foot_contact=foot_contact_loss,
            terrain=terrain_loss,
        )

    def as_jit(self) -> nn.Module:
        return _TorchBarrierEstimatorActor(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxBarrierEstimatorActor(self, verbose)
