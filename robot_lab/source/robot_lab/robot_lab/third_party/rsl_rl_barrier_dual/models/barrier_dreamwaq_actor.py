# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP


@dataclass
class BarrierDreamWaQLoss:
    total: torch.Tensor
    velocity: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor


class BarrierDreamWaQDeployWrapper(nn.Module):
    """Merged CENet + actor head for deployment: history + current -> actions."""

    is_recurrent: bool = False

    def __init__(self, actor: "BarrierDreamWaQActor") -> None:
        super().__init__()
        self.history_dim = int(actor.history_dim)
        self.current_dim = int(actor.current_dim)

        self.encoder = copy.deepcopy(actor.encoder)
        self.velocity_head = copy.deepcopy(actor.velocity_head)
        self.latent_mu_head = copy.deepcopy(actor.latent_mu_head)
        self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)
        self.mlp = copy.deepcopy(actor.mlp)
        if actor.distribution is not None:
            self.deterministic_output = actor.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, history: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        features = self.encoder(history)
        velocity = self.velocity_head(features)
        latent = self.latent_mu_head(features)
        x = torch.cat([current, velocity, latent], dim=-1)
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)

    def get_dummy_inputs(self, batch: int = 1, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        dev = device or next(self.parameters()).device
        history = torch.zeros(batch, self.history_dim, device=dev, dtype=torch.float32)
        current = torch.zeros(batch, self.current_dim, device=dev, dtype=torch.float32)
        return history, current


class _OnnxBarrierDreamWaQActor(BarrierDreamWaQDeployWrapper):
    def __init__(self, actor: "BarrierDreamWaQActor", verbose: bool) -> None:
        super().__init__(actor)
        self.verbose = verbose

    @property
    def input_names(self) -> list[str]:
        return ["history", "current"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class BarrierDreamWaQActor(MLPModel):
    """BarrierDual actor with a DreamWaQ-style CENet.

    PPO acts on current proprioception, while the actor internally encodes
    proprioceptive history into a velocity estimate and latent context.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        history_groups: list[str],
        current_groups: list[str],
        velocity_target_group: str = "base_lin_vel",
        reconstruction_target_groups: list[str] | None = None,
        latent_dim: int = 16,
        velocity_dim: int = 3,
        encoder_hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        decoder_hidden_dims: list[int] | tuple[int, ...] = (256, 512),
        beta_kl: float = 0.01,
        velocity_loss_weight: float = 1.0,
        reconstruction_loss_weight: float = 1.0,
        kl_loss_weight: float = 1.0,
        logvar_min: float = -10.0,
        logvar_max: float = 4.0,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        self.history_groups = list(history_groups)
        self.current_groups = list(current_groups)
        self.velocity_target_group = velocity_target_group
        self.reconstruction_target_groups = list(reconstruction_target_groups or current_groups)
        self.latent_dim = int(latent_dim)
        self.velocity_dim = int(velocity_dim)
        self.beta_kl = float(beta_kl)
        self.velocity_loss_weight = float(velocity_loss_weight)
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.kl_loss_weight = float(kl_loss_weight)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.history_dim = self._sum_obs_dim(obs, self.history_groups)
        self.current_dim = self._sum_obs_dim(obs, self.current_groups)
        self.reconstruction_dim = self._sum_obs_dim(obs, self.reconstruction_target_groups)

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

        self.encoder = MLP(self.history_dim, int(encoder_hidden_dims[-1]), encoder_hidden_dims, activation)
        encoder_out_dim = int(encoder_hidden_dims[-1])
        self.velocity_head = nn.Linear(encoder_out_dim, self.velocity_dim)
        self.latent_mu_head = nn.Linear(encoder_out_dim, self.latent_dim)
        self.latent_logvar_head = nn.Linear(encoder_out_dim, self.latent_dim)
        self.reconstruction_decoder = MLP(
            self.velocity_dim + self.latent_dim,
            self.reconstruction_dim,
            decoder_hidden_dims,
            activation,
        )

    @staticmethod
    def _sum_obs_dim(obs: TensorDict, groups: list[str]) -> int:
        dim = 0
        for group in groups:
            if group not in obs:
                raise KeyError(f"Observation group '{group}' is required by BarrierDreamWaQActor but is missing.")
            if len(obs[group].shape) != 2:
                raise ValueError(
                    f"BarrierDreamWaQActor only supports 1D observations, got {obs[group].shape} for {group}."
                )
            dim += int(obs[group].shape[-1])
        return dim

    def _get_latent_dim(self) -> int:
        return self.current_dim + self.velocity_dim + self.latent_dim

    def _cat_obs(self, obs: TensorDict, groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[group] for group in groups], dim=-1)

    def _encode_context(
        self, obs: TensorDict, sample_latent: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history = self._cat_obs(obs, self.history_groups)
        features = self.encoder(history)
        velocity = self.velocity_head(features)
        mu = self.latent_mu_head(features)
        logvar = torch.clamp(self.latent_logvar_head(features), self.logvar_min, self.logvar_max)
        if sample_latent:
            std = torch.exp(0.5 * logvar)
            latent = mu + torch.randn_like(std) * std
        else:
            latent = mu
        return velocity, latent, mu, logvar

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None) -> torch.Tensor:
        current = self._cat_obs(obs, self.current_groups)
        velocity, latent, _, _ = self._encode_context(obs, sample_latent=False)
        actor_latent = torch.cat([current, velocity, latent], dim=-1)
        return self.obs_normalizer(actor_latent)

    def compute_cenet_loss(self, obs: TensorDict) -> BarrierDreamWaQLoss:
        velocity, latent, mu, logvar = self._encode_context(obs, sample_latent=True)
        velocity_target = obs[self.velocity_target_group]
        velocity_loss = F.mse_loss(velocity, velocity_target)

        recon_target_groups = [group + "_next" for group in self.reconstruction_target_groups]
        recon_target = self._cat_obs(obs, recon_target_groups)
        reconstruction = self.reconstruction_decoder(torch.cat([velocity, latent], dim=-1))
        reconstruction_loss = F.mse_loss(reconstruction, recon_target)

        kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl_per_sample.mean()
        total = (
            self.velocity_loss_weight * velocity_loss
            + self.reconstruction_loss_weight * reconstruction_loss
            + self.kl_loss_weight * self.beta_kl * kl_loss
        )
        return BarrierDreamWaQLoss(total=total, velocity=velocity_loss, reconstruction=reconstruction_loss, kl=kl_loss)

    def as_jit(self) -> nn.Module:
        return BarrierDreamWaQDeployWrapper(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxBarrierDreamWaQActor(self, verbose)


def barrier_dreamwaq_actor_from_runner(runner: Any) -> BarrierDreamWaQActor | None:
    actor = getattr(runner.alg, "actor", None)
    return actor if isinstance(actor, BarrierDreamWaQActor) else None


def export_barrier_dreamwaq_full_policy_jit(
    wrapper: BarrierDreamWaQDeployWrapper,
    jit_path: str,
    *,
    device: torch.device | None = None,
) -> None:
    wrapper.eval()
    dev = device or next(wrapper.parameters()).device
    history, current = wrapper.get_dummy_inputs(batch=1, device=dev)
    traced = torch.jit.trace(wrapper, (history, current))
    parent = os.path.dirname(os.path.abspath(jit_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    traced.save(jit_path)


def export_barrier_dreamwaq_merged_jit_from_runner(
    runner: Any, export_model_dir: str, filename: str = "policy_full.pt"
) -> bool:
    actor = barrier_dreamwaq_actor_from_runner(runner)
    if actor is None:
        return False
    wrapper = BarrierDreamWaQDeployWrapper(actor).cpu()
    path = os.path.join(export_model_dir, filename)
    export_barrier_dreamwaq_full_policy_jit(wrapper, path, device=torch.device("cpu"))
    print(f"[INFO] Exported merged BarrierDreamWaQ TorchScript (history+current -> actions): {path}")
    return True
