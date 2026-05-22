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
class DreamWaQLoss:
    total: torch.Tensor
    velocity: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor


class _OnnxDreamWaQActor(nn.Module):
    """ONNX export wrapper: MLP head expects CENet output, not raw actor obs_dim.

    Input layout matches :meth:`DreamWaQActor.get_latent`: ``concat(current, v_est, z)``.
    External force / ``disturbance`` is *not* included; that is critic-only in this setup.
    """

    is_recurrent: bool = False

    def __init__(self, model: "DreamWaQActor", verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = int(model._get_latent_dim())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class DreamWaQActor(MLPModel):
    """Actor with a DreamWaQ-style context-aided estimator network.

    The actor head consumes the current proprioceptive observation plus the
    CENet estimates:
      - body linear velocity estimate
      - latent terrain/context vector

    The auxiliary loss follows the DreamWaQ CENet objective:
      MSE(v_est, v_gt) + MSE(o_next_recon, o_next) + beta * KL(q(z|o_H) || N(0, I)).
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
        encoder_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        decoder_hidden_dims: list[int] | tuple[int, ...] = (128, 256),
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

        self.encoder = MLP(self.history_dim, encoder_hidden_dims[-1], encoder_hidden_dims, activation)
        encoder_out_dim = int(encoder_hidden_dims[-1])
        self.velocity_head = torch.nn.Linear(encoder_out_dim, self.velocity_dim)
        self.latent_mu_head = torch.nn.Linear(encoder_out_dim, self.latent_dim)
        self.latent_logvar_head = torch.nn.Linear(encoder_out_dim, self.latent_dim)
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
                raise KeyError(f"Observation group '{group}' is required by DreamWaQActor but is missing.")
            if len(obs[group].shape) != 2:
                raise ValueError(f"DreamWaQActor only supports 1D observations, got {obs[group].shape} for {group}.")
            dim += int(obs[group].shape[-1])
        return dim

    def _get_latent_dim(self) -> int:
        return self.current_dim + self.velocity_dim + self.latent_dim

    def _cat_obs(self, obs: TensorDict, groups: list[str]) -> torch.Tensor:
        return torch.cat([obs[group] for group in groups], dim=-1)

    def _encode_context(self, obs: TensorDict, sample_latent: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # Keep policy logits deterministic for PPO log-prob recomputation; stochasticity is handled by the action distribution.
        velocity, latent, _, _ = self._encode_context(obs, sample_latent=False)
        actor_latent = torch.cat([current, velocity, latent], dim=-1)
        return self.obs_normalizer(actor_latent)

    def compute_cenet_loss(self, obs: TensorDict) -> DreamWaQLoss:
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
        return DreamWaQLoss(total=total, velocity=velocity_loss, reconstruction=reconstruction_loss, kl=kl_loss)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxDreamWaQActor(self, verbose)

