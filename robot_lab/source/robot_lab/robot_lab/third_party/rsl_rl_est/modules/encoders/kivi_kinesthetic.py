from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

try:
    from rsl_rl.networks import MLP
except ImportError:
    from rsl_rl.modules import MLP

ObsGroupSpec = Union[str, Tuple[str, bool]]


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class KiviKinestheticEncoder(nn.Module):
    """Single KiVi kinesthetic module: history -> shared feature -> explicit estimate and VAE latent."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: List[ObsGroupSpec],
        latent_name: str,
        latent_dim: int,
        network_cfg: dict,
    ):
        super().__init__()
        self._obs_groups_spec = obs_groups
        self.obs_groups: list[str] = []
        self._detach_flags: dict[str, bool] = {}
        self.num_obs = 0
        for spec in obs_groups:
            if isinstance(spec, str):
                name, detach = spec, False
            else:
                name, detach = spec[0], spec[1]
            assert name in obs.keys(), f"Obs key '{name}' not found in TensorDict."
            x = obs[name]
            if x.dim() != 2:
                raise ValueError(f"KiviKinestheticEncoder expects 2D obs, got {name}: {tuple(x.shape)}")
            self.obs_groups.append(name)
            self._detach_flags[name] = bool(detach)
            self.num_obs += x.shape[-1]

        self.latent_name = latent_name
        self.latent_dim = latent_dim
        self.feature_name = network_cfg.get("feature_name", "kinesthetic_feature")
        self.feature_dim = int(network_cfg.get("feature_dim", 32))
        self.explicit_latent_name = network_cfg.get("explicit_latent_name", "kinesthetic_explicit_latent")
        self.explicit_dim = int(network_cfg.get("explicit_dim", 11))
        self.implicit_latent_name = network_cfg.get("implicit_latent_name", "kinesthetic_implicit_latent")
        self.implicit_dim = int(network_cfg.get("implicit_dim", latent_dim))
        self.logvar_min = float(network_cfg.get("logvar_min", -10.0))
        self.logvar_max = float(network_cfg.get("logvar_max", 4.0))
        self.use_l2_norm = bool(network_cfg.get("normalize_output", False))

        activation = network_cfg.get("activation", "elu")
        self.posterior_depth_group = network_cfg.get("posterior_depth_group", None)
        self.posterior_depth_detach = bool(network_cfg.get("posterior_depth_detach", False))
        self.posterior_depth_dim = 0

        if self.posterior_depth_group is not None:
            assert self.posterior_depth_group in obs.keys(), (
                f"Obs key '{self.posterior_depth_group}' not found in TensorDict."
            )
            depth_shape = obs[self.posterior_depth_group].shape
            if len(depth_shape) != 4:
                raise ValueError(
                    f"posterior_depth_group must be [B,C,H,W], got {self.posterior_depth_group}: {tuple(depth_shape)}"
                )
            depth_channels = int(depth_shape[1])
            self.posterior_depth_dim = int(network_cfg.get("posterior_depth_dim", 32))
            conv_channels = list(network_cfg.get("posterior_depth_channels", [8, 16, 32]))
            conv_layers: list[nn.Module] = []
            in_channels = depth_channels
            for out_channels in conv_channels:
                conv_layers.extend(
                    [
                        nn.Conv2d(in_channels, int(out_channels), kernel_size=5, stride=2, padding=2),
                        _activation(activation),
                    ]
                )
                in_channels = int(out_channels)
            conv_layers.extend(
                [
                    nn.Conv2d(in_channels, self.posterior_depth_dim, kernel_size=3, stride=1, padding=1),
                    _activation(activation),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                ]
            )
            self.posterior_depth_encoder = nn.Sequential(*conv_layers)
        else:
            self.posterior_depth_encoder = None

        self.backbone = MLP(
            input_dim=self.num_obs,
            output_dim=self.feature_dim,
            hidden_dims=network_cfg.get("hidden_dims", [512, 256]),
            activation=activation,
        )
        self.explicit_head = MLP(
            input_dim=self.feature_dim,
            output_dim=self.explicit_dim,
            hidden_dims=network_cfg.get("explicit_hidden_dims", [64]),
            activation=activation,
        )
        self.implicit_head = MLP(
            input_dim=self.feature_dim + self.posterior_depth_dim,
            output_dim=2 * self.implicit_dim,
            hidden_dims=network_cfg.get("implicit_hidden_dims", [128, 64]),
            activation=activation,
        )

    def _get_obs_tensor(self, obs: TensorDict) -> torch.Tensor:
        tensors = []
        for name in self.obs_groups:
            x = obs[name]
            if self._detach_flags.get(name, False):
                x = x.detach()
            tensors.append(x)
        return torch.cat(tensors, dim=-1)

    def _get_posterior_depth_feature(self, obs: TensorDict) -> torch.Tensor | None:
        if self.posterior_depth_encoder is None:
            return None
        depth = obs[self.posterior_depth_group]
        if self.posterior_depth_detach:
            depth = depth.detach()
        return self.posterior_depth_encoder(depth)

    def _split_implicit(self, obs: TensorDict, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth_feature = self._get_posterior_depth_feature(obs)
        posterior_feature = feature if depth_feature is None else torch.cat([feature, depth_feature], dim=-1)
        out = self.implicit_head(posterior_feature)
        mu, logvar = torch.split(out, self.implicit_dim, dim=-1)
        return mu, torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_l2_norm:
            return F.normalize(x, dim=-1, p=2)
        return x

    def encode(self, obs: TensorDict, **kwargs) -> TensorDict:
        feature = self.backbone(self._get_obs_tensor(obs))
        explicit = self.explicit_head(feature)
        mu, logvar = self._split_implicit(obs, feature)
        std = torch.exp(0.5 * logvar)
        implicit = self._maybe_normalize(mu + torch.randn_like(std) * std)
        return TensorDict(
            {
                self.feature_name: feature,
                self.explicit_latent_name: explicit,
                self.implicit_latent_name: implicit,
                self.implicit_latent_name + "_mu": mu,
                self.implicit_latent_name + "_logvar": logvar,
            },
            batch_size=obs.batch_size,
        )

    def encode_inference(self, obs: TensorDict, **kwargs) -> TensorDict:
        feature = self.backbone(self._get_obs_tensor(obs))
        explicit = self.explicit_head(feature)
        mu, logvar = self._split_implicit(obs, feature)
        return TensorDict(
            {
                self.feature_name: feature,
                self.explicit_latent_name: explicit,
                self.implicit_latent_name: self._maybe_normalize(mu),
            },
            batch_size=obs.batch_size,
        )

    def reset(self, done=None):
        pass

    def get_hidden_state(self):
        return {}

    def get_hidden_state_inference(self):
        return {}
