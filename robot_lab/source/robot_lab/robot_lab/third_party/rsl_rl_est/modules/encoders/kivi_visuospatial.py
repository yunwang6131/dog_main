from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from rsl_rl.utils import split_and_pad_trajectories, unpad_trajectories

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


class KiviVisuospatialEncoder(nn.Module):
    """KiVi-style proprio-depth fusion for terrain-aware visual latents.

    This encoder intentionally accepts mixed vector and image observations:
    proprioceptive history is compressed into one token, depth history into
    visual tokens, and learnable memory tokens give the transformer stable
    slots for terrain context. Auxiliary height heads are returned in the same
    TensorDict so their losses train the visual fusion path directly.
    """
    is_recurrent = True

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
        for spec in obs_groups:
            if isinstance(spec, str):
                name, detach = spec, False
            else:
                name, detach = spec[0], spec[1]
            assert name in obs.keys(), f"Obs key '{name}' not found in TensorDict."
            self.obs_groups.append(name)
            self._detach_flags[name] = bool(detach)

        self.latent_name = latent_name
        self.latent_dim = latent_dim

        self.depth_group = network_cfg["depth_group"]
        self.proprio_groups = list(network_cfg["proprio_groups"])
        for name in [self.depth_group, *self.proprio_groups]:
            assert name in obs.keys(), f"Obs key '{name}' not found in TensorDict."

        self.state_latent_name = network_cfg.get("state_latent_name", "visuospatial_state_latent")
        self.state_latent_dim = int(network_cfg.get("state_latent_dim", 12))
        self.foot_latent_name = network_cfg.get("foot_latent_name", "visuospatial_foot_latent")
        self.foot_latent_dim = int(network_cfg.get("foot_latent_dim", 8))
        if self.state_latent_dim + self.foot_latent_dim != latent_dim:
            raise ValueError(
                f"state_latent_dim + foot_latent_dim must equal latent_dim: "
                f"{self.state_latent_dim} + {self.foot_latent_dim} != {latent_dim}"
            )

        self.height_latent_name = network_cfg.get("height_latent_name", "height_scan_latent")
        self.height_target_group = network_cfg.get("height_target_group", "height_scan")
        self.foot_height_latent_name = network_cfg.get("foot_height_latent_name", "height_scan_feet_latent")
        self.foot_height_target_group = network_cfg.get("foot_height_target_group", "height_scan_feet")
        self.use_l2_norm = bool(network_cfg.get("normalize_output", False))

        embed_dim = int(network_cfg.get("embed_dim", 32))
        num_heads = int(network_cfg.get("num_heads", 4))
        num_layers = int(network_cfg.get("num_layers", 2))
        dim_feedforward = int(network_cfg.get("dim_feedforward", 4 * embed_dim))
        dropout = float(network_cfg.get("dropout", 0.0))
        activation = str(network_cfg.get("activation", "elu"))
        transformer_activation = str(network_cfg.get("transformer_activation", "gelu"))
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        self.num_proprio = sum(obs[name].shape[-1] for name in self.proprio_groups)
        self.proprio_encoder = MLP(
            input_dim=self.num_proprio,
            output_dim=embed_dim,
            hidden_dims=network_cfg.get("proprio_hidden_dims", [256, 128]),
            activation=activation,
        )

        depth_shape = obs[self.depth_group].shape
        if len(depth_shape) != 4:
            raise ValueError(f"{self.depth_group} must be [B,C,H,W], got {tuple(depth_shape)}")
        self.depth_history_len = int(depth_shape[1])
        conv_channels = list(network_cfg.get("conv_channels", [16, 32]))
        conv_layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in conv_channels:
            conv_layers.extend(
                [
                    nn.Conv2d(in_channels, int(out_channels), kernel_size=5, stride=2, padding=2),
                    _activation(activation),
                ]
            )
            in_channels = int(out_channels)
        conv_layers.extend([nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1), _activation(activation)])
        self.depth_encoder = nn.Sequential(*conv_layers)
        token_grid = int(network_cfg.get("visual_token_grid", 4))
        self.visual_pool = nn.AdaptiveAvgPool2d((token_grid, token_grid))

        num_memory_tokens = int(network_cfg.get("num_memory_tokens", 4))
        self.num_memory_tokens = num_memory_tokens
        self.num_transitions_per_env = int(network_cfg.get("num_transitions_per_env", 24))
        self.initial_memory_tokens = nn.Parameter(torch.zeros(num_memory_tokens, embed_dim))
        nn.init.trunc_normal_(self.initial_memory_tokens, std=0.02)
        self.hidden_states: torch.Tensor | None = None
        self.hidden_states_inference: torch.Tensor | None = None

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=transformer_activation,
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fusion_norm = nn.LayerNorm(embed_dim)

        self.state_head = MLP(
            input_dim=embed_dim,
            output_dim=self.state_latent_dim,
            hidden_dims=network_cfg.get("state_hidden_dims", network_cfg.get("vis_hidden_dims", [128])),
            activation=activation,
        )
        self.foot_head = MLP(
            input_dim=embed_dim,
            output_dim=self.foot_latent_dim,
            hidden_dims=network_cfg.get("foot_hidden_dims", network_cfg.get("vis_hidden_dims", [128])),
            activation=activation,
        )

        self.height_dim = int(obs[self.height_target_group].shape[-1]) if self.height_target_group in obs.keys() else 0
        self.height_head = (
            MLP(
                input_dim=self.state_latent_dim,
                output_dim=self.height_dim,
                hidden_dims=network_cfg.get("height_hidden_dims", [256, 256]),
                activation=activation,
            )
            if self.height_dim > 0
            else None
        )

        self.foot_height_dim = (
            int(obs[self.foot_height_target_group].shape[-1]) if self.foot_height_target_group in obs.keys() else 0
        )
        self.foot_height_head = (
            MLP(
                input_dim=self.foot_latent_dim,
                output_dim=self.foot_height_dim,
                hidden_dims=network_cfg.get("foot_height_hidden_dims", [128, 128]),
                activation=activation,
            )
            if self.foot_height_dim > 0
            else None
        )

    def _get(self, obs: TensorDict, name: str) -> torch.Tensor:
        x = obs[name]
        if self._detach_flags.get(name, False):
            return x.detach()
        return x

    def _encode_tokens(self, obs: TensorDict) -> torch.Tensor:
        proprio = torch.cat([self._get(obs, name) for name in self.proprio_groups], dim=-1)
        proprio_token = self.proprio_encoder(proprio).unsqueeze(1)

        depth = self._get(obs, self.depth_group)
        batch, history_len, height, width = depth.shape
        depth_frames = depth.reshape(batch * history_len, 1, height, width)
        visual_map = self.visual_pool(self.depth_encoder(depth_frames))
        visual_map = visual_map.view(batch, history_len, visual_map.shape[1], visual_map.shape[2], visual_map.shape[3])
        visual_map = visual_map.mean(dim=1)
        visual_tokens = visual_map.flatten(2).transpose(1, 2)
        return torch.cat([proprio_token, visual_tokens], dim=1)

    def _initial_memory(self, batch: int, device: torch.device) -> torch.Tensor:
        return self.initial_memory_tokens.unsqueeze(1).expand(-1, batch, -1).to(device)

    def _fuse_with_memory(self, base_tokens: torch.Tensor, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory_tokens = memory.transpose(0, 1)
        fused_tokens = self.fusion(torch.cat([base_tokens, memory_tokens], dim=1))
        proprio_state = self.fusion_norm(fused_tokens[:, 0])
        visual_foot = self.fusion_norm(fused_tokens[:, 1:-self.num_memory_tokens].mean(dim=1))
        next_memory = self.fusion_norm(fused_tokens[:, -self.num_memory_tokens:]).transpose(0, 1)
        return torch.cat([proprio_state, visual_foot], dim=-1), next_memory

    def _encode_recurrent(
        self,
        base_tokens: torch.Tensor,
        dones: torch.Tensor | None = None,
        hidden_states: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if dones is None:
            batch = base_tokens.shape[0]
            hidden = None if hidden_states is None else hidden_states.get(self.latent_name, None)
            if hidden is None:
                hidden = self.hidden_states
            if hidden is None or hidden.shape[1] != batch:
                hidden = self._initial_memory(batch, base_tokens.device)
            fused, next_memory = self._fuse_with_memory(base_tokens, hidden)
            if hidden_states is None:
                self.hidden_states = next_memory.detach()
            else:
                self.hidden_states_inference = next_memory.detach()
            return fused, next_memory

        if hidden_states is None or self.latent_name not in hidden_states:
            raise ValueError("Hidden states must be passed to KiviVisuospatialEncoder during policy update.")

        num_tokens, embed_dim = base_tokens.shape[1:]
        base_tokens = base_tokens.view(self.num_transitions_per_env, -1, num_tokens, embed_dim)
        dones = dones.view(self.num_transitions_per_env, -1, 1)
        tokens_traj, masks = split_and_pad_trajectories(base_tokens, dones)
        memory = hidden_states[self.latent_name]
        fused_steps = []
        for t in range(tokens_traj.shape[0]):
            fused, memory = self._fuse_with_memory(tokens_traj[t], memory)
            fused_steps.append(fused)
        fused_padded = torch.stack(fused_steps, dim=0)
        fused_flat = unpad_trajectories(fused_padded, masks)
        return fused_flat.reshape(-1, fused_flat.shape[-1]), None

    def encode(self, obs: TensorDict, dones=None, hidden_states=None, **kwargs) -> TensorDict:
        tokens = self._encode_tokens(obs)
        fused_features, _ = self._encode_recurrent(tokens, dones=dones, hidden_states=hidden_states)
        proprio_state, visual_foot = torch.split(fused_features, fused_features.shape[-1] // 2, dim=-1)

        state_latent = self.state_head(proprio_state)
        foot_latent = self.foot_head(visual_foot)
        visuospatial = torch.cat([state_latent, foot_latent], dim=-1)
        if self.use_l2_norm:
            visuospatial = F.normalize(visuospatial, dim=-1, p=2)
            state_latent = F.normalize(state_latent, dim=-1, p=2)
            foot_latent = F.normalize(foot_latent, dim=-1, p=2)

        out = TensorDict(
            {
                self.latent_name: visuospatial,
                self.state_latent_name: state_latent,
                self.foot_latent_name: foot_latent,
            },
            batch_size=obs.batch_size,
        )
        if self.height_head is not None:
            out[self.height_latent_name] = self.height_head(state_latent)
        if self.foot_height_head is not None:
            out[self.foot_height_latent_name] = self.foot_height_head(foot_latent)
        return out

    def encode_inference(self, obs: TensorDict, **kwargs) -> TensorDict:
        return self.encode(obs, **kwargs)

    def reset(self, done=None):
        if done is None:
            self.hidden_states = None
            self.hidden_states_inference = None
            return
        done = done.squeeze(-1).bool() if done.dim() > 1 else done.bool()
        if self.hidden_states is not None:
            self.hidden_states[:, done == 1, :] = 0.0
        if self.hidden_states_inference is not None:
            self.hidden_states_inference[:, done == 1, :] = 0.0

    def get_hidden_state(self):
        return {self.latent_name: self.hidden_states} if self.hidden_states is not None else {}

    def get_hidden_state_inference(self):
        return {self.latent_name: self.hidden_states_inference} if self.hidden_states_inference is not None else {}
