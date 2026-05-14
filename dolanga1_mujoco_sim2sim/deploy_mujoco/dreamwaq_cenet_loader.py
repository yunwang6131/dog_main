"""Load DreamWaQ CENet (encoder + v_est + z mean) for sim2sim.

Supports:
  - ``cenet.pt`` from ``play.py`` (``cenet_state_dict``);
  - full RSL-RL ``model_*.pt`` (``actor_state_dict``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import MLP


class DreamWaQCenet(nn.Module):
    """Matches ``DreamWaQActor`` CENet inference (deterministic z = mu)."""

    def __init__(
        self,
        history_dim: int,
        encoder_hidden_dims: list[int],
        encoder_out_dim: int,
        velocity_dim: int,
        latent_dim: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.encoder = MLP(history_dim, encoder_out_dim, encoder_hidden_dims, activation)
        self.velocity_head = nn.Linear(encoder_out_dim, velocity_dim)
        self.latent_mu_head = nn.Linear(encoder_out_dim, latent_dim)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(history)
        v_est = self.velocity_head(feat)
        z = self.latent_mu_head(feat)
        return v_est, z


def _linear_indices_in_encoder(actor_sd: dict[str, torch.Tensor]) -> list[int]:
    indices: list[int] = []
    for key in actor_sd:
        if key.startswith("encoder.") and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                indices.append(int(parts[1]))
    return sorted(set(indices))


def build_cenet_from_actor_state_dict(
    actor_sd: dict[str, torch.Tensor],
    activation: str = "elu",
) -> DreamWaQCenet:
    """Rebuild CENet topology from checkpoint weights (no Isaac / TensorDict)."""
    idxs = _linear_indices_in_encoder(actor_sd)
    if len(idxs) < 2:
        raise ValueError("Checkpoint does not look like DreamWaQActor: missing encoder Linear layers.")

    shapes: list[tuple[int, int]] = []
    for idx in idxs:
        w = actor_sd[f"encoder.{idx}.weight"]
        if w.ndim != 2:
            raise ValueError(f"Bad encoder weight {idx}: shape {w.shape}")
        shapes.append((int(w.shape[0]), int(w.shape[1])))

    history_dim = shapes[0][1]
    hidden_dims = [s[0] for s in shapes[:-1]]
    encoder_out_dim = shapes[-1][0]
    if actor_sd["encoder.0.weight"].shape[1] != history_dim:
        raise ValueError("Encoder input dim inconsistent.")

    vel_w = actor_sd["velocity_head.weight"]
    lat_w = actor_sd["latent_mu_head.weight"]
    velocity_dim = int(vel_w.shape[0])
    latent_dim = int(lat_w.shape[0])

    net = DreamWaQCenet(
        history_dim=history_dim,
        encoder_hidden_dims=hidden_dims,
        encoder_out_dim=encoder_out_dim,
        velocity_dim=velocity_dim,
        latent_dim=latent_dim,
        activation=activation,
    )
    load_sd = {
        k: v
        for k, v in actor_sd.items()
        if k.startswith("encoder.") or k.startswith("velocity_head.") or k.startswith("latent_mu_head.")
    }
    net.load_state_dict(load_sd, strict=True)
    return net


def load_dreamwaq_cenet_from_rsl_checkpoint(path: str, device: torch.device | str = "cpu") -> DreamWaQCenet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(
            f"{path}: expected dict (play-exported cenet.pt or training model_*.pt), got {type(ckpt).__name__}."
        )
    if "cenet_state_dict" in ckpt:
        actor_sd = ckpt["cenet_state_dict"]
    elif "actor_state_dict" in ckpt:
        actor_sd = ckpt["actor_state_dict"]
    else:
        raise KeyError(f"{path}: need 'cenet_state_dict' (exported/cenet.pt) or 'actor_state_dict' (model_*.pt).")
    net = build_cenet_from_actor_state_dict(actor_sd)
    net.to(device)
    net.eval()
    return net
