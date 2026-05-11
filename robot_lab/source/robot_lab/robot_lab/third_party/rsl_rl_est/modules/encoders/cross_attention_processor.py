from .encoder import Encoder
from tensordict import TensorDict
from typing import List
import torch
import torch.nn as nn
from robot_lab.third_party.rsl_rl_est.networks import CrossAttention

class CrossAttentionProcessor(Encoder):
    """
    输入: x_flat [B, X]
    切分:
        q  = x_flat[:, :D]      -> [B,1,D]
        kv = x_flat[:, D:]      -> [B,Tk,D]
    输出:
        flatten_output=True  -> [B, D]
        flatten_output=False -> [B,1,D]
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: List[str],
        latent_name: str,
        latent_dim: int,
        network_cfg: dict,
    ):
        super().__init__(obs, obs_groups, latent_name, latent_dim)

        if "embed_dim" not in network_cfg:
            raise ValueError("CrossAttentionProcessor requires network_cfg['embed_dim'].")

        self.D = int(network_cfg["embed_dim"])
        self.num_heads = int(network_cfg.get("num_heads", 4))
        self.dropout = float(network_cfg.get("dropout", 0.0))
        self.flatten_output = bool(network_cfg.get("flatten_output", True))

        self.use_pre_ln = bool(network_cfg.get("pre_ln", True))
        self.use_post_ln = bool(network_cfg.get("post_ln", False))

        self._built = False
        self._attn = None
        self._ln_q = None
        self._ln_kv = None
        self._ln_out = None

    def _build_if_needed(self, x_flat: torch.Tensor):
        if self._built:
            return

        device = x_flat.device
        B, X = x_flat.shape

        if (X - self.D) <= 0 or (X - self.D) % self.D != 0:
            raise ValueError(
                f"X={X} must satisfy X = D + k*D (D={self.D})"
            )

        self._attn = CrossAttention(
            embed_dim=self.D,
            num_heads=self.num_heads,
            dropout=self.dropout,
        ).to(device)

        if self.use_pre_ln:
            self._ln_q = nn.LayerNorm(self.D).to(device)
            self._ln_kv = nn.LayerNorm(self.D).to(device)

        if self.use_post_ln:
            self._ln_out = nn.LayerNorm(self.D).to(device)

        self._built = True

    def encode(self, obs: TensorDict, **kwargs) -> TensorDict:
        x_flat = self.get_obs_tensor(obs)  # [B,X]
        self._build_if_needed(x_flat)

        B, X = x_flat.shape
        D = self.D
        Tk = (X - D) // D

        q = x_flat[:, :D].view(B, 1, D)          # [B,1,D]
        kv = x_flat[:, D:].view(B, Tk, D)        # [B,Tk,D]

        if self._ln_q is not None:
            q = self._ln_q(q)
            kv = self._ln_kv(kv)

        y = self._attn(q, kv)                    # [B,1,D]

        if self._ln_out is not None:
            y = self._ln_out(y)

        if self.flatten_output:
            y = y.squeeze(1)                     # [B,D]

        return TensorDict({self.latent_name: y}, batch_size=obs.batch_size)

    def encode_inference(self, obs: TensorDict, **kwargs) -> TensorDict:
        return self.encode(obs, **kwargs)