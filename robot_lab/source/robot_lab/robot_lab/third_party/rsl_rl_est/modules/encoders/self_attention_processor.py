import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from tensordict import TensorDict

from .encoder import Encoder
from robot_lab.third_party.rsl_rl_est.networks import SelfAttention


class SelfAttentionProcessor(Encoder):
    """
    输入 obs_tensor: [B, X]
    自动推断 T:  T = X / D  （要求 X % D == 0）
    reshape: [B, T, D]
    输出:
      - flatten_output=True  -> [B, T*D] (= [B, X])
      - flatten_output=False -> [B, T, D]
    永远不使用 mask（attn_mask=None）
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

        # ========== 网络共享 ==========
        if network_cfg.get("share_network", False):
            self.encoder = network_cfg["shared_network"]
            self._use_shared = True
            return
        self._use_shared = False

        # 保存 cfg（如果你想日志写回）
        self._network_cfg = dict(network_cfg)

        # ========== L2 normalize / clip ==========
        self.use_l2_norm = bool(network_cfg.get("normalize_output", False))
        clip_range = network_cfg.get("clip_range", None)
        if clip_range is not None:
            assert isinstance(clip_range, (list, tuple)) and len(clip_range) == 2
            self.clip_min, self.clip_max = clip_range
        else:
            self.clip_min = None
            self.clip_max = None

        # ========== attention cfg ==========
        # ✅ 你要求：不指定 T，只指定 D（embed_dim）
        if "embed_dim" not in network_cfg:
            raise ValueError("SelfAttentionProcessor requires network_cfg['embed_dim'] (token dim D).")
        self.D: int = int(network_cfg["embed_dim"])

        self.num_heads: int = int(network_cfg.get("num_heads", 4))
        self.dropout: float = float(network_cfg.get("dropout", 0.0))
        self.bias: bool = bool(network_cfg.get("bias", True))
        self.causal: bool = bool(network_cfg.get("causal", False))

        # 输出形态
        self.flatten_output: bool = bool(network_cfg.get("flatten_output", True))

        # 可选 LayerNorm
        self.use_pre_ln: bool = bool(network_cfg.get("pre_ln", True))
        self.use_post_ln: bool = bool(network_cfg.get("post_ln", False))

        # Lazy build（需要看到 X 才能推断 T）
        self._built = False
        self.T: Optional[int] = None
        self._attn: Optional[SelfAttention] = None
        self._ln1: Optional[nn.LayerNorm] = None
        self._ln2: Optional[nn.LayerNorm] = None

        # 基本合法性（D、heads）
        if self.D <= 0:
            raise ValueError(f"embed_dim must be > 0, got {self.D}")
        if self.D % self.num_heads != 0:
            raise ValueError(f"embed_dim={self.D} must be divisible by num_heads={self.num_heads}")

    # ---------------- utils ----------------
    def _maybe_clip(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_min is not None or self.clip_max is not None:
            return torch.clamp(x, min=self.clip_min, max=self.clip_max)
        return x

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_l2_norm:
            # flatten_output=True: [B, *]，dim=-1 OK
            # flatten_output=False: [B,T,D]，对 token dim 做 normalize
            return F.normalize(x, dim=-1, p=2)
        return x

    def _build_if_needed(self, x_flat: torch.Tensor):
        if self._built:
            return

        device = x_flat.device  # ⭐ 关键

        if x_flat.dim() != 2:
            raise ValueError(f"Expected [B, X], got {tuple(x_flat.shape)}")

        _, X = x_flat.shape
        if X % self.D != 0:
            raise ValueError(
                f"Input dim X={X} must be divisible by embed_dim D={self.D}"
            )

        self.T = X // self.D

        self._attn = SelfAttention(
            embed_dim=self.D,
            num_heads=self.num_heads,
            dropout=self.dropout,
            bias=self.bias,
            causal=self.causal,
        ).to(device)  # ⭐⭐⭐

        if self.use_pre_ln:
            self._ln1 = nn.LayerNorm(self.D).to(device)

        if self.use_post_ln:
            self._ln2 = nn.LayerNorm(self.D).to(device)

        self._built = True

    def _flat_to_btd(self, x_flat: torch.Tensor) -> torch.Tensor:
        # [B, X] -> [B, T, D]
        B, X = x_flat.shape
        # self.T 在 build 后一定存在
        return x_flat.view(B, self.T, self.D)

    # ---------------- main ----------------
    def encode(self, obs: TensorDict, **kwargs) -> TensorDict:
        x_flat = self.get_obs_tensor(obs)  # 期望 [B, X]

        if self._use_shared:
            latent = self.encoder(x_flat)
            latent = self._maybe_clip(latent)
            latent = self._maybe_normalize(latent)
            return TensorDict({self.latent_name: latent}, batch_size=obs.batch_size)

        self._build_if_needed(x_flat)

        x = self._flat_to_btd(x_flat)  # [B,T,D]

        if self._ln1 is not None:
            x = self._ln1(x)

        # self-attention（无 mask / 无 cache）
        y = self._attn(x, attn_mask=None, cache=None, return_cache=False)  # [B,T,D]

        if self._ln2 is not None:
            y = self._ln2(y)

        if self.flatten_output:
            y = y.reshape(y.shape[0], -1)  # [B, T*D] == [B, X]

        y = self._maybe_clip(y)
        y = self._maybe_normalize(y)

        return TensorDict({self.latent_name: y}, batch_size=obs.batch_size)

    def encode_inference(self, obs: TensorDict, **kwargs) -> TensorDict:
        return self.encode(obs, **kwargs)
