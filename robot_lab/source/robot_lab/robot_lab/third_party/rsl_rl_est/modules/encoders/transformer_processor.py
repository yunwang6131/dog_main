import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from tensordict import TensorDict

from .encoder import Encoder


class TransformerProcessor(Encoder):
    """
    输入 obs_tensor: [B, X]
    自动推断 T = X / D（要求 X % D == 0）
    reshape: [B, T, D]

    使用 torch.nn.TransformerEncoder:
      - 不用 mask（src_key_padding_mask=None）
      - batch_first=True

    输出:
      - pooling="none" 且 flatten_output=True  -> [B, T*D] (= [B, X])
      - pooling="none" 且 flatten_output=False -> [B, T, D]
      - pooling="mean_token"/"first_token"     -> [B, D]（flatten_output 无意义）
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

        # ========== transformer cfg ==========
        if "embed_dim" not in network_cfg:
            raise ValueError("TransformerProcessor requires network_cfg['embed_dim'] (token dim D).")
        self.D: int = int(network_cfg["embed_dim"])
        self.num_heads: int = int(network_cfg.get("num_heads", 4))
        self.num_layers: int = int(network_cfg.get("num_layers", 2))
        self.dim_feedforward: int = int(network_cfg.get("dim_feedforward", 4 * self.D))
        self.dropout: float = float(network_cfg.get("dropout", 0.0))

        # PyTorch TransformerEncoderLayer activation: "relu" or "gelu"
        self.activation: str = str(network_cfg.get("activation", "gelu")).lower()
        if self.activation not in ("relu", "gelu"):
            raise ValueError("TransformerProcessor: network_cfg['activation'] must be 'relu' or 'gelu'.")

        # norm_first=True -> Pre-LN, False -> Post-LN
        self.norm_first: bool = bool(network_cfg.get("norm_first", True))

        # 输出形态
        self.flatten_output: bool = bool(network_cfg.get("flatten_output", True))
        self.pooling: str = str(network_cfg.get("pooling", "none")).lower()
        if self.pooling not in ("none", "mean_token", "first_token"):
            raise ValueError(f"Unsupported pooling='{self.pooling}'")

        # lazy build
        self._built = False
        self.T: Optional[int] = None
        self._encoder: Optional[nn.TransformerEncoder] = None

        # 基本合法性
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
            return F.normalize(x, dim=-1, p=2)
        return x

    def _build_if_needed(self, x_flat: torch.Tensor):
        if self._built:
            return

        device = x_flat.device

        if x_flat.dim() != 2:
            raise ValueError(f"Expected [B, X], got {tuple(x_flat.shape)}")

        _, X = x_flat.shape
        if X % self.D != 0:
            raise ValueError(f"Input dim X={X} must be divisible by embed_dim D={self.D}")
        self.T = X // self.D

        layer = nn.TransformerEncoderLayer(
            d_model=self.D,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,   # 关键：输入输出 [B,T,D]
            norm_first=self.norm_first,
        )
        self._encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers).to(device)

        self._built = True

    def _flat_to_btd(self, x_flat: torch.Tensor) -> torch.Tensor:
        B, _ = x_flat.shape
        return x_flat.view(B, self.T, self.D)

    def _apply_pooling(self, y: torch.Tensor) -> torch.Tensor:
        if self.pooling == "none":
            return y
        if self.pooling == "mean_token":
            return y.mean(dim=1)      # [B,D]
        if self.pooling == "first_token":
            return y[:, 0, :]         # [B,D]
        return y

    # ---------------- main ----------------
    def encode(self, obs: TensorDict, **kwargs) -> TensorDict:
        x_flat = self.get_obs_tensor(obs)  # [B, X]

        if self._use_shared:
            latent = self.encoder(x_flat)
            latent = self._maybe_clip(latent)
            latent = self._maybe_normalize(latent)
            return TensorDict({self.latent_name: latent}, batch_size=obs.batch_size)

        self._build_if_needed(x_flat)

        x = self._flat_to_btd(x_flat)              # [B,T,D]
        y = self._encoder(x, src_key_padding_mask=None)  # [B,T,D]

        y = self._apply_pooling(y)                 # [B,T,D] or [B,D]

        # pooling 后如果变成 [B,D]，就别 flatten 了
        if y.dim() == 3 and self.flatten_output:
            y = y.reshape(y.shape[0], -1)          # [B, T*D] == [B,X]

        y = self._maybe_clip(y)
        y = self._maybe_normalize(y)

        return TensorDict({self.latent_name: y}, batch_size=obs.batch_size)

    def encode_inference(self, obs: TensorDict, **kwargs) -> TensorDict:
        return self.encode(obs, **kwargs)