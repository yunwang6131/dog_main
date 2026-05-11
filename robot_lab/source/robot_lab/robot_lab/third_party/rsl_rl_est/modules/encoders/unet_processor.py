import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from tensordict import TensorDict

from .encoder import Encoder
from robot_lab.third_party.rsl_rl_est.networks.unet import UNet  # <- 按你实际路径改


class UNetProcessor(Encoder):
    """
    输入 obs_tensor: [B, X] 或 [B,C,H,W] 或 [B,H,W,C]（可选）
    强制 reshape 到 [B,C,H,W] -> UNet -> 输出 latent

    network_cfg 关键字段：
      - input_dim: [H, W]
      - input_channels: C
      - channel_last: bool  (如果输入是 BHWC 或 flatten 时按 HWC 展开)
      - flatten_output: bool (True: 输出 [B, outC*H*W]；False: 输出 [B,outC,H,W])
      - out_channels: int (UNet 输出通道)
      - base_channels, depth, norm, activation, upsample_mode, kernel_size, dilation
      - normalize_output (L2), clip_range
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

        # share network
        if network_cfg.get("share_network", False):
            self.encoder = network_cfg["shared_network"]
            self._use_shared = True
            return
        self._use_shared = False

        self._cfg = dict(network_cfg)

        # clip / l2
        self.use_l2_norm = bool(network_cfg.get("normalize_output", False))
        clip_range = network_cfg.get("clip_range", None)
        if clip_range is not None:
            assert isinstance(clip_range, (list, tuple)) and len(clip_range) == 2
            self.clip_min, self.clip_max = float(clip_range[0]), float(clip_range[1])
        else:
            self.clip_min = None
            self.clip_max = None

        # input shape
        self.input_dim: Tuple[int, int] = tuple(network_cfg["input_dim"])  # (H,W)
        self.input_channels: int = int(network_cfg["input_channels"])
        self.channel_last: bool = bool(network_cfg.get("channel_last", False))

        # output shape
        self.flatten_output: bool = bool(network_cfg.get("flatten_output", True))
        self.out_channels: int = int(network_cfg.get("out_channels", 32))

        # unet config
        self.base_channels: int = int(network_cfg.get("base_channels", 32))
        self.depth: int = int(network_cfg.get("depth", 3))
        self.norm: str = str(network_cfg.get("norm", "none"))
        self.activation: str = str(network_cfg.get("activation", "elu"))
        self.upsample_mode: str = str(network_cfg.get("upsample_mode", "bilinear"))
        self.kernel_size: int = int(network_cfg.get("kernel_size", 3))
        self.dilation: int = int(network_cfg.get("dilation", 1))

        # lazy build
        self._built = False
        self.unet: Optional[nn.Module] = None

    def _maybe_clip(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_min is not None or self.clip_max is not None:
            return torch.clamp(x, min=self.clip_min, max=self.clip_max)
        return x

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_l2_norm:
            return x
        # flatten: [B, D] -> ok
        # not flatten: [B, C, H, W] -> 这里默认沿最后一维做 normalize 不合适，所以我们只在 flatten 时 normalize
        if x.dim() == 2:
            return F.normalize(x, dim=-1, p=2)
        return x

    def _to_bchw(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        H, W = self.input_dim
        C = self.input_channels

        # already 4D
        if obs_tensor.dim() == 4:
            # BCHW
            if obs_tensor.shape[1:] == (C, H, W):
                return obs_tensor
            # BHWC
            if obs_tensor.shape[1:] == (H, W, C):
                return obs_tensor.permute(0, 3, 1, 2).contiguous()
            raise ValueError(f"Unexpected 4D shape {tuple(obs_tensor.shape)}")

        # flat: [B, X]
        if obs_tensor.dim() == 2:
            B, X = obs_tensor.shape
            expected = C * H * W
            if X != expected:
                raise ValueError(f"Flat dim={X}, expected C*H*W={expected} (C={C},H={H},W={W})")

            if self.channel_last:
                # flat is HWC
                return obs_tensor.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            # flat is CHW
            return obs_tensor.view(B, C, H, W)

        raise ValueError(f"Unsupported obs_tensor dim={obs_tensor.dim()}")

    def _build_if_needed(self, x: torch.Tensor):
        if self._built:
            return
        device = x.device

        self.unet = UNet(
            in_channels=self.input_channels,
            out_channels=self.out_channels,
            base_channels=self.base_channels,
            depth=self.depth,
            norm=self.norm,
            activation=self.activation,
            upsample_mode=self.upsample_mode,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
        ).to(device)

        # 计算并写回 latent_dim（便于日志/对齐）
        H, W = self.input_dim
        if self.flatten_output:
            self.latent_dim = int(self.out_channels * H * W)
        else:
            self.latent_dim = (self.out_channels, H, W)

        self._built = True

    def encode(self, obs: TensorDict, **kwargs) -> TensorDict:
        obs_tensor = self.get_obs_tensor(obs)

        if self._use_shared:
            latent = self.encoder(obs_tensor)
            latent = self._maybe_clip(latent)
            latent = self._maybe_normalize(latent)
            return TensorDict({self.latent_name: latent}, batch_size=obs.batch_size)

        x = self._to_bchw(obs_tensor)
        self._build_if_needed(x)

        y = self.unet(x)  # [B, outC, H, W]

        if self.flatten_output:
            y = y.flatten(start_dim=1)  # [B, outC*H*W]

        y = self._maybe_clip(y)
        y = self._maybe_normalize(y)
        return TensorDict({self.latent_name: y}, batch_size=obs.batch_size)

    def encode_inference(self, obs: TensorDict, **kwargs) -> TensorDict:
        return self.encode(obs, **kwargs)