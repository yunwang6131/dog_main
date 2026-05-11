import torch
import torch.nn.functional as F
from typing import List, Tuple
from tensordict import TensorDict

from .encoder import Encoder
from robot_lab.third_party.rsl_rl_est.networks import CNN

# 假设 CNN 已正确导入
# from xxx import CNN


class CNNProcessor(Encoder):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: List[str],
        latent_name: str,
        latent_dim: int,      # 这里可以保留接口一致，但不再用于映射
        network_cfg: dict,
    ):
        super().__init__(obs, obs_groups, latent_name, latent_dim)

        # ========== 网络共享 ==========
        if network_cfg.get("share_network", False):
            self.encoder = network_cfg["shared_network"]
            self._use_shared = True
            return
        self._use_shared = False

        # ========== L2 normalize / clip ==========
        self.use_l2_norm = network_cfg.get("normalize_output", False)
        clip_range = network_cfg.get("clip_range", None)
        if clip_range is not None:
            assert isinstance(clip_range, (list, tuple)) and len(clip_range) == 2
            self.clip_min, self.clip_max = clip_range
        else:
            self.clip_min = None
            self.clip_max = None

        # ========== 输入形状 ==========
        self.input_dim: Tuple[int, int] = tuple(network_cfg["input_dim"])      # (H, W)
        self.input_channels: int = int(network_cfg["input_channels"])
        self.channel_last: bool = bool(network_cfg.get("channel_last", False))

        # ========== 是否 flatten ==========
        self.flatten: bool = bool(network_cfg.get("flatten", True))

        # ========== CNN (无 head) ==========
        self.cnn = CNN(
            input_dim=self.input_dim,
            input_channels=self.input_channels,
            output_channels=network_cfg.get("output_channels", [32, 64, 128]),
            kernel_size=network_cfg.get("kernel_size", 3),
            stride=network_cfg.get("stride", 1),
            dilation=network_cfg.get("dilation", 1),
            padding=network_cfg.get("padding", "none"),
            norm=network_cfg.get("norm", "none"),
            activation=network_cfg.get("activation", "elu"),
            max_pool=network_cfg.get("max_pool", False),
            global_pool=network_cfg.get("global_pool", "none"),
            flatten=self.flatten,
            channel_last=network_cfg.get("channel_last", False),
        )
        if network_cfg.get("init_cnn_weights", True):
            self.cnn.init_weights()

        if self.flatten:
            # output_dim 是 int
            self.latent_dim = int(self.cnn.output_dim)
        else:
            # output_dim 是 (H_out, W_out)，channels 用 cnn.output_channels
            H_out, W_out = self.cnn.output_dim
            C_out = int(self.cnn.output_channels)
            # 这里 latent_dim 你可以存 tuple，或者存 C_out*H_out*W_out 看你上游怎么用
            # 推荐存结构化信息，避免误用
            self.latent_dim = (C_out, int(H_out), int(W_out))

            # （可选）把推断结果写回 cfg，方便日志/对齐
        network_cfg["resolved_latent_dim"] = self.latent_dim

    def _maybe_clip(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_min is not None or self.clip_max is not None:
            return torch.clamp(x, min=self.clip_min, max=self.clip_max)
        return x

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_l2_norm:
            # flatten=True: (B, D) -> ok
            # flatten=False: (B, C, H, W) -> 沿通道/空间整体做向量归一化通常不合适
            # 这里默认只对最后一维做 normalize；如果你 flatten=False 还想 normalize，
            # 更推荐你在上游自己定义策略（比如按 channel 做 normalize）。
            return F.normalize(x, dim=-1, p=2)
        return x

    def _to_bchw(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        H, W = self.input_dim
        C = self.input_channels

        if obs_tensor.dim() == 4:
            if obs_tensor.shape[1] == C and obs_tensor.shape[2] == H and obs_tensor.shape[3] == W:
                return obs_tensor
            else:
                raise ValueError(f"Unexpected 4D shape {tuple(obs_tensor.shape)}")

        if obs_tensor.dim() == 2:
            B, N = obs_tensor.shape
            expected = C * H * W
            if N != expected:
                raise ValueError(f"Flat dim={N}, expected C*H*W={expected}")
            return obs_tensor.view(B, C, H, W)

        raise ValueError(f"Unsupported obs_tensor dim={obs_tensor.dim()}")

    def encode(self, obs: TensorDict, **kwargs) -> TensorDict:
        obs_tensor = self.get_obs_tensor(obs)

        if self._use_shared:
            latent = self.encoder(obs_tensor)
        else:
            x = self._to_bchw(obs_tensor)
            latent = self.cnn(x)   # ✅ 纯 CNN 输出，无 head

        latent = self._maybe_clip(latent)
        latent = self._maybe_normalize(latent)
        return TensorDict({self.latent_name: latent}, batch_size=obs.batch_size)

    def encode_inference(self, obs: TensorDict, **kwargs) -> TensorDict:
        return self.encode(obs, **kwargs)