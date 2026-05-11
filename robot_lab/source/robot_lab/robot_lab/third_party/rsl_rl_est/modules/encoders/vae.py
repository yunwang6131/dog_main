import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from tensordict import TensorDict

from .encoder import Encoder
from robot_lab.third_party.rsl_rl_est.networks import MLP


class VAE(Encoder):
    """
    VAE-style encoder:

    - 输入: 从 obs_groups 收集到的观测 (flatten 后维度 = num_obs)
    - 网络: encoder(x) -> [mu, logvar] (拼在一起, 维度 2 * latent_dim)
    - encode():
        * 采样 latent: z = mu + eps * exp(0.5 * logvar)
        * 返回 {latent_name: z, latent_name + '_mu': mu, latent_name + '_logvar': logvar}
    - encode_inference():
        * 只用均值: 返回 {latent_name: mu}
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: List[str],
        latent_name: str,
        latent_dim: int,
        network_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(obs, obs_groups, latent_name, latent_dim)

        if network_cfg is None:
            network_cfg = {}

        # ====== 网络部分：支持 shared_network 或自己建一个 MLP ======
        share_network = network_cfg.get("share_network", False)
        if share_network:
            # 要求外部在 network_cfg["shared_network"] 里给一个 backbone，
            # 输出维度必须是 2 * latent_dim
            if "shared_network" not in network_cfg:
                raise ValueError(
                    "VAE: share_network=True, but 'shared_network' not provided in network_cfg."
                )
            self.encoder = network_cfg["shared_network"]
        else:
            hidden_dims = network_cfg.get("hidden_dims", [256, 256])
            activation = network_cfg.get("activation", "elu")
            last_activation = network_cfg.get("last_activation", None)

            # 这里直接输出 2 * latent_dim，然后一半 mu，一半 logvar
            self.encoder = MLP(
                input_dim=self.num_obs,
                output_dim=2 * latent_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                last_activation=last_activation,
            )

        # ====== 输出 latent 的归一化 / 截断设置（参考 Processor） ======
        self.use_l2_norm = network_cfg.get("normalize_output", False)

        clip_range = network_cfg.get("clip_range", None)
        if clip_range is not None:
            assert isinstance(clip_range, (list, tuple)) and len(clip_range) == 2, \
                f"clip_range must be [min, max], got {clip_range}"
            self.clip_min, self.clip_max = float(clip_range[0]), float(clip_range[1])
        else:
            self.clip_min, self.clip_max = None, None

        # 可选: 对 logvar 做 clamp 防止数值爆炸
        self.logvar_min = float(network_cfg.get("logvar_min", -10.0))
        self.logvar_max = float(network_cfg.get("logvar_max", 4.0))

    # ====== 工具函数：对 latent 做 clip / normalize ======
    def _maybe_clip(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_min is not None or self.clip_max is not None:
            return torch.clamp(x, min=self.clip_min, max=self.clip_max)
        return x

    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_l2_norm:
            return F.normalize(x, dim=-1, p=2)
        return x

    # ====== encode: 训练用，带 reparameterization ======
    def encode(self, obs: TensorDict, **kwargs) -> TensorDict:
        """
        训练阶段：
        - 从 obs 取出 obs_groups 对应的观测并 flatten
        - 通过 encoder 得到 [mu, logvar]
        - 采样 z = mu + eps * exp(0.5*logvar)
        - 返回 z, mu, logvar
        """
        obs_tensor = self.get_obs_tensor(obs)          # [B, num_obs]
        out = self.encoder(obs_tensor)                 # [B, 2 * latent_dim]

        if out.shape[-1] != 2 * self.latent_dim:
            raise RuntimeError(
                f"VAE encoder output dim = {out.shape[-1]}, expected 2 * latent_dim = {2 * self.latent_dim}"
            )

        mu, logvar = torch.split(out, self.latent_dim, dim=-1)

        # clamp logvar 防止数值过大/过小
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)

        # reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # 对最终 latent 做可选的 clip / normalize
        z = self._maybe_clip(z)
        z = self._maybe_normalize(z)

        return TensorDict(
            {
                self.latent_name: z,
                self.latent_name + "_mu": mu,
                self.latent_name + "_logvar": logvar,
            },
            batch_size=obs.batch_size,
        )

    # ====== encode_inference: 推理用，不采样，用均值 ======
    def encode_inference(self, obs: TensorDict, **kwargs) -> TensorDict:
        """
        推理阶段：
        - 同样算出 mu / logvar，但不采样，直接使用 mu 作为 latent。
        - 按你的要求，只返回 {latent_name: mu}。
        """
        obs_tensor = self.get_obs_tensor(obs)          # [B, num_obs]
        out = self.encoder(obs_tensor)                 # [B, 2 * latent_dim]

        if out.shape[-1] != 2 * self.latent_dim:
            raise RuntimeError(
                f"VAE encoder output dim = {out.shape[-1]}, expected 2 * latent_dim = {2 * self.latent_dim}"
            )

        mu, logvar = torch.split(out, self.latent_dim, dim=-1)

        # 推理时一般也可以 clamp 一下 logvar（虽然不用它），保持数值稳定
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)

        # 对 mu 做和训练时 z 一样的后处理（clip / normalize）
        mu = self._maybe_clip(mu)
        mu = self._maybe_normalize(mu)

        return TensorDict(
            {self.latent_name: mu},
            batch_size=obs.batch_size,
        )