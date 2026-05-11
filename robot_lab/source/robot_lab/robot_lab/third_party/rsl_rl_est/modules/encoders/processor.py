import torch
import torch.nn.functional as F
from typing import List
from tensordict import TensorDict
from .encoder import Encoder
from robot_lab.third_party.rsl_rl_est.networks import MLP


class Processor(Encoder):
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
        if network_cfg.get('share_network', False):
            self.encoder = network_cfg['shared_network']
        else:
            self.encoder = MLP(
                input_dim=self.num_obs,
                output_dim=latent_dim,
                hidden_dims=network_cfg.get("hidden_dims", [256, 256]),
                activation=network_cfg.get("activation", "elu"),
                last_activation=network_cfg.get("last_activation", None),
            )

        # ========== L2 normalize ==========
        self.use_l2_norm = network_cfg.get("normalize_output", False)

        # ========== Clip range（支持 list） ==========
        clip_range = network_cfg.get("clip_range", None)

        if clip_range is not None:
            assert isinstance(clip_range, (list, tuple)) and len(clip_range) == 2, \
                f"clip_range must be [min, max], got {clip_range}"
            self.clip_min, self.clip_max = clip_range
        else:
            self.clip_min = None
            self.clip_max = None

    # clipper
    def _maybe_clip(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_min is not None or self.clip_max is not None:
            return torch.clamp(
                x,
                min=self.clip_min,
                max=self.clip_max,
            )
        return x

    # normalize
    def _maybe_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_l2_norm:
            return F.normalize(x, dim=-1, p=2)
        return x

    # ========== 编码 ==========
    def encode(self, obs: TensorDict, **kwargs) -> TensorDict:
        obs_tensor = self.get_obs_tensor(obs)
        latent = self.encoder(obs_tensor)
        latent = self._maybe_clip(latent)
        latent = self._maybe_normalize(latent)

        return TensorDict({self.latent_name: latent}, batch_size=obs.batch_size)

    def encode_inference(self, obs: TensorDict, **kwargs) -> TensorDict:
        return self.encode(obs, **kwargs)