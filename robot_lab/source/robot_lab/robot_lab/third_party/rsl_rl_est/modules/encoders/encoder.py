import torch
import torch.nn as nn
from typing import List, Union, Tuple
from tensordict import TensorDict

ObsGroupSpec = Union[str, Tuple[str, bool]]


class Encoder(nn.Module):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: List[ObsGroupSpec],
        latent_name: str,
        latent_dim: int | None = None,
    ):
        super().__init__()

        self._obs_groups_spec: List[ObsGroupSpec] = obs_groups
        self.obs_groups: List[str] = []
        self._detach_flags: dict[str, bool] = {}

        self.latent_name = latent_name
        self.latent_dim = latent_dim

        # 统计总的 obs 维度（对 4D 也给一个“flatten 维度”，以兼容旧逻辑）
        self.num_obs = 0

        # 记录一下 obs 的“模式”：vector(2D) or spatial(4D)
        self._obs_rank: int | None = None

        for spec in obs_groups:
            if isinstance(spec, str):
                name, detach = spec, False
            else:
                name, detach = spec[0], spec[1]

            self.obs_groups.append(name)
            self._detach_flags[name] = bool(detach)

            assert name in obs.keys(), f"Obs key '{name}' not found in TensorDict."

            x = obs[name]
            print(name, x.size())
            assert x.dim() in (2, 4), (
                f"Encoder supports only 2D [B,D] or 4D [B,C,H,W] observations, "
                f"but got '{name}' with shape {tuple(x.shape)}"
            )

            # 要么全 2D，要么全 4D（强烈建议这样，避免语义混乱）
            if self._obs_rank is None:
                self._obs_rank = x.dim()
            else:
                assert x.dim() == self._obs_rank, (
                    f"All obs_groups must have the same rank. "
                    f"Got '{name}' dim={x.dim()} but previous dim={self._obs_rank}."
                )

            if x.dim() == 2:
                self.num_obs += x.shape[-1]
            else:
                # 4D: [B,C,H,W]
                self.num_obs += x.shape[1] * x.shape[2] * x.shape[3]

        self.encoder = nn.Identity()

    def encode(self, obs: TensorDict, **kwargs) -> TensorDict:
        raise NotImplementedError

    def encode_inference(self, obs: TensorDict, **kwargs) -> TensorDict:
        raise NotImplementedError

    def reset(self, done=None):
        pass

    def get_hidden_state(self):
        return {}

    def get_hidden_state_inference(self):
        return {}

    # ------------ helpers ------------

    def get_obs_list(self, obs: TensorDict) -> List[torch.Tensor]:
        obs_list: List[torch.Tensor] = []
        for name in self.obs_groups:
            x = obs[name]
            if self._detach_flags.get(name, False):
                x = x.detach()
            obs_list.append(x)
        return obs_list

    def get_obs_tensor(self, obs: TensorDict) -> torch.Tensor:
        """
        - 若 obs 全是 2D: [B,D] -> 在 dim=1 拼成 [B,sumD]
        - 若 obs 全是 4D: [B,C,H,W] -> 在 dim=1 拼成 [B,sumC,H,W]
        """
        obs_list = self.get_obs_list(obs)
        assert len(obs_list) > 0, "obs_groups is empty."

        rank = obs_list[0].dim()
        assert rank in (2, 4), f"Expected 2D or 4D obs, got dim={rank}."

        # 确保 rank 一致
        for x in obs_list:
            assert x.dim() == rank, "All obs tensors must have the same rank to concat."

        if rank == 2:
            # [B, D] concat on feature dim
            return torch.cat(obs_list, dim=1)

        # rank == 4: [B, C, H, W] concat on channel dim
        B, C0, H0, W0 = obs_list[0].shape
        for x in obs_list[1:]:
            assert x.shape[0] == B, "Batch size mismatch for 4D obs."
            assert x.shape[2] == H0 and x.shape[3] == W0, (
                f"Spatial size mismatch for 4D obs: expected (H,W)=({H0},{W0}), got ({x.shape[2]},{x.shape[3]})"
            )
        return torch.cat(obs_list, dim=1)

    def get_obs_dict(self, obs: TensorDict) -> TensorDict:
        batch_size = obs.batch_size
        out = TensorDict({}, batch_size=batch_size)
        for name in self.obs_groups:
            x = obs[name]
            if self._detach_flags.get(name, False):
                x = x.detach()
            out[name] = x
        return out