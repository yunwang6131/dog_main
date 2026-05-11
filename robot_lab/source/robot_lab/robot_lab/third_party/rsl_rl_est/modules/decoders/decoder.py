import torch
import torch.nn as nn
from typing import List, Union, Tuple
from tensordict import TensorDict

ObsGroupSpec = Union[str, Tuple[str, bool]]


class Decoder(nn.Module):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: List[ObsGroupSpec],
        target_groups: List[ObsGroupSpec],
        loss_weight: float = 1.0,
    ):
        super().__init__()

        # 原始 spec（可以是 str 或 (name, detach)）
        self._obs_groups_spec: List[ObsGroupSpec] = obs_groups
        self._target_groups_spec: List[ObsGroupSpec] = target_groups

        # 对外暴露：纯字符串，方便日志 / 导出
        self.obs_groups: List[str] = []
        self.target_groups: List[str] = []

        # detach 标记
        self._obs_detach: dict[str, bool] = {}
        self._target_detach: dict[str, bool] = {}

        # 解析 obs_groups
        self.num_obs = 0
        for spec in obs_groups:
            if isinstance(spec, str):
                name = spec
                detach = False
            else:
                name, detach = spec[0], spec[1]
            self.obs_groups.append(name)
            self._obs_detach[name] = bool(detach)
            # 可选：检查维度
            assert len(obs[name].shape) == 2, "Decoder only supports 1D obs (B, D)."
            self.num_obs += obs[name].shape[-1]

        # 解析 target_groups
        self.num_targets = 0
        for spec in target_groups:
            if isinstance(spec, str):
                name = spec
                detach = False
            else:
                name, detach = spec[0], spec[1]
            self.target_groups.append(name)
            self._target_detach[name] = bool(detach)
            assert len(obs[name].shape) == 2, "Decoder only supports 1D targets (B, D)."
            self.num_targets += obs[name].shape[-1]

        self.loss_weight = loss_weight

    # ---------- obs 部分 ----------

    def get_decoder_obs_list(self, obs: TensorDict) -> List[torch.Tensor]:
        obs_list: List[torch.Tensor] = []
        for name in self.obs_groups:
            x = obs[name]
            if self._obs_detach.get(name, False):
                x = x.detach()
            obs_list.append(x)
        return obs_list

    def get_decoder_obs_dict(self, obs: TensorDict) -> TensorDict:
        batch_size = obs.batch_size
        obs_dict = TensorDict({}, batch_size=batch_size)
        for name in self.obs_groups:
            x = obs[name]
            if self._obs_detach.get(name, False):
                x = x.detach()
            obs_dict[name] = x
        return obs_dict

    def get_decoder_obs_tensor(self, obs: TensorDict) -> torch.Tensor:
        obs_list = self.get_decoder_obs_list(obs)
        return torch.cat(obs_list, dim=-1)

    # ---------- target 部分 ----------

    def get_decoder_targets_list(self, obs: TensorDict) -> List[torch.Tensor]:
        target_list: List[torch.Tensor] = []
        for name in self.target_groups:
            x = obs[name]
            if self._target_detach.get(name, False):
                x = x.detach()
            target_list.append(x)
        return target_list

    def get_decoder_targets_dict(self, obs: TensorDict) -> TensorDict:
        batch_size = obs.batch_size
        target_dict = TensorDict({}, batch_size=batch_size)
        for name in self.target_groups:
            x = obs[name]
            if self._target_detach.get(name, False):
                x = x.detach()
            target_dict[name] = x
        return target_dict

    def get_decoder_targets_tensor(self, obs: TensorDict) -> torch.Tensor:
        target_list = self.get_decoder_targets_list(obs)
        return torch.cat(target_list, dim=-1)

    def loss_fn(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        """子类需要实现."""
        raise NotImplementedError