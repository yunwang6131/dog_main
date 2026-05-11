# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import MLP, EmpiricalNormalization
from robot_lab.third_party.rsl_rl_est.modules.estimator import Estimator


class ActorCriticEst(nn.Module):
    is_recurrent: bool = False

    def __init__(
            self,
            obs: TensorDict,
            obs_groups: dict[str, list[str]],
            num_actions: int,
            actor_obs_normalization: bool = False,
            critic_obs_normalization: bool = False,
            actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
            critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
            activation: str = "elu",
            init_noise_std: float = 1.0,
            noise_std_type: str = "scalar",
            state_dependent_std: bool = False,
            estimator_cfg=None,
            **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups: dict[str, list[str]] = {}
        self._detach_flags: dict[str, dict[str, bool]] = {key: {} for key in obs_groups}

        for key, group_list in obs_groups.items():
            clean_list: list[str] = []
            for item in group_list:
                if isinstance(item, (list, tuple)):
                    name, detach_flag = item[0], item[1]
                else:
                    name, detach_flag = item, False
                clean_list.append(name)
                self._detach_flags[key][name] = bool(detach_flag)
            self.obs_groups[key] = clean_list

        self.normalizer = nn.ModuleDict()
        for obs_group in self.obs_groups["normalize"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_normalize_obs = obs[obs_group].shape[-1]
            obs_device = obs[obs_group].device
            self.normalizer.update({obs_group: EmpiricalNormalization(num_normalize_obs).to(obs_device)})
            obs.update({obs_group + '_norm': self.normalizer[obs_group](obs[obs_group])})

        self.privlege_normalizer = nn.ModuleDict()
        for obs_group in self.obs_groups["privilege_normalize"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_privilege_normalize_obs = obs[obs_group].shape[-1]
            obs_device = obs[obs_group].device
            self.privlege_normalizer.update(
                {obs_group: EmpiricalNormalization(num_privilege_normalize_obs).to(obs_device)})
            obs.update({obs_group + '_norm': self.privlege_normalizer[obs_group](obs[obs_group])})

        if estimator_cfg is not None:
            self.estimator = Estimator(
                obs=obs,
                **estimator_cfg,
            )
            obs = self.estimator.encode(obs)
        else:
            class DummyEstimator:
                def encode(self, x, **kwargs):
                    return x

                def encode_inference(self, x, **kwargs):
                    return x

                def reset(self, *args, **kwargs):
                    pass

                def loss_fn(self, *args, **kwargs):
                    pass

                def get_memory_buffers(self) -> dict[str, torch.Tensor]:
                    return {}

                def get_hidden_states(self) -> dict[str, torch.Tensor]:
                    return {}

                def get_hidden_states_inference(self) -> dict[str, torch.Tensor]:
                    return {}

            self.estimator = DummyEstimator()

        # Get the observation dimensions
        num_actor_obs = 0
        for obs_group in self.obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in self.obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std

        self._print_detach_report()
        # Actor
        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def _print_detach_report(self):
        print("\n================ Gradient Flow (ActorCritic) ================")
        for branch in ["policy", "critic"]:
            groups = self.obs_groups.get(branch, [])
            flags = self._detach_flags.get(branch, {})

            detached = [g for g in groups if flags.get(g, False)]
            attached = [g for g in groups if not flags.get(g, False)]

            print(f"[{branch.upper()}]")
            print(f"  inputs             : {groups}")
            print(f"  detached (no grad) : {detached if detached else 'None'}")
            print(f"  with grad          : {attached if attached else 'None'}")
        print("==============================================================\n")

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.estimator.reset(dones)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, obs: TensorDict) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        # obs = self.estimator.encode(obs, dones=masks, hidden_states=hidden_state)
        obs = self.get_actor_obs(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        for obs_group in self.obs_groups["normalize"]:
            obs.update({obs_group + '_norm': self.normalizer[obs_group](obs[obs_group])})
        obs = self.estimator.encode_inference(obs, dones=masks, hidden_states=hidden_state)
        obs = self.get_actor_obs(obs)
        if self.state_dependent_std:
            return self.actor(obs)[..., 0, :]
        else:
            return self.actor(obs)

    def act_inference_for_training(self, obs):
        obs = self.get_actor_obs(obs)
        if self.state_dependent_std:
            return self.actor(obs)[..., 0, :]
        else:
            return self.actor(obs)

    def encode(self, obs: TensorDict, **kwargs) -> TensorDict:
        return self.estimator.encode(obs, **kwargs)

    def encode_inference(self, obs: TensorDict, **kwargs) -> TensorDict:
        return self.estimator.encode_inference(obs, **kwargs)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        # obs = self.estimator.encode(obs)
        obs = self.get_critic_obs(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = []
        for name in self.obs_groups["policy"]:
            x = obs[name]
            if self._detach_flags["policy"].get(name, False):
                x = x.detach()
            obs_list.append(x)
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = []
        for name in self.obs_groups["critic"]:
            x = obs[name]
            if self._detach_flags["critic"].get(name, False):
                x = x.detach()
            obs_list.append(x)
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        for obs_group in self.obs_groups["normalize"]:
            self.normalizer[obs_group].update(obs[obs_group])
        for obs_group in self.obs_groups["privilege_normalize"]:
            self.privlege_normalizer[obs_group].update(obs[obs_group])

    def normalize(self, obs: TensorDict) -> TensorDict:
        for obs_group in self.obs_groups["normalize"]:
            obs.update({obs_group + '_norm': self.normalizer[obs_group](obs[obs_group])})
        for obs_group in self.obs_groups["privilege_normalize"]:
            obs.update({obs_group + '_norm': self.privlege_normalizer[obs_group](obs[obs_group])})
        return obs

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True

    def loss_fn(self, obs: TensorDict) -> TensorDict:
        return self.estimator.loss_fn(obs)

    def get_memory_buffers(self) -> dict[str, torch.Tensor]:
        return self.estimator.get_memory_buffers()

    def get_est_hidden_states(self) -> dict[str, torch.Tensor]:
        return self.estimator.get_hidden_states()

    def get_est_hidden_states_inference(self) -> dict[str, torch.Tensor]:
        return self.estimator.get_hidden_states_inference()
