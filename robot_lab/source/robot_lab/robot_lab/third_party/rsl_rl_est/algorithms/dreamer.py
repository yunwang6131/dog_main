from __future__ import annotations
from robot_lab.third_party.rsl_rl_est.modules import ActorCriticEst

from robot_lab.third_party.rsl_rl_est.algorithms.ppo_wm import PPOWM
from robot_lab.third_party.rsl_rl_est.storage import RolloutStorageWM
from tensordict import TensorDict
import torch
from torch import optim


class Dreamer:
    def __init__(
            self,
            policy: ActorCriticEst,
            num_learning_epochs: int = 5,
            num_mini_batches: int = 4,
            clip_param: float = 0.2,
            gamma: float = 0.99,
            lam: float = 0.95,
            value_loss_coef: float = 1.0,
            entropy_coef: float = 0.01,
            learning_rate: float = 0.001,
            max_grad_norm: float = 1.0,
            use_clipped_value_loss: bool = True,
            schedule: str = "adaptive",
            desired_kl: float = 0.01,
            device: str = "cpu",
            normalize_advantage_per_mini_batch: bool = False,
            obs_go_next: list[str] | None = None,
            # RND parameters
            rnd_cfg: dict | None = None,
            # Symmetry parameters
            symmetry_cfg: dict | None = None,
            # Distributed training parameters
            multi_gpu_cfg: dict | None = None,
            # Dreamer specific
            num_dreamer_steps_per_env=None,
            obs_replace_dict: dict[str, str] | None = None,
    ) -> None:
        self.alg = PPOWM(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg, )
        self.obs_go_next = obs_go_next
        self.device = device
        self.storage: RolloutStorageWM = None  # type: ignore
        self.transition = RolloutStorageWM.Transition()
        self.num_mini_batches = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.num_dreamer_steps_per_env = num_dreamer_steps_per_env
        self.num_actions = None
        self.rnd = None  # TODO: just for now

        self.alg_obs_groups = []
        for obs in policy.obs_groups['policy']:
            if obs not in self.alg_obs_groups:
                self.alg_obs_groups.append(obs)
        for obs in policy.obs_groups['critic']:
            if obs not in self.alg_obs_groups:
                self.alg_obs_groups.append(obs)
        self.obs_replace_dict = obs_replace_dict

        self.est_optimizer = optim.Adam(self.alg.policy.estimator.parameters(), lr=1e-7)

    @property
    def policy(self):
        return self.alg.policy

    @property
    def optimizer(self):
        return self.alg.optimizer

    @property
    def learning_rate(self):
        return self.alg.learning_rate

    def init_storage(
            self,
            training_type: str,
            num_envs: int,
            num_transitions_per_env: int,
            obs: TensorDict,
            actions_shape: tuple[int] | list[int],
    ) -> None:
        # Create rollout storage
        self.storage = RolloutStorageWM(
            num_envs,
            num_transitions_per_env,
            obs,
            self.device,
        )
        self.num_actions = actions_shape
        self.num_transitions_per_env = num_transitions_per_env
        self.num_env = num_envs

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.transition.est_hidden_states = self.alg.policy.get_est_hidden_states()
        # Compute the actions and values
        for key in self.obs_go_next or []:
            obs[key + "_next"] = obs[key].clone()
        obs = self.alg.policy.normalize(obs)
        obs = self.alg.policy.encode(obs)
        actions = self.alg.policy.act(obs).detach()
        obs['current_actions'] = actions
        obs['current_actions_norm'] = self.alg.policy.normalizer['actions'](obs['current_actions'])
        obs = self.alg.policy.post_encode(obs)
        # Record observations before env.step()
        self.transition.observations = obs
        return actions

    def process_env_step(
            self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        for key in self.obs_go_next or []:
            self.transition.observations[key + "_next"] = obs[key].clone()
            obs[key + "_next"] = obs[key].clone()
        # Update the normalizers
        self.alg.policy.update_normalization(obs)
        # if self.rnd:
        #     self.rnd.update_normalization(obs)

        self.transition.dones = dones

        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.alg.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        # Compute value for the last step
        pass

    def update(self):
        loss_dict = {}
        generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        # print('------------------update estimator------------------')
        for obs_batch, hid_batch, masks_batch in generator:
            estimator_losses = self.alg.policy.loss_fn(obs_batch, dones=masks_batch, hidden_states=hid_batch)
            estimator_loss = torch.mean(estimator_losses['total_loss'])
            self.alg.optimizer.zero_grad()
            estimator_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.alg.policy.parameters(), self.alg.max_grad_norm)
            self.alg.optimizer.step()
            for key, value in estimator_losses.items():
                if key not in loss_dict:
                    loss_dict[key] = 0.0
                loss_dict[key] += value.mean().item() / (self.num_mini_batches * self.num_learning_epochs)

        dreamer_generator = self.storage.mini_batch_generator(self.num_mini_batches, 1)
        dreamer_loss_dict = {}
        # # print('------------------update ac------------------')
        for obs_batch, hid_batch in dreamer_generator:
            obs = obs_batch
            hid = hid_batch
            with torch.inference_mode():
                obs = self.alg.policy.encode_inference(obs, hidden_states=hid).detach()
                hid = self.alg.policy.get_est_hidden_states_inference()
                ppo_obs = {key: obs[key].detach() for key in self.alg_obs_groups}
                for _ in range(self.num_dreamer_steps_per_env):
                    # Sample actions
                    if self.alg.storage is None:
                        self.alg.init_storage(
                            'rl',
                            int(self.num_env * self.num_transitions_per_env / self.num_mini_batches),
                            self.num_dreamer_steps_per_env,
                            ppo_obs,
                            self.num_actions,
                        )
                    actions = self.alg.act(ppo_obs)
                    with torch.no_grad():
                        obs['current_actions'] = actions.detach()
                        obs['current_actions_norm'] = self.alg.policy.normalizer['actions'](
                            obs['current_actions']).detach()
                        obs = self.alg.policy.post_encode_inference(obs, hidden_states=hid).detach()
                        # Convert reset_est to a rounded boolean (四舍五入的布尔值)
                        dones = obs['reset_est'].float().squeeze(1).round().detach()
                        rewards = obs['rewards_est'].float().squeeze(1).detach() * 10.

                        for k, v in self.obs_replace_dict.items():
                            obs[k] = obs[v]
                        obs['actions'] = obs['current_actions'].detach()
                        obs['actions_norm'] = self.alg.policy.normalizer['actions'](obs['actions']).detach()
                        obs = self.alg.policy.encode_inference(obs, hidden_states=hid).detach()
                        hid = self.alg.policy.get_est_hidden_states_inference()
                        ppo_obs = {key: obs[key].detach() for key in self.alg_obs_groups}

                    # process the step
                    self.alg.process_env_step(ppo_obs, rewards, dones, {})

                self.alg.compute_returns(ppo_obs)

            loss_dict_per_dreamer_update = self.alg.update()
            for key, value in loss_dict_per_dreamer_update.items():
                if key not in dreamer_loss_dict:
                    dreamer_loss_dict[key] = 0.0
                dreamer_loss_dict[key] += value / (self.num_mini_batches)
        # # print('------------------end update------------------')

        loss_dict.update(dreamer_loss_dict)
        self.storage.clear()
        return loss_dict
