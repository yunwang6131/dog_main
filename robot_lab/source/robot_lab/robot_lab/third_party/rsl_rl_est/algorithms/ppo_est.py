import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.algorithms.ppo import PPO
from robot_lab.third_party.rsl_rl_est.modules import ActorCriticEst
from robot_lab.third_party.rsl_rl_est.storage import RolloutStorageEst


def _masked_mean(total_est_loss, obs_td):
    # total_est_loss: [B] or [B,1]
    if total_est_loss.ndim > 1:
        total_est_loss = total_est_loss.view(total_est_loss.shape[0], -1)[:, 0]

    if "reset" in obs_td:
        reset = obs_td["reset"].detach()
        if reset.ndim > 1:
            reset = reset.view(reset.shape[0], -1)[:, 0]
        valid = (reset == 0) | (reset < 0.5)
        if valid.any():
            return total_est_loss[valid].mean()
    return total_est_loss.mean()


class PPOEst(PPO):
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCriticEst
    """The actor critic module."""

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
            obs_go_next: list | None = None,
            # RND parameters
            rnd_cfg: dict | None = None,
            # Symmetry parameters
            symmetry_cfg: dict | None = None,
            # Distributed training parameters
            multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            policy,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            device,
            normalize_advantage_per_mini_batch,
            rnd_cfg,
            symmetry_cfg,
            multi_gpu_cfg,
        )
        self.obs_go_next = obs_go_next
        # Create rollout storage
        self.storage: RolloutStorageEst | None = None
        self.transition = RolloutStorageEst.Transition()
        self.symmetry_on = True if symmetry_cfg is not None else False

    def init_storage(
            self,
            training_type: str,
            num_envs: int,
            num_transitions_per_env: int,
            obs: TensorDict,
            actions_shape: tuple[int] | list[int],
    ) -> None:
        # Create rollout storage
        self.storage = RolloutStorageEst(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )
        self.num_envs = num_envs

    def compute_returns(self, obs: TensorDict) -> None:
        # Compute value for the last step
        obs = self.policy.normalize(obs)
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        if self.policy.estimator.is_recurrent:
            self.transition.est_hidden_states = self.policy.get_est_hidden_states()
            if self.symmetry:
                self.transition.est_hidden_states_mirror = self.policy.get_est_hidden_states_inference()
        # Compute the actions and values
        for key in self.obs_go_next or []:
            obs[key + "_next"] = obs[key].detach()
        if self.symmetry and self.symmetry["use_data_augmentation"]:
            data_augmentation_func = self.symmetry["data_augmentation_func"]
            obs_all, _ = data_augmentation_func(env=None, obs=obs.detach())
            obs_m = obs_all[self.num_envs:].detach()
            obs_m = self.policy.normalize(obs_m)
            self.policy.encode(obs=obs_m, hidden_states=self.policy.get_est_hidden_states_inference())

        obs_norm = self.policy.normalize(obs)
        obs_encode = self.policy.encode(obs_norm)
        self.transition.actions = self.policy.act(obs_encode).detach()
        self.transition.values = self.policy.evaluate(obs_encode).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_est_loss_dict = {}
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        elif self.policy.estimator.is_recurrent:
            generator = self.storage.recurrent_est_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for (
                obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hidden_states_batch,
                masks_batch,
        ) in generator:
            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_all, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )

                # obs_all: [2B, ...]
                num_aug = int(obs_all.batch_size[0] / original_batch_size)  # 这里一般是 2
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

                # 1) raw obs（安全副本）
                obs_all_raw = obs_all.detach().clone()
                obs_orin_raw = obs_all_raw[:original_batch_size]
                obs_mirror_raw = obs_all_raw[original_batch_size:]

                # 2) normalize（各自做，避免任何 view/inplace 的奇怪共享）
                obs_orin_norm = self.policy.normalize(obs_orin_raw)
                obs_mirror_norm = self.policy.normalize(obs_mirror_raw)

                # 3) encode（分别用各自的 hidden；masks_batch 不需要翻倍，因为这里每次 encode 的 batch 还是 B）
                obs_orin_enc = self.policy.encode(
                    obs_orin_norm, dones=masks_batch, hidden_states=hidden_states_batch[0]
                )
                obs_mirror_enc = self.policy.encode(
                    obs_mirror_norm, dones=masks_batch, hidden_states=hidden_states_batch[1]
                )

                obs_batch = TensorDict.cat([obs_orin_enc, obs_mirror_enc], dim=0)
                estimator_losses = self.policy.loss_fn(obs_batch)
                estimator_loss = _masked_mean(estimator_losses["total_loss"], obs_batch)

            else:
                obs_batch = self.policy.normalize(obs_batch)
                obs_batch = self.policy.encode(obs_batch, dones=masks_batch, hidden_states=hidden_states_batch[0])

                estimator_losses = self.policy.loss_fn(obs_batch)
                estimator_loss = _masked_mean(estimator_losses["total_loss"], obs_batch)
            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # 总 loss 保持不变
            loss = (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_batch.mean()
                    + estimator_loss
            )

            # Symmetry loss
            if self.symmetry:
                # TODO: what about est
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference_for_training(obs_batch.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # Extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    obs_batch = self.policy.normalize(obs_batch)
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # estimator loss
            for k, v in estimator_losses.items():
                if k not in mean_est_loss_dict:
                    mean_est_loss_dict[k] = 0
                mean_est_loss_dict[k] += torch.mean(v).item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        for k in mean_est_loss_dict:
            mean_est_loss_dict[k] /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        loss_dict.update(mean_est_loss_dict)
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    def process_env_step(
            self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:

        for key in self.obs_go_next or []:
            self.transition.observations[key + "_next"] = obs[key].detach()
            obs[key + "_next"] = obs[key].detach()

        # Update the normalizers
        self.policy.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            obs = self.policy.normalize(obs)
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)
