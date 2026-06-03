# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Standard PPO with a DreamWaQ-style auxiliary CENet loss."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


_LEGACY_MLP_KWARGS = ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std")


def _sanitize_mlp_cfg(model_cfg: dict) -> dict:
    sanitized = dict(model_cfg)
    for key in _LEGACY_MLP_KWARGS:
        sanitized.pop(key, None)
    return sanitized


class DreamWaQPPO(PPO):
    """PPO plus DreamWaQ auxiliary loss, without barrier rewards or a second critic."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        *,
        estimator_loss_coef: float = 1.0,
        dreamwaq_next_obs_groups: list[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.estimator_loss_coef = float(estimator_loss_coef)
        self.dreamwaq_next_obs_groups = list(dreamwaq_next_obs_groups or [])

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        for group in self.dreamwaq_next_obs_groups:
            self.transition.observations[group + "_next"] = obs[group].clone()
        super().process_env_step(obs, rewards, dones, extras)

    def update(self) -> dict[str, float]:
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise NotImplementedError("DreamWaQPPO supports feedforward policies only.")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_cenet_loss = 0.0
        mean_cenet_velocity_loss = 0.0
        mean_cenet_reconstruction_loss = 0.0
        mean_cenet_kl_loss = 0.0
        mean_cenet_terrain_loss = 0.0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (  # type: ignore[union-attr]
                        batch.advantages.std() + 1e-8  # type: ignore[union-attr]
                    )

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore[arg-type]
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore[arg-type]
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore[arg-type]
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore[arg-type]
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            cenet_losses = None
            compute_cenet_loss = getattr(self.actor, "compute_cenet_loss", None)
            if callable(compute_cenet_loss):
                cenet_losses = compute_cenet_loss(batch.observations)
                loss = loss + self.estimator_loss_coef * cenet_losses.total

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if cenet_losses is not None:
                mean_cenet_loss += cenet_losses.total.item()
                mean_cenet_velocity_loss += cenet_losses.velocity.item()
                mean_cenet_reconstruction_loss += cenet_losses.reconstruction.item()
                mean_cenet_kl_loss += cenet_losses.kl.item()
                mean_cenet_terrain_loss += cenet_losses.terrain.item()

        n = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= n
        mean_surrogate_loss /= n
        mean_entropy /= n
        mean_cenet_loss /= n
        mean_cenet_velocity_loss /= n
        mean_cenet_reconstruction_loss /= n
        mean_cenet_kl_loss /= n
        mean_cenet_terrain_loss /= n

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if hasattr(self.actor, "compute_cenet_loss"):
            loss_dict["cenet"] = mean_cenet_loss
            loss_dict["cenet_velocity"] = mean_cenet_velocity_loss
            loss_dict["cenet_reconstruction"] = mean_cenet_reconstruction_loss
            loss_dict["cenet_kl"] = mean_cenet_kl_loss
            loss_dict["cenet_terrain"] = mean_cenet_terrain_loss
        return loss_dict

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "DreamWaQPPO":
        alg_class: type[DreamWaQPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[assignment]
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore[assignment]
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore[assignment]

        next_obs_groups = list(cfg["algorithm"].get("dreamwaq_next_obs_groups", []))
        for group in next_obs_groups:
            obs[group + "_next"] = obs[group].clone()

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor_cfg = _sanitize_mlp_cfg(cfg["actor"])
        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        print(f"DreamWaQ Actor Model: {actor}")

        share = cfg["algorithm"].pop("share_cnn_encoders", None)
        critic_cfg = _sanitize_mlp_cfg(cfg["critic"])
        if share:
            critic_cfg["cnns"] = actor.cnns  # type: ignore[index]
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        print(f"DreamWaQ Critic Model: {critic}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        alg = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
        return alg

