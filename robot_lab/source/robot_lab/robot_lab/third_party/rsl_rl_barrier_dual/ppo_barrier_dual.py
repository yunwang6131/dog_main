# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Dual-critic PPO for barrier-style rewards (Kim et al. style), built on top of ``rsl_rl``."""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer

from robot_lab.third_party.rsl_rl_barrier_dual.barrier_rollout_storage import BarrierDualRolloutStorage

# Isaac Lab's deprecation helper only migrates actor/critic, not critic_barrier.
_LEGACY_MLP_KWARGS = ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std")


def _sanitize_mlp_cfg(model_cfg: dict) -> dict:
    """Drop rsl-rl < 5 kwargs that break ``MLPModel`` on rsl-rl >= 5."""
    sanitized = dict(model_cfg)
    for key in _LEGACY_MLP_KWARGS:
        sanitized.pop(key, None)
    return sanitized


class BarrierDualPPO(PPO):
    """PPO with two critics (standard vs barrier reward streams) and a combined clipped surrogate."""

    critic_barrier: MLPModel

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        critic_barrier: MLPModel,
        storage: BarrierDualRolloutStorage,
        *,
        surrogate_barrier_weight: float = 0.5,
        estimator_loss_coef: float = 1.0,
        dreamwaq_next_obs_groups: list[str] | None = None,
        **kwargs,
    ) -> None:
        if kwargs.get("rnd_cfg"):
            raise NotImplementedError("BarrierDualPPO does not support RND in this codebase yet.")
        if kwargs.get("symmetry_cfg"):
            raise NotImplementedError("BarrierDualPPO does not support symmetry augmentation here yet.")

        opt_name = kwargs.get("optimizer", "adam")

        super().__init__(actor, critic, storage, **kwargs)

        self.critic_barrier = critic_barrier.to(self.device)

        self.optimizer = resolve_optimizer(opt_name)(
            chain(self.actor.parameters(), self.critic.parameters(), self.critic_barrier.parameters()),
            lr=self.learning_rate,
        )
        self.transition = BarrierDualRolloutStorage.Transition()
        self.surrogate_barrier_weight = float(surrogate_barrier_weight)
        self.surrogate_standard_weight = 1.0 - self.surrogate_barrier_weight
        self.estimator_loss_coef = float(estimator_loss_coef)
        self.dreamwaq_next_obs_groups = list(dreamwaq_next_obs_groups or [])

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.values_barrier = self.critic_barrier(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        self.critic_barrier.update_normalization(obs)

        rs = extras.get("reward_standard")
        rb = extras.get("reward_barrier")
        if rs is None or rb is None:
            rs_t = rewards.clone()
            rb_t = torch.zeros_like(rewards)
        else:
            rs_t = rs.to(self.device).reshape_as(rewards)
            rb_t = rb.to(self.device).reshape_as(rewards)

        self.transition.rewards_standard = rs_t.clone()
        self.transition.rewards_barrier = rb_t.clone()
        self.transition.rewards = rs_t + rb_t
        self.transition.dones = dones
        for group in self.dreamwaq_next_obs_groups:
            self.transition.observations[group + "_next"] = obs[group].clone()

        if "time_outs" in extras:
            to = extras["time_outs"].unsqueeze(-1).to(self.device).float()
            bump_std = torch.squeeze(self.transition.values * to, dim=-1)
            bump_bar = torch.squeeze(self.transition.values_barrier * to, dim=-1)
            if rewards.dim() == 2:
                bump_std = bump_std.unsqueeze(-1)
                bump_bar = bump_bar.unsqueeze(-1)
            self.transition.rewards_standard += self.gamma * bump_std
            self.transition.rewards_barrier += self.gamma * bump_bar
            self.transition.rewards = self.transition.rewards_standard + self.transition.rewards_barrier

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)
        self.critic_barrier.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        st: BarrierDualRolloutStorage = self.storage  # type: ignore[assignment]

        last_std = self.critic(obs).detach()
        last_bar = self.critic_barrier(obs).detach()

        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_std = last_std if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards_standard[step] + next_is_not_terminal * self.gamma * next_std - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

        advantage_b = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_bar = last_bar if step == st.num_transitions_per_env - 1 else st.values_barrier[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta_b = st.rewards_barrier[step] + next_is_not_terminal * self.gamma * next_bar - st.values_barrier[step]
            advantage_b = delta_b + next_is_not_terminal * self.gamma * self.lam * advantage_b
            st.returns_barrier[step] = advantage_b + st.values_barrier[step]
        st.advantages_barrier = st.returns_barrier - st.values_barrier
        if not self.normalize_advantage_per_mini_batch:
            st.advantages_barrier = (st.advantages_barrier - st.advantages_barrier.mean()) / (
                st.advantages_barrier.std() + 1e-8
            )

    def update(self) -> dict[str, float]:
        if self.actor.is_recurrent or self.critic.is_recurrent or self.critic_barrier.is_recurrent:
            raise NotImplementedError("BarrierDualPPO supports feedforward policies only.")

        mean_value_loss = 0.0
        mean_value_loss_barrier = 0.0
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
                    batch.advantages_barrier = (
                        batch.advantages_barrier - batch.advantages_barrier.mean()
                    ) / (batch.advantages_barrier.std() + 1e-8)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore[arg-type]
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            values_b = self.critic_barrier(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
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

            adv_s = torch.squeeze(batch.advantages)  # type: ignore[arg-type]
            adv_b = torch.squeeze(batch.advantages_barrier)
            surrogate_s = torch.max(
                -adv_s * ratio,
                -adv_s * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param),
            )
            surrogate_b = torch.max(
                -adv_b * ratio,
                -adv_b * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param),
            )
            surrogate_loss = (
                self.surrogate_standard_weight * surrogate_s.mean() + self.surrogate_barrier_weight * surrogate_b.mean()
            )

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()

                value_b_clipped = batch.values_barrier + (values_b - batch.values_barrier).clamp(  # type: ignore[attr-defined]
                    -self.clip_param, self.clip_param
                )
                vbl = (values_b - batch.returns_barrier).pow(2)  # type: ignore[attr-defined]
                vbcl = (value_b_clipped - batch.returns_barrier).pow(2)  # type: ignore[attr-defined]
                value_loss_b = torch.max(vbl, vbcl).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()
                value_loss_b = (batch.returns_barrier - values_b).pow(2).mean()  # type: ignore[attr-defined]

            loss = surrogate_loss + self.value_loss_coef * (value_loss + value_loss_b) - self.entropy_coef * entropy.mean()

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
            nn.utils.clip_grad_norm_(self.critic_barrier.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_value_loss_barrier += value_loss_b.item()
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
        mean_value_loss_barrier /= n
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
            "value_barrier": mean_value_loss_barrier,
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

    def save(self) -> dict:
        d = super().save()
        d["critic_barrier_state_dict"] = self.critic_barrier.state_dict()
        return d

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "critic_barrier": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }
        if load_cfg.get("critic_barrier", True) and "critic_barrier_state_dict" in loaded_dict:
            self.critic_barrier.load_state_dict(loaded_dict["critic_barrier_state_dict"], strict=strict)
        lc = dict(load_cfg)
        lc.pop("critic_barrier", None)
        return super().load(loaded_dict, lc, strict)

    def broadcast_parameters(self) -> None:
        model_params = [self.actor.state_dict(), self.critic.state_dict(), self.critic_barrier.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        self.critic_barrier.load_state_dict(model_params[2])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[3])

    def reduce_parameters(self) -> None:
        all_params = chain(
            self.actor.parameters(), self.critic.parameters(), self.critic_barrier.parameters()
        )
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        grads = [param.grad.view(-1) for param in list(all_params) if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        grouped = chain(self.actor.parameters(), self.critic.parameters(), self.critic_barrier.parameters())
        for param in grouped:
            if param.grad is not None:
                n = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + n].view_as(param.grad.data))
                offset += n

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "BarrierDualPPO":
        alg_class: type[BarrierDualPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[assignment]
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore[assignment]
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore[assignment]
        critic_barrier_class: type[MLPModel] = resolve_callable(cfg["critic_barrier"].pop("class_name"))  # type: ignore[assignment]

        next_obs_groups = list(cfg["algorithm"].get("dreamwaq_next_obs_groups", []))
        for group in next_obs_groups:
            obs[group + "_next"] = obs[group].clone()

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor_cfg = _sanitize_mlp_cfg(cfg["actor"])
        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        print(f"BarrierDual Actor Model: {actor}")
        share = cfg["algorithm"].pop("share_cnn_encoders", None)
        critic_cfg = _sanitize_mlp_cfg(cfg["critic"])
        if share:
            critic_cfg["cnns"] = actor.cnns  # type: ignore[index]
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        critic_barrier_cfg = _sanitize_mlp_cfg(cfg["critic_barrier"])
        if share:
            critic_barrier_cfg["cnns"] = actor.cnns  # type: ignore[index]
        critic_barrier = critic_barrier_class(obs, cfg["obs_groups"], "critic", 1, **critic_barrier_cfg).to(device)
        print(f"BarrierDual Critic Model: {critic}")
        print(f"BarrierDual Critic Barrier Model: {critic_barrier}")

        storage = BarrierDualRolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )
        alg = alg_class(
            actor, critic, critic_barrier, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )
        return alg

    def train_mode(self) -> None:
        super().train_mode()
        self.critic_barrier.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.critic_barrier.eval()
