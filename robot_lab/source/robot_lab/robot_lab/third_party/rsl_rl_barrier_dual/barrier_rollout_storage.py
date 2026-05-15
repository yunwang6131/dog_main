# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Rollout buffer with split standard / barrier reward streams for dual-critic PPO."""

from __future__ import annotations

from collections.abc import Generator

import torch
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage


class BarrierDualRolloutStorage(RolloutStorage):
    """Like :class:`RolloutStorage` but keeps parallel reward/value/return/advantage for a second critic."""

    class Transition(RolloutStorage.Transition):
        def __init__(self) -> None:
            super().__init__()
            self.rewards_standard = None
            self.rewards_barrier = None
            self.values_barrier = None

        def clear(self) -> None:
            super().clear()
            self.rewards_standard = None
            self.rewards_barrier = None
            self.values_barrier = None

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)
        if training_type == "rl":
            self.rewards_standard = torch.zeros_like(self.rewards)
            self.rewards_barrier = torch.zeros_like(self.rewards)
            self.values_barrier = torch.zeros_like(self.values)
            self.returns_barrier = torch.zeros_like(self.returns)
            self.advantages_barrier = torch.zeros_like(self.advantages)

    def add_transition(self, transition: Transition) -> None:
        idx = self.step
        super().add_transition(transition)
        if self.training_type != "rl":
            return
        if transition.rewards_standard is None or transition.rewards_barrier is None:
            raise ValueError("BarrierDualRolloutStorage requires rewards_standard and rewards_barrier on Transition.")
        if transition.values_barrier is None:
            raise ValueError("BarrierDualRolloutStorage requires values_barrier on Transition.")
        self.rewards_standard[idx].copy_(transition.rewards_standard.view(-1, 1))
        self.rewards_barrier[idx].copy_(transition.rewards_barrier.view(-1, 1))
        self.values_barrier[idx].copy_(transition.values_barrier)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[RolloutStorage.Batch, None, None]:
        if self.training_type != "rl":
            raise ValueError("This generator is only for RL.")

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore

        values_barrier = self.values_barrier.flatten(0, 1)
        returns_barrier = self.returns_barrier.flatten(0, 1)
        advantages_barrier = self.advantages_barrier.flatten(0, 1)

        for _epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                batch = RolloutStorage.Batch(
                    observations=observations[batch_idx],  # type: ignore[arg-type]
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                )
                batch.values_barrier = values_barrier[batch_idx]  # type: ignore[attr-defined]
                batch.returns_barrier = returns_barrier[batch_idx]  # type: ignore[attr-defined]
                batch.advantages_barrier = advantages_barrier[batch_idx]  # type: ignore[attr-defined]
                yield batch

    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        raise NotImplementedError("BarrierDualRolloutStorage recurrent policies are not supported yet.")
