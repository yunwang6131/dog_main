# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict


class RolloutStorageWM:
    class Transition:
        def __init__(self):
            self.observations = None
            self.dones = None
            self.est_hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
            self,
            num_envs,
            num_transitions_per_env,
            obs,
            device="cpu",
    ):
        # store inputs
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # Core
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For RNN networks
        self.saved_est_hidden_states = {}

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For RNN networks
        self._save_est_hidden_states(transition.est_hidden_states)

        # increment the counter
        self.step += 1

    def _save_est_hidden_states(self, est_hidden_states):
        if est_hidden_states is None:
            return

        for key, value in est_hidden_states.items():
            hid = value if isinstance(value, tuple) else (value,)
            # initialize if needed
            if key not in self.saved_est_hidden_states:
                self.saved_est_hidden_states[key] = [
                    torch.zeros(self.observations.shape[0], *hid[i].shape, device=self.device) for i in
                    range(len(hid))
                ]
            # copy the states
            for i in range(len(hid)):
                self.saved_est_hidden_states[key][i][self.step].copy_(hid[i])

    def clear(self):
        self.step = 0

    # for Estimator update
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = self.dones[:, start:stop].flatten(0, 1)
                obs_batch = self.observations[:, start:stop].flatten(0, 1)

                last_was_done = last_was_done.permute(1, 0)
                hid_batch = {}

                for key, saved_hidden_states in self.saved_est_hidden_states.items():
                    hid_a_batch = [
                        saved_hidden_states[i].permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                        .transpose(1, 0)
                        .contiguous()
                        for i in range(len(saved_hidden_states))
                    ]
                    # remove the tuple for GRU
                    hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else tuple(hid_a_batch)
                    hid_batch[key] = hid_a_batch

                yield obs_batch, hid_batch, masks_batch

                first_traj = last_traj

    # for policy and value function update
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        # --------------------------------------------------
        # 观测: [T, B, ...] -> [T*B, ...]
        # --------------------------------------------------
        observations = self.observations.flatten(0, 1)  # [T*B, obs_dim]

        # --------------------------------------------------
        # 处理 RNN hidden states
        #
        # 当前结构:
        #   self.saved_est_hidden_states['blind_latent'] 是一个 list，
        #   其中第 0 个元素是 4D tensor: [T, L, B, H]
        #
        # 目标:
        #   flat_hidden[k] 是 list，长度 L，
        #   每个元素都是 [T*B, H]，方便后续用 batch_idx 索引。
        # --------------------------------------------------
        flat_hidden = {}
        for k, v in self.saved_est_hidden_states.items():
            # v 可能是 list([T,L,B,H]) 或直接就是 tensor([T,L,B,H])
            if isinstance(v, torch.Tensor):
                h_all = v
            else:
                # 默认取第 0 个元素
                h_all = v[0]

            assert h_all.dim() == 4, \
                f"saved_est_hidden_states['{k}'] element must be [T,L,B,H], got {h_all.shape}"

            T, L, B, H = h_all.shape

            layer_list_flat = []
            for layer_idx in range(L):
                # 取出第 layer_idx 层: [T,B,H]
                h_layer = h_all[:, layer_idx, :, :]  # [T,B,H]
                # 展平成 [T*B, H]
                h_flat = h_layer.reshape(T * B, H)  # [T*B,H]
                layer_list_flat.append(h_flat)

            # 得到: list 长度 L，每个 [T*B, H]
            flat_hidden[k] = layer_list_flat

        # --------------------------------------------------
        # 采样 mini-batch
        # --------------------------------------------------
        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]  # [mini_batch]

                # obs 部分: [mini_batch, obs_dim]
                obs_batch = observations[batch_idx]

                # hidden 部分: 对每个 key、每一层，用相同 batch_idx 抽子集，
                #   然后 stack 成 [L, mini_batch, H]，符合 GRU 期望的 (num_layers, batch, hidden_size)
                hid_batch = {}
                for k, layer_list_flat in flat_hidden.items():
                    per_layer_selected = []
                    for h_flat in layer_list_flat:
                        # h_flat: [T*B, H] -> [mini_batch, H]
                        h_sel = h_flat[batch_idx]
                        per_layer_selected.append(h_sel)
                    # stack: [L, mini_batch, H]
                    h_stack = torch.stack(per_layer_selected, dim=0)
                    hid_batch[k] = h_stack

                yield obs_batch, hid_batch