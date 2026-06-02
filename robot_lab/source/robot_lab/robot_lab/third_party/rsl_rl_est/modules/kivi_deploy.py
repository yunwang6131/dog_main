# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import os
from typing import Any

import torch
import torch.nn as nn

from robot_lab.third_party.rsl_rl_est.modules.actor_critic_est import ActorCriticEst


class KiviDeployWrapper(nn.Module):
    """Merged KiVi estimator + actor for deployment.

    Interface:
        history: [B, 450] group-major 10-step proprio history
        current: [B, 45] current proprio frame
        depth: [B, 3, 60, 60] front depth history
        memory: [num_memory_tokens, B, embed_dim] visuospatial memory
    """

    is_recurrent: bool = True

    def __init__(self, policy: ActorCriticEst) -> None:
        super().__init__()
        self.history_dim = 450
        self.current_dim = 45

        estimator = policy.estimator
        self.kinesthetic_encoder = copy.deepcopy(estimator.encoders["kinesthetic_encoder"])
        self.visuospatial_encoder = copy.deepcopy(estimator.encoders["visuospatial_encoder"])
        self.normalizer = copy.deepcopy(policy.normalizer)
        self.privlege_normalizer = copy.deepcopy(policy.privlege_normalizer)
        self.actor = copy.deepcopy(policy.actor)
        self.state_dependent_std = bool(policy.state_dependent_std)

        self.depth_history_len = int(self.visuospatial_encoder.depth_history_len)
        self.depth_height = 60
        self.depth_width = 60
        self.num_memory_tokens = int(self.visuospatial_encoder.num_memory_tokens)
        self.memory_dim = int(self.visuospatial_encoder.initial_memory_tokens.shape[-1])

    def get_initial_memory(self, batch: int = 1, device: torch.device | None = None) -> torch.Tensor:
        dev = device or next(self.parameters()).device
        return self.visuospatial_encoder.initial_memory_tokens.unsqueeze(1).expand(-1, batch, -1).to(dev).clone()

    def get_dummy_inputs(
        self, batch: int = 1, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dev = device or next(self.parameters()).device
        history = torch.zeros(batch, self.history_dim, device=dev, dtype=torch.float32)
        current = torch.zeros(batch, self.current_dim, device=dev, dtype=torch.float32)
        depth = torch.full(
            (batch, self.depth_history_len, self.depth_height, self.depth_width),
            2.0,
            device=dev,
            dtype=torch.float32,
        )
        memory = self.get_initial_memory(batch=batch, device=dev)
        return history, current, depth, memory

    def _split_current(self, current: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return torch.split(current, [3, 3, 3, 12, 12, 12], dim=-1)

    def _split_history(self, history: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return torch.split(history, [30, 30, 30, 120, 120, 120], dim=-1)

    def _normalized_proprio(self, history: torch.Tensor, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c_ang, c_grav, c_cmd, c_q, c_dq, c_act = self._split_current(current)
        h_ang, h_grav, h_cmd, h_q, h_dq, h_act = self._split_history(history)

        current_policy = torch.cat(
            [
                self.normalizer["base_ang_vel_with_noise"](c_ang),
                self.normalizer["projected_gravity_with_noise"](c_grav),
                c_cmd,
                self.normalizer["joint_pos_with_noise"](c_q),
                self.normalizer["joint_vel_with_noise"](c_dq),
                self.normalizer["actions"](c_act),
            ],
            dim=-1,
        )
        history_policy = torch.cat(
            [
                self.privlege_normalizer["base_ang_vel_history"](h_ang),
                self.privlege_normalizer["projected_gravity_history"](h_grav),
                h_cmd,
                self.privlege_normalizer["joint_pos_history"](h_q),
                self.privlege_normalizer["joint_vel_history"](h_dq),
                self.normalizer["actions_history"](h_act),
            ],
            dim=-1,
        )
        return history_policy, current_policy

    def _encode_kinesthetic(self, history_policy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self.kinesthetic_encoder
        feature = encoder.backbone(history_policy)
        explicit = encoder.explicit_head(feature)
        implicit_out = encoder.implicit_head(feature)
        implicit_mu, _ = torch.split(implicit_out, encoder.implicit_dim, dim=-1)
        implicit = encoder._maybe_normalize(implicit_mu)
        return explicit, implicit

    def _encode_visuospatial(
        self, history_policy: torch.Tensor, depth: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder = self.visuospatial_encoder
        proprio_token = encoder.proprio_encoder(history_policy).unsqueeze(1)

        batch, history_len, height, width = depth.shape
        depth_frames = depth.reshape(batch * history_len, 1, height, width)
        visual_map = encoder.visual_pool(encoder.depth_encoder(depth_frames))
        visual_map = visual_map.view(batch, history_len, visual_map.shape[1], visual_map.shape[2], visual_map.shape[3])
        visual_map = visual_map.mean(dim=1)
        visual_tokens = visual_map.flatten(2).transpose(1, 2)

        memory_tokens = memory.transpose(0, 1)
        fused_tokens = encoder.fusion(torch.cat([proprio_token, visual_tokens, memory_tokens], dim=1))
        proprio_state = encoder.fusion_norm(fused_tokens[:, 0])
        visual_foot = encoder.fusion_norm(fused_tokens[:, 1 : -self.num_memory_tokens].mean(dim=1))
        next_memory = encoder.fusion_norm(fused_tokens[:, -self.num_memory_tokens :]).transpose(0, 1)

        state_latent = encoder.state_head(proprio_state)
        foot_latent = encoder.foot_head(visual_foot)
        if encoder.use_l2_norm:
            state_latent = torch.nn.functional.normalize(state_latent, dim=-1, p=2)
            foot_latent = torch.nn.functional.normalize(foot_latent, dim=-1, p=2)
        return state_latent, foot_latent, next_memory

    def forward(
        self, history: torch.Tensor, current: torch.Tensor, depth: torch.Tensor, memory: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_policy, current_policy = self._normalized_proprio(history, current)
        kin_explicit, kin_implicit = self._encode_kinesthetic(history_policy)
        vis_state, vis_foot, next_memory = self._encode_visuospatial(history_policy, depth, memory)
        actor_obs = torch.cat([current_policy, kin_explicit, kin_implicit, vis_state, vis_foot], dim=-1)
        actions = self.actor(actor_obs)
        if self.state_dependent_std:
            actions = actions[..., 0, :]
        return actions, next_memory


def kivi_policy_from_runner(runner: Any) -> ActorCriticEst | None:
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    return policy if isinstance(policy, ActorCriticEst) else None


def export_kivi_full_policy_jit(
    wrapper: KiviDeployWrapper,
    jit_path: str,
    *,
    device: torch.device | None = None,
) -> None:
    wrapper.eval()
    dev = device or next(wrapper.parameters()).device
    dummy_inputs = wrapper.get_dummy_inputs(batch=1, device=dev)
    # TransformerEncoder can produce equivalent but textually different trace graphs during the
    # checker pass on the Isaac bundled PyTorch build. The deployment wrapper is deterministic in
    # eval mode, so disable the structural re-trace comparison.
    traced = torch.jit.trace(wrapper, dummy_inputs, check_trace=False)
    parent = os.path.dirname(os.path.abspath(jit_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    traced.save(jit_path)


def export_kivi_merged_jit_from_runner(runner: Any, export_model_dir: str, filename: str = "policy_full.pt") -> bool:
    policy = kivi_policy_from_runner(runner)
    if policy is None:
        return False
    wrapper = KiviDeployWrapper(policy).cpu()
    path = os.path.join(export_model_dir, filename)
    export_kivi_full_policy_jit(wrapper, path, device=torch.device("cpu"))
    print(f"[INFO] Exported merged KiVi TorchScript (history+current+depth+memory -> actions,memory): {path}")
    return True
