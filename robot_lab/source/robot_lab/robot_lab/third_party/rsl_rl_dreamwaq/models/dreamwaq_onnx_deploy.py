# Copyright (c) 2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
"""DreamWaQ 单一 ONNX 部署封装（CENet + MLP），供脚本与 Isaac Lab 风格 exporter 共用。"""

from __future__ import annotations

import copy
import os
from typing import Any

import torch
import torch.nn as nn

from robot_lab.third_party.rsl_rl_dreamwaq.models.dreamwaq_actor import DreamWaQActor


class DreamWaQOnnxDeployWrapper(nn.Module):
    """与 ``DreamWaQActor._encode_context(..., sample_latent=False)`` + ``get_latent`` 推理链一致。"""

    is_recurrent: bool = False

    def __init__(self, actor: DreamWaQActor) -> None:
        super().__init__()
        self.history_dim = int(actor.history_dim)
        self.current_dim = int(actor.current_dim)

        self.encoder = copy.deepcopy(actor.encoder)
        self.velocity_head = copy.deepcopy(actor.velocity_head)
        self.latent_mu_head = copy.deepcopy(actor.latent_mu_head)

        self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)
        self.mlp = copy.deepcopy(actor.mlp)
        self.deterministic_output = actor.distribution.as_deterministic_output_module()

    def forward(self, history: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(history)
        v_est = self.velocity_head(feat)
        z = self.latent_mu_head(feat)
        x = torch.cat([current, v_est, z], dim=-1)
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)

    def get_dummy_inputs(self, batch: int = 1, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        dev = device or next(self.parameters()).device
        hist = torch.zeros(batch, self.history_dim, device=dev, dtype=torch.float32)
        cur = torch.zeros(batch, self.current_dim, device=dev, dtype=torch.float32)
        return hist, cur


def dreamwaq_actor_from_runner(runner: Any) -> DreamWaQActor | None:
    a = runner.alg.actor
    return a if isinstance(a, DreamWaQActor) else None


def export_dreamwaq_full_policy_jit(
    wrapper: DreamWaQOnnxDeployWrapper,
    jit_path: str,
    *,
    device: torch.device | None = None,
) -> None:
    """Export CENet + MLP as a single TorchScript module (history + current -> actions).

    Traced with batch size 1, suitable for typical on-robot inference. Inputs match
    :class:`DreamWaQOnnxDeployWrapper`: ``history`` [B, history_dim], ``current`` [B, current_dim].
    """
    wrapper.eval()
    dev = device or next(wrapper.parameters()).device
    hist, cur = wrapper.get_dummy_inputs(batch=1, device=dev)
    traced = torch.jit.trace(wrapper, (hist, cur))
    parent = os.path.dirname(os.path.abspath(jit_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    traced.save(jit_path)


def export_dreamwaq_merged_jit_from_runner(runner: Any, export_model_dir: str, filename: str = "policy_full.pt") -> bool:
    """If the loaded agent is DreamWaQ, save ``policy_full.pt`` (merged CENet + actor MLP)."""
    actor = dreamwaq_actor_from_runner(runner)
    if actor is None or not isinstance(actor, DreamWaQActor):
        return False
    # CPU trace is the most portable for embedded / libtorch on robot.
    wrapper = DreamWaQOnnxDeployWrapper(actor).cpu()
    path = os.path.join(export_model_dir, filename)
    export_dreamwaq_full_policy_jit(wrapper, path, device=torch.device("cpu"))
    print(f"[INFO] Exported merged DreamWaQ TorchScript (history+current -> actions): {path}")
    return True


def export_dreamwaq_full_policy_onnx(
    wrapper: DreamWaQOnnxDeployWrapper,
    onnx_path: str,
    *,
    opset_version: int = 18,
    dynamic_batch: bool = True,
    verbose: bool = False,
) -> None:
    wrapper.eval()
    device = next(wrapper.parameters()).device
    hist, cur = wrapper.get_dummy_inputs(batch=1, device=device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "history": {0: "batch"},
            "current": {0: "batch"},
            "actions": {0: "batch"},
        }

    parent = os.path.dirname(os.path.abspath(onnx_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (hist, cur),
        onnx_path,
        input_names=["history", "current"],
        output_names=["actions"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=verbose,
    )
