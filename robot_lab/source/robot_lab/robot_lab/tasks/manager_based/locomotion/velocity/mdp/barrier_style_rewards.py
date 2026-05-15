# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def barrier_style_stream_placeholder(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Neutrally small stream for ``barrier_style_*`` reward split plumbing (replace with real barriers)."""
    return torch.zeros(env.num_envs, device=env.device)
