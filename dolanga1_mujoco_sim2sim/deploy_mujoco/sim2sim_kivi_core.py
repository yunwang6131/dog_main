from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from deploy_mujoco.sim2sim_core import Sim2SimCfg, Sim2SimRunner


@dataclass
class KiviSim2SimCfg(Sim2SimCfg):
    depth_history_len: int = 3
    depth_height: int = 60
    depth_width: int = 60
    depth_fill_value: float = 2.0
    visual_memory_tokens: int = 4
    visual_memory_dim: int = 32


class KiviSim2SimRunner(Sim2SimRunner):
    """Dolanga1 KiVi sim2sim runner.

    The exported KiVi policy_full.pt is expected to use:
        history, current, front_camera_depth, visual_memory -> actions, next_visual_memory
    """

    cfg: KiviSim2SimCfg

    def __init__(self, cfg: KiviSim2SimCfg):
        super().__init__(cfg)
        self.input_shape = ["history", "current", "front_camera_depth", "visual_memory"]
        self.depth_history = np.full(
            (cfg.depth_history_len, cfg.depth_height, cfg.depth_width),
            cfg.depth_fill_value,
            dtype=np.float32,
        )
        self.visual_memory = torch.zeros(
            cfg.visual_memory_tokens,
            1,
            cfg.visual_memory_dim,
            dtype=torch.float32,
        )
        print(
            "[INFO] KiVi sim2sim cfg: "
            f"policy_input={self.input_shape}, "
            f"depth_shape={tuple(self.depth_history.shape)}, "
            f"visual_memory_shape={tuple(self.visual_memory.shape)}"
        )

    def _build_depth_obs(self) -> np.ndarray:
        # Placeholder for real MuJoCo/offboard depth. The value 2.0 matches the training max-distance fill.
        return self.depth_history.astype(np.float32)

    def _run_policy(self, history_obs: np.ndarray, current_obs: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            history_tensor = torch.from_numpy(history_obs[None, :]).float()
            current_tensor = torch.from_numpy(current_obs[None, :]).float()
            depth_tensor = torch.from_numpy(self._build_depth_obs()[None, :]).float()
            out = self._torch_policy(history_tensor, current_tensor, depth_tensor, self.visual_memory)
            if isinstance(out, tuple):
                actions, next_memory = out
                self.visual_memory = next_memory.detach().cpu()
            else:
                actions = out
        return actions.cpu().numpy()[0].astype(np.float32)
