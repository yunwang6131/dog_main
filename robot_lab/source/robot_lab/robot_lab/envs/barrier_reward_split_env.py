# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Splits ``reward_buf`` into standard vs barrier streams (by reward term prefix) for dual-critic PPO."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


class BarrierRewardSplitManagerBasedRLEnv(ManagerBasedRLEnv):
    """Same as :class:`ManagerBasedRLEnv`, but writes ``extras['reward_standard'/'reward_barrier']``.

    Barrier terms are any active reward keys whose names start with one of the prefixes configured on
    :attr:`~ManagerBasedRLEnv.cfg`\ ``barrier_reward_term_prefixes`` (defaults to ``("barrier_style_",)``
    accessed via ``getattr`` when absent).
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_action_delay_buffers()

    def _init_action_delay_buffers(self) -> None:
        delay_range = getattr(self.cfg, "action_delay_env_steps_range", None)
        self._action_delay_enabled = delay_range is not None and delay_range[1] > 0
        if not self._action_delay_enabled:
            self._action_delay_max = 0
            self._action_delay_buf = None
            self._action_delay_steps = None
            return

        self._action_delay_min = int(delay_range[0])
        self._action_delay_max = int(delay_range[1])
        if self._action_delay_min < 0 or self._action_delay_max < self._action_delay_min:
            raise ValueError(
                f"Invalid action_delay_env_steps_range={delay_range!r}. Expected 0 <= min <= max."
            )

        action_dim = self.action_manager.total_action_dim
        self._action_delay_buf = torch.zeros(
            self.num_envs,
            self._action_delay_max + 1,
            action_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self._action_delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _reset_action_delay(self, env_ids: torch.Tensor) -> None:
        if not self._action_delay_enabled:
            return
        self._action_delay_buf[env_ids] = 0.0
        self._action_delay_steps[env_ids] = torch.randint(
            self._action_delay_min,
            self._action_delay_max + 1,
            (len(env_ids),),
            device=self.device,
            dtype=torch.long,
        )

    def _apply_action_with_optional_delay(self, action: torch.Tensor) -> None:
        if not self._action_delay_enabled:
            self.action_manager.process_action(action.to(self.device))
            return

        action = action.to(self.device)
        self._action_delay_buf[:, 1:] = self._action_delay_buf[:, :-1].clone()
        self._action_delay_buf[:, 0] = action
        env_ids = torch.arange(self.num_envs, device=self.device)
        delayed_action = self._action_delay_buf[env_ids, self._action_delay_steps]
        self.action_manager.process_action(delayed_action)

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self._reset_action_delay(env_ids)

    def _attach_barrier_reward_splits(self) -> None:
        prefixes: tuple[str, ...] = getattr(self.cfg, "barrier_reward_term_prefixes", ("barrier_style_",))
        rm = self.reward_manager
        names = rm.active_terms
        idxs = [i for i, n in enumerate(names) if any(n.startswith(p) for p in prefixes)]
        dt = self.step_dt
        if idxs:
            idx_t = torch.as_tensor(idxs, device=rm._reward_buf.device, dtype=torch.long)  # noqa: SLF001
            r_bar = rm._step_reward.index_select(1, idx_t).sum(dim=-1) * dt  # noqa: SLF001
        else:
            r_bar = torch.zeros_like(rm._reward_buf)  # noqa: SLF001
        r_std = self.reward_buf - r_bar
        self.extras["reward_standard"] = r_std.detach()
        self.extras["reward_barrier"] = r_bar.detach()

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # Derived from Isaac Lab ManagerBasedRLEnv.step (BSD-3-Clause); insert after reward compute.
        self._apply_action_with_optional_delay(action)

        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self._attach_barrier_reward_splits()

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
