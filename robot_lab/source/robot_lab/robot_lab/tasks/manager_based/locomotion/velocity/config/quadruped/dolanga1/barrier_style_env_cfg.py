# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Dolanga1 envs with ``barrier_style_*`` reward keys feeding the barrier critic stream."""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm

from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.dolanga1.flat_env_cfg import Dolanga1FlatEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.dolanga1.rough_env_cfg import Dolanga1RoughEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.mdp import barrier_style_rewards


def _add_barrier_placeholder(self) -> None:
    self.rewards.barrier_style_placeholder = RewTerm(
        func=barrier_style_rewards.barrier_style_stream_placeholder,
        weight=1e-6,
        params={},
    )


@configclass
class Dolanga1RoughBarrierStyleEnvCfg(Dolanga1RoughEnvCfg):
    barrier_reward_term_prefixes: tuple[str, ...] = ("barrier_style_",)

    def __post_init__(self):
        super().__post_init__()
        _add_barrier_placeholder(self)
        if self.__class__.__name__ == "Dolanga1RoughBarrierStyleEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Dolanga1FlatBarrierStyleEnvCfg(Dolanga1FlatEnvCfg):
    barrier_reward_term_prefixes: tuple[str, ...] = ("barrier_style_",)

    def __post_init__(self):
        super().__post_init__()
        _add_barrier_placeholder(self)
        if self.__class__.__name__ == "Dolanga1FlatBarrierStyleEnvCfg":
            self.disable_zero_weight_rewards()
