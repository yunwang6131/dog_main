# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class BarrierDualAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "robot_lab.third_party.rsl_rl_barrier_dual.ppo_barrier_dual:BarrierDualPPO"
    surrogate_barrier_weight: float = 0.5


@configclass
class Dolanga1RoughBarrierDualRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "dolanga1_rough_barrier_dual"

    obs_groups = {
        "actor": [
            "base_ang_vel_history",
            "projected_gravity_history",
            "velocity_commands_history",
            "joint_pos_history",
            "joint_vel_history",
            "actions_history",
        ],
        "critic": [
            "base_lin_vel_history",
            "base_ang_vel_history",
            "projected_gravity_history",
            "velocity_commands_history",
            "joint_pos_history",
            "joint_vel_history",
            "actions_history",
        ],
    }

    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )

    critic = RslRlMLPModelCfg(
        obs_normalization=False,
        hidden_dims=[512, 256, 128],
        activation="elu",
    )

    critic_barrier = RslRlMLPModelCfg(
        obs_normalization=False,
        hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = BarrierDualAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Dolanga1FlatBarrierDualRunnerCfg(Dolanga1RoughBarrierDualRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 5000
        self.experiment_name = "dolanga1_flat_barrier_dual"
