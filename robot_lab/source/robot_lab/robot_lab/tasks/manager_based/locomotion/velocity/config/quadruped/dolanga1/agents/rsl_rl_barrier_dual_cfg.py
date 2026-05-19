# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


ACTOR_OBS_GROUPS = [
    "base_ang_vel_history",
    "projected_gravity_history",
    "velocity_commands_history",
    "joint_pos_history",
    "joint_vel_history",
    "actions_history",
    "foot_positions_body",
    "phase",
    "stand_mode",
]

CRITIC_OBS_GROUPS = [
    "base_lin_vel",
    "base_ang_vel_history",
    "projected_gravity_history",
    "velocity_commands_history",
    "joint_pos_history",
    "joint_vel_history",
    "actions_history",
    "foot_positions_body",
    "phase",
    "stand_mode",
    "foot_contact_state",
    "height_scan_feet",
]


@configclass
class BarrierEstimatorActorCfg(RslRlMLPModelCfg):
    class_name: str = "robot_lab.third_party.rsl_rl_barrier_dual.models:BarrierEstimatorActor"

    estimator_input_groups: list[str] = ACTOR_OBS_GROUPS
    velocity_target_group: str = "base_lin_vel"
    foot_contact_target_group: str = "foot_contact_state"
    terrain_target_group: str = "height_scan_feet"
    estimator_hidden_dims: list[int] = [256, 128]
    estimator_activation: str = "elu"
    velocity_loss_weight: float = 1.0
    foot_contact_loss_weight: float = 1.0
    terrain_loss_weight: float = 1.0


@configclass
class BarrierDualAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "robot_lab.third_party.rsl_rl_barrier_dual.ppo_barrier_dual:BarrierDualPPO"
    surrogate_barrier_weight: float = 0.5
    estimator_loss_coef: float = 1.0


@configclass
class Dolanga1RoughBarrierDualRunnerCfg(RslRlOnPolicyRunnerCfg):
    # Keep rollouts short enough for responsive debugging; override to 400 for paper-style long rollouts.
    num_steps_per_env = 64
    max_iterations = 10000
    save_interval = 100
    experiment_name = "dolanga1_rough_barrier_dual"

    obs_groups = {
        "actor": ACTOR_OBS_GROUPS,
        "critic": CRITIC_OBS_GROUPS,
    }

    actor = BarrierEstimatorActorCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )

    critic = RslRlMLPModelCfg(
        obs_normalization=False,
        hidden_dims=[256, 128, 64],
        activation="elu",
        distribution_cfg=None,
    )

    critic_barrier = RslRlMLPModelCfg(
        obs_normalization=False,
        hidden_dims=[256, 128, 64],
        activation="elu",
        distribution_cfg=None,
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
