# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


PROPRIO_GROUPS = [
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
]

HISTORY_GROUPS = [
    "base_ang_vel_history",
    "projected_gravity_history",
    "velocity_commands_history",
    "joint_pos_history",
    "joint_vel_history",
    "actions_history",
]

CRITIC_OBS_GROUPS = [
    "base_lin_vel",
    *HISTORY_GROUPS,
    "height_scan_feet",
]

FLAT_CRITIC_OBS_GROUPS = [
    "base_lin_vel",
    *HISTORY_GROUPS,
]


@configclass
class DreamWaQActorCfg(RslRlMLPModelCfg):
    class_name: str = "robot_lab.third_party.rsl_rl_barrier_dual.models:BarrierDreamWaQActor"

    history_groups: list[str] = HISTORY_GROUPS
    current_groups: list[str] = PROPRIO_GROUPS
    velocity_target_group: str = "base_lin_vel"
    reconstruction_target_groups: list[str] = PROPRIO_GROUPS
    latent_dim: int = 16
    terrain_latent_dim: int = 8
    terrain_target_group: str = "height_scan_feet"
    terrain_stats_dim: int = 20
    velocity_dim: int = 3
    encoder_hidden_dims: list[int] = [512, 256]
    decoder_hidden_dims: list[int] = [256, 512]
    beta_kl: float = 0.03
    velocity_loss_weight: float = 0.5
    reconstruction_loss_weight: float = 0.25
    terrain_loss_weight: float = 0.25
    kl_loss_weight: float = 1.0
    logvar_min: float = -10.0
    logvar_max: float = 4.0


@configclass
class DreamWaQAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "robot_lab.third_party.rsl_rl_barrier_dual.ppo_dreamwaq:DreamWaQPPO"
    estimator_loss_coef: float = 1.0
    dreamwaq_next_obs_groups: list[str] = PROPRIO_GROUPS


@configclass
class DolangD2RoughDreamWaQRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 10000
    save_interval = 100
    experiment_name = "dolangD2_rough_dreamwaq"
    logger = "wandb"
    wandb_project = "humanoid"
    obs_groups = {
        "actor": PROPRIO_GROUPS,
        "critic": CRITIC_OBS_GROUPS,
    }

    actor = DreamWaQActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )

    critic = RslRlMLPModelCfg(
        obs_normalization=False,
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=None,
    )

    algorithm = DreamWaQAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
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
class DolangD2FlatDreamWaQRunnerCfg(DolangD2RoughDreamWaQRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 5000
        self.experiment_name = "dolangD2_flat_dreamwaq"
        self.obs_groups = {
            "actor": PROPRIO_GROUPS,
            "critic": FLAT_CRITIC_OBS_GROUPS,
        }
        self.actor.terrain_loss_weight = 0.0
        self.algorithm.estimator_loss_coef = 0.2


@configclass
class DolangD2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Plain PPO fallback for quick asset bring-up."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "dolangD2_rough"
    logger = "wandb"
    wandb_project = "robot_lab"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.006,
        max_grad_norm=0.5,
    )


@configclass
class DolangD2FlatPPORunnerCfg(DolangD2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1500
        self.experiment_name = "dolangD2_flat"
