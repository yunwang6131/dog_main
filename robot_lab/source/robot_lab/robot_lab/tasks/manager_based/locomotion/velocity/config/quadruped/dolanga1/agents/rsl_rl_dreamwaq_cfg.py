# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


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


@configclass
class DreamWaQActorCfg(RslRlMLPModelCfg):
    class_name: str = "robot_lab.third_party.rsl_rl_dreamwaq.models:DreamWaQActor"

    history_groups: list[str] = HISTORY_GROUPS
    current_groups: list[str] = PROPRIO_GROUPS
    velocity_target_group: str = "base_lin_vel"
    reconstruction_target_groups: list[str] = PROPRIO_GROUPS
    latent_dim: int = 16
    velocity_dim: int = 3
    encoder_hidden_dims: list[int] = [512, 256]
    decoder_hidden_dims: list[int] = [256, 512]
    beta_kl: float = 0.01
    velocity_loss_weight: float = 1.0
    reconstruction_loss_weight: float = 1.0
    kl_loss_weight: float = 1.0
    logvar_min: float = -10.0
    logvar_max: float = 4.0


@configclass
class DreamWaQAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "robot_lab.third_party.rsl_rl_dreamwaq.algorithms:DreamWaQPPO"
    dreamwaq_next_obs_groups: list[str] = PROPRIO_GROUPS
    cenet_loss_coef: float = 1.0


@configclass
class Dolanga1RoughDreamWaQRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "dolanga1_rough_dreamwaq"

    obs_groups = {
        # Actor only consumes deployable proprioception; CENet reads HISTORY_GROUPS internally.
        "actor": PROPRIO_GROUPS,
        # DreamWaQ asymmetric critic: proprioception + true base velocity + exteroceptive height scan.
        "critic": PROPRIO_GROUPS + ["base_lin_vel", "disturbance", "height_scan"],
    }

    actor = DreamWaQActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[1024, 512, 256],
        activation="elu",
        obs_normalization=False,
    )

    algorithm = DreamWaQAlgorithmCfg(
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
class Dolanga1FlatDreamWaQRunnerCfg(Dolanga1RoughDreamWaQRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 5000
        self.experiment_name = "dolanga1_flat_dreamwaq"

