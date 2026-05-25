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

CRITIC_OBS_GROUPS = [
    "base_lin_vel",
    *HISTORY_GROUPS,
    "foot_positions_body",
    "phase",
    "stand_mode",
    "foot_contact_state",
    "height_scan_feet",
]


@configclass
class BarrierDreamWaQActorCfg(RslRlMLPModelCfg):
    class_name: str = "robot_lab.third_party.rsl_rl_barrier_dual.models:BarrierDreamWaQActor"

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
class BarrierDualAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "robot_lab.third_party.rsl_rl_barrier_dual.ppo_barrier_dual:BarrierDualPPO"
    surrogate_barrier_weight: float = 0.5
    estimator_loss_coef: float = 1.0
    dreamwaq_next_obs_groups: list[str] = PROPRIO_GROUPS


@configclass
class Dolanga1RoughBarrierDualDebugRunnerCfg(RslRlOnPolicyRunnerCfg):
    # Debug-friendly default. Use the explicit paper runner when matching the paper's long rollouts.
    num_steps_per_env = 64
    max_iterations = 10000
    save_interval = 100
    experiment_name = "dolanga1_rough_barrier_dual"

    obs_groups = {
        "actor": PROPRIO_GROUPS,
        "critic": CRITIC_OBS_GROUPS,
    }

    actor = BarrierDreamWaQActorCfg(
        hidden_dims=[512, 256, 128],
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
class Dolanga1RoughBarrierDualPaperRunnerCfg(Dolanga1RoughBarrierDualDebugRunnerCfg):
    num_steps_per_env = 400
    experiment_name = "dolanga1_rough_barrier_dual_paper"


@configclass
class Dolanga1RoughBarrierDualRunnerCfg(Dolanga1RoughBarrierDualDebugRunnerCfg):
    """Backward-compatible alias for the debug-friendly BarrierDual runner."""


@configclass
class Dolanga1FlatBarrierDualDebugRunnerCfg(Dolanga1RoughBarrierDualDebugRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 5000
        self.experiment_name = "dolanga1_flat_barrier_dual"


@configclass
class Dolanga1FlatBarrierDualPaperRunnerCfg(Dolanga1FlatBarrierDualDebugRunnerCfg):
    num_steps_per_env = 400

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "dolanga1_flat_barrier_dual_paper"


@configclass
class Dolanga1FlatBarrierDualRunnerCfg(Dolanga1FlatBarrierDualDebugRunnerCfg):
    """Backward-compatible alias for the debug-friendly flat BarrierDual runner."""
