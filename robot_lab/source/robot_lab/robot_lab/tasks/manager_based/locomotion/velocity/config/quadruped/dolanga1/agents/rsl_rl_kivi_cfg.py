# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.third_party.rsl_rl_est.cfgs import (
    DecoderCfg,
    EncoderCfg,
    EstimatorCfg,
    RslRlEstOnPolicyRunnerCfg,
    RslRlEstPpoActorCriticCfg,
    RslRlEstPpoAlgorithmCfg,
)


PROPRIO_GROUPS = [
    "base_ang_vel_with_noise_norm",
    "projected_gravity_with_noise_norm",
    "velocity_commands",
    "joint_pos_with_noise_norm",
    "joint_vel_with_noise_norm",
    "actions_norm",
]

HISTORY_GROUPS = [
    "base_ang_vel_history_norm",
    "projected_gravity_history_norm",
    "velocity_commands_history",
    "joint_pos_history_norm",
    "joint_vel_history_norm",
    "actions_history_norm",
]

NEXT_PROPRIO_TARGETS = [
    "base_ang_vel_next_norm",
    "projected_gravity_next_norm",
    "velocity_commands_next",
    "joint_pos_next_norm",
    "joint_vel_next_norm",
    "actions_next_norm",
]


@configclass
class Dolanga1RoughKiviPPORunnerCfg(RslRlEstOnPolicyRunnerCfg):
    """KiVi runner: proprioceptive backbone plus proprio-depth visuospatial fusion."""

    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 200
    experiment_name = "dolanga1_rough_kivi"

    obs_groups = {
        "normalize": [
            "base_ang_vel_with_noise",
            "projected_gravity_with_noise",
            "joint_pos_with_noise",
            "joint_vel_with_noise",
            "actions",
            "actions_history",
        ],
        "privilege_normalize": [
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "base_lin_vel_history",
            "base_ang_vel_history",
            "projected_gravity_history",
            "joint_pos_history",
            "joint_vel_history",
            "base_ang_vel_next",
            "projected_gravity_next",
            "joint_pos_next",
            "joint_vel_next",
            "actions_next",
        ],
        "policy": [
            *PROPRIO_GROUPS,
            "kinesthetic_explicit_latent",
            "kinesthetic_implicit_latent",
            "visuospatial_state_latent",
            "visuospatial_foot_latent",
        ],
        "critic": [
            "base_lin_vel_norm",
            "base_ang_vel_norm",
            "projected_gravity_norm",
            "velocity_commands",
            "joint_pos_norm",
            "joint_vel_norm",
            "actions_norm",
            "foot_contact_forces_xz",
            "height_scan",
        ],
    }

    policy = RslRlEstPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
        estimator_cfg=EstimatorCfg(
            memory_buffer_cfgs=None,
            encoder_cfgs={
                "kinesthetic_encoder": EncoderCfg(
                    type="kivi_kinesthetic",
                    obs_groups=HISTORY_GROUPS,
                    latent_name="kinesthetic_implicit_latent",
                    latent_dim=20,
                    network_cfg={
                        "feature_name": "kinesthetic_feature",
                        "feature_dim": 256,
                        "explicit_latent_name": "kinesthetic_explicit_latent",
                        "explicit_dim": 11,
                        "implicit_latent_name": "kinesthetic_implicit_latent",
                        "implicit_dim": 20,
                        "hidden_dims": [512, 256],
                        "explicit_hidden_dims": [64],
                        "implicit_hidden_dims": [128, 64],
                        "posterior_depth_group": None,
                        "activation": "elu",
                        "normalize_output": False,
                        "logvar_min": -10.0,
                        "logvar_max": 4.0,
                    },
                ),
                "visuospatial_encoder": EncoderCfg(
                    type="kivi_visuospatial",
                    obs_groups=HISTORY_GROUPS + ["front_camera_depth"],
                    latent_name="visuospatial_latent",
                    latent_dim=20,
                    network_cfg={
                        "proprio_groups": HISTORY_GROUPS,
                        "depth_group": "front_camera_depth",
                        "state_latent_name": "visuospatial_state_latent",
                        "state_latent_dim": 12,
                        "foot_latent_name": "visuospatial_foot_latent",
                        "foot_latent_dim": 8,
                        "embed_dim": 32,
                        "num_heads": 4,
                        "num_layers": 2,
                        "dim_feedforward": 128,
                        "dropout": 0.0,
                        "activation": "elu",
                        "transformer_activation": "gelu",
                        "proprio_hidden_dims": [128],
                        "conv_channels": [8, 16, 32],
                        "visual_token_grid": 4,
                        "num_memory_tokens": 4,
                        "num_transitions_per_env": num_steps_per_env,
                        "state_hidden_dims": [128],
                        "foot_hidden_dims": [128],
                        "height_latent_name": "height_scan_latent",
                        "height_target_group": "height_scan",
                        "height_hidden_dims": [256, 256],
                        "foot_height_latent_name": "height_scan_feet_latent",
                        "foot_height_target_group": "height_scan_feet",
                        "foot_height_hidden_dims": [128, 128],
                        "normalize_output": False,
                    },
                ),
            },
            decoder_cfgs={
                "kinesthetic_explicit_decoder": DecoderCfg(
                    type="explicit",
                    loss_type="mse",
                    obs_groups=["kinesthetic_explicit_latent"],
                    target_groups=["kinesthetic_explicit_target"],
                    loss_weight=1.0,
                ),
                "kinesthetic_implicit_decoder": DecoderCfg(
                    type="implicit",
                    loss_type="mse",
                    obs_groups=["kinesthetic_implicit_latent"],
                    target_groups=NEXT_PROPRIO_TARGETS,
                    network_cfg={
                        "hidden_dims": [256, 128],
                        "activation": "elu",
                    },
                    loss_weight=1.0,
                ),
                "kinesthetic_kl_decoder": DecoderCfg(
                    type="kl",
                    loss_type=None,
                    obs_groups=[
                        "kinesthetic_implicit_latent_mu",
                        "kinesthetic_implicit_latent_logvar",
                    ],
                    target_groups=[],
                    loss_weight=0.001,
                ),
                "height_scan_decoder": DecoderCfg(
                    type="explicit",
                    loss_type="l1",
                    obs_groups=["height_scan_latent"],
                    target_groups=["height_scan"],
                    loss_weight=1.0,
                ),
                "height_scan_feet_decoder": DecoderCfg(
                    type="explicit",
                    loss_type="l1",
                    obs_groups=["height_scan_feet_latent"],
                    target_groups=["height_scan_feet"],
                    loss_weight=1.0,
                ),
            },
        ),
    )

    algorithm = RslRlEstPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_go_next=[
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ],
    )
