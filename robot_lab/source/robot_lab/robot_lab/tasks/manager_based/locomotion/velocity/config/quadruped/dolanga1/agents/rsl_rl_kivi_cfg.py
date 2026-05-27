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
    """KiVi-lite runner: proprioceptive backbone plus a depth-supervised visual latent."""

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
            ("velocity_latent", True),
            ("kinesthetic_latent", True),
            ("visuospatial_latent", True),
        ],
        "critic": [
            "base_lin_vel_history_norm",
            "base_ang_vel_history_norm",
            "projected_gravity_history_norm",
            "velocity_commands_history",
            "joint_pos_history_norm",
            "joint_vel_history_norm",
            "actions_history_norm",
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
                "history_encoder": EncoderCfg(
                    type="processor",
                    obs_groups=HISTORY_GROUPS,
                    latent_name="blind_latent",
                    latent_dim=32,
                    network_cfg={
                        "hidden_dims": [512, 256],
                        "activation": "elu",
                    },
                ),
                "velocity_encoder": EncoderCfg(
                    type="processor",
                    obs_groups=["blind_latent"],
                    latent_name="velocity_latent",
                    latent_dim=3,
                    network_cfg={
                        "hidden_dims": [16],
                        "activation": "elu",
                        "normalize_output": False,
                        "clip_range": [-10.0, 10.0],
                    },
                ),
                "kinesthetic_encoder": EncoderCfg(
                    type="processor",
                    obs_groups=["blind_latent"],
                    latent_name="kinesthetic_latent",
                    latent_dim=16,
                    network_cfg={
                        "hidden_dims": [64],
                        "activation": "elu",
                        "normalize_output": True,
                    },
                ),
                "cnn_encoder": EncoderCfg(
                    type="cnn_processor",
                    obs_groups=["front_camera_depth"],
                    latent_name="cnn_latent",
                    latent_dim=None,
                    network_cfg={
                        "input_channels": 2,
                        "input_dim": (60, 60),
                        "output_channels": [8, 16, 32],
                        "kernel_size": 5,
                        "stride": 2,
                        "channel_last": False,
                    },
                ),
                "visuospatial_encoder": EncoderCfg(
                    type="processor",
                    obs_groups=["cnn_latent"],
                    latent_name="visuospatial_latent",
                    latent_dim=32,
                    network_cfg={
                        "hidden_dims": [128],
                        "activation": "elu",
                        "normalize_output": True,
                    },
                ),
                "height_scan_encoder": EncoderCfg(
                    type="processor",
                    obs_groups=["cnn_latent"],
                    latent_name="height_scan_latent",
                    latent_dim=17 * 17,
                    network_cfg={
                        "hidden_dims": [256, 256],
                        "activation": "elu",
                    },
                ),
            },
            decoder_cfgs={
                "velocity_decoder": DecoderCfg(
                    type="explicit",
                    loss_type="mse",
                    obs_groups=["velocity_latent"],
                    target_groups=["base_lin_vel_norm"],
                    loss_weight=1.0,
                ),
                "kinesthetic_decoder": DecoderCfg(
                    type="himloco",
                    loss_type=None,
                    obs_groups=["kinesthetic_latent"],
                    target_groups=NEXT_PROPRIO_TARGETS,
                    network_cfg={
                        "hidden_dims": [256, 128],
                        "activation": "elu",
                    },
                    add_on_cfg={"num_prototype": 32, "temperature": 3.0},
                    loss_weight=1.0,
                ),
                "height_scan_decoder": DecoderCfg(
                    type="explicit",
                    loss_type="l1",
                    obs_groups=["height_scan_latent"],
                    target_groups=["height_scan"],
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
