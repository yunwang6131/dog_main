# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.third_party.rsl_rl_est.cfgs.rl_cfg import (
    RslRlEstOnPolicyRunnerCfg,
    RslRlEstPpoActorCriticCfg,
    RslRlEstPpoAlgorithmCfg,
    EstimatorCfg,
    EncoderCfg,
    DecoderCfg,
)


@configclass
class Dolanga1RoughPPORunnerCfg(RslRlEstOnPolicyRunnerCfg):
    logger = "wandb"
    wandb_project = experiment_name = "dolanga1_rough_est"

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100

    obs_groups = {
        "normalize": [
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
            "actions",

            "base_ang_vel_history",
            "projected_gravity_history",
            "joint_pos_history",
            "joint_vel_history",
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

        # actor 实际输入
        # 注意：policy 不直接吃 base_lin_vel，因为真机上通常测不到真实 base linear velocity
        "policy": [
            "base_ang_vel_norm",
            "projected_gravity_norm",
            "velocity_commands",
            "joint_pos_norm",
            "joint_vel_norm",
            "actions_norm",
            ("velocity_latent", True),
            ("himloco_latent", True),
        ],

        # critic 输入，可以给更多信息
        "critic": [
            "base_lin_vel_history_norm",
            "base_ang_vel_history_norm",
            "projected_gravity_history_norm",
            "velocity_commands_history",
            "joint_pos_history_norm",
            "joint_vel_history_norm",
            "actions_history_norm",
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
            encoder_cfgs={
                # 1. 用历史 proprio 压成 blind_latent
                "history_encoder": EncoderCfg(
                    type="processor",
                    obs_groups=[
                        "base_ang_vel_history_norm",
                        "projected_gravity_history_norm",
                        "velocity_commands_history",
                        "joint_pos_history_norm",
                        "joint_vel_history_norm",
                        "actions_history_norm",
                    ],
                    latent_name="blind_latent",
                    latent_dim=32,
                    network_cfg={
                        "hidden_dims": [512, 256],
                        "activation": "elu",
                    },
                ),

                # 2. 从 blind_latent 估计 base linear velocity
                "velocity_encoder": EncoderCfg(
                    type="processor",
                    obs_groups=["blind_latent"],
                    latent_name="velocity_latent",
                    latent_dim=3,
                    network_cfg={
                        "hidden_dims": [16],
                        "activation": "elu",
                        "normalize_output": False,
                        "clip_range": [-10, 10],
                    },
                ),

                # 3. 再给 policy 一个隐式动力学 latent
                "himloco_encoder": EncoderCfg(
                    type="processor",
                    obs_groups=["blind_latent"],
                    latent_name="himloco_latent",
                    latent_dim=16,
                    network_cfg={
                        "hidden_dims": [64],
                        "activation": "elu",
                        "normalize_output": True,
                    },
                ),
            },

            decoder_cfgs={
                # 用 velocity_latent 回归真实 base_lin_vel
                "velocity_decoder": DecoderCfg(
                    type="explicit",
                    loss_type="mse",
                    obs_groups=["velocity_latent"],
                    target_groups=["base_lin_vel_norm"],
                    loss_weight=1.0,
                ),

                # 用 himloco_latent 预测下一步 proprio 状态
                "himloco_decoder": DecoderCfg(
                    type="himloco",
                    loss_type=None,
                    obs_groups=["himloco_latent"],
                    target_groups=[
                        "base_ang_vel_next_norm",
                        "projected_gravity_next_norm",
                        "velocity_commands_next",
                        "joint_pos_next_norm",
                        "joint_vel_next_norm",
                        "actions_next_norm",
                    ],
                    network_cfg={
                        "hidden_dims": [256, 128],
                        "activation": "elu",
                    },
                    add_on_cfg={
                        "num_prototype": 32,
                        "temperature": 3.0,
                    },
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

        # 这些 obs 会被额外生成 xxx_next，给 decoder 预测下一步用
        obs_go_next=[
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ],
    )


@configclass
class Dolanga1FlatPPORunnerCfg(Dolanga1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 5000
        self.experiment_name = "dolanga1_flat_est"