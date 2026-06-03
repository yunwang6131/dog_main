# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

import copy
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, add_history_term

from robot_lab.assets.dolangh1 import DOLANGH1_CFG  # isort: skip


@configclass
class DolangH1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base_link"
    foot_link_name = ".*_foot_link"

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = DOLANGH1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/.*_link"
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_fl_foot.prim_path = "{ENV_REGEX_NS}/Robot/left_foot_link"
        self.scene.height_scanner_fr_foot.prim_path = "{ENV_REGEX_NS}/Robot/right_foot_link"
        self.scene.height_scanner_hl_foot.prim_path = "{ENV_REGEX_NS}/Robot/left_foot_link"
        self.scene.height_scanner_hr_foot.prim_path = "{ENV_REGEX_NS}/Robot/right_foot_link"

        self.observations.base_ang_vel = copy.deepcopy(self.observations.base_ang_vel_with_noise)
        self.observations.projected_gravity = copy.deepcopy(self.observations.projected_gravity_with_noise)
        self.observations.joint_pos = copy.deepcopy(self.observations.joint_pos_with_noise)
        self.observations.joint_vel = copy.deepcopy(self.observations.joint_vel_with_noise)

        self.observations.base_lin_vel.base_lin_vel.scale = 2.0
        self.observations.base_ang_vel.base_ang_vel.scale = 0.25
        self.observations.joint_pos.joint_pos_rel.scale = 1.0
        self.observations.joint_vel.joint_vel_rel.scale = 0.05

        self.observations.height_scan = None

        history_length = 10
        history_groups = [
            "base_ang_vel",
            "projected_gravity",
            "velocity_commands",
            "joint_pos",
            "joint_vel",
            "actions",
        ]
        for group in history_groups:
            add_history_term(self, group, history_length)

        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.is_terminated.weight = -200.0
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.joint_torques_l2.weight = -1.5e-7
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [
            ".*_hip_.*",
            ".*_knee_joint",
            ".*_ankle_.*",
            ".*_foot.*",
        ]
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -1.25e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = [".*_hip_.*", ".*_knee_joint"]
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.1, [".*hip_yaw.*", ".*hip_roll.*"])
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_arms_l1", -0.1, [".*shoulder.*", ".*elbow.*", ".*wrist.*"])
        self.rewards.create_joint_deviation_l1_rewterm(
            "joint_deviation_torso_l1",
            -0.1,
            [
                "waist_joint",
                "lazy_waist_cross_joint",
                "lazy_torso_joint",
                "left_torso_joint",
                "right_torso_joint",
                "lazy_torso_left_stick_joint",
                "lazy_torso_right_stick_joint",
                "neck_yaw",
            ],
        )
        self.rewards.joint_pos_limits.weight = -0.5
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = 0
        self.rewards.stand_still.weight = 0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = 0
        self.rewards.joint_mirror.params["mirror_joints"] = [["left_(hip|knee|ankle|foot).*", "right_(hip|knee|ankle|foot).*"]]

        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.action_mirror.weight = 0
        self.rewards.action_mirror.params["mirror_joints"] = [["left_(hip|knee|ankle|foot).*", "right_(hip|knee|ankle|foot).*"]]

        self.rewards.undesired_contacts.weight = 0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp

        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 1.0

        if self.__class__.__name__ == "DolangH1RoughEnvCfg":
            self.disable_zero_weight_rewards()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.bad_orientation.params["asset_cfg"].body_names = [self.base_link_name]

        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
