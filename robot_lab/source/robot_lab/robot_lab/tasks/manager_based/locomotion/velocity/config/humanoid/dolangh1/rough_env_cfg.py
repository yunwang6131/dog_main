# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
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

        self.actions.joint_pos.scale = {
            ".*_hip_pitch_joint": 0.32,
            ".*_hip_roll_joint": 0.18,
            ".*_hip_yaw_joint": 0.12,
            ".*_knee_joint": 0.40,
            ".*_ankle_.*": 0.30,
            ".*_foot.*": 0.14,
            ".*_shoulder_.*": 0.20,
            ".*_elbow_.*": 0.10,
            ".*_wrist_.*": 0.06,
            "waist_joint": 0.08,
            "lazy_waist_cross_joint": 0.08,
            "lazy_torso_joint": 0.08,
            "left_torso_joint": 0.08,
            "right_torso_joint": 0.08,
            "lazy_torso_left_stick_joint": 0.08,
            "lazy_torso_right_stick_joint": 0.08,
            "neck_yaw": 0.05,
        }
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_reset_joints = None

        self.rewards.is_terminated.weight = -200.0
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.35
        self.rewards.flat_orientation_l2.weight = -0.35
        self.rewards.base_height_l2.weight = -0.3
        self.rewards.base_height_l2.params["target_height"] = 0.9
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.base_bounce_l2 = RewTerm(
            func=mdp.base_bounce_l2,
            weight=-2.0,
            params={
                "target_height": 0.9,
                "height_deadband": 0.04,
                "vertical_velocity_weight": 1.5,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
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
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_arms_l1", -0.05, [".*shoulder.*", ".*elbow.*", ".*wrist.*"])
        self.rewards.create_joint_deviation_l1_rewterm(
            "joint_deviation_torso_l1",
            -0.03,
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
        self.rewards.joint_pos_penalty.weight = -0.4
        self.rewards.joint_mirror.weight = 0
        self.rewards.joint_mirror.params["mirror_joints"] = [["left_(hip|knee|ankle|foot).*", "right_(hip|knee|ankle|foot).*"]]

        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.action_mirror.weight = -0.02
        self.rewards.action_mirror.params["mirror_joints"] = [["left_(hip|knee|ankle|foot).*", "right_(hip|knee|ankle|foot).*"]]

        self.rewards.undesired_contacts.weight = 0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp
        self.rewards.face_velocity_direction = RewTerm(
            func=mdp.face_velocity_direction_exp,
            weight=1.0,
            params={
                "command_name": "base_velocity",
                "std": 0.5,
                "min_command_speed": 0.15,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = -0.4
        self.rewards.feet_contact.params["expect_contact_num"] = 1
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.no_fly = RewTerm(
            func=mdp.no_fly,
            weight=-4.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name])},
        )
        self.rewards.no_flight = RewTerm(
            func=mdp.no_flight_biped,
            weight=-4.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            },
        )
        self.rewards.single_foot_contact = RewTerm(
            func=mdp.single_foot_contact,
            weight=0.5,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
            },
        )
        self.rewards.feet_contact_without_cmd.weight = 0.2
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.3
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0.08
        self.rewards.feet_height_body.params["target_height"] = -0.12
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.gait.weight = 0.7
        self.rewards.gait.params["period"] = 1.2
        self.rewards.gait.params["offset"] = [0.0, 0.5]
        self.rewards.gait.params["threshold"] = 0.55
        self.rewards.gait.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 0.5

        if self.__class__.__name__ == "DolangH1RoughEnvCfg":
            self.disable_zero_weight_rewards()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.bad_orientation.params["asset_cfg"].body_names = [self.base_link_name]

        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
