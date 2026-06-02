# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, \
    add_history_term
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
import copy 

##
# Pre-defined configs
##
from robot_lab.assets.dolanga import DOLANGA1_CFG  # isort: skip


@configclass
class Dolanga1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base_link"
    # Allow both explicit foot links and merged-fixed-joint fallback on calf links.
    foot_link_name = ".*_foot_link"
    # fmt: off
    joint_names = [
        "LF_hip_joint", "LF_thigh_joint", "LF_calf_joint",
        "RF_hip_joint", "RF_thigh_joint", "RF_calf_joint",
        "LH_hip_joint", "LH_thigh_joint", "LH_calf_joint",
        "RH_hip_joint", "RH_thigh_joint", "RH_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = DOLANGA1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        #self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        #self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.scene.height_scanner_fl_foot = None
        self.scene.height_scanner_fr_foot = None
        self.scene.height_scanner_hl_foot = None
        self.scene.height_scanner_hr_foot = None
        self.scene.front_depth_camera = None
        self.scene.front_depth_camera_flip = None
        # self.scene.terrain.max_init_terrain_level = 0
        terrain_generator = self.scene.terrain.terrain_generator
        terrain_generator.sub_terrains["flat"].proportion = 0.25
        terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.15
        terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.04, 0.16)
        terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.15
        terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.04, 0.16)
        terrain_generator.sub_terrains["boxes"].proportion = 0.30
        terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.08)
        terrain_generator.sub_terrains["random_rough"].proportion = 0.0
        terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.075
        terrain_generator.sub_terrains["hf_pyramid_slope"].slope_range = (0.0, 0.25)
        terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.075
        terrain_generator.sub_terrains["hf_pyramid_slope_inv"].slope_range = (0.0, 0.25)


        # ------------------------------Observations------------------------------
        # Keep canonical observation group names expected by trainer,
        # but swap selected groups to noisy variants for policy robustness.
        self.observations.base_ang_vel = copy.deepcopy(self.observations.base_ang_vel_with_noise)
        self.observations.projected_gravity = copy.deepcopy(self.observations.projected_gravity_with_noise)
        self.observations.joint_pos = copy.deepcopy(self.observations.joint_pos_with_noise)
        self.observations.joint_vel = copy.deepcopy(self.observations.joint_vel_with_noise)

        self.observations.base_lin_vel.base_lin_vel.scale = 2.0
        self.observations.base_ang_vel.base_ang_vel.scale = 0.25
        self.observations.joint_pos.joint_pos_rel.scale = 1.0
        self.observations.joint_vel.joint_vel_rel.scale = 0.05

        self.observations.joint_pos.joint_pos_rel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.joint_vel.joint_vel_rel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.joint_pos_with_noise.joint_pos_rel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.joint_vel_with_noise.joint_vel_rel.params["asset_cfg"].joint_names = self.joint_names

        self.observations.height_scan = None
        self.observations.height_scan_feet = None

        # ------------------------------history-----------------------------
        history_length = 10
        history_groups = [
            'base_lin_vel',
            'base_ang_vel',
            'projected_gravity',
            'joint_pos',
            'joint_vel',
            'velocity_commands',
            'actions',
        ]
        for group in history_groups:
            add_history_term(self, group, history_length)

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.4, "^(?!.*_hip_joint).*": 0.25} # hip 0.5
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0, 0),
                "y": (0, 0),
                "z": (0, 0),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (0, 0),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-10.0, 10.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-5.0, 5.0)
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)
        self.events.randomize_actuator_gains.params["distribution"] = "log_uniform"
        self.events.randomize_reset_joints.params["position_range"] = (0.9, 1.1)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.1, 0.1)
        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -3.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.45
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["LF_(hip|thigh|calf).*", "RH_(hip|thigh|calf).*"],
            ["RF_(hip|thigh|calf).*", "LH_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.03

        # Contact sensor
        self.rewards.undesired_contacts.weight = -10.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            ".*_thigh_link",
            ".*_hip_link",
            self.base_link_name,
        ]
        self.rewards.undesired_contacts_shank = copy.deepcopy(self.rewards.undesired_contacts)
        self.rewards.undesired_contacts_shank.params["sensor_cfg"].body_names = [".*_calf_link"]
        self.rewards.undesired_contacts_shank.weight = -0.5
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["threshold"] = 0.3
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["expect_contact_num"] = 2
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -1.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.3
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0.4
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0.3
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("LF_calf_link", "RH_calf_link"),
            ("RF_calf_link", "LH_calf_link"),
        )
        self.rewards.feet_air_time_variance.weight = -1.5
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.no_fly = RewTerm(
            func=mdp.no_fly,
            weight=-1.0,
            params={ 
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name),
            },
        )
        self.rewards.upward.weight = 0.05

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Dolanga1RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        # self.terminations.illegal_contact = None
        self.terminations.bad_orientation.params["asset_cfg"].body_names = [self.base_link_name]
        self.terminations.base_height = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ------------------------------Commands------------------------------
        # self.commands.base_velocity.ranges.lin_vel_x = (0, 2.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.lin_vel_x = (0, 0.8) #1.0
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5) #(-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
