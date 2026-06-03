"""Configuration for the DolangH1 humanoid robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


DOLANGH1_LEG_JOINTS = [
    ".*_hip_.*",
    ".*_knee_joint",
    ".*_ankle_.*",
    ".*_foot.*",
]

DOLANGH1_ARM_JOINTS = [
    ".*_shoulder_.*",
    ".*_elbow_.*",
    ".*_wrist_.*",
]

DOLANGH1_TORSO_JOINTS = [
    "waist_joint",
    "lazy_waist_cross_joint",
    "lazy_torso_joint",
    "left_torso_joint",
    "right_torso_joint",
    "lazy_torso_left_stick_joint",
    "lazy_torso_right_stick_joint",
    "neck_yaw",
]


DOLANGH1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/dolangh1/urdf/robot.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_pitch_joint": -0.15,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.30,
            ".*_ankle_.*": -0.15,
            ".*_foot.*": 0.0,
            ".*_shoulder_.*": 0.0,
            ".*_elbow_.*": 0.0,
            ".*_wrist_.*": 0.0,
            "waist_joint": 0.0,
            "lazy_waist_cross_joint": 0.0,
            "lazy_torso_joint": 0.0,
            "left_torso_joint": 0.0,
            "right_torso_joint": 0.0,
            "lazy_torso_left_stick_joint": 0.0,
            "lazy_torso_right_stick_joint": 0.0,
            "neck_yaw": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=DOLANGH1_LEG_JOINTS,
            effort_limit_sim=120.0,
            velocity_limit_sim=20.0,
            stiffness=80.0,
            damping=3.0,
            friction=0.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=DOLANGH1_ARM_JOINTS,
            effort_limit_sim=60.0,
            velocity_limit_sim=20.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=DOLANGH1_TORSO_JOINTS,
            effort_limit_sim=80.0,
            velocity_limit_sim=15.0,
            stiffness=60.0,
            damping=3.0,
            friction=0.0,
        ),
    },
)
"""DolangH1 humanoid configuration imported from the local DolangH1 reference assets."""
