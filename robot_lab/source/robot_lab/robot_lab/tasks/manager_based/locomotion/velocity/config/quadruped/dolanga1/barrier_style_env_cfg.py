# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Dolanga1 envs with paper-aligned barrier-style rewards for BarrierDual training."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.dolanga1.flat_env_cfg import Dolanga1FlatEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.dolanga1.rough_env_cfg import Dolanga1RoughEnvCfg
from robot_lab.tasks.manager_based.locomotion.velocity.mdp import barrier_style_rewards
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ObservationsCfg,
    create_obsgroup_class,
)

# Diagonal trot: LF/RH in phase, RF/LH in opposite phase (paper Sec. III-B, T=0.72 s).
_TROT_PERIOD = 0.72
_TROT_PHASE_OFFSETS = [0.0, 0.5, 0.5, 0.0]  # LF, RF, LH, RH
_FOOT_BODY_NAMES = ["LF_foot_link", "RF_foot_link", "LH_foot_link", "RH_foot_link"]
_FOOT_SENSOR_CFG = SceneEntityCfg("contact_forces", body_names=_FOOT_BODY_NAMES)
_FOOT_ASSET_CFG = SceneEntityCfg("robot", body_names=_FOOT_BODY_NAMES)
_FOOT_ASSET_AND_JOINT_CFG = SceneEntityCfg("robot", joint_names=[".*"], body_names=_FOOT_BODY_NAMES)
_ROBOT_JOINT_CFG = SceneEntityCfg("robot", joint_names=[".*"])
_FRONT_BODY_NAMES = ["LF_thigh_link", "RF_thigh_link"]
_HIND_BODY_NAMES = ["LH_thigh_link", "RH_thigh_link"]
_FOOT_TERRAIN_SENSOR_CFGS = [
    SceneEntityCfg("height_scanner_fl_foot"),
    SceneEntityCfg("height_scanner_fr_foot"),
    SceneEntityCfg("height_scanner_hl_foot"),
    SceneEntityCfg("height_scanner_hr_foot"),
]
_FRONT_TERRAIN_SENSOR_CFGS = _FOOT_TERRAIN_SENSOR_CFGS[:2]
_HIND_TERRAIN_SENSOR_CFGS = _FOOT_TERRAIN_SENSOR_CFGS[2:]


def _make_foot_height_scanner(prim_path: str) -> RayCasterCfg:
    return RayCasterCfg(
        prim_path=prim_path,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 2.0)),
        ray_alignment="yaw",
        # The paper samples terrain within 5 cm around each foot.
        pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


def _tune_paper_training_recipe(cfg) -> None:
    cfg.scene.num_envs = 400
    cfg.decimation = 2
    cfg.episode_length_s = 4.0
    cfg.sim.render_interval = cfg.decimation
    cfg.commands.base_velocity.resampling_time_range = (4.0, 4.0)


def _enable_paper_aux_sensors(cfg) -> None:
    cfg.scene.height_scanner_fl_foot = _make_foot_height_scanner("{ENV_REGEX_NS}/Robot/LF_foot_link")
    cfg.scene.height_scanner_fr_foot = _make_foot_height_scanner("{ENV_REGEX_NS}/Robot/RF_foot_link")
    cfg.scene.height_scanner_hl_foot = _make_foot_height_scanner("{ENV_REGEX_NS}/Robot/LH_foot_link")
    cfg.scene.height_scanner_hr_foot = _make_foot_height_scanner("{ENV_REGEX_NS}/Robot/RH_foot_link")
    for sensor_name in (
        "height_scanner_fl_foot",
        "height_scanner_fr_foot",
        "height_scanner_hl_foot",
        "height_scanner_hr_foot",
    ):
        getattr(cfg.scene, sensor_name).update_period = cfg.decimation * cfg.sim.dt


def _add_paper_observations(cfg) -> None:
    PhaseObsCfg = create_obsgroup_class(
        "Dolanga1BarrierPhaseCfg",
        {
            "phase": ObsTerm(
                func=mdp.phase_with_command,
                params={"cycle_time": _TROT_PERIOD, "command_name": "base_velocity", "stand_threshold": 0.2},
                clip=(-1.0, 1.0),
                scale=1.0,
            )
        },
    )
    StandModeObsCfg = create_obsgroup_class(
        "Dolanga1BarrierStandModeCfg",
        {
            "stand_mode": ObsTerm(
                func=mdp.command_stand_mode,
                params={"command_name": "base_velocity", "threshold": 0.2},
                clip=(0.0, 1.0),
                scale=1.0,
            )
        },
    )
    FootPosBodyObsCfg = create_obsgroup_class(
        "Dolanga1BarrierFootPosBodyCfg",
        {
            "foot_positions_body": ObsTerm(
                func=mdp.feet_positions_body,
                params={"asset_cfg": _FOOT_ASSET_CFG},
                clip=(-100.0, 100.0),
                scale=1.0,
            )
        },
    )
    FootContactObsCfg = create_obsgroup_class(
        "Dolanga1BarrierFootContactCfg",
        {
            "foot_contact_state": ObsTerm(
                func=mdp.foot_contact_state,
                params={"sensor_cfg": _FOOT_SENSOR_CFG, "contact_threshold": 1.0},
                clip=(0.0, 1.0),
                scale=1.0,
            )
        },
    )
    cfg.observations.phase = PhaseObsCfg()
    cfg.observations.stand_mode = StandModeObsCfg()
    cfg.observations.foot_positions_body = FootPosBodyObsCfg()
    cfg.observations.foot_contact_state = FootContactObsCfg()
    cfg.observations.height_scan_feet = ObservationsCfg.HeightScanFeetCfg()


def _add_barrier_style_rewards(cfg) -> None:
    """Register barrier_style_* terms (summed into the barrier critic stream)."""
    cfg.rewards.barrier_style_gait = RewTerm(
        func=barrier_style_rewards.barrier_style_gait,
        weight=1.0,
        params={
            "period": _TROT_PERIOD,
            "phase_offsets": _TROT_PHASE_OFFSETS,
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "d_lower": -0.6,
            "d_upper": 2.0,
            "delta": 0.1,
            "command_name": "base_velocity",
            "command_threshold": 0.2,
            "stand_threshold": 0.2,
        },
    )
    cfg.rewards.barrier_style_foot_clearance = RewTerm(
        func=barrier_style_rewards.barrier_style_foot_clearance,
        weight=1.0,
        params={
            "period": _TROT_PERIOD,
            "phase_offsets": _TROT_PHASE_OFFSETS,
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "asset_cfg": _FOOT_ASSET_CFG,
            "terrain_sensor_cfgs": _FOOT_TERRAIN_SENSOR_CFGS,
            "p_des": 0.15,
            "d_lower_gait": -0.6,
            "d_lower_clearance": -0.08,
            "delta": 0.01,
            "command_name": "base_velocity",
            "command_threshold": 0.2,
            "stand_threshold": 0.2,
        },
    )
    cfg.rewards.barrier_style_joint_position = RewTerm(
        func=barrier_style_rewards.barrier_style_joint_position,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "hip_joint_names": [".*_hip_joint"],
            "thigh_joint_names": [".*_thigh_joint"],
            "calf_joint_names": [".*_calf_joint"],
        },
    )
    cfg.rewards.barrier_style_body_height = RewTerm(
        func=barrier_style_rewards.barrier_style_body_height,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "front_body_names": _FRONT_BODY_NAMES,
            "hind_body_names": _HIND_BODY_NAMES,
            "front_terrain_sensor_cfgs": _FRONT_TERRAIN_SENSOR_CFGS,
            "hind_terrain_sensor_cfgs": _HIND_TERRAIN_SENSOR_CFGS,
            # Scaled for Dolanga1 (~0.47 m standing); paper Table I targets HOUND-scale heights.
            "front_bounds": (0.18, 0.38),
            "hind_bounds": (0.18, 0.38),
            "front_delta": 0.04,
            "hind_delta": 0.04,
        },
    )
    cfg.rewards.barrier_style_velocity_tracking = RewTerm(
        func=barrier_style_rewards.barrier_style_velocity_tracking,
        weight=1.0,
        params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards.barrier_style_base_motion = RewTerm(
        func=barrier_style_rewards.barrier_style_base_motion,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards.barrier_style_joint_velocity = RewTerm(
        func=barrier_style_rewards.barrier_style_joint_velocity,
        weight=1.0,
        params={"asset_cfg": _ROBOT_JOINT_CFG, "vel_bounds": (-8.0, 8.0), "delta": 2.0},
    )


def _add_paper_standard_reward(cfg) -> None:
    cfg.rewards.paper_standard_reward = RewTerm(
        func=mdp.PaperStandardReward,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": _FOOT_ASSET_AND_JOINT_CFG,
            "sensor_cfg": _FOOT_SENSOR_CFG,
            "front_body_names": _FRONT_BODY_NAMES,
            "hind_body_names": _HIND_BODY_NAMES,
            "front_terrain_sensor_cfgs": _FRONT_TERRAIN_SENSOR_CFGS,
            "hind_terrain_sensor_cfgs": _HIND_TERRAIN_SENSOR_CFGS,
            "lin_vel_weight": 3.0,
            "ang_vel_weight": 1.5,
            "neg_exp_scale": 0.2,
            "torque_weight": 2.5e-5,
            "action_rate_weight": 0.03,
            "foot_slip_weight": 0.3,
            "foot_position_weight": 0.5,
            "front_hind_balance_weight": 1.0,
            "use_orientation_penalty": False,
        },
    )


def _tune_for_barrier_training(cfg) -> None:
    """Replace ad-hoc standard shaping with the paper-style standard reward and terminations."""
    disabled_reward_names = (
        "lin_vel_z_l2",
        "ang_vel_xy_l2",
        "flat_orientation_l2",
        "base_height_l2",
        "body_lin_acc_l2",
        "joint_torques_l2",
        "joint_vel_l2",
        "joint_acc_l2",
        "joint_pos_limits",
        "joint_vel_limits",
        "joint_power",
        "stand_still",
        "joint_pos_penalty",
        "joint_mirror",
        "action_rate_l2",
        "undesired_contacts",
        "undesired_contacts_shank",
        "contact_forces",
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "feet_air_time",
        "feet_contact",
        "feet_contact_without_cmd",
        "feet_stumble",
        "feet_slide",
        "feet_height",
        "feet_height_body",
        "feet_gait",
        "feet_air_time_variance",
        "no_fly",
        "upward",
    )
    for reward_name in disabled_reward_names:
        reward = getattr(cfg.rewards, reward_name, None)
        if reward is not None:
            reward.weight = 0.0

    cfg.rewards.is_terminated.weight = -5.0
    cfg.terminations.illegal_contact.params["sensor_cfg"].body_names = [
        ".*_hip_link",
        ".*_thigh_link",
        cfg.base_link_name,
    ]


@configclass
class Dolanga1RoughBarrierStyleEnvCfg(Dolanga1RoughEnvCfg):
    barrier_reward_term_prefixes: tuple[str, ...] = ("barrier_style_",)

    def __post_init__(self):
        super().__post_init__()
        _tune_paper_training_recipe(self)
        _enable_paper_aux_sensors(self)
        _add_paper_observations(self)
        _add_barrier_style_rewards(self)
        _add_paper_standard_reward(self)
        _tune_for_barrier_training(self)
        if self.__class__.__name__ == "Dolanga1RoughBarrierStyleEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class Dolanga1FlatBarrierStyleEnvCfg(Dolanga1FlatEnvCfg):
    barrier_reward_term_prefixes: tuple[str, ...] = ("barrier_style_",)

    def __post_init__(self):
        super().__post_init__()
        _tune_paper_training_recipe(self)
        _enable_paper_aux_sensors(self)
        _add_paper_observations(self)
        _add_barrier_style_rewards(self)
        _add_paper_standard_reward(self)
        _tune_for_barrier_training(self)
        if self.__class__.__name__ == "Dolanga1FlatBarrierStyleEnvCfg":
            self.disable_zero_weight_rewards()
