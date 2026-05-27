# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.dolanga1.rough_env_cfg import (
    Dolanga1RoughEnvCfg,
)
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg, create_obsgroup_class


_FOOT_BODY_NAMES = ["LF_foot_link", "RF_foot_link", "LH_foot_link", "RH_foot_link"]


def _make_front_depth_camera(prim_path: str) -> RayCasterCameraCfg:
    return RayCasterCameraCfg(
        prim_path=prim_path,
        mesh_prim_paths=["/World/ground"],
        update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.25489, 0.0175, 0.07249),
            rot=(0.4056, -0.5792, 0.5792, -0.4056),
            convention="ros",
        ),
        data_types=["distance_to_image_plane"],
        max_distance=2.0,
        depth_clipping_behavior="max",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=43.984,
            vertical_aperture=18.4543,
            height=60,
            width=108,
        ),
    )


def _make_height_scanner(prim_path: str) -> RayCasterCfg:
    return RayCasterCfg(
        prim_path=prim_path,
        offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


def _make_foot_height_scanner(prim_path: str) -> RayCasterCfg:
    return RayCasterCfg(
        prim_path=prim_path,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class Dolanga1RoughKiviEnvCfg(Dolanga1RoughEnvCfg):
    """Dolanga1 rough locomotion with opt-in depth observations for KiVi."""

    def __post_init__(self):
        super().__post_init__()

        # Re-enable only the sensors/observations needed by the visual branch.
        base_prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        self.scene.front_depth_camera = _make_front_depth_camera(base_prim_path)
        self.scene.height_scanner = _make_height_scanner(base_prim_path)
        self.scene.height_scanner_fl_foot = _make_foot_height_scanner("{ENV_REGEX_NS}/Robot/LF_foot_link")
        self.scene.height_scanner_fr_foot = _make_foot_height_scanner("{ENV_REGEX_NS}/Robot/RF_foot_link")
        self.scene.height_scanner_hl_foot = _make_foot_height_scanner("{ENV_REGEX_NS}/Robot/LH_foot_link")
        self.scene.height_scanner_hr_foot = _make_foot_height_scanner("{ENV_REGEX_NS}/Robot/RH_foot_link")

        self.observations.front_camera_depth = ObservationsCfg.FrontDepthCfg()
        self.observations.front_camera_depth.front_depth_camera.params.update(
            {
                "camera_shake_deg": 2.0,
                "horizontal_fov_deg": 85.0,
                "vertical_fov_deg": 42.0,
                "history_len": 3,
                "occlusion_ratio_range": (0.0, 0.4),
                "occlusion_value": 2.0,
            }
        )
        self.observations.height_scan = ObservationsCfg.HeightScanCfg()
        self.observations.height_scan_feet = ObservationsCfg.HeightScanFeetCfg()
        self.observations.kinesthetic_explicit_target = create_obsgroup_class(
            "Dolanga1KiviKinestheticExplicitTargetCfg",
            {
                "base_lin_vel": ObsTerm(
                    func=mdp.base_lin_vel,
                    clip=(-100.0, 100.0),
                    scale=1.0,
                ),
                "foot_contact_forces_xz": ObsTerm(
                    func=mdp.foot_contact_forces_xz,
                    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_BODY_NAMES)},
                    clip=(-500.0, 500.0),
                    scale=1.0,
                ),
            },
            enable_corruption=False,
            concatenate_terms=True,
        )()
        self.observations.foot_contact_forces_xz = create_obsgroup_class(
            "Dolanga1KiviFootContactForcesXZCfg",
            {
                "foot_contact_forces_xz": ObsTerm(
                    func=mdp.foot_contact_forces_xz,
                    params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_BODY_NAMES)},
                    clip=(-500.0, 500.0),
                    scale=1.0,
                )
            },
            enable_corruption=False,
            concatenate_terms=True,
        )()

        # Keep the mirror camera disabled for now; the policy uses the real depth stream.
        self.scene.front_depth_camera_flip = None
        self.observations.front_camera_depth_flip = None

        if self.__class__.__name__ == "Dolanga1RoughKiviEnvCfg":
            self.disable_zero_weight_rewards()
