# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.sensors import RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.dolanga1.rough_env_cfg import (
    Dolanga1RoughEnvCfg,
)
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg


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


@configclass
class Dolanga1RoughKiviEnvCfg(Dolanga1RoughEnvCfg):
    """Dolanga1 rough locomotion with opt-in depth observations for KiVi-lite."""

    def __post_init__(self):
        super().__post_init__()

        # Re-enable only the sensors/observations needed by the visual branch.
        base_prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        self.scene.front_depth_camera = _make_front_depth_camera(base_prim_path)
        self.scene.height_scanner = _make_height_scanner(base_prim_path)

        self.observations.front_camera_depth = ObservationsCfg.FrontDepthCfg()
        self.observations.height_scan = ObservationsCfg.HeightScanCfg()

        # Keep the mirror camera and foot scans disabled for the first KiVi-lite pass.
        self.scene.front_depth_camera_flip = None
        self.observations.front_camera_depth_flip = None
        self.observations.height_scan_feet = None

        if self.__class__.__name__ == "Dolanga1RoughKiviEnvCfg":
            self.disable_zero_weight_rewards()
