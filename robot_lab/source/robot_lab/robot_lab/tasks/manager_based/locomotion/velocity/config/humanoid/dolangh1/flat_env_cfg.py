# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import DolangH1RoughEnvCfg


@configclass
class DolangH1FlatEnvCfg(DolangH1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.scene.height_scanner_fl_foot = None
        self.scene.height_scanner_fr_foot = None
        self.scene.height_scanner_hl_foot = None
        self.scene.height_scanner_hr_foot = None
        self.scene.front_depth_camera = None
        self.scene.front_depth_camera_flip = None
        self.observations.height_scan = None
        self.observations.height_scan_feet = None
        self.curriculum.terrain_levels = None

        if self.__class__.__name__ == "DolangH1FlatEnvCfg":
            self.disable_zero_weight_rewards()
