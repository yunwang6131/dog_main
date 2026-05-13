# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ObservationsCfg,
    create_obsgroup_class,
)

from . import dreamwaq_mdp
from .rough_env_cfg import Dolanga1RoughEnvCfg


@configclass
class Dolanga1DreamWaQRoughEnvCfg(Dolanga1RoughEnvCfg):
    """Dolanga1 rough environment variant for DreamWaQ experiments.

    This leaves the existing Dolanga1 PPO environment untouched and only
    re-enables the body height scan used as privileged critic input.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + self.base_link_name,
            offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.6]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.observations.height_scan = ObservationsCfg.HeightScanCfg()
        disturbance_cfg = SceneEntityCfg("robot", body_names=[self.base_link_name])
        DisturbanceCfg = create_obsgroup_class(
            "DreamWaQDisturbanceCfg",
            {
                "disturbance": ObsTerm(
                    func=dreamwaq_mdp.disturbance_force_torque,
                    params={"asset_cfg": disturbance_cfg},
                    clip=(-100.0, 100.0),
                    scale=1.0,
                )
            },
            enable_corruption=False,
            concatenate_terms=True,
        )
        self.observations.disturbance = DisturbanceCfg()

        if self.__class__.__name__ == "Dolanga1DreamWaQRoughEnvCfg":
            self.disable_zero_weight_rewards()

