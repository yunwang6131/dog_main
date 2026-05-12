# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.terrains.trimesh import *
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from robot_lab.terrains.trimesh import mesh_terrains

"""
Different trimesh terrain configurations.
"""

@configclass
class MeshFlatHighBoxCfg(SubTerrainBaseCfg):
    function = mesh_terrains.flat_high_box

    platform_width: float = 2.5
    height_range: tuple[float, float] = (0.1, 0.6)  # difficulty↑ →  height ↑

@configclass
class MeshGapTerrainCfg(SubTerrainBaseCfg):
    function = mesh_terrains.gap_terrain

    platform_width: float = 2.0
    gap_width_range: tuple[float, float] = (0.2, 0.5)
    gap_deepth : float = 0.5