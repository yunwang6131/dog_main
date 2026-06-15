from __future__ import annotations

import numpy as np
import trimesh
from typing import TYPE_CHECKING

from isaaclab.terrains.trimesh.utils import make_border

if TYPE_CHECKING:
    from . import mesh_terrains_cfg

_GROUND_THICK = 0.05


def _make_box(
    half_extents: tuple[float, float, float],
    center: tuple[float, float, float],
    euler_xyz: tuple[float, float, float] | None = None,
) -> trimesh.Trimesh:
    """Create an axis-aligned or rotated box (MuJoCo half-extents and body-frame euler xyz)."""
    transform = trimesh.transformations.translation_matrix(center)
    if euler_xyz is not None:
        rot = trimesh.transformations.euler_matrix(*euler_xyz, axes="rxyz")
        transform = transform @ rot
    extents = (2.0 * half_extents[0], 2.0 * half_extents[1], 2.0 * half_extents[2])
    return trimesh.creation.box(extents=extents, transform=transform)


def flat_high_box(
        difficulty: float, cfg: "mesh_terrains_cfg.MeshFlatHighBoxCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """
    Three-layer terrain:
      - Outer flat ground (z=0)
      - Middle raised ring wall (height H)
      - Inner flat platform (z=0 again)
    """
    import numpy as np
    import trimesh

    size_x, size_y = cfg.size
    cx, cy = 0.5 * size_x, 0.5 * size_y

    p = float(max(0.0, cfg.platform_width))  # 最中心平台宽度
    max_inner = min(size_x, size_y)
    p = min(p, max_inner)

    h_min, h_max = cfg.height_range
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    H = float(h_min + difficulty * (h_max - h_min))  # 中层高墙高度

    PLATFORM_THICK = 0.05
    meshes = []

    def make_box(dx: float, dy: float, dz: float, x: float, y: float, z: float):
        if dx <= 1e-8 or dy <= 1e-8 or dz <= 1e-8:
            return None
        return trimesh.creation.box(
            extents=(dx, dy, dz),
            transform=trimesh.transformations.translation_matrix((x, y, z)),
        )

    # ======================
    # first layer（z=0）
    # ======================
    outer_ground = make_box(size_x, size_y, PLATFORM_THICK, cx, cy, -0.5 * PLATFORM_THICK)
    if outer_ground is not None:
        meshes.append(outer_ground)

    # ======================
    # second layer
    # ======================
    ring_outer = min(size_x, size_y) * 0.7
    ring_inner = p
    if H > 1e-8:
        top_h = max(0.0, 0.5 * (ring_outer - ring_inner))
        side_w = top_h
        top_y = cy + 0.5 * ring_inner + 0.5 * top_h
        bottom_y = cy - 0.5 * ring_inner - 0.5 * top_h
        left_x = cx - 0.5 * ring_inner - 0.5 * side_w
        right_x = cx + 0.5 * ring_inner + 0.5 * side_w

        top = make_box(ring_outer, top_h, H, cx, top_y, 0.5 * H)
        bottom = make_box(ring_outer, top_h, H, cx, bottom_y, 0.5 * H)
        left = make_box(side_w, ring_inner, H, left_x, cy, 0.5 * H)
        right = make_box(side_w, ring_inner, H, right_x, cy, 0.5 * H)

        for m in (top, bottom, left, right):
            if m is not None:
                meshes.append(m)

    # ======================
    # middle layer
    # ======================
    center = make_box(p, p, PLATFORM_THICK, cx, cy, -0.5 * PLATFORM_THICK)
    if center is not None:
        meshes.append(center)

    # ---- origin ----
    origin = np.array([cx, cy, 0.0], dtype=np.float32)
    return meshes, origin


def gap_terrain(
        difficulty: float, cfg: mesh_terrains_cfg.MeshGapTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a gap around the platform.

    The terrain has a ground with a platform in the middle. The platform is surrounded by a gap
    of width :obj:`gap_width` on all sides.

    .. image:: ../../_static/terrains/trimesh/gap_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        A tuple containing the tri-mesh of the terrain and the origin of the terrain (in m).
    """
    # resolve the terrain configuration
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    gap_depth = cfg.gap_deepth

    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    terrain_height = gap_depth
    terrain_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -terrain_height / 2)

    # Generate the outer ring
    inner_size = (cfg.platform_width + 2 * gap_width, cfg.platform_width + 2 * gap_width)
    meshes_list += make_border(cfg.size, inner_size, terrain_height, terrain_center)
    # Generate the inner box
    box_dim = (cfg.platform_width, cfg.platform_width, terrain_height)
    box = trimesh.creation.box(box_dim, trimesh.transformations.translation_matrix(terrain_center))
    meshes_list.append(box)
    # Generate the box on the bottom of the gap
    gap_center = (terrain_center[0], terrain_center[1], -gap_depth / 2 * 3)
    gap_box_dim = (cfg.platform_width + 2 * gap_width, cfg.platform_width + 2 * gap_width, gap_depth)
    gap_box = trimesh.creation.box(gap_box_dim, trimesh.transformations.translation_matrix(gap_center))
    meshes_list.append(gap_box)

    # specify the origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], 0.0])

    return meshes_list, origin


def sim2sim_slide_course(
    difficulty: float, cfg: "mesh_terrains_cfg.MeshSim2SimSlideCourseCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Fixed sim2sim slide course from dolanga1_mujoco_sim2sim/resources/slide.xml."""
    del difficulty  # fixed geometry; difficulty is unused

    size_x, size_y = cfg.size
    cy = 0.5 * size_y
    meshes: list[trimesh.Trimesh] = []

    meshes.append(
        _make_box(
            (0.5 * size_x, 0.5 * size_y, 0.5 * _GROUND_THICK),
            (0.5 * size_x, cy, -0.5 * _GROUND_THICK),
        )
    )

    # ramp -> platform -> upstairs -> top -> downstairs -> ramp2 (world x matches slide.xml)
    meshes.append(_make_box((2.0, 3.0, 0.05), (3.0, cy, 0.0), (0.0, -0.3, 0.0)))
    meshes.append(_make_box((0.8, 3.0, 0.05), (5.7, cy, 0.6)))

    up_stair_specs = [
        (0.3, 0.71, (0.15, 3.0, 0.10)),
        (0.6, 0.91, (0.15, 3.0, 0.10)),
        (0.9, 1.11, (0.15, 3.0, 0.10)),
        (1.2, 1.31, (0.15, 3.0, 0.10)),
        (1.5, 1.51, (0.15, 3.0, 0.10)),
    ]
    for local_x, local_z, half_size in up_stair_specs:
        meshes.append(_make_box(half_size, (6.2 + local_x, cy, local_z)))

    meshes.append(_make_box((1.1, 3.0, 0.05), (8.9, cy, 1.56)))

    down_stair_specs = [
        (0.3, 1.44),
        (0.6, 1.34),
        (0.9, 1.24),
        (1.2, 1.14),
        (1.5, 1.04),
        (1.8, 0.94),
        (2.1, 0.84),
        (2.4, 0.74),
    ]
    for local_x, local_z in down_stair_specs:
        meshes.append(_make_box((0.15, 3.0, 0.05), (9.8 + local_x, cy, local_z)))

    meshes.append(_make_box((2.0, 3.0, 0.05), (14.2, cy, 0.1), (0.0, 0.3, 0.0)))

    origin = np.array([1.5, cy, 0.0], dtype=np.float32)
    return meshes, origin
