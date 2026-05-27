# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.sensors import Camera, ContactSensor, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def phase_with_command(
    env: ManagerBasedRLEnv,
    cycle_time: float,
    command_name: str = "base_velocity",
    stand_threshold: float = 0.2,
) -> torch.Tensor:
    """Return gait phase features and zero them in stand-mode, matching the paper description."""
    phase_tensor = phase(env, cycle_time)
    command = env.command_manager.get_command(command_name)
    moving_mask = (torch.linalg.norm(command, dim=1, keepdim=True) >= stand_threshold).float()
    return phase_tensor * moving_mask


def command_stand_mode(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    threshold: float = 0.2,
) -> torch.Tensor:
    """Binary stand-mode indicator used by the actor/critic observations."""
    command = env.command_manager.get_command(command_name)
    return (torch.linalg.norm(command, dim=1, keepdim=True) < threshold).float()


def foot_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Binary contact state for the selected feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return (forces > contact_threshold).float()


def feet_positions_body(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Feet positions in the body frame, flattened as [x1, y1, z1, ..., xn, yn, zn]."""
    asset: Articulation = env.scene[asset_cfg.name]
    translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[:, None, :]
    feet_pos_body = torch.zeros_like(translated)
    for i in range(len(asset_cfg.body_ids)):
        feet_pos_body[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), translated[:, i, :]
        )
    return feet_pos_body.view(env.num_envs, -1)


class RollImageHistory:
    """Store image history as a channel stack [num_envs, history_len, height, width]."""

    def __init__(self, num_env: int, history_len: int, height: int, width: int, device: str = "cuda"):
        self.num_env = num_env
        self.history_len = history_len
        self.height = height
        self.width = width
        self.buffer = torch.zeros((num_env, history_len, height, width), device=device)

    def add_frames(self, new_frames: torch.Tensor) -> None:
        """Append a single-channel frame batch to the rolling image buffer."""
        frames = new_frames
        if frames.dim() == 4:
            if frames.shape[-1] == 1:
                frames = frames.squeeze(-1)
            elif frames.shape[1] == 1:
                frames = frames.squeeze(1)
            else:
                raise RuntimeError(f"Expected single-channel frames, got shape {tuple(frames.shape)}")
        elif frames.dim() != 3:
            raise RuntimeError(f"Unsupported frame tensor shape {tuple(frames.shape)}")

        if frames.shape != (self.num_env, self.height, self.width):
            raise RuntimeError(
                f"Expected frame shape {(self.num_env, self.height, self.width)}, got {tuple(frames.shape)}"
            )

        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        self.buffer[:, -1] = frames

    def get_history(self) -> torch.Tensor:
        return self.buffer

    def reset_env(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            self.buffer.zero_()
        else:
            self.buffer[env_ids].zero_()


class image_with_history(ManagerTermBase):
    """Return depth image history as [num_envs, history_len, height, width]."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self.sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[self.sensor_cfg.name]
        self.history_length = int(cfg.params["history_len"])
        self.image_height = int(cfg.params["img_shape"][0])
        self.image_width = int(cfg.params["img_shape"][1])
        self.data_type = cfg.params.get("data_type", "distance_to_image_plane")
        if self.data_type != "distance_to_image_plane":
            raise RuntimeError(f"Unsupported camera data type: {self.data_type}")

        self.history_buffer = RollImageHistory(
            num_env=env.num_envs,
            history_len=self.history_length,
            height=self.image_height,
            width=self.image_width,
            device=str(env.device) if hasattr(env, "device") else "cuda",
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        data_type: str = "distance_to_image_plane",
        history_len: int = 1,
        img_shape: tuple[int, int] = (60, 60),
        clip_horizontal_from: int = 26,
        clip_vertical_from: int = 0,
        flip: bool = False,
    ) -> torch.Tensor:
        if history_len != self.history_length:
            raise RuntimeError(f"history_len={history_len} does not match configured {self.history_length}")
        if img_shape != (self.image_height, self.image_width):
            raise RuntimeError(f"img_shape={img_shape} does not match configured {(self.image_height, self.image_width)}")
        if data_type != self.data_type:
            raise RuntimeError(f"data_type={data_type} does not match configured {self.data_type}")

        if self.sensor._is_outdated[0]:
            images = self.sensor.data.output[data_type]
            if flip:
                images = torch.flip(images, dims=[2])
            images = images[
                :,
                clip_vertical_from : clip_vertical_from + img_shape[0],
                clip_horizontal_from : clip_horizontal_from + img_shape[1],
                :,
            ]
            self.history_buffer.add_frames(images)
        return self.history_buffer.get_history()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.history_buffer.reset_env(env_ids)
