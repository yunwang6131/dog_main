from __future__ import annotations

from dataclasses import dataclass, field, fields

import mujoco
import numpy as np
import torch
import yaml


@dataclass
class Sim2SimCfg:
    sim_duration: float = 120.0
    dt: float = 0.005
    decimation: int = 4
    history_len: int = 10
    warmup_seconds: float = 0.0
    init_base_height: float = 0.50
    base_body_name: str = "base_link"
    foot_body_names: list[str] = field(
        default_factory=lambda: ["LF_foot_link", "RF_foot_link", "LH_foot_link", "RH_foot_link"]
    )
    body_name_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "base_link": "base",
            "LF_foot_link": "LF_FOOT",
            "RF_foot_link": "RF_FOOT",
            "LH_foot_link": "LH_FOOT",
            "RH_foot_link": "RH_FOOT",
        }
    )

    mujoco_model_path: str = "path/to/scene.xml"
    policy_path: str = "path/to/policy_full.pt"
    debug_print: bool = True

    joint_names: list[str] = field(
        default_factory=lambda: [
            "LF_hip_joint",
            "LF_thigh_joint",
            "LF_calf_joint",
            "RF_hip_joint",
            "RF_thigh_joint",
            "RF_calf_joint",
            "LH_hip_joint",
            "LH_thigh_joint",
            "LH_calf_joint",
            "RH_hip_joint",
            "RH_thigh_joint",
            "RH_calf_joint",
        ]
    )
    joint_name_aliases: dict[str, str] = field(
        default_factory=lambda: {
            "LF_hip_joint": "LF_HAA",
            "LF_thigh_joint": "LF_HFE",
            "LF_calf_joint": "LF_KFE",
            "RF_hip_joint": "RF_HAA",
            "RF_thigh_joint": "RF_HFE",
            "RF_calf_joint": "RF_KFE",
            "LH_hip_joint": "LH_HAA",
            "LH_thigh_joint": "LH_HFE",
            "LH_calf_joint": "LH_KFE",
            "RH_hip_joint": "RH_HAA",
            "RH_thigh_joint": "RH_HFE",
            "RH_calf_joint": "RH_KFE",
        }
    )
    q_default: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0, 0.70, -1.50, 0.0, 0.70, -1.50, 0.0, 0.70, -1.50, 0.0, 0.70, -1.50],
            dtype=np.float32,
        )
    )
    action_scale: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.4, 0.25, 0.25, 0.4, 0.25, 0.25, 0.4, 0.25, 0.25, 0.4, 0.25, 0.25], dtype=np.float32
        )
    )
    kp: np.ndarray = field(default_factory=lambda: np.array([80.0] * 12, dtype=np.float32))
    kd: np.ndarray = field(default_factory=lambda: np.array([2.0] * 12, dtype=np.float32))
    tau_limit: np.ndarray = field(default_factory=lambda: np.array([96.0, 156.0, 156.0] * 4, dtype=np.float32))
    action_clip: float = 10.0
    cmd: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))


def load_sim2sim_cfg(path: str) -> Sim2SimCfg:
    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    valid_fields = {item.name for item in fields(Sim2SimCfg)}
    unknown_fields = sorted(set(raw_cfg) - valid_fields)
    if unknown_fields:
        raise ValueError(f"Unknown sim2sim config fields in {path}: {unknown_fields}")

    return Sim2SimCfg(**raw_cfg)


def resolve_sim2sim_cfg(
    config_path: str | None,
    load_model: str | None,
    policy: str | None,
    sim_duration: float | None,
    cmd_x: float | None,
    cmd_y: float | None,
    cmd_yaw: float | None,
) -> Sim2SimCfg:
    if config_path:
        cfg = load_sim2sim_cfg(config_path)
        if load_model:
            cfg.mujoco_model_path = load_model
        if policy:
            cfg.policy_path = policy
        if sim_duration is not None:
            cfg.sim_duration = sim_duration
    else:
        if not load_model or not policy:
            raise ValueError("Pass either --config or both --load_model and --policy.")
        cfg = Sim2SimCfg(
            mujoco_model_path=load_model,
            policy_path=policy,
            sim_duration=120.0 if sim_duration is None else sim_duration,
        )

    if cmd_x is not None or cmd_y is not None or cmd_yaw is not None:
        cfg.cmd = np.array(
            [
                cfg.cmd[0] if cmd_x is None else cmd_x,
                cfg.cmd[1] if cmd_y is None else cmd_y,
                cfg.cmd[2] if cmd_yaw is None else cmd_yaw,
            ],
            dtype=np.float32,
        )

    return cfg


def quat_to_rotmat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def build_actor_obs_frame(
    base_ang_vel_body: np.ndarray,
    projected_gravity_body: np.ndarray,
    velocity_command: np.ndarray,
    joint_pos_rel: np.ndarray,
    joint_vel_rel: np.ndarray,
    last_action: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "base_ang_vel": (base_ang_vel_body * 0.25).astype(np.float32),
        "projected_gravity": projected_gravity_body.astype(np.float32),
        "velocity_commands": velocity_command.astype(np.float32),
        "joint_pos": joint_pos_rel.astype(np.float32),
        "joint_vel": (joint_vel_rel * 0.05).astype(np.float32),
        "actions": last_action.astype(np.float32),
    }


class Sim2SimRunner:
    def __init__(self, cfg: Sim2SimCfg):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.mujoco_model_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = cfg.dt
        if not cfg.policy_path.endswith(".pt"):
            raise ValueError(
                "Dolanga1 sim2sim expects merged TorchScript policy_full.pt. "
                "Export with export_barrier_dreamwaq_merged_jit_from_runner and pass that file as --policy."
            )
        self._torch_policy = torch.jit.load(cfg.policy_path, map_location="cpu").eval()
        self.input_shape = ["history", "current"]
        self.gyro_sensor_slice = self._find_sensor_slice("imu_gyro", expected_dim=3)

        self.n_joints = len(self.cfg.joint_names)
        self.q_default = self._as_vector(self.cfg.q_default, "q_default")
        self.action_scale = self._as_vector(self.cfg.action_scale, "action_scale")
        self.kp = self._as_vector(self.cfg.kp, "kp")
        self.kd = self._as_vector(self.cfg.kd, "kd")
        self.tau_limit = self._as_vector(self.cfg.tau_limit, "tau_limit")
        self.cmd = np.asarray(self.cfg.cmd, dtype=np.float32)
        if self.cmd.shape != (3,):
            raise ValueError(f"cmd must have shape (3,), got {self.cmd.shape}")
        if self.cfg.history_len <= 0:
            raise ValueError(f"history_len must be positive, got {self.cfg.history_len}")
        if self.cfg.action_clip <= 0.0:
            raise ValueError(f"action_clip must be positive, got {self.cfg.action_clip}")

        self.qpos_idx, self.qvel_idx, self.joint_ids = self._build_joint_indices()
        self.actuator_idx = self._build_actuator_indices(self.joint_ids)
        self.foot_body_ids = self._build_body_indices(self.cfg.foot_body_names)
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)
        self.obs_history: dict[str, list[np.ndarray]] = {
            "base_ang_vel": [],
            "projected_gravity": [],
            "velocity_commands": [],
            "joint_pos": [],
            "joint_vel": [],
            "actions": [],
        }
        self._proprio_frame_dim = 9 + 3 * self.n_joints
        self._history_obs_dim = self._proprio_frame_dim * self.cfg.history_len
        self._current_obs_dim = self._proprio_frame_dim
        self._actor_obs_dim = self._history_obs_dim + self._current_obs_dim
        self._reset_to_training_init()
        print(
            "[INFO] sim2sim cfg: "
            f"policy_input={self.input_shape}, "
            f"actor_obs_dim={self._actor_obs_dim}, "
            f"history_len={self.cfg.history_len}, "
            f"warmup_seconds={self.cfg.warmup_seconds}, "
            f"kp={self.kp[0]:.1f}, kd={self.kd[0]:.1f}, "
            f"init_base_height={self.cfg.init_base_height:.3f}"
        )
        if self.cfg.debug_print:
            joint_map = ", ".join(
                f"{joint}:act{actuator}" for joint, actuator in zip(self.cfg.joint_names, self.actuator_idx)
            )
            print(f"[INFO] joint actuator map: {joint_map}")

    def _as_vector(self, value: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (self.n_joints,):
            raise ValueError(f"{name} must have shape ({self.n_joints},), got {arr.shape}")
        return arr

    def _resolve_joint_id(self, joint_name: str) -> tuple[int, str]:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            return joint_id, joint_name

        alias = self.cfg.joint_name_aliases.get(joint_name)
        if alias is not None:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, alias)
            if joint_id >= 0:
                return joint_id, alias

        available = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        raise ValueError(
            f"Joint '{joint_name}' not found in model: {self.cfg.mujoco_model_path}. "
            f"Available joints: {available}"
        )

    def _build_joint_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qpos_idx: list[int] = []
        qvel_idx: list[int] = []
        joint_ids: list[int] = []

        for joint_name in self.cfg.joint_names:
            joint_id, model_joint_name = self._resolve_joint_id(joint_name)

            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError(
                    f"Joint '{model_joint_name}' must be hinge/slide (1 DoF), got type={joint_type}"
                )

            qpos_idx.append(int(self.model.jnt_qposadr[joint_id]))
            qvel_idx.append(int(self.model.jnt_dofadr[joint_id]))
            joint_ids.append(joint_id)

        return (
            np.asarray(qpos_idx, dtype=np.int32),
            np.asarray(qvel_idx, dtype=np.int32),
            np.asarray(joint_ids, dtype=np.int32),
        )

    def _build_actuator_indices(self, joint_ids: np.ndarray) -> np.ndarray:
        actuator_idx: list[int] = []
        for joint_name, joint_id in zip(self.cfg.joint_names, joint_ids):
            candidates = np.where(self.model.actuator_trnid[:, 0] == joint_id)[0]
            if candidates.size == 0:
                raise ValueError(f"No actuator found for joint '{joint_name}'")
            if candidates.size > 1:
                raise ValueError(f"Multiple actuators found for joint '{joint_name}': {candidates.tolist()}")
            actuator_idx.append(int(candidates[0]))
        return np.asarray(actuator_idx, dtype=np.int32)

    def _resolve_body_id(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            return int(body_id)

        alias = self.cfg.body_name_aliases.get(body_name)
        if alias is not None:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, alias)
            if body_id >= 0:
                return int(body_id)

        available = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)]
        raise ValueError(
            f"Body '{body_name}' not found in model: {self.cfg.mujoco_model_path}. "
            f"Available bodies: {available}"
        )

    def _build_body_indices(self, body_names: list[str]) -> np.ndarray:
        return np.asarray([self._resolve_body_id(body_name) for body_name in body_names], dtype=np.int32)

    def _find_sensor_slice(self, sensor_name: str, expected_dim: int) -> slice | None:
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id < 0:
            return None

        sensor_dim = int(self.model.sensor_dim[sensor_id])
        if sensor_dim != expected_dim:
            raise ValueError(f"Sensor '{sensor_name}' must have dim={expected_dim}, got {sensor_dim}")

        sensor_adr = int(self.model.sensor_adr[sensor_id])
        return slice(sensor_adr, sensor_adr + sensor_dim)

    def _read_base_ang_vel_body(self) -> np.ndarray:
        if self.gyro_sensor_slice is not None:
            return self.data.sensordata[self.gyro_sensor_slice].astype(np.float32)

        # MuJoCo free-joint angular velocity is already in the body frame.
        return self.data.qvel[3:6].astype(np.float32)

    def _reset_to_training_init(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:3] = np.array([0.0, 0.0, self.cfg.init_base_height], dtype=np.float32)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.data.qpos[self.qpos_idx] = self.q_default
        mujoco.mj_forward(self.model, self.data)

    def _update_obs_history(self, frame: dict[str, np.ndarray]) -> None:
        for name, value in frame.items():
            history = self.obs_history[name]
            value = value.astype(np.float32)
            while len(history) < self.cfg.history_len:
                history.append(value.copy())
            history.append(value.copy())
            del history[: -self.cfg.history_len]

    def _build_history_obs(self) -> np.ndarray:
        # Match training history_groups order:
        # base_ang_vel_history, projected_gravity_history, velocity_commands_history,
        # joint_pos_history, joint_vel_history, actions_history.
        obs = np.concatenate(
            [
                np.concatenate(self.obs_history["base_ang_vel"], axis=0),
                np.concatenate(self.obs_history["projected_gravity"], axis=0),
                np.concatenate(self.obs_history["velocity_commands"], axis=0),
                np.concatenate(self.obs_history["joint_pos"], axis=0),
                np.concatenate(self.obs_history["joint_vel"], axis=0),
                np.concatenate(self.obs_history["actions"], axis=0),
            ],
            axis=0,
        ).astype(np.float32)
        if obs.shape[0] != self._history_obs_dim:
            raise ValueError(f"History obs dim mismatch: {obs.shape[0]} != {self._history_obs_dim}")
        return obs

    def _build_current_obs(self) -> np.ndarray:
        obs = np.concatenate(
            [
                self.obs_history["base_ang_vel"][-1],
                self.obs_history["projected_gravity"][-1],
                self.obs_history["velocity_commands"][-1],
                self.obs_history["joint_pos"][-1],
                self.obs_history["joint_vel"][-1],
                self.obs_history["actions"][-1],
            ],
            axis=0,
        ).astype(np.float32)
        if obs.shape[0] != self._current_obs_dim:
            raise ValueError(f"Current obs dim mismatch: {obs.shape[0]} != {self._current_obs_dim}")
        return obs

    def _run_policy(self, history_obs: np.ndarray, current_obs: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            history_tensor = torch.from_numpy(history_obs[None, :]).float()
            current_tensor = torch.from_numpy(current_obs[None, :]).float()
            out = self._torch_policy(history_tensor, current_tensor)
        return out.cpu().numpy()[0].astype(np.float32)

    def step(self, step_id: int) -> None:
        q = self.data.qpos[self.qpos_idx].astype(np.float32)
        dq = self.data.qvel[self.qvel_idx].astype(np.float32)

        base_quat_wxyz = self.data.qpos[3:7].astype(np.float32)
        rot_bw = quat_to_rotmat_wxyz(base_quat_wxyz).T
        base_ang_vel_body = self._read_base_ang_vel_body()
        projected_gravity_body = rot_bw @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        joint_pos_rel = q - self.q_default
        joint_vel_rel = dq

        warmup_steps = int(self.cfg.warmup_seconds / self.cfg.dt)
        if step_id % self.cfg.decimation == 0:
            frame = build_actor_obs_frame(
                base_ang_vel_body,
                projected_gravity_body,
                self.cmd,
                joint_pos_rel,
                joint_vel_rel,
                self.last_action,
            )
            self._update_obs_history(frame)

        if step_id < warmup_steps:
            action = np.zeros(self.n_joints, dtype=np.float32)
        elif step_id % self.cfg.decimation == 0:
            history_obs = self._build_history_obs()
            current_obs = self._build_current_obs()
            raw_action = self._run_policy(history_obs, current_obs)

            if raw_action.shape != (self.n_joints,):
                raise ValueError(
                    f"Policy output shape mismatch: expected ({self.n_joints},), got {raw_action.shape}"
                )

            if not np.all(np.isfinite(raw_action)):
                raise RuntimeError(f"Policy output NaN/Inf at step {step_id}: {raw_action}")

            action = np.clip(raw_action, -self.cfg.action_clip, self.cfg.action_clip)

            self.last_action = action.copy()
        else:
            action = self.last_action
            raw_action = action

        q_target = self.q_default + action * self.action_scale

        tau_raw = self.kp * (q_target - q) + self.kd * (0.0 - dq)

        if not np.all(np.isfinite(tau_raw)):
            raise RuntimeError(
                f"tau_raw NaN/Inf at step {step_id}: "
                f"action={action}, q_target={q_target}, q={q}, dq={dq}"
            )

        tau = np.clip(tau_raw, -self.tau_limit, self.tau_limit)
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.actuator_idx] = tau
        if self.cfg.debug_print and step_id % 100 == 0:
            top_action_ids = np.argsort(-np.abs(action))[: min(5, self.n_joints)]
            top_actions = ", ".join(
                f"{self.cfg.joint_names[i]}={action[i]:.2f}" for i in top_action_ids
            )
            foot_z = ", ".join(
                f"{self.cfg.foot_body_names[i]}={self.data.xpos[body_id, 2]:.3f}"
                for i, body_id in enumerate(self.foot_body_ids)
            )
            print(
                f"[step {step_id}] "
                f"z={self.data.qpos[2]:.3f}, "
                f"ncon={self.data.ncon}, "
                f"foot_z=[{foot_z}], "
                f"gravity_xy={np.linalg.norm(projected_gravity_body[:2]):.3f}, "
                f"ang_vel={np.linalg.norm(base_ang_vel_body):.3f}, "
                f"raw_action_max={np.max(np.abs(raw_action)):.3f}, "
                f"action_max={np.max(np.abs(action)):.3f}, "
                f"top_actions=[{top_actions}], "
                f"q_err_max={np.max(np.abs(q - self.q_default)):.3f}, "
                f"dq_max={np.max(np.abs(dq)):.3f}, "
                f"tau_max={np.max(np.abs(tau)):.3f}, "
                f"tau_sat={np.mean(np.abs(tau_raw) >= self.tau_limit):.2f}"
            )
        mujoco.mj_step(self.model, self.data)
