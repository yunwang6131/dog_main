from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import onnxruntime as ort


@dataclass
class Sim2SimCfg:
    sim_duration: float = 120.0
    dt: float = 0.005
    decimation: int = 4
    history_len: int = 10
    warmup_seconds: float = 0.0
    init_base_height: float = 0.5

    mujoco_model_path: str = "path/to/scene.xml"
    onnx_path: str = "path/to/policy.onnx"

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
            [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5], dtype=np.float32
        )
    )
    action_scale: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.25], dtype=np.float32
        )
    )
    kp: np.ndarray = field(default_factory=lambda: np.array([100.0] * 12, dtype=np.float32))
    kd: np.ndarray = field(default_factory=lambda: np.array([1.5] * 12, dtype=np.float32))
    tau_limit: np.ndarray = field(default_factory=lambda: np.array([96.0, 156.0, 156.0] * 4, dtype=np.float32))
    cmd: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))


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
        self.sess = ort.InferenceSession(cfg.onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.input_shape = self.sess.get_inputs()[0].shape
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

        # Policy was trained with 12 leg joints and 10 history frames.
        if self.n_joints != 12:
            raise ValueError(f"Expected 12 joints for this policy, got {self.n_joints}: {self.cfg.joint_names}")

        self.qpos_idx, self.qvel_idx, self.joint_ids = self._build_joint_indices()
        self.actuator_idx = self._build_actuator_indices(self.joint_ids)
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)
        self.obs_history: dict[str, list[np.ndarray]] = {
            "base_ang_vel": [],
            "projected_gravity": [],
            "velocity_commands": [],
            "joint_pos": [],
            "joint_vel": [],
            "actions": [],
        }
        self._reset_to_training_init()
        print(
            "[INFO] sim2sim cfg: "
            f"onnx_input={self.input_shape}, "
            f"history_len={self.cfg.history_len}, "
            f"warmup_seconds={self.cfg.warmup_seconds}, "
            f"kp={self.kp[0]:.1f}, kd={self.kd[0]:.1f}, "
            f"init_base_height={self.cfg.init_base_height:.3f}"
        )

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

    def _build_actor_obs(self) -> np.ndarray:
        # Match training obs_groups order:
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
        expected_dim = 45 * self.cfg.history_len
        if obs.shape[0] != expected_dim:
            raise ValueError(f"Actor obs dim mismatch: {obs.shape[0]} != {expected_dim}")
        return obs

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
            obs = self._build_actor_obs()
            raw_action = self.sess.run(
                [self.output_name],
                {self.input_name: obs[None, :]}
            )[0][0].astype(np.float32)

            if raw_action.shape != (self.n_joints,):
                raise ValueError(
                    f"Policy output shape mismatch: expected ({self.n_joints},), got {raw_action.shape}"
                )

            if not np.all(np.isfinite(raw_action)):
                raise RuntimeError(f"Policy output NaN/Inf at step {step_id}: {raw_action}")

            action = np.clip(raw_action, -10.0, 10.0)

            # 注意：last_action 一定存 clip 后的 action
            self.last_action = action.copy()
        else:
            action = self.last_action

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
        if step_id % 100 == 0:
            print(
                f"[step {step_id}] "
                f"z={self.data.qpos[2]:.3f}, "
                f"gravity_xy={np.linalg.norm(projected_gravity_body[:2]):.3f}, "
                f"ang_vel={np.linalg.norm(base_ang_vel_body):.3f}, "
                f"action_max={np.max(np.abs(action)):.3f}, "
                f"dq_max={np.max(np.abs(dq)):.3f}, "
                f"tau_max={np.max(np.abs(tau)):.3f}, "
                f"tau_sat={np.mean(np.abs(tau_raw) >= self.tau_limit):.2f}"
            )
        mujoco.mj_step(self.model, self.data)