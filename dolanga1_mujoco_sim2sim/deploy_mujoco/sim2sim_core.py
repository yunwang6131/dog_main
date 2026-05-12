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
    warmup_seconds: float = 2.0

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
    q_default: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 0.8, -1.5], dtype=np.float32
        )
    )
    action_scale: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.125, 0.25, 0.25, 0.125, 0.25, 0.25, 0.125, 0.25, 0.25, 0.125, 0.25, 0.25], dtype=np.float32
        )
    )
    kp: np.ndarray = field(default_factory=lambda: np.array([40.0] * 12, dtype=np.float32))
    kd: np.ndarray = field(default_factory=lambda: np.array([1.0] * 12, dtype=np.float32))
    tau_limit: np.ndarray = field(default_factory=lambda: np.array([96.0, 156.0, 156.0] * 4, dtype=np.float32))
    cmd: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.0, 0.0], dtype=np.float32))


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


def build_actor_obs(
    base_ang_vel_body: np.ndarray,
    projected_gravity_body: np.ndarray,
    velocity_command: np.ndarray,
    joint_pos_rel: np.ndarray,
    joint_vel_rel: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    obs = np.concatenate(
        [base_ang_vel_body*0.25, projected_gravity_body, velocity_command, joint_pos_rel, joint_vel_rel*0.05, last_action], axis=0
    ).astype(np.float32)
    if obs.shape[0] != 45:
        raise ValueError(f"Actor obs dim mismatch: {obs.shape[0]} != 45")
    return obs


class Sim2SimRunner:
    def __init__(self, cfg: Sim2SimCfg):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.mujoco_model_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = cfg.dt
        self.sess = ort.InferenceSession(cfg.onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

        self.n_joints = len(self.cfg.joint_names)
        self.q_default = self._as_vector(self.cfg.q_default, "q_default")
        self.action_scale = self._as_vector(self.cfg.action_scale, "action_scale")
        self.kp = self._as_vector(self.cfg.kp, "kp")
        self.kd = self._as_vector(self.cfg.kd, "kd")
        self.tau_limit = self._as_vector(self.cfg.tau_limit, "tau_limit")
        self.cmd = np.asarray(self.cfg.cmd, dtype=np.float32)
        if self.cmd.shape != (3,):
            raise ValueError(f"cmd must have shape (3,), got {self.cmd.shape}")

        # Policy was trained with 12 leg joints -> actor obs dimension is fixed at 45.
        if self.n_joints != 12:
            raise ValueError(f"Expected 12 joints for this policy, got {self.n_joints}: {self.cfg.joint_names}")

        self.qpos_idx, self.qvel_idx, self.joint_ids = self._build_joint_indices()
        self.actuator_idx = self._build_actuator_indices(self.joint_ids)
        self.last_action = np.zeros(self.n_joints, dtype=np.float32)

    def _as_vector(self, value: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != (self.n_joints,):
            raise ValueError(f"{name} must have shape ({self.n_joints},), got {arr.shape}")
        return arr

    def _build_joint_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qpos_idx: list[int] = []
        qvel_idx: list[int] = []
        joint_ids: list[int] = []

        for joint_name in self.cfg.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Joint '{joint_name}' not found in model: {self.cfg.mujoco_model_path}")

            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError(
                    f"Joint '{joint_name}' must be hinge/slide (1 DoF), got type={joint_type}"
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

    def step(self, step_id: int) -> None:
        q = self.data.qpos[self.qpos_idx].astype(np.float32)
        dq = self.data.qvel[self.qvel_idx].astype(np.float32)

        base_quat_wxyz = self.data.qpos[3:7].astype(np.float32)
        base_ang_vel_world = self.data.qvel[3:6].astype(np.float32)
        rot_bw = quat_to_rotmat_wxyz(base_quat_wxyz).T
        base_ang_vel_body = rot_bw @ base_ang_vel_world
        projected_gravity_body = rot_bw @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        joint_pos_rel = q - self.q_default
        joint_vel_rel = dq

        warmup_steps = int(self.cfg.warmup_seconds / self.cfg.dt)
        if step_id < warmup_steps:
            action = np.zeros(self.n_joints, dtype=np.float32)
        elif step_id % self.cfg.decimation == 0:
            obs = build_actor_obs(
                base_ang_vel_body,
                projected_gravity_body,
                self.cmd,
                joint_pos_rel,
                joint_vel_rel,
                self.last_action,
            )
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