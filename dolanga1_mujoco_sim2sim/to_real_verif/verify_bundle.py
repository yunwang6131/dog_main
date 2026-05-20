#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml


DEFAULT_EXPORTED_DIR = Path(
    "/home/sen/wy/dog_main/robot_lab/logs/rsl_rl/"
    "dolanga1_rough_dreamwaq/2026-05-13_17-33-33_initial/exported"
)
DEFAULT_REPORT_PATH = Path(
    "/home/sen/wy/dog_main/dolanga1_mujoco_sim2sim/to_real_verif/reports/"
    "2026-05-13_17-33-33_initial_report.md"
)
JOINT_ORDER = [
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


@dataclass
class Check:
    status: str
    title: str
    detail: str


def activation_from_name(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], activation: str) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation_from_name(activation))
        for idx in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
            layers.append(activation_from_name(activation))
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class StandaloneDreamWaQCenet(nn.Module):
    def __init__(self, actor_sd: dict[str, torch.Tensor], activation: str = "elu") -> None:
        super().__init__()
        linear_indices = sorted(
            {
                int(parts[1])
                for key in actor_sd
                if key.startswith("encoder.") and key.endswith(".weight")
                for parts in [key.split(".")]
                if len(parts) >= 3 and parts[1].isdigit()
            }
        )
        if len(linear_indices) < 2:
            raise ValueError("Checkpoint does not contain a valid DreamWaQ encoder.")

        shapes = [tuple(actor_sd[f"encoder.{idx}.weight"].shape) for idx in linear_indices]
        history_dim = int(shapes[0][1])
        hidden_dims = [int(shape[0]) for shape in shapes[:-1]]
        encoder_out_dim = int(shapes[-1][0])
        self.history_dim = history_dim
        self.encoder = SimpleMLP(history_dim, encoder_out_dim, hidden_dims, activation)
        self.velocity_head = nn.Linear(encoder_out_dim, int(actor_sd["velocity_head.weight"].shape[0]))
        self.latent_mu_head = nn.Linear(encoder_out_dim, int(actor_sd["latent_mu_head.weight"].shape[0]))

        encoder_state: dict[str, torch.Tensor] = {}
        for linear_order, state_idx in enumerate(linear_indices):
            seq_idx = linear_order * 2
            encoder_state[f"layers.{seq_idx}.weight"] = actor_sd[f"encoder.{state_idx}.weight"]
            encoder_state[f"layers.{seq_idx}.bias"] = actor_sd[f"encoder.{state_idx}.bias"]
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.velocity_head.weight.data.copy_(actor_sd["velocity_head.weight"])
        self.velocity_head.bias.data.copy_(actor_sd["velocity_head.bias"])
        self.latent_mu_head.weight.data.copy_(actor_sd["latent_mu_head.weight"])
        self.latent_mu_head.bias.data.copy_(actor_sd["latent_mu_head.bias"])

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(history)
        return self.velocity_head(feat), self.latent_mu_head(feat)


def add_check(checks: list[Check], status: str, title: str, detail: str) -> None:
    checks.append(Check(status=status, title=title, detail=detail))


def load_bundle_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} does not contain a YAML mapping.")
    return data


def load_training_env_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.unsafe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} does not contain a YAML mapping.")
    return data


def maybe_sibling_training_env(exported_dir: Path) -> Path | None:
    candidate = exported_dir.parent / "params" / "env.yaml"
    return candidate if candidate.is_file() else None


def floats_close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def lists_close(a: list[float], b: list[float], tol: float = 1e-6) -> bool:
    if len(a) != len(b):
        return False
    return all(floats_close(x, y, tol=tol) for x, y in zip(a, b))


def derive_training_q_default(env_yaml: dict[str, Any]) -> list[float]:
    joint_pos = env_yaml["scene"]["robot"]["init_state"]["joint_pos"]
    hip = float(joint_pos[".*_hip_joint"])
    thigh = float(joint_pos[".*_thigh_joint"])
    calf = float(joint_pos[".*_calf_joint"])
    return [hip, thigh, calf] * 4


def derive_training_action_scale(env_yaml: dict[str, Any]) -> list[float]:
    scale = env_yaml["actions"]["joint_pos"]["scale"]
    hip = float(scale[".*_hip_joint"])
    other = float(scale["^(?!.*_hip_joint).*"])
    return [hip, other, other] * 4


def derive_training_joint_params(env_yaml: dict[str, Any], field_name: str) -> list[float]:
    actuators = env_yaml["scene"]["robot"]["actuators"]
    hip = float(actuators["hip"][field_name])
    thigh = float(actuators["thigh"][field_name])
    calf = float(actuators["calf"][field_name])
    return [hip, thigh, calf] * 4


def derive_training_command_ranges(env_yaml: dict[str, Any]) -> dict[str, list[float]]:
    ranges = env_yaml["commands"]["base_velocity"]["ranges"]
    return {
        "lin_vel_x": [float(ranges["lin_vel_x"][0]), float(ranges["lin_vel_x"][1])],
        "lin_vel_y": [float(ranges["lin_vel_y"][0]), float(ranges["lin_vel_y"][1])],
        "ang_vel_z": [float(ranges["ang_vel_z"][0]), float(ranges["ang_vel_z"][1])],
    }


def derive_training_history_len(env_yaml: dict[str, Any]) -> int:
    obs_group = env_yaml["observations"]["base_ang_vel_history"]
    group_history_len = obs_group.get("history_length")
    if group_history_len is not None:
        return int(group_history_len)
    term_history_len = obs_group["base_ang_vel"]["history_length"]
    return int(term_history_len)


def verify_bundle_yaml_contract(bundle_yaml: dict[str, Any], checks: list[Check]) -> None:
    history_shape = bundle_yaml["policy_interface"]["inputs"]["history"]["shape"]
    current_shape = bundle_yaml["policy_interface"]["inputs"]["current"]["shape"]
    output_shape = bundle_yaml["policy_interface"]["output"]["actions"]["shape"]
    history_len = int(bundle_yaml["observation_contract"]["history_len"])
    current_dim = int(bundle_yaml["observation_contract"]["current_dim"])
    history_dim = int(bundle_yaml["observation_contract"]["history_dim"])

    per_frame_total = sum(
        int(spec["dim"]) for spec in bundle_yaml["observation_contract"]["per_frame_layout"].values()
    )
    expected_period = float(bundle_yaml["control_contract"]["physics_dt_s"]) * int(bundle_yaml["control_contract"]["decimation"])

    if history_shape == [1, history_dim]:
        add_check(checks, "OK", "history shape", f"history input shape is [1, {history_dim}].")
    else:
        add_check(checks, "MISMATCH", "history shape", f"policy_interface={history_shape}, contract history_dim={history_dim}.")

    if current_shape == [1, current_dim]:
        add_check(checks, "OK", "current shape", f"current input shape is [1, {current_dim}].")
    else:
        add_check(checks, "MISMATCH", "current shape", f"policy_interface={current_shape}, contract current_dim={current_dim}.")

    if output_shape == [1, 12]:
        add_check(checks, "OK", "output shape", "action output shape is [1, 12].")
    else:
        add_check(checks, "MISMATCH", "output shape", f"policy_interface output shape is {output_shape}, expected [1, 12].")

    if per_frame_total == current_dim:
        add_check(checks, "OK", "per-frame dim", f"sum(per_frame_layout dims) = {per_frame_total}.")
    else:
        add_check(checks, "MISMATCH", "per-frame dim", f"sum(per_frame_layout dims) = {per_frame_total}, current_dim = {current_dim}.")

    if history_len * current_dim == history_dim:
        add_check(checks, "OK", "history flatten rule", f"{history_len} * {current_dim} = {history_dim}.")
    else:
        add_check(checks, "MISMATCH", "history flatten rule", f"{history_len} * {current_dim} != {history_dim}.")

    if floats_close(expected_period, float(bundle_yaml["control_contract"]["control_period_s"])):
        add_check(checks, "OK", "control period", f"dt * decimation = {expected_period:.5f}s.")
    else:
        add_check(
            checks,
            "MISMATCH",
            "control period",
            "dt * decimation does not match control_period_s "
            f"({expected_period:.5f}s vs {bundle_yaml['control_contract']['control_period_s']}).",
        )

    array_fields = ["q_default", "action_scale", "kp", "kd", "tau_limit"]
    for field_name in array_fields:
        values = [float(v) for v in bundle_yaml["control_contract"][field_name]]
        if len(values) == len(JOINT_ORDER):
            add_check(checks, "OK", field_name, f"{field_name} has {len(values)} entries.")
        else:
            add_check(checks, "MISMATCH", field_name, f"{field_name} has {len(values)} entries, expected {len(JOINT_ORDER)}.")


def verify_artifacts(exported_dir: Path, bundle_yaml: dict[str, Any], checks: list[Check]) -> dict[str, Any]:
    artifact_paths = {
        "policy_full.pt": exported_dir / "policy_full.pt",
        "sim2real_config.yaml": exported_dir / "sim2real_config.yaml",
        "policy.pt": exported_dir / "policy.pt",
        "policy.onnx": exported_dir / "policy.onnx",
        "cenet.pt": exported_dir / "cenet.pt",
    }

    for label, path in artifact_paths.items():
        if path.exists():
            add_check(checks, "OK", label, f"found {path}.")
        else:
            status = "ERR" if label in {"policy_full.pt", "sim2real_config.yaml"} else "WARN"
            add_check(checks, status, label, f"missing {path}.")

    history_dim = int(bundle_yaml["observation_contract"]["history_dim"])
    current_dim = int(bundle_yaml["observation_contract"]["current_dim"])
    action_dim = int(bundle_yaml["policy_interface"]["output"]["actions"]["shape"][1])

    model = torch.jit.load(str(artifact_paths["policy_full.pt"]), map_location="cpu").eval()
    hist_zero = torch.zeros(1, history_dim, dtype=torch.float32)
    cur_zero = torch.zeros(1, current_dim, dtype=torch.float32)
    hist_rand = torch.randn(1, history_dim, dtype=torch.float32)
    cur_rand = torch.randn(1, current_dim, dtype=torch.float32)
    with torch.inference_mode():
        out_zero = model(hist_zero, cur_zero)
        out_rand = model(hist_rand, cur_rand)

    if tuple(out_zero.shape) == (1, action_dim):
        add_check(checks, "OK", "policy_full smoke", f"zeros forward shape={tuple(out_zero.shape)} dtype={out_zero.dtype}.")
    else:
        add_check(checks, "ERR", "policy_full smoke", f"zeros forward returned shape {tuple(out_zero.shape)}.")
    if tuple(out_rand.shape) == (1, action_dim):
        add_check(checks, "OK", "policy_full random forward", f"random forward shape={tuple(out_rand.shape)}.")
    else:
        add_check(checks, "ERR", "policy_full random forward", f"random forward returned shape {tuple(out_rand.shape)}.")

    split_max_abs = None
    split_ok = None
    if artifact_paths["policy.pt"].is_file() and artifact_paths["cenet.pt"].is_file():
        ckpt = torch.load(str(artifact_paths["cenet.pt"]), map_location="cpu", weights_only=False)
        actor_sd = ckpt["cenet_state_dict"] if "cenet_state_dict" in ckpt else ckpt["actor_state_dict"]
        cenet = StandaloneDreamWaQCenet(actor_sd).eval()
        policy_tail = torch.jit.load(str(artifact_paths["policy.pt"]), map_location="cpu").eval()
        with torch.inference_mode():
            v_est, z = cenet(hist_rand)
            split_out = policy_tail(torch.cat([cur_rand, v_est, z], dim=-1))
            merged_out = model(hist_rand, cur_rand)
        split_max_abs = float((split_out - merged_out).abs().max().item())
        split_ok = bool(torch.allclose(split_out, merged_out, rtol=1e-4, atol=1e-5))
        if split_ok:
            add_check(checks, "OK", "split vs merged", f"max_abs_diff={split_max_abs:.3e}.")
        else:
            add_check(checks, "MISMATCH", "split vs merged", f"max_abs_diff={split_max_abs:.3e}.")
    else:
        add_check(checks, "WARN", "split vs merged", "policy.pt or cenet.pt is missing, skipped consistency check.")

    return {
        "history_dim": history_dim,
        "current_dim": current_dim,
        "action_dim": action_dim,
        "split_max_abs": split_max_abs,
        "split_ok": split_ok,
    }


def compare_bundle_with_training_env(
    bundle_yaml: dict[str, Any],
    env_yaml: dict[str, Any],
    checks: list[Check],
) -> list[str]:
    mismatches: list[str] = []
    ref_runtime = bundle_yaml["reference_runtime_params"]
    control_contract = bundle_yaml["control_contract"]

    env_dt = float(env_yaml["sim"]["dt"])
    env_decimation = int(env_yaml["decimation"])
    env_history_len = derive_training_history_len(env_yaml)
    env_init_z = float(env_yaml["scene"]["robot"]["init_state"]["pos"][2])
    env_q_default = derive_training_q_default(env_yaml)
    env_action_scale = derive_training_action_scale(env_yaml)
    env_kp = derive_training_joint_params(env_yaml, "stiffness")
    env_kd = derive_training_joint_params(env_yaml, "damping")
    env_tau = derive_training_joint_params(env_yaml, "effort_limit")
    env_ranges = derive_training_command_ranges(env_yaml)

    def compare_scalar(title: str, bundle_value: float, env_value: float, label: str) -> None:
        if floats_close(bundle_value, env_value):
            add_check(checks, "OK", title, f"bundle {bundle_value} matches training env {env_value}.")
        else:
            msg = f"bundle {bundle_value} != training env {env_value}"
            add_check(checks, "MISMATCH", title, msg + ".")
            mismatches.append(f"{label}: {msg}")

    def compare_vector(title: str, bundle_values: list[float], env_values: list[float], label: str) -> None:
        if lists_close(bundle_values, env_values):
            add_check(checks, "OK", title, "bundle values match training env snapshot.")
        else:
            msg = f"bundle {bundle_values} != training env {env_values}"
            add_check(checks, "MISMATCH", title, msg + ".")
            mismatches.append(f"{label}: {msg}")

    compare_scalar("training dt", float(ref_runtime["dt"]), env_dt, "dt")
    compare_scalar("training decimation", float(ref_runtime["decimation"]), float(env_decimation), "decimation")
    compare_scalar("training history_len", float(ref_runtime["history_len"]), float(env_history_len), "history_len")
    compare_scalar("training init_base_height", float(ref_runtime["init_base_height"]), env_init_z, "init_base_height")
    compare_vector("training q_default", [float(v) for v in control_contract["q_default"]], env_q_default, "q_default")
    compare_vector("training action_scale", [float(v) for v in control_contract["action_scale"]], env_action_scale, "action_scale")
    compare_vector("training kp", [float(v) for v in control_contract["kp"]], env_kp, "kp")
    compare_vector("training kd", [float(v) for v in control_contract["kd"]], env_kd, "kd")
    compare_vector("training tau_limit", [float(v) for v in control_contract["tau_limit"]], env_tau, "tau_limit")

    bundle_ranges = bundle_yaml["command_ranges_seen_in_training"]
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        bundle_values = [float(bundle_ranges[key][0]), float(bundle_ranges[key][1])]
        env_values = env_ranges[key]
        if lists_close(bundle_values, env_values):
            add_check(checks, "OK", f"training {key}", f"bundle {bundle_values} matches training env.")
        else:
            msg = f"bundle {bundle_values} != training env {env_values}"
            add_check(checks, "MISMATCH", f"training {key}", msg + ".")
            mismatches.append(f"{key}: {msg}")

    return mismatches


def render_report(
    checks: list[Check],
    exported_dir: Path,
    bundle_yaml_path: Path,
    training_env_yaml: Path | None,
    runtime_info: dict[str, Any],
    mismatches: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# DreamWaQ Bundle Verification")
    lines.append("")
    lines.append(f"- Exported dir: `{exported_dir}`")
    lines.append(f"- Bundle yaml: `{bundle_yaml_path}`")
    lines.append(f"- Training env yaml: `{training_env_yaml}`" if training_env_yaml is not None else "- Training env yaml: `not found`")
    lines.append(f"- Python: `{sys.executable}`")
    lines.append(f"- Torch: `{torch.__version__}`")
    lines.append("")
    lines.append("## Checks")
    for check in checks:
        lines.append(f"- [{check.status}] {check.title}: {check.detail}")
    lines.append("")
    lines.append("## Verdict")
    if any(check.status == "ERR" for check in checks):
        lines.append("- Artifact verification failed; fix the `ERR` items first.")
    else:
        lines.append("- `policy_full.pt` can be loaded and called successfully.")
        if runtime_info["split_ok"] is True:
            lines.append("- `policy_full.pt` matches `cenet.pt + policy.pt` numerically.")
        if mismatches:
            lines.append("- The bundle is internally runnable, but `sim2real_config.yaml` is not a faithful copy of the sibling training snapshot.")
            for item in mismatches:
                lines.append(f"- Mismatch: {item}")
        else:
            lines.append("- `sim2real_config.yaml` matches the sibling training snapshot on the checked fields.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify DreamWaQ policy_full bundle and sim2real_config contract.")
    parser.add_argument("--exported-dir", type=Path, default=DEFAULT_EXPORTED_DIR)
    parser.add_argument("--bundle-yaml", type=Path, default=None)
    parser.add_argument("--training-env-yaml", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--strict-mismatches", action="store_true")
    args = parser.parse_args()

    exported_dir = args.exported_dir.resolve()
    bundle_yaml_path = (args.bundle_yaml or (exported_dir / "sim2real_config.yaml")).resolve()
    training_env_yaml = args.training_env_yaml.resolve() if args.training_env_yaml is not None else maybe_sibling_training_env(exported_dir)
    report_path = args.report_path.resolve()

    checks: list[Check] = []

    if not exported_dir.is_dir():
        print(f"[ERR] exported dir not found: {exported_dir}", file=sys.stderr)
        return 1
    if not bundle_yaml_path.is_file():
        print(f"[ERR] bundle yaml not found: {bundle_yaml_path}", file=sys.stderr)
        return 1

    bundle_yaml = load_bundle_yaml(bundle_yaml_path)
    verify_bundle_yaml_contract(bundle_yaml, checks)
    runtime_info = verify_artifacts(exported_dir, bundle_yaml, checks)

    mismatches: list[str] = []
    if training_env_yaml is not None and training_env_yaml.is_file():
        env_yaml = load_training_env_yaml(training_env_yaml)
        mismatches = compare_bundle_with_training_env(bundle_yaml, env_yaml, checks)
    else:
        add_check(checks, "WARN", "training env snapshot", "sibling params/env.yaml not found, skipped training-side comparison.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = render_report(checks, exported_dir, bundle_yaml_path, training_env_yaml, runtime_info, mismatches)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Report written to: {report_path}")

    if any(check.status == "ERR" for check in checks):
        return 2
    if mismatches and args.strict_mismatches:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
