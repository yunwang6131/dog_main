# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
import json
import shutil
from typing import Dict, List, Tuple, Optional, Any

import onnx
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from isaaclab.managers import ObservationTermCfg as ObsTerm

from robot_lab.third_party.rsl_rl_est.modules.actor_critic_est import ActorCriticEst
from robot_lab.third_party.rsl_rl_est.modules.encoders import MemoryEncoder


# =============================================================================
# Public API
# =============================================================================

def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename: str = "policy.pt"):
    """Export policy into a Torch JIT file."""
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object,
    path: str,
    env_cfg: object | None = None,
    normalizer: object | None = None,
    filename: str = "policy.onnx",
    verbose: bool = True,
    obs_example: TensorDict | None = None,
):
    """Export policy into ONNX.

    - 非 ActorCriticEst：用旧式 _OnnxPolicyExporter (输入是 flat tensor)。
    - ActorCriticEst：
        * 非 RNN：导出 TensorDictActorDeployWrapper (inputs=obs_* 2D/4D)
        * RNN：导出 TensorDictActorRNNDeployWrapper (inputs=obs_* + h_in + mem_*_in)
      同时写 policy_config.json（需要 env_cfg + obs_example）。
    """
    # -------------------------
    # Old-style policies
    # -------------------------
    if not isinstance(policy, ActorCriticEst):
        os.makedirs(path, exist_ok=True)
        exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
        exporter.export(path, filename)
        return

    # -------------------------
    # ActorCriticEst requires env_cfg + obs_example
    # -------------------------
    if obs_example is None:
        raise ValueError(
            "export_policy_as_onnx: policy is ActorCriticEst, but obs_example is None.\n"
            "请传入一个示例 TensorDict（跟 act_inference 用的一致）用于推断输入 shape。"
        )
    if env_cfg is None:
        raise ValueError(
            "export_policy_as_onnx: policy is ActorCriticEst, but env_cfg is None.\n"
            "需要 env_cfg 来导出 policy_config.json（obs_scales、obs_dims、commands_limits 等）。"
        )

    mem_info = _find_memory_encoder(policy)

    if mem_info is None:
        export_tensordict_policy_as_onnx(
            policy=policy,
            obs_example=obs_example,
            env_cfg=env_cfg,
            path=path,
            filename=filename,
            verbose=verbose,
        )
    else:
        export_tensordict_policy_rnn_as_onnx(
            policy=policy,
            obs_example=obs_example,
            env_cfg=env_cfg,
            mem_info=mem_info,
            path=path,
            filename=filename,
            verbose=verbose,
        )


# =============================================================================
# Helpers
# =============================================================================

def _infer_obs_shapes(obs_example: TensorDict) -> Dict[str, List[int]]:
    """Return batch-excluded shapes for each obs key.
    - 2D: [B,D] -> [D]
    - 4D: [B,C,H,W] -> [C,H,W]
    """
    obs_shapes: Dict[str, List[int]] = {}
    for k in obs_example.keys():
        x = obs_example[k]
        if x.dim() == 2:
            obs_shapes[k] = [int(x.shape[1])]
        elif x.dim() == 4:
            obs_shapes[k] = [int(x.shape[1]), int(x.shape[2]), int(x.shape[3])]
        else:
            raise ValueError(f"obs_example['{k}'] must be 2D or 4D, got {tuple(x.shape)}")
    return obs_shapes


def _make_dummy_from_shape(shape: List[int], batch: int = 1) -> torch.Tensor:
    return torch.zeros([batch] + list(shape), dtype=torch.float32)


def _find_memory_encoder(
    policy: ActorCriticEst,
) -> Optional[Tuple[str, MemoryEncoder, str, nn.Module]]:
    """If Estimator contains MemoryEncoder, return (encoder_name, encoder, latent_name, rnn_module)."""
    est = getattr(policy, "estimator", None)
    if est is None or not isinstance(est, nn.Module):
        return None
    if not hasattr(est, "encoders"):
        return None

    for name, enc in est.encoders.items():
        if isinstance(enc, MemoryEncoder):
            if not hasattr(enc, "encoder") or not hasattr(enc.encoder, "rnn"):
                continue
            latent_name = getattr(enc, "latent_name", None)
            if not isinstance(latent_name, str):
                continue
            return name, enc, latent_name, enc.encoder.rnn
    return None


# =============================================================================
# Non-RNN Wrapper
# =============================================================================

class TensorDictActorDeployWrapper(nn.Module):
    """Deploy wrapper for ActorCriticEst (non-RNN).
    Inputs:  obs_* tensors (each can be 2D or 4D, shape determined by obs_example)
    Output:  actions [B, num_actions]
    """

    def __init__(self, ac: ActorCriticEst, obs_shapes: Dict[str, List[int]]):
        super().__init__()

        # copy train-time modules
        self.actor = copy.deepcopy(ac.actor)
        self.normalizer = copy.deepcopy(ac.normalizer)
        self.state_dependent_std = bool(ac.state_dependent_std)

        self.has_estimator = isinstance(getattr(ac, "estimator", None), nn.Module)
        self.estimator = copy.deepcopy(ac.estimator) if self.has_estimator else None

        self.policy_groups: List[str] = list(ac.obs_groups["policy"])
        self.normalize_groups: List[str] = list(ac.obs_groups["normalize"])

        self.detach_flags_policy: Dict[str, bool] = {}
        if hasattr(ac, "_detach_flags") and "policy" in ac._detach_flags:
            self.detach_flags_policy = dict(ac._detach_flags["policy"])

        # encoder latent names (exclude them from external inputs)
        encoder_latent_names: set[str] = set()
        if hasattr(self.estimator, "encoders"):
            for enc in self.estimator.encoders.values():
                if hasattr(enc, "latent_name") and isinstance(enc.latent_name, str):
                    encoder_latent_names.add(enc.latent_name)
                if hasattr(enc, "obs_groups") and isinstance(enc.obs_groups, list):
                    for group in enc.obs_groups:
                        if group not in self.encoder_obs_groups:
                            self.encoder_obs_groups.append(group)

        input_groups = set(self.normalize_groups)
        for name in self.policy_groups:
            if name.endswith("_norm"):
                continue
            if name in encoder_latent_names:
                continue
            input_groups.add(name)
        for name in self.encoder_obs_groups:
            if name.endswith("_norm"):
                continue
            if name in encoder_latent_names:
                continue
            input_groups.add(name)

        self.input_groups: List[str] = sorted(list(input_groups))

        # record shapes
        self.obs_shapes: Dict[str, List[int]] = {}
        for k in self.input_groups:
            if k not in obs_shapes:
                raise KeyError(f"obs_example 缺少 key '{k}'")
            self.obs_shapes[k] = list(map(int, obs_shapes[k]))

    def forward(self, *obs_tensors: torch.Tensor) -> torch.Tensor:
        assert len(obs_tensors) == len(self.input_groups), (
            f"Expected {len(self.input_groups)} tensors, got {len(obs_tensors)}."
        )
        batch_size = int(obs_tensors[0].size(0))

        obs_dict: Dict[str, torch.Tensor] = {}
        for name, tensor in zip(self.input_groups, obs_tensors):
            obs_dict[name] = tensor  # tensor can be 2D or 4D

        obs = TensorDict(obs_dict, batch_size=[batch_size])

        # normalize (only meaningful for 2D groups; your ActorCriticEst already enforces that)
        for name in self.normalize_groups:
            if name in obs and name in self.normalizer:
                obs[name + "_norm"] = self.normalizer[name](obs[name])

        if self.estimator is not None:
            obs = self.estimator.encode_inference(obs)

        obs_list: List[torch.Tensor] = []
        for name in self.policy_groups:
            if name not in obs:
                raise RuntimeError(f"Policy obs_group '{name}' missing after encode_inference/normalize.")
            x = obs[name]
            if self.detach_flags_policy.get(name, False):
                x = x.detach()
            obs_list.append(x)

        actor_obs = torch.cat(obs_list, dim=-1)

        if self.state_dependent_std:
            return self.actor(actor_obs)[..., 0, :]
        else:
            return self.actor(actor_obs)


# =============================================================================
# RNN Wrapper
# =============================================================================

class TensorDictActorRNNDeployWrapper(nn.Module):
    """Deploy wrapper for ActorCriticEst with MemoryEncoder (GRU) + memory_buffers.

    Inputs:  obs_* tensors + h_in + mem_*_in...
    Outputs: actions + h_out + mem_*_out...
    """

    def __init__(
        self,
        ac: ActorCriticEst,
        obs_shapes: Dict[str, List[int]],
        latent_name: str,
        rnn: nn.Module,
    ):
        super().__init__()

        # copy actor & normalizer; estimator is kept (not deepcopy) to preserve memory_buffers structure
        self.actor = copy.deepcopy(ac.actor)
        self.normalizer = copy.deepcopy(ac.normalizer)
        self.state_dependent_std = bool(ac.state_dependent_std)

        assert isinstance(ac.estimator, nn.Module), "RNN export requires policy.estimator is nn.Module."
        self.estimator = ac.estimator

        self.policy_groups: List[str] = list(ac.obs_groups["policy"])
        self.normalize_groups: List[str] = list(ac.obs_groups["normalize"])
        self.encoder_obs_groups: List[str] = []

        self.detach_flags_policy: Dict[str, bool] = {}
        if hasattr(ac, "_detach_flags") and "policy" in ac._detach_flags:
            self.detach_flags_policy = dict(ac._detach_flags["policy"])

        self.latent_name = latent_name
        self.rnn_type = type(rnn).__name__.lower()
        self.num_layers = int(rnn.num_layers)
        self.hidden_size = int(rnn.hidden_size)

        if self.rnn_type != "gru":
            raise NotImplementedError(f"Only GRU is implemented now, got rnn_type='{self.rnn_type}'.")

        # exclude latent outputs from external inputs
        encoder_latent_names: set[str] = set()
        if hasattr(self.estimator, "encoders"):
            for enc in self.estimator.encoders.values():
                if hasattr(enc, "latent_name") and isinstance(enc.latent_name, str):
                    encoder_latent_names.add(enc.latent_name)
                if hasattr(enc, "obs_groups") and isinstance(enc.obs_groups, list):
                    for group in enc.obs_groups:
                        if group not in self.encoder_obs_groups:
                            self.encoder_obs_groups.append(group)

        input_groups = set(self.normalize_groups)
        for name in self.policy_groups:
            if name.endswith("_norm"):
                continue
            if name in encoder_latent_names:
                continue
            input_groups.add(name)
        for name in self.encoder_obs_groups:
            if name.endswith("_norm"):
                continue
            if name in encoder_latent_names:
                continue
            input_groups.add(name)

        self.input_groups: List[str] = sorted(list(input_groups))

        self.obs_shapes: Dict[str, List[int]] = {}
        for k in self.input_groups:
            if k not in obs_shapes:
                raise KeyError(f"obs_example 缺少 key '{k}'")
            self.obs_shapes[k] = list(map(int, obs_shapes[k]))

        # memory buffers IO
        self.memory_names: List[str] = []
        self.memory_dims: Dict[str, int] = {}
        if hasattr(self.estimator, "memory_buffers"):
            for name, tensor in self.estimator.memory_buffers.items():
                self.memory_names.append(name)
                self.memory_dims[name] = int(tensor.shape[-1])

    def forward(self, *inputs: torch.Tensor):
        num_obs = len(self.input_groups)
        num_mem = len(self.memory_names)
        expected = num_obs + 1 + num_mem
        assert len(inputs) == expected, (
            f"Expected {expected} tensors (obs_* + h_in + mem_*_in), got {len(inputs)}."
        )

        obs_tensors = list(inputs[:num_obs])
        h_in = inputs[num_obs]
        mem_in_list = list(inputs[num_obs + 1:])

        batch_size = int(obs_tensors[0].size(0))

        # build obs tensordict
        obs_dict: Dict[str, torch.Tensor] = {}
        for name, tensor in zip(self.input_groups, obs_tensors):
            obs_dict[name] = tensor
        obs = TensorDict(obs_dict, batch_size=[batch_size])

        # override estimator memory_buffers
        if self.memory_names:
            for name, mem_in in zip(self.memory_names, mem_in_list):
                self.estimator.memory_buffers[name] = mem_in

        # normalize
        for name in self.normalize_groups:
            if name in obs and name in self.normalizer:
                obs[name + "_norm"] = self.normalizer[name](obs[name])

        # pass hidden state
        hidden_states = {self.latent_name: h_in}
        obs = self.estimator.encode_inference(obs, dones=None, hidden_states=hidden_states)

        # get new hidden state
        if not hasattr(self.estimator, "get_hidden_states_inference"):
            raise RuntimeError("Estimator has no get_hidden_states_inference.")
        hidden_dict = self.estimator.get_hidden_states_inference()
        if self.latent_name not in hidden_dict:
            raise RuntimeError(f"Missing hidden state key '{self.latent_name}' in get_hidden_states_inference().")
        h_out = hidden_dict[self.latent_name]

        # collect mem outs
        mem_out_list: List[torch.Tensor] = []
        for name in self.memory_names:
            mem_out_list.append(self.estimator.memory_buffers[name])

        # build actor obs
        obs_list: List[torch.Tensor] = []
        for name in self.policy_groups:
            if name not in obs:
                raise RuntimeError(f"Policy obs_group '{name}' missing after encode_inference/normalize.")
            x = obs[name]
            if self.detach_flags_policy.get(name, False):
                x = x.detach()
            obs_list.append(x)
        actor_obs = torch.cat(obs_list, dim=-1)

        # forward actor
        if self.state_dependent_std:
            actions = self.actor(actor_obs)[..., 0, :]
        else:
            actions = self.actor(actor_obs)

        return (actions, h_out, *mem_out_list)


# =============================================================================
# Export ActorCriticEst (non-RNN)
# =============================================================================

def export_tensordict_policy_as_onnx(
    policy: ActorCriticEst,
    obs_example: TensorDict,
    env_cfg: object,
    path: str,
    filename: str = "policy.onnx",
    verbose: bool = False,
):
    os.makedirs(path, exist_ok=True)

    obs_example = copy.deepcopy(obs_example).cpu()
    obs_shapes = _infer_obs_shapes(obs_example)

    wrapper = TensorDictActorDeployWrapper(policy, obs_shapes).eval().cpu()

    dummy_inputs: List[torch.Tensor] = []
    for name in wrapper.input_groups:
        dummy_inputs.append(_make_dummy_from_shape(wrapper.obs_shapes[name], batch=1))

    opset_version = 18
    input_names = [f"obs_{name}" for name in wrapper.input_groups]
    output_names = ["actions"]
    dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    onnx_full_path = os.path.join(path, filename)

    if verbose:
        print("[ONNX EXPORT] ActorCriticEst (non-RNN)")
        print("Wrapper input groups:", wrapper.input_groups)
        print("Wrapper input shapes:", wrapper.obs_shapes)
        print("Policy concat order:", wrapper.policy_groups)
        print("Normalize groups:", wrapper.normalize_groups)
        print("-------------------------------------------------------")

    torch.onnx.export(
        wrapper,
        tuple(dummy_inputs),
        onnx_full_path,
        export_params=True,
        opset_version=opset_version,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    if verbose:
        print(f"Saved ONNX policy to: {onnx_full_path}")

    _write_policy_config_json_from_onnx(
        onnx_path=onnx_full_path,
        env_cfg=env_cfg,
        obs_example=obs_example,
        json_path=os.path.join(path, "policy_config.json"),
        verbose=verbose,
    )


# =============================================================================
# Export ActorCriticEst (RNN + memory_buffers)
# =============================================================================

def export_tensordict_policy_rnn_as_onnx(
    policy: ActorCriticEst,
    obs_example: TensorDict,
    env_cfg: object,
    mem_info: Tuple[str, MemoryEncoder, str, nn.Module],
    path: str,
    filename: str = "policy.onnx",
    verbose: bool = False,
):
    os.makedirs(path, exist_ok=True)

    enc_name, mem_enc, latent_name, rnn = mem_info
    rnn_type = type(rnn).__name__.lower()
    num_layers = int(rnn.num_layers)
    hidden_size = int(rnn.hidden_size)

    obs_example = copy.deepcopy(obs_example).cpu()
    obs_shapes = _infer_obs_shapes(obs_example)

    wrapper = TensorDictActorRNNDeployWrapper(
        ac=policy,
        obs_shapes=obs_shapes,
        latent_name=latent_name,
        rnn=rnn,
    ).eval().cpu()

    dummy_inputs: List[torch.Tensor] = []
    for name in wrapper.input_groups:
        dummy_inputs.append(_make_dummy_from_shape(wrapper.obs_shapes[name], batch=1))

    dummy_h_in = torch.zeros(num_layers, 1, hidden_size, dtype=torch.float32)
    dummy_inputs.append(dummy_h_in)

    for mem_name in wrapper.memory_names:
        mem_dim = wrapper.memory_dims[mem_name]
        dummy_inputs.append(torch.zeros(1, mem_dim, dtype=torch.float32))

    opset_version = 18

    input_names = [f"obs_{name}" for name in wrapper.input_groups] + ["h_in"]
    input_names += [f"mem_{name}_in" for name in wrapper.memory_names]

    output_names = ["actions", "h_out"]
    output_names += [f"mem_{name}_out" for name in wrapper.memory_names]

    dynamic_axes: Dict[str, Dict[int, str]] = {}

    for name in input_names:
        if name.startswith("obs_") or name.startswith("mem_"):
            dynamic_axes[name] = {0: "batch"}
    dynamic_axes["actions"] = {0: "batch"}

    dynamic_axes["h_in"] = {1: "batch"}
    dynamic_axes["h_out"] = {1: "batch"}

    for mem_name in wrapper.memory_names:
        dynamic_axes[f"mem_{mem_name}_out"] = {0: "batch"}

    onnx_full_path = os.path.join(path, filename)

    if verbose:
        print("[ONNX EXPORT] ActorCriticEst (RNN)")
        print(f"Using memory encoder '{enc_name}' latent_name='{latent_name}' rnn_type='{rnn_type}'")
        print("Wrapper input groups:", wrapper.input_groups)
        print("Wrapper input shapes:", wrapper.obs_shapes)
        print("Memory buffers:", wrapper.memory_names, wrapper.memory_dims)
        print("-------------------------------------------------------")

    torch.onnx.export(
        wrapper,
        tuple(dummy_inputs),
        onnx_full_path,
        export_params=True,
        opset_version=opset_version,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    if verbose:
        print(f"Saved ONNX policy to: {onnx_full_path}")

    _write_policy_config_json_from_onnx(
        onnx_path=onnx_full_path,
        env_cfg=env_cfg,
        obs_example=obs_example,
        json_path=os.path.join(path, "policy_config.json"),
        verbose=verbose,
        extra_rnn_cfg={
            "type": rnn_type,
            "latent_name": latent_name,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
        },
    )


# =============================================================================
# Write policy_config.json (supports 2D/4D obs)
# =============================================================================

def _write_policy_config_json_from_onnx(
    onnx_path: str,
    env_cfg: object,
    obs_example: TensorDict,
    json_path: str,
    verbose: bool = False,
    extra_rnn_cfg: Optional[Dict] = None,
):
    onnx_model = onnx.load(onnx_path)
    initializer_names = {init.name for init in onnx_model.graph.initializer}

    real_input_names: List[str] = []
    for inp in onnx_model.graph.input:
        name = inp.name
        if name in initializer_names:
            continue
        if name.startswith("obs_"):
            real_input_names.append(name)

    if verbose:
        print("Real ONNX obs inputs:", real_input_names)

    obs_scales: Dict[str, float | List[float]] = {}
    obs_dims: Dict[str, List[int]] = {}

    for onnx_name in real_input_names:
        group_name = onnx_name[4:]  # remove 'obs_'

        if not hasattr(env_cfg.observations, group_name):
            print(f"[ONNX CONFIG] Warning: '{group_name}' not in env_cfg.observations; skip.")
            continue

        group_cfg = getattr(env_cfg.observations, group_name)

        term_scales = []
        for attr_val in group_cfg.__dict__.values():
            if isinstance(attr_val, ObsTerm):
                term_scales.append(attr_val.scale)

        if not term_scales:
            print(f"[ONNX CONFIG] Warning: '{group_name}' has no ObsTerm; skip.")
            continue

        if all(s == term_scales[0] for s in term_scales):
            obs_scales[onnx_name] = float(term_scales[0])
        else:
            obs_scales[onnx_name] = [float(s) for s in term_scales]

        if group_name not in obs_example:
            raise KeyError(f"[ONNX CONFIG] obs_example missing key '{group_name}'")

        x = obs_example[group_name]

        # 2D -> [feat_dim, history_len] (keeps your deploy convention)
        if x.dim() == 2:
            total_dim = int(x.shape[-1])

            history_len = 1
            for attr_val in group_cfg.__dict__.values():
                if isinstance(attr_val, ObsTerm):
                    history_len = int(getattr(attr_val, "history_length", 1))
                    break

            if history_len > 0 and total_dim % history_len == 0:
                feat_dim = total_dim // history_len
            else:
                if verbose:
                    print(
                        f"[ONNX CONFIG] Warning: '{group_name}' total_dim={total_dim} "
                        f"not divisible by history_len={history_len}; fallback to history_len=1."
                    )
                feat_dim = total_dim
                history_len = 1

            obs_dims[onnx_name] = [feat_dim, history_len]

        # 4D -> [C,H,W]
        elif x.dim() == 4:
            C, H, W = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
            obs_dims[onnx_name] = [C, H, W]
        else:
            raise ValueError(f"[ONNX CONFIG] '{group_name}' must be 2D or 4D, got {tuple(x.shape)}")

    data = {
        "obs_scales": obs_scales,
        "obs_dims": obs_dims,
        "commands_limits": {
            "lin_vel_x": env_cfg.commands.base_velocity.ranges.lin_vel_x,
            "lin_vel_y": env_cfg.commands.base_velocity.ranges.lin_vel_y,
            "ang_vel_z": env_cfg.commands.base_velocity.ranges.ang_vel_z,
        },
        "action_scale": env_cfg.actions.joint_pos.scale,
        "action_dims": len(env_cfg.joint_names),
        "policy_dof_order": env_cfg.joint_names,
        "dof_default_pos": env_cfg.scene.robot.init_state.joint_pos,
        "stiffness": {joint: env_cfg.scene.robot.actuators[joint].stiffness for joint in env_cfg.scene.robot.actuators},
        "damping": {joint: env_cfg.scene.robot.actuators[joint].damping for joint in env_cfg.scene.robot.actuators},
    }

    if extra_rnn_cfg is not None:
        data["rnn"] = extra_rnn_cfg

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    if verbose:
        print(f"Saved policy_config.json to: {json_path}")

    # optional copy to deploy dir (keep your original behavior)
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        gdog_root = os.path.abspath(os.path.join(this_dir, "..", "..", "..", "..", ".."))
        deploy_root = os.path.join(gdog_root, "Lite3_rl_deploy", "policy", "ppo")

        export_dir = os.path.dirname(json_path)
        run_name = os.path.basename(export_dir)
        target_dir = os.path.join(deploy_root, run_name)
        os.makedirs(target_dir, exist_ok=True)

        shutil.copy2(onnx_path, os.path.join(target_dir, os.path.basename(onnx_path)))
        shutil.copy2(json_path, os.path.join(target_dir, os.path.basename(json_path)))

        if verbose:
            print(f"[ONNX EXPORT] Copied policy files to deploy dir: {target_dir}")
    except Exception as e:
        print(f"[ONNX EXPORT] Warning: failed to copy to deploy dir: {e}")


# =============================================================================
# Old-style exporters (unchanged)
# =============================================================================

class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = getattr(policy, "is_recurrent", False)

        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")

        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            if self.rnn_type == "lstm":
                self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
                self.forward = self.forward_lstm
                self.reset = self.reset_memory
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
                self.reset = self.reset_memory
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")

        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward_gru(self, x):
        x = self.normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        if hasattr(self, "cell_state"):
            self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of old-style actor-critic into ONNX file (flat obs)."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.is_recurrent = getattr(policy, "is_recurrent", False)

        if hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")

        if self.is_recurrent:
            self.rnn.cpu()
            self.rnn_type = type(self.rnn).__name__.lower()
            if self.rnn_type == "lstm":
                self.forward = self.forward_lstm
            elif self.rnn_type == "gru":
                self.forward = self.forward_gru
            else:
                raise NotImplementedError(f"Unsupported RNN type: {self.rnn_type}")

        self.normalizer = copy.deepcopy(normalizer) if normalizer else torch.nn.Identity()

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x), h, c

    def forward_gru(self, x_in, h_in):
        x_in = self.normalizer(x_in)
        x, h = self.rnn(x_in.unsqueeze(0), h_in)
        x = x.squeeze(0)
        return self.actor(x), h

    def forward(self, x):
        return self.actor(self.normalizer(x))

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        os.makedirs(path, exist_ok=True)

        opset_version = 18
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)

            if self.rnn_type == "lstm":
                c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
                torch.onnx.export(
                    self,
                    (obs, h_in, c_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in", "c_in"],
                    output_names=["actions", "h_out", "c_out"],
                    dynamic_axes={},
                )
            elif self.rnn_type == "gru":
                torch.onnx.export(
                    self,
                    (obs, h_in),
                    os.path.join(path, filename),
                    export_params=True,
                    opset_version=opset_version,
                    verbose=self.verbose,
                    input_names=["obs", "h_in"],
                    output_names=["actions", "h_out"],
                    dynamic_axes={},
                )
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=opset_version,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )