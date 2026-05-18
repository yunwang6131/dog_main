#!/usr/bin/env python3
"""Verify merged ``policy_full.pt`` (CENet + actor MLP) without Isaac / MuJoCo.

  1) Smoke: random/zeros forward — model loads and runs.
  2) Match: same tensors through ``cenet.pt`` + ``policy.pt`` vs ``policy_full.pt`` — outputs should match.

Usage (from repo ``dolanga1_mujoco_sim2sim`` root)::

    PYTHONPATH=. python deploy_mujoco/verify_policy_full_jit.py \\
      --exported /path/to/.../exported

Requires: PyTorch; optional ``onnx`` for auto dim inference from ``policy.onnx``.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from deploy_mujoco.dreamwaq_cenet_loader import load_dreamwaq_cenet_from_rsl_checkpoint


def _infer_mlp_in_from_onnx(exported_dir: str) -> int | None:
    onnx_path = os.path.join(exported_dir, "policy.onnx")
    if not os.path.isfile(onnx_path):
        return None
    try:
        import onnx  # noqa: PLC0415
    except ImportError:
        print("[WARN] policy.onnx present but ``onnx`` not installed; install or pass --current-dim.")
        return None
    model = onnx.load(onnx_path)
    for inp in model.graph.input:
        tt = inp.type.tensor_type
        if not tt.HasField("shape"):
            continue
        dims = tt.shape.dim
        if len(dims) < 2:
            continue
        d1 = dims[1]
        if d1.HasField("dim_value") and int(d1.dim_value) > 0:
            return int(d1.dim_value)
    return None


def _load_ts(path: str, map_location: torch.device) -> torch.jit.ScriptModule:
    m = torch.jit.load(path, map_location=map_location)
    m.eval()
    return m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--exported",
        type=str,
        required=True,
        help="Directory containing policy_full.pt (and optionally cenet.pt, policy.pt, policy.onnx).",
    )
    p.add_argument(
        "--current-dim",
        type=int,
        default=None,
        help="Proprio vector dim (training ``current_groups``). If omitted, infer from policy.onnx + cenet.pt.",
    )
    p.add_argument("--atol", type=float, default=1e-5, help="allclose atol for split vs merged.")
    p.add_argument("--rtol", type=float, default=1e-4, help="allclose rtol for split vs merged.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    exported = os.path.abspath(args.exported)
    full_path = os.path.join(exported, "policy_full.pt")
    cenet_path = os.path.join(exported, "cenet.pt")
    policy_tail_path = os.path.join(exported, "policy.pt")

    if not os.path.isfile(full_path):
        print(f"[ERR] Missing: {full_path}", file=sys.stderr)
        return 1

    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    merged = _load_ts(full_path, device)

    cenet = load_dreamwaq_cenet_from_rsl_checkpoint(cenet_path, device=device)
    hdim = int(cenet.encoder[0].in_features)
    vel_d = int(cenet.velocity_head.out_features)
    lat_d = int(cenet.latent_mu_head.out_features)

    mlp_in = _infer_mlp_in_from_onnx(exported)
    if args.current_dim is not None:
        current_dim = int(args.current_dim)
    elif mlp_in is not None:
        current_dim = mlp_in - vel_d - lat_d
        if current_dim <= 0:
            print(f"[ERR] Inferred current_dim invalid: mlp_in={mlp_in}, vel={vel_d}, lat={lat_d}.", file=sys.stderr)
            return 1
    else:
        print(
            "[ERR] Cannot infer current_dim: install ``onnx`` and keep policy.onnx next to policy_full.pt, "
            "or pass --current-dim (e.g. 45 for Dolanga1 DreamWaQ).",
            file=sys.stderr,
        )
        return 1

    if mlp_in is not None and current_dim + vel_d + lat_d != mlp_in:
        print(
            f"[WARN] current_dim+vel+lat = {current_dim}+{vel_d}+{lat_d} != policy.onnx obs dim {mlp_in}. "
            "Check --current-dim or checkpoint export.",
            file=sys.stderr,
        )

    hist = torch.randn(1, hdim, device=device)
    cur = torch.randn(1, current_dim, device=device)

    # ---- 1) smoke ----
    with torch.inference_mode():
        actions = merged(hist, cur)
    if actions.ndim != 2 or actions.shape[0] != 1:
        print(f"[ERR] Unexpected merged output shape: {tuple(actions.shape)}", file=sys.stderr)
        return 1
    print(f"[OK] Smoke forward: policy_full.pt -> actions shape {tuple(actions.shape)} (dtype={actions.dtype}).")

    # ---- 2) numeric match split vs merged ----
    if not os.path.isfile(policy_tail_path):
        print("[INFO] Skip split-vs-merged check (no policy.pt in exported/). Smoke test was enough.")
        return 0

    policy_tail = _load_ts(policy_tail_path, device)
    with torch.inference_mode():
        v_est, z = cenet(hist)
        split_in = torch.cat([cur, v_est, z], dim=-1)
        act_split = policy_tail(split_in)
        act_merged = merged(hist, cur)

    if act_split.shape != act_merged.shape:
        print(f"[ERR] Shape mismatch split {act_split.shape} vs merged {act_merged.shape}", file=sys.stderr)
        return 1

    max_abs = (act_split - act_merged).abs().max().item()
    ok = torch.allclose(act_split, act_merged, rtol=args.rtol, atol=args.atol)
    if ok:
        print(
            f"[OK] Split path matches merged: max_abs_diff={max_abs:.3e} (rtol={args.rtol}, atol={args.atol}). "
            f"Dims: history={hdim}, current={current_dim}, mlp_in={current_dim + vel_d + lat_d}."
        )
        return 0

    print(
        f"[ERR] Mismatch: max_abs_diff={max_abs:.3e}. "
        "Ensure policy.pt / cenet.pt / policy_full.pt are from the same play export.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
