from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class AttnCache:
    # k, v: [B, H, T_cache, Dh]
    k: torch.Tensor
    v: torch.Tensor


class SelfAttention(nn.Module):
    """
    Batch-first Multi-Head Self-Attention
    Input : x [B, T, D]
    Output: y [B, T, D]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, H, T, Dh]
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[AttnCache] = None,
        return_cache: bool = False,
    ):
        """
        Args:
            x: [B, T, D]
            attn_mask: bool tensor [B, T_k]
                       True  = valid token
                       False = padding (masked)
            cache: optional KV cache for streaming inference
            return_cache: whether to return updated cache
        """
        B, T, D = x.shape
        assert D == self.embed_dim

        # ---- QKV projection ----
        qkv = self.qkv(x)                      # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._split_heads(q)               # [B, H, T, Dh]
        k = self._split_heads(k)
        v = self._split_heads(v)

        # ---- Append cache ----
        if cache is not None:
            k = torch.cat([cache.k, k], dim=2)  # [B, H, T_k, Dh]
            v = torch.cat([cache.v, v], dim=2)

        T_k = k.size(2)

        # ---- Attention scores ----
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # shape: [B, H, T, T_k]

        # ---- Padding mask ----
        if attn_mask is not None:
            # attn_mask: [B, T_k], True = keep
            mask = ~attn_mask[:, None, None, :]  # [B,1,1,T_k]
            attn = attn.masked_fill(mask, float("-inf"))

        # ---- Causal mask ----
        if self.causal:
            q_pos = torch.arange(T_k - T, T_k, device=x.device).view(T, 1)
            k_pos = torch.arange(T_k, device=x.device).view(1, T_k)
            causal_mask = k_pos > q_pos          # [T, T_k]
            attn = attn.masked_fill(
                causal_mask.view(1, 1, T, T_k),
                float("-inf"),
            )

        # ---- Softmax ----
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # ---- Weighted sum ----
        y = torch.matmul(attn, v)               # [B, H, T, Dh]

        # ---- Merge heads ----
        y = y.transpose(1, 2).contiguous()      # [B, T, H, Dh]
        y = y.view(B, T, D)                      # [B, T, D]
        y = self.proj(y)

        if return_cache:
            new_cache = AttnCache(k=k.detach(), v=v.detach())
            return y, new_cache

        return y