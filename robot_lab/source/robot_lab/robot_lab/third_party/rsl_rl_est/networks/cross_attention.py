from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CrossAttnCache:
    # cached k,v: [B, H, Tk_cache, Dh]
    k: torch.Tensor
    v: torch.Tensor


class CrossAttention(nn.Module):
    """
    Feature-split Cross Attention
    Q : [B, 1, D]
    KV: [B, Tk, D]
    Out: [B, 1, D]
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, D] -> [B, H, T, Dh]
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q : [B, 1, D]
        kv: [B, Tk, D]
        """
        q = self._split_heads(self.q_proj(q))      # [B,H,1,Dh]
        k = self._split_heads(self.k_proj(kv))     # [B,H,Tk,Dh]
        v = self._split_heads(self.v_proj(kv))     # [B,H,Tk,Dh]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        y = torch.matmul(attn, v)                  # [B,H,1,Dh]
        y = y.transpose(1, 2).contiguous()         # [B,1,H,Dh]
        y = y.view(y.shape[0], 1, self.embed_dim)  # [B,1,D]
        return self.out_proj(y)