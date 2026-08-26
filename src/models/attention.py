"""
Multi-head causal self-attention with Rotary Position Embeddings (RoPE).

References:
    Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021).
    Touvron et al., "LLaMA: Open and Efficient Foundation Language Models" (2023).

Design choices:
    - Fused QKV linear (single matrix for all heads) instead of three separate ones.
    - RoPE applied to Q and K only, not V — exactly as in the original paper.
    - Causal mask via torch.nn.functional.scaled_dot_product_attention(is_causal=True),
      which dispatches to Flash Attention on A100/L4 and efficient SDPA elsewhere.
    - No GQA (grouped-query attention) — not needed at 250M / single-GPU scale.
    - No dropout — we're training from scratch with weight decay; dropout on attention
      weights hurts more than it helps at this model size.

RoPE rotation detail:
    For each attention head, positions 0..T-1 and feature dimensions 0..head_dim-1.
    We precompute (context_length, head_dim) cos/sin tables with entries:
        cos[m, 2i] = cos[m, 2i+1] = cos(m / base^(2i/head_dim))
        sin[m, 2i] = sin[m, 2i+1] = sin(m / base^(2i/head_dim))
    The rotation for feature pair (2i, 2i+1) at position m is:
        q'[2i]   = q[2i]*cos - q[2i+1]*sin
        q'[2i+1] = q[2i]*sin + q[2i+1]*cos
    which is equivalent to q * cos + rotate_half(q) * sin where
    rotate_half([q0, q1, q2, q3, ...]) = [-q1, q0, -q3, q2, ...].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Precomputes and buffers RoPE cos/sin tables up to context_length."""

    def __init__(self, head_dim: int, context_length: int, base: int = 10_000) -> None:
        super().__init__()
        # theta_i = 1 / base^(2i/head_dim) for i in 0..head_dim//2-1
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        m = torch.arange(context_length, dtype=torch.float32)
        freqs = torch.outer(m, theta)            # (context_length, head_dim//2)
        # Repeat each frequency for both the original and rotated partner:
        # freqs[:, i] applies to positions 2i and 2i+1.
        cos = freqs.cos().repeat_interleave(2, dim=-1)  # (context_length, head_dim)
        sin = freqs.sin().repeat_interleave(2, dim=-1)  # (context_length, head_dim)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Produce the paired-rotation of x: [-x1, x0, -x3, x2, ...]."""
    x_even = x[..., 0::2]   # q0, q2, q4, ...
    x_odd  = x[..., 1::2]   # q1, q3, q5, ...
    # Interleave so index 2i → -x_odd[i], index 2i+1 → x_even[i]
    return torch.stack([-x_odd, x_even], dim=-1).flatten(-2)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k. cos/sin: (T, head_dim); q,k: (B, n_head, T, head_dim)."""
    cos = cos[None, None]   # (1, 1, T, head_dim) — broadcasts over B and n_head
    sin = sin[None, None]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class MultiHeadAttention(nn.Module):
    """Causal MHA with RoPE using SDPA (Flash Attention on supported hardware)."""

    def __init__(self, n_embed: int, n_head: int, context_length: int) -> None:
        super().__init__()
        assert n_embed % n_head == 0, f"n_embed ({n_embed}) must be divisible by n_head ({n_head})"
        self.n_head = n_head
        self.head_dim = n_embed // n_head
        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, context_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)                              # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1)                 # each (B, T, C)

        # Reshape to (B, n_head, T, head_dim) for multi-head ops.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE: buffers are on the model's device, sliced to current T.
        cos, sin = self.rope(T)
        q, k = _apply_rope(q, k, cos, sin)

        # Causal attention — Flash Attention backend on A100/L4 via SDPA.
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # (B, n_head, T, head_dim) → (B, T, C)
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(x)
