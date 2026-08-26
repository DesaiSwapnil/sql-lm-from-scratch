"""
Pre-norm transformer block.

Structure (per block):
    x = x + Attention(RMSNorm(x))    ← attention sub-layer with residual
    x = x + MLP(RMSNorm(x))          ← MLP sub-layer with residual

Pre-norm (normalize the *input* to each sublayer) is more stable than post-norm
at large depth and matches the LLaMA / Mistral architecture convention.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention
from src.models.mlp import SwiGLUMLP
from src.models.rms_norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embed: int,
        n_head: int,
        context_length: int,
        swiglu_hidden: int,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(n_embed)
        self.attn  = MultiHeadAttention(n_embed, n_head, context_length)
        self.norm2 = RMSNorm(n_embed)
        self.mlp   = SwiGLUMLP(n_embed, swiglu_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
