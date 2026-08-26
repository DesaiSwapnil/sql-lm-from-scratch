"""
SwiGLU gated MLP.

Reference: Noam Shazeer, "GLU Variants Improve Transformer" (2020), §2.

Formula:
    SwiGLU(x) = SiLU(gate(x)) ⊙ up(x)
    output    = down(SwiGLU(x))

Three weight matrices (no bias):
    gate : n_embed → hidden_dim
    up   : n_embed → hidden_dim
    down : hidden_dim → n_embed

With hidden_dim = 8/3 * n_embed (rounded to a sensible multiple), SwiGLU's three
matrices carry slightly more parameters than a standard 4× ReLU MLP's two matrices,
but hidden_dim is sized to keep total params on budget (swiglu_hidden=2048 for n_embed=768
gives exactly 8/3 × 768 = 2048, so there's no rounding involved).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    def __init__(self, n_embed: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(n_embed, hidden_dim, bias=False)
        self.up   = nn.Linear(n_embed, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, n_embed, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))
