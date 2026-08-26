"""
RMSNorm — Root Mean Square Layer Normalization.

Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019).

Formula:
    RMS(x) = sqrt(mean(x^2) + eps)
    RMSNorm(x) = (x / RMS(x)) * weight

No mean-centering (unlike LayerNorm) and no additive bias. The norm computation
runs in float32 regardless of input dtype to avoid underflow with bf16 training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for the norm, then return in original dtype.
        # This prevents catastrophic cancellation / underflow in bf16.
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f * rms).to(x.dtype) * self.weight
