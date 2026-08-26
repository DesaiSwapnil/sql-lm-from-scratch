# Adapted from FareedKhan-dev/train-llm-from-scratch (MIT License).
# Original: https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/optim.py
# Changes: none to the core logic.
"""Optimizer and learning-rate schedule helpers shared across all training stages."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def configure_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """AdamW with weight decay applied only to >=2D parameters (matrices), not to
    biases / norms / 1D params. Standard recipe from the GPT-3 paper."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def cosine_lr(step: int, *, warmup_steps: int, max_steps: int, lr: float, min_lr: float) -> float:
    """Linear warmup to lr over warmup_steps, then cosine decay to min_lr by max_steps."""
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)
