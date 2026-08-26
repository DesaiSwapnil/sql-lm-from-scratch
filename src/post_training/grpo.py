"""
GRPO loss and group-advantage computation — implemented in Stage 7.

Reference: DeepSeek-R1 / GRPO (Shao et al. 2024).
No critic; advantages are computed group-relative within each prompt's sample group.
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """
    Normalize rewards within each prompt's sample group to get advantages.

    rewards: (N,) where N = n_prompts * group_size, group-contiguous ordering.
    Returns advantages: (N,) zero-mean, unit-std within each group (with stability eps).
    """
    raise NotImplementedError("group_advantages will be implemented in Stage 7")


def grpo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip: float,
    kl_coef: float,
) -> tuple[torch.Tensor, dict]:
    """
    Clipped GRPO surrogate loss + KL-to-reference penalty.

    Returns (loss scalar, stats dict with 'kl' and 'clipfrac').
    """
    raise NotImplementedError("grpo_loss will be implemented in Stage 7")
