"""
Rollout and log-probability helpers for GRPO.

Adapted from FareedKhan-dev/train-llm-from-scratch (MIT License):
  src/post_training/rollout.py (rollout loop shape, logprob computation).

Key additions vs. reference:
  - EOT-early-stopping so sequences are trimmed to their natural end.
  - Group-batched generation: all G completions for the same prompt are generated
    in a single (G, T) forward pass per step, giving G× GPU utilisation vs.
    sequential.  Across-group (P groups) remains sequential — no cross-group
    padding required since each group shares the same prompt.
  - Batched logprob computation for all N sequences simultaneously.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.post_training.chat_template import EOT_ID
from src.post_training.utils import unwrap

PAD_ID = 0   # token used to right-pad sequences to equal length after collection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling — works for (B, V) or (1, V)."""
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove   = (cumprobs - F.softmax(sorted_logits, dim=-1)) > top_p
    sorted_logits[remove] = float("-inf")
    return torch.scatter(logits, -1, sorted_idx, sorted_logits)


@torch.no_grad()
def _generate_one(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    device: str,
    temperature: float,
    top_p: float | None,
) -> torch.Tensor:
    """Generate one completion (fallback path, group_size=1).  Stops at EOT."""
    context_len = unwrap(model).context_length
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -context_len:])
        logits = logits[:, -1, :]
        if temperature > 0:
            logits = logits / temperature
        if top_p is not None and top_p < 1.0:
            logits = _top_p_filter(logits, top_p)
        next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        idx = torch.cat([idx, next_tok], dim=1)
        if next_tok.item() == EOT_ID:
            break

    return idx.squeeze(0)


@torch.no_grad()
def _generate_group(
    model: nn.Module,
    prompt_ids: list[int],
    group_size: int,
    max_new_tokens: int,
    device: str,
    temperature: float,
    top_p: float | None,
) -> list[torch.Tensor]:
    """
    Generate group_size completions for one prompt in a single batched loop.

    All G sequences start from the same prompt so no padding is needed during
    generation.  Sequences diverge only in content after the first sampled token.
    When a sequence hits EOT it is marked done and subsequent positions receive
    PAD_ID; the returned tensor for that sequence is sliced to exclude the padding,
    so the response_mask in rollout_prompts stays accurate.

    Returns:
        List of G 1-D tensors, each of shape (prompt_len + generation_len,).
        generation_len varies per sequence (stops at EOT or max_new_tokens).
    """
    context_len = unwrap(model).context_length
    P = len(prompt_ids)

    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    idx    = prompt.unsqueeze(0).expand(group_size, -1).clone()  # (G, P)

    done     = torch.zeros(group_size, dtype=torch.bool,  device=device)
    eff_ends = torch.full((group_size,), P + max_new_tokens,
                          dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        if done.all():
            break
        logits, _ = model(idx[:, -context_len:])  # (G, T, V)
        logits = logits[:, -1, :]                  # (G, V)
        if temperature > 0:
            logits = logits / temperature
        if top_p is not None and top_p < 1.0:
            logits = _top_p_filter(logits, top_p)

        next_toks = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)  # (G,)

        # Capture effective end for sequences that hit EOT this step.
        just_done = (~done) & (next_toks == EOT_ID)
        if just_done.any():
            end_val = torch.full_like(eff_ends, P + step + 1)
            eff_ends = torch.where(just_done, end_val, eff_ends)

        # Already-done sequences get PAD so idx stays rectangular.
        next_toks = torch.where(done, torch.full_like(next_toks, PAD_ID), next_toks)
        idx  = torch.cat([idx, next_toks.unsqueeze(1)], dim=1)
        done = done | just_done

    # Clamp eff_ends to actual tensor length (early group-exit truncates total steps).
    actual_len = idx.shape[1]
    eff_ends   = eff_ends.clamp(max=actual_len)

    # Return each sequence trimmed to its effective end (no trailing PAD).
    return [idx[i, : eff_ends[i].item()] for i in range(group_size)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rollout_prompts(
    model: nn.Module,
    prompts: list[list[int]],
    max_new_tokens: int,
    device: str,
    temperature: float = 1.0,
    top_p: float | None = None,
    group_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample one completion per prompt.

    Args:
        model:          the policy model (unwrapped, in eval mode).
        prompts:        list of N prompt token-id lists.  When group_size > 1,
                        N must equal P × group_size, and consecutive group_size
                        entries must be identical (same prompt repeated).
        max_new_tokens: max new tokens to generate per completion.
        device:         torch device string.
        temperature:    sampling temperature.
        top_p:          nucleus sampling cutoff, or None.
        group_size:     when > 1, completions within each group of identical prompts
                        are generated in a single batched forward pass (G× faster
                        than sequential on GPU without any cross-prompt padding).

    Returns:
        seqs:          (N, T_max) int64 — right-padded with PAD_ID.
        response_mask: (N, T_max) bool  — True at generated (non-prompt) positions.
        prompt_lens:   (N,) int64       — prompt length for each sequence.
    """
    model.eval()
    sequences:   list[torch.Tensor] = []
    prompt_lens: list[int]          = []

    if group_size > 1:
        assert len(prompts) % group_size == 0, \
            f"len(prompts)={len(prompts)} must be divisible by group_size={group_size}"
        P = len(prompts) // group_size
        for p_idx in range(P):
            group_prompt = prompts[p_idx * group_size]  # all G copies are identical
            group_seqs   = _generate_group(model, group_prompt, group_size,
                                           max_new_tokens, device, temperature, top_p)
            sequences.extend(group_seqs)
            prompt_lens.extend([len(group_prompt)] * group_size)
    else:
        for toks in prompts:
            sequences.append(_generate_one(model, toks, max_new_tokens, device, temperature, top_p))
            prompt_lens.append(len(toks))

    # Right-pad all sequences to the same length.
    T_max = max(s.shape[0] for s in sequences)
    N     = len(sequences)
    seqs          = torch.full((N, T_max), PAD_ID, dtype=torch.long, device=device)
    response_mask = torch.zeros((N, T_max),        dtype=torch.bool, device=device)

    for i, (seq, plen) in enumerate(zip(sequences, prompt_lens)):
        L = seq.shape[0]
        seqs[i, :L] = seq
        if L > plen:
            response_mask[i, plen:L] = True   # True only at real generated positions

    return seqs, response_mask, torch.tensor(prompt_lens, dtype=torch.long, device=device)


def compute_logprobs(
    model: nn.Module,
    seqs: torch.Tensor,
    response_mask: torch.Tensor,
    temperature: float = 1.0,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token log-probabilities for all positions.

    logprobs[:, t] = log π(seqs[:, t+1] | seqs[:, :t+1]).
    To select response-token positions in the target space: use response_mask[:, 1:].

    Args:
        model:         policy or reference model.
        seqs:          (N, T) int64 padded sequences.
        response_mask: (N, T) bool mask from rollout_prompts.
        temperature:   logit scaling (1.0 for training; match rollout temp for diagnostics).
        requires_grad: False wraps in torch.no_grad() for ref/old-policy passes.

    Returns:
        logprobs: (N, T-1) per-token log-probs.
        entropy:  (N,) mean entropy over response tokens (monitoring).
    """
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx:
        inp = seqs[:, :-1]
        tgt = seqs[:, 1:]

        logits, _ = model(inp)          # (N, T-1, V)
        if temperature != 1.0:
            logits = logits / temperature

        log_probs_all = F.log_softmax(logits, dim=-1)
        tok_logprobs  = log_probs_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

        resp_mask_f = response_mask[:, 1:].float()
        probs       = log_probs_all.exp()
        entropy_all = -(probs * log_probs_all).sum(-1)
        mean_ent    = (entropy_all * resp_mask_f).sum(-1) / resp_mask_f.sum(-1).clamp(min=1)

    return tok_logprobs, mean_ent
