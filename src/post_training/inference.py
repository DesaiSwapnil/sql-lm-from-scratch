"""
Inference helpers for chat and eval.

Adapted from FareedKhan-dev/train-llm-from-scratch (MIT License):
  src/post_training/inference.py (generate_response wrapper).

Provides a high-level generate_response that wraps the model's generate
method with the chat template and decoding, and a batch variant used by
the eval harness to amortise tokenisation overhead.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.post_training.chat_template import decode, encode_prompt, EOT_ID
from src.post_training.utils import unwrap


@torch.no_grad()
def generate_response(
    model: nn.Module,
    messages: list[dict] | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
    prompt_ids: list[int] | None = None,
) -> str:
    """
    Format messages through the chat template and generate a response.

    Returns the decoded assistant response string (without role markers or EOT).
    The model is set to eval mode for the duration and restored afterward.

    Args:
        model:          Transformer (unwrapped — pass the underlying module).
        messages:       list of {"role": "user"|"assistant"|"system", "content": str}.
        max_new_tokens: maximum tokens to generate.
        temperature:    sampling temperature (0 = greedy).
        top_p:          nucleus sampling cutoff (1.0 = disabled).
        device:         torch device string.
        prompt_ids:     pre-tokenized token ids — skips encode_prompt when provided
                        (used by smoke eval where tiktoken ids exceed the vocab size).
    """
    model.eval()
    if prompt_ids is None:
        prompt_ids = encode_prompt(messages)
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    context_len = unwrap(model).context_length
    import torch.nn.functional as F

    def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = (cumprobs - F.softmax(sorted_logits, dim=-1)) > p
        sorted_logits[remove] = float("-inf")
        return torch.scatter(logits, -1, sorted_idx, sorted_logits)

    prompt_end = len(prompt_ids)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if temperature > 0:
            logits = logits / temperature
        if top_p < 1.0:
            logits = _top_p_filter(logits, top_p)
        probs    = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_tok], dim=1)
        if next_tok.item() == EOT_ID:
            break

    response_ids = idx[0, prompt_end:].tolist()
    # Strip trailing EOT if present.
    if response_ids and response_ids[-1] == EOT_ID:
        response_ids = response_ids[:-1]
    return decode(response_ids)


@torch.no_grad()
def generate_sql_response(
    model: nn.Module,
    user_content: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
    prompt_ids: list[int] | None = None,
) -> str:
    """Convenience wrapper: single user message → decoded assistant response."""
    return generate_response(
        model,
        [{"role": "user", "content": user_content}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
        prompt_ids=prompt_ids,
    )
