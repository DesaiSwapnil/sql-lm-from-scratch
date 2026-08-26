# Adapted from FareedKhan-dev/train-llm-from-scratch (MIT License).
# Original: https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/chat_template.py
# Changes: switched tokenizer from r50k_base to cl100k_base; replaced math answer markers
# with SQL markers (<sql>...</sql>); kept <think>...</think> for reasoning chains.
"""
Chat formatting + loss masking for post-training.

Tokenizer: tiktoken cl100k_base (vocab 0-100255; special tokens at 100256+).
EOT token: <|endoftext|> at id 100257 (the only special token we use as a stop signal).

We cannot register new special tokens in tiktoken without a custom encoding, so role
markers are plain-text strings that tokenize as ordinary multi-token sequences.
The model learns them during SFT.

Conversation format::

    <|user|>
    {user content}<|endoftext|><|assistant|>
    {assistant content}<|endoftext|>

For SQL tasks the assistant content carries the reasoning + SQL structure::

    <think>step-by-step reasoning</think><sql>SELECT ...</sql>

Loss mask: 1 on assistant content tokens (and the closing EOT); 0 on everything else.
This is the standard SFT prompt-masking mechanism and doubles as the GRPO response_mask.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import tiktoken

# Special token ids for cl100k_base.
# <|endoftext|> is id 100257; regular BPE tokens run 0-100255.
_BPE_VOCAB_SIZE = 100256   # first index that is NOT a regular BPE token
EOT_ID = 100257            # <|endoftext|> in cl100k_base

# Plain-text role markers (ordinary tokens, not registered specials).
USER_HEADER = "<|user|>\n"
ASSISTANT_HEADER = "<|assistant|>\n"
SYSTEM_HEADER = "<|system|>\n"

# Reasoning + SQL structure markers used by the SFT data formatter and reward verifier.
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
SQL_OPEN, SQL_CLOSE = "<sql>", "</sql>"


@lru_cache(maxsize=1)
def get_tokenizer() -> tiktoken.Encoding:
    """Return the shared cl100k_base encoder (cached so we build it once per process)."""
    return tiktoken.get_encoding("cl100k_base")


def _encode_ordinary(text: str) -> list[int]:
    return get_tokenizer().encode_ordinary(text)


def _header_for(role: str) -> str:
    if role == "user":
        return USER_HEADER
    if role == "assistant":
        return ASSISTANT_HEADER
    if role == "system":
        return SYSTEM_HEADER
    raise ValueError(f"Unknown chat role: {role!r}")


def render_chat(messages: Iterable[dict], add_generation_prompt: bool = False) -> str:
    """Render messages to the plain-text chat format (for display/debugging only)."""
    parts: list[str] = []
    for m in messages:
        parts.append(_header_for(m["role"]))
        parts.append(m["content"])
        parts.append("<|endoftext|>")
    if add_generation_prompt:
        parts.append(ASSISTANT_HEADER)
    return "".join(parts)


def encode_chat(
    messages: Iterable[dict],
    add_generation_prompt: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Tokenize a conversation and build an aligned per-token loss mask.

    mask=1 over assistant content and the closing EOT; 0 everywhere else.
    When add_generation_prompt=True the mask is all zeros (prompt-only form for rollouts).

    Returns:
        (ids, loss_mask) as equal-length python lists of ints.
    """
    ids: list[int] = []
    mask: list[int] = []

    for m in messages:
        role = m["role"]
        header_ids = _encode_ordinary(_header_for(role))
        ids.extend(header_ids)
        mask.extend([0] * len(header_ids))

        content_ids = _encode_ordinary(m["content"])
        is_completion = role == "assistant"
        ids.extend(content_ids)
        mask.extend([1 if is_completion else 0] * len(content_ids))

        ids.append(EOT_ID)
        mask.append(1 if is_completion else 0)

    if add_generation_prompt:
        header_ids = _encode_ordinary(ASSISTANT_HEADER)
        ids.extend(header_ids)
        mask.extend([0] * len(header_ids))

    return ids, mask


def encode_prompt(messages: Iterable[dict]) -> list[int]:
    """Token ids for the prompt form (ends in the assistant header, ready to generate)."""
    ids, _ = encode_chat(messages, add_generation_prompt=True)
    return ids


def decode(ids: Iterable[int]) -> str:
    """
    Decode token ids back to text.

    Filters out EOT and any ids outside the regular BPE range (0-100255) so that
    padding ids emitted by an under-trained model don't crash tiktoken's decoder.
    """
    clean = [t for t in ids if 0 <= t < _BPE_VOCAB_SIZE]
    return get_tokenizer().decode(clean)
