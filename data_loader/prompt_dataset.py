"""
RL prompt dataset loader for GRPO training.

Reads JSONL files written by scripts/prepare_rl_prompts.py.
Each line: {"prompt": <str|list[int]>, "gold_sql": "...", "db_id": "...", "schema_path": "..."}

When "prompt" is a string it is treated as the user-message content and tokenised
via encode_prompt at iteration time.  When it is a list of ints it is used directly
(smoke mode, where the vocab is too small for the real tiktoken tokenizer).
"""

from __future__ import annotations

import json
import random
from typing import Iterator


def get_prompt_iterator(
    path: str,
    prompts_per_iter: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> Iterator[list[dict]]:
    """
    Infinite iterator that yields lists of prompts_per_iter prompt dicts per call.

    Each dict has keys: prompt (str | list[int]), gold_sql, db_id, schema_path.
    Shards the prompt list across ranks; shuffles at the start of each epoch.
    """
    with open(path) as fh:
        all_prompts = [json.loads(line) for line in fh if line.strip()]

    # Shard by rank for multi-GPU consistency (each rank sees different prompts).
    all_prompts = all_prompts[rank::world_size]
    if not all_prompts:
        raise ValueError(f"No prompts left after rank sharding (rank={rank}, world_size={world_size})")

    rng = random.Random(seed + rank)
    epoch = 0
    while True:
        rng.shuffle(all_prompts)
        epoch += 1
        i = 0
        while i + prompts_per_iter <= len(all_prompts):
            yield all_prompts[i : i + prompts_per_iter]
            i += prompts_per_iter
        # Leftover prompts at end of epoch: yield them too if non-empty.
        if i < len(all_prompts):
            yield all_prompts[i:]
