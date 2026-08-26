"""
Interactive chat with a trained checkpoint.

Adapted from FareedKhan-dev/train-llm-from-scratch (MIT License):
  scripts/chat.py (REPL loop, prompt formatting, streaming output shape).

Usage:
    PYTHONPATH=. python scripts/chat.py --ckpt /content/ckpts/grpo.pt
    PYTHONPATH=. python scripts/chat.py --ckpt /tmp/sql_lm_smoke/sft.pt --smoke
"""

from __future__ import annotations

import argparse
import os

import torch

from config.loader import load_config
from config.training_config import BaseModelConfig
from src.post_training.inference import generate_response
from src.post_training.rewards.parsing import extract_sql, extract_think
from src.post_training.utils import load_backbone_from_ckpt, unwrap


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt",           required=True, help="Path to checkpoint (.pt)")
    p.add_argument("--config",         default=None,
                   help="Optional JSON config path for architecture overrides")
    p.add_argument("--max_new_tokens", type=int,   default=512)
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--top_p",          type=float, default=0.9)
    p.add_argument("--smoke",          action="store_true",
                   help="Use smoke architecture config (configs/smoke/base.json)")
    return p.parse_args()


def _format_response(text: str) -> str:
    """Pretty-print a model response: show think and SQL blocks clearly."""
    think = extract_think(text)
    sql   = extract_sql(text)
    parts = []
    if think:
        parts.append(f"\033[90m[think] {think}\033[0m")    # grey
    if sql:
        parts.append(f"\033[96m[sql]\n{sql}\033[0m")        # cyan
    if not parts:
        parts.append(text)
    return "\n".join(parts)


def main() -> None:
    args   = _parse_args()
    config_path = args.config or ("configs/smoke/base.json" if args.smoke else None)
    cfg    = load_config(BaseModelConfig, config_path)

    device = cfg.device if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        object.__setattr__(cfg, "device",   "cpu")
        object.__setattr__(cfg, "amp_dtype", None)

    print(f"Loading {args.ckpt} …")
    model = load_backbone_from_ckpt(cfg, args.ckpt, device)
    model.eval()
    raw = unwrap(model)

    print(f"Model ready on {device}. Type 'quit' or Ctrl-C to exit.\n")
    history: list[dict] = []

    while True:
        try:
            user_input = input("\033[1mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        if user_input.lower() == "/clear":
            history.clear()
            print("(conversation cleared)")
            continue

        if user_input.lower() == "/history":
            for m in history:
                print(f"  [{m['role']}] {m['content'][:80]}")
            continue

        history.append({"role": "user", "content": user_input})

        response = generate_response(
            raw,
            history,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        history.append({"role": "assistant", "content": response})

        print(f"\033[1mAssistant:\033[0m")
        print(_format_response(response))
        print()


if __name__ == "__main__":
    main()
