"""
SQL execution accuracy evaluation on Spider / BIRD held-out splits.

Runs the model on a held-out split and reports:
  - Execution accuracy (EX): fraction where generated SQL result set == gold result set.
  - Format rate: fraction where the model produced a well-formed <sql>…</sql> block.
  - Valid-exec rate: fraction where the generated SQL executed without error.
  - Mean reward: mean of the 4-level reward signal (0 / 0.1 / 0.3 / 1.0).

Can compare multiple checkpoints in one run (--ckpt may be given more than once or
as a glob) to show pretrain → SFT → GRPO progression.

Spider directory layout expected:
    spider_dir/
        dev.json                           — held-out examples
        database/<db_id>/<db_id>.sqlite    — SQLite files

Smoke mode (--smoke) runs on the tiny in-memory schema from prepare_rl_prompts.py
so the full pipeline can be validated without downloading Spider.

Usage:
    # Evaluate one checkpoint:
    PYTHONPATH=. python scripts/eval_sql.py \\
        --ckpt /content/ckpts/grpo.pt \\
        --spider_dir /content/data/spider

    # Compare multiple checkpoints:
    PYTHONPATH=. python scripts/eval_sql.py \\
        --ckpt /content/ckpts/pretrain.pt \\
        --ckpt /content/ckpts/sft.pt \\
        --ckpt /content/ckpts/grpo.pt \\
        --spider_dir /content/data/spider

    # Smoke test (no Spider download needed):
    PYTHONPATH=. python scripts/eval_sql.py \\
        --ckpt /tmp/sql_lm_smoke/sft.pt \\
        --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from dataclasses import dataclass

import torch

from config.loader import load_config
from config.training_config import BaseModelConfig
from src.post_training.inference import generate_sql_response
from src.post_training.rewards.parsing import (
    extract_sql,
    has_well_formed_sql,
)
from src.post_training.rewards.sql_reward import (
    REWARD_CORRECT,
    reward_sql,
)
from src.post_training.utils import load_backbone_from_ckpt, unwrap


# ---------------------------------------------------------------------------
# Spider loader
# ---------------------------------------------------------------------------

def _schema_desc(db_path: str) -> str:
    """Minimal schema description by introspecting the SQLite file."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
        lines = []
        for tbl in tables:
            lines.append(f"Table: {tbl}")
            for col in conn.execute(f"PRAGMA table_info({tbl})").fetchall():
                lines.append(f"  {tbl}.{col[1]}")
        conn.close()
        return "\n".join(lines)
    except Exception:
        return ""


def _load_spider_examples(spider_dir: str, split: str) -> list[dict]:
    """
    Load examples from spider_dir/{split}.json.

    Each returned dict has: user_content, gold_sql, db_id, schema_path.
    Examples whose SQLite file is missing are skipped.
    """
    split_file = os.path.join(spider_dir, f"{split}.json")
    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"Spider split file not found: {split_file}\n"
            f"Available files: {os.listdir(spider_dir)}"
        )
    with open(split_file) as fh:
        raw = json.load(fh)

    examples = []
    skipped  = 0
    for ex in raw:
        db_id    = ex.get("db_id", "")
        question = ex.get("question", "").strip()
        gold_sql = ex.get("query", "").strip()
        if not db_id or not question or not gold_sql:
            skipped += 1
            continue
        db_path = os.path.join(spider_dir, "database", db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            skipped += 1
            continue
        schema  = _schema_desc(db_path)
        examples.append({
            "user_content": f"Schema:\n{schema}\n\nQuestion: {question}",
            "gold_sql":     gold_sql,
            "db_id":        db_id,
            "schema_path":  db_path,
        })
    if skipped:
        print(f"  [loader] skipped {skipped} examples (missing db or empty fields)")
    return examples


# ---------------------------------------------------------------------------
# Smoke examples
# ---------------------------------------------------------------------------

def _load_smoke_examples(smoke_db: str, smoke_prompts: str) -> list[dict]:
    """
    Load smoke examples from the pre-tokenized JSONL written by prepare_rl_prompts.py.

    Using pre-tokenized ids avoids OOB embedding errors: the smoke model has
    vocab_size=512, but tiktoken (cl100k_base) produces ids up to 100352.
    """
    examples = []
    with open(smoke_prompts) as fh:
        for line in fh:
            if line.strip():
                ex = json.loads(line)
                examples.append({
                    "user_content": ex["gold_sql"],   # display label only
                    "prompt_ids":   ex["prompt"],      # pre-tokenized, in [0, vocab_size)
                    "gold_sql":     ex["gold_sql"],
                    "db_id":        ex.get("db_id", "smoke"),
                    "schema_path":  ex.get("schema_path", smoke_db),
                })
    return examples


# ---------------------------------------------------------------------------
# Per-example scoring
# ---------------------------------------------------------------------------

@dataclass
class ExResult:
    correct:    bool
    has_format: bool
    exec_ok:    bool    # generated SQL executed without error
    reward:     float
    generated:  str     # raw model response


def _score_example(response: str, gold_sql: str, schema_path: str) -> ExResult:
    r = reward_sql(response, gold_sql, schema_path)
    return ExResult(
        correct    = r >= REWARD_CORRECT,
        has_format = has_well_formed_sql(response),
        exec_ok    = r >= 0.3,  # REWARD_WRONG or REWARD_CORRECT → executed OK
        reward     = r,
        generated  = response,
    )


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    ex_accuracy:  float   # execution accuracy (primary metric)
    format_rate:  float   # fraction with well-formed <sql> tags
    valid_rate:   float   # fraction where generated SQL executed
    mean_reward:  float   # mean 4-level reward
    n_examples:   int


def _aggregate(results: list[ExResult]) -> EvalMetrics:
    n = len(results)
    return EvalMetrics(
        ex_accuracy = sum(r.correct    for r in results) / n,
        format_rate = sum(r.has_format for r in results) / n,
        valid_rate  = sum(r.exec_ok    for r in results) / n,
        mean_reward = sum(r.reward     for r in results) / n,
        n_examples  = n,
    )


# ---------------------------------------------------------------------------
# Single checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    ckpt_path: str,
    examples:  list[dict],
    cfg:       BaseModelConfig,
    args:      argparse.Namespace,
) -> EvalMetrics:
    """Load a checkpoint, run inference on all examples, return metrics."""
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    print(f"\n  Loading {os.path.basename(ckpt_path)} …", flush=True)
    model = load_backbone_from_ckpt(cfg, ckpt_path, device)
    model.eval()
    raw = unwrap(model)

    results: list[ExResult] = []
    t0 = time.perf_counter()

    for i, ex in enumerate(examples):
        response = generate_sql_response(
            raw,
            ex["user_content"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            prompt_ids=ex.get("prompt_ids"),
        )
        result = _score_example(response, ex["gold_sql"], ex["schema_path"])
        results.append(result)

        if args.verbose and i < args.show_n:
            _print_example(i, ex, result)

        if (i + 1) % 50 == 0 or (i + 1) == len(examples):
            elapsed = time.perf_counter() - t0
            running = _aggregate(results)
            print(f"  [{i+1:4d}/{len(examples)}] "
                  f"EX {running.ex_accuracy:.1%}  "
                  f"fmt {running.format_rate:.1%}  "
                  f"valid {running.valid_rate:.1%}  "
                  f"({elapsed:.0f}s)", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return _aggregate(results)


def _print_example(idx: int, ex: dict, res: ExResult) -> None:
    print(f"\n  ── example {idx} ──────────────────────────────")
    print(f"  Question : {ex['user_content'].split('Question:')[-1].strip()}")
    print(f"  Gold SQL : {ex['gold_sql']}")
    gen_sql = extract_sql(res.generated) or "(none)"
    print(f"  Gen SQL  : {gen_sql}")
    print(f"  Correct  : {'✓' if res.correct else '✗'}  reward={res.reward:.1f}")


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _print_table(rows: list[tuple[str, EvalMetrics]]) -> None:
    print("\n" + "=" * 72)
    print(f"{'checkpoint':35s}  {'EX%':>6}  {'fmt%':>6}  {'valid%':>6}  {'reward':>6}  n")
    print("-" * 72)
    for name, m in rows:
        print(f"{name:35s}  {m.ex_accuracy:6.1%}  {m.format_rate:6.1%}  "
              f"{m.valid_rate:6.1%}  {m.mean_reward:6.3f}  {m.n_examples}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt",           action="append", default=[],
                   metavar="PATH",
                   help="Checkpoint path(s) to evaluate (repeat for comparison)")
    p.add_argument("--config",         default=None,
                   help="Optional JSON config for architecture overrides")
    p.add_argument("--spider_dir",     default="/content/data/spider",
                   help="Root Spider directory (contains dev.json + database/)")
    p.add_argument("--split",          default="dev",
                   help="Spider split filename stem (default: dev → dev.json)")
    p.add_argument("--max_examples",   type=int, default=None,
                   help="Cap number of examples (for quick checks; default: all)")
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Max generation tokens per example (default: 256)")
    p.add_argument("--temperature",    type=float, default=0.0,
                   help="Sampling temperature (0=greedy, default: 0)")
    p.add_argument("--top_p",          type=float, default=1.0)
    p.add_argument("--verbose",        action="store_true",
                   help="Print first --show_n decoded examples per checkpoint")
    p.add_argument("--show_n",         type=int, default=3,
                   help="Number of examples to print when --verbose (default: 3)")
    p.add_argument("--out_json",       default=None,
                   help="Optional path to write results as JSON")
    p.add_argument("--smoke",          action="store_true",
                   help="Evaluate on the tiny smoke SQLite (no Spider download)")
    p.add_argument("--smoke_db",       default="/tmp/sql_lm_smoke/smoke.sqlite",
                   help="Path to smoke SQLite file (default: /tmp/sql_lm_smoke/smoke.sqlite)")
    p.add_argument("--smoke_prompts",  default="/tmp/sql_lm_smoke/sql_prompts_train.jsonl",
                   help="Pre-tokenized smoke JSONL from prepare_rl_prompts.py --smoke")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if not args.ckpt:
        raise SystemExit("Provide at least one --ckpt PATH to evaluate.")

    # --- Load architecture config ---
    cfg = load_config(
        BaseModelConfig,
        args.config or ("configs/smoke/base.json" if args.smoke else None),
    )
    # Force CPU for smoke (no GPU needed) and when CUDA is absent.
    if args.smoke or not torch.cuda.is_available():
        object.__setattr__(cfg, "device",   "cpu")
        object.__setattr__(cfg, "amp_dtype", None)

    # --- Load examples ---
    if args.smoke:
        for path, label in [(args.smoke_db, "smoke SQLite"), (args.smoke_prompts, "smoke prompts JSONL")]:
            if not os.path.exists(path):
                raise SystemExit(
                    f"{label} not found at {path!r}.\n"
                    "Run: python scripts/prepare_rl_prompts.py --smoke --out_dir /tmp/sql_lm_smoke"
                )
        examples = _load_smoke_examples(args.smoke_db, args.smoke_prompts)
        print(f"Smoke mode: {len(examples)} examples from {args.smoke_db}")
    else:
        print(f"Loading Spider {args.split} split from {args.spider_dir} …")
        examples = _load_spider_examples(args.spider_dir, args.split)
        print(f"  {len(examples)} examples loaded")

    if args.max_examples is not None:
        examples = examples[: args.max_examples]
        print(f"  Capped to {len(examples)} examples")

    if not examples:
        raise SystemExit("No examples to evaluate.")

    # --- Evaluate each checkpoint ---
    rows: list[tuple[str, EvalMetrics]] = []
    all_results: dict[str, dict] = {}

    for ckpt_path in args.ckpt:
        label = os.path.basename(ckpt_path)
        print(f"\n{'─'*60}")
        print(f"Evaluating: {ckpt_path}")
        metrics = evaluate_checkpoint(ckpt_path, examples, cfg, args)
        rows.append((label, metrics))
        all_results[label] = {
            "ex_accuracy":  metrics.ex_accuracy,
            "format_rate":  metrics.format_rate,
            "valid_rate":   metrics.valid_rate,
            "mean_reward":  metrics.mean_reward,
            "n_examples":   metrics.n_examples,
        }

    # --- Print comparison table ---
    _print_table(rows)

    # --- Optional JSON output ---
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nResults written to {args.out_json}")


if __name__ == "__main__":
    main()
