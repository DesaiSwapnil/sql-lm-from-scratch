"""
Build the SFT dataset from general instruction + SQL-specific data.

Adapted from FareedKhan-dev/train-llm-from-scratch (MIT License):
  scripts/prepare_sft_data.py (HDF5 packing, tokenisation, loss-mask logic).

Data sources:
  - tatsu-lab/alpaca              (general instruction following, ~52k examples)
  - databricks/databricks-dolly-15k (general instruction following, ~15k examples)
  - spider (Yale Semantic Parsing) — NL question + schema + gold SQL
  - BIRD                         — NL question + schema + gold SQL (harder)

Each example is formatted as a single-turn chat with encode_chat() and packed
into fixed-length context_length windows. Two parallel HDF5 datasets are written:
  tokens    shape (n_rows, context_length)  dtype int32  — input ids
  loss_mask shape (n_rows, context_length)  dtype uint8  — 1 on assistant tokens

HDF5 layout is consumed by data_loader/sft_dataset.py.

Usage:
    # Full run:
    PYTHONPATH=. HF_HOME=/content/hf_cache python scripts/prepare_sft_data.py \\
        --out_dir /content/data --context_length 1024

    # Smoke mode (no download, instant):
    PYTHONPATH=. python scripts/prepare_sft_data.py --smoke --out_dir /tmp/sql_lm_smoke
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Iterator

import h5py
import numpy as np

from src.post_training.chat_template import SQL_CLOSE, SQL_OPEN, THINK_CLOSE, THINK_OPEN, encode_chat


# ---- SQL formatting helpers ----------------------------------------------------

def _sql_turn(question: str, schema_desc: str, gold_sql: str) -> tuple[str, str]:
    user_content = f"Schema:\n{schema_desc}\n\nQuestion: {question}"
    assistant_content = (
        f"{THINK_OPEN}Let me write the SQL query.{THINK_CLOSE}"
        f"{SQL_OPEN}{gold_sql}{SQL_CLOSE}"
    )
    return user_content, assistant_content


# ---- HDF5 helpers --------------------------------------------------------------

def _open_sft_h5(path: str, context_length: int) -> tuple[h5py.File, h5py.Dataset, h5py.Dataset]:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = h5py.File(path, "w")
    tok_ds  = f.create_dataset("tokens",    shape=(0, context_length), maxshape=(None, context_length),
                                dtype="i4", chunks=(256, context_length))
    mask_ds = f.create_dataset("loss_mask", shape=(0, context_length), maxshape=(None, context_length),
                                dtype="u1", chunks=(256, context_length))
    return f, tok_ds, mask_ds


def _pack_and_flush(
    tok_ds: h5py.Dataset,
    mask_ds: h5py.Dataset,
    ids: list[int],
    mask: list[int],
    context_length: int,
    pad_id: int = 0,
) -> int:
    """Slice ids/mask into context_length rows, pad the final partial row, append to HDF5."""
    rows_tok, rows_mask = [], []
    for start in range(0, len(ids), context_length):
        chunk_ids  = ids [start : start + context_length]
        chunk_mask = mask[start : start + context_length]
        if len(chunk_ids) < context_length:
            pad = context_length - len(chunk_ids)
            chunk_ids  = chunk_ids  + [pad_id] * pad
            chunk_mask = chunk_mask + [0]       * pad
        rows_tok.append(chunk_ids)
        rows_mask.append(chunk_mask)

    if not rows_tok:
        return 0

    arr_tok  = np.array(rows_tok,  dtype=np.int32)
    arr_mask = np.array(rows_mask, dtype=np.uint8)
    n   = len(arr_tok)
    old = tok_ds.shape[0]
    tok_ds.resize(old + n,  axis=0)
    mask_ds.resize(old + n, axis=0)
    tok_ds [old : old + n] = arr_tok
    mask_ds[old : old + n] = arr_mask
    return n


# ---- Data source iterators -----------------------------------------------------

def _iter_alpaca(hf_cache: str) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=hf_cache)
    for ex in ds:
        instr = ex.get("instruction", "").strip()
        inp   = ex.get("input",       "").strip()
        out   = ex.get("output",      "").strip()
        if not instr or not out:
            continue
        user = f"{instr}\n{inp}".strip() if inp else instr
        yield user, out


def _iter_dolly(hf_cache: str) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train", cache_dir=hf_cache)
    for ex in ds:
        instr    = ex.get("instruction", "").strip()
        context  = ex.get("context",     "").strip()
        response = ex.get("response",    "").strip()
        if not instr or not response:
            continue
        user = f"{instr}\n\n{context}".strip() if context else instr
        yield user, response


def _iter_spider(hf_cache: str) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset
    try:
        ds = load_dataset("spider", split="train", cache_dir=hf_cache, trust_remote_code=True)
    except Exception as e:
        print(f"[spider] skipping ({e})")
        return
    for ex in ds:
        question = ex.get("question", "").strip()
        gold_sql = ex.get("query",    "").strip()
        db_id    = ex.get("db_id",    "")
        if not question or not gold_sql:
            continue
        user, asst = _sql_turn(question, f"Database: {db_id}", gold_sql)
        yield user, asst


def _iter_bird(hf_cache: str) -> Iterator[tuple[str, str]]:
    from datasets import load_dataset
    try:
        ds = load_dataset("birdbench/bird", split="train", cache_dir=hf_cache, trust_remote_code=True)
    except Exception as e:
        print(f"[bird] skipping ({e})")
        return
    for ex in ds:
        question = ex.get("question", "").strip()
        gold_sql = ex.get("SQL", ex.get("query", "")).strip()
        db_id    = ex.get("db_id", "")
        if not question or not gold_sql:
            continue
        user, asst = _sql_turn(question, f"Database: {db_id}", gold_sql)
        yield user, asst


# ---- Packing pipeline ----------------------------------------------------------

def _examples_to_h5(
    examples: list[tuple[str, str]],
    tok_ds: h5py.Dataset,
    mask_ds: h5py.Dataset,
    context_length: int,
    *,
    label: str,
) -> int:
    total_rows = 0
    for user, assistant in examples:
        ids, mask = encode_chat([
            {"role": "user",      "content": user},
            {"role": "assistant", "content": assistant},
        ])
        total_rows += _pack_and_flush(tok_ds, mask_ds, ids, mask, context_length)
    print(f"  [{label}] {len(examples):,} examples -> {total_rows:,} rows")
    return total_rows


# ---- Smoke mode ----------------------------------------------------------------

def _make_smoke_sft_h5(path: str, context_length: int, n_rows: int) -> None:
    """Write tiny random SFT HDF5 for smoke-testing. Loss mask on the last half."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rng = np.random.default_rng(42)
    tokens    = rng.integers(0, 512, size=(n_rows, context_length), dtype=np.int32)
    loss_mask = np.zeros((n_rows, context_length), dtype=np.uint8)
    loss_mask[:, context_length // 2 :] = 1
    with h5py.File(path, "w") as f:
        f.create_dataset("tokens",    data=tokens)
        f.create_dataset("loss_mask", data=loss_mask)
    print(f"[smoke] {n_rows} rows -> {path}")


# ---- Main ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir",        default="/content/data")
    p.add_argument("--hf_cache",       default="/content/hf_cache")
    p.add_argument("--context_length", type=int,   default=1024)
    p.add_argument("--dev_fraction",   type=float, default=0.05,
                   help="Fraction held out for dev (default: 0.05)")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--smoke",          action="store_true",
                   help="Write tiny random HDF5 files for smoke-testing; no download")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "sft_train.h5")
    dev_path   = os.path.join(args.out_dir, "sft_dev.h5")

    if args.smoke:
        print("=== Smoke mode: writing random SFT HDF5 files ===")
        # Smoke model context_length is 64 (configs/smoke/base.json).
        ctx_len = args.context_length if args.context_length != 1024 else 64
        print(f"  context_length={ctx_len}")
        _make_smoke_sft_h5(train_path, ctx_len, n_rows=200)
        _make_smoke_sft_h5(dev_path,   ctx_len, n_rows=20)
        print("Done.")
        return

    # --- Collect all (user, assistant) pairs, tracking per-source counts ---
    rng = random.Random(args.seed)
    all_examples: list[tuple[str, str]] = []
    source_counts: dict[str, int] = {}

    for name, iterator in [
        ("alpaca", _iter_alpaca(args.hf_cache)),
        ("dolly",  _iter_dolly(args.hf_cache)),
        ("spider", _iter_spider(args.hf_cache)),
        ("bird",   _iter_bird(args.hf_cache)),
    ]:
        print(f"Collecting {name} …")
        before = len(all_examples)
        all_examples.extend(list(iterator))
        source_counts[name] = len(all_examples) - before

    # --- Per-source summary — loud warning when SQL sources return nothing ---
    sql_sources = {"spider", "bird"}
    print("\n=== Source counts ===")
    sql_total = 0
    for name, count in source_counts.items():
        tag = ""
        if name in sql_sources:
            sql_total += count
            if count == 0:
                tag = "  *** WARNING: 0 examples — SQL data missing from this source ***"
        print(f"  {name:8s}: {count:6,}{tag}")
    print(f"  {'TOTAL':8s}: {len(all_examples):6,}")
    if sql_total == 0:
        print("\n  *** CRITICAL: Both Spider and BIRD returned 0 examples.")
        print("  *** The SFT dataset will contain NO SQL data.")
        print("  *** The model will NOT learn SQL generation from this checkpoint.")
        print("  *** Check HuggingFace access and run `huggingface-cli login` if needed.")
    print()

    rng.shuffle(all_examples)

    n_dev          = max(1, int(len(all_examples) * args.dev_fraction))
    dev_examples   = all_examples[:n_dev]
    train_examples = all_examples[n_dev:]

    # --- Pack into HDF5 ---
    print(f"Packing train ({len(train_examples):,} examples) -> {train_path}")
    tr_f, tr_tok, tr_mask = _open_sft_h5(train_path, args.context_length)
    train_rows = _examples_to_h5(train_examples, tr_tok, tr_mask, args.context_length, label="train")
    tr_f.close()

    print(f"\nPacking dev ({len(dev_examples):,} examples) -> {dev_path}")
    dv_f, dv_tok, dv_mask = _open_sft_h5(dev_path, args.context_length)
    dev_rows = _examples_to_h5(dev_examples, dv_tok, dv_mask, args.context_length, label="dev")
    dv_f.close()

    print("\n=== Done ===")
    print(f"  train: {train_rows:,} rows  ({train_path})")
    print(f"  dev:   {dev_rows:,} rows  ({dev_path})")


if __name__ == "__main__":
    main()
