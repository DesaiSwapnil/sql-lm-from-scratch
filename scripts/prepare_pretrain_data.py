"""
Download, tokenize, and pack pretraining data into flat-token HDF5 files.

Data sources:
    FineWeb-Edu sample-10BT   — quality-filtered web text (public, no auth required).
    The Stack dedup (Python)  — deduplicated Python source files.
    The Stack dedup (SQL)     — deduplicated SQL files.

Tokenizer: tiktoken cl100k_base.  EOT token (100257) is appended after every document
to mark boundaries; the model sees these during pretraining.

HDF5 output layout:
    file['tokens']  shape (N,), dtype int32, flat concatenation of all document tokens.

The data_loader reads windows of context_length + 1 tokens at stride context_length.

Token budget:
    Default produces ~2.6B training tokens and ~50M dev tokens.
    At batch_size=16, grad_accum=8, context_length=1024:
        tokens/step = 16 × 8 × 1024 = 131,072
        2.6B / 131,072 ≈ 19,836 steps  (maps to train_steps=20000 in configs/pretrain.json)

Data mix (defaults):
    85% FineWeb-Edu (general text)  → 2.21B train tokens
    15% The Stack Python/SQL        → 0.39B train tokens

Usage:
    # Full run on Colab:
    PYTHONPATH=. HF_HOME=/content/hf_cache python scripts/prepare_pretrain_data.py \
        --out_dir /content/data

    # Smoke mode (no download, random tokens, ~1 second):
    PYTHONPATH=. python scripts/prepare_pretrain_data.py --smoke --out_dir /tmp
"""

from __future__ import annotations

import argparse
import os
import time

import h5py
import numpy as np
import tiktoken
from tqdm import tqdm

# cl100k_base vocabulary boundary and EOT token.
_BPE_VOCAB = 100256    # regular BPE token ids: 0 .. 100255
EOT_ID = 100257        # <|endoftext|> in cl100k_base

WRITE_CHUNK = 8_000_000   # flush to HDF5 every ~8M tokens (~32MB at int32)
ENCODE_BATCH = 512        # documents per tiktoken batch-encode call

# The Stack: these are the data_dir paths on HuggingFace.
# Note: bigcode/the-stack-dedup may require accepting a license at
#   https://huggingface.co/datasets/bigcode/the-stack-dedup
# before it can be streamed. Run `huggingface-cli login` if you hit a 401.
_STACK_LANGS = [
    ("data/python", "content"),
    ("data/sql",    "content"),
]


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _get_enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _tokenize_batch(enc: tiktoken.Encoding, texts: list[str]) -> list[int]:
    """Encode a list of texts (ordinary tokens only), appending EOT after each."""
    out: list[int] = []
    for ids in enc.encode_ordinary_batch(texts):
        out.extend(ids)
        out.append(EOT_ID)
    return out


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _open_h5(path: str) -> tuple[h5py.File, h5py.Dataset]:
    """Open (or create) a resizable flat-token HDF5 dataset."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = h5py.File(path, "a")
    if "tokens" not in f:
        ds = f.create_dataset(
            "tokens", shape=(0,), maxshape=(None,),
            dtype="i4", chunks=(WRITE_CHUNK,),
        )
    else:
        ds = f["tokens"]
    return f, ds


def _flush(ds: h5py.Dataset, buf: list[int]) -> int:
    """Append buf to ds; return number of tokens flushed."""
    if not buf:
        return 0
    arr = np.asarray(buf, dtype=np.int32)
    old = ds.shape[0]
    ds.resize(old + arr.size, axis=0)
    ds[old : old + arr.size] = arr
    return arr.size


# ---------------------------------------------------------------------------
# Source iterators
# ---------------------------------------------------------------------------

def _iter_fineweb(hf_cache: str):
    """Stream text from FineWeb-Edu sample-10BT."""
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        cache_dir=hf_cache,
    )
    for ex in ds:
        txt = ex.get("text", "")
        if txt:
            yield txt


def _iter_stack(lang_dir: str, text_field: str, hf_cache: str, max_doc_tokens: int):
    """Stream code from bigcode/the-stack-dedup for one language."""
    from datasets import load_dataset
    enc = _get_enc()
    ds = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir=lang_dir,
        split="train",
        streaming=True,
        cache_dir=hf_cache,
    )
    for ex in ds:
        content = ex.get(text_field, "")
        if not content:
            continue
        # Skip files that are extremely long — likely minified or generated code.
        if len(content) > max_doc_tokens * 4:  # rough char estimate before tokenising
            continue
        ids = enc.encode_ordinary(content)
        if len(ids) > max_doc_tokens:
            continue
        yield content


# ---------------------------------------------------------------------------
# Core pipeline: stream → tokenise → write HDF5
# ---------------------------------------------------------------------------

def _stream_to_h5(
    source,                  # iterator over text strings
    ds: h5py.Dataset,
    enc: tiktoken.Encoding,
    token_target: int,
    desc: str,
) -> int:
    """
    Stream documents from source, tokenise, and append to ds until token_target is reached.

    Returns the number of tokens written.
    """
    buf: list[int] = []
    written = 0
    batch: list[str] = []

    pbar = tqdm(source, desc=desc, unit="doc", dynamic_ncols=True)
    for text in pbar:
        batch.append(text)
        if len(batch) >= ENCODE_BATCH:
            buf.extend(_tokenize_batch(enc, batch))
            batch = []
            if len(buf) >= WRITE_CHUNK:
                n = _flush(ds, buf)
                written += n
                buf = []
                pbar.set_postfix(tokens=f"{written / 1e6:.0f}M / {token_target / 1e6:.0f}M")
        if written + len(buf) >= token_target:
            break

    # Flush remaining batch and buffer.
    if batch:
        buf.extend(_tokenize_batch(enc, batch))
    n = _flush(ds, buf)
    written += n
    return written


# ---------------------------------------------------------------------------
# Smoke mode
# ---------------------------------------------------------------------------

def _make_smoke_h5(path: str, n_tokens: int = 200_000, vocab_size: int = 512) -> None:
    """Write a tiny random-token HDF5 for smoke-testing the training loop."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tokens = np.random.randint(0, vocab_size, size=n_tokens, dtype=np.int32)
    with h5py.File(path, "w") as f:
        f.create_dataset("tokens", data=tokens, dtype="i4")
    print(f"[smoke] wrote {n_tokens:,} random tokens -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="/content/data",
                   help="Directory for output HDF5 files (default: /content/data)")
    p.add_argument("--hf_cache", default="/content/hf_cache",
                   help="HuggingFace dataset cache directory")
    p.add_argument("--target_train_tokens", type=int, default=2_600_000_000,
                   help="Target number of training tokens (default: 2.6B)")
    p.add_argument("--dev_tokens", type=int, default=50_000_000,
                   help="Tokens reserved for the dev set (default: 50M)")
    p.add_argument("--code_fraction", type=float, default=0.15,
                   help="Fraction of training tokens from The Stack (default: 0.15)")
    p.add_argument("--max_doc_tokens", type=int, default=50_000,
                   help="Skip code documents longer than this after tokenisation (default: 50k)")
    p.add_argument("--smoke", action="store_true",
                   help="Write tiny random HDF5 files for smoke-testing; no download")
    p.add_argument("--smoke_vocab", type=int, default=512,
                   help="Vocab size for smoke random tokens (default: 512)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "pretrain_train.h5")
    dev_path   = os.path.join(args.out_dir, "pretrain_dev.h5")

    # ------------------------------------------------------------------
    # Smoke mode: write random HDF5 files and exit.
    # ------------------------------------------------------------------
    if args.smoke:
        print("=== Smoke mode: writing random token HDF5 files ===")
        # Use a vocab small enough for the smoke model (vocab_size=512 in smoke config).
        _make_smoke_h5(train_path, n_tokens=200_000, vocab_size=args.smoke_vocab)
        _make_smoke_h5(dev_path,   n_tokens=20_000,  vocab_size=args.smoke_vocab)
        print("Done. Run training with --config configs/smoke/pretrain.json")
        return

    # ------------------------------------------------------------------
    # Real mode: stream + tokenise + write.
    # ------------------------------------------------------------------
    enc = _get_enc()
    t0 = time.perf_counter()

    fineweb_tokens = args.dev_tokens + int(args.target_train_tokens * (1 - args.code_fraction))
    code_tokens    = int(args.target_train_tokens * args.code_fraction)

    print(f"Token targets:")
    print(f"  dev (FineWeb-Edu):     {args.dev_tokens / 1e6:.0f}M")
    print(f"  train FineWeb-Edu:     {(fineweb_tokens - args.dev_tokens) / 1e6:.0f}M")
    print(f"  train The Stack (Py+SQL): {code_tokens / 1e6:.0f}M")
    print(f"  total train:           {args.target_train_tokens / 1e6:.0f}M")

    # --- Stage 1: FineWeb-Edu → dev first, then train ---
    print("\n--- Stage 1: FineWeb-Edu ---")
    fineweb_iter = _iter_fineweb(args.hf_cache)

    dev_f, dev_ds = _open_h5(dev_path)
    dev_written = _stream_to_h5(
        fineweb_iter, dev_ds, enc,
        token_target=args.dev_tokens,
        desc="FineWeb-Edu dev",
    )
    dev_f.close()
    print(f"  dev:   {dev_written:,} tokens -> {dev_path}")

    train_f, train_ds = _open_h5(train_path)
    train_written = _stream_to_h5(
        fineweb_iter, train_ds, enc,
        token_target=fineweb_tokens - args.dev_tokens,
        desc="FineWeb-Edu train",
    )
    print(f"  train: {train_written:,} tokens from FineWeb-Edu -> {train_path}")

    # --- Stage 2: The Stack (Python + SQL) → train ---
    print("\n--- Stage 2: The Stack (Python + SQL) ---")
    tokens_per_lang = code_tokens // len(_STACK_LANGS)
    for lang_dir, field in _STACK_LANGS:
        lang = lang_dir.split("/")[-1]
        try:
            stack_iter = _iter_stack(lang_dir, field, args.hf_cache, args.max_doc_tokens)
            n = _stream_to_h5(
                stack_iter, train_ds, enc,
                token_target=tokens_per_lang,
                desc=f"The Stack ({lang})",
            )
            train_written += n
            print(f"  {lang}: {n:,} tokens added")
        except Exception as e:
            print(f"  WARNING: skipping {lang} ({e})")
            print("  Tip: run `huggingface-cli login` if you get a 401 error.")

    train_f.close()

    elapsed = time.perf_counter() - t0
    print(f"\n=== Done in {elapsed/60:.1f} min ===")
    print(f"  pretrain_train.h5: {train_written:,} tokens")
    print(f"  pretrain_dev.h5:   {dev_written:,} tokens")
    print(f"  Train examples at context_length=1024: {(train_written - 1) // 1024:,}")
    print(f"  At 20k steps × 131k tok/step = 2.62B tokens per epoch.")


if __name__ == "__main__":
    main()
