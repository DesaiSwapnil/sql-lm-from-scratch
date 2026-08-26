"""
Build GRPO prompt JSONL files from Spider/BIRD SQL datasets.

Each output line:
  {"prompt":      "<user message content — schema + question>",
   "gold_sql":    "SELECT ...",
   "db_id":       "chinook",
   "schema_path": "/content/data/spider_dbs/chinook/chinook.sqlite"}

The "prompt" value is the raw user-message content string; train_grpo.py wraps it
in the chat template and tokenises it at iteration time.

Usage:
    # Full run (requires Spider SQLite databases on disk):
    PYTHONPATH=. python scripts/prepare_rl_prompts.py \\
        --spider_dir /content/data/spider \\
        --out_dir /content/data

    # Smoke (no download — creates tiny SQLite + fake prompts):
    PYTHONPATH=. python scripts/prepare_rl_prompts.py --smoke --out_dir /tmp/sql_lm_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3


# ---------------------------------------------------------------------------
# Smoke helpers
# ---------------------------------------------------------------------------

_SMOKE_SCHEMA = """
CREATE TABLE users  (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL);
INSERT INTO users  VALUES (1,'Alice',30),(2,'Bob',25),(3,'Charlie',35);
INSERT INTO orders VALUES (1,1,49.99),(2,1,19.99),(3,2,99.00);
"""

_SMOKE_EXAMPLES: list[tuple[str, str]] = [
    ("How many users are there?",                    "SELECT COUNT(*) FROM users"),
    ("List all user names.",                         "SELECT name FROM users"),
    ("What is the total order amount?",              "SELECT SUM(amount) FROM orders"),
    ("Which user has the most orders?",              "SELECT user_id FROM orders GROUP BY user_id ORDER BY COUNT(*) DESC LIMIT 1"),
    ("How many orders does Alice have?",             "SELECT COUNT(*) FROM orders WHERE user_id=1"),
    ("What is the average age of users?",            "SELECT AVG(age) FROM users"),
    ("List users older than 28.",                    "SELECT name FROM users WHERE age > 28"),
    ("What is the maximum order amount?",            "SELECT MAX(amount) FROM orders"),
]

_SMOKE_SCHEMA_DESC = (
    "Table: users\n  users.id\n  users.name\n  users.age\n"
    "Table: orders\n  orders.id\n  orders.user_id\n  orders.amount"
)


def _make_smoke_db(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.executescript(_SMOKE_SCHEMA)
    conn.close()


def _make_smoke_prompts(db_path: str, smoke_vocab: int = 512, prompt_len: int = 10) -> list[dict]:
    """
    Smoke prompts use pre-tokenised integer lists in [0, smoke_vocab) instead of
    text strings, because the smoke model's embedding table only has smoke_vocab
    entries and tiktoken would produce ids >> smoke_vocab for real text.
    """
    import random
    rng = random.Random(0)
    prompts = []
    for _, gold_sql in _SMOKE_EXAMPLES:
        fake_ids = [rng.randint(0, smoke_vocab - 1) for _ in range(prompt_len)]
        prompts.append({
            "prompt":      fake_ids,       # list[int] — signals pre-tokenised
            "gold_sql":    gold_sql,
            "db_id":       "smoke",
            "schema_path": db_path,
        })
    return prompts


# ---------------------------------------------------------------------------
# Spider helpers
# ---------------------------------------------------------------------------

def _schema_desc(db_path: str) -> str:
    """Build a minimal schema description by introspecting the SQLite file."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()]
        lines = []
        for tbl in tables:
            lines.append(f"Table: {tbl}")
            cols = conn.execute(f"PRAGMA table_info({tbl})").fetchall()
            for col in cols:
                lines.append(f"  {tbl}.{col[1]}")
        conn.close()
        return "\n".join(lines)
    except Exception:
        return ""


def _load_spider(spider_dir: str) -> list[dict]:
    """
    Load Spider examples from the standard Spider directory layout:
        spider_dir/
            train_spider.json     — training examples
            dev.json              — dev examples
            database/<db_id>/<db_id>.sqlite
    """
    prompts: list[dict] = []
    for split_file in ("train_spider.json", "train_others.json", "dev.json"):
        fpath = os.path.join(spider_dir, split_file)
        if not os.path.exists(fpath):
            continue
        with open(fpath) as fh:
            examples = json.load(fh)
        for ex in examples:
            db_id    = ex.get("db_id", "")
            question = ex.get("question", "").strip()
            gold_sql = ex.get("query", "").strip()
            if not db_id or not question or not gold_sql:
                continue
            db_path = os.path.join(spider_dir, "database", db_id, f"{db_id}.sqlite")
            if not os.path.exists(db_path):
                continue
            schema  = _schema_desc(db_path)
            user_content = f"Schema:\n{schema}\n\nQuestion: {question}"
            prompts.append({
                "prompt":      user_content,
                "gold_sql":    gold_sql,
                "db_id":       db_id,
                "schema_path": db_path,
            })
    return prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spider_dir", default="/content/data/spider",
                   help="Root of the Spider dataset directory (contains database/ subdir)")
    p.add_argument("--out_dir",    default="/content/data")
    p.add_argument("--dev_fraction", type=float, default=0.1,
                   help="Fraction held out for dev prompts (default: 0.1)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--smoke",      action="store_true",
                   help="Write tiny fake prompts + SQLite for smoke-testing; no download")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "sql_prompts_train.jsonl")
    dev_path   = os.path.join(args.out_dir, "sql_prompts_dev.jsonl")

    if args.smoke:
        print("=== Smoke mode: building fake SQL prompts ===")
        db_path = os.path.join(args.out_dir, "smoke.sqlite")
        _make_smoke_db(db_path)
        print(f"  SQLite schema -> {db_path}")
        prompts = _make_smoke_prompts(db_path)
        # Split: last 2 as dev.
        train_prompts = prompts[:-2]
        dev_prompts   = prompts[-2:]
        for path, split in [(train_path, train_prompts), (dev_path, dev_prompts)]:
            with open(path, "w") as fh:
                for pr in split:
                    fh.write(json.dumps(pr) + "\n")
            print(f"  {len(split)} prompts -> {path}")
        print("Done.")
        return

    # --- Real mode ---
    print("Loading Spider …")
    all_prompts = _load_spider(args.spider_dir)

    print(f"Total prompts: {len(all_prompts):,}")
    if len(all_prompts) == 0:
        print("WARNING: 0 prompts found. Check --spider_dir path.")
        return

    rng = random.Random(args.seed)
    rng.shuffle(all_prompts)

    n_dev          = max(1, int(len(all_prompts) * args.dev_fraction))
    dev_prompts    = all_prompts[:n_dev]
    train_prompts  = all_prompts[n_dev:]

    for path, split, label in [
        (train_path, train_prompts, "train"),
        (dev_path,   dev_prompts,   "dev"),
    ]:
        with open(path, "w") as fh:
            for pr in split:
                fh.write(json.dumps(pr) + "\n")
        print(f"  {label}: {len(split):,} prompts -> {path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
