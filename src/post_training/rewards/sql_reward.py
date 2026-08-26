"""
SQL-execution verifier reward — 4-level graduated scheme.

Reward levels (discrete, not additive):
    1.0  SQL executes and result set matches gold (correct answer).
    0.3  SQL executes without error but result differs from gold (valid but wrong).
    0.1  <sql>…</sql> tags present but SQL fails to execute (syntax/runtime error).
    0.0  No well-formed <sql>…</sql> block found.

Rationale for 4 levels vs. binary:
    GRPO advantage = (r_i - mean_group) / std_group.  When all rollouts score 0
    (binary, nothing correct) std=0 and the gradient vanishes — wasted compute.
    The 0.3 vs 0.1 split keeps σ > 0 on hard-prompt groups where nothing is yet
    correct but some rollouts at least produce executable SQL.  The 0.3→1.0 gap
    (0.7) is large enough that correct completions always dominate group advantage
    even when they're a minority.

Execution sandbox:
    Each call opens a read-only SQLite connection (uri mode=ro), enforced by
    SQLite's VFS layer — any write, ATTACH, or DDL raises OperationalError before
    execution.  The query runs in a daemon thread capped at EXEC_TIMEOUT seconds;
    conn.interrupt() is called on timeout to abort quickly.

Result comparison:
    Both result sets are normalised to sorted lists of string-valued row tuples.
    Numeric cells coerced to canonical integer string when integral (1.0 → "1").
    Row order ignored unless the outermost query has ORDER BY (detected via
    parenthesis-depth traversal so subquery ORDER BYs don't count).
"""

from __future__ import annotations

import concurrent.futures
import re
import sqlite3
from typing import Any

from src.post_training.rewards.parsing import extract_sql, has_well_formed_sql

# 4-level reward values (approved design).
REWARD_CORRECT = 1.0   # executes, result matches gold
REWARD_WRONG   = 0.3   # executes without error, result differs
REWARD_FORMAT  = 0.1   # has <sql> tags but SQL fails to execute
REWARD_NONE    = 0.0   # no <sql> tags

# Legacy aliases kept so existing imports of FORMAT_BONUS / CORRECT_BONUS don't break.
CORRECT_BONUS = REWARD_CORRECT
FORMAT_BONUS  = REWARD_FORMAT
REWARD_CLIP   = REWARD_CORRECT   # no clipping needed; max is exactly 1.0

EXEC_TIMEOUT = 5.0    # seconds per SQL execution
MAX_ROWS     = 500    # cap fetched rows to prevent OOM on runaway queries


# ---------------------------------------------------------------------------
# ORDER BY detection
# ---------------------------------------------------------------------------

def _has_order_by(sql: str) -> bool:
    """True if the outermost SELECT carries an ORDER BY clause."""
    depth = 0
    i = 0
    while i < len(sql):
        ch = sql[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0 and re.match(r'ORDER\s+BY\b', sql[i:], re.IGNORECASE):
            return True
        i += 1
    return False


# ---------------------------------------------------------------------------
# Statement filter
# ---------------------------------------------------------------------------

def _is_select(sql: str) -> bool:
    """Accept only SELECT statements and CTEs (WITH ... SELECT ...)."""
    s = sql.strip()
    return bool(re.match(r'\s*(SELECT|WITH)\b', s, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Result normalisation
# ---------------------------------------------------------------------------

def _cell_str(v: Any) -> str:
    """Canonical string for one cell value."""
    if v is None:
        return "NULL"
    try:
        f = float(v)
        # Map 1.0 → "1", 3.14 → "3.14" etc.
        if f == int(f) and abs(f) < 1e15:
            return str(int(f))
        return f"{f:.10g}"
    except (ValueError, TypeError):
        return str(v).strip()


def _normalize(rows: list[tuple], ordered: bool) -> list[tuple]:
    str_rows = [tuple(_cell_str(c) for c in row) for row in rows]
    return str_rows if ordered else sorted(str_rows)


# ---------------------------------------------------------------------------
# Sandboxed execution
# ---------------------------------------------------------------------------

def _execute_sql(db_path: str, sql: str) -> list[tuple] | None:
    """
    Execute sql against db_path in a read-only SQLite connection.

    Returns the (capped) result rows, or None on any error or timeout.
    """
    if not _is_select(sql):
        return None

    conn_holder: list[sqlite3.Connection] = []

    def _run() -> list[tuple] | None:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn_holder.append(conn)
            cur = conn.execute(sql)
            rows = cur.fetchmany(MAX_ROWS)
            conn.close()
            return rows
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_run)
        try:
            return fut.result(timeout=EXEC_TIMEOUT)
        except concurrent.futures.TimeoutError:
            if conn_holder:
                try:
                    conn_holder[0].interrupt()
                except Exception:
                    pass
            return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reward_sql(text: str, gold_sql: str, schema_path: str) -> float:
    """
    Compute the 4-level SQL execution reward for a model response.

    Args:
        text:        decoded model response (may contain <think>…</think><sql>…</sql>).
        gold_sql:    the ground-truth SQL query for this example.
        schema_path: path to the SQLite database file for execution.

    Returns:
        One of {REWARD_NONE, REWARD_FORMAT, REWARD_WRONG, REWARD_CORRECT}.
    """
    # Level 0: no <sql> tags.
    if not has_well_formed_sql(text):
        return REWARD_NONE

    gen_sql = extract_sql(text)   # guaranteed non-None here

    # Try to execute the generated SQL.
    gen_rows = _execute_sql(schema_path, gen_sql)
    if gen_rows is None:
        # Level 1: tags present but SQL fails to execute.
        return REWARD_FORMAT

    # Try to execute the gold SQL (bad gold → can't score).
    gold_rows = _execute_sql(schema_path, gold_sql)
    if gold_rows is None:
        return REWARD_NONE

    # Level 2 vs 3: compare result sets.
    ordered = _has_order_by(gold_sql)
    if _normalize(gen_rows, ordered) == _normalize(gold_rows, ordered):
        return REWARD_CORRECT   # 1.0
    return REWARD_WRONG         # 0.3


def sql_is_correct(text: str, gold_sql: str, schema_path: str) -> bool:
    """Whether the generated SQL produces the same result set as gold_sql."""
    return reward_sql(text, gold_sql, schema_path) >= REWARD_CORRECT
