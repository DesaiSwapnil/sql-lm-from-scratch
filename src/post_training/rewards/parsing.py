"""
Response parsing utilities for the SQL reward verifier.

Extracts SQL queries from model responses formatted as <sql>...</sql> and checks
for well-formed structure. Mirrors the reference's rewards/parsing.py but for SQL.
"""

from __future__ import annotations

import re

# DOTALL so SQL can span multiple lines; IGNORECASE for robustness.
_SQL_RE = re.compile(r"<sql>(.*?)</sql>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str | None:
    """
    Extract the SQL query from a model response containing <sql>...</sql>.

    Returns the SQL string (stripped) if exactly one well-formed block is found,
    else None. Handles multi-line SQL and surrounding whitespace.
    """
    matches = _SQL_RE.findall(text)
    if len(matches) == 1:
        return matches[0].strip()
    return None


def extract_think(text: str) -> str | None:
    """Extract the reasoning chain from <think>...</think>, or None if absent/malformed."""
    matches = _THINK_RE.findall(text)
    if len(matches) == 1:
        return matches[0].strip()
    return None


def has_well_formed_sql(text: str) -> bool:
    """True if the response contains exactly one properly opened and closed <sql>...</sql> block."""
    return extract_sql(text) is not None
