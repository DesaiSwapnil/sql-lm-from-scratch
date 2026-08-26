# Adapted from FareedKhan-dev/train-llm-from-scratch (MIT License).
# Original: https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/config/loader.py
# Changes: none to the core logic; docstrings trimmed.
"""
JSON config loader for the SQL-LM training stages.

Merges four layers (lowest precedence first):
    1. dataclass field defaults
    2. configs/base.json
    3. the stage JSON (configs/sft.json, ...)
    4. CLI --field overrides

When json_path lives in a sub-dir with its own base.json (e.g. configs/smoke/sft.json),
that sibling base.json is used automatically.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields
from typing import Any


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _resolve_base(json_path: str | None, base_path: str | None) -> str:
    if base_path is not None:
        return base_path
    if json_path:
        sibling = os.path.join(os.path.dirname(json_path), "base.json")
        if os.path.exists(sibling):
            return sibling
    return "configs/base.json"


def load_config(
    cfg_cls,
    json_path: str | None = None,
    overrides: dict[str, Any] | None = None,
    *,
    base_path: str | None = None,
):
    """Resolve cfg_cls from base.json + stage JSON + CLI overrides."""
    base = _resolve_base(json_path, base_path)
    merged: dict[str, Any] = {}
    for path in (base, json_path):
        if path and os.path.exists(path):
            with open(path) as fh:
                _deep_merge(merged, json.load(fh))

    field_names = {f.name for f in fields(cfg_cls)}
    for key in list(merged):
        if key not in field_names:
            print(f"[config] ignoring unknown key '{key}' for {cfg_cls.__name__}")
            merged.pop(key)

    if overrides:
        merged.update({k: v for k, v in overrides.items() if k in field_names and v is not None})

    return cfg_cls(**merged)
