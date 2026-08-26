# Adapted from FareedKhan-dev/train-llm-from-scratch (MIT License).
# Original: https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/cli.py
# Changes: none to the core logic; updated import path.
"""
Shared CLI helper: turn any stage config dataclass into --field value arguments so
every training script can override hyperparameters without bespoke argparse blocks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from types import SimpleNamespace

from config.loader import load_config


def _add_typed(p: argparse.ArgumentParser, name: str, type_str) -> None:
    t = str(type_str)
    if "int" in t and "|" not in t:
        p.add_argument(f"--{name}", type=int)
    elif "float" in t:
        p.add_argument(f"--{name}", type=float)
    elif t == "bool" or t == "<class 'bool'>":
        p.add_argument(f"--{name}", type=lambda s: s.lower() in ("1", "true", "yes"))
    else:
        p.add_argument(f"--{name}", type=str)


def parse_config(cfg_cls, extra: dict | None = None):
    """Parse CLI overrides for a config dataclass (no JSON file)."""
    p = argparse.ArgumentParser()
    field_names = {f.name for f in fields(cfg_cls)}
    for f in fields(cfg_cls):
        _add_typed(p, f.name, f.type)
    for flag, kwargs in (extra or {}).items():
        p.add_argument(flag, **kwargs)
    args = vars(p.parse_args())
    overrides = {k: v for k, v in args.items() if k in field_names and v is not None}
    extras = {k: v for k, v in args.items() if k not in field_names}
    return cfg_cls(**overrides), SimpleNamespace(**extras)


def parse_config_with_json(cfg_cls, default_json: str, extra: dict | None = None):
    """
    Resolve a stage config from a JSON file plus optional CLI --field overrides.

    Adds two flags:
      --config PATH      : the stage JSON to load (default: default_json).
      --print-config     : print the fully resolved config as JSON and exit.

    Returns (cfg, extras) where extras is a SimpleNamespace of non-config args.
    """
    p = argparse.ArgumentParser()
    field_names = {f.name for f in fields(cfg_cls)}
    for f in fields(cfg_cls):
        _add_typed(p, f.name, f.type)
    p.add_argument("--config", default=default_json, help="stage JSON config to load")
    p.add_argument("--print-config", action="store_true", help="print resolved config and exit")
    for flag, kwargs in (extra or {}).items():
        p.add_argument(flag, **kwargs)
    args = vars(p.parse_args())

    reserved = {"config", "print_config"}
    overrides = {k: v for k, v in args.items() if k in field_names and v is not None}
    extras = {k: v for k, v in args.items() if k not in field_names and k not in reserved}

    cfg = load_config(cfg_cls, args["config"], overrides)
    if args["print_config"]:
        print(json.dumps(asdict(cfg), indent=2))
        raise SystemExit(0)
    return cfg, SimpleNamespace(**extras)
