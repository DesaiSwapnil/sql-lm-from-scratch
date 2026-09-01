# Adapted from FareedKhan-dev/train-llm-from-scratch (MIT License).
# Original: https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/utils.py
# Changes: added Drive sync to save_stage_ckpt; updated build_model_from_config for our
# architecture (passes swiglu_hidden; imports our Transformer, not theirs).
"""
Shared helpers: model construction, frozen-copy creation, checkpoint I/O with
optional Google Drive sync, masked reductions, and seeding.
"""

from __future__ import annotations

import contextlib
import copy
import os
import random
import shutil
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


_AMP_RESOLVE_LOGGED = False


def resolve_amp_dtype(amp_dtype: str | None, device: str) -> str | None:
    """Map a requested AMP dtype to what this GPU can actually run.

    Returns ``"bf16"``, ``"fp16"``, or ``None`` (AMP off).

    Turing GPUs (T4, compute 7.5) have no native bf16 tensor cores.
    ``torch.cuda.is_bf16_supported()`` is False there; requesting bf16 would
    still construct a bfloat16 autocast, but matmuls are emulated and some
    kernels are unreliable. We fall back to fp16, which needs a GradScaler.
    Ampere+ (A100, L4) keep bf16 and skip the scaler.
    """
    global _AMP_RESOLVE_LOGGED
    raw = amp_dtype
    requested = (amp_dtype or "").strip().lower() or None
    if requested in (None, "none", "off", "fp32"):
        return None

    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if not cuda:
        if requested in ("bf16", "fp16", "auto") and not _AMP_RESOLVE_LOGGED:
            print(f"[amp] CUDA not available; disabling AMP (requested {raw!r})")
            _AMP_RESOLVE_LOGGED = True
        return None

    bf16_ok = torch.cuda.is_bf16_supported()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.device_count() else device

    if requested == "auto":
        chosen = "bf16" if bf16_ok else "fp16"
    elif requested == "bf16":
        if bf16_ok:
            chosen = "bf16"
        else:
            chosen = "fp16"
            if not _AMP_RESOLVE_LOGGED:
                print(
                    f"[amp] bf16 requested but not natively supported on {gpu} "
                    f"(compute capability < 8.0); falling back to fp16 + GradScaler"
                )
                _AMP_RESOLVE_LOGGED = True
    elif requested == "fp16":
        chosen = "fp16"
    else:
        raise ValueError(
            f"Unknown amp_dtype {amp_dtype!r}; expected bf16, fp16, auto, or none"
        )

    if not _AMP_RESOLVE_LOGGED:
        cap = torch.cuda.get_device_capability(0)
        print(
            f"[amp] using {chosen} on {gpu} (cc {cap[0]}.{cap[1]}, "
            f"bf16_supported={bf16_ok}, requested={raw!r})"
        )
        _AMP_RESOLVE_LOGGED = True
    return chosen


def amp_autocast(amp_dtype: str | None, device: str):
    """Autocast context for the resolved AMP dtype (bf16, fp16, or no-op)."""
    resolved = resolve_amp_dtype(amp_dtype, device)
    if resolved == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if resolved == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(amp_dtype: str | None, device: str) -> torch.amp.GradScaler:
    """GradScaler is required for fp16 AMP; bf16 and fp32 skip it."""
    enabled = resolve_amp_dtype(amp_dtype, device) == "fp16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Model construction ------------------------------------------------------

def _cfg_get(cfg: Any, key: str) -> Any:
    if isinstance(cfg, dict):
        return cfg[key]
    return getattr(cfg, key)


def build_model_from_config(cfg: Any):
    """Construct a fresh Transformer from a config carrying the standard architecture keys."""
    from src.models.transformer import Transformer
    return Transformer(
        vocab_size=_cfg_get(cfg, "vocab_size"),
        n_embed=_cfg_get(cfg, "n_embed"),
        n_head=_cfg_get(cfg, "n_head"),
        n_blocks=_cfg_get(cfg, "n_blocks"),
        context_length=_cfg_get(cfg, "context_length"),
        swiglu_hidden=_cfg_get(cfg, "swiglu_hidden"),
    )


def _strip_ddp_prefix(state_dict: dict) -> dict:
    if any(k.startswith("module.") for k in state_dict):
        return {k.removeprefix("module."): v for k, v in state_dict.items()}
    return state_dict


def load_backbone_from_ckpt(cfg: Any, ckpt_path: str, device: str):
    """Build a Transformer from cfg and load weights from a checkpoint."""
    model = build_model_from_config(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    state = _strip_ddp_prefix(state)
    backbone_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state.items() if k in backbone_keys}
    missing, _ = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"[load_backbone] {len(missing)} missing keys (e.g. {missing[:3]})")
    return model.to(device)


def unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module behind a DDP wrapper (or the model itself)."""
    return model.module if hasattr(model, "module") else model


def make_frozen_copy(model: nn.Module, device: str | None = None) -> nn.Module:
    """Deep-copy a model, put it in eval mode, and freeze all gradients."""
    ref = copy.deepcopy(unwrap(model))
    if device is not None:
        ref = ref.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


# --- Masked reductions -------------------------------------------------------

def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def masked_mean_per_row(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)


def gather_last(values: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
    """Return values[i, seq_lengths[i]-1] for each row i."""
    idx = (seq_lengths - 1).clamp(min=0).long()
    return values[torch.arange(values.size(0), device=values.device), idx]


# --- Checkpoint I/O ----------------------------------------------------------

def _cfg_to_dict(cfg: Any) -> Any:
    return asdict(cfg) if is_dataclass(cfg) else cfg


def sync_to_drive(local_path: str, drive_dir: str) -> None:
    """Copy a checkpoint to Google Drive if drive_dir is set and the mount is accessible."""
    if not drive_dir:
        return
    try:
        os.makedirs(drive_dir, exist_ok=True)
    except OSError:
        print(f"[sync] Drive not mounted or inaccessible: {drive_dir!r} — skipping.")
        return
    dest = os.path.join(drive_dir, os.path.basename(local_path))
    shutil.copy2(local_path, dest)
    print(f"[sync] Drive <- {dest}")


def save_stage_ckpt(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    stage: str,
    cfg: Any,
    step: int,
    metrics: dict | None = None,
    extra: dict | None = None,
    drive_dir: str | None = None,
) -> None:
    """
    Save a checkpoint locally and optionally sync it to Google Drive.

    The checkpoint shape is: {model_state_dict, optimizer_state_dict, stage, cfg, step, metrics}.
    DDP wrappers are unwrapped so checkpoints load cleanly on any world_size.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "stage": stage,
        "cfg": _cfg_to_dict(cfg),
        "step": step,
        "metrics": metrics or {},
        "pytorch_version": torch.__version__,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    if drive_dir:
        sync_to_drive(path, drive_dir)
