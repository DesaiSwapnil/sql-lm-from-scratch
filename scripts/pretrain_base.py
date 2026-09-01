"""
Pretrain the ~247M base model from scratch on FineWeb-Edu + code.

Adapted from FareedKhan-dev/train-llm-from-scratch (MIT License):
  scripts/pretrain_base.py (training loop skeleton, checkpoint/resume, DDP boilerplate).

Key additions vs. reference:
  - Checkpoints to Google Drive on every save (sync_to_drive in utils.py).
  - Our Transformer (RMSNorm, SwiGLU, RoPE) instead of the reference's GPT-2-style model.
  - cl100k_base tokenizer (vocab=100352) instead of r50k_base.
  - Single checkpoint load on resume (reference loads the file twice).

Single GPU:
    PYTHONPATH=. python scripts/pretrain_base.py
Resume after disconnect:
    PYTHONPATH=. python scripts/pretrain_base.py --resume /content/ckpts/pretrain.pt
Smoke test (CPU, seconds):
    PYTHONPATH=. python scripts/pretrain_base.py --config configs/smoke/pretrain.json
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch

from config.loader import load_config
from config.training_config import PretrainConfig
from data_loader.data_loader import get_batch_iterator
from src.post_training.distributed import (
    DDPContext,
    barrier,
    cleanup,
    ddp_setup,
    ddp_wrap,
    reduce_scalar,
)
from src.post_training.logging_utils import MetricsLogger
from src.post_training.optim import configure_optimizer, cosine_lr
from src.post_training.utils import (
    amp_autocast,
    build_model_from_config,
    make_grad_scaler,
    resolve_amp_dtype,
    save_stage_ckpt,
    set_seed,
    unwrap,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/pretrain.json",
                   help="Path to stage JSON config (default: configs/pretrain.json)")
    p.add_argument("--resume", default=None,
                   help="Checkpoint path to resume from. Pass 'latest' to use cfg.out_ckpt.")
    p.add_argument("--print-config", action="store_true",
                   help="Print resolved config and exit.")
    return p.parse_known_args()


def _apply_cli_overrides(cfg: PretrainConfig, extras: list[str]) -> PretrainConfig:
    """Parse remaining --key value pairs and update cfg in place."""
    it = iter(extras)
    overrides: dict = {}
    for tok in it:
        if tok.startswith("--"):
            key = tok.lstrip("-").replace("-", "_")
            try:
                val_str = next(it)
            except StopIteration:
                raise SystemExit(f"Missing value for flag {tok!r}")
            field_map = {f.name: f for f in cfg.__dataclass_fields__.values()}
            if key not in field_map:
                print(f"[cli] ignoring unknown flag --{key}")
                continue
            current = getattr(cfg, key)
            field_type = type(current) if current is not None else str
            # Dataclass Optional[str] is NoneType when unset; treat as str.
            if key == "amp_dtype":
                field_type = str
            if field_type is bool:
                overrides[key] = val_str.lower() not in ("false", "0", "no")
            elif field_type in (int, float, str):
                overrides[key] = field_type(val_str)
    for k, v in overrides.items():
        object.__setattr__(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, dev_iter, cfg: PretrainConfig, ctx: DDPContext) -> float:
    """Run eval_iters batches on the dev set; return mean loss averaged across ranks."""
    model.eval()
    ac = amp_autocast(cfg.amp_dtype, ctx.device)
    total = 0.0
    for _ in range(cfg.eval_iters):
        xb, yb = next(dev_iter)
        with ac:
            _, loss = model(xb, targets=yb)
        total += loss.item()
    model.train()
    mean_loss = total / cfg.eval_iters
    return reduce_scalar(mean_loss, ctx)


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------

def _load_resume_ckpt(resume_path: str, model, optimizer, device: str) -> int:
    """Load model + optimizer state from a checkpoint. Returns the saved step."""
    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # Strip DDP prefix if present.
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    unwrap(model).load_state_dict(state)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Move optimizer state tensors to the right device.
        for og in optimizer.state.values():
            for k, v in og.items():
                if isinstance(v, torch.Tensor):
                    og[k] = v.to(device)
    start_step = ckpt.get("step", 0)
    print(f"[resume] loaded step={start_step} from {resume_path!r}")
    return start_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args, extras = _parse_args()

    cfg: PretrainConfig = load_config(PretrainConfig, args.config)
    cfg = _apply_cli_overrides(cfg, extras)

    if args.print_config:
        import dataclasses, json
        print(json.dumps(dataclasses.asdict(cfg), indent=2))
        return

    # DDP / device setup.
    ctx = ddp_setup(cfg.device)
    set_seed(cfg.seed + ctx.rank)
    object.__setattr__(cfg, "amp_dtype", resolve_amp_dtype(cfg.amp_dtype, ctx.device))

    if ctx.is_main:
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        if ctx.device.startswith("cuda") and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(
                f"[gpu] {torch.cuda.get_device_name(0)} | "
                f"{total / 1024**3:.1f} GB total, {free / 1024**3:.1f} GB free"
            )
        else:
            print(f"[gpu] device={ctx.device} | CUDA unavailable")
        print(
            f"[amp] resolved={cfg.amp_dtype!r} | "
            f"scaler={cfg.amp_dtype == 'fp16'}"
        )
        print(
            f"[grad_checkpointing] {cfg.grad_checkpointing} | "
            f"batch_size={cfg.batch_size} grad_accum={cfg.grad_accum} "
            f"(effective {cfg.batch_size * cfg.grad_accum})"
        )

    # Build model.
    model = build_model_from_config(cfg).to(ctx.device)

    if cfg.grad_checkpointing:
        unwrap(model).gradient_checkpointing = True

    if cfg.compile:
        model = torch.compile(model)

    model = ddp_wrap(model, ctx)

    optimizer = configure_optimizer(
        unwrap(model),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Resume.
    start_step = 0
    resume_path = args.resume
    if resume_path == "latest":
        resume_path = cfg.out_ckpt
    if resume_path and os.path.exists(resume_path):
        start_step = _load_resume_ckpt(resume_path, model, optimizer, ctx.device)
    elif resume_path:
        print(f"[resume] checkpoint not found at {resume_path!r}; starting from scratch")

    # Data iterators — all ranks open the same file (read-only, safe).
    train_iter = get_batch_iterator(cfg.train_path, cfg.batch_size, cfg.context_length, ctx.device)
    dev_iter   = get_batch_iterator(cfg.dev_path,   cfg.batch_size, cfg.context_length, ctx.device)

    # Skip to the right position when resuming.
    for _ in range(start_step * cfg.grad_accum):
        next(train_iter)

    logger = MetricsLogger(
        "pretrain",
        cfg.log_dir,
        use_wandb=cfg.use_wandb,
        wandb_project=cfg.wandb_project,
        config={"train_steps": cfg.train_steps, "batch_size": cfg.batch_size,
                "grad_accum": cfg.grad_accum, "lr": cfg.lr},
    ) if ctx.is_main else None

    ac = amp_autocast(cfg.amp_dtype, ctx.device)
    scaler = make_grad_scaler(cfg.amp_dtype, ctx.device)

    model.train()
    t0 = time.perf_counter()
    tokens_seen = start_step * cfg.grad_accum * cfg.batch_size * cfg.context_length

    for step in range(start_step, cfg.train_steps):
        # LR schedule.
        lr = cosine_lr(step, warmup_steps=cfg.warmup_steps, max_steps=cfg.train_steps,
                       lr=cfg.lr, min_lr=cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation.
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(cfg.grad_accum):
            xb, yb = next(train_iter)
            # In DDP, skip the sync on all but the last micro-step.
            sync_ctx = (
                model.no_sync()
                if ctx.enabled and micro < cfg.grad_accum - 1
                else _null_context()
            )
            with sync_ctx:
                with ac:
                    _, loss = model(xb, targets=yb)
                scaler.scale(loss / cfg.grad_accum).backward()
            accum_loss += loss.item() / cfg.grad_accum

        tokens_seen += cfg.grad_accum * cfg.batch_size * cfg.context_length * ctx.world_size

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # Logging (main rank only).
        if ctx.is_main:
            elapsed = time.perf_counter() - t0
            tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
            if step % 10 == 0 or step < 5:
                print(f"step {step:6d}/{cfg.train_steps} | loss {accum_loss:.4f} | "
                      f"lr {lr:.2e} | {tok_per_sec/1e6:.2f}M tok/s")
            if logger:
                logger.log(step, {"train_loss": accum_loss, "lr": lr,
                                  "tokens_seen": tokens_seen})

        # Periodic eval.
        if cfg.eval_steps > 0 and (step + 1) % cfg.eval_steps == 0:
            val_loss = evaluate(model, dev_iter, cfg, ctx)
            if ctx.is_main:
                print(f"  [eval] step {step+1} | val_loss {val_loss:.4f}")
                if logger:
                    logger.log(step + 1, {"val_loss": val_loss})

        # Periodic save (main rank only).
        if ctx.is_main and cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            save_stage_ckpt(
                cfg.out_ckpt,
                model,
                optimizer,
                stage="pretrain",
                cfg=cfg,
                step=step + 1,
                metrics={"train_loss": accum_loss},
                drive_dir=cfg.drive_ckpt_dir or None,
            )

        barrier(ctx)

    # Final save.
    if ctx.is_main:
        val_loss = evaluate(model, dev_iter, cfg, ctx)
        print(f"\n[done] final val_loss {val_loss:.4f}")
        save_stage_ckpt(
            cfg.out_ckpt,
            model,
            optimizer,
            stage="pretrain",
            cfg=cfg,
            step=cfg.train_steps,
            metrics={"val_loss": val_loss},
            drive_dir=cfg.drive_ckpt_dir or None,
        )
        if logger:
            logger.log(cfg.train_steps, {"val_loss": val_loss})
            logger.close()

    cleanup(ctx)


import contextlib

def _null_context():
    return contextlib.nullcontext()


if __name__ == "__main__":
    main()
