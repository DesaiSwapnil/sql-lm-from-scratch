"""
SFT training script: fine-tune the pretrained base model on instruction + SQL data.

Adapted from FareedKhan-dev/train-llm-from-scratch (MIT License):
  scripts/train_sft.py (training loop, SFT loss masking mechanism).

Loss is computed only over assistant tokens (per the loss_mask in the HDF5).
Loads the pretrained backbone from cfg.pretrained_ckpt, then trains for cfg.epochs
epochs (or up to cfg.max_steps if set).

Usage:
    PYTHONPATH=. python scripts/train_sft.py
    PYTHONPATH=. python scripts/train_sft.py --config configs/smoke/sft.json
Resume:
    PYTHONPATH=. python scripts/train_sft.py --resume /content/ckpts/sft.pt
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time

import torch
from torch.utils.data import DataLoader

from config.loader import load_config
from config.training_config import SFTConfig
from data_loader.sft_dataset import SFTDataset
from src.post_training.distributed import (
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
    load_backbone_from_ckpt,
    save_stage_ckpt,
    set_seed,
    unwrap,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",  default="configs/sft.json")
    p.add_argument("--resume",  default=None,
                   help="SFT checkpoint to resume from (pass 'latest' for cfg.out_ckpt)")
    p.add_argument("--print-config", action="store_true")
    return p.parse_known_args()


def _apply_cli_overrides(cfg: SFTConfig, extras: list[str]) -> SFTConfig:
    it = iter(extras)
    for tok in it:
        if tok.startswith("--"):
            key = tok.lstrip("-").replace("-", "_")
            try:
                val_str = next(it)
            except StopIteration:
                raise SystemExit(f"Missing value for flag {tok!r}")
            if key not in cfg.__dataclass_fields__:
                print(f"[cli] ignoring unknown flag --{key}")
                continue
            field_type = type(getattr(cfg, key))
            if field_type is bool:
                v = val_str.lower() not in ("false", "0", "no")
            elif field_type in (int, float, str):
                v = field_type(val_str)
            else:
                v = val_str
            object.__setattr__(cfg, key, v)
    return cfg


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader: DataLoader, cfg: SFTConfig, ctx) -> float:
    model.eval()
    ac = amp_autocast(cfg.amp_dtype, ctx.device)
    total, count = 0.0, 0
    for xb, yb, mb in loader:
        xb, yb, mb = xb.to(ctx.device), yb.to(ctx.device), mb.to(ctx.device)
        with ac:
            _, loss = model(xb, targets=yb, loss_mask=mb)
        if loss is not None:
            total += loss.item()
            count += 1
    model.train()
    mean = total / max(count, 1)
    return reduce_scalar(mean, ctx)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_sft_resume(path: str, model, optimizer, device: str) -> tuple[int, int]:
    """Load SFT checkpoint; return (global_step, start_epoch)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    unwrap(model).load_state_dict(state)
    if optimizer and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for og in optimizer.state.values():
            for k, v in og.items():
                if isinstance(v, torch.Tensor):
                    og[k] = v.to(device)
    step  = ckpt.get("step",  0)
    epoch = ckpt.get("extra", {}).get("epoch", 0)
    print(f"[resume] SFT step={step}, epoch={epoch} from {path!r}")
    return step, epoch


# ---------------------------------------------------------------------------
# Total-step budget
# ---------------------------------------------------------------------------

def _max_steps(cfg: SFTConfig, steps_per_epoch: int) -> int:
    """Total optimizer steps across all epochs, capped at cfg.max_steps if set."""
    total = cfg.epochs * steps_per_epoch
    if cfg.max_steps > 0:
        total = min(total, cfg.max_steps)
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args, extras = _parse_args()

    cfg: SFTConfig = load_config(SFTConfig, args.config)
    cfg = _apply_cli_overrides(cfg, extras)

    if args.print_config:
        import dataclasses, json
        print(json.dumps(dataclasses.asdict(cfg), indent=2))
        return

    ctx = ddp_setup(cfg.device)
    set_seed(cfg.seed + ctx.rank)

    if ctx.is_main:
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        os.makedirs(cfg.log_dir,  exist_ok=True)

    # --- Datasets / loaders ---
    train_ds = SFTDataset(cfg.train_path)
    dev_ds   = SFTDataset(cfg.dev_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=(ctx.device.startswith("cuda")),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    steps_per_epoch = len(train_loader) // cfg.grad_accum
    total_steps     = _max_steps(cfg, steps_per_epoch)

    if ctx.is_main:
        print(f"[sft] train rows={len(train_ds):,}  dev rows={len(dev_ds):,}")
        print(f"[sft] steps/epoch={steps_per_epoch}  total_steps={total_steps}")

    # --- Model ---
    # Load pretrained backbone, then optionally resume from an SFT checkpoint.
    model = load_backbone_from_ckpt(cfg, cfg.pretrained_ckpt, ctx.device)

    if cfg.grad_checkpointing:
        unwrap(model).gradient_checkpointing = True

    if cfg.compile:
        model = torch.compile(model)

    model = ddp_wrap(model, ctx)

    optimizer = configure_optimizer(unwrap(model), lr=cfg.lr, weight_decay=cfg.weight_decay)

    global_step, start_epoch = 0, 0
    resume_path = args.resume
    if resume_path == "latest":
        resume_path = cfg.out_ckpt
    if resume_path and os.path.exists(resume_path):
        global_step, start_epoch = _load_sft_resume(resume_path, model, optimizer, ctx.device)
    elif resume_path:
        print(f"[resume] not found at {resume_path!r}; starting from pretrained backbone")

    logger = MetricsLogger(
        "sft", cfg.log_dir,
        use_wandb=cfg.use_wandb,
        wandb_project=cfg.wandb_project,
        config={"epochs": cfg.epochs, "lr": cfg.lr, "total_steps": total_steps},
    ) if ctx.is_main else None

    ac = amp_autocast(cfg.amp_dtype, ctx.device)
    model.train()
    t0 = time.perf_counter()

    for epoch in range(start_epoch, cfg.epochs):
        if ctx.is_main:
            print(f"\n[epoch {epoch+1}/{cfg.epochs}]")

        micro_buf_loss = 0.0
        micro_count    = 0

        for batch_idx, (xb, yb, mb) in enumerate(train_loader):
            if global_step >= total_steps:
                break

            xb = xb.to(ctx.device)
            yb = yb.to(ctx.device)
            mb = mb.to(ctx.device)

            # On the first micro-step of each optimizer step, zero grads.
            if micro_count == 0:
                optimizer.zero_grad(set_to_none=True)
                lr = cosine_lr(global_step, warmup_steps=cfg.warmup_steps,
                               max_steps=total_steps, lr=cfg.lr, min_lr=cfg.min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            is_last_micro = (micro_count == cfg.grad_accum - 1) or \
                            (batch_idx == len(train_loader) - 1)

            sync_ctx = (
                model.no_sync()
                if ctx.enabled and not is_last_micro
                else contextlib.nullcontext()
            )
            with sync_ctx:
                with ac:
                    _, loss = model(xb, targets=yb, loss_mask=mb)
                if loss is None:
                    micro_count += 1
                    continue
                (loss / cfg.grad_accum).backward()
            micro_buf_loss += loss.item() / cfg.grad_accum
            micro_count    += 1

            if is_last_micro:
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                global_step  += 1
                accum_loss    = micro_buf_loss
                micro_buf_loss = 0.0
                micro_count    = 0

                if ctx.is_main:
                    if global_step % 10 == 0 or global_step < 5:
                        elapsed = time.perf_counter() - t0
                        print(f"  step {global_step:5d}/{total_steps} | "
                              f"loss {accum_loss:.4f} | lr {lr:.2e} | "
                              f"{elapsed/60:.1f}m")
                    if logger:
                        logger.log(global_step, {"train_loss": accum_loss, "lr": lr})

                # Periodic eval.
                if cfg.eval_steps > 0 and global_step % cfg.eval_steps == 0:
                    val_loss = evaluate(model, dev_loader, cfg, ctx)
                    if ctx.is_main:
                        print(f"  [eval] step {global_step} | val_loss {val_loss:.4f}")
                        if logger:
                            logger.log(global_step, {"val_loss": val_loss})

                # Periodic save.
                if ctx.is_main and cfg.save_every > 0 and global_step % cfg.save_every == 0:
                    save_stage_ckpt(
                        cfg.out_ckpt, model, optimizer,
                        stage="sft", cfg=cfg, step=global_step,
                        metrics={"train_loss": accum_loss},
                        extra={"epoch": epoch},
                        drive_dir=cfg.drive_ckpt_dir or None,
                    )

                barrier(ctx)

        if global_step >= total_steps:
            break

    # --- Final eval + save ---
    val_loss = evaluate(model, dev_loader, cfg, ctx)
    if ctx.is_main:
        print(f"\n[done] final val_loss {val_loss:.4f}")
        save_stage_ckpt(
            cfg.out_ckpt, model, optimizer,
            stage="sft", cfg=cfg, step=global_step,
            metrics={"val_loss": val_loss},
            extra={"epoch": cfg.epochs},
            drive_dir=cfg.drive_ckpt_dir or None,
        )
        if logger:
            logger.log(global_step, {"val_loss": val_loss})
            logger.close()

    cleanup(ctx)


if __name__ == "__main__":
    main()
