"""
GRPO training with SQL execution verifier.

Adapted from FareedKhan-dev/train-llm-from-scratch (MIT License):
  scripts/train_grpo.py (GRPO loop shape, rollout/advantage/loss structure).

Core change vs. reference: replaces the GSM8K arithmetic checker with a SQL
execution verifier (src/post_training/rewards/sql_reward.py). Reward is based
on whether the generated SQL's result set matches the gold query's result set
when executed against the relevant SQLite database. No string-matching.

Algorithm per iteration:
    1. Sample P prompts; expand by group_size G → N = P × G prompt copies.
    2. Rollout: generate one completion per copy (no grad).
    3. Score: reward_sql for each decoded completion.
    4. Advantages: group-relative normalisation within each P-sized group.
    5. Old logprobs: snapshot of current policy (no grad).
    6. Ref logprobs:  frozen SFT model (no grad).
    7. grpo_epochs policy-update steps with clipped PPO surrogate + KL penalty.

Usage:
    PYTHONPATH=. python scripts/train_grpo.py
    PYTHONPATH=. python scripts/train_grpo.py --config configs/smoke/grpo.json
Resume:
    PYTHONPATH=. python scripts/train_grpo.py --resume /content/ckpts/grpo.pt
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn

from config.loader import load_config
from config.training_config import GRPOConfig
from data_loader.prompt_dataset import get_prompt_iterator
from src.post_training.chat_template import decode, encode_prompt
from src.post_training.distributed import barrier, cleanup, ddp_setup, ddp_wrap, reduce_scalar
from src.post_training.logging_utils import MetricsLogger
from src.post_training.optim import configure_optimizer
from src.post_training.rewards.sql_reward import reward_sql
from src.post_training.rollout import compute_logprobs, rollout_prompts
from src.post_training.utils import (
    load_backbone_from_ckpt,
    make_frozen_copy,
    save_stage_ckpt,
    set_seed,
    unwrap,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",  default="configs/grpo.json")
    p.add_argument("--resume",  default=None,
                   help="GRPO checkpoint to resume from (pass 'latest' for cfg.out_ckpt)")
    p.add_argument("--print-config", action="store_true")
    return p.parse_known_args()


def _apply_cli_overrides(cfg: GRPOConfig, extras: list[str]) -> GRPOConfig:
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
            v = (val_str.lower() not in ("false", "0", "no")) if field_type is bool else field_type(val_str)
            object.__setattr__(cfg, key, v)
    return cfg


# ---------------------------------------------------------------------------
# Prompt tokenisation
# ---------------------------------------------------------------------------

def _prompt_ids(entry: dict) -> list[int]:
    """
    Return token ids for a prompt dict.

    If "prompt" is already a list of ints (smoke mode, pre-tokenised), use it
    directly.  Otherwise tokenise the user-message content via the chat template.
    """
    p = entry["prompt"]
    if isinstance(p, list):
        return p
    return encode_prompt([{"role": "user", "content": p}])


# ---------------------------------------------------------------------------
# Reward scoring (CPU, sequential — SQL execution uses thread timeout internally)
# ---------------------------------------------------------------------------

def _score_rewards(
    seqs: torch.Tensor,
    prompt_lens: torch.Tensor,
    prompt_dicts: list[dict],
) -> torch.Tensor:
    """Decode each completion and compute reward_sql. Returns (N,) float tensor."""
    N = seqs.shape[0]
    rewards = []
    for i in range(N):
        plen  = prompt_lens[i].item()
        resp  = seqs[i, plen:].tolist()
        text  = decode(resp)
        gold  = prompt_dicts[i]["gold_sql"]
        dbp   = prompt_dicts[i]["schema_path"]
        rewards.append(reward_sql(text, gold, dbp))
    return torch.tensor(rewards, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def _compute_advantages(rewards: torch.Tensor, P: int, G: int) -> torch.Tensor:
    """
    Group-relative advantage normalisation.

    rewards: (N,) with N = P × G, ordered as [group_0 × G, group_1 × G, …].
    Returns (N,) advantages, zero-centred and unit-variance within each group.
    Groups where all rewards are equal get advantage 0 (std clamped to avoid NaN).
    """
    r = rewards.view(P, G)
    mean = r.mean(dim=1, keepdim=True)              # (P, 1)
    std  = r.std(dim=1, keepdim=True).clamp(min=1e-6)  # (P, 1)
    adv  = (r - mean) / std                         # (P, G)
    return adv.view(P * G)                          # (N,)


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------

def _grpo_loss(
    logprobs:     torch.Tensor,   # (N, T-1)  current policy, requires_grad
    old_logprobs: torch.Tensor,   # (N, T-1)  snapshot, no grad
    ref_logprobs: torch.Tensor,   # (N, T-1)  frozen SFT, no grad
    advantages:   torch.Tensor,   # (N,)
    response_mask: torch.Tensor,  # (N, T)    bool
    clip_eps:     float,
    kl_coef:      float,
) -> torch.Tensor:
    """
    Clipped PPO surrogate + per-token KL penalty, averaged over response tokens.

    Importance-sampling ratio uses old_logprobs (snapshot at start of grpo_epochs).
    KL is the first-order approximation: logprob_θ − logprob_ref (always >= 0 near
    the reference, acts as a regulariser away from SFT weights).
    """
    resp = response_mask[:, 1:].float()   # (N, T-1) — targets aligned with logprobs
    n_tokens = resp.sum().clamp(min=1)

    # Importance-sampling ratio.
    log_ratio = logprobs - old_logprobs   # (N, T-1)
    ratio     = log_ratio.exp()

    # Clipped surrogate (per token, advantage broadcast from rollout level).
    adv = advantages.unsqueeze(1)         # (N, 1) → broadcasts over T-1
    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.min(surr1, surr2)   # (N, T-1)

    # KL penalty: positive when policy assigns higher prob than reference.
    kl = logprobs - ref_logprobs          # (N, T-1)

    loss = (
        -(surrogate * resp).sum() / n_tokens
        + kl_coef * (kl * resp).sum() / n_tokens
    )
    return loss


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_prompts: list[dict],
    cfg: GRPOConfig,
    ctx,
) -> dict[str, float]:
    """Sample one completion per eval prompt, return mean reward and format rate."""
    model.eval()
    prompt_ids_list = [_prompt_ids(p) for p in eval_prompts]
    seqs, _, prompt_lens = rollout_prompts(
        unwrap(model), prompt_ids_list, cfg.rollout_len, ctx.device,
        temperature=cfg.temperature,
        top_p=cfg.top_p if cfg.top_p < 1.0 else None,
    )
    rewards = _score_rewards(seqs, prompt_lens, eval_prompts)
    mean_r  = reduce_scalar(rewards.mean().item(), ctx)
    # Format rate: fraction with well-formed <sql> tags.
    from src.post_training.rewards.parsing import has_well_formed_sql
    fmt_rate = sum(
        has_well_formed_sql(decode(seqs[i, prompt_lens[i]:].tolist()))
        for i in range(len(eval_prompts))
    ) / max(1, len(eval_prompts))
    fmt_rate = reduce_scalar(fmt_rate, ctx)
    model.train()
    return {"eval_reward": mean_r, "eval_fmt_rate": fmt_rate}


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------

def _load_grpo_resume(path: str, model, optimizer, device: str) -> int:
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
    iteration = ckpt.get("step", 0)
    print(f"[resume] GRPO iteration={iteration} from {path!r}")
    return iteration


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args, extras = _parse_args()
    cfg: GRPOConfig = load_config(GRPOConfig, args.config)
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

    # --- Models ---
    model = load_backbone_from_ckpt(cfg, cfg.sft_ckpt, ctx.device)
    ref_model = make_frozen_copy(model, ctx.device)   # frozen SFT reference

    model = ddp_wrap(model, ctx)
    optimizer = configure_optimizer(unwrap(model), lr=cfg.lr, weight_decay=0.0)

    # --- Resume ---
    start_iter = 0
    resume_path = args.resume
    if resume_path == "latest":
        resume_path = cfg.out_ckpt
    if resume_path and os.path.exists(resume_path):
        start_iter = _load_grpo_resume(resume_path, model, optimizer, ctx.device)
    elif resume_path:
        print(f"[resume] not found at {resume_path!r}; starting from SFT weights")

    # --- Prompt iterators ---
    prompt_iter = get_prompt_iterator(
        cfg.prompt_path, cfg.prompts_per_iter,
        rank=ctx.rank, world_size=ctx.world_size, seed=cfg.seed,
    )
    # Load a fixed eval prompt batch (up to prompts_per_iter examples).
    eval_prompts: list[dict] = []
    try:
        import json as _json
        with open(cfg.eval_prompt_path) as fh:
            for line in fh:
                if line.strip():
                    eval_prompts.append(_json.loads(line))
                    if len(eval_prompts) >= cfg.prompts_per_iter:
                        break
    except FileNotFoundError:
        if ctx.is_main:
            print(f"[grpo] eval prompt file not found: {cfg.eval_prompt_path!r} — skipping eval")

    logger = MetricsLogger(
        "grpo", cfg.log_dir,
        use_wandb=cfg.use_wandb,
        wandb_project=cfg.wandb_project,
        config={"iterations": cfg.iterations, "group_size": cfg.group_size,
                "rollout_len": cfg.rollout_len, "kl_coef": cfg.kl_coef},
    ) if ctx.is_main else None

    t0 = time.perf_counter()
    P  = cfg.prompts_per_iter
    G  = cfg.group_size

    for iteration in range(start_iter, cfg.iterations):
        # ── 1. Sample prompts ──────────────────────────────────────────────
        prompts = next(prompt_iter)          # list of P dicts
        # Expand each prompt G times so we get G completions per prompt.
        all_dicts = [p for p in prompts for _ in range(G)]  # N = P × G
        N = len(all_dicts)

        # ── 2. Rollout ─────────────────────────────────────────────────────
        prompt_ids_list = [_prompt_ids(p) for p in all_dicts]
        model.eval()
        with torch.no_grad():
            seqs, response_mask, prompt_lens = rollout_prompts(
                unwrap(model), prompt_ids_list, cfg.rollout_len, ctx.device,
                temperature=cfg.temperature,
                top_p=cfg.top_p if cfg.top_p < 1.0 else None,
                group_size=G,
            )

        # ── 3. Reward scoring ──────────────────────────────────────────────
        rewards = _score_rewards(seqs, prompt_lens, all_dicts).to(ctx.device)

        # ── 4. Group-relative advantages ──────────────────────────────────
        advantages = _compute_advantages(rewards, P, G)   # (N,)

        # ── 5. Snapshot logprobs (no grad) ─────────────────────────────────
        with torch.no_grad():
            old_logprobs, _ = compute_logprobs(
                unwrap(model), seqs, response_mask, requires_grad=False)
            ref_logprobs, _ = compute_logprobs(
                ref_model, seqs, response_mask, requires_grad=False)

        # ── 6. Policy update epochs ────────────────────────────────────────
        model.train()
        for grpo_ep in range(cfg.grpo_epochs):
            logprobs, entropy = compute_logprobs(
                model, seqs, response_mask, requires_grad=True)
            loss = _grpo_loss(
                logprobs, old_logprobs, ref_logprobs,
                advantages, response_mask,
                cfg.clip, cfg.kl_coef,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        # ── 7. Logging ─────────────────────────────────────────────────────
        if ctx.is_main:
            mean_r    = rewards.mean().item()
            frac_corr = (rewards >= 1.0).float().mean().item()  # full-credit fraction
            elapsed   = time.perf_counter() - t0
            if iteration % 10 == 0 or iteration < 5:
                print(
                    f"iter {iteration:5d}/{cfg.iterations} | "
                    f"loss {loss.item():.4f} | "
                    f"reward {mean_r:.3f} | "
                    f"correct {frac_corr:.2%} | "
                    f"entropy {entropy.mean().item():.3f} | "
                    f"{elapsed/60:.1f}m"
                )
            if logger:
                logger.log(iteration, {
                    "loss":       loss.item(),
                    "reward":     mean_r,
                    "correct":    frac_corr,
                    "entropy":    entropy.mean().item(),
                    "kl":         (logprobs - ref_logprobs).mean().item(),
                })

        # ── 8. Periodic eval ───────────────────────────────────────────────
        if eval_prompts and cfg.eval_every > 0 and (iteration + 1) % cfg.eval_every == 0:
            eval_metrics = evaluate(model, eval_prompts, cfg, ctx)
            if ctx.is_main:
                print(f"  [eval] iter {iteration+1} | " +
                      " | ".join(f"{k} {v:.4f}" for k, v in eval_metrics.items()))
                if logger:
                    logger.log(iteration + 1, eval_metrics)

        # ── 9. Periodic save ───────────────────────────────────────────────
        if ctx.is_main and cfg.save_every > 0 and (iteration + 1) % cfg.save_every == 0:
            save_stage_ckpt(
                cfg.out_ckpt, model, optimizer,
                stage="grpo", cfg=cfg, step=iteration + 1,
                metrics={"reward": rewards.mean().item()},
                drive_dir=cfg.drive_ckpt_dir or None,
            )

        barrier(ctx)

    # --- Final save ---
    if ctx.is_main:
        save_stage_ckpt(
            cfg.out_ckpt, model, optimizer,
            stage="grpo", cfg=cfg, step=cfg.iterations,
            metrics={},
            drive_dir=cfg.drive_ckpt_dir or None,
        )
        if logger:
            logger.close()

    cleanup(ctx)


if __name__ == "__main__":
    main()
