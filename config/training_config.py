"""
Configuration dataclasses for the SQL-LM training pipeline.

Architecture: ~247M params, decoder-only, RMSNorm + SwiGLU + RoPE, cl100k_base tokenizer.
  n_embed=768, n_head=12, n_blocks=24, swiglu_hidden=2048, context_length=1024, vocab_size=100352

Hardware target: Google Colab Pro, single GPU (A100/L4/V100), 12-hour sessions.
Checkpoints write to local /content/ckpts and optionally sync to Google Drive.
"""

from __future__ import annotations

from dataclasses import dataclass

# Colab paths.
_DATA = "/content/data"
_CKPT = "/content/ckpts"
_LOG = "/content/logs"
_DRIVE = "/content/drive/MyDrive/llm_ckpts"


@dataclass
class BaseModelConfig:
    # Model architecture — must be identical across all pipeline stages.
    vocab_size: int = 100352       # cl100k_base BPE vocab rounded to next multiple of 128
    context_length: int = 1024
    n_embed: int = 768
    n_head: int = 12
    n_blocks: int = 24
    swiglu_hidden: int = 2048      # 8/3 * 768 = 2048 exactly; SwiGLU has 3 weight matrices

    # Runtime.
    device: str = "cuda"
    amp_dtype: str | None = "auto"  # auto → bf16 on Ampere+, fp16+GradScaler on T4
    seed: int = 1337
    compile: bool = False          # torch.compile: large speedup, slow first step
    grad_checkpointing: bool = False  # stage configs should enable this for T4 16GB

    # Paths.
    ckpt_dir: str = _CKPT
    drive_ckpt_dir: str = _DRIVE  # set to "" to disable Drive sync
    log_dir: str = _LOG

    # Weights & Biases (optional).
    use_wandb: bool = False
    wandb_project: str = "sql-lm"


@dataclass
class PretrainConfig(BaseModelConfig):
    train_path: str = f"{_DATA}/pretrain_train.h5"
    dev_path: str = f"{_DATA}/pretrain_dev.h5"
    # T4 16GB-safe microbatch. Effective batch stays 128 (4 × 32).
    # A100 40GB can raise batch_size and drop grad_accum proportionally (e.g. 16 × 8).
    # Logits are (B, T, 100352) — this, not attention, is the T4 OOM cliff.
    batch_size: int = 4
    grad_accum: int = 32
    train_steps: int = 20_000
    eval_steps: int = 1_000
    eval_iters: int = 100
    warmup_steps: int = 2_000
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    out_ckpt: str = f"{_CKPT}/pretrain.pt"
    save_every: int = 1_000


@dataclass
class SFTConfig(BaseModelConfig):
    pretrained_ckpt: str = f"{_CKPT}/pretrain.pt"
    train_path: str = f"{_DATA}/sft_train.h5"
    dev_path: str = f"{_DATA}/sft_dev.h5"
    out_ckpt: str = f"{_CKPT}/sft.pt"
    batch_size: int = 8
    grad_accum: int = 4
    epochs: int = 3
    max_steps: int = -1             # -1 = run full epochs
    eval_steps: int = 200
    warmup_steps: int = 100
    lr: float = 1e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    save_every: int = 500


@dataclass
class GRPOConfig(BaseModelConfig):
    sft_ckpt: str = f"{_CKPT}/sft.pt"
    prompt_path: str = f"{_DATA}/sql_prompts_train.jsonl"
    eval_prompt_path: str = f"{_DATA}/sql_prompts_dev.jsonl"
    out_ckpt: str = f"{_CKPT}/grpo.pt"
    iterations: int = 2_000
    prompts_per_iter: int = 8       # distinct SQL prompts per iteration
    group_size: int = 8             # completions sampled per prompt
    rollout_len: int = 512          # max new tokens (SQL + reasoning chain)
    temperature: float = 1.0
    top_p: float = 1.0
    grpo_epochs: int = 1
    clip: float = 0.2
    kl_coef: float = 0.04           # KL(policy || ref) penalty
    lr: float = 1e-6
    grad_clip: float = 1.0
    eval_every: int = 100
    save_every: int = 200


# Tiny config for fast smoke tests: runs on CPU in seconds.
# Use: load_config(PretrainConfig, "configs/smoke/pretrain.json")
# The smoke base.json overrides these fields automatically.
SMOKE = dict(
    vocab_size=512,
    context_length=64,
    n_embed=128,
    n_head=4,
    n_blocks=2,
    swiglu_hidden=341,      # 8/3 * 128 ≈ 341 (rounded down to odd is fine for smoke)
    device="cpu",
    amp_dtype=None,
    compile=False,
    grad_checkpointing=False,
    drive_ckpt_dir="",      # no Drive sync in smoke runs
)
