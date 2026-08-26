# SQL-LM: a 250M-parameter SQL-specialized language model from scratch

A decoder-only transformer trained from scratch with a SQL-execution RLVR objective.
Pretrains on FineWeb-Edu + code, fine-tunes on Spider/BIRD instruction data, then
uses GRPO with an execution-based verifier to teach precise SQL generation.

## Architecture (~247M parameters)

| Component | Choice | Notes |
|---|---|---|
| Norm | RMSNorm (pre-norm) | Replaces LayerNorm |
| MLP | SwiGLU, hidden=2048 | ≈ 8/3 × n\_embed; 3 weight matrices |
| Position | RoPE | No learned position embeddings |
| Attention | Standard MHA, n\_head=12, head\_dim=64 | Fused QKV projection |
| Vocabulary | `cl100k_base` via tiktoken | vocab\_size=100352 (padded for CUDA) |

Config: `n_embed=768`, `n_head=12`, `n_blocks=24`, `context_length=1024`.
Weight-tied lm\_head (embedding table reused) so the vocab doesn't dominate the param count.

## Training pipeline

| Stage | Data | Script |
|---|---|---|
| Pretrain | FineWeb-Edu sample-10BT + Python/SQL (the-stack-dedup) | `scripts/pretrain_base.py` |
| SFT | Alpaca + Dolly (general) + Spider/BIRD (NL→SQL) | `scripts/train_sft.py` |
| GRPO | Spider/BIRD prompts, SQL execution reward | `scripts/train_grpo.py` |
| Eval | Spider/BIRD held-out split | `scripts/eval_sql.py` |

DPO is a stretch goal — see `configs/dpo.json` when implemented.

## Hardware target

Google Colab Pro, single GPU (A100 40GB / L4 / V100), ~12-hour sessions.
All training scripts checkpoint locally **and** sync to Google Drive. Resume is always
available via `--resume latest` — designed from day one for frequent disconnects.

## Setup

```bash
pip install -r requirements.txt
```

Mount Google Drive before any training run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Checkpoints save to `/content/drive/MyDrive/llm_ckpts/` by default.
Override with `--drive_ckpt_dir /your/path`.

## Running (session-by-session sketch)

```bash
# Session 1: download + tokenize pretraining data
PYTHONPATH=. python scripts/prepare_pretrain_data.py --out_dir /content/data

# Session 2+: pretrain (add --resume latest on reconnect)
PYTHONPATH=. python scripts/pretrain_base.py --config configs/pretrain.json

# After pretraining: prepare SFT data
PYTHONPATH=. python scripts/prepare_sft_data.py --out_dir /content/data

# SFT
PYTHONPATH=. python scripts/train_sft.py

# GRPO with SQL execution verifier
PYTHONPATH=. python scripts/train_grpo.py

# Evaluate on Spider/BIRD held-out split
PYTHONPATH=. python scripts/eval_sql.py
```

Smoke test (CPU, seconds):

```bash
PYTHONPATH=. python scripts/pretrain_base.py --config configs/smoke/pretrain.json
```

## Config system

Each stage has a small JSON in `configs/`. Configs layer as:

```
dataclass defaults < configs/base.json < configs/<stage>.json < --field CLI overrides
```

`configs/smoke/` mirrors the full config tree with tiny model dims for fast CPU tests.

## Credits

Training-loop structure, checkpoint/resume pattern, config-loading system, GRPO loop
shape, DDP helpers, optimizer setup, and SFT packing/masking mechanism are adapted from:

> **Fareed Khan** — [train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch) (MIT License)

Files that directly incorporate adapted code carry an inline MIT license notice at the
top. The model architecture (RMSNorm, SwiGLU, RoPE), tokenizer (cl100k\_base),
pretraining data sources (FineWeb-Edu, The Stack), and post-training objective
(SQL execution verifier) are original to this repository.

## License

MIT — see [LICENSE](LICENSE).
