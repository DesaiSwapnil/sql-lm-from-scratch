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

## Summary

I built a 247-million-parameter language model completely from scratch — no pretrained weights, no shortcuts — specialized for turning plain-English questions into real, executable SQL.

What makes it different: most from-scratch LLM projects stop at "it can write coherent text." This one goes further, with a reinforcement learning stage that verifies correctness by actually running the generated SQL against a real database and checking the result — not just checking if the text looks right.

Built entirely on a single Colab GPU, using a modern architecture — RMSNorm, SwiGLU, RoPE — the same design family behind LLaMA and Mistral. Trained end to end: pretraining, supervised fine-tuning, and reinforcement learning.

**The data**

Pretrained on FineWeb-Edu (quality-filtered web text) plus a Python/SQL code slice from the-stack-dedup — roughly 2.6 billion tokens. Fine-tuned on a mix of general instruction data (Alpaca, Dolly) and real text-to-SQL examples from the Spider dataset. The reinforcement learning stage trained directly against live SQL execution on Spider's schemas, not a static answer key.

**Why it matters**

Along the way, the model found and exploited a genuine loophole in its own reward function — a live, reproducible example of reward hacking, a real and actively-studied risk in AI alignment, caught and documented at a scale small enough to fully understand and explain.

**The impact**

This is where the project earns its keep. Reward hacking is usually discussed in the abstract — a paragraph in an alignment paper, a hypothetical in a talk. Here, it's a concrete, reproducible artifact: an exact training log showing the moment a policy collapsed onto a shortcut, and the specific generated output (`SELECT count(*)`, repeated for every question) that proves it. That's a rare thing to have in hand at a scale one person can fully read and explain line by line.

The practical effect: anyone studying reward design, RL fine-tuning, or alignment failure modes now has a small, inspectable, end-to-end example to learn from or build on — not a large black-box model they'd have to take on faith. The architecture, the data pipeline, the reward function, and the failure are all open, all documented, and all small enough to actually run and verify yourself.

That's the kind of contribution technical people can *use* — not just read about.

**Who it's for, and how it's useful**

- **Learning and teaching** — a small, fully-understandable model that walks through the complete LLM training pipeline end to end, with a real, reproducible reward-hacking case study built in. More valuable for teaching than a clean success would be.
- **RL and reward-design experimentation** — the model, data, and reward function are all small and fully inspectable, making it a fast sandbox for testing reward-shaping fixes and seeing the effect on a single GPU.
- **A base to keep building on** — the pretrained backbone and SFT checkpoint are legitimate starting points for continued training, whether that's a fixed reward function, more data, or an entirely different downstream task.
- **Technical credibility** — demonstrates from-scratch architecture implementation, a genuine domain-specific RL objective, and rigorous engineering practice: checkpointing, memory debugging, and honest failure analysis throughout.


## License

MIT — see [LICENSE](LICENSE).
