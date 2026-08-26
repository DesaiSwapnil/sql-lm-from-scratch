# Adapted from FareedKhan-dev/train-llm-from-scratch (MIT License).
# Original: https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/src/post_training/logging_utils.py
# Changes: updated default wandb_project name.
"""
Lightweight metrics logging shared by every training stage.

Always writes one JSON object per logged step to a JSONL file under log_dir.
Optionally mirrors to Weights & Biases when use_wandb=True.
Only the main process should construct/use a logger.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any


class MetricsLogger:
    def __init__(
        self,
        stage: str,
        log_dir: str,
        *,
        use_wandb: bool = False,
        wandb_project: str = "sql-lm",
        config: dict | None = None,
        run_name: str | None = None,
    ) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.stage = stage
        stamp = int(time.time())
        self.path = os.path.join(log_dir, f"{stage}_{stamp}.jsonl")
        self._fh = open(self.path, "a")
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=run_name or f"{stage}-{stamp}", config=config or {})
                self._wandb = wandb
            except Exception as e:
                print(f"[logger] wandb disabled ({e}); JSONL logging only -> {self.path}")
        print(f"[logger] stage={stage} -> {self.path}")

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        record = {"step": step, "wall": time.time(), **metrics}
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            if self._wandb is not None:
                self._wandb.finish()
