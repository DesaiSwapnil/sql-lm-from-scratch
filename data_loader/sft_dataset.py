"""
SFT dataset loader over packed HDF5 files written by scripts/prepare_sft_data.py.

Each HDF5 file contains two 2-D datasets of shape (n_rows, context_length):
  tokens    dtype int32  — input token ids
  loss_mask dtype uint8  — 1 on assistant-turn tokens, 0 elsewhere

The file is kept open for the lifetime of the dataset and closed on garbage
collection via __del__ so that the DataLoader worker can stream large files
without loading them fully into RAM.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """torch Dataset over a packed SFT HDF5 file (tokens + loss_mask)."""

    def __init__(self, path: str) -> None:
        self._f    = h5py.File(path, "r")
        self._tok  = self._f["tokens"]
        self._mask = self._f["loss_mask"]
        self._len  = self._tok.shape[0]
        self._ctx  = self._tok.shape[1]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (input_ids, target_ids, loss_mask) all of shape (context_length,)."""
        tokens = np.array(self._tok[idx],  dtype=np.int64)
        mask   = np.array(self._mask[idx], dtype=np.float32)

        input_ids  = torch.from_numpy(tokens[:-1] if self._ctx > 1 else tokens)
        target_ids = torch.from_numpy(tokens[1:]  if self._ctx > 1 else tokens)
        loss_mask  = torch.from_numpy(mask  [1:]  if self._ctx > 1 else mask)

        # Rows are already padded to context_length by the packer, so both slices
        # have length context_length - 1.  Callers should treat this as the sequence
        # length and the model uses targets + mask directly.
        return input_ids, target_ids, loss_mask

    def __del__(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
