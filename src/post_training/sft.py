"""
SFT packing and dataset utilities — implemented in Stage 6.

Provides:
  pack_examples: concatenate variable-length (ids, mask) examples into fixed-length rows.
  SFTDataset: torch Dataset over a packed HDF5 file.
"""

from __future__ import annotations

import numpy as np


def pack_examples(
    examples: list[tuple[list[int], list[int]]],
    context_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack (ids, loss_mask) examples into fixed-length rows by concatenating them.

    Rows do not span examples: each example starts fresh at a position boundary.
    Pads with zeros where necessary. Returns uint16 tokens and uint8 masks shaped
    (n_rows, context_length).
    """
    raise NotImplementedError("pack_examples will be implemented in Stage 6")
