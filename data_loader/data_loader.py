# Adapted from FareedKhan-dev/train-llm-from-scratch (MIT License).
# Original: https://github.com/FareedKhan-dev/train-llm-from-scratch/blob/main/data_loader/data_loader.py
# Changes: updated dtype handling for cl100k_base (int32 tokens exceed uint16 range);
#          added sorted reads within each batch for better HDF5 I/O locality.
"""
Pretraining batch iterator over flat-token HDF5 files.

HDF5 layout written by scripts/prepare_pretrain_data.py:
    file['tokens']  shape (N,), dtype int32
    N is the total number of tokens in the corpus (FineWeb-Edu + code, concatenated
    with EOT=100257 separators between documents).

Each "example" i occupies positions [i*T, i*T + T + 1) — context_length + 1 tokens.
    x = tokens[i*T : i*T + T]       (input)
    y = tokens[i*T + 1 : i*T + T+1] (target = x shifted right by 1)

n_examples = (N - 1) // context_length

The iterator shuffles example indices at the start of each epoch. Within a batch,
indices are sorted before reading so HDF5 disk reads are roughly sequential.
"""

from __future__ import annotations

from typing import Iterator

import h5py
import numpy as np
import torch


def get_batch_iterator(
    data_path: str,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Infinite iterator yielding (x, y) batches from a flat-token HDF5 file.

    Args:
        data_path: path to the HDF5 file (must have a 'tokens' dataset).
        batch_size: number of sequences per batch.
        context_length: sequence length T; each sequence has T tokens.
        device: torch device string.

    Yields:
        xb: (batch_size, context_length) int64 tensor — input token ids.
        yb: (batch_size, context_length) int64 tensor — target token ids (shifted by 1).
    """
    # Open explicitly so the finally block can close cleanly when GeneratorExit is raised
    # (e.g. when the training script finishes and the iterator is garbage-collected).
    f = h5py.File(data_path, "r")
    try:
        dataset = f["tokens"]
        N = dataset.shape[0]
        n_examples = (N - 1) // context_length
        if n_examples < batch_size:
            raise ValueError(
                f"HDF5 has only {n_examples} examples at context_length={context_length}; "
                f"need at least batch_size={batch_size}. Run prepare_pretrain_data.py first."
            )

        idxs = np.arange(n_examples, dtype=np.int64)
        np.random.shuffle(idxs)
        epoch, i = 0, 0

        while True:
            if i + batch_size > n_examples:
                np.random.shuffle(idxs)
                i = 0
                epoch += 1
                print(f"[data] epoch {epoch} (n_examples={n_examples:,})")

            batch_idxs = idxs[i : i + batch_size]
            i += batch_size

            # Sort starts for roughly sequential reads (better HDF5 I/O performance).
            starts = np.sort(batch_idxs) * context_length
            samples = np.array([dataset[s : s + context_length + 1] for s in starts], dtype=np.int64)

            xb = torch.tensor(samples[:, :context_length],     dtype=torch.long).to(device)
            yb = torch.tensor(samples[:, 1 : context_length + 1], dtype=torch.long).to(device)
            yield xb, yb
    finally:
        f.close()
