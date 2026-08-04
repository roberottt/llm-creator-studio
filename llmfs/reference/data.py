"""Reference for module 04: from a list of ids to batches on the GPU."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

#: uint16 goes up to 65,535. With vocab_size=4096 that is plenty, and it takes half the
#: space of uint32: 500M tokens are 1 GB instead of 2 GB.
MAX_UINT16 = 2**16


def pack_tokens_uint16(ids: Sequence[int], vocab_size: int) -> np.ndarray:
    """Pack ids into a `uint16` array, validating that they fit.

    The validation is not paranoia: if an id goes out of range, numpy silently *wraps
    around* (65536 becomes 0) and you end up training on corrupted data with no visible
    error. The model simply learns worse and you do not know why.

    Raises:
        ValueError: if `vocab_size` does not fit in uint16 or if some id is outside the
            vocabulary.
    """
    if vocab_size > MAX_UINT16:
        raise ValueError(
            f"vocab_size={vocab_size} does not fit in uint16. Use uint32 (and twice the disk)."
        )

    array = np.asarray(ids, dtype=np.int64)
    if array.size and (array.min() < 0 or array.max() >= vocab_size):
        raise ValueError(
            f"ids outside the vocabulary [0, {vocab_size}): "
            f"min={array.min()}, max={array.max()}"
        )
    return array.astype(np.uint16)


def train_val_split(tokens: np.ndarray, val_fraction: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """Split the corpus into training and validation.

    The cut is **contiguous and from the end**, not random. Reason: with a sliding window,
    two overlapping samples share almost all of their tokens. If you shuffled at the token
    level, the validation set would contain pieces of sequences seen during training and
    the perplexity would come out optimistic. Cutting from the end guarantees that
    validation is made of stories the model has never seen.

    Returns:
        `(train, val)` as views of the original array (nothing is copied).
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), not {val_fraction}")

    n_val = max(1, int(len(tokens) * val_fraction))
    if n_val >= len(tokens):
        raise ValueError("val_fraction leaves the training set empty")
    return tokens[:-n_val], tokens[-n_val:]


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device | str | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of contiguous windows.

    `batch_size` starting positions are picked at random and `context_length + 1` tokens
    are taken from each: the first `context_length` are the input and the last
    `context_length` (shifted by one) are the target. In other words, the task is "predict
    the next token" at every position at once.

        data  = [ 5, 8, 2, 9, 1, ...]
        x     = [ 5, 8, 2, 9]
        y     = [ 8, 2, 9, 1]

    With `T` positions you get `T` predictions per sample, not one. That is why training a
    language model is so data-efficient.

    Args:
        data: 1-D array of tokens (typically a uint16 `np.memmap`).
        device: if it is CUDA, `pin_memory` + `non_blocking` are used to overlap the copy
            with the computation.
        rng: numpy generator. Fix it to get reproducible batches.

    Returns:
        `(x, y)`, both `int64` with shape `(batch_size, context_length)`.
    """
    rng = rng or np.random.default_rng()
    max_start = len(data) - context_length - 1
    if max_start < 1:
        raise ValueError(
            f"The corpus ({len(data)} tokens) is shorter than the context "
            f"({context_length} + 1)."
        )

    starts = rng.integers(0, max_start, size=batch_size)
    # astype(int64) also materializes the memmap: without the copy, torch would be left
    # pointing at disk-mapped memory and every access would be a read.
    x_np = np.stack([data[i : i + context_length] for i in starts]).astype(np.int64)
    y_np = np.stack([data[i + 1 : i + 1 + context_length] for i in starts]).astype(np.int64)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    if device is not None:
        device = torch.device(device)
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

    return x, y
