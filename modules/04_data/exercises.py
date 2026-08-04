"""Module 04 - Data: from text to batches on the GPU.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement -> `llmfs check 04` -> `llmfs hint 04 -e N`
-> `SOLUTION.md` has the complete code.

WHAT YOU ARE GOING TO BUILD
===========================

The bridge between tokenized text and the GPU. Three functions:

    pack_tokens_uint16   (ex. 1)  ids -> a 2-bytes-per-token array, with validation
    train_val_split      (ex. 2)  set a chunk aside for validation
    get_batch            (ex. 3)  draw a batch of random windows

Your training run will execute the third one tens of thousands of times, and it is where the
module's important idea lives: how text is turned into a learning task.

VOCABULARY YOU ARE GOING TO NEED
================================

- **batch**: a group of samples processed at once. Going one at a time wastes the GPU.
- **window / context**: how many consecutive tokens the model sees at once. Ours, 512.
- **memmap**: an array that lives on disk but is used as if it were in memory. The operating
  system loads only what you touch.
- **validation set**: text the model does NOT see during training, so you can tell whether
  it is learning or just memorizing.
- **uint16**: unsigned 2-byte integer, from 0 to 65,535.

    llmfs demo 04     the whole pipeline, from text to a batch on the GPU
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

#: uint16 goes up to 65,535. With vocab_size=4096 that is plenty, and it is half of uint32.
MAX_UINT16 = 2**16


def pack_tokens_uint16(ids: Sequence[int], vocab_size: int) -> np.ndarray:
    """Turns a list of ids into a `uint16` array, validating that they fit.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Four steps, and three of them are the validation.

        1. If `vocab_size > 2**16`, raise `ValueError` (it does not fit in uint16, you would
           need uint32).

        2. Convert to `int64` FIRST, which is where everything fits:

               array = np.asarray(ids, dtype=np.int64)

        3. If the array is NOT empty, check the range:

               if array.size and (array.min() < 0 or array.max() >= vocab_size):
                   raise ValueError(f"... min={array.min()}, max={array.max()}")

        4. And only then convert:

               return array.astype(np.uint16)

    WHY VALIDATE BEFORE CONVERTING
    ------------------------------
    Numpy does NOT warn you if a number does not fit: it *wraps around* silently.

        np.array([65536], dtype=np.int64).astype(np.uint16)   ->   array([0])

    No exception, no warning. Your data is corrupted, the model trains worse, and there is
    absolutely nothing pointing at the cause. You could spend days looking for it.

    If you converted first and validated afterwards, the overflow would already have
    happened and you would be checking data that is already corrupted. That is why step 2
    comes before step 3.

    WHY THE `array.size and ...`
    ----------------------------
    On an empty array, `.min()` raises an exception about reductions over empty sequences: a
    real error, but one that has nothing to do with what you are validating and sends you off
    track. The short-circuit avoids it.

    PUT THE VALUES IN THE ERROR MESSAGE
    -----------------------------------
    "ids out of range" does not help. "max=9999" tells you instantly that your tokenizer is
    producing ids it should not, and by how much. There is a test that checks the value
    appears in the message.

    WHY uint16
    ----------
        int64  (python's)  ->  500M tokens = 4 GB
        uint32             ->  2 GB
        uint16             ->  1 GB

    uint16 goes up to 65,535 and our ids run from 0 to 4095: plenty to spare.

    Args:
        ids: the ids to pack.
        vocab_size: the vocabulary size. Every id must be in [0, vocab_size).

    Returns:
        An `np.ndarray` of dtype `uint16`.

    Raises:
        ValueError: if `vocab_size` does not fit in uint16, or if some id is out of range.
    """
    raise NotImplementedError("TODO: module 04, exercise 1 - pack_tokens_uint16")


def train_val_split(tokens: np.ndarray, val_fraction: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """Sets a chunk of the end aside for validation.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Four lines.

        1. Validate that `val_fraction` is in (0, 1) and raise `ValueError` if not.

        2. Compute how many tokens go to validation:

               n_val = max(1, int(len(tokens) * val_fraction))

        3. If `n_val >= len(tokens)`, raise `ValueError` (it would leave training empty).

        4. Return the two parts:

               return tokens[:-n_val], tokens[-n_val:]

    THE ONLY DECISION IN THE EXERCISE: WHY THE CUT IS CONTIGUOUS AND FROM THE END
    ----------------------------------------------------------------------------
    The usual reflex would be to shuffle and split. Here that is a mistake.

    The training windows OVERLAP: the one starting at position 100 and the one starting at
    101 share 511 of their 512 tokens. If you split at random — at the token level or even
    the window level — your validation set would be full of fragments the model has already
    seen.

    The symptom would be beautiful and misleading: a very low validation loss, almost
    identical to the training one, and never any sign of overfitting. You would be measuring
    memorization and calling it generalization.

    By cutting a contiguous block off the end, what you set aside are COMPLETE stories the
    model has never seen.

    TWO DETAILS
    -----------
    **The `max(1, ...)`.** With a 50-token corpus and `val_fraction=0.005`, `int(50*0.005)`
    gives 0 and you would be left with no validation set.

    **Return VIEWS, not copies.** Numpy slicing does not copy, and that is what you want:
    with 500M tokens, a needless `.copy()` would be 1 GB of RAM thrown away. There is a test
    that checks it with `np.shares_memory`.

    Args:
        tokens: 1-D array with the whole corpus.
        val_fraction: the fraction for validation. Must be in (0, 1).

    Returns:
        `(train, val)`.

    Raises:
        ValueError: if `val_fraction` is not in (0, 1), or if it would leave training empty.
    """
    raise NotImplementedError("TODO: module 04, exercise 2 - train_val_split")


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device | str | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draws a batch of random windows. It is the most-executed function in the course.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Six steps.

        1. If `rng` is None, create one: `rng = rng or np.random.default_rng()`

        2. Compute how far you can start, and validate:

               max_start = len(data) - context_length - 1
               if max_start < 1:
                   raise ValueError(f"the corpus ({len(data)}) is shorter than the context ...")

        3. Pick the starting positions:

               starts = rng.integers(0, max_start, size=batch_size)

        4. Stack the windows, the input one and the shifted one:

               x_np = np.stack([data[i : i+context_length]     for i in starts]).astype(np.int64)
               y_np = np.stack([data[i+1 : i+1+context_length] for i in starts]).astype(np.int64)

        5. Turn them into tensors:

               x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)

        6. If there is a `device`, move them:

               device = torch.device(device)
               if device.type == "cuda":
                   x = x.pin_memory().to(device, non_blocking=True)
                   y = y.pin_memory().to(device, non_blocking=True)
               else:
                   x, y = x.to(device), y.to(device)

    THE IDEA, WITH NUMBERS
    ----------------------
        data = [5, 8, 2, 9, 1, 7, ...]     starting at 0, with context_length = 4

            x = [5, 8, 2, 9]      <- what the model sees
            y = [8, 2, 9, 1]      <- what it has to predict

    Read it column by column:

        seeing [5]          -> predict 8
        seeing [5,8]        -> predict 2
        seeing [5,8,2]      -> predict 9
        seeing [5,8,2,9]    -> predict 1

    ONE 4-token window is FOUR training examples. With a context of 512, that is 512
    predictions per sample, and a 48x512 batch is 24,576. That is why language models use
    data so well.

    (This works thanks to module 06's causal mask, which stops position 2 from seeing token
    3. Without it the model would see the answer.)

    THE `-1` IN STEP 2 IS THE EXERCISE'S OFF-BY-ONE
    -----------------------------------------------
    `y` needs one token MORE than the end of `x`. If `x` reaches `i + context_length - 1`,
    `y` reaches `i + context_length`. Without the `-1`, the last possible window overflows.

    And numpy does not raise on out-of-range slicing: it simply returns fewer elements. What
    you will see is an `np.stack` failing on incompatible shapes, three lines further down
    and with no clue about the real cause.

    THE `.astype(np.int64)` IN STEP 4 DOES TWO THINGS
    ------------------------------------------------
    The obvious one: `nn.Embedding` requires `int64` indices, and the data is `uint16`.

    The less obvious one: the conversion COPIES. Without that copy, `torch.from_numpy` would
    be left pointing at disk-mapped memory and every model access would potentially be a file
    read.

    THE `pin_memory` IN STEP 6, CUDA ONLY
    -------------------------------------
    "Pinned" memory is memory the operating system commits to not moving, which lets the GPU
    read it by DMA with no CPU involvement. With `non_blocking=True` the call returns
    immediately and the copy overlaps with whatever the GPU is computing.

    On MPS it does not apply (the memory is unified, there is no copy to overlap) and neither
    does it on CPU.

    Args:
        data: 1-D array of tokens (usually a uint16 `np.memmap`).
        batch_size: how many windows.
        context_length: how many tokens per window.
        device: where to move the tensors. `None` leaves them on the CPU.
        rng: numpy generator. Pass one with a fixed seed and you will always get the same
            batch, which is what lets the test compare against the reference.

    Returns:
        `(x, y)`, both `int64` tensors of shape `(batch_size, context_length)`.

    Raises:
        ValueError: if the corpus is shorter than `context_length + 1`.
    """
    raise NotImplementedError("TODO: module 04, exercise 3 - get_batch")
