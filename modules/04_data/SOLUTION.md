# 04 — Annotated solution

## Exercise 1 — `pack_tokens_uint16`

```python
if vocab_size > MAX_UINT16:
    raise ValueError(f"vocab_size={vocab_size} does not fit in uint16...")

array = np.asarray(ids, dtype=np.int64)
if array.size and (array.min() < 0 or array.max() >= vocab_size):
    raise ValueError(
        f"ids outside the vocabulary [0, {vocab_size}): "
        f"min={array.min()}, max={array.max()}"
    )
return array.astype(np.uint16)
```

**The order matters: `int64` first, check, `uint16` afterwards.** If you converted straight
to `uint16`, the overflow would already have happened and you would be checking data that is
already corrupted. You have to validate in a type where everything fits.

**Why this deserves its own exercise.** NumPy's behaviour here is genuinely dangerous:

```python
np.array([65536], dtype=np.int64).astype(np.uint16)   # -> array([0], dtype=uint16)
```

No exception, no warning. In a data pipeline, a silent bug that corrupts a small fraction of
the corpus is one of the worst there is: the model trains, the loss drops, everything
*appears* to work, and the result is simply worse than it should be with nothing pointing at
the cause. The validation costs three lines.

**The `array.size and ...`.** On an empty array, `.min()` raises `ValueError` with a message
about reduction operations on empty sequences — a real error, but one that has nothing to do
with what you are validating and sends you off track. The short-circuit avoids it.

**The error message states the values.** `"ids out of range"` does not help; `"max=9999"`
tells you immediately that your tokenizer is producing ids it should not, and by how much.

## Exercise 2 — `train_val_split`

```python
if not 0.0 < val_fraction < 1.0:
    raise ValueError(...)

n_val = max(1, int(len(tokens) * val_fraction))
if n_val >= len(tokens):
    raise ValueError("val_fraction leaves the training set empty")
return tokens[:-n_val], tokens[-n_val:]
```

Two lines of logic and one design decision that does matter.

**Why contiguous and from the end.** It is the only thing to understand in this exercise.
The training windows overlap massively: the one starting at position 100 and the one
starting at 101 share 511 of their 512 tokens. If you split at random — at the token level
or even the window level — the validation set would be riddled with fragments the model
already saw in training.

The symptom would be beautiful and misleading: a very low validation loss, almost identical
to the training one, and never any sign of overfitting. You would be measuring memorization
and calling it generalization.

By cutting a contiguous block off the end, what you set aside are complete TinyStories
stories the model has never seen. It is the right unit: the validation set has to be
independent of the training one *in the unit that matters*, and here the unit is not the
token, it is the story.

**Returning views, not copies.** NumPy slicing does not copy. With 500M tokens, a needless
`.copy()` would be 1 GB of RAM thrown in the bin. The test
`test_it_returns_views_and_not_copies` checks it with `np.shares_memory`.

**The `max(1, ...)`.** With a 50-token corpus and `val_fraction=0.005`, `int(50*0.005)` is
0, and you would be left with no validation set. The `max` guarantees at least one token.

## Exercise 3 — `get_batch`

```python
rng = rng or np.random.default_rng()
max_start = len(data) - context_length - 1
if max_start < 1:
    raise ValueError(...)

starts = rng.integers(0, max_start, size=batch_size)
x_np = np.stack([data[i : i+context_length]     for i in starts]).astype(np.int64)
y_np = np.stack([data[i+1 : i+1+context_length] for i in starts]).astype(np.int64)

x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)

if device is not None:
    device = torch.device(device)
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
return x, y
```

**The `-1` in `max_start`.** It is the exercise's off-by-one. `y` needs one token beyond the
end of `x`: if `x` reaches `i + context_length - 1`, `y` reaches `i + context_length`.
Without the `-1`, the last possible window overflows the array. NumPy does not raise on
out-of-range slicing — it simply returns fewer elements — so what you would get is an
`np.stack` failing with a message about incompatible shapes, three lines further down and
with no clue about the real cause.

**The `.astype(np.int64)` does two things at once.** The obvious one: `nn.Embedding`
requires `int64` indices, and the data is `uint16`. The less obvious one: the conversion
**copies**. Without that copy, `torch.from_numpy` would be left pointing at disk-mapped
memory, and every model access would potentially be a file read.

**`pin_memory` + `non_blocking`, CUDA only.** "Pinned" memory is memory the operating system
commits to not moving, which lets the GPU read it by DMA with no CPU involvement. With
`non_blocking=True`, the call returns immediately and the copy overlaps with whatever the
GPU is computing. On MPS it does not apply — the memory is unified, there is no copy to
overlap — and neither does it on CPU.

**The `rng` generator as a parameter.** It is what makes the exercise testable: two calls
with `np.random.default_rng(42)` produce exactly the same batch, and the test can compare it
with the reference element by element. Without it, you could only check shapes.

**About sampling with replacement.** Picking random positions on each call means some
windows will come up several times and others never. It is not an "epoch" in the strict
sense. It is what nanoGPT does and it works well with a single pass over 500M tokens; with
many epochs over a small corpus, a shuffled traversal would give better coverage guarantees.

## What you should see in the demo

The `x`/`y` correspondence over real text:

```
x = 'ot accidenta'
y = 't accidental'

seeing 'o'      -> it must predict 't'
seeing 'ot'     -> it must predict ' '
seeing 'ot '    -> it must predict 'a'
```

And the number that sums the module up: **an 8×64 batch is 512 predictions**, not 8. With
the final configuration, 48×512 is 24,576 predictions per batch. That is the reason training
a language model is so data-efficient compared with almost any other supervised learning
task.

The speed section is worth looking at carefully in module 12: if `get_batch` takes longer
than a training step, the GPU spends its time waiting and the data loading has to be moved
to a separate thread.
