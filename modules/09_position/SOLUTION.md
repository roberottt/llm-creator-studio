# 09 — Annotated solution

## Exercise 1 — `sinusoidal_embeddings`

```python
position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)      # (T, 1)
div_term = torch.exp(
    torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(base) / d_model)
)                                                                        # (d/2,)

embeddings = torch.zeros(seq_len, d_model)
embeddings[:, 0::2] = torch.sin(position * div_term)
embeddings[:, 1::2] = torch.cos(position * div_term)
return embeddings
```

**The `exp(-log(base) · 2i/d)` trick.** It is mathematically identical to `base ** (-2i/d)`,
but far more stable. Raising 10000 to a large negative power loses floating-point precision;
going through logarithms does not. It is the standard idiom and it is worth having in the
repertoire: whenever you see a power with a large exponent, `exp(log(...))` is usually
better.

**`position * div_term` broadcasts by itself.** `(T, 1)` against `(d/2,)` gives `(T, d/2)`:
every angle of every position at once, with no loops.

**`0::2` and `1::2`** mean "from 0 in steps of two" and "from 1 in steps of two". It is the
way to interleave sines and cosines without writing a `for`.

## Exercise 2 — `rope_frequencies`

```python
if head_dim % 2 != 0:
    raise ValueError(f"head_dim ({head_dim}) has to be even")

inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
positions = torch.arange(max_seq_len, dtype=torch.float32)

angles = torch.outer(positions, inv_freq)          # (T, head_dim/2)
angles = torch.cat([angles, angles], dim=-1)       # (T, head_dim)

cos, sin = angles.cos(), angles.sin()
if device is not None:
    cos, sin = cos.to(device), sin.to(device)
return cos, sin
```

**The confusing step is the duplication.** The tables come out with `head_dim` columns, not
`head_dim/2`, and it is not a mistake: it is what makes exercise 3 a single line.

The reason is the pairing convention. With the **halves** convention we use (Llama's and
HuggingFace's), dimension `i` is paired with `i + head_dim/2`. Both need the **same angle**,
so each frequency appears twice, once in each half of the table. With the original paper's
convention — pairing consecutive ones — you would have to interleave instead of concatenate,
and `apply_rope` would need to reorder dimensions.

The two conventions are equivalent up to a permutation of the dimensions, which the network
learns without noticing. One is cleaner to implement and that is why it won.

**`torch.outer(a, b)[i,j] = a[i] * b[j]`.** Exactly what is needed: every position ×
frequency combination.

## Exercise 3 — `apply_rope`

```python
def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    seq_len = x.shape[-2]
    cos = cos[:seq_len].to(dtype=x.dtype, device=x.device)
    sin = sin[:seq_len].to(dtype=x.dtype, device=x.device)
    return x * cos + rotate_half(x) * sin
```

**Where that line comes from.** Rotating a pair $(x_1, x_2)$ by an angle $t$:

```
x1' = x1·cos(t) − x2·sin(t)
x2' = x2·cos(t) + x1·sin(t)
```

And now check that `x * cos + rotate_half(x) * sin` produces exactly that, knowing that
`rotate_half([a, b]) = [-b, a]`:

```
component 1:  x1·cos + (−x2)·sin  =  x1·cos − x2·sin    ✓
component 2:  x2·cos + ( x1)·sin  =  x2·cos + x1·sin    ✓
```

It is the good old rotation matrix, written with vectorized operations.

**The slice `cos[:seq_len]`** is not optional. The tables are precomputed up to
`max_seq_len` (512 in the final model) and your sequence almost never has exactly that
length. Without slicing, the broadcast fails or — worse — succeeds by accident with the
wrong shapes.

**The `.to(dtype=x.dtype)`** matters under AMP: the tables are created in fp32 and `x`
arrives in fp16. Mixing types makes PyTorch promote, and you end up with half the
computation at a precision you did not want.

**No `unsqueeze` is needed.** `x` is `(B, n_heads, T, head_dim)` and `cos` is
`(T, head_dim)`; the broadcast aligns from the right and takes care of the first two
dimensions by itself.

## What you should see in the demo

**The relative invariance**, which is the property that justifies everything:

| positions (q,k) | distance | score |
|---|---|---|
| (0, 3) | 3 | 0.1264068037 |
| (2, 5) | 3 | 0.1264068037 |
| (100, 103) | 3 | 0.1264068037 |
| (200, 203) | 3 | 0.1264068037 |

**Identical to the last decimal.** The model does not learn "token number 3", it learns "the
token three positions back", and that is why it can apply what it learned anywhere in the
sequence.

**The extrapolation**, which is the experiment that really matters. Three identical models
except for the positional encoding, trained with a context of 32:

| context | learned | sinusoidal | RoPE |
|---|---|---|---|
| 16 | 2.1139 | 2.1088 | 2.0665 |
| **32 (trained)** | **2.1296** | **2.0823** | **2.0376** |
| 48 | **cannot** | 2.3490 | 2.1049 |
| 128 | **cannot** | 2.7601 | 2.6324 |

Three readings.

**The learned one has a hard ceiling.** It is not that it works badly beyond 32: it *cannot*.
There is no row in the table to look up. The code raises an exception, and that is the honest
thing to do.

**RoPE wins at every context**, including inside the trained range. Relative encoding helps
even without extrapolating.

**And both degrade.** From 2.04 to 2.63 is 29% worse. Here it is worth being careful with
what you read out there: it is often repeated that "RoPE extrapolates", and what the demo
shows is that it extrapolates *better than the alternatives*, not that it extrapolates
*well*.

The reason: the slow frequencies barely complete a fraction of a turn within the trained
range — look at the period table from the first experiment — so the large angles are
literally unseen territory. There is a whole family of techniques for extending the context
after training (position interpolation, NTK-aware scaling, YaRN) precisely because direct
extrapolation is not enough.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def sinusoidal_embeddings(seq_len: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    # exp(-ln(base) * 2i/d) is numerically more stable than base ** (-2i/d)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(base) / d_model)
    )  # (d/2,)

    embeddings = torch.zeros(seq_len, d_model)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings


def rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) has to be even: RoPE rotates pairs.")

    # theta^(-2i/d) para i = 0, 1, ..., d/2-1
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)

    angles = torch.outer(positions, inv_freq)  # (T, head_dim/2)
    angles = torch.cat([angles, angles], dim=-1)  # (T, head_dim), duplicated by halves

    cos, sin = angles.cos(), angles.sin()
    if device is not None:
        cos, sin = cos.to(device), sin.to(device)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[-2]
    cos = cos[:seq_len].to(dtype=x.dtype, device=x.device)
    sin = sin[:seq_len].to(dtype=x.dtype, device=x.device)
    return x * cos + rotate_half(x) * sin
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
