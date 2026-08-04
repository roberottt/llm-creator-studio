"""Reference for module 09: how the model is told what position each token is in.

Attention is a weighted sum, and a sum has no order: without positional information,
"the dog bites the man" and "the man bites the dog" produce exactly the same output.
"""

from __future__ import annotations

import math

import torch


def sinusoidal_embeddings(seq_len: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    """The sine and cosine table from the original 2017 paper.

    $$PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{base^{2i/d}}\\right), \\qquad
      PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{base^{2i/d}}\\right)$$

    Each pair of dimensions oscillates at a different frequency: the first ones change fast
    (they distinguish neighbouring positions), the last ones change extremely slowly (they
    distinguish the start of the sequence from the end). Together they form a unique
    signature per position, much like a binary number identifies a value with bits that
    flip at different rates.

    They are added to the token embeddings, not concatenated.

    Returns:
        `(seq_len, d_model)`.
    """
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
    """Precompute RoPE's cosine and sine tables.

    RoPE does not ADD anything to the embedding: it ROTATES pairs of dimensions by an angle
    proportional to the position. Pair `i` is rotated by an angle `pos * theta^(-2i/d)`.

    The frequencies run from fast to slow just like the sinusoidal ones: the first pairs
    spin quickly and capture short relative positions, the last ones spin extremely slowly
    and capture long distances.

    PAIRING CONVENTION: the "halves" convention is used (Llama's and HuggingFace's), not
    the consecutive-pairs one from the original paper. That is, dimension `i` is paired
    with `i + head_dim/2`, not with `i+1`. The two versions are equivalent up to a
    permutation of the dimensions, which the network learns without noticing, and the
    halves version is implemented with much cleaner vector operations.

    That is why the returned tables have `head_dim` columns and not `head_dim/2`: each
    frequency appears DUPLICATED, once for each half. That makes `apply_rope` a one-liner.

    Returns:
        `(cos, sin)`, both of shape `(max_seq_len, head_dim)`.
    """
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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """`[a, b]` -> `[-b, a]`, splitting the last dimension in half.

    It is the vectorized version of rotating a 2-D vector by 90 degrees. Combined with cos
    and sin it reproduces a rotation by any angle:

        x * cos + rotate_half(x) * sin

    which for a pair `(x1, x2)` is exactly

        x1' = x1*cos - x2*sin
        x2' = x2*cos + x1*sin

    that is, the good old rotation matrix.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the positional rotation to Q or to K.

    Args:
        x: `(B, n_heads, T, head_dim)`.
        cos, sin: `(max_seq_len, head_dim)` from `rope_frequencies`. They are sliced to `T`
            and broadcast over batch and heads.

    Returns:
        The same shape as `x`.

    Why this works: the dot product of two rotated vectors depends ONLY on the difference
    of angles, that is, on the relative distance between the two positions.

        <R(m)q, R(n)k> = <q, R(n-m)k>

    So the attention score between tokens 5 and 3 is the same as between 105 and 103. The
    model learns "the token two positions back", not "token number 3". That is why RoPE
    extrapolates better to sequences longer than the training ones, and it is the property
    the module's demo measures.

    It is applied only to Q and K, never to V: what has to depend on position is the
    attention SCORES, not the content being transported.
    """
    seq_len = x.shape[-2]
    cos = cos[:seq_len].to(dtype=x.dtype, device=x.device)
    sin = sin[:seq_len].to(dtype=x.dtype, device=x.device)
    return x * cos + rotate_half(x) * sin


class LearnedPositionalEmbedding(torch.nn.Module):
    """Learned positional embeddings, as in GPT-2. For comparison in the demo.

    A `(max_seq_len, d_model)` table trained like any other parameter. Simple, and it
    works, but it has a hard ceiling: for positions beyond `max_seq_len` there is no row to
    look up, and the model cannot process them at all.
    """

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions)
