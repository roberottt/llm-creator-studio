"""Module 09 - Positional information and RoPE.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement in order -> `llmfs check 09` -> `llmfs hint 09 -e N`
-> `SOLUTION.md` has the complete code.

WHAT YOU ARE GOING TO BUILD
===========================

The way of telling the model what position each token is in:

    sinusoidal_embeddings  (ex. 1)  the 2017 paper's table (historical)
    rope_frequencies       (ex. 2)  precompute the rotation angles
            |
            v
    apply_rope             (ex. 3)  rotate Q and K. ONE LINE, but only after ex. 2

Exercise 2 is the hard one, and it is hard because of a single step: the one that duplicates
the frequencies. Exercise 3 is a single line, but it only makes sense after understanding
exercise 2.

Watch out for one thing: exercise 1 is NOT used by our model. It is the 2017 option and it is
here because it introduces the frequency ladder that RoPE reuses, because you will run into it
in a lot of code, and because the demo trains it to compare it against the other two. The
"Exercise 1" section of `THEORY.md` explains this.

And a familiar face: the `apply_rope` of exercise 3 is one you already called in module 06,
inside `MultiHeadAttention`, imported from the reference with a comment telling you to ignore it
for now. Today you write it, and the `cos` and `sin` tables of that signature come out of
exercise 2.

`THEORY.md` is ordered just like this list and each docstring here tells you which section it
maps to.

THE PROBLEM IT SOLVES
=====================

Look again at the attention formula (module 06): it is a weighted sum, and a sum has no
order. To the attention mechanism, "the dog bites the man" and "the man bites the dog"
produce exactly the same thing.

VOCABULARY YOU ARE GOING TO NEED
================================

- **positional embedding**: the information that tells the model where each token is.
- **absolute / relative position**: "I am token 7" versus "I am two positions behind that
  one". The relative one generalizes better.
- **RoPE** (Rotary Position Embedding): instead of ADDING something to the vector, it
  ROTATES it by an angle proportional to the position.
- **head_dim**: the dimension of each attention head. In our model, 40. RoPE works on this,
  not on d_model's 320.
- **extrapolate**: using the model with sequences longer than the ones it saw in training.

    llmfs demo 09     draws the frequencies and measures extrapolation for real
"""

from __future__ import annotations

import math

import torch


def sinusoidal_embeddings(seq_len: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    """The 2017 paper's table of sines and cosines.

    Context in `THEORY.md`: section "Exercise 1: sines and cosines", which starts by explaining
    why you are writing something the final model does not use, and carries the whole 5x4 table
    your function has to return, read by columns so the frequency ladder shows up.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Five lines, no loops.

        1. The positions, as a column:

               position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)   # (T, 1)

        2. The frequencies, one per PAIR of dimensions:

               div_term = torch.exp(
                   torch.arange(0, d_model, 2, dtype=torch.float32)
                   * (-math.log(base) / d_model)
               )                                                                    # (d/2,)

        3. The empty table:

               embeddings = torch.zeros(seq_len, d_model)

        4. Fill it, interleaving:

               embeddings[:, 0::2] = torch.sin(position * div_term)
               embeddings[:, 1::2] = torch.cos(position * div_term)

        5. Return it.

    HOW IT WORKS
    ------------
    `position * div_term` broadcasts `(T,1)` against `(d/2,)` and gives `(T, d/2)`: every
    angle of every position at once, with no loops.

    `[:, 0::2]` means "every row, columns from 0 in steps of two", that is, the even ones.
    And `[:, 1::2]` the odd ones. It is the way to interleave sine and cosine without writing
    a `for`.

    THE TRICK IN STEP 2, WORTH KNOWING
    ----------------------------------
    `exp(-log(base) * 2i/d)` is mathematically identical to `base ** (-2i/d)`, but far more
    STABLE. Raising 10000 to a large negative power loses floating-point precision; going
    through logarithms does not.

    A general rule that will serve you elsewhere: if you see a power with a large exponent,
    `exp(log(...))` is usually better.

    THE IDEA: A BINARY COUNTER
    --------------------------
    When counting in binary, each bit oscillates at a different rate:

        0000    the rightmost bit changes at every step
        0001    the next one, every two
        0010    the next one, every four

    The combination of all of them identifies a number uniquely. Here it is the same but with
    continuous waves: the first pairs of dimensions oscillate fast (distinguishing
    neighbouring positions) and the last ones extremely slowly (distinguishing the start from
    the end).

    Args:
        seq_len: how many positions to generate.
        d_model: the model dimension. Assumed even.
        base: the paper's constant, 10000.

    Returns:
        A `(seq_len, d_model)` tensor.
    """
    raise NotImplementedError("TODO: module 09, exercise 1 - sinusoidal_embeddings")


def rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precomputes the cosine and sine tables RoPE will use.

    Context in `THEORY.md`: section "Exercise 2: the angle tables". If step 4 (the `cat` that
    duplicates) looks like a bug to you, that section has the complete tables for head_dim=4 with
    the repeated columns marked, and the reason: with the halves convention, both components of a
    pair need THE SAME angle.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Five steps.

        1. Validate that `head_dim` is EVEN and raise `ValueError` if not. (RoPE rotates
           pairs: with an odd dimension one would be left over.)

        2. The inverse frequencies, one per pair:

               inv_freq = 1.0 / (theta ** (
                   torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
               ))                                                        # (head_dim/2,)

        3. Every angle at once:

               positions = torch.arange(max_seq_len, dtype=torch.float32)
               angles = torch.outer(positions, inv_freq)                 # (T, head_dim/2)

        4. DUPLICATE by halves:

               angles = torch.cat([angles, angles], dim=-1)              # (T, head_dim)

        5. Return the cosine and the sine, moved to `device` if asked:

               cos, sin = angles.cos(), angles.sin()
               if device is not None:
                   cos, sin = cos.to(device), sin.to(device)
               return cos, sin

    `torch.outer(a, b)[i,j] = a[i] * b[j]`, which is exactly what is needed: every
    position x frequency combination.

    STEP 4 IS THE CONFUSING ONE, AND THIS IS WHY
    --------------------------------------------
    There are two ways of pairing the dimensions for rotation:

        - the original paper pairs CONSECUTIVE ones:  (x0,x1), (x2,x3), ...
        - Llama and HuggingFace pair by HALVES:       (x0, x_{d/2}), (x1, x_{d/2+1}), ...

    We use the halves one. With that convention, dimension `i` and dimension `i + head_dim/2`
    form a pair and need THE SAME angle. That is why each frequency appears TWICE and the
    tables have `head_dim` columns instead of `head_dim/2`.

    The two conventions are equivalent up to a permutation of the dimensions, which the
    network learns without noticing. The halves one won because it makes exercise 3 ONE LINE
    with no reordering.

    WHAT IS HAPPENING
    -----------------
    RoPE does not ADD anything to the vector: it ROTATES it. Pair `i` at position `pos` is
    rotated by an angle `pos * theta^(-2i/head_dim)`.

    The frequencies run from fast to slow: the first pairs turn quickly and capture short
    relationships, the last ones turn extremely slowly and capture long distances.

    Args:
        head_dim: each head's dimension (40 in the final model). It has to be even.
        max_seq_len: up to which position to precompute.
        theta: the base, 10000 by default.
        device: where to leave the tensors.

    Returns:
        `(cos, sin)`, both of shape `(max_seq_len, head_dim)`.

    Raises:
        ValueError: if `head_dim` is odd.
    """
    raise NotImplementedError("TODO: module 09, exercise 2 - rope_frequencies")


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies the positional rotation to Q or to K.

    Context in `THEORY.md`: section "Exercise 3: actually rotating", with the check that this line
    IS the rotation matrix and an example with head_dim=4 you can follow by hand: the vector
    [1, 0, 0, 1] rotated at positions 0, 1 and 2, with the two pairs taken separately and the norm
    unchanged.

    WHAT YOU HAVE TO WRITE
    ----------------------
    **A helper** (put it as a separate function in the same file):

        def rotate_half(x):
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([-x2, x1], dim=-1)

    **And the function**, three lines:

        seq_len = x.shape[-2]
        cos = cos[:seq_len].to(dtype=x.dtype, device=x.device)
        sin = sin[:seq_len].to(dtype=x.dtype, device=x.device)
        return x * cos + rotate_half(x) * sin

    WHERE THAT LAST LINE COMES FROM
    -------------------------------
    Rotating a pair `(x1, x2)` by an angle `t` is the usual rotation matrix:

        x1' = x1*cos(t) - x2*sin(t)
        x2' = x2*cos(t) + x1*sin(t)

    And now check that `x * cos + rotate_half(x) * sin` produces exactly that, knowing that
    `rotate_half([a, b]) = [-b, a]`:

        component 1:  x1*cos + (-x2)*sin  =  x1*cos - x2*sin    OK
        component 2:  x2*cos + ( x1)*sin  =  x2*cos + x1*sin    OK

    TWO DETAILS THAT BREAK IF YOU SKIP THEM
    ---------------------------------------
    **The slice `cos[:seq_len]`.** The tables are precomputed up to `max_seq_len` (512 in the
    final model) and your sequence is almost never exactly that long. Without slicing, the
    broadcast fails or — worse — succeeds by accident with the wrong shapes.

    **The `.to(dtype=x.dtype)`.** Under AMP the tables are in fp32 and `x` arrives in fp16.
    Mixing them makes PyTorch promote, and you end up computing at a precision you did not
    want.

    NO `unsqueeze` IS NEEDED
    ------------------------
    `x` is `(B, n_heads, T, head_dim)` and `cos` is `(T, head_dim)`. The broadcast aligns
    from the right and takes care of the first two dimensions by itself.

    WHY THIS WORKS (the property that justifies RoPE)
    -------------------------------------------------
    The dot product of two rotated vectors depends ONLY on the difference of angles:

        <R(m)q, R(n)k> = <q, R(n-m)k>

    That is: the attention score between tokens 5 and 3 is the SAME as between 105 and 103.
    The model learns "the token two positions back", not "token number 3". The demo checks it
    with numbers and it comes out equal to the last decimal.

    And on top of that, rotating does NOT change the vector's length, which adding a
    positional embedding does.

    Args:
        x: `(B, n_heads, T, head_dim)`, usually Q or K.
        cos, sin: `(max_seq_len, head_dim)`, from `rope_frequencies`.

    Returns:
        The same shape as `x`.
    """
    raise NotImplementedError("TODO: module 09, exercise 3 - apply_rope")
