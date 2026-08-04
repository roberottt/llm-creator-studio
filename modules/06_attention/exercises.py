"""Module 06 - Self-attention.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement in order -> `llmfs check 06` -> `llmfs hint 06 -e N`
-> `SOLUTION.md` has the complete code.

The "the cat I saw yesterday was sleeping" example, worked by hand with three words and two
dimensions, is EXACTLY what you are going to program. Keep it in front of you.

WHAT YOU ARE GOING TO BUILD
===========================

The heart of the Transformer. Three exercises that fit together like this:

    causal_mask             (ex. 1)  stop a token from looking into the future
            |
            v
    single_head_attention   (ex. 2)  the formula, with a single head
            |
            v
    MultiHeadAttention      (ex. 3)  eight in parallel, which is what the model uses

Exercise 2 is four lines, and each one has a trap. Exercise 3 is the same computation with
one more dimension.

THE IDEA, IN ONE SENTENCE
=========================

Each token asks a QUESTION, all the previous ones ANSWER, how well each answer fits is
measured, and their CONTENT is mixed accordingly.

    output = softmax( Q K^T / sqrt(d_k) + mask ) V

VOCABULARY YOU ARE GOING TO NEED
================================

- **Q, K, V** (query, key, value): the three projections. The query is the question a token
  asks, the key is the label each token advertises itself with, and the value is the content
  it contributes if it gets chosen.
- **softmax**: turns a list of arbitrary numbers into probabilities that sum to 1. It
  exponentiates each one and divides by the sum.
- **dot product**: multiplying two vectors component by component and adding. It measures
  how similar they are: the more aligned, the larger the number.
- **head**: one independent attention. The model has 8 in parallel, each working in 40
  dimensions.
- **causal mask**: the one that stops position 3 from looking at position 4. Without it the
  model would see the answer.

    llmfs demo 06     trains an attention model and draws what each letter looks at
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# RoPE is from module 09; it is given ready-made so MultiHeadAttention can use it. If you
# have not got there yet, ignore it: this module's tests pass cos=None and sin=None.
from llmfs.reference import apply_rope


def causal_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """The triangular mask that stops you looking into the future.

    WHAT YOU HAVE TO WRITE
    ----------------------
    One line.

        return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()

    `tril` = *triangular lower*. By default it uses `diagonal=0`, which INCLUDES the
    diagonal, which is what you want: a token can look at itself.

    WHAT IT HAS TO PRODUCE
    ----------------------
    For `seq_len = 4`, with the convention `True = looking IS allowed`:

        [[ True, False, False, False],     token 0 only sees itself
         [ True,  True, False, False],     token 1 sees token 0 and itself
         [ True,  True,  True, False],
         [ True,  True,  True,  True]]

    IF IT COMES OUT WRONG
    ---------------------
        - inverted        -> you used `triu` instead of `tril`
        - diagonal False  -> you passed `diagonal=-1`

    WHY IT IS NEEDED
    ----------------
    During training we give the model the whole sentence at once and ask it to predict each
    token from the previous ones. Without a mask, position 3 could look at position 4, which
    is literally the answer it has to give.

    That is the most expensive bug in the course: the loss drops spectacularly, everything
    looks wonderful, and the model is useless because at generation time that future does
    not exist.

    A WARNING FOR LATER
    -------------------
    We use `True = allowed` because it is `F.scaled_dot_product_attention`'s convention. But
    PyTorch's `nn.MultiheadAttention` uses the OPPOSITE: its boolean `attn_mask` marks with
    `True` what has to be FORBIDDEN. That is why the test comparing against it passes
    `~mask`. It is a real inconsistency inside the library itself.

    Args:
        seq_len: the sequence length.
        device: where to create the tensor.

    Returns:
        A boolean `(seq_len, seq_len)` tensor.
    """
    raise NotImplementedError("TODO: module 06, exercise 1 - causal_mask")


def single_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-head attention. The heart of the Transformer, in four lines.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Four steps, and each one has a trap.

        1. Take the dimension and compute the SCORES:

               d_k = q.shape[-1]
               scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

        2. If there is a mask, cover what is forbidden:

               if mask is not None:
                   scores = scores.masked_fill(~mask, float("-inf"))

        3. Turn it into weights that sum to 1:

               weights = F.softmax(scores, dim=-1)

        4. Mix the values and return both things:

               return weights @ v, weights

    WHAT IS HAPPENING AT EACH STEP
    ------------------------------
    **Step 1.** `q @ k.transpose(-2,-1)` multiplies `(B, T, d_k)` by `(B, d_k, S)` and gives
    `(B, T, S)`. Cell `[b, i, j]` is the dot product of token `i`'s query with token `j`'s
    key: how interested `i` is in token `j`.

    **Step 2.** The `~` inverts the boolean: where the mask says `False` (forbidden), put
    `-inf`. Since `e^(-inf) = 0`, those positions receive exactly zero weight.

    **Step 3.** Softmax exponentiates and normalizes, so each row ends up summing to 1.

    **Step 4.** `weights @ v` is the weighted average: each token takes away a mixture of the
    values, weighted by how interested it is in each.

    THE THREE TRAPS
    ---------------
    **`transpose(-2, -1)` with NEGATIVE indices.** They count from the end, so they work the
    same with `(B, T, d)` as with `(B, heads, T, d)`. If you write `transpose(1, 2)`, this
    exercise passes and exercise 3 breaks with a shape error that is hard to connect back to
    the cause.

    **`dim=-1` in the softmax.** You are normalizing over WHO IS BEING LOOKED AT, so each row
    sums to 1. With `dim=-2` you would normalize over who is looking, which means nothing.
    And it gives no error: the shapes are identical, the model trains, and it learns worse.
    There is a test that checks each row sums to 1.

    **The mask goes BEFORE the softmax.** If you zeroed the weights afterwards, the rows
    would stop summing to 1 and you would be scaling the output by an arbitrary factor that
    differs at every position.

    WHY WE DIVIDE BY sqrt(d_k)
    --------------------------
    The dot product of two `d_k`-dimensional vectors has variance `d_k`. Without dividing,
    with a large `d_k` the scores shoot up, and since softmax is exponential, it returns
    almost `[0,0,...,1,...,0]`: attention collapses into picking a single token.

    And the real problem is not the forward pass, it is the GRADIENT: the softmax's
    derivative is `p(1-p)`, and with `p` pinned to 0 or 1 it is practically zero. The layer
    stops learning. The module's demo measures it.

    Args:
        q: `(B, T, d_k)` the queries.
        k: `(B, S, d_k)` the keys.
        v: `(B, S, d_v)` the values.
        mask: `(T, S)` or `(B, T, S)` boolean, `True` = allowed. `None` = no mask.

    Returns:
        `(output, weights)` with output `(B, T, d_v)` and weights `(B, T, S)`. The weights
        are returned because they are what the demo's heatmap draws.
    """
    raise NotImplementedError("TODO: module 06, exercise 2 - single_head_attention")


class MultiHeadAttention(nn.Module):
    """Several attentions in parallel, each with its own projections.

    WHAT YOU HAVE TO WRITE
    ----------------------
    **In `__init__`:**

        1. Validate that `d_model` is divisible by `n_heads`, and raise `ValueError` if not.

        2. Store: `self.d_model`, `self.n_heads`, `self.head_dim = d_model // n_heads`,
           `self.dropout` and `self.use_sdpa`.

        3. Create the four projections and the two dropouts. The names matter (the test
           copies weights by name):

               self.q_proj = nn.Linear(d_model, d_model, bias=bias)
               self.k_proj = nn.Linear(d_model, d_model, bias=bias)
               self.v_proj = nn.Linear(d_model, d_model, bias=bias)
               self.out_proj = nn.Linear(d_model, d_model, bias=bias)
               self.attn_dropout = nn.Dropout(dropout)
               self.resid_dropout = nn.Dropout(dropout)

    **Two helpers** (write them as methods):

        def _split_heads(self, x):          # (B, T, d_model) -> (B, n_heads, T, head_dim)
            batch, seq_len, _ = x.shape
            return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        def _merge_heads(self, x):          # (B, n_heads, T, head_dim) -> (B, T, d_model)
            batch, _, seq_len, _ = x.shape
            return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    **In `forward`:**

        1. `seq_len = x.shape[1]`, and if `mask` is None, build it with `causal_mask`.

        2. Project and split into heads:

               q = self._split_heads(self.q_proj(x))
               k = self._split_heads(self.k_proj(x))
               v = self._split_heads(self.v_proj(x))

        3. If `cos` and `sin` are not None, apply RoPE to q and k (NEVER to v):

               q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        4. The attention. If `self.use_sdpa` and no weights are requested:

               out = F.scaled_dot_product_attention(
                   q, k, v, attn_mask=mask,
                   dropout_p=self.dropout if self.training else 0.0,
               )
               weights = None

           Otherwise, the explicit computation (the same as exercise 2, but with 4
           dimensions):

               scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
               scores = scores.masked_fill(~mask, float("-inf"))
               weights = F.softmax(scores, dim=-1)
               out = self.attn_dropout(weights) @ v

        5. Merge the heads and project:

               out = self.resid_dropout(self.out_proj(self._merge_heads(out)))
               return (out, weights) if return_weights else out

    FOUR DETAILS THAT BREAK IF YOU DO NOT WATCH THEM
    ------------------------------------------------
    **The ORDER of the view in `_split_heads`.** First `view(B, T, n_heads, head_dim)` and
    THEN `transpose`. If you did `view(B, n_heads, T, head_dim)` directly you would be mixing
    positions with heads: right shape, wrong data, zero errors. There is a test that detects
    it by checking the heads do not give identical patterns.

    **The `.contiguous()` in `_merge_heads`.** `transpose` does not move data, it only
    changes how it is walked (the "strides"), and `view` demands contiguous memory. Without
    it, PyTorch raises an error that talks about strides and does not clearly say what to do.

    **RoPE goes AFTER splitting into heads.** The rotation depends on `head_dim`, not on
    `d_model`. And only to q and k: what should depend on position is the SCORES, not the
    content being transported.

    **The `if self.training` in SDPA's dropout.** `F.scaled_dot_product_attention` does not
    check the mode on its own: if you pass it a fixed `dropout_p`, it would apply dropout at
    evaluation time too and your samples would come out noisy and non-reproducible.

    WHY ONE BIG PROJECTION AND NOT EIGHT SMALL ONES
    -----------------------------------------------
    `nn.Linear(320, 320)` followed by a `view` is mathematically identical to eight
    `nn.Linear(320, 40)` whose results get concatenated. But it is ONE big matmul instead of
    eight small ones, and as you saw in module 01, large matrices use the GPU far better.

    forward(x, mask=None, cos=None, sin=None, return_weights=False):
        Args:
            x: `(B, T, d_model)`.
            mask: `(T, T)` boolean. If it is `None`, build a causal one.
            cos, sin: RoPE tables (module 09), or `None`.
            return_weights: if `True`, return `(output, weights)`.
        Returns:
            `(B, T, d_model)`, or the tuple if `return_weights`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_sdpa: bool = False,
    ) -> None:
        super().__init__()
        raise NotImplementedError("TODO: module 06, exercise 3 - MultiHeadAttention.__init__")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TODO: module 06, exercise 3 - MultiHeadAttention.forward")
