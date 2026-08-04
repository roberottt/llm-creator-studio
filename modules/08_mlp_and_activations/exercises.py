"""Module 08 - FFN, GELU and SwiGLU.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement -> `llmfs check 08` -> `llmfs hint 08 -e N`
-> `SOLUTION.md` has the complete code.

WHAT YOU ARE GOING TO BUILD
===========================

The part of the Transformer where TWO THIRDS of the parameters live:

    gelu               (ex. 1)  the classic nonlinearity
    swiglu_hidden_dim  (ex. 2)  arithmetic: the 896 in the final config comes from here
    SwiGLU             (ex. 3)  the gated FFN the model uses

Exercise 2 is the shortest in the course and produces a number you have already seen in the
YAML.

WHY THIS IS NEEDED
==================

Attention is a weighted average, that is, a LINEAR operation. And two linear operations in a
row are one:

    W2 · (W1 · x) = (W2 · W1) · x

A hundred stacked linear layers are equivalent to ONE. The only thing stopping the whole
Transformer from collapsing is this module's nonlinearity.

VOCABULARY YOU ARE GOING TO NEED
================================

- **FFN / MLP** (feed-forward network): the part of each block that is NOT attention. It
  processes each token separately, without looking at the others.
- **activation**: the nonlinear function that goes between layers. ReLU, GELU, Swish.
- **nonlinearity**: any function that is not `f(ax+b) = a·f(x)+b`. It is what makes stacking
  layers worth anything.
- **gate**: in SwiGLU, one of the two branches multiplies the other and decides how much
  signal passes through each dimension. Unlike a normal activation, that filtering depends
  on the input.
- **d_ff**: the FFN's inner dimension. In our model, 896.

    llmfs demo 08     shows the linear collapse and compares the activations
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU with the tanh approximation.

    WHAT YOU HAVE TO WRITE
    ----------------------
    One line, transcribing the formula as it stands:

        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
        ))

    No loops and no branches. The two possible mistakes are mistyping a constant
    (`sqrt(2/pi) ~= 0.7978`) or regrouping the expression in a way that changes the order of
    operations.

    VALUES TO CHECK AGAINST
    -----------------------
        x = -3  ->  -0.0036     almost cancelled
        x = -1  ->  -0.1588     partially
        x =  0  ->   0.0000
        x =  1  ->   0.8412     almost whole
        x =  3  ->   2.9964     whole

    WHAT IT DOES AND WHY IT IS NEEDED
    ---------------------------------
    It is the NONLINEARITY. Without something like it between layers, stacking layers
    achieves nothing: two matrix multiplications in a row are one matrix multiplication.

    GELU multiplies `x` by the probability that a standard normal comes out below `x`.
    Instead of deciding with a hard cut whether to let the value through (as ReLU does), it
    attenuates it gradually.

    WHY THE APPROXIMATION AND NOT THE EXACT ONE
    -------------------------------------------
    The real definition uses `erf`, which was slow on 2016 GPUs. Today the difference is
    irrelevant, but GPT-2 was trained with the approximation and it is still used for
    compatibility.

    Your result has to match `F.gelu(x, approximate="tanh")`, NOT plain `F.gelu(x)`: they are
    different functions and the test compares against the first.

    WHAT MATTERS IS NOT THE FORMULA, IT IS THE DERIVATIVE
    -----------------------------------------------------
        x       ReLU     dReLU/dx      GELU      dGELU/dx
        -3.0    0.0000   0.0000       -0.0036   -0.0119
        -1.0    0.0000   0.0000       -0.1588   -0.0833

    With ReLU the derivative throughout the negative zone is EXACTLY ZERO. A neuron that ends
    up always producing negative values stops receiving gradient forever: it is dead and
    there is no way to revive it. GELU's derivative is small but not zero, so it can come
    back.

    Args:
        x: any shape.

    Returns:
        The same shape as `x`.
    """
    raise NotImplementedError("TODO: module 08, exercise 1 - gelu")


def swiglu_hidden_dim(
    d_model: int, multiple_of: int = 64, ffn_dim_multiplier: float | None = None
) -> int:
    """Computes `d_ff` for SwiGLU. This exercise produces the 896 in the final config.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Three lines.

        1. Two thirds of the classic 4x:

               hidden = int(2 * (4 * d_model) / 3)

        2. The optional multiplier:

               if ffn_dim_multiplier is not None:
                   hidden = int(ffn_dim_multiplier * hidden)

        3. Round UP to the next multiple of `multiple_of`:

               return multiple_of * ((hidden + multiple_of - 1) // multiple_of)

    THE ROUNDING IN STEP 3, WITHOUT `math.ceil`
    -------------------------------------------
    Adding `multiple_of - 1` before the integer division forces rounding up, and if the value
    was already an exact multiple it does not change it. It is the standard idiom and it
    avoids putting floats where they are not needed.

    CHECK IT WITH THE COURSE'S TWO CASES
    ------------------------------------
        d_model = 320:  int(2*1280/3) = 853  ->  64 * ((853+63)//64) = 64 * 14 = 896
        d_model = 128:  int(2*512/3)  = 341  ->  64 * ((341+63)//64) = 64 *  6 = 384

    The 896 is the final model's `d_ff`, and the 384 is the toy's.

    WHERE THE 2/3 COMES FROM
    ------------------------
    It is the only conceptual part of the exercise.

        classic FFN:  2 matrices x d x 4d           = 8d²
        SwiGLU:       3 matrices x d x (2/3 * 4d)   = 8d²      <- same budget
        SwiGLU unadjusted: 3 x d x 4d               = 12d²     <- 50% more

    SwiGLU has THREE matrices where the classic FFN has two. With the same `d_ff` it would
    cost 50% more, so the hidden size is cut to two thirds to spend the same and be able to
    compare fairly.

    WHY WE ROUND
    ------------
    It is not cosmetic. Dimensions aligned to powers of two let the tensor cores take their
    fast paths. A matrix with 853 columns is noticeably slower than one with 896, while
    having FEWER parameters.

    Args:
        d_model: the model dimension.
        multiple_of: what to round to a multiple of.
        ffn_dim_multiplier: optional extra factor (Llama uses it to tune by hand).

    Returns:
        `d_ff` as an integer.
    """
    raise NotImplementedError("TODO: module 08, exercise 2 - swiglu_hidden_dim")


class SwiGLU(nn.Module):
    """The gated FFN the final model uses. Two thirds of its parameters are here.

    WHAT YOU HAVE TO WRITE
    ----------------------
    **In `__init__`** (four lines besides the `super()`). The names matter:

        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout   = nn.Dropout(dropout)

    **In `forward`** (one line):

        return self.dropout(self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        ))

    THREE DETAILS
    -------------
    **The `*` is ELEMENTWISE multiplication**, not matrix multiplication. Both branches come
    out with the same shape `(B, T, d_ff)` and are multiplied point by point. If you put `@`
    the shapes would not even line up.

    **The activation goes on `gate_proj`, NOT on `up_proj`.** Numerically the module would
    work just as well with the assignment swapped — it is symmetric apart from which weights
    learn what — but it would NOT match the reference when copying weights and the test would
    fail with a difference that is hard to interpret. There is a dedicated test that points
    it out.

    **`F.silu` is Swish**, that is, `z * sigmoid(z)`. You can write it by hand
    (`x * torch.sigmoid(x)`) and it gives the same thing, but `F.silu` has a fused kernel.

    WHAT IS HAPPENING
    -----------------
    There are TWO projections in parallel from the same input. One of them, after going
    through Swish, acts as a GATE: it multiplies the other and decides how much signal it
    lets through each dimension.

    The difference from a normal activation is that this filtering DEPENDS ON THE INPUT. An
    activation applies the same function to everything; a gate decides, for each dimension
    and each token, how much gets through.

    A POINT WORTH BEING CLEAR ON
    ----------------------------
    The FFN processes each token SEPARATELY. It does not mix information between positions:
    that is attention's job. No mask or anything like it is needed here.

    `bias=False` by default is the final model's config, and it is what makes the parameter
    count come out at exactly `3 * d_model * d_ff`.

    forward(x):
        Args:
            x: `(B, T, d_model)`.
        Returns:
            `(B, T, d_model)`.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        raise NotImplementedError("TODO: module 08, exercise 3 - SwiGLU.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: module 08, exercise 3 - SwiGLU.forward")
