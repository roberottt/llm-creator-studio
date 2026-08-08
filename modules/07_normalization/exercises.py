"""Module 07 - Normalization and residual connections.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement -> `llmfs check 07` -> `llmfs hint 07 -e N`
-> `SOLUTION.md` has the complete code.

WHAT YOU ARE GOING TO BUILD
===========================

The two pieces that make a deep network train instead of returning NaN:

    layer_norm         (ex. 1)  center at 0 and scale to variance 1
    RMSNorm            (ex. 2)  the same but without the mean (what Llama uses)
    prenorm_residual   (ex. 3)  ONE LINE, and it is the most important in the module

The third one has barely any code. What matters is understanding WHY the parentheses go
where they go.

`THEORY.md` is ordered just like this list: each exercise has its own section there with its
numeric example, and each docstring here tells you which one. If you are wondering why exercise
1 is a plain function and exercise 2 is a class, the answer is in the section "Why 1 is a
function and 2 is a class": exercise 2 is the first layer in the course with weights of its
own.

VOCABULARY YOU ARE GOING TO NEED
================================

- **normalize**: rescale some numbers so they have a known mean and spread. Here, mean 0 and
  variance 1.
- **variance**: how much the values spread around their mean.
- **residual connection**: adding a block's input to its output (`x + f(x)`). It is what
  makes training deep networks possible.
- **residual stream**: that accumulated sum flowing through the whole network. Each layer
  adds its contribution to it.
- **vanishing gradient**: when the gradient becomes so small as it crosses layers that the
  first ones stop receiving signal and do not learn.
- **pre-norm / post-norm**: whether the normalization goes inside the branch
  (`x + f(norm(x))`) or wrapped around the sum (`norm(x + f(x))`).

    llmfs demo 07     measures how much gradient reaches the first layer in each setup
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Normalizes each token to mean 0 and variance 1, then applies scale and shift.

    Context in `THEORY.md`: section "Exercise 1: LayerNorm by hand", with the example worked out
    by hand, which dimension the mean is computed over (and how that differs from BatchNorm),
    and the two versions of the variance side by side.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Five lines.

        1. The mean and the variance, over the LAST dimension:

               mean = x.mean(dim=-1, keepdim=True)
               var = x.var(dim=-1, keepdim=True, unbiased=False)

        2. Normalize:

               normalized = (x - mean) / torch.sqrt(var + eps)

        3. Apply the parameters IF they exist:

               if weight is not None:
                   normalized = normalized * weight
               if bias is not None:
                   normalized = normalized + bias

        4. `return normalized`

    EXAMPLE TO CHECK AGAINST
    ------------------------
        x = [2.0, 8.0, 4.0, 6.0]

        mean     = (2+8+4+6)/4 = 5.0
        variance = ((2-5)² + (8-5)² + (4-5)² + (6-5)²)/4 = 20/4 = 5.0
        sqrt(5)  = 2.236

        y = [(2-5)/2.236, (8-5)/2.236, (4-5)/2.236, (6-5)/2.236]
          = [-1.342, 1.342, -0.447, 0.447]

    THE TRAP: `unbiased=False`
    --------------------------
    `torch.var` divides by (n-1) by default (the sample variance, with Bessel's correction).
    LayerNorm uses the POPULATION one, which divides by n.

    Without the `unbiased=False`, your result will look a lot like `F.layer_norm` but will
    not match it. With d=320 the difference is 0.3% and you might not see it; with d=4 it is
    33%. The test compares your result against both versions and tells you which one you
    resemble.

    THE `keepdim=True` IS NOT OPTIONAL EITHER
    -----------------------------------------
    Without it, `mean(dim=-1)` over `(4, 8, 32)` returns `(4, 8)` instead of `(4, 8, 1)`,
    and the subtraction `x - mean` tries to broadcast the dimensions wrongly. Sometimes it
    raises and sometimes — when the shapes happen to line up — it silently produces garbage.

    THE eps GOES INSIDE THE SQUARE ROOT
    -----------------------------------
    `sqrt(var + eps)`, not `sqrt(var) + eps`. It is what `F.layer_norm` does, and with a
    small variance the difference matters. It is there so you do not divide by zero when
    every component is equal.

    WHY OVER THE LAST DIMENSION
    ---------------------------
    Those are each token's features. Each token is normalized on its own, independently of
    the others and of the batch.

    That is what distinguishes LayerNorm from BatchNorm, and the reason it works the same
    with a batch of 1 as with 1000 and does not need stored statistics for inference.

    Args:
        x: `(..., d)`. The last dimension is normalized.
        weight: `(d,)` or None.
        bias: `(d,)` or None.
        eps: so you do not divide by zero.

    Returns:
        The same shape as `x`.
    """
    raise NotImplementedError("TODO: module 07, exercise 1 - layer_norm")


class RMSNorm(nn.Module):
    """LayerNorm without the mean and without the bias. What Llama, Mistral and our model use.

    Context in `THEORY.md`: section "Exercise 2: dropping half of it", with the measured table of
    what LayerNorm and RMSNorm do to centered and shifted data, why this layer is a class and not
    a function, and why the demo's timings do NOT say RMSNorm is slower.

    WHAT YOU HAVE TO WRITE
    ----------------------
    **In `__init__`** (two lines besides the `super()`):

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    **A helper** (or inline it, as you prefer):

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    **In `forward`** (one line):

        return self._norm(x.float()).type_as(x) * self.weight

    EXAMPLE TO CHECK AGAINST
    ------------------------
        x = [2.0, 8.0, 4.0, 6.0]

        RMS = sqrt((4+64+16+36)/4) = sqrt(30) = 5.477
        y   = [2/5.477, 8/5.477, 4/5.477, 6/5.477]
            = [0.365, 1.461, 0.730, 1.096]

    `torch.ones` AND NOT `torch.randn`
    ----------------------------------
    At initialization the layer has to be pure normalization. If `weight` started random, you
    would be scaling each dimension by an arbitrary factor before having learned anything,
    and the step-0 loss would not match `ln(V)`.

    THE `.float()` IS NOT PARANOIA
    ------------------------------
    With activations in fp16, squaring overflows sooner than you would expect:
    `300**2 = 90,000` and fp16 runs out at 65,504.

    The result would be `inf`, then the mean would be `inf`, and `rsqrt(inf)` would be 0: the
    layer would return ZEROS. There is a test that reproduces exactly that case.

    A SURPRISING DETAIL
    -------------------
    Even if you call `.type_as(x)` to go back to fp16, the output ends up being **fp32**,
    because afterwards you multiply by `self.weight`, which is an fp32 parameter, and PyTorch
    promotes.

    It is not a bug: it is what Llama's implementation does and it is what you want. Under
    autocast the weights stay in fp32 and the following operations convert what they need.
    There is a test that documents it.

    `torch.rsqrt(z)` computes `1/sqrt(z)` in one go: one kernel fewer than dividing and
    slightly more stable.

    WHAT CHANGES COMPARED WITH LAYERNORM
    ------------------------------------
    The mean is not subtracted and there is no beta. It only rescales by the root mean
    square (hence the name).

    Zhang and Sennrich (2019) observed that almost all of LayerNorm's benefit comes from
    RESCALING, not from RECENTERING. Dropping it saves a pass over the data and an
    intermediate tensor, with no measurable loss of quality.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        raise NotImplementedError("TODO: module 07, exercise 2 - RMSNorm.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: module 07, exercise 2 - RMSNorm.forward")


def prenorm_residual(
    x: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    norm: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Pre-norm. It is ONE LINE and it is the most important exercise in the module.

    Context in `THEORY.md`: section "Exercise 3: where the parentheses go", with the gradient
    table for the four configurations and the scale of the residual stream measured block by
    block on the real model, which is where the need for `norm_f` comes from.

    WHAT YOU HAVE TO WRITE
    ----------------------
        return x + fn(norm(x))

    That is all.

    WHAT YOU HAVE TO UNDERSTAND
    ---------------------------
    There are two ways to place the normalization, and only the parentheses move:

        post-norm (the 2017 paper):     norm(x + fn(x))
        pre-norm  (everything modern):  x + fn(norm(x))

    What changes is WHERE THE GRADIENT GOES.

    In **pre-norm**, the path from `x` to the output includes a `+x` with nothing in between.
    Differentiating:

        d(output)/dx = 1 + d(fn(norm(x)))/dx

    That **1** is a highway: even if the second term is tiny, the gradient reaches the layers
    below intact, layer after layer.

    In **post-norm**, the normalization sits ON TOP of the sum, so the gradient goes through
    it at every layer and gets rescaled. With 6 layers you barely notice; with 40 you need a
    careful warmup so training does not explode.

    IF IT COMES OUT WRONG
    ---------------------
    If you write `norm(x + fn(x))` you have written post-norm, and there is a test that
    detects it.

    HOW IT GETS CHECKED
    -------------------
    There is a test that completely nulls the branch's gradient (`fn` returns a detached
    tensor times zero) and verifies that the gradient at the input is still EXACTLY 1. That 1
    is the whole reason the residual exists.

    A CONSEQUENCE FOR MODULE 10
    ---------------------------
    Since the residual stream is never normalized along the way, it reaches the output at a
    scale that grows with depth. That is why pre-norm models ALWAYS carry a final
    normalization before the logits layer. It will be called `norm_f`.

    Args:
        x: the input.
        fn: the block (attention or FFN).
        norm: the normalization layer.

    Returns:
        The same shape as `x`.
    """
    raise NotImplementedError("TODO: module 07, exercise 3 - prenorm_residual")
