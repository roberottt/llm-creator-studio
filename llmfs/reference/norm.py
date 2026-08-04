"""Reference for module 07: normalization and residual connections."""

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
    """LayerNorm: center at 0, scale to variance 1, then apply gamma and beta.

    $$y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\cdot \\gamma + \\beta$$

    The mean and the variance are computed over the LAST dimension (each token's
    features), separately for each token. Nothing to do with BatchNorm, which normalizes
    along the batch: LayerNorm treats each token independently, which is why it works the
    same with a batch of 1 as with 1000 and needs no running statistics.

    WATCH OUT with the variance: the POPULATION one is used (divide by n), not the sample
    one (by n-1). That is what `F.layer_norm` does, and using `torch.var` with its default
    `unbiased=True` gives a slightly different result.
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(var + eps)

    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias
    return normalized


class RMSNorm(nn.Module):
    """RMSNorm: like LayerNorm but without subtracting the mean or adding a bias.

    $$y = \\frac{x}{\\sqrt{\\frac{1}{d}\\sum_i x_i^2 + \\epsilon}} \\cdot \\gamma$$

    Zhang and Sennrich (2019) observed that almost all of LayerNorm's benefit comes from
    rescaling, not from recentering. Dropping the mean saves a pass over the data and an
    intermediate tensor: between 7% and 64% faster depending on the case, with no
    measurable loss of quality. That is why Llama, Mistral and practically everything
    modern use it.

    An important implementation detail: the computation is done in float32 even if the
    input arrives in float16. Squaring the values of a large activation can overflow the
    fp16 range (65504), and the result would be `inf`.

    Submodules:
        weight: a parameter of shape `(dim,)`, initialized to ones.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


def prenorm_residual(
    x: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    norm: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Pre-norm: `x + fn(norm(x))`.

    Compare with post-norm, which is `norm(x + fn(x))`. The difference looks cosmetic and
    it decides whether a deep network trains at all.

    In pre-norm, the path `x -> output` includes a `+x` with nothing in between.
    Differentiating, that term contributes a 1 to the gradient which arrives intact at the
    layers below, layer after layer. It is a highway for the gradient.

    In post-norm, the normalization sits ON TOP of the residual path, so the gradient goes
    through it at every layer and keeps getting rescaled. With 6 layers you barely notice;
    with 40 you need a careful warmup so it does not explode or die.

    The price of pre-norm is that the block's output has a scale that grows with depth
    (every layer adds something to the residual stream). That is why pre-norm models ALWAYS
    have a final normalization before the output layer.
    """
    return x + fn(norm(x))


def postnorm_residual(
    x: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    norm: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Post-norm: `norm(x + fn(x))`. What the original 2017 paper did.

    Not an exercise; it is here so the demo can compare the two empirically.
    """
    return norm(x + fn(x))
