"""Reference for module 08: the feed-forward network and its activations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU with the tanh approximation.

    $$\\text{GELU}(x) \\approx 0.5x\\left(1 + \\tanh\\left[\\sqrt{2/\\pi}(x + 0.044715x^3)\\right]\\right)$$

    The exact definition is `x * Phi(x)`, with Phi the cumulative distribution function of
    the standard normal. Interpretation: instead of deciding whether to let `x` through
    with a hard cut (like ReLU), it multiplies it by the probability that a normal variable
    is less than `x`. Very negative values are almost entirely cancelled, very positive
    ones pass through almost whole, and in between there is a smooth transition.

    The tanh approximation exists because `erf` was slow on 2016 GPUs. Today the difference
    is negligible, but GPT-2 was trained with the approximation and it is still used for
    compatibility. It is equivalent to `F.gelu(x, approximate="tanh")`.

    Compared with ReLU, the practical advantage is that the derivative is not exactly zero
    for negative inputs, so a neuron that drifts into the negative zone can recover.
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def swiglu_hidden_dim(
    d_model: int, multiple_of: int = 64, ffn_dim_multiplier: float | None = None
) -> int:
    """Compute `d_ff` for SwiGLU: two thirds of the classic 4x, rounded up.

    Where it comes from:

    - The classic FFN uses `d_ff = 4 * d_model` and has TWO matrices: `d -> 4d` and
      `4d -> d`. That is `8 * d^2` parameters.
    - SwiGLU has THREE matrices (gate, up, down), so with the same `d_ff` it would have
      `12 * d^2`: 50% more.
    - To spend the same number of parameters you multiply by 2/3: `d_ff = (2/3) * 4 * d`.
      Then `3 * d * (8/3)d = 8 * d^2`, the same as before.

    After that it is rounded UP to the next multiple of `multiple_of` (64 by default). This
    is not cosmetic: dimensions aligned to powers of two let the tensor cores take their
    fast paths. A matrix with 853 columns is noticeably slower than one with 896.

    With `d_model=320`:
        (2/3) * 4 * 320 = 853.33  ->  ceil to a multiple of 64  ->  896   <- the config's
    """
    hidden = int(2 * (4 * d_model) / 3)
    if ffn_dim_multiplier is not None:
        hidden = int(ffn_dim_multiplier * hidden)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):
    """Gated FFN: `down(Swish(gate(x)) * up(x))`.

    $$\\text{SwiGLU}(x) = \\left(\\text{Swish}(x W_{gate}) \\odot x W_{up}\\right) W_{down}$$

    with `Swish(z) = z * sigmoid(z)` (also called SiLU).

    The idea behind the GLU variants is that one of the two branches acts as a GATE: it
    multiplies the other element by element and decides how much signal passes through each
    dimension. Unlike a normal activation, that filtering depends on the input.

    Shazeer (2020) tried them all and SwiGLU came out best consistently. His own conclusion
    about why, quoted literally, is: "We offer no explanation as to why these architectures
    seem to work; we attribute their success, as all else, to divine benevolence." It is
    one of the most used and least understood architecture decisions in the field, and it
    is worth knowing that.

    Submodules:
        gate_proj: nn.Linear(d_model, d_ff, bias=bias)
        up_proj:   nn.Linear(d_model, d_ff, bias=bias)
        down_proj: nn.Linear(d_ff, d_model, bias=bias)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class MLP(nn.Module):
    """The classic two-matrix FFN with GELU, for comparison in the demo.

    Not an exercise: it serves as a reference point against SwiGLU.

    Submodules:
        fc_in:  nn.Linear(d_model, d_ff, bias=bias)
        fc_out: nn.Linear(d_ff, d_model, bias=bias)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff, bias=bias)
        self.fc_out = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(gelu(self.fc_in(x))))
