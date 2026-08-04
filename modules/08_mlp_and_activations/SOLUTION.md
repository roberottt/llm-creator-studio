# 08 — Annotated solution

## Exercise 1 — `gelu`

```python
return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
```

One line, transcribing the formula as it stands. The two possible mistakes are mistyping a
constant (`sqrt(2/pi) ≈ 0.7978`) or regrouping the expression in a way that changes the
order of operations.

**Why the approximation and not the exact one.** The real definition is $x \cdot \Phi(x)$,
with $\Phi$ the normal's cumulative distribution, computed with `erf`. In 2016 `erf` was
slow on GPUs, so Hendrycks and Gimpel proposed this tanh approximation. Today the speed
difference is irrelevant, but GPT-2 was trained with it and it is still used for historical
compatibility. Your result has to match `F.gelu(x, approximate="tanh")`, not plain
`F.gelu(x)` — they are different functions and the test compares against the first.

**What to take from the exercise is not the formula, it is the derivative.** The demo
tabulates it:

| x | ReLU | dReLU/dx | GELU | dGELU/dx |
|---|---|---|---|---|
| −3.0 | 0.0000 | **0.0000** | −0.0036 | −0.0119 |
| −1.0 | 0.0000 | **0.0000** | −0.1588 | −0.0833 |

With ReLU, the derivative throughout the negative zone is **exactly zero**. A neuron that
ends up always producing negative values stops receiving gradient forever: it is dead and
there is no way to revive it. GELU's derivative is small but not zero, so it can come back.

## Exercise 2 — `swiglu_hidden_dim`

```python
hidden = int(2 * (4 * d_model) / 3)
if ffn_dim_multiplier is not None:
    hidden = int(ffn_dim_multiplier * hidden)
return multiple_of * ((hidden + multiple_of - 1) // multiple_of)
```

**Rounding up without `math.ceil`.** Adding `multiple_of - 1` before the integer division
forces rounding up, and if the value was already an exact multiple it does not change it. It
is the standard idiom for this and it avoids putting floats where they are not needed.

Check it with the course's two cases:

```
d_model = 320:  int(2·1280/3) = 853  ->  64 · ((853+63)//64) = 64 · 14 = 896   ✓
d_model = 128:  int(2·512/3)  = 341  ->  64 · ((341+63)//64) = 64 ·  6 = 384   ✓
```

**Where the 2/3 comes from**, which is the only conceptual part of the exercise:

```
classic FFN:  2 matrices × d × 4d           = 8d²
SwiGLU:       3 matrices × d × (2/3 · 4d)   = 8d²      ✓ same budget
SwiGLU unadjusted: 3 × d × 4d               = 12d²     ✗ 50% more
```

The demo shows it with the model's numbers: classic FFN 819,200, SwiGLU unadjusted 1,228,800
(+50%), SwiGLU with the 2/3 → 860,160 (+5%).

That residual +5%, rather than 0%, is due to the rounding to a multiple of 64. **The budget
equality is asymptotic, not exact**: at $d = 4096$ the drift falls to 0.2%.

**Why we round.** It is not cosmetic. Aligned dimensions let the tensor cores take their
fast paths; a matrix with 853 columns is slower than one with 896 despite having fewer
parameters.

## Exercise 3 — `SwiGLU`

```python
def __init__(self, d_model, d_ff, dropout=0.0, bias=False):
    super().__init__()
    self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
    self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
    self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
    self.dropout = nn.Dropout(dropout)

def forward(self, x):
    return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
```

**The `*` is elementwise multiplication**, not matrix multiplication. Both branches come out
with the same shape `(B, T, d_ff)` and are multiplied point by point. If you put `@` the
shapes would not even line up.

**The activation goes on `gate_proj`, not on `up_proj`.** Numerically the module would work
just as well with the assignment swapped — it is symmetric apart from which weights learn
what — but it **would not match the reference when copying weights** and the test would fail
with a difference that is hard to interpret. The test
`test_swiglu_applies_the_activation_to_the_gate_branch` is there to point it out directly.

**`F.silu` is Swish.** $\text{Swish}(z) = z \cdot \sigma(z)$. You can write it by hand
(`x * torch.sigmoid(x)`) and it gives the same thing, but `F.silu` has a fused kernel.

**No biases by default.** `bias=False` is the final model's config, and it is what makes the
count come out at exactly $3 \cdot d \cdot d_{ff}$. Modern LLMs have been dropping biases:
they add little and they complicate module 11's weight decay.

## What you should see in the demo

**The linear collapse**, which is the whole argument of the module:

```
5 stacked layers  vs  1 single matrix  ->  maximum difference: 2.38e-07   (that is, zero)
the same 5 layers WITH GELU            ->  difference: 4.7
```

Five linear layers with no activation **are** one layer. Since attention is a weighted
average — linear — the FFN is literally the only thing stopping the whole Transformer from
collapsing into a single matrix multiplication.

**The parameter split:**

| d_model | attention | FFN | % FFN |
|---|---|---|---|
| 320 | 409,600 | 860,160 | **68%** |
| 4096 | 67,108,864 | 134,479,872 | **67%** |

Two thirds of the model is FFN. When you read that a model has N parameters, most of them
are here, not in the attention.

**The SwiGLU / classic FFN comparison** deserves a note on method. In the first version of
this demo the task was so easy that both reached a loss of `0.00000` and the code declared
the winner to be whichever had the lower last decimal: that is reading noise. Now the task
is harder, the results are printed in scientific notation, and **if the difference falls
below 10% the demo itself says the experiment does not distinguish between the two
architectures**.

And even when it does distinguish, a toy experiment on an invented task with a single seed
proves nothing about language models. Shazeer (2020) tried every GLU variant training real
transformers. His explanation of why SwiGLU wins, quoted literally:

> *"We offer no explanation as to why these architectures seem to work; we attribute their
> success, as all else, to divine benevolence."*

It is one of the most used and least understood architecture decisions in the field, and it
is worth knowing when you read explanations that sound very sure of themselves.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def swiglu_hidden_dim(
    d_model: int, multiple_of: int = 64, ffn_dim_multiplier: float | None = None
) -> int:
    hidden = int(2 * (4 * d_model) / 3)
    if ffn_dim_multiplier is not None:
        hidden = int(ffn_dim_multiplier * hidden)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
