# 07 — Annotated solution

## Exercise 1 — `layer_norm`

```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
normalized = (x - mean) / torch.sqrt(var + eps)
if weight is not None:
    normalized = normalized * weight
if bias is not None:
    normalized = normalized + bias
return normalized
```

**`unbiased=False` is the exercise's trap.** `torch.var` divides by $n-1$ by default (sample
variance, Bessel's correction). LayerNorm uses the **population** one, which divides by $n$.
With $d = 320$ the difference is 0.3% and you might not notice it; with $d = 4$ it is 33%
and it is obvious. The test compares your result against both versions and tells you which
one it resembles more.

**`keepdim=True`.** Without it, `mean(dim=-1)` over `(4, 8, 32)` returns `(4, 8)` instead of
`(4, 8, 1)`, and the subtraction `x - mean` tries to broadcast the dimensions wrongly.
Sometimes it raises and sometimes — when the shapes happen to line up — it silently produces
garbage.

**The $\epsilon$ goes inside the square root**, not outside:
$\sqrt{\sigma^2 + \epsilon}$, not $\sqrt{\sigma^2} + \epsilon$. It is what `F.layer_norm`
does and with a small variance the difference matters.

**The optional parameters.** Letting `weight` and `bias` be `None` makes it possible to
compare pure normalization against `F.layer_norm(x, (d,))` with no affine arguments. It is
not a whim: it is what gives the test a clean oracle.

## Exercise 2 — `RMSNorm`

```python
def __init__(self, dim, eps=1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

def forward(self, x):
    return self._norm(x.float()).type_as(x) * self.weight
```

**`torch.ones` and not `torch.randn`.** At initialization, the layer has to be pure
normalization. If `weight` started random, you would be scaling each dimension by an
arbitrary factor before having learned anything, and the step-0 loss would not match
$\ln(V)$ — module 05's bug detector.

**`torch.rsqrt(z)` instead of `1/torch.sqrt(z)`.** It computes the inverse square root in
one go. It is one kernel fewer and slightly more numerically stable.

**The `.float()` is not paranoia.** With fp16 activations, squaring overflows sooner than
you would expect: $300^2 = 90,000$ and fp16 runs out at 65,504. The result would be `inf`,
then the mean would be `inf`, and `rsqrt(inf)` would be 0: the layer would return zeros. The
test `test_rmsnorm_computes_in_fp32_and_does_not_overflow` reproduces exactly that case.

**A surprising detail worth seeing once.** Even if you call `.type_as(x)` to go back to
fp16, the output ends up being **fp32**, because afterwards you multiply by `self.weight`,
which is an fp32 parameter, and PyTorch promotes. It is not a bug: it is what Llama's
implementation does and it is what you want. Under autocast the weights stay in fp32 and the
following operations convert what they need; leaving a normalization's output at high
precision is free and gives numerical headroom. There is a test that documents it.

## Exercise 3 — `prenorm_residual`

```python
return x + fn(norm(x))
```

One line. And it is the most important exercise in the module.

The difference from post-norm — `norm(x + fn(x))` — looks like a matter of parentheses and
it decides whether a deep network trains. Differentiating pre-norm with respect to `x`:

$$\frac{\partial}{\partial x}\big(x + f(\text{norm}(x))\big) = 1 + \frac{\partial f(\text{norm}(x))}{\partial x}$$

That **1** reaches the layers below intact however many layers there are. The test
`test_the_gradient_arrives_intact_through_the_residual` checks it in the most direct way
possible: it completely nulls the branch's gradient and verifies that the gradient at the
input is still exactly 1.

## What you should see in the demo, and a correction to the usual narrative

The experiment stacks $N$ blocks and measures the norm of the gradient reaching the input:

| layers | nothing | norm only | post-norm | pre-norm |
|---|---|---|---|---|
| 4 | 3.2e-01 | 1.4e+01 | 7.1e+01 | 7.9e+01 |
| 16 | 1.1e-07 | 1.9e+01 | 6.4e+01 | 9.7e+01 |
| 64 | **0.0e+00** | 5.1e+00 | 5.9e+01 | **1.5e+02** |

Read it carefully, because there is a nuance the textbook explanation usually skips.

**With nothing**, the gradient reaches **exactly zero** with 64 layers. Not "small": zero,
through floating-point underflow. The first layers receive no signal at all.

**With normalization only** — no residuals — the gradient already recovers to 5.1. This
matters: **normalization on its own solves much of the vanishing problem**, because it
returns the scale to 1 at every step and cuts the chain of multiplicative factors.

So the argument that "residuals exist so the gradient does not vanish" is half true.
Normalization and residuals attack the same problem by different routes and they are
complements, not alternatives. What sets **pre-norm** apart is not avoiding the vanishing,
but that it is the only setup whose gradient **grows** with depth instead of shrinking: the
path $x \to x$ has no toll at all.

## About the LayerNorm / RMSNorm comparison

The demo also measures how they really differ, and there is a methodology lesson here.

My first version compared the two outputs with the **correlation coefficient**, and it gave
0.998 with centered data and 0.999 with shifted data. That is, the correlation *rose* when
the data was shifted, exactly the opposite of what you would expect.

The error was in the metric, not in the result: **correlation is invariant to affine
transformations**, so it gives ~1 even if one of the two leaves an offset that the other
removes. That is exactly what I was trying to measure, and it was blind to it.

The right metric is looking at the **mean of the output**: LayerNorm always leaves it at 0,
and RMSNorm preserves it. With already-centered data — which is the normal case inside a
network — the two do practically the same thing, hence you can do without subtracting the
mean. With a large offset they diverge.

That this does not hurt in practice is an **empirical result**, not a theorem. Zhang and
Sennrich checked it by training models, not by proving it.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(var + eps)

    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias
    return normalized


class RMSNorm(nn.Module):

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
    return x + fn(norm(x))
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
