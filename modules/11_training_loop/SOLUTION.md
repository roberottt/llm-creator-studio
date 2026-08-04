# 11 — Annotated solution

## Exercise 1 — `AdamWScratch.step`

```python
@torch.no_grad()
def step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        beta1, beta2 = group["betas"]
        lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            t = state["step"]
            m, v = state["exp_avg"], state["exp_avg_sq"]

            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1**t
            bias_correction2 = 1 - beta2**t

            denom = (v / bias_correction2).sqrt_().add_(eps)
            step_size = lr / bias_correction1

            if wd != 0.0:
                p.mul_(1 - lr * wd)
            p.addcdiv_(m, denom, value=-step_size)

    return loss
```

### The three mistakes the test catches

**Starting `t` at 0.** With $t=0$, $1 - \beta^0 = 0$ and you divide by zero. You have to
increment `state["step"]` **before** using it.

**Forgetting the bias correction.** The test `test_the_bias_correction_is_applied` measures
it in the most direct way: a single step with gradient 1 and `lr=0.1` has to move the
parameter by exactly 0.1. Without the correction, with $\beta_2 = 0.95$, $v = 0.05$ and
$\sqrt{v} = 0.224$: the step would come out $0.1/0.224 = 0.447$, **4.5 times larger than it
should be**. With that, the first steps wreck the initialization.

**Adding the weight decay to the gradient.** That is the difference between Adam+L2 and
AdamW. The test `test_the_weight_decay_is_decoupled` tells them apart with an elegant trick:
it sets the **gradient to zero** and checks the parameter still shrinks. With decoupled
decay, `p ← p·(1 − lr·wd)` goes from 2.0 to 1.9. With L2, the "gradient" would be `wd·p` and
it would go through the division by $\sqrt{v}$, which with $v \approx 0$ gives something very
different.

### About `p.mul_(1 - lr * wd)`

It is equivalent to `p -= lr * wd * p` but in a single operation. Note that **the decay is
applied before** Adam's update, just as in PyTorch's implementation. The order matters for
the weights to match to the last decimal.

### About the in-place operations

`addcmul_(a, b, value=v)` computes `self += v * a * b`, and `addcdiv_(a, b, value=v)`
computes `self += v * a / b`. They are cryptic but they avoid allocating intermediate
tensors. With 8.9 million parameters and 10,000 steps, that shows.

If they feel uncomfortable, write it with normal operations first: the test compares results,
not style.

## Exercise 2 — `lr_at_step`

```python
min_lr = lr * min_lr_ratio

if warmup_steps > 0 and step < warmup_steps:
    return lr * (step + 1) / warmup_steps

if step >= max_steps:
    return min_lr

if schedule == "constant":
    return lr

progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
progress = min(1.0, max(0.0, progress))

if schedule == "linear":
    return lr - (lr - min_lr) * progress

coef = 0.5 * (1.0 + math.cos(math.pi * progress))
return min_lr + (lr - min_lr) * coef
```

**The `+1` in the warmup** stops step 0 from having an `lr` of exactly zero. A step with lr=0
learns nothing: it is a wasted step, and with a 500-step warmup that would be 500 wasted at
the start of every run.

**Check the cosine's endpoints** instead of trusting it: with `progress=0`, $\cos(0)=1$ and
`coef=1`, so it returns `lr`. With `progress=1`, $\cos(\pi)=-1$ and `coef=0`, so it returns
`min_lr`. Correct.

**The order of the guards matters.** The `step >= max_steps` has to come before the cosine
computation; otherwise `progress` would come out greater than 1 and the cosine would start
*rising* again. The `min(1.0, max(0.0, ...))` is an extra belt.

## Exercise 3 — `clip_grad_norm`

```python
grads = [p.grad for p in parameters if p.grad is not None]
if not grads:
    return 0.0

total = torch.sqrt(sum(g.detach().pow(2).sum() for g in grads))
total_f = float(total)

if max_norm > 0 and total_f > max_norm:
    factor = max_norm / (total_f + 1e-6)
    for g in grads:
        g.mul_(factor)

return total_f
```

**The norm is global**, not per tensor. That is the only conceptual part of the exercise, and
the test `test_it_preserves_the_gradients_direction` checks it directly: it computes the
cosine between the gradient vector before and after clipping, and requires it to be > 0.9999.
If you clipped each tensor separately, each would be scaled by a different factor and the
combined direction would change.

The demo measures it: the norm goes from **112,858 to 1.0000** with a cosine of
**0.99999994**. Only the magnitude changes.

**Returning the norm before clipping** is what `torch.nn.utils.clip_grad_norm_` does and it
is the useful one: logging it warns you that training is destabilizing before it blows up.

## Exercise 4 — `build_param_groups`

```python
decay, no_decay = [], []
for param in model.parameters():
    if not param.requires_grad:
        continue
    (decay if param.dim() >= 2 else no_decay).append(param)

return [
    {"params": decay, "weight_decay": weight_decay},
    {"params": no_decay, "weight_decay": 0.0},
]
```

Five lines, and the rule is surprisingly simple: **`param.dim() >= 2`**. Matrices decay;
vectors do not.

With our model the split comes out like this:

| group | tensors | parameters |
|---|---|---|
| with decay (matrices) | 43 | 8,929,280 |
| without decay (RMSNorm) | 13 | 4,160 |

The 4,160 are the 13 normalizations × 320. They are 0.05% of the parameters and applying
decay to them **would produce no visible error**: the model would simply train slightly worse
and you would only detect it by comparing two complete runs.

## A bug I made writing these tests

`assert model[0].weight not in everything`, where `everything` is a list of tensors, blows up
with `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`.

The `in` operator uses `==`, and on tensors `==` returns an **elementwise** comparison, not a
boolean. You have to compare by identity:

```python
ids = {id(p) for g in groups for p in g["params"]}
assert id(model[0].weight) not in ids
```

It is a classic PyTorch stumble and it is worth keeping in mind: any use of tensors in a
boolean context — `if tensor:`, `x in list_of_tensors`, `assert tensor` — does something
different from what it looks like.

## What you should see in the demo

**Your AdamW against PyTorch's:** a maximum error of $2 \times 10^{-7}$ in the weights after
200 steps. Not similar: identical apart from fp32 rounding.

**The clipping in the face of a toxic batch**, injected at step 50:

```
without clipping  : the loss RISES 3.0x after the toxic batch
with grad_clip=1.0: it does not even notice (0.8x, still dropping)
```

In a 10,000-step run, a single odd batch can cost you the whole training. `grad_clip=1.0`
bounds the maximum damage from any batch.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
class AdamWScratch(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr cannot be negative: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"the betas must be in [0, 1): {betas}")
        if eps < 0.0:
            raise ValueError(f"eps cannot be negative: {eps}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                # Running averages, in-place so no new tensors are allocated each step.
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t

                denom = (v / bias_correction2).sqrt_().add_(eps)
                step_size = lr / bias_correction1

                # DECOUPLED weight decay: applied directly to the parameter.
                if wd != 0.0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m, denom, value=-step_size)

        return loss


def lr_at_step(
    step: int,
    max_steps: int,
    lr: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
    schedule: str = "cosine",
) -> float:
    min_lr = lr * min_lr_ratio

    if warmup_steps > 0 and step < warmup_steps:
        # +1 so that step 0 does not get an lr of exactly zero (it would learn nothing).
        return lr * (step + 1) / warmup_steps

    if step >= max_steps:
        return min_lr

    if schedule == "constant":
        return lr

    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))

    if schedule == "linear":
        return lr - (lr - min_lr) * progress

    coef = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (lr - min_lr) * coef


def clip_grad_norm(parameters: Iterable[nn.Parameter], max_norm: float) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    total = torch.sqrt(sum(g.detach().pow(2).sum() for g in grads))
    total_f = float(total)

    if max_norm > 0 and total_f > max_norm:
        # 1e-6 so we do not divide by zero if the norm is tiny.
        factor = max_norm / (total_f + 1e-6)
        for g in grads:
            g.mul_(factor)

    return total_f


def build_param_groups(
    model: nn.Module, weight_decay: float = 0.1
) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
