# 13 — Annotated solution

## Exercise 1 — `overfit_single_batch`

```python
factory = optimizer_factory or (lambda params: torch.optim.AdamW(params, lr=lr))
opt = factory(model.parameters())

model.train()
history = []
for _ in range(steps):
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    history.append(float(loss.detach()))

return history
```

The simplest possible loop: no scheduler, no accumulation, no AMP. **On purpose**: the fewer
pieces, the fewer places a bug can hide.

**The `model.train()` is not decorative.** If the model arrived in `eval` mode, dropout would
be switched off and you would not be testing the same code path the real training will use.
There is a test that checks it.

**`float(loss.detach())`** and not `float(loss)`: without the detach, PyTorch raises a warning
about converting tensors with a gradient into scalars. It works, but it clutters the output.

### Why this exercise is the most useful one in the module

In the demo, on the real model:

```
step   0:  4.1856   ← ln(65) = 4.1744, correct
step  10:  3.4090
step 100:  0.4363
step 299:  0.0173   ← memorized
```

A model with 800,000 parameters memorizes four sequences of 128 characters without breaking a
sweat. **If it cannot, there is a bug**, and you know it in 30 seconds.

And the warning that goes with it: if it dropped to zero in five steps, suspect an information
leak. Check that the targets are shifted one token relative to the input — the same bug I made
writing the module 10 tests.

## Exercise 2 — `format_eta`

```python
if not math.isfinite(seconds) or seconds < 0:
    return "?"

secs = int(seconds)
if secs < 60:
    return f"{secs}s"
if secs < 3600:
    return f"{secs // 60}m {secs % 60}s"
if secs < 86400:
    return f"{secs // 3600}h {(secs % 3600) // 60}m"
return f"{secs // 86400}d {(secs % 86400) // 3600}h"
```

**From one hour on, the seconds stop being shown.** When there are two hours left, the seconds
are noise: `2h 1m` reads at a glance and `2h 1m 5s` adds nothing.

**Non-finite values return `"?"`.** It is the honest thing when there is not yet enough data to
estimate, and it avoids printing things like `-1s` or `infd 0h`. The `math.isfinite()` covers
`inf`, `-inf` and `nan` in one go.

It looks like a cosmetic exercise and it is not: you are going to look at that number many
times during a run that lasts hours.

## The real run

With everything implemented:

```bash
uv run python -m llmfs train --config tiny_char
```

On this hardware (MPS) it is about 70 seconds for 1,500 steps. On the RTX 2060 it should be
similar or somewhat faster.

### What you should see

**The step-0 loss against `ln(V)`.** The trainer checks it by itself and prints it in green or
in red:

```
initial loss: 4.2325  (ln(65) = 4.1744, deviation +0.0581)
```

**The curve.** It drops fast at first and flattens out; the validation one follows the training
one with a gap that grows slowly. That is incipient overfitting and it is normal.

**And the samples**, which are the part that really teaches you something:

```
step 0     kUU$sbpKKMMbbbPcxfffffTjjfNLL --TJ??333OOqIw
step 300   MAPCHASTING Yrace not be town, bunders. CAMILLY: Mare striset mist
step 600   Which begane of schame a loved, this show as friar, But there appos
step 1500  KING RICHARD III: That's such heaven dull sented braw and starm
```

Pure noise → recognizable words → sentence structure → play formatting with character names.
**That file read top to bottom is the model learning to write**, and it is more informative
than the loss curve.

### Before launching the long run

Two things, in this order:

1. **The overfit on a batch.** 30 seconds.
2. **`--max-steps 100`** to measure the real pace and see the ETA. If it says 40 hours when
   you expected 4, something is wrong and it is better to know before leaving it all night.

## About resuming

The checkpoint saves the weights, **the optimizer state**, the GradScaler's and the step
number. If you resumed with the weights only, Adam would start with its moments at zero and
the model would lurch: it shows up as a spike in the curve, exactly where you resumed.

And the implementation detail: it writes to a temporary file and renames at the end. If the
process dies halfway through writing, the previous checkpoint is still intact. **A half-written
checkpoint is worse than no checkpoint.**

Try it: interrupt with Ctrl+C and resume with `--resume`. The trainer saves before exiting.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
def overfit_single_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 200,
    lr: float = 1e-3,
    optimizer_factory: Callable[..., Any] | None = None,
) -> list[float]:
    factory = optimizer_factory or (lambda params: torch.optim.AdamW(params, lr=lr))
    opt = factory(model.parameters())

    model.train()
    history: list[float] = []
    for _ in range(steps):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))

    return history


def format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?"

    secs = int(seconds)
    if secs < 60:
        return f"{secs}s"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60}s"
    if secs < 86400:
        return f"{secs // 3600}h {(secs % 3600) // 60}m"
    return f"{secs // 86400}d {(secs % 86400) // 3600}h"
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
