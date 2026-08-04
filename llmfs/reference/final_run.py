"""Reference for module 13: the real run."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch


def overfit_single_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 200,
    lr: float = 1e-3,
    optimizer_factory: Callable[..., Any] | None = None,
) -> list[float]:
    """Train on ONE SINGLE batch until it memorizes it. The test that catches almost any bug.

    THE IDEA. A model with millions of parameters has more than enough capacity to memorize
    four sequences. If you feed it the same batch over and over, the loss HAS to drop to
    practically zero.

    If it does not drop, there is a bug, and you know it in 30 seconds instead of four hours.

    WHAT IT CATCHES
        - gradients that do not reach some part of the model (one `detach()` too many)
        - the forgotten `zero_grad()`
        - an absurd learning rate
        - a badly placed mask (though in that case it drops TOO fast, watch out)
        - a layer that is not connected to the graph
        - the optimizer built over the wrong parameters

    WHAT IT DOES NOT CATCH
        Anything about generalization: the data, the model size, the regularizer. A model
        that memorizes a batch can still be useless.

    IT IS THE FIRST CHECK TO RUN, always, before launching any long training run. Karpathy
    has been repeating it for years and it is still the best cost-to-benefit advice in all
    of deep learning.

    Returns:
        The loss history, one per step.
    """
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
    """Format a duration into something readable at a glance.

        45      -> "45s"
        125     -> "2m 5s"
        3725    -> "1h 2m"
        90000   -> "1d 1h"

    It looks cosmetic and it is not: you are going to look at this number many times during
    a run that lasts hours, and "1h 2m" reads instantly while "3725 s" has to be divided in
    your head.

    Negative or non-finite values return "?", which is the honest answer when there is not
    enough data to estimate yet.
    """
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


def estimate_remaining(
    step: int, max_steps: int, elapsed_seconds: float, warmup_steps: int = 10
) -> float:
    """Seconds remaining, based on the throughput measured so far.

    The first steps are slower (kernel compilation, cold caches), so including them in the
    average gives pessimistic estimates. `warmup_steps` ignores them.

    Returns `inf` while there is not enough data, and `format_eta` turns that into "?". That
    is more honest than making a number up.
    """
    if step <= warmup_steps or elapsed_seconds <= 0:
        return float("inf")

    per_step = elapsed_seconds / step
    return per_step * max(0, max_steps - step)
