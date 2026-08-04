"""Reference for module 11: the training loop.

Here are the four pieces that turn "I have a model" into "I have a trained model": the
optimizer, the learning-rate scheduler, gradient clipping, and splitting the parameters for
weight decay.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn as nn


class AdamWScratch(torch.optim.Optimizer):
    """AdamW implemented from scratch, so it stops being a black box.

    Adam combines two ideas:

    1. **Momentum** (first moment, `m`): instead of moving along this step's gradient, it
       moves along a running average of the recent gradients. This smooths out the noise
       from batch sampling.

    2. **Per-dimension scaling** (second moment, `v`): each parameter is divided by the
       square root of the running average of its SQUARED gradients. Parameters with
       consistently large gradients move little, and those that almost never receive signal
       move a lot when they do. That is what makes a single learning rate work for the
       whole network.

    The formulas, with `t` the step number (starting at 1):

        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g^2

        m_hat = m / (1 - beta1^t)          <- bias correction
        v_hat = v / (1 - beta2^t)

        p -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)

    THE BIAS CORRECTION. `m` and `v` start at zero, so in the first steps they underestimate
    the real magnitude: with beta2=0.95, after one step `v` is only 5% of g^2. Dividing by
    `1 - beta^t` compensates for that exactly. Without it, the first steps take enormous
    jumps and training can diverge before it begins.

    THE W IN ADAMW. Weight decay is applied DIRECTLY to the parameter, not added to the
    gradient. The difference matters: if you added it to the gradient, it would go through
    `v`'s scaling and its real effect would depend on the magnitude of that parameter's
    gradients. Decoupled (Loshchilov and Hutter, 2019) the decay is uniform and predictable.
    """

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
    """The learning rate for a given step: linear warmup + cosine decay.

    Three segments:

        1. `step < warmup_steps`   -> linear ramp from 0 to `lr`
        2. `step >= max_steps`     -> stays at `lr * min_lr_ratio`
        3. in between              -> cosine from `lr` down to `lr * min_lr_ratio`

    THE WARMUP. In the first steps, Adam's moments are nearly empty and its estimates are
    noisy. On top of that the weights are freshly initialized and the gradients are large.
    Starting at full lr usually produces a loss spike that the model sometimes never
    recovers from. Ramping up slowly avoids that initial wreckage.

    THE COSINE. At the start you want to move fast; at the end, to fine-tune. The cosine
    drops slowly at first, fast in the middle and slowly at the end. Compared with a linear
    decay the difference is small but consistent, and it is what everyone uses.

    THE 10% FLOOR. It does not decay to zero: below a certain point the model stops learning
    entirely and compute is wasted. 10% is the convention.

    Formula for segment 3, with `progress` running from 0 to 1:

        coef = 0.5 * (1 + cos(pi * progress))       goes from 1 to 0
        lr_t = lr * (min_lr_ratio + (1 - min_lr_ratio) * coef)
    """
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
    """Clip the gradients so their GLOBAL norm does not exceed `max_norm`.

    The norm of all the gradients together is computed, as if they were a single giant
    vector:

        norm = sqrt( sum of ||g_i||^2 over every parameter )

    If it exceeds `max_norm`, ALL the gradients are multiplied by `max_norm / norm`.

    WHY THE GLOBAL NORM AND NOT ONE PER TENSOR. Clipping each tensor separately would change
    the DIRECTION of the combined gradient, which is exactly what you do not want: the
    gradient points where you have to go, and you are only limiting how far you step. With
    the global norm, the direction is preserved exactly.

    WHAT IT IS FOR. Occasionally a batch produces enormous gradients (an odd sequence, a
    very rare token). Without clipping, that single batch can take a jump that destroys
    hours of training. With grad_clip=1.0 the maximum damage is bounded.

    Returns:
        The norm BEFORE clipping. It is worth logging: if it rises steadily, training is
        destabilizing.
    """
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
    """Split the parameters into two groups: with weight decay and without it.

    THE RULE: **decay only on matrix weights** (anything with 2 dimensions or more). Biases,
    normalization scales and any 1-dimensional parameter go WITHOUT decay.

    WHY. Weight decay pushes weights towards zero, and that makes sense in a projection
    matrix: penalizing large magnitudes reduces overfitting.

    On an RMSNorm scale it makes no sense at all. That parameter starts at 1 and its job is
    to rescale; pushing it towards zero is pushing the layer's output towards zero, which is
    exactly the opposite of what is needed. The same goes for biases: they are offsets, not
    magnitudes worth limiting.

    Applying decay to everything is a common mistake, produces no visible error, and
    degrades the result. Detecting it means comparing two training runs, which is expensive.

    A NOTE ON WEIGHT TYING. `model.parameters()` deduplicates by identity, so the tied
    embedding appears once and goes into the decay group (it has 2 dimensions).

    Returns:
        A list of two dicts in the format PyTorch optimizers expect:
        `[{"params": [...], "weight_decay": wd}, {"params": [...], "weight_decay": 0.0}]`
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
