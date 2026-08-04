"""Module 11 - The training loop.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement -> `llmfs check 11` -> `llmfs hint 11 -e N`
-> `SOLUTION.md` has the complete code.

Exercise 1 is the longest in the course. The other three are short.

WHAT YOU ARE GOING TO BUILD
===========================

The four pieces that make a training run work at scale:

    AdamWScratch        (ex. 1)  the optimizer, from scratch
    lr_at_step          (ex. 2)  how the learning rate changes during the run
    clip_grad_norm      (ex. 3)  stopping an odd batch destroying hours of work
    build_param_groups  (ex. 4)  which parameters decay and which do not

When all four are green, the final model will train with YOUR optimizer.

VOCABULARY YOU ARE GOING TO NEED
================================

- **optimizer**: the algorithm that decides how to apply the gradients to the weights.
- **learning rate** (lr): how far the weights move on each step. The hyperparameter that
  ruins the most training runs.
- **momentum**: a running average of the recent gradients, to smooth out the noise.
- **weight decay**: pushing the weights towards zero so they do not grow without control.
- **warmup**: raising the learning rate slowly over the first steps.
- **AMP / GradScaler**: training in 16 bits by multiplying the loss by a large number so the
  gradients do not go to zero.
- **parameter groups**: subsets with different hyperparameters. PyTorch accepts them as a
  list of dicts.

    llmfs demo 11     compares your AdamW with PyTorch's and measures the clipping
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn as nn


class AdamWScratch(torch.optim.Optimizer):
    """AdamW from scratch. It inherits from `torch.optim.Optimizer` and you only write `step`.

    WHAT YOU HAVE TO WRITE
    ----------------------
    The `__init__` IS ALREADY DONE. Your only job is the `step()` method, and it has a fixed
    structure: two nested loops (per group, per parameter) and six operations inside.

        1. The skeleton of the two loops:

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

        2. That parameter's state, creating it the first time:

                           state = self.state[p]
                           if len(state) == 0:
                               state["step"] = 0
                               state["exp_avg"] = torch.zeros_like(p)
                               state["exp_avg_sq"] = torch.zeros_like(p)

                           state["step"] += 1
                           t = state["step"]
                           m, v = state["exp_avg"], state["exp_avg_sq"]

        3. THE WEIGHT DECAY, DECOUPLED (this goes on the parameter, NOT on the gradient):

                           if wd != 0.0:
                               p.mul_(1.0 - lr * wd)

        4. The two running averages:

                           m.mul_(beta1).add_(grad, alpha=1 - beta1)
                           v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        5. The bias correction:

                           bias1 = 1 - beta1 ** t
                           bias2 = 1 - beta2 ** t
                           step_size = lr / bias1
                           denom = (v.sqrt() / math.sqrt(bias2)).add_(eps)

        6. The update:

                           p.addcdiv_(m, denom, value=-step_size)

        7. And outside all the loops: `return loss`

    WHAT ADAM DOES, IN TWO IDEAS
    ----------------------------
    **Momentum.** Instead of moving along THIS step's gradient, it uses a running average of
    the recent ones (that is `m`). Each batch is a different sample and its gradients are
    noisy; averaging cancels the noise.

    **Per-dimension scaling.** It also keeps a running average of the SQUARED gradient (`v`)
    and divides by its square root. A parameter with always-large gradients moves little; one
    that almost never receives signal moves a lot when it does. Each parameter ends up with
    its own effective learning rate, which is why a single global `lr` works for the whole
    model.

    THE FORMULAS, with t the step number STARTING AT 1
    -------------------------------------------------
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g²

        m_hat = m / (1 - beta1^t)          <- the bias correction
        v_hat = v / (1 - beta2^t)

        p -= lr * m_hat / (sqrt(v_hat) + eps)
        p -= lr * weight_decay * p         <- separately, see below

    In step 5 above the correction is rearranged (`lr/bias1` and `sqrt(v)/sqrt(bias2)`) to
    save one division per tensor. It is the same formula.

    HOW YOU WRITE AN OPTIMIZER IN PYTORCH
    -------------------------------------
    **`self.param_groups`** are exercise 4's groups: each with its own parameters and its own
    `weight_decay`. That is why the hyperparameters are read INSIDE the group loop and not
    once at the start.

    **`self.state[p]`** is a per-parameter `defaultdict` where you store m, v and the counter.
    The first time you touch a parameter it is empty (`len(state) == 0`) and it has to be
    initialized. PyTorch serializes it for you in `optimizer.state_dict()`, which is what
    makes resuming a training run halfway through possible.

    **The `@torch.no_grad()`** is mandatory. You are modifying parameters that have
    `requires_grad=True`; without it you would be building an autograd graph over the updates
    themselves, which besides being wrong would eat memory.

    THE THREE MISTAKES TO AVOID
    ---------------------------
    **Starting t at 0.** With t=0, `1 - beta^0 = 0` and you divide by zero. Increment
    `state["step"]` BEFORE using it.

    **Forgetting the bias correction.** `m` and `v` start at zero, so the first steps
    underestimate the magnitudes. With beta2=0.95, after one step `v` is only 5% of g², and
    dividing by its square root would give a step 4.5 times larger than it should be. Without
    the correction, training can diverge in the first steps and you will never blame this.

    **Adding the weight decay to the gradient.** That is THE difference between Adam+L2 and
    AdamW:

        Adam + L2:  grad = grad + wd * p     <- WRONG, not what we want
        AdamW:      p.mul_(1 - lr * wd)      <- directly on the parameter

    If you add it to the gradient, the decay goes through the division by `sqrt(v)` and its
    real effect ends up depending on the magnitude of that parameter's gradients. Decoupled it
    is uniform and predictable. There is a test that tells the two versions apart.

    THE IN-PLACE OPERATIONS (recommended, not mandatory)
    ----------------------------------------------------
    With 8.9 million parameters, allocating new tensors on every step shows. The in-place
    versions end in an underscore:

        m.mul_(beta1).add_(grad, alpha=1 - beta1)            # m = beta1*m + (1-beta1)*g
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v = beta2*v + (1-beta2)*g²
        p.addcdiv_(m, denom, value=-step_size)               # p -= step_size * m/denom

    If they look cryptic, write it with normal operations first (`m = beta1*m + ...`) and
    optimize afterwards. The test compares results, not style. But watch out: with the
    non-in-place version you have to store the result back into `state["exp_avg"]`, because
    you would be creating new tensors instead of modifying the existing ones.

    HOW TO KNOW IF IT IS RIGHT
    --------------------------
    The test trains the same problem for 50 steps with your optimizer and with
    `torch.optim.AdamW`, and compares the final weights. They have to match with
    `torch.allclose`.
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
        """One optimization step. The full instructions are in the class docstring.

        The `closure` is a PyTorch convention almost nobody uses, but it is respected for
        compatibility. If it is not None, it is called inside `torch.enable_grad()` and its
        result is returned:

            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            ... the rest ...
            return loss
        """
        raise NotImplementedError("TODO: module 11, exercise 1 - AdamWScratch.step")


def lr_at_step(
    step: int,
    max_steps: int,
    lr: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
    schedule: str = "cosine",
) -> float:
    """The learning rate for a given step: linear warmup + cosine decay.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Three segments, in this order (the order matters: warmup gets checked first).

        1. The floor, used in every segment:

               min_lr = lr * min_lr_ratio

        2. WARMUP. If `step < warmup_steps`:

               return lr * (step + 1) / warmup_steps

        3. If `schedule == "constant"`, you are done: `return lr`.

        4. The progress, clamped to [0, 1]:

               progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
               progress = min(1.0, max(0.0, progress))

        5. And according to the schedule:

               if schedule == "cosine":
                   coef = 0.5 * (1.0 + math.cos(math.pi * progress))
               elif schedule == "linear":
                   coef = 1.0 - progress
               else:
                   raise ValueError(f"unknown schedule: {schedule}")

               return min_lr + (lr - min_lr) * coef

    CHECK THE ENDPOINTS BY HAND
    ---------------------------
    It is the way to know the formula is right without running anything:

        progress = 0  ->  cos(0) = 1   ->  coef = 1  ->  returns lr          OK
        progress = 1  ->  cos(pi) = -1 ->  coef = 0  ->  returns min_lr      OK

    If it comes out backwards, you left out the `0.5 * (1 + ...)` and you are using the raw
    cosine, which runs from 1 to -1 instead of 1 to 0.

    THE `+1` IN THE WARMUP
    ----------------------
    `lr * (step + 1) / warmup_steps` instead of `lr * step / warmup_steps`. Without it, step 0
    would have an lr of exactly zero: a step that learns nothing and is wasted. It is a minor
    detail but the tests check it.

    WHY WARMUP EXISTS
    -----------------
    In the first steps Adam's moments are nearly empty and its estimates are extremely noisy
    (it is the same problem the bias correction fixes, but the correction does not fully solve
    it). And on top of that the freshly initialized weights produce large gradients. Starting
    at full lr usually produces a loss spike the model sometimes never recovers from.

    WHY A COSINE AND NOT A STRAIGHT LINE
    ------------------------------------
    It drops slowly at the start (you still want to move fast and explore), fast in the
    middle, and slowly again at the end (fine-tuning in a good region). The difference from
    linear is small but consistent across every paper that has measured it.

    WHY IT DOES NOT DECAY TO ZERO
    -----------------------------
    `min_lr_ratio=0.1` leaves a 10% floor. Below a certain point the model stops learning
    entirely and every extra step is wasted compute. If you are going to stop, better to stop.

    TWO PROTECTIONS THAT ARE NOT DECORATIVE
    ---------------------------------------
    The `max(1, ...)` in the denominator avoids dividing by zero if
    `max_steps <= warmup_steps`. The clamp to [0, 1] means calling with `step > max_steps`
    returns `min_lr` instead of starting to RISE again (the cosine is periodic: with progress
    > 1 it would grow again).

    Args:
        step: the current step, starting at 0.
        max_steps: the total number of steps in the run.
        lr: the maximum learning rate, the peak one.
        warmup_steps: how many steps the ramp lasts.
        min_lr_ratio: the fraction of `lr` that is the floor.
        schedule: "cosine" (default), "linear" or "constant".

    Returns:
        That step's learning rate.

    Raises:
        ValueError: if `schedule` is none of the three.
    """
    raise NotImplementedError("TODO: module 11, exercise 2 - lr_at_step")


def clip_grad_norm(parameters: Iterable[nn.Parameter], max_norm: float) -> float:
    """Clips the gradients so their GLOBAL norm does not exceed `max_norm`.

    WHAT YOU HAVE TO WRITE
    ----------------------
        1. Gather the gradients that exist:

               grads = [p.grad for p in parameters if p.grad is not None]
               if not grads:
                   return 0.0

        2. The global norm, as if they were all ONE giant vector:

               total = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
               total_norm = float(total)

        3. The clipping, only if needed:

               if max_norm > 0 and total_norm > max_norm:
                   factor = max_norm / (total_norm + 1e-6)
                   for g in grads:
                       g.mul_(factor)

        4. `return total_norm`   <- the norm BEFORE clipping

    Watch out in step 1: `parameters` can be a generator (`model.parameters()` is one), and a
    generator is exhausted once you walk it. If you walk it twice, the second time it is
    empty. That is why the list of gradients is materialized ONCE and everything works over
    it.

    WHAT PROBLEM IT SOLVES
    ----------------------
    Occasionally a batch produces enormous gradients: an odd sequence, a very rare token, a
    corrupted line in the dataset. Without protection, that SINGLE batch can take a jump that
    destroys hours of training, and you will see it as a vertical spike in the loss curve the
    model takes a long time to recover from (or never does).

    WHY THE GLOBAL NORM AND NOT ONE PER TENSOR
    ------------------------------------------
    Clipping each tensor separately would change the DIRECTION of the combined gradient, which
    is exactly what you do not want to touch. The gradient tells you which way to go; you are
    only limiting HOW FAR you move in that direction. By multiplying every tensor by the same
    scalar, the direction is preserved exactly.

    There is a test that checks it: the normalized vector before and after clipping has to be
    identical.

    WHY THE NORM **BEFORE** CLIPPING IS RETURNED
    --------------------------------------------
    It is what `torch.nn.utils.clip_grad_norm_` does, and it is the useful one. If you log it
    and see it rising steadily, training is destabilizing and you find out BEFORE it blows up.
    If you returned the post-clipping norm you would see `max_norm` pinned and learn nothing.

    THE `1e-6` AND THE `.detach()`
    ------------------------------
    The `1e-6` in the denominator avoids dividing by zero if the norm is tiny. It never
    happens in practice, but it costs five characters.

    The `.detach()` when computing the norm: gradients do not require gradient, so right now
    it makes no difference. It is the correct habit and it avoids surprises if you ever use
    higher-order graphs.

    Args:
        parameters: the model's parameters (`model.parameters()`).
        max_norm: the threshold. If it is <= 0, nothing is clipped (but the norm is still
            returned).

    Returns:
        The global norm BEFORE clipping. `0.0` if there is no gradient at all.
    """
    raise NotImplementedError("TODO: module 11, exercise 3 - clip_grad_norm")


def build_param_groups(model: nn.Module, weight_decay: float = 0.1) -> list[dict[str, Any]]:
    """Splits the parameters into two groups: with weight decay and without it.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Five lines.

        1. The two lists:

               decay, no_decay = [], []

        2. The split, skipping anything frozen:

               for param in model.parameters():
                   if not param.requires_grad:
                       continue
                   (decay if param.dim() >= 2 else no_decay).append(param)

        3. The two groups, IN THIS ORDER (there are tests that depend on the order):

               return [
                   {"params": decay, "weight_decay": weight_decay},
                   {"params": no_decay, "weight_decay": 0.0},
               ]

    THE RULE, and it is surprisingly simple
    ---------------------------------------
        Parameters with 2 dimensions or more  ->  WITH weight decay
        Parameters with 1 dimension           ->  WITHOUT weight decay

    That is: the matrices decay, and the biases and normalization scales do not.
    `param.dim()` gives the number of dimensions: a weight matrix has 2, a bias has 1.

    WHY THAT RULE
    -------------
    Weight decay pushes weights towards zero. On a projection matrix that makes sense:
    penalizing large magnitudes reduces overfitting.

    On an RMSNorm's scale it makes NONE. That parameter starts at 1 and its job is to rescale
    the layer's output; pushing it towards zero is pushing the output towards zero, which is
    exactly the opposite of what is needed.

    The same goes for biases: they are offsets, not magnitudes worth limiting.

    Applying decay to everything is a common mistake, it produces NO visible error, and it
    degrades the result. It can only be detected by comparing two complete training runs,
    which is expensive. That is why it is worth getting right from the start.

    THE FORMAT PYTORCH EXPECTS
    --------------------------
    A list of dictionaries, each with at least the `"params"` key. Any extra key (`lr`,
    `betas`, `weight_decay`...) overrides the optimizer's default value for that group ONLY.
    It is PyTorch's standard mechanism, and it is what your `AdamWScratch.step` reads when it
    does `for group in self.param_groups`.

    TWO DETAILS
    -----------
    **Skipping `requires_grad=False`.** Those parameters are not going to be updated; putting
    them in the optimizer only wastes state memory (two tensors per parameter). In module 16,
    with LoRA, this goes from a detail to essential: almost the whole model is frozen.

    **Tied weights.** `model.parameters()` already deduplicates by identity, so the tied
    embedding appears ONCE and goes into the decay group (it has 2 dimensions). Nothing
    special needs doing.

    Args:
        model: the model.
        weight_decay: the value for the group that does decay.

    Returns:
        The list of two groups, the first with decay and the second without, IN THAT ORDER.
    """
    raise NotImplementedError("TODO: module 11, exercise 4 - build_param_groups")
