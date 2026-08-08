"""Module 13 - The real run.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement the two exercises -> `llmfs check 13` -> and then TRAIN:

    llmfs train --config tiny_char

WHAT YOU ARE GOING TO BUILD
===========================

    overfit_single_batch  (ex. 1)  the 30-second check that catches almost everything
    format_eta            (ex. 2)  how long is left, in something readable

And with that you launch the real training. Careful though: the training itself IS ALREADY
WRITTEN, assembled on top of the pieces from modules 04 to 12. Your job here is to launch it,
know how to read it and decide whether it is going well.

`THEORY.md` follows this same order and each docstring here tells you which section it maps to.
And there is one section that matches no exercise and is the skill you actually take away: "How
to read the log line", which takes apart field by field the line you will be staring at for
hours. Each of its six fields comes from a different module of the course.

EXERCISE 1 IS THE ONE THAT MATTERS
==================================

A model with millions of parameters memorizes four sequences without breaking a sweat. If you
give it the SAME batch over and over and the loss does not drop to almost zero, there is a bug.

And you know it in 30 seconds instead of in four hours. It is the best cost/benefit advice in
all of deep learning, and almost nobody applies it.

VOCABULARY YOU ARE GOING TO NEED
================================

- **overfit**: the model memorizing instead of generalizing. Normally it is bad; here it is
  wanted on purpose, as a sanity check.
- **checkpoint**: a snapshot of the training (weights, optimizer state, step number) so you
  can resume.
- **ETA**: how long is left to finish, estimated from the measured pace.
- **step**: one update of the weights. Not to be confused with epoch, which is a complete
  pass over the data.

    llmfs demo 13     does the overfit and trains a complete model
"""

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
    """Trains ONE SINGLE batch until it memorizes it. The test that catches almost any bug.

    Context in `THEORY.md`: section "Exercise 1: the 30-second check", with the measured table of
    how the loss drops (from 4.29 to 0.09 in 300 steps and two seconds on the clock) and the list
    of what it catches and what it does not.

    WHAT YOU HAVE TO WRITE
    ----------------------
    The simplest training loop there is. Four steps.

        1. The optimizer, using the one they pass you or AdamW if not:

               if optimizer_factory is None:
                   opt = torch.optim.AdamW(model.parameters(), lr=lr)
               else:
                   opt = optimizer_factory(model.parameters())

        2. Training mode and the list where you keep the history:

               model.train()
               history = []

        3. The loop, `steps` times, always with THE SAME batch:

               for _ in range(steps):
                   _, loss = model(x, y)
                   opt.zero_grad(set_to_none=True)
                   loss.backward()
                   opt.step()
                   history.append(float(loss.detach()))

        4. `return history`

    No scheduler, no gradient accumulation, no AMP. On purpose: the fewer pieces there are,
    the fewer places a bug can hide. This loop is the reference pattern you will compare the
    real loop against when something fails.

    WHAT SHOULD COME OUT
    --------------------
    The loss starts at `ln(vocab_size)` and drops to almost zero. With the toy model:

        step   0:  4.17
        step  50:  1.12
        step 100:  0.21
        step 200:  0.02

    If `history[-1]` is not far below `history[0]`, STOP and look for the bug. Do not launch
    the long training run.

    THE IDEA
    --------
    A model with millions of parameters has more than enough capacity to memorize four
    sequences. If you give it the SAME batch over and over, the loss HAS to drop practically
    to zero. There is nothing to generalize: only to memorize.

    If it does not drop, there is a bug. And you know it in 30 seconds instead of in four
    hours.

    WHAT IT CATCHES
    ---------------
        - gradients that do not reach some part of the model (one `detach()` too many)
        - the forgotten `zero_grad()` (gradients ACCUMULATE by default in PyTorch)
        - an absurd learning rate, too high or too low
        - a layer disconnected from the graph
        - the optimizer built on the wrong parameters

    WHAT IT DOES NOT CATCH
    ----------------------
    Anything to do with generalization. A model that memorizes a batch perfectly can still be
    completely useless on new data. This checks that the MACHINERY works, not that the model
    is good.

    IF IT DROPS TOO FAST, ALSO BE SUSPICIOUS
    ----------------------------------------
    If the loss settles at zero in 5 steps, look for an information leak: check that `y` is
    shifted ONE token relative to `x`. If you pass `model(x, x)` the model only has to copy
    the input and the loss collapses. The symptom is identical to that of a broken causal
    mask.

    THE `set_to_none=True`
    ----------------------
    `opt.zero_grad(set_to_none=True)` sets the `.grad`s to `None` instead of to zeros. It
    saves a write pass over every gradient and some memory. It is the default since PyTorch
    2.0; it is written explicitly because it reads better.

    THIS IS THE FIRST CHECK YOU HAVE TO DO, always, before launching any long training run.
    It is the best cost/benefit advice in all of deep learning, and almost nobody applies it.

    Args:
        model: the model, with forward `(idx, targets) -> (logits, loss)`.
        x, y: the batch to memorize. `y` shifted one token relative to `x`.
        steps: how many steps to take.
        lr: learning rate (only used if you do not pass `optimizer_factory`).
        optimizer_factory: `fn(params) -> optimizer`, so you can try yours from module 11.

    Returns:
        The loss history, one per step.
    """
    raise NotImplementedError("TODO: module 13, exercise 1 - overfit_single_batch")


def format_eta(seconds: float) -> str:
    """Formats a duration into something readable at a glance.

    Context in `THEORY.md`: section "Exercise 2: how much is left", with the two reasons this is
    not cosmetic. The second is a rule that serves you well beyond this exercise: the useful
    precision of an estimate is always proportional to its magnitude.

    WHAT YOU HAVE TO WRITE
    ----------------------
    A chain of `if`s, from smallest to largest.

        1. The odd cases, FIRST:

               if not math.isfinite(seconds) or seconds < 0:
                   return "?"

        2. To an integer, since fractional seconds add nothing:

               seconds = int(seconds)

        3. The four bands:

               if seconds < 60:
                   return f"{seconds}s"
               if seconds < 3600:
                   return f"{seconds // 60}m {seconds % 60}s"
               if seconds < 86400:
                   return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
               return f"{seconds // 86400}d {(seconds % 86400) // 3600}h"

    WHAT SHOULD COME OUT
    --------------------
            45   ->  "45s"
           125   ->  "2m 5s"
          3725   ->  "1h 2m"
         90000   ->  "1d 1h"
            -1   ->  "?"
        float("inf")  ->  "?"

    Check the 3725 by hand: 3725 // 3600 = 1, and (3725 % 3600) // 60 = 125 // 60 = 2. That
    gives "1h 2m". And notice that the 5 leftover seconds are lost, which is exactly what you
    want.

    WHY THE SECONDS STOP BEING SHOWN FROM ONE HOUR ON
    -------------------------------------------------
    When there are two hours left, the seconds are noise: they change constantly, they add no
    information and they make the number dance on screen. The useful precision of an estimate
    is always proportional to its magnitude.

    WHY "?" AND NOT A 0
    -------------------
    Returning "?" is the honest thing when there is not yet enough data to estimate (in the
    first few steps the average speed means nothing). And it avoids printing things like "-1s"
    or "infd 0h", which besides being ugly make you doubt whether the training is going well.

    `math.isfinite(x)` is False for `inf`, `-inf` and `nan`. All three come out of dividing by
    zero, or a zero over zero, when computing the pace.

    THIS LOOKS COSMETIC AND IT IS NOT
    ---------------------------------
    You are going to look at this number many times during a run that lasts hours. "1h 2m"
    reads instantly; "3725s" you have to divide mentally every time.

    Args:
        seconds: the duration in seconds.

    Returns:
        The formatted string, or "?" if the input is not usable.
    """
    raise NotImplementedError("TODO: module 13, exercise 2 - format_eta")
