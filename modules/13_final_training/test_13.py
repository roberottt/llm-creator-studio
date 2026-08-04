"""Module 13 tests. Run them with `llmfs check 13`."""

from __future__ import annotations

import math

import pytest
import torch

import llmfs.reference as ref
from llmfs.config import ModelConfig
from llmfs.device import set_seed
from llmfs.testing import load_exercises

ex = load_exercises(__file__)


def small_model() -> torch.nn.Module:
    set_seed(0)
    return ref.GPT(
        ModelConfig(vocab_size=64, n_layers=2, d_model=32, n_heads=4, d_ff=96, context_length=16)
    )


def batch(vocab: int = 64, b: int = 4, t: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    set_seed(1)
    seq = torch.randint(0, vocab, (b, t + 1))
    return seq[:, :-1], seq[:, 1:]


# ------------------------------------------------- exercise 1: overfit_single_batch


def test_the_loss_drops_to_almost_zero():
    """THE check: a healthy model memorizes four sequences without breaking a sweat."""
    x, y = batch()
    history = ex.overfit_single_batch(small_model(), x, y, steps=300, lr=3e-3)
    assert history[-1] < 0.1, (
        f"the loss got stuck at {history[-1]:.4f} memorizing a single batch. "
        "If it does not drop, there is a bug in the model or in the loop."
    )


def test_it_starts_close_to_ln_of_the_vocabulary():
    x, y = batch()
    history = ex.overfit_single_batch(small_model(), x, y, steps=10, lr=1e-3)
    assert abs(history[0] - math.log(64)) < 0.5, (
        f"it starts at {history[0]:.4f} and should be around ln(64)={math.log(64):.4f}"
    )


def test_it_returns_one_loss_per_step():
    x, y = batch()
    assert len(ex.overfit_single_batch(small_model(), x, y, steps=37, lr=1e-3)) == 37


def test_the_loss_is_roughly_monotonic():
    x, y = batch()
    h = ex.overfit_single_batch(small_model(), x, y, steps=200, lr=3e-3)
    first_third = sum(h[:60]) / 60
    last_third = sum(h[-60:]) / 60
    assert last_third < first_third / 5


def test_it_accepts_your_own_optimizer():
    x, y = batch()
    h = ex.overfit_single_batch(
        small_model(),
        x,
        y,
        steps=100,
        optimizer_factory=lambda p: ref.AdamWScratch(p, lr=3e-3),
    )
    assert h[-1] < h[0]


def test_the_values_are_finite():
    x, y = batch()
    h = ex.overfit_single_batch(small_model(), x, y, steps=100, lr=3e-3)
    assert all(math.isfinite(v) for v in h), "there are infs or nans in the history"


def test_it_leaves_the_model_in_training_mode():
    x, y = batch()
    model = small_model()
    model.eval()
    ex.overfit_single_batch(model, x, y, steps=5, lr=1e-3)
    assert model.training, "you have to call model.train() before the loop"


def test_the_overfit_matches_the_reference():
    x, y = batch()
    set_seed(0)
    mine = ex.overfit_single_batch(small_model(), x, y, steps=50, lr=3e-3)
    set_seed(0)
    theirs = ref.overfit_single_batch(small_model(), x, y, steps=50, lr=3e-3)
    for i, (a, b) in enumerate(zip(mine, theirs)):
        assert abs(a - b) < 1e-4, f"they diverge at step {i}: {a:.6f} vs {b:.6f}"


# ------------------------------------------------------------ exercise 2: format_eta


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "0s"),
        (1, "1s"),
        (45, "45s"),
        (59, "59s"),
        (60, "1m 0s"),
        (125, "2m 5s"),
        (3599, "59m 59s"),
        (3600, "1h 0m"),
        (3725, "1h 2m"),
        (86399, "23h 59m"),
        (86400, "1d 0h"),
        (90000, "1d 1h"),
    ],
)
def test_the_formats_of_each_band(seconds, expected):
    assert ex.format_eta(seconds) == expected


def test_from_one_hour_on_the_seconds_are_not_shown():
    """When there are two hours left, the seconds are noise."""
    assert ex.format_eta(7265).count("s") == 0 or "m" in ex.format_eta(7265)
    assert ex.format_eta(7265) == "2h 1m"


def test_the_odd_values_return_a_question_mark():
    for bad in (-1, -1000, float("inf"), float("nan"), float("-inf")):
        assert ex.format_eta(bad) == "?", f"{bad} should give '?'"


def test_it_accepts_floats():
    assert ex.format_eta(45.7) == "45s"


def test_the_format_matches_the_reference():
    for s in (0, 30, 59, 60, 125, 3599, 3600, 3725, 86400, 123456, -5, float("inf")):
        assert ex.format_eta(s) == ref.format_eta(s), f"they disagree at {s}"
