"""Tests for module 01. Run them with `llmfs check 01`."""

from __future__ import annotations

import math

import pytest
import torch

import llmfs.reference as ref
from llmfs.device import get_device
from llmfs.testing import assert_scalar_close, load_exercises

ex = load_exercises(__file__)


# --------------------------------------------------------------- exercise 2: FLOPs


@pytest.mark.parametrize(
    "params",
    [
        # this course's final model
        dict(n_layers=6, d_model=320, d_ff=896, context_length=512, vocab_size=4096),
        # the shakespeare toy
        dict(n_layers=4, d_model=128, d_ff=384, context_length=128, vocab_size=65),
        # something large, so the attention term weighs differently
        dict(n_layers=32, d_model=4096, d_ff=11008, context_length=4096, vocab_size=32000),
    ],
)
def test_flops_per_token_matches_the_reference(params):
    assert ex.transformer_flops_per_token(**params) == ref.transformer_flops_per_token(**params)


def test_the_9m_models_count_is_the_expected_one():
    """A fixed number: if it changes, THEORY.md and module 11 are lying."""
    flops = ex.transformer_flops_per_token(
        n_layers=6, d_model=320, d_ff=896, context_length=512, vocab_size=4096
    )
    assert flops == 65_372_160


def test_the_backward_costs_twice_the_forward():
    common = dict(n_layers=6, d_model=320, d_ff=896, context_length=512, vocab_size=4096)
    fwd_only = ex.transformer_flops_per_token(**common, include_backward=False)
    with_bwd = ex.transformer_flops_per_token(**common, include_backward=True)
    assert with_bwd == 3 * fwd_only


def test_the_attention_term_grows_with_the_context():
    """The 6N part does not depend on T; the attention one does, linearly."""
    base = dict(n_layers=6, d_model=320, d_ff=896, vocab_size=4096)
    short = ex.transformer_flops_per_token(**base, context_length=512)
    long = ex.transformer_flops_per_token(**base, context_length=1024)
    attention_term = 3 * 4 * 6 * 512 * 320
    assert long - short == attention_term


def test_a_classic_two_matrix_ffn_costs_less_than_swiglu():
    base = dict(n_layers=6, d_model=320, d_ff=896, context_length=512, vocab_size=4096)
    assert ex.transformer_flops_per_token(**base, n_ffn_matrices=2) < ex.transformer_flops_per_token(
        **base, n_ffn_matrices=3
    )


def test_it_returns_an_integer():
    flops = ex.transformer_flops_per_token(
        n_layers=6, d_model=320, d_ff=896, context_length=512, vocab_size=4096
    )
    assert isinstance(flops, int)


# ------------------------------------------------------- exercise 3: tokens/second


def test_tokens_per_second_matches_the_reference():
    assert_scalar_close(
        ex.estimate_tokens_per_second(51.6, 65_372_160, mfu=0.15),
        ref.estimate_tokens_per_second(51.6, 65_372_160, mfu=0.15),
        what="tokens/s",
    )


def test_tokens_per_second_scales_linearly_with_mfu():
    a = ex.estimate_tokens_per_second(50.0, 1_000_000, mfu=0.1)
    b = ex.estimate_tokens_per_second(50.0, 1_000_000, mfu=0.2)
    assert_scalar_close(b, 2 * a, what="twice the MFU")


def test_a_non_positive_flops_per_token_is_an_error():
    with pytest.raises(ValueError):
        ex.estimate_tokens_per_second(50.0, 0)


# ------------------------------------------------------- exercise 1: real measurement


def test_the_measurement_returns_a_plausible_number():
    """It cannot be compared against a reference: it is a measurement. It gets bounded."""
    tflops = ex.measure_matmul_tflops(size=512, warmup=2, iters=5)
    assert isinstance(tflops, float)
    assert math.isfinite(tflops)
    assert 0.001 < tflops < 10_000, (
        f"{tflops:.3f} TFLOPS is not plausible. If it shoots up (thousands), you forgot "
        "to synchronize and you are measuring the time to enqueue."
    )


def test_measuring_on_cpu_gives_no_more_than_the_default_device():
    """On CPU there is no asynchrony, so this number IS trustworthy without synchronizing."""
    cpu = ex.measure_matmul_tflops(cfg=get_device("cpu"), size=512, warmup=2, iters=5)
    assert 0.001 < cpu < 100


def test_larger_matrices_do_not_tank_the_throughput():
    """A bigger matmul should use the hardware better, not worse.

    If this fails by a huge margin, it is almost always the missing warmup: the fixed cost
    of the first call is spread differently depending on the size.
    """
    small = ex.measure_matmul_tflops(size=256, warmup=3, iters=10)
    large = ex.measure_matmul_tflops(size=1024, warmup=3, iters=10)
    assert large > small / 10


def test_it_accepts_an_explicit_dtype():
    value = ex.measure_matmul_tflops(cfg=get_device("cpu"), size=256, dtype=torch.float32, iters=3)
    assert math.isfinite(value) and value > 0
