"""Tests for module 09. Run them with `llmfs check 09`."""

from __future__ import annotations

import math

import pytest
import torch

import llmfs.reference as ref
from llmfs.testing import assert_close, assert_scalar_close, load_exercises

ex = load_exercises(__file__)


# ---------------------------------------------- exercise 1: sinusoidal_embeddings


def test_the_sinusoidals_have_the_right_shape():
    assert ex.sinusoidal_embeddings(64, 32).shape == (64, 32)


def test_the_sinusoidals_match_the_reference():
    assert_close(
        ex.sinusoidal_embeddings(64, 32), ref.sinusoidal_embeddings(64, 32), what="the table"
    )


def test_the_even_dimensions_are_sine_and_the_odd_ones_cosine():
    table = ex.sinusoidal_embeddings(16, 8)
    # At position 0 every angle is 0: sin(0)=0 and cos(0)=1
    assert_close(table[0, 0::2], torch.zeros(4), atol=1e-6, what="the even dimensions at pos 0")
    assert_close(table[0, 1::2], torch.ones(4), atol=1e-6, what="the odd ones at pos 0")


def test_the_values_are_bounded_between_minus_one_and_one():
    table = ex.sinusoidal_embeddings(500, 64)
    assert float(table.max()) <= 1.0 + 1e-6 and float(table.min()) >= -1.0 - 1e-6


def test_each_position_has_a_distinct_signature():
    table = ex.sinusoidal_embeddings(128, 32)
    distances = torch.cdist(table, table)
    distances.fill_diagonal_(float("inf"))
    assert float(distances.min()) > 1e-3, "there are two positions with the same vector"


def test_the_first_dimensions_oscillate_faster_than_the_last():
    """The frequency ladder, which is the binary-counter idea."""
    table = ex.sinusoidal_embeddings(200, 64)
    fast_variation = float((table[1:, 0] - table[:-1, 0]).abs().mean())
    slow_variation = float((table[1:, -2] - table[:-1, -2]).abs().mean())
    assert fast_variation > slow_variation * 10, (
        f"the first dimension varies {fast_variation:.4f} per position and the last one "
        f"{slow_variation:.6f}: they should differ by orders of magnitude"
    )


def test_it_is_defined_for_any_position():
    """Unlike a learned table, there is no ceiling."""
    assert torch.isfinite(ex.sinusoidal_embeddings(100_000, 16)).all()


# ------------------------------------------------------ exercise 2: rope_frequencies


def test_rope_returns_two_tables_with_head_dim_columns():
    cos, sin = ex.rope_frequencies(16, 64)
    assert cos.shape == (64, 16), "the tables have head_dim columns, not head_dim/2"
    assert sin.shape == (64, 16)


def test_rope_matches_the_reference():
    my_cos, my_sin = ex.rope_frequencies(40, 512)
    ref_cos, ref_sin = ref.rope_frequencies(40, 512)
    assert_close(my_cos, ref_cos, what="the cosine table")
    assert_close(my_sin, ref_sin, what="the sine table")


def test_rope_duplicates_the_frequencies_by_halves():
    """The halves convention: column i and column i+head_dim/2 carry the SAME angle."""
    cos, sin = ex.rope_frequencies(16, 32)
    assert_close(cos[:, :8], cos[:, 8:], what="the two halves of cos")
    assert_close(sin[:, :8], sin[:, 8:], what="the two halves of sin")


def test_at_position_zero_there_is_no_rotation():
    cos, sin = ex.rope_frequencies(16, 32)
    assert_close(cos[0], torch.ones(16), atol=1e-6, what="cos at position 0")
    assert_close(sin[0], torch.zeros(16), atol=1e-6, what="sin at position 0")


def test_cos_squared_plus_sin_squared_is_one():
    """A basic trig identity: if it fails, the tables are not of a real angle."""
    cos, sin = ex.rope_frequencies(16, 64)
    assert_close(cos**2 + sin**2, torch.ones(64, 16), atol=1e-5, what="cos^2 + sin^2")


def test_an_odd_head_dim_is_an_error():
    with pytest.raises(ValueError):
        ex.rope_frequencies(15, 32)


def test_the_first_frequency_turns_faster_than_the_last():
    cos, _ = ex.rope_frequencies(64, 256)
    fast = float((cos[1:, 0] - cos[:-1, 0]).abs().mean())
    slow = float((cos[1:, 31] - cos[:-1, 31]).abs().mean())
    assert fast > slow * 100


def test_it_respects_the_device():
    from llmfs.device import get_device

    cfg = get_device()
    cos, sin = ex.rope_frequencies(16, 32, device=cfg.device)
    assert cos.device.type == cfg.kind and sin.device.type == cfg.kind


# ------------------------------------------------------------ exercise 3: apply_rope


def test_apply_rope_preserves_the_shape():
    cos, sin = ref.rope_frequencies(16, 64)
    x = torch.randn(2, 4, 8, 16)
    assert ex.apply_rope(x, cos, sin).shape == (2, 4, 8, 16)


def test_apply_rope_matches_the_reference():
    torch.manual_seed(0)
    cos, sin = ref.rope_frequencies(16, 64)
    x = torch.randn(2, 4, 8, 16)
    assert_close(ex.apply_rope(x, cos, sin), ref.apply_rope(x, cos, sin), what="RoPE")


def test_rotating_does_not_change_the_vectors_length():
    """The advantage over adding a positional embedding: only the direction changes."""
    torch.manual_seed(0)
    cos, sin = ref.rope_frequencies(16, 64)
    x = torch.randn(2, 4, 8, 16)
    assert_close(
        ex.apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5, what="the norm"
    )


def test_at_position_zero_the_vector_does_not_change():
    cos, sin = ref.rope_frequencies(16, 64)
    x = torch.randn(1, 1, 1, 16)
    assert_close(ex.apply_rope(x, cos, sin), x, atol=1e-6, what="the vector at position 0")


def test_different_positions_rotate_differently():
    cos, sin = ref.rope_frequencies(16, 64)
    x = torch.randn(1, 1, 1, 16).repeat(1, 1, 5, 1)  # the same vector at 5 positions
    rotated = ex.apply_rope(x, cos, sin)
    for pos in range(1, 5):
        assert not torch.allclose(rotated[0, 0, 0], rotated[0, 0, pos], atol=1e-4)


def test_the_relative_invariance_is_exact():
    """THE property that justifies RoPE.

    The dot product of two rotated vectors depends only on the DIFFERENCE of positions:
    <R(m)q, R(n)k> = <q, R(n-m)k>.

    Here we check that the score between positions (2,5) is the same as between (4,7): in
    both cases the distance is 3.
    """
    torch.manual_seed(0)
    cos, sin = ref.rope_frequencies(16, 64)
    q_base, k_base = torch.randn(16), torch.randn(16)

    def score(pos_q: int, pos_k: int) -> float:
        q = torch.zeros(1, 1, 32, 16)
        k = torch.zeros(1, 1, 32, 16)
        q[0, 0, pos_q] = q_base
        k[0, 0, pos_k] = k_base
        qr = ex.apply_rope(q, cos, sin)
        kr = ex.apply_rope(k, cos, sin)
        return float(qr[0, 0, pos_q] @ kr[0, 0, pos_k])

    assert_scalar_close(
        score(2, 5), score(4, 7), rtol=1e-5, what="the score at distance 3"
    )
    assert_scalar_close(
        score(0, 10), score(20, 30), rtol=1e-5, what="the score at distance 10"
    )
    # And at different distances it has to give different things
    assert abs(score(2, 5) - score(2, 9)) > 1e-4


def test_it_works_with_a_sequence_shorter_than_the_tables():
    """cos/sin come with max_seq_len rows: they have to be sliced to the real length."""
    cos, sin = ref.rope_frequencies(16, 512)
    x = torch.randn(2, 4, 7, 16)
    assert ex.apply_rope(x, cos, sin).shape == (2, 4, 7, 16)


def test_it_works_in_fp16():
    """With AMP, x arrives in fp16 and the tables are in fp32: conversion is needed."""
    cos, sin = ref.rope_frequencies(16, 64)
    x = torch.randn(1, 2, 4, 16, dtype=torch.float16)
    out = ex.apply_rope(x, cos, sin)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()


def test_it_is_linear_in_x():
    """Rotating is a linear transformation: R(2x) = 2·R(x)."""
    cos, sin = ref.rope_frequencies(16, 64)
    x = torch.randn(1, 1, 4, 16)
    assert_close(ex.apply_rope(2 * x, cos, sin), 2 * ex.apply_rope(x, cos, sin), atol=1e-5)
