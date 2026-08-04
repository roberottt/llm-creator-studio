"""Tests for module 06. Run them with `llmfs check 06`.

The ground truth comes from two oracles: `llmfs.reference` and
`torch.nn.MultiheadAttention`.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import llmfs.reference as ref
from llmfs.testing import assert_close, copy_parameters, load_exercises

ex = load_exercises(__file__)


# ------------------------------------------------------------- exercise 1: causal_mask


def test_the_mask_is_lower_triangular():
    m = ex.causal_mask(4)
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(m, expected)


def test_the_mask_is_boolean():
    assert ex.causal_mask(5).dtype == torch.bool


def test_the_diagonal_is_included():
    """A token can look at itself."""
    m = ex.causal_mask(6)
    assert bool(m.diagonal().all())


def test_no_position_looks_into_the_future():
    m = ex.causal_mask(8)
    assert not bool(m.triu(diagonal=1).any()), "there are positions looking forwards"


def test_the_mask_matches_the_reference():
    for n in (1, 2, 7, 64):
        assert torch.equal(ex.causal_mask(n), ref.causal_mask(n))


def test_the_mask_respects_the_device():
    from llmfs.device import get_device

    cfg = get_device()
    assert ex.causal_mask(4, device=cfg.device).device.type == cfg.kind


# --------------------------------------------------- exercise 2: single_head_attention


def data(batch=2, t=5, d=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, t, d), torch.randn(batch, t, d), torch.randn(batch, t, d)


def test_the_output_shapes_are_right():
    q, k, v = data()
    out, weights = ex.single_head_attention(q, k, v)
    assert out.shape == (2, 5, 8)
    assert weights.shape == (2, 5, 5)


def test_each_row_of_weights_sums_to_one():
    q, k, v = data()
    _, weights = ex.single_head_attention(q, k, v, ref.causal_mask(5))
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6), (
        "each row must sum to 1. If not, you probably used dim=-2 in the softmax"
    )


def test_the_mask_zeroes_out_the_future():
    q, k, v = data()
    _, weights = ex.single_head_attention(q, k, v, ref.causal_mask(5))
    assert torch.allclose(weights[0].triu(1), torch.zeros(5, 5), atol=1e-9), (
        "there is weight on future positions: the mask is not being applied"
    )


def test_single_head_matches_the_reference():
    q, k, v = data()
    mask = ref.causal_mask(5)
    my_out, my_weights = ex.single_head_attention(q, k, v, mask)
    ref_out, ref_weights = ref.single_head_attention(q, k, v, mask)
    assert_close(my_out, ref_out, what="the output")
    assert_close(my_weights, ref_weights, what="the weights")


def test_it_also_works_without_a_mask():
    q, k, v = data()
    out, weights = ex.single_head_attention(q, k, v)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    assert torch.isfinite(out).all()


def test_the_scaling_by_sqrt_dk_is_applied():
    """Checked by comparing against the explicit computation with and without scaling."""
    torch.manual_seed(1)
    q, k, v = torch.randn(1, 4, 64), torch.randn(1, 4, 64), torch.randn(1, 4, 64)
    _, weights = ex.single_head_attention(q, k, v)

    scaled = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(64), dim=-1)
    unscaled = torch.softmax(q @ k.transpose(-2, -1), dim=-1)

    err_scaled = (weights - scaled).abs().max()
    err_unscaled = (weights - unscaled).abs().max()
    assert err_scaled < err_unscaled, (
        f"your weights look more like the UNSCALED version (err {err_unscaled:.2e}) than "
        f"the scaled one (err {err_scaled:.2e}). The division by sqrt(d_k) is missing."
    )
    assert err_scaled < 1e-6


def test_the_scaling_keeps_the_softmax_from_saturating():
    """The point of the exercise: with a large d_k and no scaling, attention collapses."""
    torch.manual_seed(2)
    q, k, v = torch.randn(1, 16, 256), torch.randn(1, 16, 256), torch.randn(1, 16, 256)
    _, weights = ex.single_head_attention(q, k, v)

    # Mean entropy of the distributions. Saturated = entropy close to 0.
    entropy = float(-(weights * torch.log(weights + 1e-12)).sum(-1).mean())
    max_entropy = math.log(16)
    assert entropy > 0.3 * max_entropy, (
        f"the mean entropy is {entropy:.3f} out of a maximum of {max_entropy:.3f}: "
        "the softmax is saturated, which happens when you do not divide by sqrt(d_k)"
    )


def test_it_works_with_q_and_k_of_different_lengths():
    """Not the course's case, but the formula is general."""
    torch.manual_seed(3)
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 7, 8)
    v = torch.randn(2, 7, 16)
    out, weights = ex.single_head_attention(q, k, v)
    assert out.shape == (2, 3, 16) and weights.shape == (2, 3, 7)


def test_with_a_single_token_attention_is_the_identity():
    """With T=1 it can only look at itself, so the output is v."""
    q, k, v = torch.randn(1, 1, 8), torch.randn(1, 1, 8), torch.randn(1, 1, 8)
    out, weights = ex.single_head_attention(q, k, v, ref.causal_mask(1))
    assert_close(out, v, what="the output with a single token")
    assert_close(weights, torch.ones(1, 1, 1), what="the single weight")


# --------------------------------------------------- exercise 3: MultiHeadAttention


def test_mha_has_the_expected_architecture():
    mine = ex.MultiHeadAttention(32, 4)
    copy_parameters(ref.MultiHeadAttention(32, 4), mine)


def test_mha_returns_the_right_shape():
    mha = ex.MultiHeadAttention(32, 4)
    out = mha(torch.randn(2, 7, 32))
    assert out.shape == (2, 7, 32)


def test_mha_validates_that_d_model_is_divisible():
    with pytest.raises(ValueError):
        ex.MultiHeadAttention(d_model=32, n_heads=5)


def test_mha_matches_the_reference():
    torch.manual_seed(0)
    mine, theirs = ex.MultiHeadAttention(32, 4), ref.MultiHeadAttention(32, 4)
    copy_parameters(theirs, mine)
    mine.eval()
    theirs.eval()
    x = torch.randn(2, 7, 32)
    assert_close(mine(x), theirs(x), what="the multi-head output")


def test_mha_matches_nn_multiheadattention():
    """The external oracle: PyTorch's implementation.

    `nn.MultiheadAttention` stores the three projections concatenated in a single
    `in_proj_weight` of (3*d, d), so the weights have to be transferred by hand.
    And its `attn_mask` uses the OPPOSITE convention: True means FORBIDDEN.
    """
    torch.manual_seed(0)
    d_model, n_heads, seq = 32, 4, 6

    mine = ex.MultiHeadAttention(d_model, n_heads, bias=True)
    theirs = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=True)

    with torch.no_grad():
        theirs.in_proj_weight.copy_(
            torch.cat([mine.q_proj.weight, mine.k_proj.weight, mine.v_proj.weight])
        )
        theirs.in_proj_bias.copy_(
            torch.cat([mine.q_proj.bias, mine.k_proj.bias, mine.v_proj.bias])
        )
        theirs.out_proj.weight.copy_(mine.out_proj.weight)
        theirs.out_proj.bias.copy_(mine.out_proj.bias)

    mine.eval()
    theirs.eval()
    x = torch.randn(2, seq, d_model)
    mask = ref.causal_mask(seq)

    my_out = mine(x, mask=mask)
    their_out, _ = theirs(x, x, x, attn_mask=~mask, need_weights=False)
    assert_close(my_out, their_out, atol=1e-5, what="the output against PyTorch")


def test_mha_returns_weights_if_you_ask_for_them():
    mha = ex.MultiHeadAttention(32, 4)
    out, weights = mha(torch.randn(2, 7, 32), return_weights=True)
    assert out.shape == (2, 7, 32)
    assert weights.shape == (2, 4, 7, 7), "the weights are (B, n_heads, T, T), one per head"


def test_mha_applies_the_causal_mask_by_default():
    mha = ex.MultiHeadAttention(32, 4)
    mha.eval()
    _, weights = mha(torch.randn(1, 6, 32), return_weights=True)
    assert torch.allclose(weights[0, 0].triu(1), torch.zeros(6, 6), atol=1e-9)


def test_mha_does_not_mix_information_between_heads():
    """Each head has to operate in its own head_dim-dimensional subspace."""
    torch.manual_seed(0)
    mha = ex.MultiHeadAttention(32, 4)
    mha.eval()
    _, weights = mha(torch.randn(1, 5, 32), return_weights=True)
    # If the heads were identical, that would be a sign the split is wrong.
    assert not torch.allclose(weights[0, 0], weights[0, 1], atol=1e-4), (
        "every head gives the same pattern: check the view/transpose of the split"
    )


def test_mha_has_the_expected_parameters():
    """4 * d_model^2 without biases, which is what the final model's count assumes."""
    mha = ex.MultiHeadAttention(320, 8, bias=False)
    assert sum(p.numel() for p in mha.parameters()) == 4 * 320 * 320


def test_mha_accepts_rope_and_changes_the_result():
    """With cos/sin, q and k get rotated and the output has to be different."""
    torch.manual_seed(0)
    mha = ex.MultiHeadAttention(32, 4)
    mha.eval()
    x = torch.randn(2, 6, 32)
    cos, sin = ref.rope_frequencies(8, 16)

    without_rope = mha(x)
    with_rope = mha(x, cos=cos, sin=sin)
    assert not torch.allclose(without_rope, with_rope, atol=1e-5), "RoPE is not being applied"

    expected = ref.MultiHeadAttention(32, 4)
    copy_parameters(mha, expected)
    expected.eval()
    assert_close(with_rope, expected(x, cos=cos, sin=sin), what="the output with RoPE")


def test_mha_with_sdpa_gives_the_same_result():
    """Module 12's fused kernel cannot change the output, only the speed."""
    torch.manual_seed(0)
    mha = ex.MultiHeadAttention(32, 4)
    mha.eval()
    x = torch.randn(2, 7, 32)

    explicit = mha(x)
    mha.use_sdpa = True
    fused = mha(x)
    assert_close(fused, explicit, atol=1e-5, what="the output with SDPA")
