"""Tests for module 08. Run them with `llmfs check 08`.

The external oracle is `F.gelu(x, approximate="tanh")`.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import llmfs.reference as ref
from llmfs.testing import assert_close, assert_scalar_close, copy_parameters, load_exercises

ex = load_exercises(__file__)


# --------------------------------------------------------------------- exercise 1: gelu


def test_it_matches_the_approximate_f_gelu():
    x = torch.linspace(-6, 6, 500)
    assert_close(ex.gelu(x), F.gelu(x, approximate="tanh"), what="GELU")


def test_the_values_from_the_statement():
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    expected = torch.tensor([-0.0036, -0.1588, 0.0, 0.8412, 2.9964])
    assert_close(ex.gelu(x), expected, atol=1e-3, what="the THEORY.md values")


def test_gelu_at_zero_is_zero():
    assert_scalar_close(ex.gelu(torch.zeros(1))[0], 0.0, atol=1e-9, what="GELU(0)")


def test_it_is_almost_the_identity_for_large_values():
    x = torch.tensor([10.0, 20.0])
    assert_close(ex.gelu(x), x, rtol=1e-4, what="GELU of very positive values")


def test_it_almost_completely_cancels_very_negative_values():
    assert bool((ex.gelu(torch.tensor([-10.0, -20.0])).abs() < 1e-6).all())


def test_the_derivative_is_not_zero_in_the_negative_zone():
    """The advantage over ReLU: a neuron in the negative zone can recover."""
    x = torch.tensor([-1.0], requires_grad=True)
    ex.gelu(x).backward()
    assert abs(float(x.grad)) > 0.01, (
        "the derivative at x=-1 is practically zero: that is ReLU, not GELU"
    )


def test_it_is_not_relu():
    x = torch.tensor([-2.0, -1.0, -0.5])
    assert not torch.allclose(ex.gelu(x), torch.zeros(3), atol=1e-3), (
        "you cancel the negatives entirely: you have implemented ReLU"
    )


def test_it_preserves_the_shape():
    for shape in [(10,), (4, 8), (2, 3, 16)]:
        assert ex.gelu(torch.randn(*shape)).shape == shape


def test_gelu_matches_the_reference():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)
    assert_close(ex.gelu(x), ref.gelu(x), what="GELU")


# -------------------------------------------------------- exercise 2: swiglu_hidden_dim


def test_it_produces_the_896_of_the_final_config():
    assert ex.swiglu_hidden_dim(320) == 896


def test_it_produces_the_384_of_the_toy_config():
    assert ex.swiglu_hidden_dim(128) == 384


@pytest.mark.parametrize("d_model", [64, 128, 256, 320, 512, 768, 1024, 4096])
def test_d_ff_matches_the_reference(d_model):
    assert ex.swiglu_hidden_dim(d_model) == ref.swiglu_hidden_dim(d_model)


def test_the_result_is_a_multiple_of_64():
    for d in (64, 100, 320, 777, 1024):
        assert ex.swiglu_hidden_dim(d) % 64 == 0


def test_it_rounds_up_and_not_down():
    """int(2*4*320/3) = 853, and the nearest multiple of 64 above is 896, not 832."""
    assert ex.swiglu_hidden_dim(320) == 896
    assert ex.swiglu_hidden_dim(320) != 832


def test_it_respects_another_multiple_of():
    value = ex.swiglu_hidden_dim(320, multiple_of=256)
    assert value % 256 == 0 and value >= 853


def test_a_value_already_a_multiple_does_not_get_inflated():
    """96 -> int(2*384/3) = 256, which is already a multiple of 64: it must stay at 256."""
    assert ex.swiglu_hidden_dim(96) == 256


def test_the_extra_multiplier_gets_applied():
    """Llama uses it to tune the FFN size by hand."""
    normal = ex.swiglu_hidden_dim(4096)
    with_factor = ex.swiglu_hidden_dim(4096, ffn_dim_multiplier=1.3)
    assert with_factor > normal


def test_swiglu_spends_the_same_parameters_as_a_classic_ffn():
    """The reason for the 2/3, checked with numbers."""
    d = 320
    classic = 2 * d * (4 * d)
    swiglu = 3 * d * ex.swiglu_hidden_dim(d)
    assert abs(swiglu - classic) / classic < 0.06, (
        f"SwiGLU spends {swiglu:,} and the classic one {classic:,}: they should be close"
    )


def test_it_returns_an_integer():
    assert isinstance(ex.swiglu_hidden_dim(320), int)


# ------------------------------------------------------------------ exercise 3: SwiGLU


def test_swiglu_has_the_expected_architecture():
    copy_parameters(ref.SwiGLU(64, 128), ex.SwiGLU(64, 128))


def test_swiglu_returns_the_right_shape():
    ffn = ex.SwiGLU(64, 128)
    assert ffn(torch.randn(2, 7, 64)).shape == (2, 7, 64)


def test_swiglu_matches_the_reference():
    torch.manual_seed(0)
    mine, theirs = ex.SwiGLU(64, 128), ref.SwiGLU(64, 128)
    copy_parameters(theirs, mine)
    mine.eval()
    theirs.eval()
    x = torch.randn(2, 5, 64)
    assert_close(mine(x), theirs(x), what="SwiGLU's output")


def test_swiglu_matches_the_formula():
    torch.manual_seed(0)
    ffn = ex.SwiGLU(64, 128)
    ffn.eval()
    x = torch.randn(2, 5, 64)
    manual = ffn.down_proj(F.silu(ffn.gate_proj(x)) * ffn.up_proj(x))
    assert_close(ffn(x), manual, what="SwiGLU against the formula")


def test_swiglu_applies_the_activation_to_the_gate_branch():
    """If you apply it to up_proj instead of gate_proj, it does not match the reference."""
    torch.manual_seed(0)
    ffn = ex.SwiGLU(64, 128)
    ffn.eval()
    x = torch.randn(2, 5, 64)
    swapped = ffn.down_proj(ffn.gate_proj(x) * F.silu(ffn.up_proj(x)))
    assert not torch.allclose(ffn(x), swapped, atol=1e-5), (
        "you put the activation on up_proj; it goes on gate_proj"
    )


def test_swiglu_has_three_matrices():
    ffn = ex.SwiGLU(320, 896, bias=False)
    assert sum(p.numel() for p in ffn.parameters()) == 3 * 320 * 896


def test_swiglu_reproduces_the_final_models_per_layer_parameters():
    """860,160, the number from the config's breakdown."""
    assert sum(p.numel() for p in ex.SwiGLU(320, 896, bias=False).parameters()) == 860_160


def test_swiglu_processes_each_token_separately():
    """The FFN does not mix positions: that is attention's job."""
    torch.manual_seed(0)
    ffn = ex.SwiGLU(32, 64)
    ffn.eval()
    x = torch.randn(1, 4, 32)

    whole = ffn(x)
    just_one = ffn(x[:, 2:3, :])
    assert_close(just_one[0, 0], whole[0, 2], atol=1e-5, what="token 2 in isolation")


def test_swiglu_is_nonlinear():
    """If it were linear, stacking blocks would achieve nothing."""
    torch.manual_seed(0)
    ffn = ex.SwiGLU(32, 64)
    ffn.eval()
    x = torch.randn(1, 3, 32)
    assert not torch.allclose(ffn(2 * x), 2 * ffn(x), atol=1e-3), (
        "f(2x) == 2*f(x): your module is linear, the activation is missing"
    )


def test_swiglu_has_no_biases_by_default():
    ffn = ex.SwiGLU(32, 64)
    assert ffn.gate_proj.bias is None, "bias=False by default, like the model's config"
