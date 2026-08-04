"""Tests for module 07. Run them with `llmfs check 07`.

The external oracle is PyTorch's `F.layer_norm`.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import llmfs.reference as ref
from llmfs.testing import assert_close, copy_parameters, load_exercises

ex = load_exercises(__file__)


# -------------------------------------------------------------- exercise 1: layer_norm


def test_the_theory_example():
    x = torch.tensor([[2.0, 8.0, 4.0, 6.0]])
    expected = torch.tensor([[-1.3416, 1.3416, -0.4472, 0.4472]])
    assert_close(ex.layer_norm(x), expected, atol=1e-3, what="the THEORY.md example")


def test_it_matches_f_layer_norm_without_parameters():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)
    assert_close(ex.layer_norm(x), F.layer_norm(x, (32,)), what="layer_norm without affine")


def test_it_matches_f_layer_norm_with_weight_and_bias():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)
    w, b = torch.randn(32), torch.randn(32)
    assert_close(ex.layer_norm(x, w, b), F.layer_norm(x, (32,), w, b), what="affine layer_norm")


def test_it_uses_the_population_variance_and_not_the_sample_one():
    """`torch.var` divides by (n-1) by default. LayerNorm divides by n."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    correct = F.layer_norm(x, (4,))
    with_unbiased = (x - x.mean()) / torch.sqrt(x.var(unbiased=True) + 1e-5)

    mine = ex.layer_norm(x)
    err_right = (mine - correct).abs().max()
    err_wrong = (mine - with_unbiased).abs().max()
    assert err_right < err_wrong, (
        "your result looks more like the sample variance (n-1) than the population one (n). "
        "Pass unbiased=False to torch.var."
    )
    assert err_right < 1e-5


def test_the_output_has_mean_zero_and_variance_one():
    torch.manual_seed(0)
    y = ex.layer_norm(torch.randn(4, 8, 64) * 10 + 5)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(4, 8), atol=1e-5)
    assert torch.allclose(y.var(dim=-1, unbiased=False), torch.ones(4, 8), atol=1e-3)


def test_it_normalizes_each_token_separately():
    """Two tokens at very different scales must come out equally normalized."""
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [100.0, 200.0, 300.0, 400.0]]])
    y = ex.layer_norm(x)
    assert_close(y[0, 0], y[0, 1], atol=1e-4, what="the two normalized tokens")


def test_it_does_not_depend_on_the_batch_size():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16)
    alone = ex.layer_norm(x)
    batched = ex.layer_norm(x.repeat(8, 1, 1))
    assert_close(batched[0], alone[0], what="the result with batch 1 and with batch 8")


def test_the_epsilon_avoids_dividing_by_zero():
    """A constant vector has variance 0."""
    y = ex.layer_norm(torch.full((1, 8), 3.0))
    assert torch.isfinite(y).all(), "there is inf or nan: the eps in the denominator is missing"


def test_layer_norm_matches_the_reference():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    w, b = torch.randn(16), torch.randn(16)
    assert_close(ex.layer_norm(x, w, b), ref.layer_norm(x, w, b), what="layer_norm")


# ----------------------------------------------------------------- exercise 2: RMSNorm


def test_rmsnorm_reproduces_the_theory_example():
    norm = ex.RMSNorm(4)
    x = torch.tensor([[2.0, 8.0, 4.0, 6.0]])
    expected = torch.tensor([[0.3651, 1.4606, 0.7303, 1.0954]])
    assert_close(norm(x), expected, atol=1e-3, what="the THEORY.md example")


def test_rmsnorm_has_the_expected_architecture():
    copy_parameters(ref.RMSNorm(32), ex.RMSNorm(32))


def test_rmsnorm_starts_with_weights_at_one():
    """At initialization it has to be pure normalization, scaling nothing."""
    assert torch.allclose(ex.RMSNorm(16).weight, torch.ones(16))


def test_rmsnorm_does_not_subtract_the_mean():
    """The difference from LayerNorm: if it subtracted the mean, the output would sum to 0."""
    x = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    y = ex.RMSNorm(4)(x)
    assert not torch.allclose(y.mean(), torch.zeros(1), atol=1e-3), (
        "the output has mean 0: you are subtracting the mean, and RMSNorm does not"
    )


def test_rmsnorm_matches_the_formula():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)
    norm = ex.RMSNorm(32, eps=1e-6)
    manual = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    assert_close(norm(x), manual, atol=1e-4, what="RMSNorm against the formula")


def test_rmsnorm_matches_the_reference():
    torch.manual_seed(0)
    mine, theirs = ex.RMSNorm(32), ref.RMSNorm(32)
    torch.nn.init.normal_(theirs.weight)
    copy_parameters(theirs, mine)
    x = torch.randn(2, 5, 32)
    assert_close(mine(x), theirs(x), atol=1e-5, what="RMSNorm")


def test_rmsnorm_has_only_dim_parameters():
    """Half of LayerNorm, which has scale and bias. It counts towards the model total."""
    assert sum(p.numel() for p in ex.RMSNorm(320).parameters()) == 320


def test_rmsnorm_computes_in_fp32_and_does_not_overflow():
    """With large fp16 activations, x^2 goes out of range if you do not promote."""
    x = torch.full((1, 8), 300.0, dtype=torch.float16)
    y = ex.RMSNorm(8)(x)
    assert torch.isfinite(y).all(), (
        "there is inf: 300^2 = 90,000 does not fit in fp16. Do the computation in float() "
        "and return to the original dtype with .type_as(x)"
    )


def test_rmsnorm_returns_fp32_even_if_the_input_is_fp16():
    """Deliberate behaviour, and worth understanding before it surprises you.

    The `.type_as(x)` returns the normalization to fp16, but afterwards it is multiplied by
    `self.weight`, which is an fp32 parameter. PyTorch's type promotion makes the result come
    out in fp32.

    It is not a bug: it is what Llama's implementation does and it is what you want. Under
    autocast, the model's weights stay in fp32 and the following operations convert what they
    need. Leaving a normalization's output at the higher precision is free and gives
    numerical headroom.
    """
    x = torch.randn(2, 8, dtype=torch.float16)
    out = ex.RMSNorm(8)(x)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_rmsnorm_the_weight_scales_the_output():
    norm = ex.RMSNorm(4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    base = norm(x).clone()
    with torch.no_grad():
        norm.weight.mul_(2.0)
    assert_close(norm(x), base * 2, atol=1e-4, what="the output with the weight doubled")


# -------------------------------------------------------- exercise 3: prenorm_residual


def test_prenorm_is_x_plus_fn_of_norm_of_x():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    norm = ref.RMSNorm(8)
    fn = torch.nn.Linear(8, 8)
    assert_close(
        ex.prenorm_residual(x, fn, norm), x + fn(norm(x)), what="the pre-norm formula"
    )


def test_prenorm_is_not_postnorm():
    """The classic mistake: norm(x + fn(x)) instead of x + fn(norm(x))."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    norm = ref.RMSNorm(8)
    fn = torch.nn.Linear(8, 8)
    assert not torch.allclose(ex.prenorm_residual(x, fn, norm), norm(x + fn(x)), atol=1e-4), (
        "your result matches post-norm: you put the normalization outside the branch"
    )


def test_prenorm_matches_the_reference():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    norm, fn = ref.RMSNorm(8), torch.nn.Linear(8, 8)
    assert_close(ex.prenorm_residual(x, fn, norm), ref.prenorm_residual(x, fn, norm))


def test_if_the_block_does_nothing_the_input_passes_through_untouched():
    """The residual's key property: the path x -> x is clear."""
    x = torch.randn(2, 4, 8)
    assert_close(
        ex.prenorm_residual(x, lambda z: torch.zeros_like(z), ref.RMSNorm(8)),
        x,
        what="the output with a null block",
    )


def test_the_gradient_arrives_intact_through_the_residual():
    """The 1 in d(x + f(x))/dx = 1 + df/dx. It is the whole reason the residual exists."""
    x = torch.randn(2, 4, 8, requires_grad=True)
    # A block that completely nulls its branch's gradient.
    out = ex.prenorm_residual(x, lambda z: z.detach() * 0, ref.RMSNorm(8))
    out.sum().backward()
    assert_close(x.grad, torch.ones_like(x), what="the gradient along the residual shortcut")


def test_stacking_many_blocks_does_not_vanish_the_gradient():
    """With 40 layers and no residual the gradient dies; with a residual, it does not."""
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, requires_grad=True)
    norm = ref.RMSNorm(16)
    layer = torch.nn.Linear(16, 16)
    with torch.no_grad():
        layer.weight.mul_(0.1)  # a layer that shrinks its output a lot

    h = x
    for _ in range(40):
        h = ex.prenorm_residual(h, layer, norm)
    h.sum().backward()

    norm_value = float(x.grad.norm())
    assert norm_value > 0.1, (
        f"the gradient norm at the input is {norm_value:.2e}. With residuals it should not "
        "vanish even with 40 layers."
    )
    assert math.isfinite(norm_value)
