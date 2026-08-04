"""Module 17 tests. Run them with `llmfs check 17`. They are the last ones of the course."""

from __future__ import annotations

import torch

import llmfs.reference as ref
from llmfs.config import ModelConfig
from llmfs.device import set_seed
from llmfs.testing import assert_close, assert_scalar_close, load_exercises

ex = load_exercises(__file__)


# --------------------------------------------------- exercise 1: quantize_int8_symmetric


def test_the_output_dtype_is_int8():
    q, _ = ex.quantize_int8_symmetric(torch.randn(4, 8))
    assert q.dtype == torch.int8


def test_the_values_fit_in_int8():
    q, _ = ex.quantize_int8_symmetric(torch.randn(16, 16) * 100)
    assert int(q.max()) <= 127 and int(q.min()) >= -127


def test_the_example_from_the_docstring():
    w = torch.tensor([[0.12, -0.45, 0.03, 0.28]])
    q, scale = ex.quantize_int8_symmetric(w)
    assert_scalar_close(float(scale), 0.45 / 127, rtol=1e-4, what="the scale")
    assert q[0].tolist() == [34, -127, 8, 79]


def test_the_absolute_maximum_maps_to_127():
    w = torch.tensor([[0.1, -0.5, 0.3]])
    q, _ = ex.quantize_int8_symmetric(w)
    assert int(q.abs().max()) == 127


def test_zero_is_represented_exactly():
    """That is why 127 is used and not 128: the range stays symmetric."""
    w = torch.tensor([[0.0, 0.5, -0.5]])
    q, scale = ex.quantize_int8_symmetric(w)
    assert int(q[0, 0]) == 0
    assert float(ex.dequantize_int8(q, scale)[0, 0]) == 0.0


def test_per_channel_gives_one_scale_per_row():
    q, scale = ex.quantize_int8_symmetric(torch.randn(5, 8), per_channel=True)
    assert scale.shape == (5, 1)


def test_per_tensor_gives_a_single_scale():
    _, scale = ex.quantize_int8_symmetric(torch.randn(5, 8), per_channel=False)
    assert scale.numel() == 1


def test_per_channel_is_more_accurate_than_per_tensor():
    """A row with large values must not drag the others along."""
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    w[0] *= 100  # one very large row

    e_channel = ex.quantization_error(w, per_channel=True)["relative_error"]
    e_tensor = ex.quantization_error(w, per_channel=False)["relative_error"]
    assert e_channel < e_tensor, (
        f"per channel {e_channel:.4f} should be better than per tensor {e_tensor:.4f}"
    )


def test_a_row_of_zeros_does_not_blow_up():
    """The scale's clamp_min avoids dividing by zero."""
    w = torch.zeros(3, 4)
    q, scale = ex.quantize_int8_symmetric(w)
    assert torch.isfinite(scale).all() and int(q.abs().max()) == 0


def test_the_quantization_matches_the_reference():
    torch.manual_seed(0)
    w = torch.randn(8, 16)
    q_mine, s_mine = ex.quantize_int8_symmetric(w)
    q_ref, s_ref = ref.quantize_int8_symmetric(w)
    assert torch.equal(q_mine, q_ref)
    assert_close(s_mine, s_ref, what="the scales")


# --------------------------------------------------------- exercise 2: dequantize_int8


def test_the_roundtrip_gets_close_to_the_original():
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    q, scale = ex.quantize_int8_symmetric(w)
    recovered = ex.dequantize_int8(q, scale)
    assert_close(recovered, w, atol=0.02, what="the roundtrip")


def test_it_returns_float():
    q, scale = ex.quantize_int8_symmetric(torch.randn(4, 8))
    assert ex.dequantize_int8(q, scale).dtype == torch.float32


def test_it_keeps_the_shape():
    w = torch.randn(6, 10)
    q, scale = ex.quantize_int8_symmetric(w)
    assert ex.dequantize_int8(q, scale).shape == w.shape


def test_the_roundtrip_is_not_exact():
    """Information is lost: that is what you pay."""
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    q, scale = ex.quantize_int8_symmetric(w)
    assert not torch.allclose(ex.dequantize_int8(q, scale), w, atol=1e-6)


def test_the_dequantization_matches_the_reference():
    torch.manual_seed(0)
    q, scale = ref.quantize_int8_symmetric(torch.randn(4, 16))
    assert_close(ex.dequantize_int8(q, scale), ref.dequantize_int8(q, scale))


# ------------------------------------------------------ exercise 3: quantization_error


def test_it_has_all_the_keys():
    e = ex.quantization_error(torch.randn(4, 8))
    for key in (
        "relative_error",
        "max_error",
        "mean_error",
        "compression",
        "original_bytes",
        "quantized_bytes",
    ):
        assert key in e, f"the key {key!r} is missing"


def test_the_compression_from_fp32_to_int8_is_4x():
    assert ex.quantization_error(torch.randn(4, 8))["compression"] == 4.0


def test_the_relative_error_of_real_weights_is_small():
    """With the weights of a trained network, int8 per channel is around 0.5-1%."""
    set_seed(0)
    w = ref.GPT(ModelConfig()).blocks[0].attn.q_proj.weight.data
    error = ex.quantization_error(w)["relative_error"]
    assert 0.001 < error < 0.03, f"relative error {error:.4f} outside the expected range"


def test_the_quantized_bytes_include_the_scales():
    """The scales take space too: counting them is the honest thing."""
    w = torch.randn(10, 20)
    e = ex.quantization_error(w, per_channel=True)
    assert e["quantized_bytes"] > w.numel(), (
        "the quantized bytes have to include the vector of scales"
    )
    assert e["original_bytes"] == w.numel() * 4


def test_the_relative_error_is_independent_of_the_scale():
    """Multiplying the weights by 1000 does not change the RELATIVE error."""
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    a = ex.quantization_error(w)["relative_error"]
    b = ex.quantization_error(w * 1000)["relative_error"]
    assert_scalar_close(a, b, rtol=1e-3, what="the relative error")


def test_the_error_of_the_whole_model_is_acceptable():
    """Over every matrix of the final model."""
    set_seed(0)
    model = ref.GPT(ModelConfig())
    errors = [
        ex.quantization_error(p.data)["relative_error"]
        for p in model.parameters()
        if p.dim() >= 2
    ]
    assert max(errors) < 0.05, f"the worst matrix has an error of {max(errors):.1%}"


def test_the_error_matches_the_reference():
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    mine, theirs = ex.quantization_error(w), ref.quantization_error(w)
    for key in theirs:
        assert_scalar_close(mine[key], theirs[key], what=key)
