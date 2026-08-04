"""Module 12 tests. Run them with `llmfs check 12`."""

from __future__ import annotations

import pytest

import llmfs.reference as ref
from llmfs.config import ModelConfig
from llmfs.testing import assert_scalar_close, load_exercises

ex = load_exercises(__file__)


# ------------------------------------------------- exercise 1: model_flops_per_token


def test_the_total_is_the_sum_of_the_parts():
    f = ex.model_flops_per_token(ModelConfig())
    assert f["total"] == f["matmul"] + f["attention"]


def test_it_reproduces_the_number_from_module_01():
    """65.4 MFLOPs per token. The same computation, now broken down."""
    assert ex.model_flops_per_token(ModelConfig())["total"] == 65_372_160


def test_the_breakdown_of_the_final_model():
    f = ex.model_flops_per_token(ModelConfig())
    assert f["attention"] / f["total"] == pytest.approx(0.18, abs=0.02), (
        "with context 512 attention should be ~18% of the total"
    )


def test_attention_grows_with_the_context_and_matmul_does_not():
    short = ex.model_flops_per_token(ModelConfig(context_length=512))
    long = ex.model_flops_per_token(ModelConfig(context_length=1024))
    assert long["matmul"] == short["matmul"], "the matmul term does not depend on the context"
    assert long["attention"] == 2 * short["attention"], "attention grows linearly with T"


def test_with_a_very_long_context_attention_dominates():
    f = ex.model_flops_per_token(ModelConfig(context_length=4096))
    assert f["attention"] > f["matmul"], (
        "with context 4096 attention should weigh more than the matmuls"
    )


def test_the_backward_triples_the_cost():
    fwd_only = ex.model_flops_per_token(ModelConfig(), include_backward=False)
    with_bwd = ex.model_flops_per_token(ModelConfig(), include_backward=True)
    assert with_bwd["total"] == 3 * fwd_only["total"]


def test_swiglu_costs_more_than_a_classic_ffn():
    swiglu = ex.model_flops_per_token(ModelConfig(activation="swiglu"))
    classic = ex.model_flops_per_token(ModelConfig(activation="gelu"))
    assert swiglu["matmul"] > classic["matmul"]


def test_the_final_projection_counts_even_with_tying():
    """Tying the weights saves memory, not compute: the matmul happens all the same."""
    tied = ex.model_flops_per_token(ModelConfig(tie_embeddings=True))
    untied = ex.model_flops_per_token(ModelConfig(tie_embeddings=False))
    assert tied["total"] == untied["total"]


@pytest.mark.parametrize(
    "cfg",
    [
        ModelConfig(),
        ModelConfig(n_layers=1),
        ModelConfig(context_length=128),
        ModelConfig(activation="gelu"),
        ModelConfig(vocab_size=128, d_model=64, n_heads=8, d_ff=192),
    ],
)
def test_the_flops_match_the_reference(cfg):
    assert ex.model_flops_per_token(cfg) == ref.model_flops_per_token(cfg)


# ---------------------------------------------------------- exercise 2: compute_mfu


def test_the_formula_is_the_expected_one():
    assert_scalar_close(ex.compute_mfu(1000, 1_000_000, 10.0), 1e9 / 1e13, rtol=1e-9, what="the MFU")


def test_mfu_of_one_when_everything_is_used():
    """If tokens/s * flops/token == the exact peak, the MFU is 1."""
    assert_scalar_close(ex.compute_mfu(1e6, 1e7, 10.0), 1.0, rtol=1e-9, what="the maximum MFU")


def test_it_scales_linearly_with_the_tokens_per_second():
    a = ex.compute_mfu(1000, 65_372_160, 51.6)
    b = ex.compute_mfu(2000, 65_372_160, 51.6)
    assert_scalar_close(b, 2 * a, what="twice the tokens/s")


def test_a_non_positive_peak_is_an_error():
    for bad in (0.0, -10.0):
        with pytest.raises(ValueError):
            ex.compute_mfu(1000, 1_000_000, bad)


def test_the_case_of_the_final_model_on_the_2060():
    """3,000 tok/s on an RTX 2060 with our model."""
    mfu = ex.compute_mfu(3000, 65_372_160, 51.6)
    assert 0.001 < mfu < 0.02, f"MFU {mfu:.4f} outside the expected range"


def test_the_mfu_matches_the_reference():
    assert_scalar_close(
        ex.compute_mfu(3000, 65_372_160, 51.6),
        ref.compute_mfu(3000, 65_372_160, 51.6),
        what="the MFU",
    )


# ------------------------------------------ exercise 3: chinchilla_optimal_allocation


def test_it_reproduces_chinchillas_size():
    """THE test of the module. With Chinchilla's real budget (5.76e23 FLOPs), the formula
    has to give ~70 billion parameters, which is what the model had."""
    a = ex.chinchilla_optimal_allocation(5.76e23)
    assert 6.5e10 < a["params"] < 7.5e10, (
        f"the formula gives {a['params']:.2e} parameters and Chinchilla had 7.0e10"
    )


def test_it_reproduces_chinchillas_tokens():
    """1.4 trillion tokens."""
    a = ex.chinchilla_optimal_allocation(5.76e23)
    assert 1.2e12 < a["tokens"] < 1.6e12


def test_the_ratio_is_twenty_tokens_per_parameter():
    a = ex.chinchilla_optimal_allocation(1e21)
    assert_scalar_close(a["tokens"] / a["params"], 20.0, rtol=1e-6, what="tokens per parameter")


def test_the_budget_matches_6nd():
    """C = 6ND has to give back the starting budget."""
    C = 1e22
    a = ex.chinchilla_optimal_allocation(C)
    assert_scalar_close(6 * a["params"] * a["tokens"], C, rtol=1e-6, what="C = 6ND")


def test_quadrupling_the_compute_doubles_both():
    """N and D grow with the square root of the budget."""
    a = ex.chinchilla_optimal_allocation(1e21)
    b = ex.chinchilla_optimal_allocation(4e21)
    assert_scalar_close(b["params"] / a["params"], 2.0, rtol=1e-6, what="the factor in N")
    assert_scalar_close(b["tokens"] / a["tokens"], 2.0, rtol=1e-6, what="the factor in D")


def test_gpt3_was_undertrained():
    """With its budget, GPT-3 should have had far fewer parameters."""
    C_gpt3 = 6 * 175e9 * 300e9
    a = ex.chinchilla_optimal_allocation(C_gpt3)
    assert a["params"] < 175e9 / 2, (
        f"with GPT-3's budget the optimum was {a['params']:.2e} parameters, not 1.75e11"
    )


def test_another_value_of_tokens_per_param():
    """Llama-3 uses ~1800, far above Chinchilla."""
    a = ex.chinchilla_optimal_allocation(1e22, tokens_per_param=1800.0)
    assert_scalar_close(a["tokens"] / a["params"], 1800.0, rtol=1e-6)
    assert a["params"] < ex.chinchilla_optimal_allocation(1e22)["params"]


def test_a_non_positive_budget_is_an_error():
    for bad in (0.0, -1e20):
        with pytest.raises(ValueError):
            ex.chinchilla_optimal_allocation(bad)


def test_it_has_all_the_keys():
    a = ex.chinchilla_optimal_allocation(1e21)
    for key in ("params", "tokens", "tokens_per_param", "compute"):
        assert key in a


def test_chinchilla_matches_the_reference():
    mine = ex.chinchilla_optimal_allocation(5.76e23)
    theirs = ref.chinchilla_optimal_allocation(5.76e23)
    for key in theirs:
        assert_scalar_close(mine[key], theirs[key], what=key)
