"""Module 15 tests. Run them with `llmfs check 15`."""

from __future__ import annotations

import math

import pytest

import llmfs.reference as ref
from llmfs.testing import assert_scalar_close, load_exercises

ex = load_exercises(__file__)


# ------------------------------------------------------ exercise 1: perplexity_from_loss


def test_it_is_the_exponential_of_the_loss():
    assert_scalar_close(ex.perplexity_from_loss(1.6), math.exp(1.6), what="the perplexity")


def test_the_untrained_model_has_a_perplexity_equal_to_the_vocabulary():
    """With loss ln(V), the perplexity is exactly V. It is the useful check."""
    for vocab in (65, 4096, 50257):
        assert_scalar_close(
            ex.perplexity_from_loss(math.log(vocab)),
            vocab,
            rtol=1e-9,
            what=f"the perplexity with loss ln({vocab})",
        )


def test_a_loss_of_zero_gives_a_perplexity_of_one():
    assert_scalar_close(ex.perplexity_from_loss(0.0), 1.0, what="the perfect model")


def test_a_non_finite_loss_gives_infinity_without_blowing_up():
    """`math.exp(inf)` raises OverflowError: it has to be caught."""
    for bad in (float("inf"), float("nan")):
        assert ex.perplexity_from_loss(bad) == float("inf")


def test_it_is_monotonically_increasing():
    values = [ex.perplexity_from_loss(x) for x in (0.5, 1.0, 2.0, 4.0)]
    assert all(a < b for a, b in zip(values, values[1:]))


def test_the_perplexity_matches_the_reference():
    for p in (0.0, 1.6, 8.317):
        assert_scalar_close(ex.perplexity_from_loss(p), ref.perplexity_from_loss(p))


# ------------------------------------------------------------ exercise 2: bits_per_byte


def test_the_formula_is_the_expected_one():
    assert_scalar_close(
        ex.bits_per_byte(1000.0, 200, 500), 1000.0 / math.log(2) / 500, what="bits per byte"
    )


def test_it_converts_nats_into_bits():
    """One nat is 1/ln(2) = 1.4427 bits."""
    assert_scalar_close(
        ex.bits_per_byte(math.log(2), 1, 1), 1.0, rtol=1e-9, what="ln(2) nats over 1 byte"
    )


def test_more_bytes_for_the_same_loss_gives_fewer_bits_per_byte():
    """This is what makes the metric comparable across tokenizers."""
    assert ex.bits_per_byte(1000.0, 200, 1000) < ex.bits_per_byte(1000.0, 200, 500)


def test_it_does_not_depend_on_the_number_of_tokens():
    """The point of the exercise: the metric normalizes by BYTES, not by tokens."""
    a = ex.bits_per_byte(1000.0, 200, 500)
    b = ex.bits_per_byte(1000.0, 800, 500)
    assert_scalar_close(a, b, what="the result with a different number of tokens")


def test_a_realistic_value_is_in_the_expected_range():
    """Mean loss 1.2 nats/token, 3 bytes per token: ~0.58 bits/byte."""
    n_tokens, mean_loss, bytes_per_token = 1000, 1.2, 3
    bpb = ex.bits_per_byte(mean_loss * n_tokens, n_tokens, n_tokens * bytes_per_token)
    assert 0.3 < bpb < 1.5, f"{bpb:.3f} outside the range of a reasonable model"


def test_non_positive_bytes_is_an_error():
    for bad in (0, -100):
        with pytest.raises(ValueError):
            ex.bits_per_byte(1000.0, 200, bad)


def test_bits_per_byte_matches_the_reference():
    assert_scalar_close(ex.bits_per_byte(1234.5, 300, 900), ref.bits_per_byte(1234.5, 300, 900))


# --------------------------------------------------------- exercise 3: run_prompt_battery


def fake_generator(prompt: str) -> str:
    return prompt + " [generated continuation]"


def test_it_returns_one_result_per_prompt():
    results = ex.run_prompt_battery(fake_generator)
    assert len(results) == len(ref.PROMPTS_TINYSTORIES)


def test_each_result_has_the_three_keys():
    for case in ex.run_prompt_battery(fake_generator):
        assert set(case) == {"prompt", "tests", "completion"}


def test_the_completion_comes_from_the_generator():
    results = ex.run_prompt_battery(fake_generator)
    for case in results:
        assert case["completion"] == fake_generator(case["prompt"])


def test_it_keeps_the_order_and_the_labels():
    results = ex.run_prompt_battery(fake_generator)
    for case, (prompt, label) in zip(results, ref.PROMPTS_TINYSTORIES):
        assert case["prompt"] == prompt
        assert case["tests"] == label


def test_it_accepts_your_own_battery():
    mine = (("Hello", "greeting"), ("Goodbye", "farewell"))
    results = ex.run_prompt_battery(fake_generator, mine)
    assert len(results) == 2
    assert results[0]["tests"] == "greeting"


def test_the_battery_tests_different_things():
    """Six prompts with six different labels: they are not variations of the same thing."""
    labels = [label for _, label in ref.PROMPTS_TINYSTORIES]
    assert len(set(labels)) == len(labels)


def test_the_battery_matches_the_reference():
    mine = ex.run_prompt_battery(fake_generator)
    theirs = ref.run_prompt_battery(fake_generator)
    assert mine == theirs
