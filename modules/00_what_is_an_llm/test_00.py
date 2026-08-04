"""Tests for module 00. Run them with `llmfs check 00`."""

from __future__ import annotations

import random

import pytest

import llmfs.reference as ref
from llmfs.testing import assert_scalar_close, load_exercises

ex = load_exercises(__file__)

TEXT = "banana bandana cabana " * 30


# ------------------------------------------------- exercise 1: next_token_probs


def test_the_probabilities_sum_to_one():
    probs = ex.next_token_probs({"a": 40, "b": 25, "c": 20, "d": 15})
    assert_scalar_close(sum(probs.values()), 1.0, what="the sum of the probabilities")


def test_the_example_from_the_statement():
    probs = ex.next_token_probs({"b": 3, "c": 1})
    assert_scalar_close(probs["b"], 0.75, what="P(b)")
    assert_scalar_close(probs["c"], 0.25, what="P(c)")


def test_it_preserves_the_keys_and_their_order():
    """The order matters: sampling walks the dictionary and depends on it."""
    counts = {"z": 1, "a": 2, "m": 3}
    assert list(ex.next_token_probs(counts).keys()) == ["z", "a", "m"]


def test_a_single_character_has_probability_one():
    assert_scalar_close(ex.next_token_probs({"x": 7})["x"], 1.0, what="P(x) when it is the only one")


def test_probs_match_the_reference():
    counts = {"n": 40, "r": 25, " ": 20, "s": 15}
    mine, theirs = ex.next_token_probs(counts), ref.next_token_probs(counts)
    assert mine == theirs


def test_an_empty_table_is_an_explicit_error():
    with pytest.raises(ValueError):
        ex.next_token_probs({})


def test_all_zero_counts_are_also_an_error():
    with pytest.raises(ValueError):
        ex.next_token_probs({"a": 0, "b": 0})


# ------------------------------------------------ exercise 2: sample_next_token


def test_it_always_returns_a_key_of_the_distribution():
    probs = {"a": 0.5, "b": 0.3, "c": 0.2}
    rng = random.Random(0)
    for _ in range(200):
        assert ex.sample_next_token(probs, rng) in probs


def test_with_probability_one_the_same_one_always_comes_out():
    rng = random.Random(0)
    for _ in range(50):
        assert ex.sample_next_token({"only": 1.0}, rng) == "only"


def test_the_frequencies_resemble_the_probabilities():
    """Sampling 20,000 times from {a:0.7, b:0.3} should give ~70% 'a'."""
    probs = {"a": 0.7, "b": 0.3}
    rng = random.Random(1234)
    samples = [ex.sample_next_token(probs, rng) for _ in range(20_000)]
    frequency_a = samples.count("a") / len(samples)
    assert abs(frequency_a - 0.7) < 0.02, (
        f"'a' came out {frequency_a:.1%} of the time and should be around 70%. "
        "Check the roulette's running total."
    )


def test_it_respects_the_dictionary_order():
    """A small r has to land in the FIRST slice of the wheel, whatever it is."""

    class FakeRng:
        def random(self) -> float:
            return 0.01

    assert ex.sample_next_token({"first": 0.5, "second": 0.5}, FakeRng()) == "first"


def test_an_r_close_to_one_lands_on_the_last():
    class FakeRng:
        def random(self) -> float:
            return 0.999999999

    assert ex.sample_next_token({"a": 0.5, "b": 0.5}, FakeRng()) == "b"


def test_it_never_returns_none_even_if_the_probabilities_do_not_sum_exactly():
    """The classic rounding error: 0.1 * 10 does not give 1.0 in floating point."""

    class FakeRng:
        def random(self) -> float:
            return 0.9999999999

    probs = {chr(97 + i): 0.1 for i in range(10)}
    assert ex.sample_next_token(probs, FakeRng()) is not None


def test_the_same_seed_gives_the_same_sequence_as_the_reference():
    probs = {"n": 0.4, "r": 0.25, " ": 0.2, "s": 0.15}
    mine = [ex.sample_next_token(probs, random.Random(7)) for _ in range(1)]
    theirs = [ref.sample_next_token(probs, random.Random(7)) for _ in range(1)]
    assert mine == theirs

    rng_mine, rng_theirs = random.Random(99), random.Random(99)
    assert [ex.sample_next_token(probs, rng_mine) for _ in range(50)] == [
        ref.sample_next_token(probs, rng_theirs) for _ in range(50)
    ]


# --------------------------------------------------- exercise 3: generate_naive


def test_it_returns_the_requested_length():
    table = ref.build_count_table(TEXT)
    out = ex.generate_naive(table, "b", length=50, rng=random.Random(0))
    assert len(out) == 50


def test_it_starts_with_the_given_context():
    table = ref.build_count_table(TEXT, context_size=2)
    out = ex.generate_naive(table, "ba", length=30, rng=random.Random(0))
    assert out.startswith("ba")


def test_the_length_includes_start():
    """start='ba' and length=5 returns 5 characters, not 7."""
    table = ref.build_count_table(TEXT, context_size=2)
    assert len(ex.generate_naive(table, "ba", length=5, rng=random.Random(0))) == 5


def test_the_length_of_start_has_to_match_the_tables():
    """The classic trap: a context-1 table and a 2-character start.

    The table has single-character keys ('a', 'b'...), so looking up 'ba' finds nothing and
    generation stops on the first step. It is not a bug: it is the documented behaviour for
    an unknown context. But it is worth having seen it once.
    """
    table_of_1 = ref.build_count_table(TEXT, context_size=1)
    assert ex.generate_naive(table_of_1, "ba", length=50, rng=random.Random(0)) == "ba"


def test_it_only_generates_characters_that_exist_in_the_text():
    table = ref.build_count_table(TEXT)
    out = ex.generate_naive(table, "b", length=300, rng=random.Random(3))
    assert set(out) <= set(TEXT)


def test_it_stops_if_the_context_is_unknown():
    """With a table that only knows 'a'->'b', after the 'b' there is nowhere to go."""
    table = {"a": {"b": 1}}
    out = ex.generate_naive(table, "a", length=100, rng=random.Random(0))
    assert out == "ab", f"it should stop on reaching an unknown context, it gave {out!r}"


def test_the_generated_text_is_identical_to_the_references():
    """Same text, same seed, exactly the same result."""
    table = ref.build_count_table(TEXT, context_size=2)
    mine = ex.generate_naive(table, "ba", length=200, rng=random.Random(42))
    theirs = ref.generate_naive(table, "ba", length=200, rng=random.Random(42))
    assert mine == theirs


def test_it_works_with_contexts_longer_than_one_character():
    table = ref.build_count_table(TEXT, context_size=3)
    out = ex.generate_naive(table, "ban", length=60, rng=random.Random(5))
    assert len(out) == 60 and out.startswith("ban")


def test_with_a_longer_context_the_text_resembles_the_original_more():
    """The module's pedagogical point: more context, better imitation.

    It is measured as the fraction of generated WORDS that exist in the corpus. Counting
    trigrams would not do: with context >= 2 it would come out 100% by construction, because
    every generated trigram is context + next, and that was already in the table.
    """
    real_words = set(TEXT.split())

    def realism(context_size: int) -> float:
        table = ref.build_count_table(TEXT, context_size=context_size)
        start = TEXT[:context_size]
        out = ex.generate_naive(table, start, length=600, rng=random.Random(11))
        generated = out.split()
        return sum(w in real_words for w in generated) / max(1, len(generated))

    with_little, with_lots = realism(1), realism(4)
    assert with_lots > with_little, (
        f"with a context of 4, {with_lots:.0%} of the words are real and with a context "
        f"of 1 it is {with_little:.0%}. Looking further back has to help."
    )
