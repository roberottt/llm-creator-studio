"""Module 14 tests. Run them with `llmfs check 14`.

The test that matters most is `test_the_cache_gives_exactly_the_same_output`.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import llmfs.reference as ref
from llmfs.config import ModelConfig
from llmfs.device import set_seed
from llmfs.testing import assert_close, load_exercises

ex = load_exercises(__file__)


def small_model(vocab: int = 64) -> torch.nn.Module:
    set_seed(0)
    return ref.GPT(
        ModelConfig(
            vocab_size=vocab, n_layers=2, d_model=32, n_heads=4, d_ff=96, context_length=64
        )
    ).eval()


# ------------------------------------------- exercise 1: apply_repetition_penalty


def test_positive_logits_are_divided():
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    out = ex.apply_repetition_penalty(logits, torch.tensor([[0]]), penalty=2.0)
    assert abs(float(out[0, 0]) - 1.5) < 1e-5, "a positive logit has to be DIVIDED"


def test_negative_logits_are_multiplied():
    """The detail almost everyone implements wrong."""
    logits = torch.tensor([[-3.0, 2.0, 1.0]])
    out = ex.apply_repetition_penalty(logits, torch.tensor([[0]]), penalty=2.0)
    assert abs(float(out[0, 0]) - (-6.0)) < 1e-5, (
        "a negative logit has to be MULTIPLIED. If you divide it, the token becomes MORE "
        "likely: exactly the opposite of penalizing it."
    )


def test_penalizing_always_lowers_the_probability():
    """The property that really matters, whatever the sign."""
    torch.manual_seed(0)
    logits = torch.randn(1, 20)
    generated = torch.tensor([[3, 7, 15]])
    before = F.softmax(logits, dim=-1)
    after = F.softmax(ex.apply_repetition_penalty(logits, generated, 1.5), dim=-1)
    for tok in (3, 7, 15):
        assert after[0, tok] < before[0, tok], f"token {tok} has not gone down"


def test_it_does_not_touch_the_tokens_that_have_not_come_out():
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    out = ex.apply_repetition_penalty(logits, torch.tensor([[0]]), penalty=2.0)
    assert abs(float(out[0, 1]) - 2.0) < 1e-6
    assert abs(float(out[0, 2]) - 1.0) < 1e-6


def test_a_penalty_of_one_does_nothing():
    logits = torch.randn(2, 10)
    out = ex.apply_repetition_penalty(logits, torch.tensor([[0, 1], [2, 3]]), penalty=1.0)
    assert_close(out, logits, what="the logits with penalty=1.0")


def test_it_does_not_modify_the_input():
    logits = torch.tensor([[3.0, 2.0]])
    copy = logits.clone()
    ex.apply_repetition_penalty(logits, torch.tensor([[0]]), penalty=2.0)
    assert_close(logits, copy, what="the original logits")


def test_it_works_with_a_batch():
    logits = torch.randn(3, 10)
    out = ex.apply_repetition_penalty(logits, torch.tensor([[0], [1], [2]]), 1.5)
    assert out.shape == (3, 10)


def test_the_penalty_matches_the_reference():
    torch.manual_seed(0)
    logits, gen = torch.randn(2, 20), torch.tensor([[1, 5, 9], [2, 6, 10]])
    assert_close(
        ex.apply_repetition_penalty(logits, gen, 1.2),
        ref.apply_repetition_penalty(logits, gen, 1.2),
    )


# ----------------------------------------------------------- exercise 2: top_k_filter


def test_top_k_leaves_only_the_k_largest():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, -1.0]])
    out = ex.top_k_filter(logits, 2)
    assert float(out[0, 0]) == 3.0 and float(out[0, 1]) == 2.0
    assert all(math.isinf(float(out[0, i])) for i in (2, 3, 4))


def test_top_k_keeps_the_threshold():
    """The k-th logit has to survive: use < and not <=."""
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    assert float(ex.top_k_filter(logits, 2)[0, 1]) == 2.0


def test_top_k_with_extreme_values_does_not_filter():
    logits = torch.randn(1, 10)
    assert_close(ex.top_k_filter(logits, 0), logits, what="k=0")
    assert_close(ex.top_k_filter(logits, 100), logits, what="k larger than the vocabulary")


def test_top_k_works_with_a_batch():
    logits = torch.randn(4, 50)
    out = ex.top_k_filter(logits, 10)
    for row in range(4):
        assert int(torch.isfinite(out[row]).sum()) == 10


def test_top_k_matches_the_reference():
    torch.manual_seed(0)
    logits = torch.randn(3, 100)
    assert_close(ex.top_k_filter(logits, 20), ref.top_k_filter(logits, 20))


# ----------------------------------------------------------- exercise 3: top_p_filter


def test_top_p_of_the_example_in_the_docstring():
    """probs [0.60, 0.25, 0.10, 0.03, 0.02] with p=0.9 leaves 3 candidates.

    The token that CROSSES the threshold goes in: Holtzman's definition is "the smallest set
    whose cumulative probability EXCEEDS p", and [0.60, 0.25] adds up to 0.85, which does not
    exceed 0.9. The third one is needed to reach 0.95.
    """
    probs = torch.tensor([[0.60, 0.25, 0.10, 0.03, 0.02]])
    out = ex.top_p_filter(torch.log(probs), 0.9)
    assert int(torch.isfinite(out).sum()) == 3
    # And the three that remain add up to more than 0.9
    survivors = probs[0][torch.isfinite(out[0])]
    assert float(survivors.sum()) > 0.9


def test_top_p_keeps_the_SMALLEST_set_that_exceeds_p():
    """Keeping more than needed will not do: dropping the last one, it would no longer reach p."""
    probs = torch.tensor([[0.60, 0.25, 0.10, 0.03, 0.02]])
    out = ex.top_p_filter(torch.log(probs), 0.9)
    n = int(torch.isfinite(out).sum())
    without_the_last = float(probs[0][: n - 1].sum())
    assert without_the_last <= 0.9, (
        f"with {n - 1} candidates it would already reach {without_the_last:.2f} > 0.9: "
        "there is one too many"
    )


def test_the_number_of_candidates_adapts():
    """What sets top-p apart from top-k."""
    sure = torch.log(torch.tensor([[0.9, 0.05, 0.03, 0.02]]))
    unsure = torch.log(torch.tensor([[0.25, 0.25, 0.25, 0.25]]))
    n_sure = int(torch.isfinite(ex.top_p_filter(sure, 0.9)).sum())
    n_unsure = int(torch.isfinite(ex.top_p_filter(unsure, 0.9)).sum())
    assert n_sure < n_unsure, (
        f"with the sure model it leaves {n_sure} candidates and with the unsure one "
        f"{n_unsure}: the number has to ADAPT"
    )


def test_the_most_likely_one_always_survives():
    """With p=0.5 and a token of probability 0.9, without this guard none would be left."""
    logits = torch.log(torch.tensor([[0.9, 0.05, 0.05]]))
    out = ex.top_p_filter(logits, 0.5)
    assert torch.isfinite(out[0, 0]), "the most likely token can never be filtered out"
    assert int(torch.isfinite(out).sum()) >= 1


def test_top_p_of_one_filters_nothing():
    logits = torch.randn(1, 10)
    assert_close(ex.top_p_filter(logits, 1.0), logits)


def test_a_very_low_top_p_leaves_only_one():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    assert int(torch.isfinite(ex.top_p_filter(logits, 0.01)).sum()) == 1


def test_top_p_works_with_a_batch():
    torch.manual_seed(0)
    out = ex.top_p_filter(torch.randn(4, 50), 0.9)
    assert out.shape == (4, 50)
    assert all(int(torch.isfinite(out[r]).sum()) >= 1 for r in range(4))


def test_top_p_matches_the_reference():
    torch.manual_seed(0)
    logits = torch.randn(3, 100)
    assert_close(ex.top_p_filter(logits, 0.9), ref.top_p_filter(logits, 0.9))


# ---------------------------------------------------------------- exercise 4: KVCache


def test_the_cache_starts_empty():
    cache = ex.KVCache(4)
    assert cache.seq_len == 0


def test_the_cache_accumulates_along_the_time_dimension():
    cache = ex.KVCache(2)
    k = torch.randn(1, 4, 3, 8)
    v = torch.randn(1, 4, 3, 8)
    K, V = cache.update(0, k, v)
    assert K.shape == (1, 4, 3, 8)

    k2, v2 = torch.randn(1, 4, 1, 8), torch.randn(1, 4, 1, 8)
    K, V = cache.update(0, k2, v2)
    assert K.shape == (1, 4, 4, 8), "it has to concatenate along dim=-2 (time)"
    assert_close(K[:, :, :3], k, what="what was already saved")
    assert_close(K[:, :, 3:], k2, what="the new part")


def test_the_layers_are_independent():
    cache = ex.KVCache(3)
    cache.update(0, torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))
    cache.update(1, torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4))
    assert cache.keys[0].shape[-2] == 5
    assert cache.keys[1].shape[-2] == 2
    assert cache.keys[2] is None


def test_seq_len_reflects_what_is_stored():
    cache = ex.KVCache(2)
    for i in range(1, 4):
        cache.update(0, torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))
        assert cache.seq_len == i


def test_reset_empties_it():
    cache = ex.KVCache(2)
    cache.update(0, torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))
    cache.reset()
    assert cache.seq_len == 0 and cache.keys[0] is None


def test_memory_bytes_counts_what_is_stored():
    cache = ex.KVCache(1)
    cache.update(0, torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8))
    expected = 2 * 1 * 2 * 10 * 8 * 4  # K and V, float32
    assert cache.memory_bytes() == expected


# ------------------------------------------------------- exercise 5: generate_with_cache


def test_the_cache_gives_exactly_the_same_output():
    """THE test of the module. Not similar: identical, token by token.

    With temperature=0 generation is greedy and therefore deterministic, so the two versions
    have to match exactly.
    """
    model = small_model()
    prompt = torch.randint(0, 64, (1, 8))

    without_cache = model.generate(prompt.clone(), 30, temperature=0.0)
    with_cache = ex.generate_with_cache(model, prompt.clone(), 30, temperature=0.0)

    assert torch.equal(without_cache, with_cache), (
        f"they diverge.\n  without cache: {without_cache[0, -10:].tolist()}\n"
        f"  with cache: {with_cache[0, -10:].tolist()}\n"
        "The first thing to look at is RoPE's pos_offset: when generating token 50 it is "
        "given a tensor of length 1, and it has to be rotated with the angle of position "
        "50, not of position 0."
    )


def test_the_output_shape_is_the_expected_one():
    model = small_model()
    out = ex.generate_with_cache(model, torch.randint(0, 64, (1, 5)), 20)
    assert out.shape == (1, 25)


def test_the_prompt_is_kept_intact():
    model = small_model()
    prompt = torch.randint(0, 64, (1, 6))
    out = ex.generate_with_cache(model, prompt.clone(), 10)
    assert torch.equal(out[:, :6], prompt)


def test_all_the_tokens_are_in_the_vocabulary():
    model = small_model()
    out = ex.generate_with_cache(model, torch.randint(0, 64, (1, 4)), 40, top_k=10)
    assert int(out.max()) < 64 and int(out.min()) >= 0


def test_temperature_zero_is_deterministic():
    model = small_model()
    prompt = torch.randint(0, 64, (1, 5))
    a = ex.generate_with_cache(model, prompt.clone(), 15, temperature=0.0)
    b = ex.generate_with_cache(model, prompt.clone(), 15, temperature=0.0)
    assert torch.equal(a, b)


def test_the_filters_are_applied():
    """With top_k=1 generation is deterministic even if the temperature is not zero."""
    model = small_model()
    prompt = torch.randint(0, 64, (1, 5))
    a = ex.generate_with_cache(model, prompt.clone(), 15, temperature=1.0, top_k=1)
    b = ex.generate_with_cache(model, prompt.clone(), 15, temperature=1.0, top_k=1)
    assert torch.equal(a, b), "with top_k=1 there is only one candidate: it has to be deterministic"


def test_it_stops_when_it_finds_the_end_token():
    model = small_model()
    prompt = torch.randint(0, 64, (1, 4))
    # The eos is forced to the token that would come out anyway with greedy
    normal = ex.generate_with_cache(model, prompt.clone(), 20, temperature=0.0)
    eos = int(normal[0, 4])
    with_eos = ex.generate_with_cache(model, prompt.clone(), 20, temperature=0.0, eos_token=eos)
    assert with_eos.shape[1] < normal.shape[1], "it should have stopped earlier"


def test_it_works_with_top_p():
    model = small_model()
    out = ex.generate_with_cache(
        model, torch.randint(0, 64, (1, 4)), 20, temperature=0.8, top_p=0.9
    )
    assert out.shape == (1, 24)


def test_it_stops_when_it_reaches_the_context_limit():
    """Cropping with a cache would require remapping RoPE's positions: it just stops."""
    model = small_model()  # context_length=64
    prompt = torch.randint(0, 64, (1, 50))
    out = ex.generate_with_cache(model, prompt, 100, temperature=0.0)
    assert out.shape[1] <= 64, f"it generated {out.shape[1]} tokens with a context of 64"


def test_a_prompt_that_already_fills_the_context_is_an_error():
    model = small_model()
    with pytest.raises(ValueError):
        ex.generate_with_cache(model, torch.randint(0, 64, (1, 70)), 10)


def test_the_generation_matches_the_reference():
    model = small_model()
    prompt = torch.randint(0, 64, (1, 6))
    mine = ex.generate_with_cache(model, prompt.clone(), 20, temperature=0.0)
    theirs = ref.generate_with_cache(model, prompt.clone(), 20, temperature=0.0)
    assert torch.equal(mine, theirs)
