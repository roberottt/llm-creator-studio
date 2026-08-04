"""Tests for module 03. Run them with `llmfs check 03`."""

from __future__ import annotations

import pytest

import llmfs.reference as ref
from llmfs.reference import GPT4_SPLIT_PATTERN
from llmfs.testing import load_exercises

ex = load_exercises(__file__)

# Text with enough variety that at least 90 merges fit without running out of pairs.
TEXT = (
    "the black cat eats fresh fish in the morning. "
    "the little girl watches the white dog from the big window. "
    "the dog runs fast towards the green garden and barks a lot. "
    "tomorrow we will go back to the beach with the children and their toys. "
    "a blue bird sings on the highest branch of the old tree. "
) * 12
UNICODE = "ñandú 🦙 café über 日本語 naïve"


# ----------------------------------------------------------------- exercise 1: get_stats


def test_it_counts_the_pairs_from_the_statement():
    assert ex.get_stats([97, 97, 97, 98]) == {(97, 97): 2, (97, 98): 1}


def test_pairs_overlap_when_counting():
    """In [1,1,1,1] the pair (1,1) comes up 3 times, not 2."""
    assert ex.get_stats([1, 1, 1, 1]) == {(1, 1): 3}


def test_a_one_element_sequence_has_no_pairs():
    assert ex.get_stats([5]) == {}
    assert ex.get_stats([]) == {}


def test_it_accumulates_into_the_dictionary_you_pass_in():
    """This is what lets train_bpe add up several chunks without concatenating them."""
    accumulator: dict[tuple[int, int], int] = {}
    ex.get_stats([1, 2], accumulator)
    ex.get_stats([1, 2, 3], accumulator)
    assert accumulator == {(1, 2): 2, (2, 3): 1}


def test_it_returns_the_same_dictionary_it_receives():
    accumulator: dict[tuple[int, int], int] = {}
    assert ex.get_stats([1, 2], accumulator) is accumulator


def test_get_stats_matches_the_reference():
    ids = list(TEXT.encode("utf-8"))
    assert ex.get_stats(ids) == ref.get_stats(ids)


# -------------------------------------------------------------------- exercise 2: merge


def test_it_merges_the_example_from_the_statement():
    assert ex.merge([97, 97, 97, 98, 97, 97], (97, 97), 256) == [256, 97, 98, 256]


def test_merging_does_not_overlap():
    """[1,1,1] merging (1,1) gives [256,1], not [256,256]."""
    assert ex.merge([1, 1, 1], (1, 1), 256) == [256, 1]


def test_it_touches_nothing_if_the_pair_does_not_appear():
    assert ex.merge([1, 2, 3], (7, 8), 256) == [1, 2, 3]


def test_it_merges_a_pair_at_the_end_of_the_list():
    assert ex.merge([1, 2, 3, 4], (3, 4), 256) == [1, 2, 256]


def test_it_returns_a_new_list_without_modifying_the_original():
    original = [1, 1, 2]
    result = ex.merge(original, (1, 1), 256)
    assert original == [1, 1, 2], "you modified the input list"
    assert result is not original


def test_merge_matches_the_reference_on_real_text():
    ids = list(TEXT.encode("utf-8"))
    pair = max(ref.get_stats(ids), key=lambda p: (ref.get_stats(ids)[p], p))
    assert ex.merge(ids, pair, 256) == ref.merge(ids, pair, 256)


# ---------------------------------------------------------------- exercise 3: train_bpe


def test_the_theory_example_step_by_step():
    """'aaabdaaabac' with 2 merges: (a,a)->256 and (256,a)->257."""
    merges, vocab = ex.train_bpe("aaabdaaabac", 258)
    assert merges == {(97, 97): 256, (256, 97): 257}
    assert vocab[256] == b"aa"
    assert vocab[257] == b"aaa"


def test_the_vocabulary_starts_with_the_256_bytes():
    _, vocab = ex.train_bpe(TEXT, 300)
    for i in range(256):
        assert vocab[i] == bytes([i]), f"byte {i} is missing from the vocabulary"


def test_it_produces_the_requested_number_of_merges():
    merges, vocab = ex.train_bpe(TEXT, 300)
    assert len(merges) == 300 - 256
    assert len(vocab) == 300


def test_the_new_ids_start_at_256_and_are_consecutive():
    merges, _ = ex.train_bpe(TEXT, 290)
    assert list(merges.values()) == list(range(256, 290))


def test_a_vocabulary_smaller_than_256_is_an_error():
    with pytest.raises(ValueError):
        ex.train_bpe(TEXT, 100)


def test_it_stops_if_it_runs_out_of_pairs_to_merge():
    """With a tiny text there are not 1000 possible merges: it has to stop, not blow up."""
    merges, vocab = ex.train_bpe("abc", 1000)
    assert len(merges) < 744
    assert len(vocab) == 256 + len(merges)


def test_the_merges_match_the_reference_exactly():
    assert ex.train_bpe(TEXT, 320)[0] == ref.train_bpe(TEXT, 320)[0]


def test_it_also_matches_when_using_the_pre_tokenizer():
    mine = ex.train_bpe(TEXT, 320, pattern=GPT4_SPLIT_PATTERN)[0]
    theirs = ref.train_bpe(TEXT, 320, pattern=GPT4_SPLIT_PATTERN)[0]
    assert mine == theirs


def test_the_pre_tokenizer_blocks_merges_across_spaces():
    """Without a pattern, BPE learns 'e t' joining the end of one word to the next."""
    _, vocab_with = ex.train_bpe(TEXT, 340, pattern=GPT4_SPLIT_PATTERN)
    tokens_with = [vocab_with[i] for i in range(256, 340)]
    # A token with a space IN THE MIDDLE means the merge crossed a boundary.
    for token in tokens_with:
        token_text = token.decode("utf-8", errors="replace")
        assert " " not in token_text.strip(), (
            f"the token {token_text!r} contains an interior space: the pre-tokenizer "
            "is not being applied"
        )


def test_the_vocabulary_is_consistent_with_the_merges():
    """vocab[new] has to be the concatenation of the bytes of its two parents."""
    merges, vocab = ex.train_bpe(TEXT, 300)
    for (a, b), new in merges.items():
        assert vocab[new] == vocab[a] + vocab[b]


# --------------------------------------------------------- exercises 4 and 5: encode/decode


def test_it_encodes_the_theory_example():
    merges, _ = ref.train_bpe("aaabdaaabac", 258)
    assert ex.bpe_encode("aaabdaaabac", merges) == [257, 98, 100, 257, 98, 97, 99]


def test_encoding_then_decoding_returns_the_original_text():
    merges, vocab = ref.train_bpe(TEXT, 400)
    assert ex.bpe_decode(ex.bpe_encode(TEXT, merges), vocab) == TEXT


def test_the_roundtrip_works_with_text_not_used_in_training():
    merges, vocab = ref.train_bpe(TEXT, 400)
    new_text = "a completely different text, with numbers 12345 and signs !?"
    assert ex.bpe_decode(ex.bpe_encode(new_text, merges), vocab) == new_text


def test_the_roundtrip_survives_unicode():
    """Emoji, accents and Japanese. This is where you see why we work with bytes."""
    merges, vocab = ref.train_bpe(TEXT, 400)
    assert ex.bpe_decode(ex.bpe_encode(UNICODE, merges), vocab) == UNICODE


def test_the_roundtrip_works_with_the_pre_tokenizer():
    merges, vocab = ref.train_bpe(TEXT, 400, pattern=GPT4_SPLIT_PATTERN)
    ids = ex.bpe_encode(UNICODE, merges, pattern=GPT4_SPLIT_PATTERN)
    assert ex.bpe_decode(ids, vocab) == UNICODE


def test_encoding_really_compresses():
    merges, _ = ref.train_bpe(TEXT, 400)
    ids = ex.bpe_encode(TEXT, merges)
    assert len(ids) < len(TEXT.encode("utf-8")) / 2, (
        "with 144 merges over a very repetitive text it should compress at least by half"
    )


def test_the_encoding_matches_the_reference():
    merges, _ = ref.train_bpe(TEXT, 400)
    assert ex.bpe_encode(TEXT, merges) == ref.bpe_encode(TEXT, merges)


def test_the_merges_are_applied_in_the_order_they_were_learned():
    """The case that separates 'apply in learning order' from 'apply whatever you find'.

    merges = {(a,b): 256, (a,a): 257}  over the text "aab".

    Both pairs are present. Since (a,b) was learned FIRST (id 256 < 257), that is the one to
    apply first:

        "aab" -> [a, 256]

    If you applied (a,a) because it is the first one you find, or by frequency, you would
    get [257, b]. Both tokenizations have two tokens and both are "valid", but only one is
    the one the model saw during training. It does not understand the other one.
    """
    merges = {(97, 98): 256, (97, 97): 257}
    assert ex.bpe_encode("aab", merges) == [97, 256]


def test_a_merge_consumes_all_its_occurrences_before_moving_to_the_next():
    """A subtle consequence of the algorithm, and worth seeing once.

    With merges (a,a)->256 and (256,a)->257, the string "aaaa" does NOT give [257, a].

    It gives [256, 256]: the first merge is applied to the WHOLE sequence at once and takes
    the four 'a's two at a time, so the pair (256, a) never gets to exist. With three 'a's
    you would get [257], because [256, a] would be left and there the second merge applies.
    """
    merges = {(97, 97): 256, (256, 97): 257}
    assert ex.bpe_encode("aaaa", merges) == [256, 256]
    assert ex.bpe_encode("aaa", merges) == [257]


def test_decoding_does_not_break_on_invalid_bytes():
    """The bytes fallback: a half-trained model produces this constantly."""
    vocab = {i: bytes([i]) for i in range(256)}
    result = ex.bpe_decode([0xC3], vocab)  # half of a multi-byte character
    assert isinstance(result, str), "it must not raise, it must use errors='replace'"


def test_decoding_joins_the_bytes_before_decoding():
    """Two tokens that together form an 'ñ', but separately are not valid."""
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256], vocab[257] = b"\xc3", b"\xb1"
    assert ex.bpe_decode([256, 257], vocab) == "ñ"


def test_an_empty_list_decodes_to_an_empty_string():
    assert ex.bpe_decode([], {i: bytes([i]) for i in range(256)}) == ""
