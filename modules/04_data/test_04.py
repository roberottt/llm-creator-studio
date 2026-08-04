"""Tests for module 04. Run them with `llmfs check 04`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import llmfs.reference as ref
from llmfs.testing import load_exercises

ex = load_exercises(__file__)


def corpus(n: int = 10_000, vocab: int = 4096, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, vocab, size=n).astype(np.uint16)


# ------------------------------------------------------- exercise 1: pack_tokens_uint16


def test_it_returns_a_uint16_array():
    out = ex.pack_tokens_uint16([0, 1, 2, 4095], 4096)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint16


def test_it_preserves_the_values():
    ids = [0, 1, 2, 4095, 300]
    assert ex.pack_tokens_uint16(ids, 4096).tolist() == ids


def test_it_takes_two_bytes_per_token():
    """The whole point of the exercise: half the disk of uint32."""
    out = ex.pack_tokens_uint16(list(range(1000)), 4096)
    assert out.nbytes == 2000


def test_an_id_outside_the_vocabulary_is_an_error():
    """Numpy's silent wrap-around is what has to be avoided."""
    with pytest.raises(ValueError):
        ex.pack_tokens_uint16([0, 1, 4096], 4096)


def test_a_negative_id_is_an_error():
    with pytest.raises(ValueError):
        ex.pack_tokens_uint16([0, -1, 2], 4096)


def test_a_vocabulary_that_does_not_fit_in_uint16_is_an_error():
    with pytest.raises(ValueError):
        ex.pack_tokens_uint16([0, 1], 100_000)


def test_the_error_message_says_which_value_went_out_of_range():
    with pytest.raises(ValueError) as exc:
        ex.pack_tokens_uint16([0, 1, 9999], 4096)
    assert "9999" in str(exc.value), "the error should say which id is the problem"


def test_an_empty_list_does_not_blow_up():
    """`.min()` on an empty array raises: the size has to be checked."""
    out = ex.pack_tokens_uint16([], 4096)
    assert out.dtype == np.uint16 and len(out) == 0


def test_the_packing_matches_the_reference():
    ids = list(range(0, 4096, 7))
    assert np.array_equal(ex.pack_tokens_uint16(ids, 4096), ref.pack_tokens_uint16(ids, 4096))


# ---------------------------------------------------------- exercise 2: train_val_split


def test_the_two_parts_add_up_to_the_total():
    data = corpus(10_000)
    train, val = ex.train_val_split(data, 0.01)
    assert len(train) + len(val) == len(data)


def test_the_fraction_is_the_requested_one():
    data = corpus(10_000)
    _, val = ex.train_val_split(data, 0.05)
    assert len(val) == 500


def test_validation_is_the_end_of_the_corpus():
    """Contiguous and from the end, not random. See THEORY.md."""
    data = corpus(1000)
    train, val = ex.train_val_split(data, 0.1)
    assert np.array_equal(val, data[-100:])
    assert np.array_equal(train, data[:-100])


def test_there_is_no_overlap_between_the_two_parts():
    data = np.arange(1000, dtype=np.uint16)
    train, val = ex.train_val_split(data, 0.1)
    assert set(train.tolist()).isdisjoint(set(val.tolist()))


def test_it_returns_views_and_not_copies():
    """With 500M tokens, copying is a gigabyte thrown away."""
    data = corpus(1000)
    train, _ = ex.train_val_split(data, 0.1)
    assert train.base is data or np.shares_memory(train, data)


def test_validation_is_never_left_empty():
    """With a small corpus, int(len * fraction) could give 0."""
    _, val = ex.train_val_split(corpus(50), 0.005)
    assert len(val) >= 1


def test_an_invalid_fraction_is_an_error():
    data = corpus(1000)
    for fraction in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            ex.train_val_split(data, fraction)


def test_the_split_matches_the_reference():
    data = corpus(5000)
    mine_tr, mine_val = ex.train_val_split(data, 0.02)
    ref_tr, ref_val = ref.train_val_split(data, 0.02)
    assert np.array_equal(mine_tr, ref_tr) and np.array_equal(mine_val, ref_val)


# ----------------------------------------------------------------- exercise 3: get_batch


def test_the_shapes_are_right():
    x, y = ex.get_batch(corpus(), batch_size=4, context_length=16, rng=np.random.default_rng(0))
    assert x.shape == (4, 16) and y.shape == (4, 16)


def test_the_dtype_is_int64():
    """nn.Embedding's indices have to be int64."""
    x, y = ex.get_batch(corpus(), 4, 16, rng=np.random.default_rng(0))
    assert x.dtype == torch.int64 and y.dtype == torch.int64


def test_y_is_x_shifted_by_one_token():
    """The property that defines the task: predict the next one."""
    x, y = ex.get_batch(corpus(), 8, 32, rng=np.random.default_rng(1))
    assert torch.equal(x[:, 1:], y[:, :-1]), (
        "y[i] has to be the token that follows x[i]. If this fails, the model would be "
        "learning to predict something else entirely."
    )


def test_the_tokens_come_from_the_corpus_and_are_in_order():
    """Checks against the real corpus that the window is contiguous."""
    data = np.arange(5000, dtype=np.uint16)
    x, y = ex.get_batch(data, 4, 10, rng=np.random.default_rng(3))
    for row in range(4):
        start = int(x[row, 0])
        assert x[row].tolist() == list(range(start, start + 10))
        assert y[row].tolist() == list(range(start + 1, start + 11))


def test_it_never_runs_off_the_end_of_the_corpus():
    """The -1 in max_start. Without it, the last window of `y` overflows."""
    data = np.arange(200, dtype=np.uint16)
    for seed in range(30):
        x, y = ex.get_batch(data, 16, 50, rng=np.random.default_rng(seed))
        assert int(y.max()) <= 199


def test_a_corpus_shorter_than_the_context_is_an_error():
    with pytest.raises(ValueError):
        ex.get_batch(corpus(10), batch_size=2, context_length=64)


def test_the_same_seed_gives_the_same_batch():
    data = corpus()
    a = ex.get_batch(data, 4, 16, rng=np.random.default_rng(42))
    b = ex.get_batch(data, 4, 16, rng=np.random.default_rng(42))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_different_seeds_give_different_batches():
    data = corpus()
    a = ex.get_batch(data, 8, 32, rng=np.random.default_rng(1))
    b = ex.get_batch(data, 8, 32, rng=np.random.default_rng(2))
    assert not torch.equal(a[0], b[0])


def test_get_batch_matches_the_reference_exactly():
    data = corpus()
    mine = ex.get_batch(data, 8, 32, rng=np.random.default_rng(7))
    theirs = ref.get_batch(data, 8, 32, rng=np.random.default_rng(7))
    assert torch.equal(mine[0], theirs[0]) and torch.equal(mine[1], theirs[1])


def test_it_works_over_a_memmap_on_disk(tmp_path):
    """The real case: the corpus is not in RAM, it is in a file."""
    path = tmp_path / "tokens.bin"
    corpus(20_000).tofile(path)
    data = np.memmap(path, dtype=np.uint16, mode="r")

    x, y = ex.get_batch(data, 4, 64, rng=np.random.default_rng(0))
    assert x.shape == (4, 64)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_it_moves_the_tensors_to_the_device():
    from llmfs.device import get_device

    cfg = get_device()
    x, y = ex.get_batch(corpus(), 4, 16, device=cfg.device, rng=np.random.default_rng(0))
    assert x.device.type == cfg.kind and y.device.type == cfg.kind
