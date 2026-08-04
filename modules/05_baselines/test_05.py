"""Tests for module 05. Run them with `llmfs check 05`."""

from __future__ import annotations

import math

import pytest
import torch

import llmfs.reference as ref
from llmfs.testing import assert_close, assert_scalar_close, copy_parameters, load_exercises

ex = load_exercises(__file__)

SEQUENCE = [0, 1, 2, 1, 0, 1, 2, 2, 1, 0] * 40
VOCAB = 3


# --------------------------------------------------- exercise 1: uniform_baseline_loss


@pytest.mark.parametrize("vocab", [2, 65, 4096, 50257])
def test_it_is_the_logarithm_of_the_vocabulary(vocab):
    assert_scalar_close(ex.uniform_baseline_loss(vocab), math.log(vocab), what=f"ln({vocab})")


def test_the_final_models_number():
    """8.317 is what the step-0 loss has to be in module 11."""
    assert_scalar_close(ex.uniform_baseline_loss(4096), 8.3178, rtol=1e-4, what="ln(4096)")


def test_a_vocabulary_of_one_gives_zero_loss():
    """If there is only one possible option, getting it right is free."""
    assert_scalar_close(ex.uniform_baseline_loss(1), 0.0, atol=1e-12, what="ln(1)")


def test_a_non_positive_vocabulary_is_an_error():
    for bad in (0, -5):
        with pytest.raises(ValueError):
            ex.uniform_baseline_loss(bad)


def test_it_matches_torchs_cross_entropy():
    """A cross-check: uniform logits must give exactly ln(V)."""
    vocab = 512
    logits = torch.zeros(1, vocab)  # all equal -> uniform softmax
    target = torch.tensor([7])
    assert_scalar_close(
        ex.uniform_baseline_loss(vocab),
        float(torch.nn.functional.cross_entropy(logits, target)),
        rtol=1e-5,
        what="ln(V) vs cross_entropy with flat logits",
    )


# ---------------------------------------------------------- exercise 2: bigram_counts


def test_it_counts_the_example_from_the_statement():
    counts = ex.bigram_counts([0, 1, 0, 1, 2], 3)
    assert counts[0][1].item() == 2
    assert counts[1][0].item() == 1
    assert counts[1][2].item() == 1
    assert counts.sum().item() == 4


def test_the_shape_and_type_are_right():
    counts = ex.bigram_counts(SEQUENCE, VOCAB)
    assert counts.shape == (VOCAB, VOCAB)
    assert counts.dtype == torch.int64


def test_the_total_is_the_number_of_pairs():
    counts = ex.bigram_counts(SEQUENCE, VOCAB)
    assert counts.sum().item() == len(SEQUENCE) - 1


def test_repeats_accumulate_rather_than_overwrite():
    """Without accumulate=True every count would come out 1."""
    counts = ex.bigram_counts([0, 0, 0, 0, 0], 2)
    assert counts[0][0].item() == 4


def test_a_short_sequence_gives_the_matrix_of_zeros():
    assert ex.bigram_counts([5], 10).sum().item() == 0
    assert ex.bigram_counts([], 10).sum().item() == 0


def test_the_counts_match_the_reference():
    assert torch.equal(ex.bigram_counts(SEQUENCE, VOCAB), ref.bigram_counts(SEQUENCE, VOCAB))


# ------------------------------------------------------------- exercise 3: bigram_nll


def test_the_loss_matches_the_reference():
    counts = ref.bigram_counts(SEQUENCE, VOCAB)
    assert_scalar_close(
        ex.bigram_nll(counts, SEQUENCE), ref.bigram_nll(counts, SEQUENCE), what="the loss"
    )


def test_the_bigram_beats_the_uniform_baseline():
    """If this fails, the bigram model is not learning anything."""
    counts = ref.bigram_counts(SEQUENCE, VOCAB)
    loss = ex.bigram_nll(counts, SEQUENCE)
    assert loss < math.log(VOCAB), (
        f"the bigram gives {loss:.4f} and guessing at random gives {math.log(VOCAB):.4f}"
    )


def test_a_perfectly_predictable_sequence_gives_almost_zero_loss():
    cycle = [0, 1] * 500
    counts = ref.bigram_counts(cycle, 2)
    assert ex.bigram_nll(counts, cycle, alpha=1e-6) < 0.01


def test_the_smoothing_prevents_infinity():
    """The point of the exercise: an unseen pair cannot send the loss to infinity."""
    counts = ref.bigram_counts([0, 1, 0, 1], 3)  # token 2 never appears
    loss = ex.bigram_nll(counts, [2, 0, 1, 2], alpha=1.0)
    assert math.isfinite(loss), "with alpha > 0 the loss cannot be infinite"


def test_more_smoothing_moves_the_loss_towards_the_uniform_baseline():
    """A huge alpha flattens the counts and the model becomes uniform."""
    counts = ref.bigram_counts(SEQUENCE, VOCAB)
    little = ex.bigram_nll(counts, SEQUENCE, alpha=0.01)
    lots = ex.bigram_nll(counts, SEQUENCE, alpha=1e6)
    assert little < lots
    assert_scalar_close(lots, math.log(VOCAB), rtol=1e-3, what="loss with a huge alpha")


def test_each_rows_probabilities_sum_to_one():
    """An indirect check on the denominator: you have to add alpha*V, not just alpha."""
    counts = ref.bigram_counts(SEQUENCE, VOCAB)
    smoothed = counts.double() + 2.5
    probs = smoothed / smoothed.sum(dim=1, keepdim=True)
    assert torch.allclose(probs.sum(dim=1), torch.ones(VOCAB, dtype=torch.float64))


def test_a_sequence_of_fewer_than_two_tokens_is_an_error():
    counts = ref.bigram_counts(SEQUENCE, VOCAB)
    with pytest.raises(ValueError):
        ex.bigram_nll(counts, [1])


# ----------------------------------------------------------- exercise 4: NeuralBigram


def test_neural_bigram_has_the_expected_architecture():
    model = ex.NeuralBigram(VOCAB)
    expected = ref.NeuralBigram(VOCAB)
    copy_parameters(expected, model)  # fails with a useful message if the names do not match


def test_neural_bigram_returns_the_right_shapes():
    model = ex.NeuralBigram(VOCAB)
    idx = torch.randint(0, VOCAB, (4, 8))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, 8, VOCAB)
    assert loss is not None and loss.ndim == 0


def test_neural_bigram_without_targets_returns_no_loss():
    model = ex.NeuralBigram(VOCAB)
    logits, loss = model(torch.randint(0, VOCAB, (2, 5)))
    assert loss is None


def test_neural_bigram_matches_the_reference():
    torch.manual_seed(0)
    mine, theirs = ex.NeuralBigram(VOCAB), ref.NeuralBigram(VOCAB)
    copy_parameters(theirs, mine)

    idx = torch.randint(0, VOCAB, (4, 8))
    tgt = torch.randint(0, VOCAB, (4, 8))
    my_logits, my_loss = mine(idx, tgt)
    their_logits, their_loss = theirs(idx, tgt)

    assert_close(my_logits, their_logits, what="the logits")
    assert_scalar_close(my_loss, their_loss, what="the loss")


def test_neural_bigram_has_v_squared_parameters():
    model = ex.NeuralBigram(50)
    assert sum(p.numel() for p in model.parameters()) == 50 * 50


def test_a_trained_neural_bigram_approaches_the_count_based_bigram():
    """Counting and learning by gradient give the same thing with a model this simple."""
    torch.manual_seed(0)
    data = torch.tensor(SEQUENCE, dtype=torch.long)
    model = ex.NeuralBigram(VOCAB)
    opt = torch.optim.AdamW(model.parameters(), lr=0.5)

    x, y = data[:-1].unsqueeze(0), data[1:].unsqueeze(0)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    by_counting = ref.bigram_nll(ref.bigram_counts(SEQUENCE, VOCAB), SEQUENCE, alpha=1e-4)
    assert abs(float(loss) - by_counting) < 0.1, (
        f"trained it gives {float(loss):.4f} and counting gives {by_counting:.4f}; "
        "they should converge to the same place"
    )


# ------------------------------------------------------------- exercise 5: BengioMLP


def test_bengio_has_the_expected_architecture():
    model = ex.BengioMLP(VOCAB, block_size=4)
    copy_parameters(ref.BengioMLP(VOCAB, block_size=4), model)


def test_bengio_returns_the_right_shapes():
    model = ex.BengioMLP(VOCAB, block_size=4)
    idx = torch.randint(0, VOCAB, (6, 4))
    logits, loss = model(idx, torch.randint(0, VOCAB, (6,)))
    assert logits.shape == (6, VOCAB), "one logits vector per sample, not one per token"
    assert loss is not None and loss.ndim == 0


def test_bengio_matches_the_reference():
    torch.manual_seed(0)
    mine = ex.BengioMLP(VOCAB, block_size=4, d_embed=8, n_hidden=16)
    theirs = ref.BengioMLP(VOCAB, block_size=4, d_embed=8, n_hidden=16)
    copy_parameters(theirs, mine)

    idx = torch.randint(0, VOCAB, (6, 4))
    tgt = torch.randint(0, VOCAB, (6,))
    my_logits, my_loss = mine(idx, tgt)
    their_logits, their_loss = theirs(idx, tgt)

    assert_close(my_logits, their_logits, what="the logits")
    assert_scalar_close(my_loss, their_loss, what="the loss")


def test_bengio_concatenates_instead_of_averaging():
    """If you averaged the embeddings, the token order would not matter. It does matter."""
    torch.manual_seed(0)
    model = ex.BengioMLP(VOCAB, block_size=3, d_embed=8, n_hidden=16)
    a = model(torch.tensor([[0, 1, 2]]))[0]
    b = model(torch.tensor([[2, 1, 0]]))[0]
    assert not torch.allclose(a, b), (
        "the model gives the same thing with the context reversed: you are averaging, "
        "not concatenating"
    )


def test_bengios_parameters_grow_with_the_context():
    """Its limitation, and the reason attention exists."""
    small = sum(p.numel() for p in ex.BengioMLP(VOCAB, 2, d_embed=8, n_hidden=16).parameters())
    large = sum(p.numel() for p in ex.BengioMLP(VOCAB, 8, d_embed=8, n_hidden=16).parameters())
    assert large > small


def test_bengio_really_learns():
    torch.manual_seed(0)
    data = torch.tensor(SEQUENCE, dtype=torch.long)
    block = 3
    x = torch.stack([data[i : i + block] for i in range(len(data) - block)])
    y = data[block:]

    model = ex.BengioMLP(VOCAB, block, d_embed=8, n_hidden=32)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)
    initial = None
    for _ in range(200):
        _, loss = model(x, y)
        initial = initial if initial is not None else float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert float(loss) < initial
    assert float(loss) < math.log(VOCAB), "it should beat the uniform baseline"
