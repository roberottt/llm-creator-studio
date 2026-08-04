"""Tests for module 10. Run them with `llmfs check 10`.

The test that matters most is `test_the_final_model_has_8_933_440_parameters`.
"""

from __future__ import annotations

import math

import pytest
import torch

import llmfs.reference as ref
from llmfs.config import ModelConfig, RunConfig
from llmfs.paths import configs_dir
from llmfs.testing import assert_close, assert_scalar_close, copy_parameters, load_exercises

ex = load_exercises(__file__)


def small_cfg(**kwargs) -> ModelConfig:
    """A tiny model so the tests run fast."""
    base = dict(vocab_size=64, n_layers=2, d_model=32, n_heads=4, d_ff=96, context_length=16)
    base.update(kwargs)
    return ModelConfig(**base)


# ------------------------------------------------- exercise 1: expected_param_count


def test_the_final_model_has_8_933_440_parameters():
    """THE number of the course. If this fails, the config or the formula is lying."""
    assert ex.expected_param_count(ModelConfig()) == 8_933_440


def test_the_theory_breakdown_adds_up_term_by_term():
    m = ModelConfig()
    embeddings = m.vocab_size * m.d_model
    attention = 4 * m.d_model**2
    swiglu = 3 * m.d_model * m.d_ff
    norms = 2 * m.d_model
    per_layer = attention + swiglu + norms

    assert embeddings == 1_310_720
    assert attention == 409_600
    assert swiglu == 860_160
    assert per_layer == 1_270_400
    assert embeddings + m.n_layers * per_layer + m.d_model == ex.expected_param_count(m)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"n_layers": 1},
        {"tie_embeddings": False},
        {"norm": "layernorm"},
        {"activation": "gelu"},
        {"pos": "learned"},
        {"bias": True},
        {"vocab_size": 128, "d_model": 64, "n_heads": 8, "d_ff": 192},
    ],
)
def test_it_matches_the_reference_across_many_configurations(kwargs):
    cfg = small_cfg(**kwargs)
    assert ex.expected_param_count(cfg) == ref.expected_param_count(cfg)


def test_without_tying_there_are_1_310_720_more_parameters():
    with_tying = ex.expected_param_count(ModelConfig(tie_embeddings=True))
    without = ex.expected_param_count(ModelConfig(tie_embeddings=False))
    assert without - with_tying == 4096 * 320 == 1_310_720


def test_rope_adds_not_one_parameter():
    """Its tables are buffers computed from a formula, not parameters."""
    with_rope = ex.expected_param_count(small_cfg(pos="rope"))
    without_position = ex.expected_param_count(small_cfg(pos="none"))
    assert with_rope == without_position


def test_learned_positions_do_add_parameters():
    learned = ex.expected_param_count(small_cfg(pos="learned"))
    rope = ex.expected_param_count(small_cfg(pos="rope"))
    assert learned - rope == 16 * 32  # context_length * d_model


def test_it_returns_an_integer():
    assert isinstance(ex.expected_param_count(ModelConfig()), int)


# ----------------------------------------------------- exercise 2: count_parameters


def test_the_real_count_matches_the_formula():
    """Both exercises have to say the same thing. It is the cross-check."""
    for kwargs in ({}, {"tie_embeddings": False}, {"norm": "layernorm"}, {"pos": "learned"}):
        cfg = small_cfg(**kwargs)
        model = ref.GPT(cfg)
        assert ex.count_parameters(model)["total"] == ex.expected_param_count(cfg), (
            f"with {kwargs} the formula and the count do not match"
        )


def test_the_total_matches_sum_of_parameters():
    model = ref.GPT(small_cfg())
    assert ex.count_parameters(model)["total"] == sum(p.numel() for p in model.parameters())


def test_it_does_not_count_tied_weights_twice():
    """With weight tying, the same tensor appears under two different names.

    A useful and little-known fact: `named_parameters()` DEDUPLICATES by default
    (`remove_duplicate=True`), so summing over it already comes out right. With
    `remove_duplicate=False` you see the tensor repeated and the naive count shoots up.
    """
    model = ref.GPT(small_cfg(tie_embeddings=True))
    counts = ex.count_parameters(model)

    with_duplicates = sum(
        p.numel() for _, p in model.named_parameters(remove_duplicate=False)
    )
    assert with_duplicates > counts["total"], "the test model should have tied weights"
    assert with_duplicates - counts["total"] == 64 * 32, "the embedding counted twice"
    assert counts["total"] == sum(p.numel() for p in model.parameters())


def test_it_has_every_key():
    counts = ex.count_parameters(ref.GPT(small_cfg()))
    for key in ("embeddings", "attention", "ffn", "norms", "lm_head", "other",
                "total", "non_embedding"):
        assert key in counts, f"the key {key!r} is missing"


def test_non_embedding_is_total_minus_embeddings():
    counts = ex.count_parameters(ref.GPT(small_cfg()))
    assert counts["non_embedding"] == counts["total"] - counts["embeddings"]


def test_the_final_models_breakdown():
    """The numbers module 12 will use for the scaling laws."""
    counts = ex.count_parameters(ref.GPT(ModelConfig()))
    assert counts["embeddings"] == 1_310_720
    assert counts["attention"] == 6 * 409_600
    assert counts["ffn"] == 6 * 860_160
    assert counts["total"] == 8_933_440
    assert counts["non_embedding"] == 7_622_720


def test_with_tying_the_lm_head_contributes_nothing():
    counts = ex.count_parameters(ref.GPT(small_cfg(tie_embeddings=True)))
    assert counts["lm_head"] == 0


def test_the_count_matches_the_reference():
    model = ref.GPT(small_cfg())
    assert ex.count_parameters(model) == ref.count_parameters(model)


# ---------------------------------------------------- exercise 3: TransformerBlock


def test_the_block_has_the_expected_architecture():
    cfg = small_cfg()
    copy_parameters(ref.TransformerBlock(cfg), ex.TransformerBlock(cfg))


def test_the_block_preserves_the_shape():
    block = ex.TransformerBlock(small_cfg())
    assert block(torch.randn(2, 8, 32)).shape == (2, 8, 32)


def test_the_block_matches_the_reference():
    torch.manual_seed(0)
    cfg = small_cfg()
    mine, theirs = ex.TransformerBlock(cfg), ref.TransformerBlock(cfg)
    copy_parameters(theirs, mine)
    mine.eval()
    theirs.eval()
    x = torch.randn(2, 8, 32)
    cos, sin = ref.rope_frequencies(cfg.head_dim, 16)
    assert_close(
        mine(x, cos=cos, sin=sin), theirs(x, cos=cos, sin=sin), what="the block's output"
    )


def test_the_block_uses_residuals():
    """With the output weights at zero, the output has to be exactly the input."""
    torch.manual_seed(0)
    block = ex.TransformerBlock(small_cfg())
    block.eval()
    with torch.no_grad():
        for name, p in block.named_parameters():
            if name.endswith(("out_proj.weight", "down_proj.weight", "fc_out.weight")):
                p.zero_()
    x = torch.randn(2, 8, 32)
    assert_close(block(x), x, atol=1e-5, what="the output with the branches nulled")


def test_the_block_has_the_parameters_from_the_breakdown():
    """1,270,400 per layer with the final config."""
    block = ex.TransformerBlock(ModelConfig())
    assert sum(p.numel() for p in block.parameters()) == 1_270_400


# ------------------------------------------------------------- exercise 4: GPT


def test_the_complete_gpt_has_8_933_440_parameters():
    """The test that closes Part II."""
    model = ex.GPT(ModelConfig())
    assert sum(p.numel() for p in model.parameters()) == 8_933_440


def test_the_gpt_returns_the_right_shapes():
    model = ex.GPT(small_cfg())
    idx = torch.randint(0, 64, (2, 8))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 8, 64)
    assert loss is not None and loss.ndim == 0


def test_without_targets_there_is_no_loss():
    model = ex.GPT(small_cfg())
    _, loss = model(torch.randint(0, 64, (2, 8)))
    assert loss is None


def test_the_tied_weights_are_the_same_tensor():
    model = ex.GPT(small_cfg(tie_embeddings=True))
    assert model.lm_head.weight is model.token_embedding.weight, (
        "they have to be the SAME object, not copies with the same values"
    )


def test_without_tying_they_are_different_tensors():
    model = ex.GPT(small_cfg(tie_embeddings=False))
    assert model.lm_head.weight is not model.token_embedding.weight


def test_the_initial_loss_equals_ln_of_the_vocabulary():
    """Module 05's bug detector, applied to the real model.

    If it comes out much higher, the initialization is too aggressive.
    If it comes out lower, there is an information leak (badly placed causal mask).

    WATCH how the batch is built: the targets have to be SHIFTED by one token. If you pass
    `model(idx, idx)`, at position t the model sees token idx[t] and you ask it to predict
    idx[t]: it can read it off its own input. With weight tying the leak is even more direct,
    because the logits are `x @ W_emb.T` and the product of an embedding with itself is
    large. The loss comes out below ln(V) and it looks like a model bug when it is a test
    bug.
    """
    torch.manual_seed(0)
    cfg = small_cfg(vocab_size=256)
    model = ex.GPT(cfg)
    model.eval()
    sequence = torch.randint(0, 256, (8, 17))
    x, y = sequence[:, :-1], sequence[:, 1:]
    with torch.no_grad():
        _, loss = model(x, y)

    expected = math.log(256)
    assert abs(float(loss) - expected) < 0.15, (
        f"the initial loss is {float(loss):.4f} and it should be around "
        f"ln(256)={expected:.4f}. Higher: init too aggressive. Lower: information leak."
    )


def test_the_model_is_causal():
    """Changing a token CANNOT affect the earlier predictions."""
    torch.manual_seed(0)
    model = ex.GPT(small_cfg())
    model.eval()
    idx = torch.randint(0, 64, (1, 10))

    with torch.no_grad():
        original = model(idx)[0]
        modified = idx.clone()
        modified[0, 7] = (modified[0, 7] + 1) % 64
        altered = model(modified)[0]

    assert_close(altered[:, :7], original[:, :7], atol=1e-5,
                 what="the logits before the changed token")
    assert not torch.allclose(altered[:, 7:], original[:, 7:], atol=1e-5), (
        "changing token 7 should affect the predictions from 7 onwards"
    )


def test_the_depth_scaled_initialization_is_applied():
    """The projections that write into the residual start smaller."""
    torch.manual_seed(0)
    cfg = small_cfg(n_layers=8, d_model=128, n_heads=8, d_ff=256)
    model = ex.GPT(cfg)

    residual, normal = [], []
    for name, p in model.named_parameters():
        if name.endswith(("out_proj.weight", "down_proj.weight")):
            residual.append(float(p.std()))
        elif name.endswith(("q_proj.weight", "gate_proj.weight")):
            normal.append(float(p.std()))

    residual_std = sum(residual) / len(residual)
    normal_std = sum(normal) / len(normal)
    expected = 0.02 / math.sqrt(2 * cfg.n_layers)

    assert abs(residual_std - expected) < 0.002, (
        f"the residual projections have std {residual_std:.5f} and it should be "
        f"0.02/sqrt(2*{cfg.n_layers}) = {expected:.5f}"
    )
    assert normal_std > residual_std * 2, "the rest should start with std=0.02"


def test_rope_is_stored_as_a_non_persistent_buffer():
    model = ex.GPT(small_cfg(pos="rope"))
    buffer_names = [n for n, _ in model.named_buffers()]
    assert any("rope" in n for n in buffer_names), "RoPE's tables must be buffers"
    assert not any("rope" in n for n in model.state_dict()), (
        "with persistent=False they must not appear in the state_dict: they get recomputed"
    )


def test_a_sequence_longer_than_the_context_is_an_error():
    model = ex.GPT(small_cfg(context_length=16))
    with pytest.raises(ValueError):
        model(torch.randint(0, 64, (1, 32)))


def test_the_gpt_matches_the_reference():
    torch.manual_seed(0)
    cfg = small_cfg()
    mine, theirs = ex.GPT(cfg), ref.GPT(cfg)
    copy_parameters(theirs, mine)
    mine.eval()
    theirs.eval()
    idx = torch.randint(0, 64, (2, 8))
    my_logits, my_loss = mine(idx, idx)
    their_logits, their_loss = theirs(idx, idx)
    assert_close(my_logits, their_logits, atol=1e-5, what="the logits")
    assert_scalar_close(my_loss, their_loss, what="the loss")


def test_the_model_can_learn():
    """Overfit a tiny batch. If this does not drop, something is broken."""
    torch.manual_seed(0)
    cfg = small_cfg()
    model = ex.GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    idx = torch.randint(0, 64, (4, 8))

    initial = None
    for _ in range(120):
        _, loss = model(idx, idx)
        initial = initial if initial is not None else float(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    assert float(loss) < initial * 0.2, (
        f"the loss went from {initial:.4f} to {float(loss):.4f} memorizing a single batch: "
        "it should drop much further"
    )


def test_the_repos_yaml_config_produces_the_expected_model():
    """End to end: the config file gives the 8,933,440."""
    cfg = RunConfig.from_yaml(configs_dir() / "tinystories_9m.yaml")
    assert ex.expected_param_count(cfg.model) == 8_933_440
