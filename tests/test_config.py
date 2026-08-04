"""The configuration has to validate early and add up to the 9M model."""

from __future__ import annotations

import pytest
import yaml

from llmfs.config import ModelConfig, RunConfig, TrainConfig


def test_the_defaults_are_the_final_model():
    m = ModelConfig()
    assert (m.vocab_size, m.n_layers, m.d_model, m.n_heads) == (4096, 6, 320, 8)
    assert m.head_dim == 40
    assert m.d_ff == 896
    assert m.context_length == 512
    assert m.tie_embeddings is True


def test_d_model_must_divide_by_the_heads():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=320, n_heads=7)


def test_rope_requires_an_even_head_dim():
    with pytest.raises(ValueError, match="even"):
        ModelConfig(d_model=15, n_heads=5, pos="rope")


def test_dropout_out_of_range():
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=1.5)


def test_betas_must_have_two_elements():
    with pytest.raises(ValueError, match="betas"):
        TrainConfig(betas=(0.9, 0.95, 0.99))  # type: ignore[arg-type]


def test_flat_and_sectioned_yaml_are_equivalent():
    flat = {"n_layers": 4, "d_model": 128, "n_heads": 4, "lr": 3.0e-4, "batch_size": 16}
    nested = {
        "model": {"n_layers": 4, "d_model": 128, "n_heads": 4},
        "train": {"lr": 3.0e-4, "batch_size": 16},
    }
    assert RunConfig.from_dict(flat) == RunConfig.from_dict(nested)


def test_an_unknown_key_blows_up_at_startup():
    """A typo in the config cannot be discovered three hours into the run."""
    with pytest.raises(ValueError) as exc:
        RunConfig.from_dict({"n_layerz": 6})
    assert "n_layerz" in str(exc.value)
    assert "n_layers" in str(exc.value), "the error should list the valid keys"


def test_yaml_accepts_underscores_in_integers():
    """`max_tokens: 500_000_000` has to parse as an int, not as a string."""
    loaded = yaml.safe_load("max_tokens: 500_000_000\n")
    assert loaded["max_tokens"] == 500_000_000


def test_the_arithmetic_of_steps_and_tokens():
    cfg = RunConfig()
    assert cfg.tokens_per_step == 48 * 2 * 512 == 49_152
    assert cfg.max_steps == 500_000_000 // 49_152 == 10_172


def test_yaml_roundtrip(tmp_path):
    cfg = RunConfig(name="test")
    target = tmp_path / "cfg.yaml"
    cfg.to_yaml(target)
    assert RunConfig.from_yaml(target) == cfg


def test_the_name_comes_from_the_file_when_not_given(tmp_path):
    target = tmp_path / "tinystories_9m.yaml"
    target.write_text("n_layers: 6\n", encoding="utf-8")
    assert RunConfig.from_yaml(target).name == "tinystories_9m"


def test_character_tokenizer_with_an_absurd_vocabulary():
    with pytest.raises(ValueError, match="character"):
        RunConfig.from_dict({"tokenizer": "char", "vocab_size": 4096})


def test_the_repos_real_configs_load():
    from llmfs.paths import configs_dir

    found = sorted(p.name for p in configs_dir().glob("*.yaml"))
    assert found == ["tiny_char.yaml", "tinystories_9m.yaml"]
    for path in configs_dir().glob("*.yaml"):
        RunConfig.from_yaml(path)


def test_the_9m_config_matches_the_documented_breakdown():
    """The count module 10 promises. If this changes, THEORY.md is lying."""
    from llmfs.paths import configs_dir

    cfg = RunConfig.from_yaml(configs_dir() / "tinystories_9m.yaml")
    m = cfg.model

    embeddings = m.vocab_size * m.d_model
    attention = 4 * m.d_model**2
    swiglu = 3 * m.d_model * m.d_ff
    norms = 2 * m.d_model
    per_layer = attention + swiglu + norms
    total = embeddings + m.n_layers * per_layer + m.d_model  # + the final rmsnorm

    assert embeddings == 1_310_720
    assert attention == 409_600
    assert swiglu == 860_160
    assert per_layer == 1_270_400
    assert total == 8_933_440
    assert m.tie_embeddings, "without tying there would be 1.3M more parameters in lm_head"


def test_d_ff_is_the_right_multiple_of_64_in_both_configs():
    """SwiGLU uses 2/3 of the hidden size to make up for having 3 matrices instead of 2."""
    from llmfs.paths import configs_dir

    for name in ("tiny_char.yaml", "tinystories_9m.yaml"):
        m = RunConfig.from_yaml(configs_dir() / name).model
        target = 2 / 3 * 4 * m.d_model
        expected = 64 * -(-int(target) // 64)  # ceil to a multiple of 64
        assert m.d_ff == expected, f"{name}: d_ff={m.d_ff}, {expected} was expected"


def test_the_toy_model_trains_in_seconds_not_hours():
    from llmfs.paths import configs_dir

    cfg = RunConfig.from_yaml(configs_dir() / "tiny_char.yaml")
    assert cfg.max_steps < 5_000, "the toy config has to be genuinely a toy"
    assert cfg.model.dropout > 0, "1MB of text without dropout gets memorized"


def test_the_summary_mentions_what_matters():
    text = RunConfig(name="x").summary()
    for expected in ("6L x 320d x 8h", "head_dim=40", "rmsnorm", "rope", "swiglu"):
        assert expected in text
