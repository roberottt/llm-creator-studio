"""Module 16 tests. Run them with `llmfs check 16`."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import llmfs.reference as ref
from llmfs.device import set_seed
from llmfs.testing import assert_close, load_exercises

ex = load_exercises(__file__)


# ------------------------------------------------------ exercise 1: build_chat_template


def test_the_example_from_the_docstring():
    msgs = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "How are you"}]
    assert (
        ex.build_chat_template(msgs) == "<|user|>Hello<|end|><|assistant|>How are you<|end|>"
    )


def test_the_generation_prompt_leaves_the_string_open():
    msgs = [{"role": "user", "content": "Hello"}]
    out = ex.build_chat_template(msgs, add_generation_prompt=True)
    assert out == "<|user|>Hello<|end|><|assistant|>"
    assert not out.endswith("<|end|>"), "it has to be left OPEN so the model can continue"


def test_it_accepts_a_system_message():
    msgs = [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hello"}]
    assert ex.build_chat_template(msgs).startswith("<|system|>Be brief<|end|>")


def test_several_turns_are_chained():
    msgs = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
    ]
    out = ex.build_chat_template(msgs)
    assert out.count("<|user|>") == 2 and out.count("<|assistant|>") == 1


def test_every_message_is_closed():
    """The <|end|> is what teaches the model when to stop."""
    msgs = [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]
    assert ex.build_chat_template(msgs).count("<|end|>") == 2


def test_an_unknown_role_is_an_error():
    with pytest.raises(ValueError):
        ex.build_chat_template([{"role": "robot", "content": "beep"}])


def test_an_empty_conversation_gives_an_empty_string():
    assert ex.build_chat_template([]) == ""


def test_the_template_matches_the_reference():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]
    assert ex.build_chat_template(msgs) == ref.build_chat_template(msgs)
    assert ex.build_chat_template(msgs, True) == ref.build_chat_template(msgs, True)


# ------------------------------------------------------- exercise 2: mask_prompt_tokens


def test_the_example_from_the_docstring_with_its_off_by_one():
    """TWO ignored positions at the start with prompt_len=3, not three."""
    assert ex.mask_prompt_tokens([10, 11, 12, 20, 21, 22], 3) == [-100, -100, 20, 21, 22, -100]


def test_the_last_position_of_the_prompt_does_learn():
    """It is the question->answer transition: the most important thing it has to learn."""
    targets = ex.mask_prompt_tokens([10, 11, 12, 20, 21], 3)
    assert targets[2] == 20, (
        "at the last position of the prompt the target is already the first token of the "
        "answer, and THAT one does matter"
    )


def test_the_last_position_is_always_ignored():
    """There is no input_ids[i+1] to predict."""
    assert ex.mask_prompt_tokens([1, 2, 3, 4], 2)[-1] == -100


def test_the_targets_are_shifted_one_token():
    ids = [5, 6, 7, 8, 9]
    targets = ex.mask_prompt_tokens(ids, 1)
    for i in range(len(ids) - 1):
        assert targets[i] == ids[i + 1]


def test_the_length_is_preserved():
    for n in (2, 5, 10):
        assert len(ex.mask_prompt_tokens(list(range(n)), 1)) == n


def test_with_a_prompt_of_one_almost_everything_learns():
    targets = ex.mask_prompt_tokens([1, 2, 3, 4], 1)
    assert targets.count(-100) == 1  # only the last one


def test_it_accepts_another_ignore_index():
    targets = ex.mask_prompt_tokens([1, 2, 3, 4], 2, ignore_index=-1)
    assert targets[0] == -1 and -100 not in targets


def test_an_invalid_prompt_len_is_an_error():
    for bad in (0, -1, 10):
        with pytest.raises(ValueError):
            ex.mask_prompt_tokens([1, 2, 3], bad)


def test_cross_entropy_ignores_the_marked_positions():
    """The check that it does what it is for."""
    logits = torch.randn(5, 10)
    targets = torch.tensor(ex.mask_prompt_tokens([1, 2, 3, 4, 5], 3))
    loss = F.cross_entropy(logits, targets, ignore_index=-100)
    assert torch.isfinite(loss), "-100 has to be ignored, not indexed"


def test_the_mask_matches_the_reference():
    ids = [10, 11, 12, 20, 21, 22, 23]
    assert ex.mask_prompt_tokens(ids, 3) == ref.mask_prompt_tokens(ids, 3)


# ------------------------------------------------------------- exercise 3: LoRALinear


def test_on_initialization_the_output_is_identical_to_the_base():
    """LoRA's property: it starts without perturbing anything, because B starts at zeros."""
    set_seed(0)
    base = nn.Linear(64, 64)
    lora = ex.LoRALinear(base, r=8)
    x = torch.randn(4, 64)
    assert_close(lora(x), base(x), atol=1e-6, what="the output on initialization")


def test_lora_b_starts_at_zeros():
    lora = ex.LoRALinear(nn.Linear(32, 32), r=4)
    assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))


def test_lora_a_does_not_start_at_zeros():
    """If both were zero, the gradient of both would be zero and they would never learn."""
    lora = ex.LoRALinear(nn.Linear(32, 32), r=4)
    assert not torch.allclose(lora.lora_A, torch.zeros_like(lora.lora_A))


def test_the_base_layer_is_frozen():
    lora = ex.LoRALinear(nn.Linear(32, 32), r=4)
    assert not lora.base.weight.requires_grad, (
        "the base has to be frozen: it is the point of LoRA"
    )
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad


def test_the_shapes_of_the_adapters():
    lora = ex.LoRALinear(nn.Linear(64, 128), r=8)
    assert lora.lora_A.shape == (8, 64)
    assert lora.lora_B.shape == (128, 8)


def test_only_a_fraction_of_the_parameters_is_trained():
    lora = ex.LoRALinear(nn.Linear(320, 320), r=8)
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora.parameters())
    assert trainable == 2 * 8 * 320
    assert trainable / total < 0.06, f"it should train ~5%, it trains {trainable / total:.1%}"


def test_the_output_changes_when_b_stops_being_zero():
    set_seed(0)
    base = nn.Linear(32, 32)
    lora = ex.LoRALinear(base, r=4)
    x = torch.randn(2, 32)
    before = lora(x).clone()
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)
    assert not torch.allclose(lora(x), before, atol=1e-4)


def test_the_output_shape_is_the_base_one():
    lora = ex.LoRALinear(nn.Linear(64, 128), r=8)
    assert lora(torch.randn(4, 64)).shape == (4, 128)


def test_a_non_positive_rank_is_an_error():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            ex.LoRALinear(nn.Linear(32, 32), r=bad)


def test_lora_matches_the_reference():
    set_seed(0)
    base = nn.Linear(64, 64)
    mine, theirs = ex.LoRALinear(base, r=8, alpha=16.0), ref.LoRALinear(base, r=8, alpha=16.0)
    with torch.no_grad():
        theirs.lora_A.copy_(mine.lora_A)
        theirs.lora_B.copy_(mine.lora_B)
        mine.lora_B.normal_(0, 0.1)
        theirs.lora_B.copy_(mine.lora_B)
    mine.eval()
    theirs.eval()
    x = torch.randn(4, 64)
    assert_close(mine(x), theirs(x), what="LoRA's output")


# -------------------------------------------------------- exercise 4: merge_lora_weights


def test_the_merged_layer_gives_the_same_output():
    """The property that makes LoRA useful: the adaptation is absorbed without approximating."""
    set_seed(0)
    lora = ex.LoRALinear(nn.Linear(64, 64), r=8)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)
    lora.eval()

    merged = ex.merge_lora_weights(lora)
    x = torch.randn(4, 64)
    assert_close(merged(x), lora(x), atol=1e-5, what="the output of the merged layer")


def test_it_returns_a_normal_linear():
    merged = ex.merge_lora_weights(ex.LoRALinear(nn.Linear(32, 64), r=4))
    assert isinstance(merged, nn.Linear)
    assert not isinstance(merged, ex.LoRALinear)
    assert merged.in_features == 32 and merged.out_features == 64


def test_without_adapting_the_weights_do_not_change():
    """With B at zeros, W_new == W."""
    set_seed(0)
    base = nn.Linear(32, 32)
    merged = ex.merge_lora_weights(ex.LoRALinear(base, r=4))
    assert_close(merged.weight, base.weight, atol=1e-6, what="the unadapted weights")


def test_it_keeps_the_bias():
    base = nn.Linear(32, 32, bias=True)
    merged = ex.merge_lora_weights(ex.LoRALinear(base, r=4))
    assert merged.bias is not None
    assert_close(merged.bias, base.bias, what="the bias")


def test_it_works_without_a_bias():
    merged = ex.merge_lora_weights(ex.LoRALinear(nn.Linear(32, 32, bias=False), r=4))
    assert merged.bias is None


def test_the_merged_weights_are_the_expected_sum():
    set_seed(0)
    base = nn.Linear(32, 32)
    lora = ex.LoRALinear(base, r=4, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)
    expected = base.weight + (lora.lora_B @ lora.lora_A) * (8.0 / 4)
    assert_close(ex.merge_lora_weights(lora).weight, expected, atol=1e-5)


def test_the_merge_matches_the_reference():
    set_seed(0)
    lora = ex.LoRALinear(nn.Linear(64, 64), r=8)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)
    mine = ex.merge_lora_weights(lora)
    theirs = ref.merge_lora_weights(lora)
    assert_close(mine.weight, theirs.weight, what="the merged weights")
