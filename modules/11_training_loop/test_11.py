"""Tests for module 11. Run them with `llmfs check 11`.

The external oracles are `torch.optim.AdamW` and `torch.nn.utils.clip_grad_norm_`.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import llmfs.reference as ref
from llmfs.config import ModelConfig
from llmfs.testing import assert_close, assert_scalar_close, load_exercises

ex = load_exercises(__file__)


def small_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))


def train(opt_class, steps: int = 50, **kwargs) -> tuple[list[float], list[torch.Tensor]]:
    """Trains a small model with the given optimizer and returns losses and final weights."""
    model = small_model()
    opt = opt_class(model.parameters(), **kwargs)
    torch.manual_seed(1)
    x, y = torch.randn(32, 8), torch.randn(32, 1)

    history = []
    for _ in range(steps):
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return history, [p.detach().clone() for p in model.parameters()]


# ---------------------------------------------------------- exercise 1: AdamWScratch


def test_adamw_matches_torchs():
    """The external oracle. Same weights after 50 steps, to fp32 precision."""
    kwargs = dict(lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    _, my_weights = train(ex.AdamWScratch, **kwargs)
    _, torch_weights = train(torch.optim.AdamW, **kwargs)

    for i, (mine, theirs) in enumerate(zip(my_weights, torch_weights)):
        assert_close(mine, theirs, atol=1e-5, what=f"parameter {i} after 50 steps")


def test_adamw_matches_without_weight_decay():
    kwargs = dict(lr=1e-2, betas=(0.9, 0.999), weight_decay=0.0)
    _, mine = train(ex.AdamWScratch, **kwargs)
    _, theirs = train(torch.optim.AdamW, **kwargs)
    for a, b in zip(mine, theirs):
        assert_close(a, b, atol=1e-5, what="the weights without weight decay")


def test_adamw_brings_the_loss_down():
    history, _ = train(ex.AdamWScratch, steps=100, lr=1e-2)
    assert history[-1] < history[0] / 2, (
        f"the loss went from {history[0]:.4f} to {history[-1]:.4f}"
    )


def test_the_bias_correction_is_applied():
    """On the first step, the jump size has to be ~lr, not lr/sqrt(1-beta2).

    Without the bias correction, v is (1-beta2)*g^2 after one step, and dividing by its
    square root gives a step 1/sqrt(1-beta2) times larger: with beta2=0.95, about 4.5 times.
    """
    torch.manual_seed(0)
    p = nn.Parameter(torch.zeros(100))
    opt = ex.AdamWScratch([p], lr=0.1, betas=(0.9, 0.95), weight_decay=0.0)
    p.grad = torch.ones(100)
    opt.step()

    jump = float(p.detach().abs().mean())
    assert abs(jump - 0.1) < 0.01, (
        f"the first step moves the parameter {jump:.4f} and it should move it ~0.1 (the lr). "
        f"If it comes out ~{0.1 / math.sqrt(0.05):.2f}, the bias correction is missing."
    )


def test_the_weight_decay_is_decoupled():
    """AdamW (decay on the parameter) against Adam+L2 (decay added to the gradient).

    With ZERO gradient, AdamW still shrinks the parameter; Adam+L2 would not touch it because
    the gradient reaching Adam would be wd*p and it would go through the division by sqrt(v).
    """
    p = nn.Parameter(torch.full((10,), 2.0))
    opt = ex.AdamWScratch([p], lr=0.1, weight_decay=0.5)
    p.grad = torch.zeros(10)
    opt.step()

    value = float(p.detach()[0])
    expected = 2.0 * (1 - 0.1 * 0.5)
    assert abs(value - expected) < 1e-4, (
        f"with a zero gradient and wd=0.5, the parameter should go from 2.0 to {expected} "
        f"(decoupled decay), and it is {value:.4f}"
    )


def test_the_groups_have_their_own_weight_decay():
    a = nn.Parameter(torch.full((5,), 1.0))
    b = nn.Parameter(torch.full((5,), 1.0))
    opt = ex.AdamWScratch(
        [{"params": [a], "weight_decay": 0.5}, {"params": [b], "weight_decay": 0.0}],
        lr=0.1,
    )
    a.grad, b.grad = torch.zeros(5), torch.zeros(5)
    opt.step()
    assert float(a[0]) < 1.0, "the group with decay should have shrunk"
    assert_scalar_close(float(b[0]), 1.0, what="the group without decay")


def test_adamw_ignores_parameters_with_no_gradient():
    a = nn.Parameter(torch.ones(5))
    b = nn.Parameter(torch.ones(5))
    opt = ex.AdamWScratch([a, b], lr=0.1)
    a.grad = torch.ones(5)  # b is left with no gradient
    opt.step()
    assert_scalar_close(float(b[0]), 1.0, what="the parameter with no gradient")


def test_the_state_can_be_saved_and_restored():
    """Essential for resuming a training run without the model lurching."""
    model = small_model()
    opt = ex.AdamWScratch(model.parameters(), lr=1e-2)
    x, y = torch.randn(8, 8), torch.randn(8, 1)
    for _ in range(5):
        ((model(x) - y) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()

    state = opt.state_dict()
    opt2 = ex.AdamWScratch(model.parameters(), lr=1e-2)
    opt2.load_state_dict(state)

    some_state = next(iter(opt2.state.values()))
    assert "exp_avg" in some_state or len(some_state) > 0


def test_it_validates_the_hyperparameters():
    p = [nn.Parameter(torch.ones(2))]
    with pytest.raises(ValueError):
        ex.AdamWScratch(p, lr=-1.0)
    with pytest.raises(ValueError):
        ex.AdamWScratch(p, betas=(1.5, 0.9))


def test_adamw_matches_the_reference():
    kwargs = dict(lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    _, mine = train(ex.AdamWScratch, **kwargs)
    _, theirs = train(ref.AdamWScratch, **kwargs)
    for a, b in zip(mine, theirs):
        assert_close(a, b, what="the weights")


# -------------------------------------------------------------- exercise 2: lr_at_step


def test_the_warmup_rises_linearly():
    lrs = [ex.lr_at_step(s, 1000, 1e-3, warmup_steps=100) for s in range(100)]
    assert lrs[0] < lrs[50] < lrs[99]
    assert_scalar_close(lrs[99], 1e-3, rtol=1e-9, what="the lr at the end of the warmup")
    # Linear: the difference between consecutive steps is constant
    diffs = [b - a for a, b in zip(lrs, lrs[1:])]
    assert max(diffs) - min(diffs) < 1e-12


def test_step_zero_does_not_have_zero_lr():
    """A step with lr=0 learns nothing: it is a wasted step."""
    assert ex.lr_at_step(0, 1000, 1e-3, warmup_steps=100) > 0


def test_right_after_the_warmup_it_is_at_the_maximum():
    assert_scalar_close(
        ex.lr_at_step(500, 10000, 1e-3, warmup_steps=500),
        1e-3,
        rtol=1e-6,
        what="the lr at the peak",
    )


def test_the_cosine_decays_to_the_floor():
    lr, ratio = 1e-3, 0.1
    final = ex.lr_at_step(10000, 10000, lr, warmup_steps=500, min_lr_ratio=ratio)
    assert_scalar_close(final, lr * ratio, rtol=1e-6, what="the lr at the end")


def test_it_never_goes_below_the_floor():
    for s in (10_000, 15_000, 1_000_000):
        assert ex.lr_at_step(s, 10_000, 1e-3, 500, 0.1) >= 1e-4 - 1e-12


def test_the_cosine_is_monotonically_decreasing():
    lrs = [ex.lr_at_step(s, 10_000, 1e-3, 500, 0.1) for s in range(500, 10_000, 100)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "the cosine cannot rise"


def test_halfway_it_is_at_half_height():
    """cos(pi/2) = 0, so coef = 0.5 and the lr is the average of lr and min_lr."""
    lr, ratio = 1e-3, 0.1
    middle = ex.lr_at_step(5000, 10000, lr, warmup_steps=0, min_lr_ratio=ratio)
    expected = lr * ratio + (lr - lr * ratio) * 0.5
    assert_scalar_close(middle, expected, rtol=1e-6, what="the lr halfway through training")


def test_without_warmup_it_starts_at_the_maximum():
    assert_scalar_close(
        ex.lr_at_step(0, 1000, 1e-3, warmup_steps=0),
        1e-3,
        rtol=1e-6,
        what="the lr without warmup",
    )


def test_the_constant_schedule():
    for s in (0, 500, 5000):
        assert_scalar_close(
            ex.lr_at_step(s, 10000, 1e-3, warmup_steps=0, schedule="constant"), 1e-3
        )


def test_the_linear_schedule():
    lr = 1e-3
    middle = ex.lr_at_step(5000, 10000, lr, warmup_steps=0, min_lr_ratio=0.0, schedule="linear")
    assert_scalar_close(middle, lr * 0.5, rtol=1e-6, what="the linear lr halfway")


@pytest.mark.parametrize("step", [0, 1, 250, 499, 500, 501, 2500, 9999, 10172, 20000])
def test_it_matches_the_reference_in_every_segment(step):
    assert_scalar_close(
        ex.lr_at_step(step, 10172, 1e-3, 500, 0.1),
        ref.lr_at_step(step, 10172, 1e-3, 500, 0.1),
        what=f"the lr at step {step}",
    )


# --------------------------------------------------------- exercise 3: clip_grad_norm


def with_gradients(scale: float = 1.0, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    model = nn.Linear(16, 16)
    ((model(torch.randn(4, 16)) ** 2).sum() * scale).backward()
    return model


def test_it_matches_torchs_clip():
    """The external oracle."""
    mine = with_gradients(100.0)
    theirs = with_gradients(100.0)

    my_norm = ex.clip_grad_norm(mine.parameters(), 1.0)
    their_norm = float(torch.nn.utils.clip_grad_norm_(list(theirs.parameters()), 1.0))

    assert_scalar_close(my_norm, their_norm, rtol=1e-5, what="the returned norm")
    for a, b in zip(mine.parameters(), theirs.parameters()):
        assert_close(a.grad, b.grad, atol=1e-6, what="the clipped gradients")


def test_it_returns_the_norm_before_clipping():
    model = with_gradients(100.0)
    expected = float(
        torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters()))
    )
    returned = ex.clip_grad_norm(model.parameters(), 1.0)
    assert_scalar_close(returned, expected, rtol=1e-5, what="the norm BEFORE clipping")
    assert returned > 1.0, "the test case should exceed the threshold"


def test_the_final_norm_is_the_threshold():
    model = with_gradients(100.0)
    ex.clip_grad_norm(model.parameters(), 1.0)
    final = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters())))
    assert_scalar_close(final, 1.0, rtol=1e-3, what="the norm after clipping")


def test_it_touches_nothing_if_the_norm_is_below():
    model = with_gradients(0.001)
    before = [p.grad.clone() for p in model.parameters()]
    ex.clip_grad_norm(model.parameters(), 1000.0)
    for a, p in zip(before, model.parameters()):
        assert_close(p.grad, a, what="the unclipped gradients")


def test_it_preserves_the_gradients_direction():
    """What distinguishes global clipping from per-tensor clipping."""
    model = with_gradients(100.0)
    before = torch.cat([p.grad.flatten().clone() for p in model.parameters()])
    ex.clip_grad_norm(model.parameters(), 1.0)
    after = torch.cat([p.grad.flatten() for p in model.parameters()])

    cosine = float(
        torch.dot(before, after) / (before.norm() * after.norm())
    )
    assert cosine > 0.9999, (
        f"the direction has changed (cosine {cosine:.6f}): you are clipping per tensor "
        "instead of with the global norm"
    )


def test_with_no_gradients_it_returns_zero():
    model = nn.Linear(4, 4)  # no backward, every grad is None
    assert ex.clip_grad_norm(model.parameters(), 1.0) == 0.0


def test_the_clip_ignores_parameters_with_no_gradient():
    a = nn.Parameter(torch.ones(4))
    b = nn.Parameter(torch.ones(4))
    a.grad = torch.full((4,), 3.0)
    norm = ex.clip_grad_norm([a, b], 100.0)
    assert_scalar_close(norm, 6.0, rtol=1e-5, what="the norm of a single tensor")


def test_the_clip_matches_the_reference():
    mine, theirs = with_gradients(50.0), with_gradients(50.0)
    assert_scalar_close(
        ex.clip_grad_norm(mine.parameters(), 1.0),
        ref.clip_grad_norm(theirs.parameters(), 1.0),
        what="the norm",
    )


# ------------------------------------------------------ exercise 4: build_param_groups


def test_there_are_exactly_two_groups():
    groups = ex.build_param_groups(ref.GPT(ModelConfig()), 0.1)
    assert len(groups) == 2


def test_the_first_group_has_decay_and_the_second_does_not():
    groups = ex.build_param_groups(ref.GPT(ModelConfig()), 0.1)
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0


def test_the_matrices_go_with_decay_and_the_vectors_without():
    groups = ex.build_param_groups(ref.GPT(ModelConfig()), 0.1)
    assert all(p.dim() >= 2 for p in groups[0]["params"]), (
        "the decay group should only contain parameters with 2+ dimensions"
    )
    assert all(p.dim() < 2 for p in groups[1]["params"]), (
        "the no-decay group should only contain 1-dimensional parameters"
    )


def test_no_parameter_is_lost_or_duplicated():
    model = ref.GPT(ModelConfig())
    groups = ex.build_param_groups(model, 0.1)
    total = sum(p.numel() for g in groups for p in g["params"])
    assert total == 8_933_440
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "there are parameters repeated across the groups"


def test_the_final_models_norms_are_left_without_decay():
    """4,160 parameters: 13 RMSNorms x 320. All of them have to stay out of the decay."""
    groups = ex.build_param_groups(ref.GPT(ModelConfig()), 0.1)
    assert sum(p.numel() for p in groups[1]["params"]) == 4_160


def test_it_skips_frozen_parameters():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    model[0].weight.requires_grad_(False)
    groups = ex.build_param_groups(model, 0.1)
    # By identity: `x in list_of_tensors` uses `==`, which on tensors returns an elementwise
    # comparison and blows up with "Boolean value ... is ambiguous".
    ids = {id(p) for g in groups for p in g["params"]}
    assert id(model[0].weight) not in ids
    assert id(model[1].weight) in ids


def test_the_format_is_accepted_by_torchs_optimizer():
    model = ref.GPT(ModelConfig(n_layers=1, vocab_size=64, d_model=32, n_heads=4, d_ff=64))
    opt = torch.optim.AdamW(ex.build_param_groups(model, 0.1), lr=1e-3)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 0.1


def test_the_groups_match_the_reference():
    model = ref.GPT(ModelConfig())
    mine = ex.build_param_groups(model, 0.1)
    theirs = ref.build_param_groups(model, 0.1)
    for a, b in zip(mine, theirs):
        assert [id(p) for p in a["params"]] == [id(p) for p in b["params"]]
        assert a["weight_decay"] == b["weight_decay"]
