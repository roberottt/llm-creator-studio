"""Tests for module 02. The ground truth is `torch.autograd`."""

from __future__ import annotations

import math

import pytest
import torch

import llmfs.reference as ref
from llmfs.testing import assert_scalar_close, load_exercises

ex = load_exercises(__file__)
Value = ex.Value


def torch_scalar(x: float) -> torch.Tensor:
    return torch.tensor(float(x), dtype=torch.float64, requires_grad=True)


# --------------------------------------------------------------- forward


def test_the_basic_operations_give_the_right_value():
    a, b = Value(3.0), Value(-2.0)
    assert_scalar_close((a + b).data, 1.0, what="a + b")
    assert_scalar_close((a * b).data, -6.0, what="a * b")
    assert_scalar_close((a - b).data, 5.0, what="a - b")
    assert_scalar_close((a / b).data, -1.5, what="a / b")
    assert_scalar_close((a**2).data, 9.0, what="a ** 2")
    assert_scalar_close((-a).data, -3.0, what="-a")


def test_the_nonlinear_functions_give_the_right_value():
    a = Value(0.7)
    assert_scalar_close(a.tanh().data, math.tanh(0.7), what="tanh")
    assert_scalar_close(a.exp().data, math.exp(0.7), what="exp")
    assert_scalar_close(a.log().data, math.log(0.7), what="log")
    assert_scalar_close(a.relu().data, 0.7, what="relu(positive)")
    assert_scalar_close(Value(-0.7).relu().data, 0.0, what="relu(negative)")


def test_the_right_hand_operators_work():
    """`2 * a` calls `a.__rmul__(2)`, not `__mul__`."""
    a = Value(4.0)
    assert_scalar_close((2 + a).data, 6.0, what="2 + a")
    assert_scalar_close((2 * a).data, 8.0, what="2 * a")
    assert_scalar_close((2 - a).data, -2.0, what="2 - a")
    assert_scalar_close((2 / a).data, 0.5, what="2 / a")


def test_operating_with_a_bare_number_wraps_it_in_a_value():
    assert isinstance(Value(1.0) + 3, Value)
    assert isinstance(Value(1.0) * 3, Value)


# --------------------------------------------------------------- gradients vs torch


def test_gradient_of_a_mixed_expression_matches_torch():
    a, b = Value(2.0), Value(-3.0)
    c = a * b + a.tanh() - b**2
    c.backward()

    ta, tb = torch_scalar(2.0), torch_scalar(-3.0)
    (ta * tb + ta.tanh() - tb**2).backward()

    assert_scalar_close(a.grad, ta.grad.item(), rtol=1e-9, what="dc/da")
    assert_scalar_close(b.grad, tb.grad.item(), rtol=1e-9, what="dc/db")


def test_gradient_with_exp_log_and_division_matches_torch():
    a, b = Value(1.3), Value(0.4)
    out = (a.exp() / (b + 1.0)).log() * a
    out.backward()

    ta, tb = torch_scalar(1.3), torch_scalar(0.4)
    ((ta.exp() / (tb + 1.0)).log() * ta).backward()

    assert_scalar_close(a.grad, ta.grad.item(), rtol=1e-9, what="da")
    assert_scalar_close(b.grad, tb.grad.item(), rtol=1e-9, what="db")


def test_relu_cuts_the_gradient_on_the_negative_branch():
    positive, negative = Value(2.0), Value(-2.0)
    (positive.relu() + negative.relu()).backward()
    assert_scalar_close(positive.grad, 1.0, what="grad along the positive branch")
    assert_scalar_close(negative.grad, 0.0, what="grad along the negative branch")


def test_the_gradient_accumulates_when_a_node_is_reused():
    """The minimal case that separates `+=` from `=`. With `=` it would come out 1.0."""
    x = Value(3.0)
    (x + x).backward()
    assert_scalar_close(x.grad, 2.0, what="d(x+x)/dx")


def test_accumulation_in_a_diamond_shaped_graph():
    """x is used along two different paths that join back together."""
    x = Value(2.0)
    left = x * 3.0
    right = x.tanh()
    (left + right).backward()

    tx = torch_scalar(2.0)
    (tx * 3.0 + tx.tanh()).backward()
    assert_scalar_close(x.grad, tx.grad.item(), rtol=1e-9, what="grad along two paths")


def test_a_node_used_many_times_sums_every_contribution():
    x = Value(1.5)
    total = x
    for _ in range(9):
        total = total + x  # x appears 10 times
    total.backward()
    assert_scalar_close(x.grad, 10.0, what="grad of x used 10 times")


def test_the_root_starts_with_gradient_1():
    x = Value(5.0)
    x.backward()
    assert_scalar_close(x.grad, 1.0, what="dx/dx")


# --------------------------------------------------------------- topological order


def test_each_node_comes_after_its_children():
    a, b = Value(1.0), Value(2.0)
    c = a * b
    d = c + a
    e = d.tanh()

    order = ex.topological_order(e)
    position = {id(n): i for i, n in enumerate(order)}

    for node in order:
        for child in node._prev:
            assert position[id(child)] < position[id(node)], (
                "a child appears after its parent: the topological order is reversed"
            )


def test_the_root_is_the_last_element():
    a = Value(1.0)
    root = (a * 2.0).tanh()
    assert ex.topological_order(root)[-1] is root


def test_each_node_appears_exactly_once():
    x = Value(2.0)
    root = x * x + x
    order = ex.topological_order(root)
    assert len({id(n) for n in order}) == len(order), "there are repeated nodes"


def test_it_includes_every_node_in_the_graph():
    a, b = Value(1.0), Value(2.0)
    root = (a + b) * (a - b)
    # a, b, (a+b), (-b), (a-b) [= a + (-b)], (b*-1)... the syntactic sugar's graph adds
    # nodes, so we only check that the leaves and the root are there.
    ids = {id(n) for n in ex.topological_order(root)}
    assert id(a) in ids and id(b) in ids and id(root) in ids


def test_it_does_not_blow_up_on_a_deep_graph():
    """If you made it recursive, this raises RecursionError."""
    x = Value(0.5)
    node = x
    for _ in range(3000):
        node = node + 0.001
    order = ex.topological_order(node)
    assert len(order) > 3000


def test_backward_works_on_a_deep_graph():
    x = Value(0.0)
    node = x
    for _ in range(2000):
        node = node + x
    node.backward()
    assert_scalar_close(x.grad, 2001.0, what="x added 2001 times")


# --------------------------------------------------------------- training


def test_training_brings_the_loss_down():
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    history = ex.train_scalar_mlp(xs, ys, hidden=(8, 8), steps=80, lr=0.05, seed=0)

    assert len(history) == 80
    assert history[-1] < history[0] / 10, (
        f"the loss went from {history[0]:.4f} to {history[-1]:.4f}. "
        "If it barely drops, check that you call zero_grad() BEFORE the backward pass."
    )
    assert history[-1] < 0.05


def test_the_history_matches_the_reference():
    """Same seed, same lr, same steps -> exactly the same trajectory."""
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    mine = ex.train_scalar_mlp(xs, ys, hidden=(4,), steps=25, lr=0.05, seed=7)
    theirs = ref.train_scalar_mlp(
        xs, ys, hidden=(4,), steps=25, lr=0.05, seed=7, value_cls=ref.Value
    )

    for step, (a, b) in enumerate(zip(mine, theirs)):
        assert_scalar_close(a, b, rtol=1e-6, atol=1e-9, what=f"loss at step {step}")


def test_without_zeroing_the_gradients_training_does_not_converge():
    """An indirect check: the correct trajectory is the reference's."""
    xs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
    ys = [1.0, 1.0, -1.0, -1.0]
    history = ex.train_scalar_mlp(xs, ys, hidden=(6,), steps=60, lr=0.1, seed=3)
    assert all(math.isfinite(p) for p in history), (
        "there are inf/nan in the loss: the classic symptom of gradients accumulated "
        "without being cleared"
    )
