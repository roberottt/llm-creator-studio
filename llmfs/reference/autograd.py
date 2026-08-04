"""Reference for module 02: a scalar autodifferentiation engine.

A `Value` is a number that also remembers where it came from. Chaining operations together
builds a directed acyclic graph; `backward()` walks it in reverse topological order
applying the chain rule.

It is exactly what PyTorch does, only PyTorch does it over tensors and in C++.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Sequence


class Value:
    """A scalar with a gradient.

    Attributes:
        data: the forward value.
        grad: derivative of the final output with respect to this node. It accumulates.
        _prev: the nodes this one depends on.
        _op: label for the operation, only for debugging and drawing the graph.
        _backward: closure that propagates this node's gradient down to its children.
    """

    __slots__ = ("data", "grad", "_prev", "_op", "_backward", "label")

    def __init__(
        self,
        data: float,
        _children: Iterable["Value"] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self.label = label
        self._backward: Callable[[], None] = lambda: None

    # ------------------------------------------------------------------ operations

    def __add__(self, other: Any) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Any) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> "Value":
        if not isinstance(exponent, (int, float)):
            raise TypeError("only constant int or float exponents are supported")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            # d(a^n)/da = n * a^(n-1)
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            # d(e^a)/da = e^a, which is exactly out.data
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            # d(tanh)/dx = 1 - tanh^2
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward() -> None:
            # The derivative at 0 does not exist; by convention we take 0.
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ sugar

    def __neg__(self) -> "Value":
        return self * -1

    def __radd__(self, other: Any) -> "Value":
        return self + other

    def __sub__(self, other: Any) -> "Value":
        return self + (-(other if isinstance(other, Value) else Value(other)))

    def __rsub__(self, other: Any) -> "Value":
        return (other if isinstance(other, Value) else Value(other)) + (-self)

    def __rmul__(self, other: Any) -> "Value":
        return self * other

    def __truediv__(self, other: Any) -> "Value":
        return self * (other if isinstance(other, Value) else Value(other)) ** -1

    def __rtruediv__(self, other: Any) -> "Value":
        return (other if isinstance(other, Value) else Value(other)) * self**-1

    def __repr__(self) -> str:
        label = f" {self.label}" if self.label else ""
        return f"Value({self.data:.6g}, grad={self.grad:.6g}{label})"

    # ------------------------------------------------------------------ backward

    def backward(self) -> None:
        """Propagate the gradient from this node down to the leaves.

        Two details that matter:

        1. It starts with `self.grad = 1.0`, because the derivative of the output with
           respect to itself is 1.
        2. You have to walk in reverse topological order. If you propagated a node's
           gradient before ALL of its parents had contributed theirs, that node would send
           an incomplete gradient downwards. On a tree-shaped graph you would not notice;
           as soon as there is a reused node (and in a neural network there are thousands),
           it gives wrong results.
        """
        self.grad = 1.0
        for node in reversed(topological_order(self)):
            node._backward()


def topological_order(root: Value) -> list[Value]:
    """Topological order of the graph: each node appears after all of its children.

    Iterative post-order DFS. Iterative rather than recursive because the graph of an MLP
    with a few hundred neurons already gets close to python's recursion limit.
    """
    order: list[Value] = []
    visited: set[int] = set()
    # (node, children_already_expanded)
    stack: list[tuple[Value, bool]] = [(root, False)]

    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))
        for child in node._prev:
            if id(child) not in visited:
                stack.append((child, False))

    return order


# ---------------------------------------------------------------------------- neural net


class Neuron:
    """A neuron: `act(w . x + b)`."""

    def __init__(self, nin: int, nonlin: bool = True, value_cls: type = Value, rng: Any = None) -> None:
        rng = rng or random.Random(0)
        self.w = [value_cls(rng.uniform(-1, 1)) for _ in range(nin)]
        self.b = value_cls(0.0)
        self.nonlin = nonlin

    def __call__(self, x: Sequence[Any]) -> Any:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self) -> list[Any]:
        return [*self.w, self.b]


class Layer:
    def __init__(self, nin: int, nout: int, value_cls: type = Value, rng: Any = None, **kw: Any) -> None:
        self.neurons = [Neuron(nin, value_cls=value_cls, rng=rng, **kw) for _ in range(nout)]

    def __call__(self, x: Sequence[Any]) -> Any:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Any]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Multilayer perceptron built on top of `value_cls`.

    `value_cls` is a parameter so you can build it with YOUR `Value` class without touching
    this code. That is what the `train_scalar_mlp` exercise does.
    """

    def __init__(
        self,
        nin: int,
        nouts: Sequence[int],
        value_cls: type = Value,
        seed: int = 0,
    ) -> None:
        rng = random.Random(seed)
        sizes = [nin, *nouts]
        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                value_cls=value_cls,
                rng=rng,
                nonlin=i < len(nouts) - 1,  # the last layer is linear
            )
            for i in range(len(nouts))
        ]

    def __call__(self, x: Sequence[Any]) -> Any:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Any]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0


def train_scalar_mlp(
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    hidden: Sequence[int] = (8, 8),
    steps: int = 100,
    lr: float = 0.05,
    seed: int = 0,
    value_cls: type = Value,
) -> list[float]:
    """Train an MLP with gradient descent and return the loss history.

    Loss: mean squared error. The loop is the same one you will use in module 10 with
    PyTorch, and it is worth seeing it here with no abstraction in the way.

    Returns:
        A list of `steps` losses, one per step.
    """
    model = MLP(len(xs[0]), [*hidden, 1], value_cls=value_cls, seed=seed)
    history: list[float] = []

    for _ in range(steps):
        # forward: predictions and loss
        preds = [model(x) for x in xs]
        loss = sum(((p - y) ** 2 for p, y in zip(preds, ys)), value_cls(0.0)) * (1.0 / len(ys))

        # backward: ALWAYS zero the gradients first, because they accumulate.
        # Forgetting this is the number one beginner bug.
        model.zero_grad()
        loss.backward()

        # descent: one step against the gradient
        for p in model.parameters():
            p.data -= lr * p.grad

        history.append(loss.data)

    return history
