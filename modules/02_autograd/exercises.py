"""Module 02 - Autodifferentiation from scratch.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement -> `llmfs check 02` -> `llmfs hint 02 -e N` if you get stuck
-> `SOLUTION.md` has the complete code if you need it.

This is the densest module in the course. If something does not fit, do not move on:
everything else rests on this.

WHAT YOU ARE GOING TO BUILD
===========================

The engine that makes `loss.backward()` work. In about 100 lines and without using PyTorch.

    Value               (exercise 1) a number that remembers where it came from
    topological_order   (exercise 2) what order to walk the graph backwards in
    train_scalar_mlp    (exercise 3) train a network using only your engine

By the end, `loss.backward()` will be code you understand line by line.

WHERE EACH THING IS EXPLAINED
=============================

If anything in the code below sounds like it came out of nowhere, it is explained in
`THEORY.md`:

    where the Value class comes from    -> "From the chain rule to a Python class"
    what .data / .grad / ._prev are     -> "The five fields, and why each one is there"
    why there is a `def` inside a `def` -> "The closure: the part that is hardest to see"
    the full trace with numbers         -> "A full trace, with the numbers from the example"
    why exercise 2 is needed at all     -> "The order matters"
    what a neuron and an MLP are        -> "What a neuron is, and what an MLP is"
    where the 6-step loop comes from    -> "The training loop, step by step"

VOCABULARY YOU ARE GOING TO NEED
================================

- **gradient**: the derivative of the loss with respect to a parameter. It says which way to
  move that parameter so the model gets things less wrong.
- **backward / backpropagation**: the algorithm that computes ALL the gradients at once, by
  walking the network backwards.
- **compute graph**: the record of which operations were done and in what order. It is what
  makes walking backwards possible.
- **chain rule**: if y depends on u and u depends on x, then dy/dx = (dy/du) x (du/dx). The
  whole machine rests on this.

    llmfs demo 02     draws the graph inside and compares against torch.autograd
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Sequence

# The MLP is already written: it is not the point of the module. What matters is that it
# accepts `value_cls`, so you can build it with YOUR Value class.
from llmfs.reference import MLP


class Value:
    """A scalar that remembers where it came from. It is the longest exercise in the course.

    If you cannot see where this class comes from, read the section "From the chain rule to a
    Python class" in `THEORY.md` first: it builds these nine methods from the ground up, with
    a field-by-field trace of `d = a * b + 10`.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Nine methods, but SEVEN of them follow the same mould. Learn the mould and the rest is
    copying and changing two lines.

    **Step 1 - `__init__`.** Store four things:

        self.data = float(data)      # the forward value
        self.grad = 0.0              # the gradient, STARTS AT ZERO
        self._prev = tuple(_children)  # what this node depends on
        self._op = _op               # a label, only for debugging
        self._backward = lambda: None  # for now, does nothing

    **Step 2 - the mould for the operations.** They are all written like this:

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)   # (a)
            out = Value(self.data * other.data, (self, other), '*')        # (b)

            def _backward():                                               # (c)
                self.grad  += other.data * out.grad
                other.grad += self.data  * out.grad

            out._backward = _backward                                      # (d)
            return out

        (a) wrap a bare number, so `a * 3` works
        (b) create the result node, telling it who its children are
        (c) the closure: it does NOT run now, it is stored for the backward pass
        (d) it gets attached to the node

    **Step 3 - the local derivatives.** Only step (c) changes in each operation:

        a + b     ->  self.grad += out.grad             ;  other.grad += out.grad
        a * b     ->  self.grad += other.data*out.grad  ;  other.grad += self.data*out.grad
        a ** n    ->  self.grad += n * self.data**(n-1) * out.grad
        exp(a)    ->  self.grad += out.data * out.grad        (e^a is its own derivative)
        log(a)    ->  self.grad += (1/self.data) * out.grad
        tanh(a)   ->  self.grad += (1 - out.data**2) * out.grad
        relu(a)   ->  self.grad += (out.data > 0) * out.grad

    **Step 4 - the sugar.** It needs no new derivatives, it builds on the above:

        -a          ->  return self * -1
        a - b       ->  return self + (-other)
        a / b       ->  return self * other ** -1

    **Step 5 - `backward()`.** Three lines:

        self.grad = 1.0
        for node in reversed(topological_order(self)):
            node._backward()

    THE HARDEST PART TO SEE
    -----------------------
    The closure in step (c) does NOT run when you write it: it is stored. When the backward
    pass runs it, `out.grad` will already hold whatever its parents put there, because the
    closure captures `out` BY REFERENCE.

    You are building a to-do list during the forward pass, and the backward pass executes it
    in reverse order.

    THE MISTAKE TO AVOID: `+=` AND NEVER `=`
    ----------------------------------------
    Try it with `y = x + x`:

        with `=`  : the first branch sets x.grad = 1, the second OVERWRITES it -> 1.  WRONG
        with `+=` : x.grad = 1 + 1 = 2.                                               RIGHT

    It is correct because y = 2x and its derivative is 2. If a variable influences things
    through several paths, its total derivative is the SUM of what each path contributes, and
    in a network that happens constantly.

    And that is where the `optimizer.zero_grad()` you will see in module 11 comes from: since
    gradients accumulate, they have to be zeroed before every step.

    TWO WARNINGS
    ------------
    - The `self.grad = 1.0` in `backward()` is the seed: the derivative of something with
      respect to itself. Without it every gradient comes out 0 and nothing happens.
    - Do not use `__slots__` unless you know exactly what it does.
    - `_prev` can be a tuple or a set of the children.

    The methods `__radd__`, `__rmul__`, `__rsub__` and `__rtruediv__` are already written
    below: they are what makes `2 * a` work (python tries `(2).__mul__(a)`, fails, and calls
    `a.__rmul__(2)`).
    """

    def __init__(
        self,
        data: float,
        _children: Iterable["Value"] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        raise NotImplementedError("TODO: module 02, exercise 1 - Value.__init__")

    def __add__(self, other: Any) -> "Value":
        raise NotImplementedError("TODO: module 02, exercise 1 - Value.__add__")

    def __mul__(self, other: Any) -> "Value":
        raise NotImplementedError("TODO: module 02, exercise 1 - Value.__mul__")

    def __pow__(self, exponent: float) -> "Value":
        raise NotImplementedError("TODO: module 02, exercise 1 - Value.__pow__")

    def exp(self) -> "Value":
        raise NotImplementedError("TODO: module 02, exercise 1 - Value.exp")

    def log(self) -> "Value":
        raise NotImplementedError("TODO: module 02, exercise 1 - Value.log")

    def tanh(self) -> "Value":
        raise NotImplementedError("TODO: module 02, exercise 1 - Value.tanh")

    def relu(self) -> "Value":
        raise NotImplementedError("TODO: module 02, exercise 1 - Value.relu")

    def backward(self) -> None:
        """Propagates the gradient from this node down to the leaves.

        Two steps:
          1. `self.grad = 1.0`  (the derivative of something with respect to itself is 1)
          2. Walk `topological_order(self)` IN REVERSE calling `node._backward()`.
        """
        raise NotImplementedError("TODO: module 02, exercise 1 - Value.backward")

    # -------------------------------------------------- sugar (implement it too)

    def __neg__(self) -> "Value":
        raise NotImplementedError("TODO: module 02, exercise 1 - Value.__neg__")

    def __sub__(self, other: Any) -> "Value":
        raise NotImplementedError("TODO: module 02, exercise 1 - Value.__sub__")

    def __truediv__(self, other: Any) -> "Value":
        raise NotImplementedError("TODO: module 02, exercise 1 - Value.__truediv__")

    def __radd__(self, other: Any) -> "Value":
        return self + other

    def __rmul__(self, other: Any) -> "Value":
        return self * other

    def __rsub__(self, other: Any) -> "Value":
        return (other if isinstance(other, Value) else Value(other)) + (-self)

    def __rtruediv__(self, other: Any) -> "Value":
        return (other if isinstance(other, Value) else Value(other)) * self**-1

    def __repr__(self) -> str:
        return f"Value({getattr(self, 'data', '?')}, grad={getattr(self, 'grad', '?')})"


def topological_order(root: "Value") -> list["Value"]:
    """Orders the graph's nodes so it can be walked backwards.

    The section "The order matters" in `THEORY.md` has the diamond graph where you can see a
    wrongly computed order give 6 instead of 14, without warning you about anything.

    WHAT YOU HAVE TO WRITE
    ----------------------
    A depth-first traversal with an EXPLICIT STACK (not recursive). The trick is a flag
    saying whether you already expanded that node's children.

        order, visited = [], set()
        stack = [(root, False)]          # (node, did I already expand its children?)

        while stack:
            node, expanded = stack.pop()

            if expanded:                  # second visit: its children are already in
                order.append(node)
                continue

            if id(node) in visited:
                continue
            visited.add(id(node))

            stack.append((node, True))    # requeue myself for AFTER my children
            for child in node._prev:
                if id(child) not in visited:
                    stack.append((child, False))

        return order

    HOW THE TRICK WORKS
    -------------------
    Each node goes onto the stack TWICE:
      - the first time with `expanded=False`, to push its children
      - the second time with `expanded=True`, and that one gets processed AFTER all of them,
        because the children were stacked on top

    That is exactly what produces post-order without using recursion.

    WHAT THE RESULT HAS TO SATISFY
    ------------------------------
    **Each node appears AFTER all of its children.** That is, `root` ends up LAST in the
    list, and `backward()` walks it in reverse.

    WHY IT MATTERS
    --------------
    `node._backward()` passes that node's gradient to its children. If you call it before
    every one of that node's PARENTS has contributed its share, you will be passing on an
    incomplete gradient.

    With a tree-shaped graph, any reasonable order works and the bug goes unnoticed. As soon
    as there is a reused node — and in a network there are thousands — a wrongly computed
    order gives silently incorrect gradients.

    TWO WARNINGS
    ------------
    **Iterative, not recursive.** The graph of an MLP with a few hundred neurons already
    exceeds python's recursion limit. There is a test with 3000 nodes that checks it.

    **Use `id(node)` in the visited set, not the node.** If you overload operators on a
    class, relying on its default hash is asking for trouble. `id()` is the object's
    identity, which is exactly what you want.

    If `root` comes out FIRST, you have the order reversed: the symptom will be gradients
    that are right on simple graphs and wrong as soon as there is a reused node.

    Args:
        root: the root node, typically the loss.

    Returns:
        A list of every node reachable from `root`, in topological order.
    """
    raise NotImplementedError("TODO: module 02, exercise 2 - topological_order")


def train_scalar_mlp(
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    hidden: Sequence[int] = (8, 8),
    steps: int = 100,
    lr: float = 0.05,
    seed: int = 0,
    value_cls: type = Value,
) -> list[float]:
    """Trains an MLP with YOUR autodiff engine.

    If you do not know what an MLP is, or what `model(x)`, `parameters()` or `zero_grad()`
    do, read the sections "What a neuron is, and what an MLP is" and "The training loop, step
    by step" in `THEORY.md` first: they count the 113 parameters one by one and walk through a
    full descent step with numbers.

    WHAT YOU HAVE TO WRITE
    ----------------------
    The training loop in its most naked form. Six steps, and the order matters.

        1. Build the model (the MLP class is already written, imported above):

               model = MLP(len(xs[0]), [*hidden, 1], value_cls=value_cls, seed=seed)
               history = []

        2. Repeat `steps` times:

             a. FORWARD - predict for each input:

                    preds = [model(x) for x in xs]

             b. LOSS - the mean squared error:

                    loss = sum(((p - y)**2 for p, y in zip(preds, ys)),
                               value_cls(0.0)) * (1.0 / len(ys))

             c. CLEAR THE GRADIENTS, **before** the backward pass:

                    model.zero_grad()

             d. BACKWARD:

                    loss.backward()

             e. MOVE THE WEIGHTS, against the gradient:

                    for p in model.parameters():
                        p.data -= lr * p.grad

             f. RECORD:

                    history.append(loss.data)

        3. Return `history`.

    STEP 2c IS THE ONE EVERYBODY FORGETS
    ------------------------------------
    Gradients ACCUMULATE (you made them do that in exercise 1). Without zeroing them, step 50
    uses the sum of the gradients from steps 1 to 50.

    And it gives NO error: the loss drops a little at first and then stalls or explodes. It
    is probably the most frustrating bug in all of deep learning, because there is no signal
    pointing at the cause.

    Putting it AFTER the backward pass also works, but only by accident (it leaves them clean
    for the next step). It is fragile: as soon as the loop has a `continue` or a branch, it
    stops being valid.

    TWO PYTHON DETAILS
    ------------------
    **The `value_cls(0.0)` in step 2b.** Python's `sum()` starts accumulating from the
    integer `0`. It would work through `__radd__`, but giving it the right initial value is
    cleaner and more explicit.

    **`p.data -= ...` and not `p -= ...`.** You are modifying the number inside the node, not
    creating a new node. If you created new nodes, the next step's graph would hang off the
    previous one and grow without end until memory ran out. In PyTorch, the equivalent of
    this distinction is the `with torch.no_grad():` block around the optimizer step.

    Args:
        xs: list of input vectors, all of the same length.
        ys: the scalar targets, one per input.
        hidden: the sizes of the hidden layers. The output one (1 neuron) is added for you.
        steps: how many gradient descent steps.
        lr: the learning rate.
        seed: the initialization seed, for reproducibility.
        value_cls: the scalar class to use. Yours by default.

    Returns:
        A list of `steps` losses (floats), one per step.
    """
    raise NotImplementedError("TODO: module 02, exercise 3 - train_scalar_mlp")
