# 02 — Annotated solution

## Exercise 1 — The `Value` class

### The mould

Every operation has the same shape. If you understand one, you understand all seven:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)   # 1
    out = Value(self.data * other.data, (self, other), '*')       # 2

    def _backward():                                              # 3
        self.grad  += other.data * out.grad
        other.grad += self.data  * out.grad

    out._backward = _backward                                     # 4
    return out
```

1. **Wrap the bare number.** That way `a * 3` works without writing `a * Value(3)`.
2. **Create the result node**, telling it who its children are. That tuple is what makes it
   possible to walk the graph later.
3. **The closure.** This is the only conceptually new thing. It does not run now: it is
   stored for later. When it does run, `out.grad` will already hold whatever its parents put
   there, because the closure captures `out` **by reference**, not by value.
4. **Attach it to the node**, so `backward()` can call it.

That step 3 does not run at the time is what makes all of this work, and it is the part that
is hardest to see the first time. You are building a to-do list while doing the forward pass,
and the backward pass executes it in reverse order.

### The derivatives, one by one

| operation | local derivative | why |
|---|---|---|
| `a + b` | `1` for both | raising `a` by 1 raises the sum by 1 |
| `a * b` | `b` for `a`, `a` for `b` | raising `a` by 1 raises the product by `b` |
| `a ** n` | `n * a^(n-1)` | the usual rule |
| `exp(a)` | `out.data` | the exponential is its own derivative, and you already have it |
| `log(a)` | `1 / a.data` | |
| `tanh(a)` | `1 - out.data²` | a known identity, and it reuses the forward value |
| `relu(a)` | `1` if it came out positive, `0` otherwise | |

Two of them — `exp` and `tanh` — use `out.data` instead of recomputing. It is not just for
speed: it is the same trick PyTorch uses, storing in the forward pass what will be needed in
the backward.

About `relu`: at exactly 0 the derivative does not exist (there is a kink). By convention we
take 0. It does not matter which you pick: the probability of landing exactly on 0 with
floats is negligible.

### The syntactic sugar

No new derivatives need writing. It all builds on the previous ones:

```python
-a       ->  a * -1
a - b    ->  a + (-b)
a / b    ->  a * b**-1
```

And the `__r*__` versions are for when the `Value` is on the right. When Python evaluates
`2 * a`, it first tries `(2).__mul__(a)`, which returns `NotImplemented` because `int` does
not know what to do with a `Value`. Then it tries `a.__rmul__(2)`. That is why `__radd__` and
`__rmul__` can delegate directly (addition and multiplication are commutative) but
`__rsub__` and `__rtruediv__` cannot: `2 - a` is not `a - 2`.

### `backward()`

```python
def backward(self):
    self.grad = 1.0
    for node in reversed(topological_order(self)):
        node._backward()
```

Three lines. The `self.grad = 1.0` is the seed: the derivative of the loss with respect to
itself. Without it every gradient would be 0 and nothing would happen.

## Exercise 2 — `topological_order`

Post-order DFS. The recursive version is five lines and **does not work**: with a few
hundred neurons it blows up with `RecursionError`. Iteratively:

```python
order, visited = [], set()
stack = [(root, False)]          # (node, did I already expand its children?)

while stack:
    node, expanded = stack.pop()
    if expanded:                  # second visit: all its children are in
        order.append(node)
        continue
    if id(node) in visited:
        continue
    visited.add(id(node))
    stack.append((node, True))    # requeue myself for after my children
    for child in node._prev:
        if id(child) not in visited:
            stack.append((child, False))
```

The trick is the flag. Each node goes onto the stack twice: the first time to expand its
children, the second — processed **after** all of them — to append itself to the result.
That is exactly what produces post-order without recursion.

**Use `id(node)` and not `node` in the visited set.** If you overload operators on the
class, relying on the default hash or equality is asking for trouble. `id()` is the object's
identity and it is exactly what you want here.

**The result has `root` at the end**, and `backward()` walks it in reverse. If it comes out
at the start, you have the order reversed: the symptom will be gradients that come out wrong
as soon as there is a reused node, but *right* on simple graphs. A test that passes with a
tree and fails with a diamond is almost always this.

## Exercise 3 — `train_scalar_mlp`

```python
model = MLP(len(xs[0]), [*hidden, 1], value_cls=value_cls, seed=seed)
history = []

for _ in range(steps):
    preds = [model(x) for x in xs]
    loss = sum(((p - y)**2 for p, y in zip(preds, ys)), value_cls(0.0)) * (1.0/len(ys))

    model.zero_grad()      # BEFORE the backward
    loss.backward()

    for p in model.parameters():
        p.data -= lr * p.grad

    history.append(loss.data)
```

**The `value_cls(0.0)` in the `sum()`.** Python's `sum()` starts accumulating from the
integer `0`. The first addition would be `0 + Value(...)`, which works through `__radd__`,
but giving it the right initial value is cleaner and more explicit.

**The order of `zero_grad()` and `backward()`.** Gradients accumulate (exercise 1). If you do
not clear them before each backward pass, step 50 uses the sum of the gradients from steps 1
to 50. The symptom is that the loss drops a little at first and then stalls or explodes, and
**there is no error message**. It is probably the most frequent bug in all of deep learning.

Putting it *after* the `backward()` instead of before also works, but only by accident (it
leaves them clean for the next step). It is fragile: as soon as the loop has a `continue` or
a branch, it stops being valid. Put it before.

**`p.data -= lr * p.grad`, not `p -= lr * p.grad`.** You are modifying the number inside the
node, not creating a new node. If you created new nodes, the next step's graph would hang off
the previous one and grow without end until memory ran out. In PyTorch, the equivalent of
this distinction is the `with torch.no_grad():` block around the optimizer step.

## What you should see in the demo

Your engine's gradients identical to `torch.autograd`'s down to the last decimal. Not
similar: **identical**, because you are doing literally the same operations in the same
order.

And the two loss curves with different `lr`. The lesson from that plot is the one that comes
back in module 11: the learning rate is the hyperparameter that ruins the most training
runs, and there is no way to know the right one without trying.
