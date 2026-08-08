# 02 — Autodifferentiation from scratch

## Why this module matters

**So that `loss.backward()` stops being magic.**

That line is what makes a network learn. It computes, in one go, how each of your model's
8.9 million parameters has to move so it gets things less wrong. And almost everyone who
writes it has no idea what it does inside.

In this module you write it yourself, in about 100 lines and without using PyTorch. By the
end, when your training does not converge, you will know where to look instead of trying
things at random.

It is the most "mathematical" module in the course, and also the one that will help you most
in debugging everything else.

### What you will know by the end

- What a gradient is and why the network needs them
- How **all** the gradients get computed at once, in the time of two forward passes
- Why `optimizer.zero_grad()` exists and what exactly happens if you forget it
- What PyTorch does inside when you call `.backward()`
- What a neuron is, what an MLP is, and what *training* is, in its barest form

### What you are going to write

Three exercises, and this theory is laid out so you read it in this order:

| Exercise | What it is | Where it is explained |
|---|---|---|
| 1. `Value` | A number that remembers where it came from | [§ From the chain rule to a class](#from-the-chain-rule-to-a-python-class) |
| 2. `topological_order` | What order to walk the graph backwards in | [§ The order matters](#the-order-matters-and-that-is-why-exercise-2-exists) |
| 3. `train_scalar_mlp` | Training a network with your engine | [§ What a neuron is](#what-a-neuron-is-and-what-an-mlp-is) and [§ The loop](#the-training-loop-step-by-step) |

### What it costs

3 hours. It is dense, but it is the foundation of everything that follows.

---

## The problem: which way do I move each weight?

Training is this: you have a function that measures how badly you are doing (the loss), and
you want to change the weights so that number goes down.

With **one** weight it is easy. Nudge it up and see whether the loss goes up or down. If it
goes down, keep nudging. The derivative of the loss with respect to a weight is called the
**gradient**, and it is literally that: how much the loss changes if I move this weight a
little.

With 8.9 million weights, brute force is no good. Trying weight by weight means evaluating
the whole network 8.9 million times per training step. A single step would take days.

## Three ways to differentiate, and why only one works

**Numerical.** What I just described: move the weight a little and see what happens. Besides
being extremely expensive, it is imprecise: if you move too little, the difference gets lost
in floating-point rounding; if you move too much, the approximation is bad. It is useful for
*checking* that your gradients are right, not for training.

**Symbolic.** Deriving the closed-form derivative, like in a calculus class. As soon as you
compose a few functions, the expression grows until it is unmanageable.

**Reverse-mode automatic.** What you are going to implement. It breaks the computation into
elementary operations (additions, multiplications) and applies the chain rule backwards.
Cost: **one forward plus a backward that costs twice as much, and that is it** — it makes no
difference whether there are 10 parameters or 10 billion. That result is what makes deep
learning possible.

## The chain rule, with numbers

The whole machine rests on something you already know from school. If $y$ depends on $u$ and
$u$ depends on $x$:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

Let us see it working. Let $a = 2$, $b = -3$, and:

```
c = a · b        ->  c = -6
d = c + 10       ->  d = 4
```

We want $\partial d/\partial a$: if I move $a$, how much does $d$ move?

We start at the end and work backwards:

```
∂d/∂d = 1                      (something with respect to itself)
∂d/∂c = 1                      (d = c + 10, adding a constant does not change the slope)
∂c/∂a = b = -3                 (c = a·b, the derivative with respect to 'a' is 'b')

∂d/∂a = ∂d/∂c · ∂c/∂a = 1 · (-3) = -3
```

If I raise $a$ by 0.1, $d$ drops by 0.3. Check it: with $a = 2.1$ you get $c = -6.3$ and
$d = 3.7$. Exactly.

**That is the whole algorithm.** Going from back to front multiplying local derivatives. All
that is missing is organizing it so it works with thousands of operations. That "organizing"
is the `Value` class, and it is what comes next.

## Why backwards and not forwards

You could walk the chain the other way and it would also work. But there is an asymmetry
that changes everything: **many inputs, one single output**. You have millions of parameters
and a single loss.

Going backwards, you start at the loss (one number) and distribute it. Each intermediate
value you carry is a single number per node. Going forwards you would have to drag along one
value for each parameter you differentiate with respect to, that is, millions of parallel
computations.

General rule: reverse mode wins when there are many inputs and few outputs. Which is exactly
the case for any loss function.

---

## From the chain rule to a Python class

This is where `Value` shows up, and it is worth saying where it comes from: **it is not a
machine learning concept, it is an engineering decision**. We have an algorithm (go backwards
multiplying local derivatives) and we need a data structure that supports it. The name comes
from [micrograd](https://github.com/karpathy/micrograd), by Karpathy, which is where this
module borrows the idea from.

### Why a plain number is not enough

Write this in Python:

```python
a = 2.0
b = -3.0
c = a * b        # -6.0
```

After that line, `c` is `-6.0` and nothing else. Python has thrown away the two things you
need for the backward pass:

- that `c` came from `a` and `b` (not from any two other numbers), and
- that it came out of a **multiplication** (not an addition, which would have a different
  derivative).

The backward pass cannot reconstruct that. So the idea is: **instead of returning a number,
return an object that also stores its origin**. That object is `Value`.

### The five fields, and why each one is there

```python
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)          # the usual number
        self.grad = 0.0                  # ∂L/∂(this node). Starts at zero.
        self._prev = tuple(_children)    # who it came from
        self._op = _op                   # which operation it came out of
        self._backward = lambda: None    # how to pass the gradient down
```

One by one, against the example above (`c = a * b`):

- **`data`** is `-6.0`. What you already had.
- **`grad`** is where $\partial L/\partial c$ will accumulate. During the forward pass it is
  0; the backward pass fills it in from the top down. Mind the asymmetry: `data` gets filled
  going forwards, `grad` going backwards.
- **`_prev`** is `(a, b)`. The **edges of the graph**, stored backwards. It is what lets
  `backward()` know who to distribute to.
- **`_op`** is `'*'`. It is only there for debugging and for drawing the pretty graph in the
  demo. If you removed it, everything would still work.
- **`_backward`** is the odd one out, and it gets its own section below.

The leading underscore on `_prev`, `_op` and `_backward` is Python's convention for "this is
class internals". Whoever uses `Value` touches `data` and `grad`, nothing else.

### Why the operators are overloaded

You could build the graph by hand: `c = multiply(a, b)`. It would work, but by exercise 3
you would have to write the whole MLP in that notation and it would be unreadable.

By defining `__mul__`, Python translates `a * b` into `a.__mul__(b)` automatically. That is:
**you write ordinary maths and the graph builds itself as a side effect**. That is exactly
what PyTorch does when you write `x @ w + b` on tensors with `requires_grad=True`. There is
no trick beyond that.

The `__radd__`, `__rmul__` and friends — already written for you in `exercises.py` — cover
the `2 * a` case: Python first tries `(2).__mul__(a)`, the `int` does not know what to do
with a `Value`, returns `NotImplemented`, and then Python calls `a.__rmul__(2)`.

### The mould: every operation is written the same way

Learn this mould and seven of the nine methods are copy-and-change-two-lines:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)   # (a)
    out = Value(self.data * other.data, (self, other), '*')       # (b)

    def _backward():                                              # (c)
        self.grad  += other.data * out.grad
        other.grad += self.data  * out.grad

    out._backward = _backward                                     # (d)
    return out
```

- **(a)** wraps the bare number, so that `a * 3` works.
- **(b)** creates the result node and tells it who its children are. That is the forward
  pass, done.
- **(c)** defines how to distribute the gradient. **It does not run now.** It is only
  defined.
- **(d)** hangs it off the node, to be run during the backward pass.

### The closure: the part that is hardest to see

`_backward` is a function defined inside another function. When `__mul__` returns, that inner
function stays alive and **keeps seeing the variables `self`, `other` and `out`**, by
reference. That is called a *closure*, and it is what makes all of this work.

The consequence is the key to the module: when `_backward()` runs — many lines of code later,
already in the backward pass — the line `self.grad += other.data * out.grad` will read the
`out.grad` **of that moment**, not the one that was there when it was defined (which was 0).
By then `out`'s parents will have contributed their share.

Another way to see it: during the forward pass you are building a **to-do list**, one item
per operation. The backward pass runs them in reverse order.

### A full trace, with the numbers from the example

Let us follow `d = a * b + 10` with $a = 2$, $b = -3$, field by field.

**Forward** — four nodes get created:

| node | `data` | `_op` | `_prev` | `grad` |
|---|---|---|---|---|
| `a` | 2.0 | `''` (leaf) | `()` | 0.0 |
| `b` | -3.0 | `''` (leaf) | `()` | 0.0 |
| `c` | -6.0 | `'*'` | `(a, b)` | 0.0 |
| `10` | 10.0 | `''` (leaf) | `()` | 0.0 |
| `d` | 4.0 | `'+'` | `(c, 10)` | 0.0 |

Every gradient is 0. Nothing has been differentiated yet: the graph has only been built
while the result was being computed.

**Backward** — `d.backward()` does three things:

1. `d.grad = 1.0` — the seed. The derivative of something with respect to itself. **Without
   this line everything stays at zero and absolutely nothing happens**, because every
   `_backward` multiplies by the gradient of the node above.
2. It asks for the topological order: `[10, b, a, c, d]` (that is the real order the
   implementation returns; what matters is that `d` ends up last).
3. It walks it **in reverse** calling `node._backward()`:

```
d._backward()  ->  c.grad  += 1.0        ->  c.grad  = 1.0
                   10.grad += 1.0
c._backward()  ->  a.grad  += b.data * c.grad = -3 * 1 = -3
                   b.grad  += a.data * c.grad =  2 * 1 =  2
a._backward()  ->  does nothing: it is a leaf, its _backward is `lambda: None`
b._backward()  ->  same
```

Result: `a.grad = -3.0`, `b.grad = 2.0`. Which is what came out by hand in the chain rule
section. You can verify it yourself: `llmfs demo 02` prints this table next to what
`torch.autograd` says for the same expression.

### The local derivatives: one line per method

The only thing that changes between operations is step (c). This table **is** exercise 1:

| operation | what goes in `_backward` |
|---|---|
| `a + b` | `self.grad += out.grad` ; `other.grad += out.grad` |
| `a * b` | `self.grad += other.data * out.grad` ; `other.grad += self.data * out.grad` |
| `a ** n` | `self.grad += n * self.data**(n-1) * out.grad` |
| `exp(a)` | `self.grad += out.data * out.grad` (because $e^a$ is its own derivative) |
| `log(a)` | `self.grad += (1/self.data) * out.grad` |
| `tanh(a)` | `self.grad += (1 - out.data**2) * out.grad` |
| `relu(a)` | `self.grad += (out.data > 0) * out.grad` |

Notice the pattern: **local derivative × gradient arriving from above**. That is literally
the chain rule, written once per operation. Addition passes the gradient through untouched
(local derivative 1), which is why it is said that "addition is a gradient router".

### The sugar needs no new derivatives

`__neg__`, `__sub__` and `__truediv__` lean on what you already have:

```python
-a       ->  self * -1
a - b    ->  self + (-other)
a / b    ->  self * other ** -1
```

You do not need to write the quotient rule: it falls out of composing `*` and `**-1`, and
the chain rule reconstructs it. This is why an autodiff engine needs so few primitive
operations.

## The detail that breaks everything if you get it wrong: the `+=`

Note that the whole table says `+=` and not `=`. That is not a whim.

Try the simplest possible case: $y = x + x$, with $x = 3$.

- With `=`: the first branch sets `x.grad = 1`, the second **overwrites** it and sets
  `x.grad = 1` again. Result: 1. **Wrong.**
- With `+=`: `x.grad = 1 + 1 = 2`. And that is correct, because $y = 2x$ and its derivative
  is 2.

The mathematical reason is the multivariable chain rule: if a variable influences the result
through several paths, its total derivative is the **sum** of what each path contributes.

$$\frac{\partial L}{\partial x} = \sum_{k} \frac{\partial L}{\partial u_k} \frac{\partial u_k}{\partial x}$$

In a network this happens constantly: every residual connection, every shared weight, every
embedding that appears several times in a sentence.

**A practical consequence that will bite you:** since gradients add up, they have to be
**zeroed before every step**. That `optimizer.zero_grad()` you will see in every training
loop is exactly this. If you forget it, no error fires: each step uses the sum of every
gradient from the beginning, and the model simply does not learn. It is one of the most
frustrating bugs there is because it gives no signal at all.

---

## The order matters (and that is why exercise 2 exists)

When a node passes its gradient down, it must already have received **everything** due to
it from above. If it distributes too early, it sends an incomplete gradient and everything
below comes out wrong.

Look at it in the smallest graph where it shows, a **diamond**: $x$ influences $L$ through
two paths.

```
x = 3
u = x * 2        ->  6        (path 1)
v = x + 1        ->  4        (path 2)
L = u * v        ->  24
```

The correct answer: $L = 2x(x+1) = 2x^2 + 2x$, so $dL/dx = 4x + 2 = 14$.

And through the graph: $\partial L/\partial u = v = 4$, $\partial L/\partial v = u = 6$, so
$\partial L/\partial x = 4 \cdot 2 + 6 \cdot 1 = 14$. It matches.

Now, **the order**. If you process the nodes like this:

```
L  ->  distributes to u and v    u.grad = 4,  v.grad = 6      fine
u  ->  distributes to x          x.grad += 4*2 = 8
v  ->  distributes to x          x.grad += 6*1 = 6            total 14   ✓
```

But if you process `u` **before** `L`, `u.grad` is still 0 and `x` receives 0 through that
path: you get 6 instead of 14. Silently wrong.

The rule is: **each node is only processed once everything that depends on it has been
processed**. That order is called a *topological order*, and you get it with a depth-first
traversal that records each node **after** its children (post-order). `backward()` walks that
list in reverse, so the root — the loss — goes first.

With a simple, tree-shaped graph any reasonable order works and the bug goes unnoticed. As
soon as there is a reused node, a wrongly computed order produces incorrect gradients **with
no visible symptom**: the loss drops somewhat, the model learns worse, and there is nothing
pointing at the culprit.

An implementation note: do it **iteratively, with a stack**, not recursively. The graph of
an MLP with a few hundred neurons already exceeds Python's recursion limit — the one in
exercise 3, with only 113 parameters, has **1,068 nodes**. The docstring of exercise 2 has
the skeleton with the "children already expanded" flag, which is the standard trick for
doing a post-order without recursion.

---

## What a neuron is, and what an MLP is

So far you have built a derivative engine. Exercise 3 uses it to train something, and that
something is an **MLP**. If you have never seen one, this is the section that was missing.

The `MLP` class **is already written** and gets imported in `exercises.py`; it is not the
point of the module. But it should not be a black box, so here it is from the inside. The
code lives in `llmfs/reference/autograd.py` and it is 60 lines: open it while you read this.

### One neuron

A neuron does **two things**, in this order:

1. a **weighted sum** of its inputs, plus a constant term:
   $\text{act} = w_1 x_1 + w_2 x_2 + \dots + b$
2. it passes the result through a **non-linear** function, here `tanh`.

The $w_i$ are the **weights** and $b$ is the **bias**. They are the parameters: the numbers
training is going to move. The $x_i$ are the input, and they do not get touched.

With numbers. Input $x = [1.0,\ -2.0]$, weights $w = [0.5,\ 0.25]$, bias $b = 0.1$:

```
act = 0.5·1.0 + 0.25·(-2.0) + 0.1 = 0.5 - 0.5 + 0.1 = 0.1
o   = tanh(0.1) = 0.099668
```

In code it is four lines, and note that it is **written with your `Value`s**: each `wi * xi`
creates a node, each addition creates another, and the `tanh` one more. By the time the
neuron's forward pass is done, its little piece of the graph is already built.

```python
class Neuron:
    def __init__(self, nin, nonlin=True, value_cls=Value, rng=None):
        self.w = [value_cls(rng.uniform(-1, 1)) for _ in range(nin)]
        self.b = value_cls(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return [*self.w, self.b]
```

The `value_cls` parameter is what lets you build the network with **your** class instead of
the reference one. That is why `train_scalar_mlp` receives it and passes it down.

**Why the `tanh` is needed.** If you remove the non-linearity, each layer is a linear
combination of the previous one, and composing linear functions gives another linear
function: a 50-layer network would collapse into a single one. The non-linearity is the only
thing that makes stacking layers worth anything. `tanh` squashes any number into the interval
$(-1, 1)$: `tanh(0.1) = 0.0997`, `tanh(3) = 0.995`, `tanh(-3) = -0.995`. In module 08 you
will see why transformers use GELU instead of `tanh`.

### A layer and an MLP

A **layer** is several neurons looking at the same input, each with its own weights. An
**MLP** (multi-layer perceptron) is layers chained together: the output of one is the input
of the next.

```python
class MLP:
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

The one in exercise 3 is `MLP(3, [8, 8, 1])`: a vector of 3 numbers goes in, passes through
two layers of 8 neurons and **one** number comes out. Count the parameters: the first layer
has 8 neurons × (3 weights + 1 bias) = 32; the second 8 × (8 + 1) = 72; the output one
1 × (8 + 1) = 9. Total **113 parameters**, which is what `len(model.parameters())` returns.

Two details of the constructor worth noting:

- **The last layer is linear**, with no `tanh` (`nonlin=i < len(nouts) - 1`). If the output
  has to be able to be 2.7 or -40, squashing it into $(-1,1)$ stops you. It is the same
  reason the GPT in module 10 ends in a linear layer producing unbounded logits.
- **The weights start out random** (`rng.uniform(-1, 1)`) and the biases at zero. If they all
  started equal, every neuron in a layer would compute exactly the same thing and receive the
  same gradient forever: the whole layer would behave like a single neuron. It is called
  *symmetry breaking*, and it comes back with more care in module 10.

The other two methods are the ones the training loop uses:

- `parameters()` flattens the whole network into a list of 113 `Value`s. That is what gradient
  descent gets applied to.
- `zero_grad()` sets all 113 `.grad`s to zero. You already know why it exists.

None of this is specific to MLPs: `parameters()` and `zero_grad()` are called the same in
PyTorch and do exactly the same thing.

---

## The training loop, step by step

Exercise 3 has four input points and their four targets:

```
xs = [[2,3,-1], [3,-1,0.5], [0.5,1,1], [1,1,-1]]
ys = [   1,        -1,         -1,        1    ]
```

The network is asked to return $+1$ for the first and fourth, and $-1$ for the other two. At
first it does not: with the random weights, the prediction for the first point is 0.939 when
it should be 1, and the rest are worse.

### Measuring the error: MSE

We need a number that says how badly it is going. The **mean squared error** (MSE) is the
average of the squared errors:

$$L = \frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)^2$$

It is squared for two reasons: so that errors of opposite sign do not cancel out, and because
the square is differentiable everywhere (the absolute value is not, at zero). With this
exercise's initial weights it comes to $L = 1.4334$.

In code, with your `Value`s:

```python
loss = sum(((p - y)**2 for p, y in zip(preds, ys)), value_cls(0.0)) * (1.0 / len(ys))
```

That `value_cls(0.0)` is the `sum`'s start value: without it Python starts accumulating from
the integer `0`, and although it would work through `__radd__`, being explicit is cleaner.
And note the important part: **`loss` is a `Value`**, the root of a graph of 1,068 nodes
reaching every one of the 113 parameters. That is why you can call `loss.backward()` on it.

### Gradient descent

After the backward pass, each parameter has its `.grad`: how much the loss goes up if I raise
that parameter. Since you want the loss to go **down**, you move against it:

```python
for p in model.parameters():
    p.data -= lr * p.grad
```

`lr` is the **learning rate**: how much notice you take of the gradient. The gradient gives
you the direction, not the distance — it is a local slope, and it is only trustworthy very
close to the point where you computed it.

A full step, with the neuron from before ($w = [0.5,\ 0.25]$, $b = 0.1$, $x = [1,\ -2]$) and
target $y = 1$:

```
forward:   o = tanh(0.1) = 0.099668
           L = (o - 1)² = 0.810598
backward:  dL/dw0 = -1.782777
           dL/dw1 = +3.565553
           dL/db  = -1.782777
step:      w0 = 0.5 - 0.1·(-1.782777) = 0.678278      (lr = 0.1)
```

$w_0$'s gradient is negative, so $w_0$ **goes up**, and with that `act` goes up, `o` moves
closer to 1 and the loss goes down. The signs add up: $x_1$ is $-2$, negative, and that is
why its weight moves the other way.

Those three gradients are checked against `torch.autograd` — they agree to six decimal
places.

### The six steps, and why in that order

```python
for _ in range(steps):
    preds = [model(x) for x in xs]                    # 1. forward
    loss  = mse(preds, ys)                            # 2. loss
    model.zero_grad()                                 # 3. clear the gradients
    loss.backward()                                   # 4. backward
    for p in model.parameters():                      # 5. move the weights
        p.data -= lr * p.grad
    history.append(loss.data)                         # 6. record
```

**Step 3 is the one everybody forgets**, and it goes **before** the backward pass. Gradients
accumulate (you wrote it that way in exercise 1), so without clearing them step 50 uses the
sum of the gradients from steps 1 to 50. And it raises no error: the loss drops a bit at
first and then stalls or blows up.

Putting it *after* the backward pass also works, but only by accident — it leaves the
gradients clean for the next round. It is fragile: as soon as the loop has a `continue` or a
branch, it stops holding.

**`p.data -= ...` and not `p -= ...`.** You are modifying the number inside the node, not
creating a new node. If you created new nodes, the next step's graph would hang off the
previous one and would grow without bound until it exhausted memory. In PyTorch the exact
equivalent of this distinction is the `with torch.no_grad():` that wraps the optimizer step.

### What should come out

With `steps=100, lr=0.05, seed=0`, the loss `train_scalar_mlp` returns:

| step | 0 | 1 | 5 | 10 | 25 | 50 | 99 |
|---|---|---|---|---|---|---|---|
| loss | 1.4334 | 0.9573 | 0.4246 | 0.1760 | 0.0024 | 6.7·10⁻⁷ | ~10⁻¹¹ |

If your first value is 1.4334 and it drops monotonically, your engine works. And with
`lr=0.005`, ten times smaller, after 100 steps the loss is still at 0.1701: it has not had
time. It is the same model and the same initialization; only the learning rate changes. Run
`llmfs demo 02` to see the two curves side by side.

---

## What PyTorch does differently

Conceptually, nothing. A `torch.Tensor` with `requires_grad=True` builds the same graph,
stores the same backward functions and does the same topological traversal:

| your code | PyTorch |
|---|---|
| `Value` | `torch.Tensor` with `requires_grad=True` |
| `.data` | `.data` |
| `.grad` | `.grad` |
| `_prev` | the graph's internal edges, in C++ |
| `_backward` | `.grad_fn` (`<MulBackward0>`, `<TanhBackward0>`…) |
| `topological_order` | the autograd engine's traversal |
| `model.zero_grad()` | `optimizer.zero_grad()` |
| `p.data -= lr * p.grad` | `optimizer.step()` inside `torch.no_grad()` |

The differences are engineering:

- It operates on **tensors** instead of individual numbers: one operation is a GPU kernel
  instead of a million Python objects. Your engine takes a good second to do 100 steps on a
  113-parameter network; module 13 trains 8.9 million.
- The graph lives in C++.
- It frees the intermediate results as it walks them, to save memory. That is why calling
  `backward()` twice fails unless you ask for `retain_graph=True`.

By the end of this module, `loss.backward()` is code you understand line by line. That is
the only reason this module exists.

## Where the debate is

Not much here. Reverse-mode autodifferentiation has been settled mathematics since the
1970s, and its rediscovery as "backpropagation" (Rumelhart, Hinton and Williams, 1986) is a
classic case of reinvention in the scientific literature. What is still open is whether the
brain does anything similar: the *weight transport problem* — that the backward pass needs
the same weights as the forward, transposed — has no known biological analogue, and there is
a whole line of research looking for plausible alternatives.

---

**Further reading:** Karpathy, [micrograd](https://github.com/karpathy/micrograd) (150
lines, worth reading in full after doing the exercise) · Baydin et al. 2018,
[Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767).
Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
