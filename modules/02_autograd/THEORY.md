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
that is missing is organizing it so it works with thousands of operations.

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

## The graph: each operation remembers itself

The structure is this: every time you do an operation, instead of returning just the result
you return an object that also stores **where it came from**.

For `c = a * b`, the object `c` stores:

- its value (`-6`),
- who its children are (`a` and `b`),
- and a function that knows how to pass backwards whatever gradient reaches it.

That last part, for a multiplication, is:

$$\frac{\partial L}{\partial a} \mathrel{+}= b \cdot \frac{\partial L}{\partial c},
\qquad
\frac{\partial L}{\partial b} \mathrel{+}= a \cdot \frac{\partial L}{\partial c}$$

Chaining operations together builds a graph, and the backward pass walks all of it.

## The detail that breaks everything if you get it wrong: the `+=`

Note that it says `+=` above and not `=`. That is not a whim.

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

## The order matters

When a node passes its gradient down, it must already have received **everything** due to
it from above. If it distributes too early, it sends an incomplete gradient and everything
below comes out wrong.

That forces you to walk the nodes in a specific order: **each node is only processed once
everything that depends on it has been processed**. That order is called a *topological
order*, and it is computed with a depth-first traversal.

With a simple, tree-shaped graph any reasonable order works and the bug goes unnoticed. As
soon as there is a reused node, a wrongly computed order produces incorrect gradients **with
no visible symptom**: the loss drops somewhat, the model learns worse, and there is nothing
pointing at the culprit.

An implementation note: do it **iteratively, with a stack**, not recursively. The graph of
an MLP with a few hundred neurons already exceeds Python's recursion limit.

## What PyTorch does differently

Conceptually, nothing. A `torch.Tensor` with `requires_grad=True` builds the same graph,
stores the same backward functions (there they are called `grad_fn`) and does the same
topological traversal. The differences are engineering:

- It operates on **tensors** instead of individual numbers: one operation is a GPU kernel
  instead of a million Python objects.
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
