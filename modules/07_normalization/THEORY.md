# 07 — Normalization and residual connections

## Why this module matters

**Because without this, a deep network does not train. Full stop.**

Two pieces that compute nothing interesting and that are the difference between a model that
learns and one that returns `NaN` after three steps. They are the Transformer's plumbing:
nobody mentions them in the headlines and without them there is nothing.

The problem they solve is concrete and you are going to see it measured: the numbers flowing
through a deep network tend to explode or vanish, and with 40 layers the gradient reaches
**exactly** zero. The demo measures it.

On top of that, this is one of the design decisions where you learn most by comparing: where
you put the normalization changes whether your network needs warmup or not.

### What you will know by the end

- Why the numbers get out of control when you stack layers, with the arithmetic that
  explains it
- What LayerNorm does exactly, and **what is surplus to it** (that is RMSNorm)
- Why `x + f(x)` is one of the most important ideas in deep learning
- Pre-norm against post-norm, measured: how much gradient reaches the first layer in each
  case

### What it costs

1.5 hours. Three short exercises, and the third is literally one line.

---

## The problem: the numbers get out of control

A deep network is a composition of functions. Each layer multiplies by a matrix, and those
multiplications accumulate.

Imagine each layer multiplies the magnitude of its inputs by 1.2. It looks harmless:

```
layer 1:  ×1.2  ->  1.2
layer 2:  ×1.2  ->  1.44
layer 3:  ×1.2  ->  1.73
...
layer 40: ×1.2  ->  1470
```

And if the factor were 0.8 instead of 1.2, after 40 layers you would have 0.00013. In one
case the numbers explode, in the other they vanish. **And the same happens to the gradient
going backwards**, which is what really hurts: if the gradient vanishes, the layers below
receive no signal and learn nothing.

With fp16, which only goes up to 65504 at the top and down to $6\times10^{-5}$ at the
bottom, this stops being an annoyance and becomes `inf` and zeros.

## Solution 1: normalize

The idea is brutal in its simplicity: **after each block, put the numbers back on a known
scale.** It does not matter what the layer did; on the way out, you renormalize.

### LayerNorm, with numbers

Take one token's vector, say 4-dimensional:

```
x = [2.0, 8.0, 4.0, 6.0]
```

Compute its mean and its variance:

```
mean     = (2+8+4+6)/4 = 5.0
variance = ((2-5)² + (8-5)² + (4-5)² + (6-5)²)/4 = (9+9+1+1)/4 = 5.0
std dev  = √5 = 2.236
```

Subtract the mean and divide by the standard deviation:

```
x_norm = [(2-5)/2.236, (8-5)/2.236, (4-5)/2.236, (6-5)/2.236]
       = [-1.342, 1.342, -0.447, 0.447]
```

Now it has mean 0 and variance 1, wherever the input came from.

But always forcing mean 0 and variance 1 takes freedom away from the model, so it is given
back with two learned parameters, $\gamma$ (scale) and $\beta$ (shift):

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

The $\epsilon$ (typically $10^{-5}$) avoids dividing by zero when every component is equal.

**Important: the mean and the variance are computed over each token's dimensions,
separately.** Nothing to do with BatchNorm, which normalizes along the batch. LayerNorm
treats each token independently, which is why it works the same with a batch of 1 as with
1000 and does not need to store statistics for inference.

### RMSNorm: dropping half of it

In 2019, Zhang and Sennrich tried something: what if we do not subtract the mean?

$$y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

It only rescales by the root mean square. No mean subtraction, no $\beta$. With the same
vector:

```
RMS = √((4+64+16+36)/4) = √30 = 5.477
x_norm = [0.365, 1.461, 0.730, 1.096]
```

Result: **between 7% and 64% faster depending on the case, with no measurable loss of
quality**. It saves a pass over the data and an intermediate tensor. That is why Llama,
Mistral and practically everything modern use it, and why our model uses it.

An implementation detail that matters: **the computation is done in float32 even if the
input arrives in float16**. Squaring large activations can overflow the fp16 range and give
`inf`.

## Solution 2: residual connections

The second piece, and the more important of the two.

Instead of each block *replacing* the representation, it is asked to *modify* it:

$$x_{\text{output}} = x + f(x)$$

The block computes a correction, not a replacement. That accumulated sum flowing through the
whole network is called the **residual stream**.

**Why this changes everything:** differentiating that expression with respect to $x$,

$$\frac{\partial x_{\text{output}}}{\partial x} = 1 + \frac{\partial f(x)}{\partial x}$$

That **1** is a highway. Even if $\partial f/\partial x$ is tiny, the gradient reaching the
layers below never drops below 1 along that path. Without residuals, the factors multiply
and vanish; with residuals, there is always a direct route.

## Pre-norm against post-norm

Now the question that decides whether your network trains: **where does the normalization
go?**

```
post-norm (the 2017 paper):     x = norm(x + f(x))
pre-norm  (everything modern):  x = x + f(norm(x))
```

It looks cosmetic. It is not.

In **post-norm**, the normalization sits *on top of* the residual path. The gradient goes
through it at every layer and gets rescaled: the highway has a toll at every exit. With 6
layers you barely notice; with 40, you need a careful learning-rate warmup so training does
not explode in the first steps.

In **pre-norm**, the normalization sits *inside* the branch. The path $x \to x$ is
completely clear and the gradient reaches the first layer intact. You can train without
warmup and with higher learning rates.

The price of pre-norm: the residual stream grows with depth, because each layer adds its
contribution with nobody normalizing it again. That is why pre-norm models **always** carry
a final normalization before the output layer. If you forget it, the logits come out at an
arbitrary scale.

The module's demo measures this empirically: it trains the same network with both variants
and compares the norm of the gradient reaching the first layer.

## Where the debate is

There is more here than it looks.

**Why normalization works is still unclear.** BatchNorm's original explanation (2015) was
*internal covariate shift*: that normalizing stabilizes the distribution of each layer's
inputs. Santurkar et al. (2018) put it to the test by injecting noise *after* normalizing —
deliberately destroying that stability — and the network kept training just as well. The
original explanation is today largely discarded, and its replacement — that it smooths the
loss landscape — is more an empirical observation than a theory.

**Pre-norm is not free.** It is fairly accepted that pre-norm trains more easily, but there
is evidence that post-norm, when it converges, reaches better final quality. There are
recent architectures using hybrid variants for exactly this reason. We use pre-norm because
it is the robust option and what everyone does, not because it is proven superior.

---

**Further reading:** Ba et al. 2016, [Layer Normalization](https://arxiv.org/abs/1607.06450)
· Zhang & Sennrich 2019, [RMSNorm](https://arxiv.org/abs/1910.07467) · Xiong et al. 2020,
[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
(the pre/post-norm analysis) · He et al. 2015,
[Deep Residual Learning](https://arxiv.org/abs/1512.03385). Stray terms are in
[GLOSSARY.md](../../GLOSSARY.md).
