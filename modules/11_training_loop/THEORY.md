# 11 — The training loop

## Why this module matters

**Because having a model is not the same as having it trained.**

You already know how to build a GPT that produces logits and how to measure how wrong it is.
What is missing is the part that turns that into learning: how 8.9 million parameters move
so the loss goes down.

You wrote the loop itself in module 02 with your autodiff engine: predict, measure,
gradients, move, repeat. What gets added here are four pieces that make that loop work **at
scale** instead of diverging after fifty steps.

Each one solves a concrete problem you would see if it were not there. And all four are what
will let you debug a training run that is going wrong instead of changing numbers at random.

### What you will know by the end

- Why a single learning rate works for the whole network (and what Adam does to achieve it)
- What warmup is and why without it the model sometimes never recovers
- How to stop **a single odd batch** from destroying hours of training
- Which parameters must NOT decay, and why applying it to all of them is a silent error
- An AMP detail that always gets forgotten and makes training crawl

### What it costs

4 hours. The first exercise (AdamW from scratch) is the longest in the course; the other
three are short.

---

## The loop, in four lines

```
for each step:
    predict and measure the loss       (forward)
    compute the gradients              (backward)
    move the parameters                (optimizer step)
    zero the gradients
```

That is all. You already wrote it in module 02 with your autodiff engine. The rest of the
module is four pieces that make that loop work at scale.

## Piece 1: the optimizer

The simplest version of "move the parameters" is gradient descent:

```
p ← p − lr · gradient
```

It works, and it has a serious problem. Think about two parameters in your model: one from
the embedding of the word `the`, which appears in almost every sentence, and one from the
embedding of a rare word. The first receives large gradients constantly; the second, almost
never. With a single `lr` for both, either the first takes absurd jumps or the second never
moves.

**Adam** solves this with two ideas.

**Momentum.** Instead of moving along this step's gradient, a running average of the recent
ones is used:

```
m = 0.9·m + 0.1·gradient
```

Since each batch is a different sample, its gradients are noisy. Averaging cancels the noise
and leaves the consistent direction.

**Per-dimension scaling.** A running average of the **squared** gradient is also kept, and
you divide by its square root:

```
v = 0.95·v + 0.05·gradient²
step = m / √v
```

A parameter with consistently large gradients has a large $v$ and moves little. One that
almost never receives signal has a small $v$ and moves a lot when it does. **Each parameter
ends up with its own effective learning rate**, which is why a single global `lr` works.

### The bias correction

$m$ and $v$ start at zero, so the first steps underestimate the real magnitudes. With
$\beta_2 = 0.95$, after one step $v$ is only 5% of $g^2$: dividing by its square root would
give a step 4.5 times larger than it should be.

The correction fixes it exactly:

$$\hat{m} = \frac{m}{1-\beta_1^t}, \qquad \hat{v} = \frac{v}{1-\beta_2^t}$$

At step 1 with $\beta_2 = 0.95$: $1 - 0.95 = 0.05$, and dividing by 0.05 multiplies by 20,
which is exactly the missing factor. As $t$ advances, $\beta^t \to 0$ and the correction
fades away by itself.

Without it, the first steps take enormous jumps and training can diverge before it begins.

### The W in AdamW

*Weight decay* is pushing the weights towards zero so they do not grow without control. There
are two ways of doing it, and the difference matters:

```
Adam + L2:   g ← g + λ·p       then Adam processes g
AdamW:       p ← p − lr·λ·p    directly, separately from Adam
```

In the first, the decay goes through the division by $\sqrt{v}$, so its real effect depends
on the magnitude of that parameter's gradients. A weight with large gradients barely decays;
one with small gradients decays enormously. Nobody wants that.

Loshchilov and Hutter (2019) decoupled it and it worked better consistently. Hence the W.

## Piece 2: the learning-rate scheduler

The `lr` is not constant during training. It has two segments.

**Warmup: rising slowly at the start.** In the first steps, Adam's moments are nearly empty
and its estimates are noisy; on top of that the weights are freshly initialized and the
gradients are large. Starting at full `lr` usually produces a loss spike that the model
sometimes never recovers from. It rises linearly from 0 to `lr` over 500 steps.

**Cosine: falling at the end.** At the start you want to move fast; at the end, to
fine-tune. The cosine drops slowly, then fast, then slowly again:

$$\text{lr}(t) = \text{lr}_{\min} + (\text{lr} - \text{lr}_{\min}) \cdot \frac{1 + \cos(\pi \cdot \text{progress})}{2}$$

It does not decay to zero, but to 10% of the initial `lr`: below a certain point the model
stops learning entirely and compute is wasted.

## Piece 3: gradient clipping

Occasionally a batch produces enormous gradients — an odd sequence, a very rare token.
Without protection, that single batch can take a jump that destroys hours of training.

The fix: compute the **global** norm of all the gradients together, as if they were one
vector, and if it exceeds a threshold, multiply them all by the same factor.

```
norm = √(Σ ‖g_i‖²)
if norm > max_norm:  every g ×= max_norm / norm
```

**Global, not per tensor.** Clipping each tensor separately would change the *direction* of
the combined gradient, which is exactly what you do not want: the gradient points where to
go, and you are only limiting how far you step. With the global norm the direction is
preserved exactly.

## Piece 4: which parameters decay and which do not

Weight decay **only on the matrices** (parameters with 2 dimensions or more). Biases and
normalization scales, no.

Think about it: an RMSNorm's scale starts at 1 and its job is to rescale the output. Pushing
it towards zero is pushing the layer's output towards zero, which is exactly the opposite of
what is needed.

Applying decay to everything is a common mistake, it **produces no visible error** and it
degrades the result. It can only be detected by comparing two complete training runs.

## And mixed precision

On the RTX 2060 (Turing, no bf16) training runs in fp16, whose range runs out at the bottom
at $6\times10^{-5}$. The gradients of the deep layers are smaller than that and become zero.

`GradScaler` solves it: it multiplies the loss by ~65,000 before the backward pass, which
raises every gradient into the representable range, and divides before the optimizer step. If
some value overflows, it discards that step and lowers the factor.

**There is a detail that gets forgotten and is silent:** with AMP you have to **unscale the
gradients before clipping them**. Otherwise their norm is multiplied by 65,000 and you would
be clipping to a threshold 65,000 times smaller than you think. Training crawls with nothing
to indicate it.

```python
scaler.unscale_(optimizer)      # this first
clip_grad_norm(params, 1.0)     # and now the clipping
```

## Where the debate is

Adam **dominates without anyone really knowing why**. The usual justification — that it
approximates second-order information — does not survive analysis: $\sqrt{v}$ is not the
Hessian's diagonal or anything like it. There is work suggesting its real advantage lies in
scale invariance, or in how it interacts with normalization. It is still open.

The same goes for **warmup**: it is essential in practice and the explanations are post-hoc.
There are results suggesting that with pre-norm and careful initialization you can do without
it, which points to it compensating for problems in other parts of the architecture.

And about our config's **hyperparameters**: `lr=1e-3`, `betas=(0.9, 0.95)`,
`weight_decay=0.1`, `warmup=500`. They are standard values inherited from GPT-2/GPT-3 and
eyeballed for this scale. They are not optimal; they are reasonable. A hyperparameter sweep
would probably find something better, and it would cost more compute than the training run
itself.

---

**Further reading:** Loshchilov & Hutter 2019,
[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) · Kingma & Ba
2015, [Adam](https://arxiv.org/abs/1412.6980) · Micikevicius et al. 2018,
[Mixed Precision Training](https://arxiv.org/abs/1710.03740). Stray terms are in
[GLOSSARY.md](../../GLOSSARY.md).
