# 11 — The training loop: the four pieces that stop it blowing up

## Why this module matters

**Because having a model is not the same as having it trained.**

You already know how to build a GPT that produces logits and how to measure how wrong it is. What
is missing is the part that turns that into learning: how 8,933,440 parameters get moved so that
the loss goes down.

And the loop itself **you already wrote**, in module 02, with your own derivative engine: predict,
measure, gradients, move, zero out, repeat. That does not change. What gets added here are four
pieces that make that loop work **at scale** instead of diverging after fifty steps.

Each one solves a concrete problem you would see if it were not there, and the demo shows you all
four problems happening. They are also the four things that will let you debug a training run
that is going wrong instead of changing numbers at random, which is what everybody does.

### What you will know by the end

- Why a single learning rate works for the whole network, and what Adam does to achieve that
- How an optimizer is written in PyTorch: `param_groups`, `state`, and why `step` has the shape it
  has
- What warmup is and why without it the model sometimes never recovers
- How to stop **a single odd batch** destroying hours of training, measured
- Which parameters must NOT decay, and why applying it to all of them is a silent mistake
- A detail of AMP that always gets forgotten and makes training crawl

### What you are going to write

Four exercises, and this theory follows them in order:

| Exercise | What it does |
|---|---|
| 1. `AdamWScratch` | The optimizer, from scratch |
| 2. `lr_at_step` | How the learning rate changes over the run |
| 3. `clip_grad_norm` | Stopping one odd batch destroying hours of work |
| 4. `build_param_groups` | Which parameters decay and which do not |

Exercise 1 is **the longest in the course** and the other three are short: number 4 is five lines.
There is a small circular dependency between 1 and 4 — the `step` you write in 1 walks over the
groups that 4 builds — but neither needs the other to work or to pass its tests. If exercise 1
bogs you down, do 2, 3 and 4 first and come back.

When all four are green, **the final model will train with your optimizer**.

### What it costs

4 hours. It opens Part III: this is where you go from having a model to training it.

---

## The loop, and what it is missing

Let us start with what you already know. The training loop, bare, is this:

```
   for each step:
       predict and measure the loss      (forward)
       compute the gradients             (backward)
       move the parameters               (optimizer step)
       zero the gradients
```

Four lines, and you wrote them in module 02 over 113 parameters. With 8.9 million and ten thousand
steps, that loop as it stands **does not make it to the end**. What it is missing is what you are
writing today, and each piece goes in a specific place:

```
   for each step:
       lr = lr_at_step(step, ...)          ← exercise 2: which lr applies now
       set that lr on the optimizer's groups

       forward, backward                    (you already have this)

       clip_grad_norm(params, 1.0)         ← exercise 3: cap the damage of an odd batch
       optimizer.step()                    ← exercise 1: YOUR AdamW
       optimizer.zero_grad()

   and once only, when building the optimizer:
       build_param_groups(model, 0.1)      ← exercise 4: who decays and who does not
```

That is the full picture. The rest of the theory explains each piece in that order.

---

## Exercise 1: the optimizer (`AdamWScratch`)

### The problem: one learning rate does not fit all

The simplest version of "move the parameters" is gradient descent, module 02's:

```
   p ← p − lr · gradient
```

It works, and it has a serious problem. Think about two parameters in your model: one from the
embedding of the word `the`, which appears in almost every sentence, and one from the embedding of
a rare word that shows up once in a thousand. The first receives large gradients constantly; the
second, almost never. With a single `lr` for both, either the first takes absurd jumps or the
second never moves at all over the whole run.

And you cannot set one `lr` per parameter by hand: there are 8.9 million of them.

### Adam's two ideas

**Momentum.** Instead of moving according to this step's gradient, a moving average of the recent
ones is used:

```
   m = 0.9·m + 0.1·gradient
```

Since every batch is a different sample, its gradients are noisy. Averaging cancels the noise and
leaves the consistent direction.

**Per-dimension scaling.** A moving average of the **squared** gradient is also kept, and we divide
by its root:

```
   v = 0.95·v + 0.05·gradient²
   step = m / √v
```

A parameter with consistently large gradients has a large $v$ and moves little. One that hardly
ever receives signal has a small $v$ and moves a lot when it does. **Each parameter ends up with
its own effective learning rate**, worked out on its own, and that is why a single global `lr`
works for the whole model.

The demo measures it on the same task, same `lr` and same number of steps:

```
   SGD (no momentum, no scaling)   final loss 0.309866
   full AdamW                      final loss 0.000200
```

Three orders of magnitude, and the only difference is the per-dimension scaling.

### The bias correction

$m$ and $v$ start at zero, so the first steps underestimate the real magnitudes. With
$\beta_2 = 0.95$, after one step $v$ is only 5% of $g^2$: dividing by its root would give a step
4.5 times bigger than it should be.

The correction fixes it exactly:

$$\hat{m} = \frac{m}{1-\beta_1^t}, \qquad \hat{v} = \frac{v}{1-\beta_2^t}$$

At step 1 with $\beta_2 = 0.95$: $1 - 0.95 = 0.05$, and dividing by 0.05 multiplies by 20, which is
exactly the missing factor. As $t$ advances, $\beta^t \to 0$ and the correction fades out on its
own.

**And watch the `t`: it starts at 1, not 0.** With $t=0$, $1 - \beta^0 = 0$ and you are dividing by
zero. Increment the counter **before** using it. It is the exercise's first classic mistake.

### The W in AdamW

*Weight decay* means pushing weights towards zero so they do not grow without control. There are
two ways to do it and the difference is the whole letter W:

```
   Adam + L2:   g ← g + λ·p       and then Adam processes that g
   AdamW:       p ← p − lr·λ·p    directly on the parameter, separately from Adam
```

In the first one, the decay goes through the division by $\sqrt{v}$, so its real effect ends up
depending on the magnitude of that parameter's gradients: a weight with large gradients barely
decays, one with small gradients decays enormously. Nobody wants that. Loshchilov and Hutter (2019)
decoupled it and it worked better consistently.

**This is the second classic mistake**: adding the weight decay to the gradient instead of applying
it to the parameter. There is a test that tells the two versions apart
(`test_the_weight_decay_is_decoupled`).

### How an optimizer is written in PyTorch

The `__init__` **is already done**. Your only job is the `step()` method, and it has a fixed shape:
two nested loops with six operations inside. Three things about the API to know before writing it:

**`self.param_groups`** are exercise 4's groups: each with its own parameters and its own
`weight_decay`. That is why the hyperparameters are read **inside** the group loop and not once at
the top — if you read them outside, every parameter would share the same decay and exercise 4 would
be pointless.

**`self.state[p]`** is a per-parameter dictionary where you keep $m$, $v$ and the step counter. The
first time you touch a parameter it is empty (`len(state) == 0`) and has to be initialized. PyTorch
serializes it on its own in `optimizer.state_dict()`, which is what lets you resume a training run
midway — and with 8.9 million parameters that is two additional tensors per parameter, the 71.5 MB
from module 10's memory breakdown.

**The `@torch.no_grad()`** is mandatory. You are modifying parameters that have
`requires_grad=True`; without it you would be building an autograd graph over the updates
themselves, which besides being conceptually wrong would eat the memory.

### The in-place operations

The docstring writes the updates like this:

```python
m.mul_(beta1).add_(grad, alpha=1 - beta1)            # m = beta1*m + (1-beta1)*g
v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v = beta2*v + (1-beta2)*g²
p.addcdiv_(m, denom, value=-step_size)               # p -= step_size * m/denom
```

The trailing underscore means **in-place**: it modifies the tensor instead of creating a new one.
With 8.9 million parameters, allocating new tensors on each of the 10,172 steps adds up.

If they look cryptic, write it with ordinary operations first (`m = beta1*m + ...`) and optimize
later: the test compares results, not style. But **watch out for one thing**: with the non-in-place
version you are creating new tensors, so you have to store them back into `state["exp_avg"]`
yourself. If you do not, the state stays at zeros forever and the optimizer behaves as if it had no
memory.

### How to know it is right

The test trains the same problem for 50 steps with your optimizer and with `torch.optim.AdamW`, and
compares the final weights with `torch.allclose`. It is an external oracle: either you match the
world's reference implementation, or you do not.

**A warning about the demo, so you do not lose your mind.** The demo trains 200 steps to draw the
curve, but it compares weights at 50, just like the test. The reason is numerical and worth
understanding: with that task the loss is practically converged by step 100, and then $m$ and $v$
are both nearly zero. The quotient $m/(\sqrt{v}+\epsilon)$ has a tiny numerator and a tiny
denominator, so any last-bit difference between two implementations gets amplified without limit.
Measured against the reference:

```
    50 steps  ->  error 8e-07     (the loss is still 2.3e-01)
   200 steps  ->  error 1.5e-04
   400 steps  ->  error 4.2e-02
```

Two **identical** implementations drift apart like that. It is not a bug of yours or anybody's: it
is what happens when you divide tiny numbers by tiny numbers.

---

## Exercise 2: the learning-rate scheduler (`lr_at_step`)

The `lr` is not constant during training. It has two segments, and the function you write returns
the one that applies at each step.

**Warmup: going up slowly at the start.** In the first steps Adam's moments are nearly empty and
their estimates are extremely noisy — it is the same problem the bias correction attacks, but the
correction does not fully solve it. And on top of that, freshly initialized weights produce large
gradients. Starting at full `lr` tends to produce a loss spike the model sometimes never recovers
from. It goes up linearly from nearly 0 to `lr` over 500 steps.

**Cosine: coming down at the end.** At the start you want to move fast and explore; at the end, to
fine-tune in a good region. The cosine comes down slowly, then fast, then slowly again:

$$\text{lr}(t) = \text{lr}_{\min} + (\text{lr} - \text{lr}_{\min}) \cdot \frac{1 + \cos(\pi \cdot \text{progress})}{2}$$

The difference against a straight line is small but consistent across every paper that has measured
it.

And it **does not decay to zero**, but to 10% of the initial `lr`: below a certain point the model
stops learning altogether and every extra step is wasted compute. If you are going to stop, better
to stop.

### The numbers for the final run

Here is how it looks with the config's values (`lr=1e-3`, `warmup=500`, 10,172 steps):

| step | lr | segment |
|---|---|---|
| 0 | 2.000e-06 | warmup, starts nearly at zero |
| 250 | 5.020e-04 | warmup, halfway |
| 500 | 1.000e-03 | end of warmup: the maximum |
| 2,500 | 9.083e-04 | cosine |
| 5,086 | 5.865e-04 | cosine, halfway through training |
| 10,172 | 1.000e-04 | end: the 10% floor |
| 12,000 | 1.000e-04 | past the end: stays at the floor |

### Check the formula by hand before running anything

It is the way to know the cosine is right without running the tests:

```
   progress = 0  ->  cos(0) = 1    ->  coef = 1  ->  returns lr
   progress = 1  ->  cos(π) = −1   ->  coef = 0  ->  returns min_lr
```

If it comes out backwards, you have left out the `0.5 * (1 + ...)` and are using the raw cosine,
which goes from 1 to −1 instead of 1 to 0.

### Three details that look decorative and are not

**The `+1` in the warmup.** It is `lr * (step + 1) / warmup_steps`, not
`lr * step / warmup_steps`. Without it, step 0 would have an `lr` of exactly zero: a step that
learns nothing, wasted. That is why the table starts at 2e-06 and not at 0.

**The `max(1, ...)` in the denominator** avoids dividing by zero if `max_steps <= warmup_steps`.

**Clamping progress to [0, 1]** is what produces the last row of the table. The cosine is
**periodic**: with progress above 1 it would start going *up* again, and a run that overshoots its
planned steps would see the `lr` climbing out of nowhere. Clamped, it stays on the floor.

---

## Exercise 3: gradient clipping (`clip_grad_norm`)

Occasionally a batch produces enormous gradients: an odd sequence, a very infrequent token, a
corrupt line in the dataset. Without protection, that **single** batch can take a jump that
destroys hours of training, and you will see it as a vertical spike in the loss curve that the
model takes a long time to recover from, or never does.

The fix: compute the **global** norm of all the gradients together, as if they were one giant
vector, and if it exceeds a threshold, multiply them all by the same factor.

```
   norm = √(Σ ‖g_i‖²)
   if norm > max_norm:  all g ×= max_norm / norm
```

### Global, not per tensor, and this is the heart of the exercise

Clipping each tensor separately would change the **direction** of the joint gradient, which is
exactly what you do not want to touch. The gradient tells you *where* to go; you are only limiting
*how far* you move in that direction. By multiplying every tensor by the same scalar, the direction
is preserved exactly.

The demo checks it by measuring the cosine between the two directions:

```
   norm BEFORE clipping : 112,858.7
   norm AFTER           : 1.0000
   cosine between the two directions: 0.99999994
```

That cosine of 1 is the point. And the effect on a real training run, with a poisoned batch at step
50:

```
   without clipping     step 49: 0.0561  ->  step 55: 0.1698     the loss goes UP 3×
   with grad_clip=1.0   step 49: 0.0501  ->  step 55: 0.0409     it does not even notice
```

### Why the norm before clipping is returned

It is what `torch.nn.utils.clip_grad_norm_` does, and it is the useful one. If you log it and see
it rising steadily, training is destabilizing and you find out **before** it blows up. If you
returned the post-clipping norm you would see `max_norm` pinned and learn nothing.

### The generator trap

`parameters` may be a generator — `model.parameters()` is one — and **a generator is exhausted once
you walk it**. If you walk it once to compute the norm and again to multiply, the second time it is
empty: the function returns the correct norm and clips nothing. No error, no warning.

That is why the list of gradients is materialized **once**, at the start, and everything works over
that list.

---

## Exercise 4: which parameters decay (`build_param_groups`)

Five lines, and the rule is surprisingly simple:

```
   parameters with 2 dimensions or more   ->  WITH weight decay
   parameters with 1 dimension            ->  WITHOUT weight decay
```

That is: matrices decay, biases and normalization scales do not. `param.dim()` gives the number of
dimensions: a weight matrix has 2, an RMSNorm scale has 1.

**Why.** Weight decay pushes weights towards zero. On a projection matrix that makes sense:
penalizing large magnitudes reduces overfitting. On an RMSNorm scale it makes none: that parameter
starts at 1 — you wrote that yourself in module 07 — and its job is to rescale the layer's output.
Pushing it towards zero is pushing the output towards zero, which is exactly the opposite of what
is needed.

Here is how our model splits:

| group | weight_decay | tensors | parameters | what is in it |
|---|---|---|---|---|
| with decay | 0.1 | 43 | 8,929,280 | matrices: embeddings and projections |
| without decay | 0.0 | 13 | 4,160 | RMSNorm scales |

Those 13 tensors and 4,160 parameters are exactly the thirteen normalization layers you counted in
module 10. This is where having them identified paid off.

**Applying decay to everything is a common mistake, gives no visible error, and degrades the
result.** It only shows up by comparing two full training runs, which is expensive. That is why it
is worth getting right from the start.

### Two details

**Skipping `requires_grad=False`.** Those parameters are not going to be updated; putting them in
the optimizer only wastes state memory, two tensors per parameter. Right now it is a minor
optimization; in module 16, with LoRA, it becomes essential because almost the whole model is
frozen.

**The tied weights.** `model.parameters()` already deduplicates by identity, so module 10's tied
embedding appears **once** and goes into the decay group, because it has 2 dimensions. Nothing
special to do.

**And the format:** a list of dictionaries, each with at least the `"params"` key. Any additional
key (`lr`, `weight_decay`...) overrides the optimizer's default **only for that group**. It is
PyTorch's standard mechanism, and it is exactly what your `AdamWScratch.step` reads when it does
`for group in self.param_groups`. That is where the circle between exercise 1 and this one closes.

---

## The piece you do not write: mixed precision

There is no exercise for this because `GradScaler` comes ready-made in PyTorch, but it has a detail
that will bite you in module 13 if you do not know it now.

On the RTX 2060 (Turing, no bf16 in hardware, module 01) training runs in **fp16**, whose range ends
at the bottom at $6\times10^{-5}$. The gradients of the deep layers are smaller than that and turn
into zero: the model stops learning from below without raising any error.

`GradScaler` fixes it by multiplying the loss by ~65,000 before the backward, which lifts all the
gradients into representable range, and dividing before the optimizer step. If any value overflows,
it discards that step and lowers the factor.

**And here is the silent detail:** with AMP you have to **unscale the gradients before clipping
them**. If you do not, their norm is multiplied by 65,000, and your `clip_grad_norm(params, 1.0)`
from exercise 3 would be clipping to an effective threshold 65,000 times smaller than you think.
Training crawls and nothing indicates why.

```python
scaler.unscale_(optimizer)      # this first
clip_grad_norm(params, 1.0)     # and now yes
```

---

## Where the debate is

Adam **dominates without anyone really knowing why**. The usual justification — that it approximates
second-order information — does not survive analysis: $\sqrt{v}$ is not the diagonal of the Hessian
or anything like it. There is work suggesting its real advantage lies in scale invariance, or in how
it interacts with normalization. Still open.

The same goes for **warmup**: it is indispensable in practice and the explanations are post-hoc.
There are results suggesting that with pre-norm and careful initialization you can do without it,
which points to it compensating for problems elsewhere in the architecture — and you have pre-norm
and careful initialization, so the question is not rhetorical.

And about our config's **hyperparameters**: `lr=1e-3`, `betas=(0.9, 0.95)`, `weight_decay=0.1`,
`grad_clip=1.0`, `warmup=500`. They are standard values inherited from GPT-2/GPT-3 and eyeballed for
this scale. They are not optimal; they are reasonable. A hyperparameter sweep would probably find
something better, and would cost more compute than the training run itself — which is, incidentally,
the reason almost nobody tunes them and everybody copies the same numbers from one paper to the
next.

---

**Further reading:** Loshchilov & Hutter 2019,
[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) · Kingma & Ba 2015,
[Adam](https://arxiv.org/abs/1412.6980) · Micikevicius et al. 2018,
[Mixed Precision Training](https://arxiv.org/abs/1710.03740). Stray terms are in
[GLOSSARY.md](../../GLOSSARY.md).
