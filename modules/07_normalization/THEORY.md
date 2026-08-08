# 07 — Normalization and residual connections: the plumbing that makes it train

## Why this module matters

**Because without this, a deep network does not train. Full stop.**

Two pieces that compute nothing interesting and that are the difference between a model that
learns and one that returns `NaN` after three steps. They are the Transformer's plumbing:
nobody mentions them in the headlines and without them there is nothing.

To show you just how much plumbing they are, one number: the thirteen normalization layers of
the final model add up to **4,160 parameters out of 8,933,440**, or 0.047%. And they are
non-negotiable. This module is about the two cheapest pieces in the whole model and about why
without them there is no model.

The problem they solve is concrete and you are going to see it measured: the numbers flowing
through a deep network tend to explode or vanish, and with 64 layers the gradient that reaches
the bottom is **exactly zero**. Not "very small": zero, by floating-point underflow. The demo
measures it and the table is further down.

On top of that, this is one of the design decisions where you learn most by comparing: where
you put the normalization changes whether your network needs warmup or not.

### What you will know by the end

- Why the numbers get out of control when you stack layers, with the arithmetic that explains
  it and the measurement that confirms it
- What LayerNorm does exactly, and **what is surplus to it** (that is RMSNorm)
- Why `x + f(x)` is one of the most important ideas in deep learning
- Why exercise 1 is a plain function and exercise 2 is a class: it is the first time you write
  a layer **with weights of its own**
- Pre-norm against post-norm, measured: how much gradient reaches the first layer in each
  case, and why the usual argument in favour of pre-norm is half wrong
- Two numerical traps that raise no error and ruin the result: `unbiased` and fp16

### What you are going to write

Three exercises. This theory is ordered so that you read them in this order, and **each one
has its own section with the matching numeric example**:

| Exercise | What it does | Where it is explained |
|---|---|---|
| 1. `layer_norm` | Center at 0 and scale to variance 1 | [§ LayerNorm](#exercise-1-layernorm-by-hand-layer_norm) |
| 2. `RMSNorm` | The same without the mean: what Llama uses and what we use | [§ RMSNorm](#exercise-2-dropping-half-of-it-rmsnorm) |
| 3. `prenorm_residual` | One line, and the most important one in the module | [§ Pre-norm](#exercise-3-where-the-parentheses-go-prenorm_residual) |

All three are short: the first is five lines, the second three, and the third is **one**. It is
the module with the least code in the course and it is not the easiest, because what you learn
here is not in the code but in understanding what breaks without it. The 1.5 hours go into the
theory and the demo, not into typing.

A note about exercise 2, because it is where most people wonder whether they missed something:
`layer_norm` is a **function** and `RMSNorm` is a **class inheriting from `nn.Module`**. It is
not a whim or an inconsistency; the reason is in
[its section](#why-1-is-a-function-and-2-is-a-class).

### What it costs

1.5 hours.

---

## Which part of the LLM this is

This is the Transformer block you started in module 06. What you already have is the box in the
middle; what you assemble today is **everything else in the drawing**:

```
    x ──┬──> NORM ──> attention (module 06) ──┐
        │   (ex. 1-2)                         ├──> +  ──┬──> NORM ──> MLP ──┐
        └─────────────────────────────────────┘  (ex.3) │  (ex. 1-2)  (mod 8)├──> +
                                                        └────────────────────┘
                                                                        (ex. 3)
```

The two norms and the two additions in every block. Six blocks, plus a final normalization
before the logits that will be called `norm_f` and which we will get to in exercise 3. Thirteen
normalization layers in total:

```
   6 blocks × 2 norms   =  12
   the final one (norm_f) =  1
   ──────────────────────────────
                            13  ×  320 parameters  =  4,160
```

And the residual additions do not have a single parameter: they are a `+`.

Compare that with the 409,600 parameters per layer of the previous module's attention and you
will see the mismatch between what this piece costs and what it contributes. That is the idea
to take away from the module: **not everything that matters in an architecture has parameters.**

---

## The problem: the numbers get out of control

A deep network is a composition of functions. Each layer multiplies by a matrix, and those
multiplications compound.

Imagine each layer multiplies the magnitude of its inputs by 1.2. It looks harmless:

```
   layer 1:   ×1.2  ->  1.2
   layer 2:   ×1.2  ->  1.44
   layer 3:   ×1.2  ->  1.73
   ...
   layer 40:  ×1.2  ->  1470
```

And if the factor were 0.8 instead of 1.2, after 40 layers you would be left with 0.00013. In
one case the numbers explode, in the other they vanish. **And the same happens to the gradient
on the way back**, which is what really hurts: if the gradient vanishes, the layers at the
bottom get no signal and learn nothing. They stay with the random weights they were born with.

With fp16, which only goes up to 65,504 and down to $6\times10^{-5}$, this stops being a
nuisance and turns into `inf` and zeros.

### The measurement, which is more brutal than the arithmetic

The demo stacks N identical blocks and measures the **norm of the gradient reaching the input**.
Here is the table, and it is worth looking at slowly because it contains the whole module:

| layers | nothing | norm only | post-norm | pre-norm |
|---|---|---|---|---|
| 4 | 3.230e-01 | 1.379e+01 | 7.062e+01 | 7.876e+01 |
| 8 | 2.709e-03 | 1.892e+01 | 6.750e+01 | 8.678e+01 |
| 16 | 1.094e-07 | 1.947e+01 | 6.387e+01 | 9.707e+01 |
| 32 | 2.086e-16 | 1.343e+01 | 5.689e+01 | 1.236e+02 |
| 64 | **0.000e+00** | 5.103e+00 | 5.941e+01 | 1.467e+02 |

The first column is a network with nothing: linear layers chained together. With 4 layers the
gradient has already lost two thirds; with 16 it is $10^{-7}$; with 64 it is **exactly zero**,
not an approximation. It has dropped below the smallest representable number and floating point
has rounded it to zero. That network does not learn slowly: its first layers **do not move at
all**, however many steps you take.

The other three columns are the three things you are about to build. Come back to this table as
you finish each exercise.

---

## Solution 1: normalize

The idea is brutal in its simplicity: **before each block, put the numbers back on a known
scale.** It does not matter what the previous layer did; on the way into the next one, you
renormalize. That cuts the chain of factors from the example: it no longer matters that one
layer multiplies by 1.2, because the next step puts everything back at scale 1.

### LayerNorm, with numbers

Take a token's vector, say with 4 dimensions:

```
   x = [2.0, 8.0, 4.0, 6.0]
```

Compute its mean and its variance:

```
   mean      = (2+8+4+6)/4 = 5.0
   variance  = ((2-5)² + (8-5)² + (4-5)² + (6-5)²)/4 = (9+9+1+1)/4 = 5.0
   std dev   = √5 = 2.236
```

Subtract the mean and divide by the standard deviation:

```
   x_norm = [(2-5)/2.236, (8-5)/2.236, (4-5)/2.236, (6-5)/2.236]
          = [-1.342, 1.342, -0.447, 0.447]
```

Now mean 0 and variance 1, wherever the input came from.

But always forcing mean 0 and variance 1 takes freedom away from the model — maybe that layer
was better off with dimension 7 being systematically large — so it is given back with two
learned parameters, $\gamma$ (scale) and $\beta$ (shift):

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

Each is a vector of size `d`, one entry per dimension, and they come out of training like any
other weight. The $\epsilon$ (typically $10^{-5}$) avoids dividing by zero when every component
is equal.

### What the mean is computed over: the point to get clear

**The mean and the variance are computed over each token's dimensions, separately.** If your
tensor is `(B, T, d)`, the normalization runs along `d`, and there are `B × T` independent
normalizations.

```
   tensor (2, 3, 4):  2 sequences × 3 tokens × 4 dimensions

   token (0,0): [2.0, 8.0, 4.0, 6.0]   ──> its own mean and its own variance
   token (0,1): [1.0, 1.0, 1.0, 9.0]   ──> its own, without looking at the one above
   token (0,2): ...
```

Nothing to do with BatchNorm, which normalizes along the batch: there each dimension is
normalized using the values of *that same dimension across every example in the batch*. That
brings two problems that are unacceptable in a language model: one example's result depends on
who it happened to share a batch with, and at inference time, with a single example, there is no
batch to get statistics from, so you have to store running averages from training and hope they
hold.

LayerNorm has neither: it treats each token independently, works the same with a batch of 1 as
with 1000, and stores nothing. That is why Transformers use it.

---

## Exercise 1: LayerNorm by hand (`layer_norm`)

It is five lines and they are the formula translated directly:

```
   1.  mean = x.mean(dim=-1, keepdim=True)                 the μ
   2.  var  = x.var(dim=-1, keepdim=True, unbiased=False)  the σ²
   3.  normalized = (x - mean) / torch.sqrt(var + eps)     the whole fraction
   4.  if weight: multiply;  if bias: add                  γ and β
   5.  return normalized
```

The `weight` and `bias` arguments are optional (`None`) because the function has to serve two
purposes: checking the pure normalization against the example above, and acting as a full layer
when you hand it the parameters. With `weight=None` and `bias=None`, your function on
`[2.0, 8.0, 4.0, 6.0]` has to return exactly `[-1.3416, 1.3416, -0.4472, 0.4472]`, which is what
PyTorch's `F.layer_norm` returns.

### The trap: `unbiased=False`

`torch.var` divides by $n-1$ by default (the *sample* variance, with Bessel's correction),
because that is what you want when estimating a population's variance from a sample. LayerNorm
estimates nothing: it normalizes the numbers it has. It uses the **population** variance, which
divides by $n$.

With our 4-component vector:

```
   correct (÷4):        variance 5.000  ->  [-1.3416,  1.3416, -0.4472,  0.4472]
   with Bessel (÷3):    variance 6.667  ->  [-1.1619,  1.1619, -0.3873,  0.3873]
```

A 13.4% difference in the result, and **no error at all**. And the size of the mistake depends
on the dimension: with `d=4` the variance comes out 33% too high, but with `d=320` — the real
size — it comes out only 0.3% too high and the difference in the result is 0.16%. That is: if
you test it with the real model, the result looks fine. This is the archetype of the bug this
course tries to teach you to fear: the one you only see if you go looking with a small example.
The test compares your result against both versions and tells you which one you match.

### The other two details

**`keepdim=True`.** Without it, `mean(dim=-1)` on `(4, 8, 32)` returns `(4, 8)` instead of
`(4, 8, 1)`, and the subtraction `x - mean` broadcasts wrongly. Sometimes it raises an error and
sometimes — when the shapes happen to line up — it silently produces garbage. It is exactly the
same `keepdim` trap as in module 05, and it will not be the last time.

**The `eps` goes inside the square root:** `sqrt(var + eps)`, not `sqrt(var) + eps`. That is what
`F.layer_norm` does, and with a small variance the difference matters: if the variance is
$10^{-8}$, the first form gives $\sqrt{10^{-5}}\approx 0.0032$ and the second gives
$10^{-4} + 10^{-5}$, about thirty times less, so you multiply the result by thirty.

---

## Exercise 2: dropping half of it (`RMSNorm`)

In 2019, Zhang and Sennrich tried something: **what if we do not subtract the mean?**

$$y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

You only rescale by the root mean square (hence the name). No subtracting the mean, no $\beta$.
With the same vector:

```
   RMS = √((4+64+16+36)/4) = √30 = 5.477
   y   = [2/5.477, 8/5.477, 4/5.477, 6/5.477]
       = [0.3651, 1.4606, 0.7303, 1.0954]
```

Notice that all four numbers are still positive: RMSNorm **centers nothing**, it only adjusts
the size. If your vector was shifted, it stays shifted.

### And that does not matter?

That is the right question, and the honest answer is: in practice it does not, and nobody knows
how to prove why. The demo measures it by feeding both the same tensor, first centered and then
deliberately shifted:

| input | mean after LayerNorm | mean after RMSNorm | max difference |
|---|---|---|---|
| mean 0 | +0.0000 | −0.0135 | 0.171 |
| mean +5 | −0.0000 | +0.9806 | 3.839 |
| mean +50 | −0.0000 | +0.9998 | 4.924 |

With already-centered data the two do practically the same thing. With a large shift they
diverge: LayerNorm removes it and RMSNorm keeps almost all of it. The thing is that **inside a
network the activations tend to be centered**, so the case where they differ hardly ever comes
up.

Zhang and Sennrich observed that almost all of LayerNorm's benefit comes from **rescaling**, not
from **recentering**, and that dropping the mean part saves one pass over the data and one
intermediate tensor: between 7% and 64% faster depending on the case, with no measurable loss of
quality. That is an **empirical** result, verified by training, not a theorem. But it is robust
enough that Llama, Mistral and practically everything modern use it, and that is why our model
uses it.

The other half of the gain is the parameters: `320` instead of `640` per layer, because there is
no $\beta$.

### Careful reading the demo's timings

The demo times the three implementations and prints this, which at first glance contradicts
everything I just told you:

```
   F.layer_norm (PyTorch)   0.158 ms      640 parameters
   your layer_norm          0.445 ms      640
   RMSNorm                  0.512 ms      320       <- the slowest?
```

Do not conclude from that that RMSNorm is worse. These layers are **memory-bound**, module 01's
term: they are limited not by how many computations they do but by moving the data between
memory and the processor. At this scale what dominates the stopwatch is the fixed cost of
launching each PyTorch kernel, and RMSNorm as you write it is several separate operations
(`pow`, `mean`, `rsqrt`, two multiplications) against `F.layer_norm`'s single compiled kernel.
You are comparing your teaching implementation against optimized C++, not one algorithm against
another.

The real gain Zhang and Sennrich report is measured over whole training runs and with fused
implementations of both. What is a checkable fact here is half the parameters.

### Why 1 is a function and 2 is a class

Here is the conceptual difference of the module, and it is about PyTorch, not about
normalization.

`layer_norm` receives its parameters as arguments: `weight` and `bias` come in through the door
and the function remembers nothing between calls. It is a pure operation.

`RMSNorm` **owns its parameter**. In `__init__` you write:

```python
self.weight = nn.Parameter(torch.ones(dim))
```

`nn.Parameter` is a tensor with a sign on it saying "this gets trained". That does three things
at once, all of them the ones you saw in module 05: it shows up in `model.parameters()`, so the
optimizer will update it; it moves to the GPU with `model.to(device)`; and it is saved with the
model. A plain `torch.ones(dim)` would do none of that — it would be a constant.

Hence exercise 2 being a class: **it is the first layer in the course that has weights of its
own not coming from an `nn.Linear` or an `nn.Embedding`**, and they are 4,160 of the final
model's 8,933,440.

**And that is why it is initialized with `torch.ones` and not with `torch.randn`.** At startup
the layer has to be the pure normalization, that is, multiply by 1. If `weight` started random
you would be scaling each dimension by an arbitrary factor before having learned anything, and
the loss at step 0 would not match `ln(V)` — module 05's bug detector, which has already caught
one case for you in the neural bigram.

### The `.float()` is not paranoia

The `forward` the docstring dictates carries a conversion that looks superfluous:

```python
return self._norm(x.float()).type_as(x) * self.weight
```

With fp16 activations, squaring overflows sooner than one expects: `300² = 90,000` and fp16 tops
out at 65,504. You can check it in two lines:

```python
>>> h = torch.tensor([300.0]).half()
>>> (h * h).item()
inf
```

And from there the whole chain collapses: the mean would be `inf`, `rsqrt(inf)` is 0, and **the
layer would return zeros**. Not `NaN`, not an error: zeros, silently. There is a test that
reproduces that exact case. Since the course's RTX 2060 trains in fp16 out of necessity (it has
no bf16 in hardware, module 01), this is not a hypothetical.

`torch.rsqrt(z)` computes $1/\sqrt{z}$ in one go: one kernel fewer than dividing, and slightly
more numerically stable.

### A detail that surprises people

Even if you call `.type_as(x)` to go back to fp16, the output ends up being **fp32**, because
you then multiply by `self.weight`, which is an fp32 parameter, and PyTorch promotes the result
to the wider of the two types.

It is not a bug: it is exactly what Llama's implementation does and it is what you want. Under
autocast the weights stay in fp32 and the following operations convert whatever they need. There
is a test that documents it, and I mention it here so you do not lose half an hour chasing a
`dtype` that is fine.

---

## Solution 2: residual connections

The second piece, and the more important of the two.

Instead of each block *replacing* the representation, it is asked to *modify* it:

$$x_{\text{output}} = x + f(x)$$

The block computes a correction, not a replacement. That accumulated sum flowing through the
whole network is called the **residual stream**, and it is a very useful way to think about a
Transformer: there is a main channel carrying each token's representation, and every block reads
from it, computes something and **adds** its contribution back.

**Why this changes everything:** differentiating that expression with respect to $x$,

$$\frac{\partial x_{\text{output}}}{\partial x} = 1 + \frac{\partial f(x)}{\partial x}$$

That **1** is a highway. Even if $\partial f/\partial x$ is tiny, the gradient reaching the
layers below does not drop below 1 along that path. Without residuals, each layer's factors
multiply together and the product vanishes; with residuals, there is always a direct route the
gradient passes along untouched.

---

## Exercise 3: where the parentheses go (`prenorm_residual`)

The whole exercise is this:

```python
return x + fn(norm(x))
```

One line. And now the part that matters: there are two ways to place the normalization, and only
the parentheses move.

```
   post-norm (the 2017 paper):     norm(x + fn(x))
   pre-norm  (everything modern):   x  + fn(norm(x))
```

It looks cosmetic. It is not, and what changes is **where the gradient goes**.

In **post-norm**, the normalization sits *on top of* the residual path: the gradient goes through
it at every layer and gets rescaled. The highway has a toll at every exit. With 6 layers you
barely notice; with 40, you need a careful learning-rate warmup so training does not explode in
the first steps.

In **pre-norm**, the normalization sits *inside* the branch. The path $x \to x$ is completely
clear and the gradient reaches the first layer intact. You can train without warmup and with
higher learning rates.

If you write `norm(x + fn(x))` you have done post-norm, and there is a test that detects it.

### How it gets checked

There is a test worth understanding because it is the whole module in three lines: it completely
kills the branch's gradient — `fn` returns a tensor detached from the graph, multiplied by zero
— and verifies that the gradient at the input is still **exactly 1.0**.

That 1 is the entire reason residuals exist. Even if the whole block were useless and contributed
no signal, the information still flows and the gradient still arrives. A Transformer block can
never make the network *worse*, because it always has the option of contributing nothing and
letting the stream through. That is the underlying reason you can stack 100 layers.

### What the table actually says, which is not what usually gets told

Go back now to the table at the top, to the last two columns, at 64 layers:

```
   nothing      0.000e+00      <- dead
   norm only    5.103e+00      <- alive
   post-norm    5.941e+01
   pre-norm     1.467e+02
```

The usual reading is "pre-norm stops the gradient vanishing". Looking at the numbers, that is
half wrong: **normalization on its own already rescues the problem**. Going from exactly zero to
5.1 is the big jump, and it is made by the column that does not even have residuals.

What really distinguishes pre-norm is something else, and you can see it in the shape of the
column: it is the only one of the four where the gradient **grows** with depth (78 → 146) instead
of shrinking. The path $x \to x$ has no toll at all, so every layer you add contributes rather
than attenuating what came before. Normalization and residuals attack the same problem by
different routes and they are not alternatives, they are complements.

### The consequence: `norm_f`, and why it exists

Pre-norm has a price. Since the residual stream **is never normalized along the way** — each
layer adds its contribution and nobody adjusts it again — it arrives at the end at a scale that
grows with depth. It can be measured on the real model, taking the average norm of each token's
vector on the way out of each block (freshly initialized model, untrained):

```
   after the embeddings:  0.357
   after block 1:         0.450
   after block 2:         0.559
   after block 3:         0.692
   after block 4:         0.868
   after block 5:         1.002
   after block 6:         1.183     <- 3.3 times what went in
   ──────────────────────────────
   after norm_f:         17.886     <- √320, i.e. RMS 1 per component
```

If those arbitrarily-scaled vectors went straight to the logits layer, the loss at step 0 would
not be `ln(V)` and the model would start out with strong opinions by accident — module 05's "bad
initialization" case, again.

That is why **pre-norm models always carry a final normalization** before the output layer. In
module 10 it will be called `norm_f`, and now you know where it comes from: it is not an
ornament, it is the mandatory patch for the design you have just written.

---

## Where the debate is

There is more here than meets the eye.

**Why normalization works is still unclear.** BatchNorm's original explanation (2015) was
*internal covariate shift*: that normalizing stabilizes the distribution of each layer's inputs.
Santurkar et al. (2018) put that to the test by injecting noise *after* normalizing —
deliberately destroying that stability — and the network kept training just as well. The original
explanation is now largely discarded, and its replacement — that it smooths the loss landscape —
is more an empirical observation than a theory. In other words: you are using two indispensable
pieces whose mechanism nobody can fully explain, and that is more normal in this field than
anyone would like.

**That RMSNorm suffices is empirical.** The section above says so and it bears repeating: there
is no theoretical reason why recentering should be dispensable. It was verified by training, it
has held up since 2019 and it has spread to almost every large model, which is the best evidence
available; but it is evidence, not proof.

**Pre-norm is not free.** It is fairly well accepted that pre-norm trains more easily, but there
is evidence that post-norm, when it converges, reaches better final quality. Some recent
architectures use hybrid variants for exactly this reason. We use pre-norm because it is the
robust choice and what everybody does, not because it has been proven superior.

---

**Further reading:** Ba et al. 2016, [Layer Normalization](https://arxiv.org/abs/1607.06450) ·
Zhang & Sennrich 2019, [RMSNorm](https://arxiv.org/abs/1910.07467) · Xiong et al. 2020,
[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
(the pre/post-norm analysis) · He et al. 2015,
[Deep Residual Learning](https://arxiv.org/abs/1512.03385). Stray terms are in
[GLOSSARY.md](../../GLOSSARY.md).
