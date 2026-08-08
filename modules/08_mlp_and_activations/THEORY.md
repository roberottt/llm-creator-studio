# 08 — Every block's MLP: FFN, GELU and SwiGLU

## Why this module matters

**Because most of your model is here, and almost nobody tells you.**

When someone says a model has N parameters, everybody's intuition is that they are in the
attention. They are not. In our model:

```
   the MLP (this module)      5,160,960     57.8%
   attention (module 06)      2,457,600     27.5%
   embeddings                 1,310,720     14.7%
   normalization (module 07)      4,160      0.05%
   ───────────────────────────────────────────────
                              8,933,440
```

Inside each block, setting the embeddings aside, the proportion is even clearer: **68% of a
Transformer block is this piece and 32% is the attention.** The module everybody explains is
the small one.

And there is a deeper reason than size. Attention is a weighted average, that is, a **linear
operation**, and stacking linear operations achieves nothing: a hundred layers are equivalent
to one. What stops the whole Transformer from collapsing into a single matrix multiplication is
precisely this module. The demo measures it, and the number is blunt: five linear layers with
no activation give the same result as a single matrix, to within $5.6 \times 10^{-8}$, which is
floating-point noise.

### What you will know by the end

- **What an FFN actually is**, why it has three different names and why this module is called
  "MLP and activations"
- Why without a nonlinearity a network's depth is an illusion, with the measurement
- What happens to a ReLU neuron when it drifts into the negative zone (it dies, literally) and
  why GELU lets it come back
- What those $d_{ff}$ intermediate neurons really do, which is the part almost nobody explains
- What a **gate** is and how it differs from a normal activation, with the numeric example
- **Where the 896** in the final model's config comes from, and why the adjustment that
  produces it does not add up as neatly as it is usually told
- A case where the paper's own author writes that he does not know why it works

### What you are going to write

Three exercises. This theory is ordered so that you read them in this order, and **each one has
its own section with the matching numeric example**:

| Exercise | What it does | Where it is explained |
|---|---|---|
| 1. `gelu` | The classic nonlinearity | [§ GELU](#exercise-1-a-soft-cut-gelu) |
| 2. `swiglu_hidden_dim` | Arithmetic: the 896 in the config comes from here | [§ The 896](#exercise-2-where-the-896-comes-from-swiglu_hidden_dim) |
| 3. `SwiGLU` | The gated FFN the model uses | [§ SwiGLU](#exercise-3-adding-a-gate-swiglu) |

Exercise 1 is one line transcribing a formula. Exercise 2 is **the shortest in the course**,
three lines of integer arithmetic, and it produces a number you have already seen in the config
YAML. Exercise 3 is five lines. As in module 07, the work is not in the typing: it is in
understanding what each piece does and why it is there.

### What it costs

1.5 hours.

---

## What an FFN is, and why the module is called "MLP and activations"

First of all, the vocabulary, because this module drags along a mess of names that is not your
fault: **three different terms for the same box.**

### Feed-forward: what it means

**FFN** stands for *feed-forward network*. It is the oldest and simplest class of network there
is, and the name literally describes how the information flows: it goes in one side, crosses
the layers in order, and comes out the other. No loops, no going back, and no looking sideways.

That "no looking sideways" is the part that matters here, and it is best understood by contrast
with what you have already seen:

```
   RECURRENT network    the output feeds back in; it processes the sentence token by token
                        (what dominated before 2017)

   ATTENTION (mod. 06)  each token looks at the other tokens
                        information moves SIDEWAYS

   FEED-FORWARD (today) each token is processed on its own, with no idea the others
                        exist. Information only goes FORWARD
```

That is why the full name in the 2017 paper is *position-wise feed-forward network*:
"position-wise" means it is applied **per position**, the same function to each token
separately. If you hand it a `(B, T, 320)` tensor, the FFN does `B × T` independent
computations.

### It is module 02's MLP, literally

And now the good news: **you have already built one.** In module 02 you assembled an `MLP` by
hand with your own derivative engine, and it was exactly this:

- a **neuron** = a weighted sum of its inputs + a bias, passed through a nonlinear function
- a **layer** = several neurons looking at the same input
- an **MLP** (*multi-layer perceptron*) = chained layers, one's output being the next one's
  input

A Transformer's classic FFN **is a two-layer MLP**. No more, no less. The only things that
change from module 02's are the sizes and the nonlinear function:

```
   module 02:   MLP(3, [8, 8, 1])        tanh    113 parameters in total
   module 08:   MLP(320, [1280, 320])    GELU    819,200 parameters per block
```

(That `MLP(320, [1280, 320])` is the classic FFN, the one from the 2017 paper. Our model's is a
variant with one extra matrix, SwiGLU, and it shows up in exercise 3: same skeleton, 860,160
parameters. But the way to think about it is the one above.)

That `tanh` you put in because module 02 asked for it is the direct ancestor of the GELU you are
about to write, and module 02 already announced it: *"in module 08 you will see why transformers
use GELU instead of `tanh`"*. That is the second half of this module's name, the
**activations**: the nonlinear function that goes inside, between the two layers. Exercise 1 is
one of them and exercise 3 uses another.

### The three names

| name | where it comes from | where you will see it |
|---|---|---|
| **FFN** | *feed-forward network*, how the information flows | Transformer papers |
| **MLP** | *multi-layer perceptron*, how it is built | the code (`llmfs`, nanoGPT, Llama), and this module's name |
| **feed-forward** | the same as FFN, spelled out | the diagrams |

All three are the same box in the drawing. In this course you will mostly see "FFN" in the
theory and "MLP" in file and directory names, and there is no difference between them.

**Watch out for one name collision**, which is why this confuses everybody: in module 02, "MLP"
meant **the whole network**. In a Transformer, "MLP" is **a sub-block of each layer**, the box
next to the attention. The same term for two things of very different scale, depending on
context. It is flagged in the [GLOSSARY.md](../../GLOSSARY.md) too.

With that clear, the rest of the module is about three things: why this box has to exist, which
nonlinear function goes inside it, and which specific variant (SwiGLU) our model uses.

---

## Which part of the LLM this is

This module closes off the Transformer block. Look at module 07's drawing with the last box
filled in:

```
    x ──┬──> norm ──> attention ──┐
        │  (mod 07)   (mod 06)    ├──> +  ──┬──> norm ──> FFN ──┐
        └─────────────────────────┘         │  (mod 07)  (TODAY)├──> +
                                            └───────────────────┘

    what is left:  module 09  ->  telling the model which position each token is at
                   module 10  ->  stacking six of these blocks and assembling the GPT
```

And it is worth keeping in mind how the work is split between the two big boxes, because it is
the underlying structure of every Transformer:

```
   ATTENTION                          FFN
   moves information BETWEEN tokens   processes the information INSIDE each token
   mixes positions                    does not look at the other tokens at all
   linear (a weighted average)        nonlinear
   32% of the block's parameters      68%
```

That is why they alternate: move, process, move, process. Six times. The FFN not looking at the
other tokens has a practical consequence you will appreciate when writing exercise 3: **there is
no mask or anything like it**. If your tensor is `(B, T, d_model)`, the FFN applies exactly the
same function to each of the `B × T` vectors separately. You could flatten it to
`(B*T, d_model)`, pass it through, and unflatten it, and the result would be identical. There is
a test that checks this.

---

## The problem: attention alone is not enough

Look at what attention does: it mixes vectors with weights. A weighted average. And a weighted
average is a **linear operation**.

That is a serious problem, and two lines of algebra show it. Imagine you stack two linear layers
with nothing in between:

```
   layer 1:  y = W₁ · x
   layer 2:  z = W₂ · y = W₂ · (W₁ · x) = (W₂ · W₁) · x
```

$W_2 W_1$ is **a single matrix**. A hundred stacked linear layers are exactly equivalent to one
linear layer, with far more parameters and not one bit more capacity. All the depth collapses.

The demo checks this instead of asserting it: it stacks 5 linear layers, multiplies their five
matrices to get a single one, and compares.

```
   5 stacked layers  vs  1 single matrix          max difference: 5.59e-08
   the same 5 layers WITH GELU  vs  1 matrix      difference:     0.298
```

The first number is not "similar", it is **zero**: $5.6\times10^{-8}$ is what accumulates when
you multiply 32-bit floats. The second says that the moment you put a nonlinearity between the
layers, the network stops being reducible.

For stacking to be worth anything you need something nonlinear between layer and layer. That is
the FFN's job, and the reason it exists.

## The classic shape: expand, bend, contract

$$\text{FFN}(x) = W_2 \cdot \text{activation}(W_1 x)$$

With $W_1$ going $d \to 4d$ and $W_2$ going $4d \to d$. It expands to four times the size,
applies the nonlinearity, and compresses again. With our dimensions and any one token:

```
   x              (320,)      the token's vector, coming out of the norm
     │  W₁
     ▼
   h             (1280,)      expanded 4×
     │  activation            <- here is the nonlinearity, and only here
     ▼
   h'            (1280,)
     │  W₂
     ▼
   output         (320,)      same shape as the input, ready for the residual
```

**Why 4x?** Honestly: because the 2017 paper said so and it worked. There is no derivation.
Other factors have been tried and 4 is still a reasonable point, but it is convention, not
theorem.

### What those 1280 middle neurons do

This is the part that is rarely explained and that stops the FFN looking like "just another
layer".

Look at the operation by rows and columns instead of as two matrices:

- **Each row of $W_1$ is a detector.** Number $i$ of the expanded vector is the dot product of
  row $i$ of $W_1$ with the token. That is exactly the same operation as in module 06: it
  measures **how similar** the token is to a specific pattern stored in that row. If it is
  similar, a large number comes out; if not, a small or negative one, and the activation
  squashes it.
- **Each column of $W_2$ is what gets written back.** If detector $i$ fires, column $i$ of $W_2$
  gets added to the output, scaled by how much it fired.

Put the two together and the FFN reads as: *"if the token looks like pattern $i$, add vector $i$
to the residual stream"*. With 1280 pattern-response pairs per layer, and six layers.

This is called the **key-value memory** interpretation of the FFN: the rows of $W_1$ are the
keys, the columns of $W_2$ the values. It is a hypothesis with evidence behind it — there is work
that locates specific factual claims in specific neurons and edits them by changing those
weights — but it is **not an established result**, and it is worth reading as a useful way to
think about the operation rather than as what the network really does. What is literal is the
rows-and-columns mechanics of the previous paragraph: that is arithmetic, not interpretation.

---

## Exercise 1: a soft cut (`gelu`)

### The problem: ReLU and dead neurons

The simplest nonlinearity is ReLU: $\max(0, x)$. It lets the positive through and zeroes the
negative. It works, it is dirt cheap, and it has a flaw you can see by looking at the derivative:

| x | ReLU | dReLU/dx | GELU | dGELU/dx |
|---|---|---|---|---|
| −3.0 | 0.0000 | **0.0000** | −0.0036 | −0.0116 |
| −1.0 | 0.0000 | **0.0000** | −0.1588 | −0.0830 |
| 0.0 | 0.0000 | — | 0.0000 | 0.5000 |
| +1.0 | 1.0000 | 1.0000 | 0.8412 | 1.0830 |
| +3.0 | 3.0000 | 1.0000 | 2.9964 | 1.0116 |

With ReLU, the derivative across the whole negative zone is **exactly zero**. And now remember
what a zero derivative means, which you saw in module 07: no gradient arrives. If during training
a neuron ends up always producing negative values, it stops receiving any signal **forever**: it
never updates, so it never gets out of there. It is dead and there is no resurrecting it. The
phenomenon has its own name, *dying ReLU*, and in large networks it can kill a far from
negligible fraction of the neurons.

### The fix: multiply by a probability

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the probability that a standard normal comes out smaller than $x$.

The intuition: instead of deciding with a hard cut whether to let $x$ through, it multiplies it
by the probability that $x$ "stands out". With numbers:

```
   x = -3   ->  Φ(-3) = 0.001   ->  GELU = -0.0036    almost cancelled
   x = -1   ->  Φ(-1) = 0.159   ->  GELU = -0.1588    partly
   x =  0   ->  Φ(0)  = 0.5     ->  GELU =  0.0000
   x =  1   ->  Φ(1)  = 0.841   ->  GELU =  0.8412    almost whole
   x =  3   ->  Φ(3)  = 0.999   ->  GELU =  2.9964    whole
```

Those are exactly the five values your function has to return, and the test checks them one by
one.

The transition is smooth, so the derivative is never exactly zero: look again at the `dGELU/dx`
column of the table, −0.0116 at $x=-3$. It is a tiny gradient, but not a null one, and that is
enough for a neuron sunk in the negative zone to come back.

### What you actually write

In practice a `tanh` approximation is used, because `erf` (the function needed for the exact
$\Phi$) was slow on 2016 GPUs:

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}\,(x + 0.044715x^3)\right]\right)$$

And that is the exercise: one line transcribing that formula, with no loops and no conditionals.
The two possible mistakes are mistyping a constant ($\sqrt{2/\pi} \approx 0.7978$) or regrouping
the expression in a way that changes the order of operations.

**Careful what you compare against.** Your result has to match
`F.gelu(x, approximate="tanh")`, **not** plain `F.gelu(x)`: they are different functions — the
second uses `erf` — and the test compares against the first. Today the speed difference between
them is irrelevant, but GPT-2 was trained with the approximation and for compatibility it is
still used everywhere.

### A warning about the demo's table

The demo plots the three activations over a grid of 400 points between −4 and 4 and then prints
the row closest to each round value. That is why its table says things like
`GELU(+0.0) = -0.0050`, which throws you if you have just read here that GELU(0) = 0 exactly. It
is not a contradiction: that row is really $x = -0.01$, the grid point nearest zero. The exact
values are the ones above.

### GELU and Swish, almost the same curve

In the demo you will also see **Swish** (which PyTorch calls `SiLU`), $z \cdot \sigma(z)$, and it
is the one you will use in exercise 3. It is worth noticing that the two curves are almost
identical despite having completely different origins: GELU comes from a probabilistic argument
and Swish from an automated search over activation functions. They ended up in practically the
same place, which suggests that what matters is the *shape* — smooth, near-identity in the
positive range, non-null attenuation in the negative — and not the specific formula.

---

## Exercise 2: where the 896 comes from (`swiglu_hidden_dim`)

This is the shortest exercise in the course: three lines of integer arithmetic, no tensors. And
it produces a number you have already seen, the `d_ff: 896` from the config. It is one of those
exercises whose entire value is in understanding **why** that number is that number.

There are two chained adjustments, and they are independent of each other.

### Adjustment 1: the 2/3 factor

SwiGLU (exercise 3) has **three** matrices where the classic FFN has two. With the same $d_{ff}$
that would be 50% more parameters, and then any comparison between the two architectures would
be rigged: you would not know whether SwiGLU wins by being better or by being bigger.

The fix is to shrink $d_{ff}$ to two thirds so the budgets line up:

```
   classic FFN:  2 matrices × d × 4d           = 8d²
   SwiGLU:       3 matrices × d × (2/3 · 4d)   = 3 · d · (8/3)d = 8d²   ✓
   SwiGLU unadjusted: 3 × d × 4d               = 12d²                   +50%
```

With our $d_{\text{model}} = 320$: $(2/3) \times 4 \times 320 = 853.33$, truncated to 853.

### Adjustment 2: round up to a multiple of 64

$853 \to 896$. And this is not cosmetic: dimensions aligned to powers of two let the GPU's tensor
cores use their fast paths. A 853-column matrix is noticeably slower than an 896-column one
**while having fewer parameters**, which is one of the most counterintuitive things in module 01
and here you see it applied.

The rounding is written without `math.ceil` and without floats:

```python
multiple_of * ((hidden + multiple_of - 1) // multiple_of)
```

Adding `multiple_of - 1` before the integer division forces rounding up, and if the value was
already an exact multiple it leaves it alone. It is the standard idiom for this and it is worth
recognizing.

The course's two cases, so you can check:

```
   d_model = 320:  int(2*1280/3) = 853  ->  64 * ((853+63)//64) = 64 * 14 = 896
   d_model = 128:  int(2*512/3)  = 341  ->  64 * ((341+63)//64) = 64 *  6 = 384
```

896 is the final model's `d_ff`; 384 is the toy config's.

### The 2/3 does not add up as neatly as it is usually told

Here is a detail that is almost never mentioned and that only shows up if you do the arithmetic
for several sizes. Adjustment 2's rounding breaks adjustment 1's balance, and how much it breaks
it depends on the model's size:

| d_model | d_ff | classic FFN | SwiGLU | extra |
|---|---|---|---|---|
| 128 | 384 | 131,072 | 147,456 | **+12.5%** |
| 320 | 896 | 819,200 | 860,160 | **+5.0%** |
| 768 | 2048 | 4,718,592 | 4,718,592 | 0.0% |
| 4096 | 10944 | 134,217,728 | 134,479,872 | +0.2% |

That is: **the 2/3 equalizes the budgets asymptotically, not exactly.** In large models the
rounding to 64 is a marginal adjustment and the equality holds almost perfectly; in a small model
like ours, 64 is an appreciable fraction of $d_{ff}$ and we end up spending 5% more. It is not a
problem — 5% changes no conclusion — but it is worth knowing before you read somewhere that
SwiGLU "costs exactly the same".

---

## Exercise 3: adding a gate (`SwiGLU`)

Here comes the change our model uses, and every modern one.

The idea behind the **GLU** (*Gated Linear Unit*) variants is to have **two** projections instead
of one, both coming from the same input. One of them acts as a **gate**: it multiplies the other
element by element and decides how much signal passes through each dimension.

$$\text{SwiGLU}(x) = \big(\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}}\big) W_{\text{down}}$$

with $\text{Swish}(z) = z \cdot \sigma(z)$, the curve almost identical to GELU from the previous
section. In PyTorch it is called `F.silu`, and you can write it by hand as `x * torch.sigmoid(x)`
without anything changing except that you lose the fused kernel.

### The route, with the shapes

```
   x                    (B, T, 320)
     ├── gate_proj ──> (B, T, 896) ── Swish ──┐
     │                                        ⊙   ELEMENT-BY-ELEMENT multiplication
     └── up_proj   ──> (B, T, 896) ───────────┘
                                              │
                                          (B, T, 896)
                                              │  down_proj
                                              ▼
                                        (B, T, 320)      same shape as the input
```

Both branches come out with the same shape, so the `⊙` is a Python `*`, **not** a `@`. If you put
`@` the shapes would not even line up, which is one of the few mistakes in this module that does
raise an immediate error.

### What the gate does, with numbers

This is the conceptual difference of the module. Imagine that at one particular position the two
branches produce these three numbers (in the real model there would be 896):

```
   up    = [ 2.0,  -1.0,   4.0]      the content
   gate  = [ 3.0,  -3.0,   0.0]      the gate, before Swish

   Swish(gate) = [2.8577, -0.1423, 0.0000]

   product     = [5.7154,  0.1423, 0.0000]
                    │         │       │
                    │         │       └── dimension SHUT: nothing gets through
                    │         └────────── nearly shut, and with the sign flipped
                    └──────────────────── wide open, amplified ×2.9
```

Compare that with a normal activation, which would apply the same function to `up` and be done.
Here the network decides, **for each dimension and for each token**, how much to let through —
and that decision is computed from the input itself, with learned weights. The third dimension is
shut off entirely not because its content was small (it was 4.0, the largest of the three) but
because the gate decided it was not relevant in this context.

It is the same kind of idea as attention — letting the content decide what gets through — applied
inside a token instead of between tokens.

### The details that break if you do not watch them

**The activation goes on `gate_proj`, not on `up_proj`.** Numerically the module would work just
as well with the assignment reversed: it is symmetric apart from which weights learn what. But it
**would not match the reference** when copying weights and the test would fail with a difference
that is hard to interpret. There is a test dedicated to pointing this out.

**The names matter**: `gate_proj`, `up_proj`, `down_proj`. The test copies weights by name, just
like in modules 05 and 06.

**`bias=False` by default**, which is the final model's config. That is what makes the parameter
count come out at exactly $3 \cdot d_{\text{model}} \cdot d_{ff} = 3 \times 320 \times 896 =
860{,}160$, with no loose addends. There is a test that checks that exact number.

**The `dropout` goes at the end**, on `down_proj`'s output, and in this course's model it is 0. It
is the same piece as in module 06 and it is explained in module 11.

---

## Where the parameters are

With the final config, per layer:

| component | parameters | % of the block |
|---|---|---|
| attention ($4d^2$) | 409,600 | 32% |
| SwiGLU ($3 \cdot d \cdot d_{ff}$) | 860,160 | **68%** |

And the proportion holds whatever the size, because both grow with $d^2$:

| d_model | d_ff | attention | FFN | % FFN |
|---|---|---|---|---|
| 128 | 384 | 65,536 | 147,456 | 69% |
| 320 | 896 | 409,600 | 860,160 | 68% |
| 768 | 2048 | 2,359,296 | 4,718,592 | 67% |
| 4096 | 10944 | 67,108,864 | 134,479,872 | 67% |

Two thirds of every block is FFN. Over the whole model the figure drops to 57.8% because the
embeddings take their share, but the idea holds: **when you read that a model has N parameters,
most of them are here, not in the attention.**

## Where the debate is

This module is probably where the "we do not know why" is most explicit, and it comes from the
author himself.

Shazeer (2020) systematically tried every GLU variant and SwiGLU came out best consistently. His
conclusion, quoted literally from the paper:

> *"We offer no explanation as to why these architectures seem to work; we attribute their
> success, as all else, to divine benevolence."*

It is not a joke: it is honesty about the state of the matter. SwiGLU is used today in Llama,
Mistral, PaLM and almost everything else, and the justification is that it does better on the
benchmarks. There is no theory.

The same goes for the 4x, a constant nobody has derived, and for the key-value memory
interpretation of the FFN, which is a reasonable hypothesis with partial evidence. And watch out
for what the demo teaches too: its fourth experiment trains a classic FFN and a SwiGLU at roughly
equal parameter counts on a made-up task, and SwiGLU wins by a wide margin. That **proves
nothing** about language models, and the demo itself says so. A toy experiment that confirms what
you already believed is the easiest way to fool yourself.

All of this is worth keeping in mind when you read explanations that sound very sure of
themselves, including the ones in this file.

---

**Further reading:** Hendrycks & Gimpel 2016,
[Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) · Shazeer 2020,
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (the paper with the quote) ·
Geva et al. 2021, [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913).
Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
