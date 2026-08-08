# 01 — Environment and hardware

## Why this module matters

**This module saves you a week.**

You are going to train a model on your own computer. Before writing a single line of neural
network it is worth knowing whether that will take two hours or two weeks, because the
answer changes every decision that follows: how many layers, what context, what batch size,
and whether the final run gets launched on a Friday afternoon or has to be rethought from
scratch.

There is a second reason, less obvious and more important in the long run. Almost everyone
who trains models copies the numbers from a tutorial without knowing where they come from,
and when something does not add up — it does not fit in memory, it runs four times slower
than expected — they have nothing to diagnose it with. By the end of this module you will be
able to **compute** what a model costs before building it. That is what separates designing
from copying.

A word on expectations, because this module is unlike the others: nothing here is part of
the model. What you build here is the measuring instrument. Three short functions and a demo
that genuinely interrogates your machine.

### What you will know by the end

- How many operations it costs to process one token, and **where the formula comes from**,
  term by term.
- How many TFLOPS your GPU really delivers, as opposed to what the spec sheet claims — and
  why measuring it badly is so easy that the exercise is built around the three ways to get
  it wrong.
- Why small matrices leave a GPU idling, with the measured curve.
- Why a GPU without bf16 is forced into `float16`, and what silent problem that brings.
- **Where the memory goes** in a training run, which is almost never where you think.
- How to estimate how long a training run will take before launching it.

### What it costs

45 minutes. Three short functions, and the demo measures your hardware for real.

---

## 1. The question: how do you measure "what it costs"?

A computer does not take the same time on two different tasks, so we need a common unit. The
one used is the **FLOP**: an operation on decimal numbers, an addition or a multiplication.
Training a model is a great many FLOPs, and a GPU can do a few trillion per second.

Two numbers and one division:

```
time = total FLOPs to be done / FLOPs per second my machine delivers
```

The whole module is about estimating those two numbers well. The numerator can be worked out
with pencil and paper — it is exact arithmetic over the architecture — and the denominator
**has to be measured**, because the one printed on the box bears no resemblance to the one
you are going to get.

## 2. Where a network's FLOPs come from

Almost everything a neural network does is **multiply matrices**. So let us start by counting
exactly what one costs, with numbers you can follow by hand.

Multiply a 2×3 matrix by a 3×2 one. The result is 2×2, that is, 4 numbers. Each of those 4
comes from pairing up 3 values with another 3, multiplying them and adding them together: 3
multiplications and 2 additions, which by convention we round to 6 operations (2 per pair).
Total:

```
4 output numbers × 6 operations = 24 FLOPs
```

In general, multiplying an $m \times k$ matrix by a $k \times n$ one costs $2mnk$ FLOPs.

Now the step that turns this into a rule you will use daily. A layer of the network stores
its weights in a matrix. If that matrix holds $P$ numbers, pushing **one token** through it
costs $2P$ FLOPs. And it makes sense without any formula: every weight is touched exactly
once, in one multiplication and one addition.

With that you can already estimate the forward pass of any network: count its parameters and
multiply by two.

### And the backward pass

Training is not just pushing data forwards. You have to work out how to adjust every weight,
and that is the *backward* pass (module 02). It costs roughly **twice** as much as the
forward, because for every multiplication the forward did it has to do two: one to work out
how to change the layer's input — the gradient to hand back to the previous layer — and one
to work out how to change its weights.

Forward plus backward, one plus two, gives the rule you will see quoted everywhere:

$$C_{\text{token}} \approx 6N$$

where $N$ is the model's parameter count. Six FLOPs per parameter per token. That is it:
that is the formula multi-million-dollar training runs get budgeted with.

### Attention gets counted separately

There is one part of the Transformer that does not fit the rule, because it does not come
from multiplying by weights but from multiplying tokens **by each other**: every token gets
compared against all the previous ones. That is attention (module 06), and its cost does not
depend on the size of the model but on how many tokens are in the window:

$$C_{\text{token}} \approx 6N + 12 \cdot n_{\text{layers}} \cdot T \cdot d_{\text{model}}$$

With our model's numbers ($T=512$, 6 layers, $d_{\text{model}}=320$, $d_{ff}=896$, vocabulary
4096), the exact breakdown is:

```
from parameters (6N)   53,575,680 FLOPs/token     82%
from attention         11,796,480 FLOPs/token     18%
                       ----------
total                  65,372,160 FLOPs/token
```

That `65,372,160` is the exact number your exercise 2 has to return. And now look at what
happens if you leave the model alone and only stretch the window to 4096 tokens: the first
term does not move — the parameters are the same — and the second gets multiplied by eight.
Attention goes from 18% to **64%** of the total cost. That is why long-context models are
expensive, and why there is an entire research field devoted to that one term.

### What this calculation ignores, and why it matters

The formula does not count normalizations, activations or the softmax. It is not that they
are free: it is that their cost is not in computing, it is in **moving data between memory
and the processor**. A GELU does one operation per number, but to do it, the number has to
be fetched from memory and written back, and at GPU scale that is glacial compared to
multiplying.

Out of that comes a distinction worth carrying for the rest of the course:

- an operation is **compute-bound** when it is limited by raw arithmetic throughput (a large
  matmul);
- it is **memory-bound** when it is limited by memory bandwidth (an activation, a
  normalization, a dropout).

In a model as small as ours, the memory-bound ones take a far from negligible share of the
real time. That difference between "the FLOPs I count" and "the seconds I wait" is exactly
what MFU measures, which is what comes next.

## 3. MFU: how much of your machine you are actually using

$$\text{MFU} = \frac{\text{tokens/s} \times C_{\text{token}}}{\text{hardware peak FLOPS}}$$

On top, the useful FLOPs you are doing per second. Underneath, the ones the machine could be
doing. The ratio is the fraction you are getting.

If your GPU can do 50 trillion FLOPs per second and you are only getting 10 out of it, your
MFU is 0.2. **Nobody reaches 1.** A large, well-optimized model runs at 0.4-0.5. Ours, at 9
million parameters, will land well below that, and it is not your fault.

The reason is size. A GPU has thousands of compute units, and to keep them all busy it needs
big matrices. Ours are 320×320, which in GPU terms is tiny: it spends more time receiving
instructions and waiting for memory than it does multiplying.

This is not something to take on faith, it is something to measure. Here is a real curve,
measured on a reference machine — a laptop with an integrated GPU, MPS backend. Yours will
give different figures; what matters is the **shape**, which is the same on any GPU:

| matrix side | fp32 | fp16 |
|---|---|---|
| 320 (our model's) | 0.87 TFLOPS | 4.01 TFLOPS |
| 640 | 3.61 | 11.14 |
| 1024 | 3.72 | 14.74 |
| 2048 | **3.80** | 15.72 |
| 4096 | 3.60 | **15.91** |

Read the first row against the best one. **The same chip, the same operation, the same data
type, and a factor of 4 in throughput** purely from the size of the matrix. At 320 — our
model's size — that machine is delivering 23% of what it can do in fp32 and 25% of what it
can do in fp16. Everything from 1024 upwards is essentially the plateau.

That table explains three things at once that would otherwise look arbitrary: why large
models are *more* efficient per FLOP than small ones, why it pays to raise the batch size as
far as it will fit, and why your 9M model is never going to approach the figures quoted in
papers.

### Measure more than once

Those numbers are medians of five repetitions with a generous warmup. That detail is not
pedantry, and it is worth showing you what happens without it.

At 4096 the measurement is rock solid: five runs give between 15.90 and 15.91 TFLOPS. At
320, the same five runs spread between 0.60 and 1.12 in fp32 — a spread of nearly 2× on the
*same* machine measuring the *same* operation. And if on top of that you use a short warmup
and few iterations, the small sizes come out even lower, because a fixed overhead is being
spread over too little work.

So: small matrices are both slow and hard to measure, and the two are easy to confuse. When
a number surprises you, run it again before building a theory on top of it.

## 4. This is why exercise 1 makes you measure

Spec sheets for mid-range GPUs quote figures in the tens of TFLOPS. Those numbers are real
in the sense that they can be reached: with enormous matrices, in 16 bits, with the compute
units perfectly fed and nothing else going on. In an actual training run you will never see
them.

Here is what happens when you really train the `tiny_char` config on that same reference
machine (a complete run, ~70 seconds):

```
step  2,925/2,929  loss 1.3369  lr 3.00e-04  |g| 0.33  112.5k tok/s  MFU 4.8%
```

112,500 tokens per second, and `tiny_char` costs 5,948,160 FLOPs per token. Multiplying:
**0.67 effective TFLOPS**. Now divide that by a "peak" and watch what happens depending on
which one you pick:

```
0.67 / 14.0  (the backend's nominal peak, which is what the logger uses)   =  4.8%
0.67 /  3.80 (the fp32 peak MEASURED on that machine, and the run is fp32) = 17.6%
```

**It is the same run.** The MFU has not changed; the denominator has. The 4.8% invites you
to think something is broken and lose an afternoon optimizing; the 17.6% tells the truth,
which is that for a 0.8M-parameter model with 128×128 matrices that is roughly where it
ought to be.

That is the entire moral of exercise 1, and the reason it makes you time things rather than
look them up: **the only number worth using as a denominator is the one from your machine,
measured with your dtype and your sizes.**

### The three traps of timing a GPU

Getting this wrong is trivially easy, and the exercise is built around the three ways to do
it:

1. **Not warming up.** The first multiplication at a given size is between 10 and 100 times
   slower than the ones after it: the GPU is picking which kernel to use and allocating
   memory. With ten timed iterations, that first one dominates the average and ruins the
   result.
2. **Not synchronizing.** This is the one that bites. `a @ b` on a GPU **does not wait for
   anything**: it queues the work and returns control immediately. If you time without
   synchronizing, you are measuring how long the CPU takes to queue an instruction — some 20
   microseconds — and you get thousands of TFLOPS. The result is so absurd that there is a
   test in place specifically to catch it.
3. **Measuring one size and calling it "the peak".** You have seen the table: the peak
   depends on the size. That is why the demo sweeps six of them.

## 5. Precision: why 16 bits and not 32

Decimal numbers are stored by splitting bits between the *exponent* — how large or small the
number can be — and the *mantissa* — how many significant digits it has:

| format | exponent | mantissa | range |
|---|---|---|---|
| fp32 | 8 bits | 23 bits | $10^{\pm 38}$ |
| fp16 | 5 bits | 10 bits | $6\times10^{-5}$ to $65504$ |
| bf16 | 8 bits | 7 bits | $10^{\pm 38}$ |

Using 16 bits instead of 32 takes half the memory and runs roughly twice as fast — in the
table above, more than four times. The catch is the range.

**fp16 has a minuscule range.** During training, the gradients of the deep layers are very
small numbers, on the order of $10^{-7}$. Look at the table: the smallest positive number
fp16 represents normally is $6\times10^{-5}$. A gradient of $10^{-7}$ **is zero in fp16**.
And when a gradient is zero, its weight does not move: that layer stops learning. No error
message, no exception, nothing. Just a loss curve that flattens out and an afternoon spent
trying to understand why.

The fix has a name and is simpler than it sounds: **`GradScaler`**. Before computing the
gradients, it multiplies the loss by a large number (on the order of 65,000). Since the
gradient is a derivative and differentiation is linear, **every** gradient ends up
multiplied by that same factor, which lifts them into the representable range. Just before
the weights get updated, everything is divided by that same number again. The mathematical
result is identical; all that changed is that the numbers made the journey through a region
where fp16 can count. If some value overflows to infinity, that whole step is discarded and
the factor is lowered.

**bf16 needs none of this**, because it keeps fp32's 8 bits of exponent. It pays for that in
mantissa — only 7 bits of precision — and it turns out that in deep learning range matters
enormously more than precision. That is why bf16 took over as soon as the hardware supported
it.

### What each backend gets

None of this is yours to decide: `llmfs/device.py` detects the hardware and chooses. But it
is worth knowing what it is choosing and why, because it explains a fair number of oddities.

**On NVIDIA GPUs older than Ampere** (`sm_75` and below: the RTX 20 series, the GTX cards)
**there is no bf16 in hardware**, so it has to be fp16 + GradScaler. Three known traps live
there:

- `torch.cuda.is_bf16_supported()` **returns `True` on those cards**. It is true and it is
  useless: it counts a software emulation that gives the right answer and is glacially slow.
  That is why the code looks at the *compute capability* directly instead of trusting that
  function; real bf16 starts at `sm_80`.
- **FlashAttention-2 does not work below `sm_80` either.** That is fine:
  `F.scaled_dot_product_attention` detects the GPU and falls back to another algorithm
  (*memory-efficient*) that does work and that also avoids the memory spike of the naive
  method.
- **`torch.compile` is off by default** on those generations, because it fails to compile
  fairly often and when it does compile it does not always win. It is an optional flag,
  never the default.

**On Ampere or newer** (`sm_80`+) bf16 is native, and then no GradScaler is needed: that is
the comfortable path.

**On Apple Silicon (MPS backend)** the default is fp32. Memory is unified, so there is no
PCIe traffic to amortize and 16 bits wins less than on a discrete GPU. There is also a detail
that causes inexplicable slowness: `PYTORCH_ENABLE_MPS_FALLBACK=1` is active —
`llmfs/__init__.py` sets it before importing torch — which makes operations without a Metal
kernel fall back to CPU **silently**. If something runs a hundred times slower than it should
on a Mac, look there first.

`uv run python -m llmfs device` tells you what it detected and what it decided in your case.

## 6. Where the memory goes

FLOPs decide how long you wait; memory decides whether you start at all. And the split is
almost never the one people expect, so let us do the arithmetic for our 8,933,440-parameter
model with the final run's real config (batch 48, context 512, vocabulary 4096).

What the model occupies, in fp32, 4 bytes per number:

```
weights                        8.93M × 4 B  =   35.7 MB
gradients (one per weight)     8.93M × 4 B  =   35.7 MB
AdamW state (m and v)          8.93M × 8 B  =   71.5 MB
                                               --------
                                                142.9 MB
```

143 MB. On any current card, nothing. Intuition says the model is what fills the GPU, and
with a small model intuition is completely wrong.

Now the logits tensor, which is the model's output before the softmax: one number for every
token in the batch and every word in the vocabulary.

```
48 × 512 = 24,576 tokens per micro-batch
24,576 × 4096 = 100,663,296 logits
in fp32:  402.7 MB     ...and its gradient, another 402.7 MB
```

**Eight hundred megabytes for one tensor, against 143 MB for the entire model and all of its
optimizer state.** That tensor is the biggest memory consumer of the final run, above even
the intermediate activations. If you ever run out of memory, that is where to look first,
and the lever that works is lowering the batch size or the context — not slimming down the
model.

The reason it is so large is structural and worth seeing: that tensor scales with
`batch × context × vocabulary`, and the vocabulary (4096) is far larger than the `d_model`
(320) that every internal layer works with. The final projection inflates the data by a
factor of 12.8 right at the end.

## 7. Putting it together: how long the final run takes

You now have all three pieces. The full estimate, using the reference machine's measured peak
(15.9 TFLOPS):

```
total FLOPs = 65,372,160 FLOPs/token × 500,000,000 tokens = 3.27 × 10^16 FLOPs

at MFU 0.10  ->  5.7 hours
at MFU 0.20  ->  2.9 hours
at MFU 0.40 (which you will not see with a model this size)  ->  1.4 hours
```

That is what `llmfs demo 01` prints, already using **your** machine's peak. And it is an
honest estimate because it carries its uncertainty range explicitly: between three and six
hours, not "a few hours". With that number in hand you can decide whether the run gets
launched tonight or whether the token budget needs cutting.

One note of honesty about this repository: the times quoted in the README for the 500M-token
run are **estimates computed with this same formula**, not measurements, because that run has
not yet been executed on CUDA. When somebody does run it, the real number is the first thing
that will need fixing there.

## Where the debate is

The $6N$ rule gets quoted as if it were physics, and it is not: it is a cost model, with
debatable assumptions and a convention stuck on top.

It assumes the backward pass costs exactly twice the forward, which depends on which
activations you store and which you recompute: with *gradient checkpointing* — recomputing
the forward pass of some blocks instead of storing it, to save memory — the factor rises to
4 and the formula becomes $8N$. It ignores everything memory-bound, which in small models is
not a detail. And there is an arbitrary decision in the attention term: since the causal mask
only needs half the triangle computed, you could divide by two, but the convention (nanoGPT,
the papers) is not to.

We follow the convention so that your MFU is comparable with everyone else's, not because it
is more correct. If you ever see an MFU that looks twice as good as yours on comparable
hardware, check first whether the other party counted attention the same way.

On MFU itself there is a deeper criticism, which is that it has become a metric optimized for
its own sake. A high MFU means you are using the hardware well, not that you are training
well: you can raise MFU with a worse model — fatter matrices, fewer layers — and end up with
a higher loss. It is an efficiency diagnostic, not a goal. The goal is still validation loss.

---

**Further reading:** Kaplan et al. 2020,
[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (appendix B,
where the $6N$ comes from) · Micikevicius et al. 2018,
[Mixed Precision Training](https://arxiv.org/abs/1710.03740) (the GradScaler paper) ·
Chowdhery et al. 2022, [PaLM](https://arxiv.org/abs/2204.02311) (where MFU is defined).
Loose terms are in [GLOSSARY.md](../../GLOSSARY.md).
