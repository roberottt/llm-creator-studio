# 12 — Efficiency and scaling laws: am I going fast, and am I spending well?

## Why this module matters

**Because "bigger is better" turned out to be false, and that changed the field.**

Two questions that look different and are the same one: am I making use of my GPU? and how
should I split my budget between model size and amount of data?

The second has a concrete answer, and in 2022 it turned out the entire industry was getting it
wrong. GPT-3 had 175 billion parameters and was **twelve times under-trained**: with the same
compute budget, a model three times smaller trained on more data would have been better. They
proved it by training one.

In this module you are going to reproduce that formula and check that it predicts Chinchilla's
real size to three significant figures. And you are going to measure your own training run's
efficiency with the field's standard metric, which is also what will tell you where to poke when
module 13 runs slow.

### What you will know by the end

- Where your model's FLOPs go, and why lengthening the context is expensive
- **Which of the course's three parameter counts goes into each formula**, which is where
  everybody gets lost
- What MFU is, what value is reasonable, and why yours is going to be low without it being your
  fault
- Where to look when a training run is slower than it should be
- Chinchilla's formula, **verified against real historical models**
- Why our model is over-trained on purpose, and why Llama-3 is 90 times over

### What you are going to write

Three functions, and this theory follows them in order:

| Exercise | What it does |
|---|---|
| 1. `model_flops_per_token` | What a token costs, broken down |
| 2. `compute_mfu` | What fraction of your GPU you are using |
| 3. `chinchilla_optimal_allocation` | How to split the compute budget |

**None is more than five lines of code** and there are no tensors, no models, no training: just
arithmetic over the config's fields. This module is short to type and long to understand, which
is exactly the opposite of module 11. All the difficulty is in knowing what the numbers mean and
what decisions get made with them.

Exercises 1 and 2 chain: the `total` the first one returns is the `flops_per_token` the second
one eats. The third is independent.

### What it costs

2 hours.

---

## Exercise 1: where the FLOPs go (`model_flops_per_token`)

In module 01 you estimated a token's cost with the quick rule. Here you break it down properly,
and the breakdown has a design consequence.

The function separates the cost into two terms because **they grow with different things**:

```
   matmul     grows with the SIZE of the model     (d_model, n_layers, d_ff, vocab)
   attention  grows with the CONTEXT               (context_length)
```

And that is the whole point. With our config the split is 82/18, but look at what happens when
you lengthen the context leaving the model alone:

| context | matmul | attention | total | % attention |
|---|---|---|---|---|
| 128 | 53.6M | 2.9M | 56.5M | 5% |
| **512 (ours)** | **53.6M** | **11.8M** | **65.4M** | **18%** |
| 1024 | 53.6M | 23.6M | 77.2M | 31% |
| 2048 | 53.6M | 47.2M | 100.8M | 47% |
| 4096 | 53.6M | 94.4M | 147.9M | 64% |
| 8192 | 53.6M | 188.7M | 242.3M | 78% |

The matmul column **does not move**: it does not depend on the context at all. The attention one
grows linearly and from 2048 onwards it dominates. This tells you instantly whether lengthening
the context is going to be expensive, without having to try it — and it connects to the quadratic
memory cost you saw in module 06.

### Where each constant comes from

Three numbers, and none of them is arbitrary.

**The 2 in `matmul = 2 * params_matmul`.** Multiplying a matrix by a vector does, per weight, one
multiplication and one addition: 2 operations per parameter.

**The 4 in `attention = 4 * n_layers * T * d_model`.** These are attention's two matmuls that
**do not involve parameters**: $QK^\top$ and $\text{weights} \times V$. Each costs $2 T d$ per
token, and there are two. That is why they do not show up in the matmul term: there are no
weights to count, but the computation happens all the same.

**The 3 for the backward.** The backward pass costs roughly **twice** the forward, because you
have to compute the gradient with respect to the input *and* with respect to the weights: two
matmuls where the forward did one. Forward + backward = 1 + 2 = 3. You can check it by calling
the function with `include_backward=False` and dividing: it comes out at exactly 3.0.

### The three parameter counts, which is where everybody gets lost

Here is what confuses people most in this module, and it is because the course drags along
**three different numbers** that look very much alike:

| count | value | what it includes | where it is used |
|---|---|---|---|
| **total** | 8,933,440 | everything | module 10, "how many parameters it has" |
| **non-embedding** | 7,622,720 | total − the embedding table | Chinchilla, exercise 3 |
| **params_matmul** | 8,929,280 | everything except the normalization scales | the FLOPs, exercise 1 |

They look alike and they are not interchangeable. Two observations that sort them out:

**`params_matmul` includes the final projection even with weight tying.** Tying the weights saves
**memory**, not **computation**: the multiplication by the $320 \times 4096$ matrix happens all
the same on every token. That is why the term adds `d_model * vocab_size` even though module 10
counted that `lm_head` as 0 parameters. There is a dedicated test
(`test_the_final_projection_counts_even_with_tying`).

**And `params_matmul` is exactly module 11's *decay* group**, the 8,929,280 across 43 tensors.
That is no coincidence: the two criteria are the same one, "everything that is a matrix". What
has 2 dimensions or more takes part in matrix multiplications and receives weight decay; the
RMSNorm scales, which have 1 dimension, get neither.

**So watch out with module 01's `6N`.** The rule says "cost per token ≈ 6 × number of
parameters", and the `N` to put in there is `params_matmul`:

```
   6 × 8,929,280  = 53,575,680   = the matmul term  ✓
   6 × 7,622,720  = 45,736,320   ✗ does not match
```

And yet Chinchilla's `N`, in exercise 3, **is** the non-embedding one. They are two different
formulas with two different conventions, and mixing them is this module's silent mistake.

---

## Exercise 2: how much of your GPU you are using (`compute_mfu`)

You already have what a token costs and in module 01 you measured your hardware's peak. **MFU**
(*Model FLOPs Utilization*) puts the two together:

$$\text{MFU} = \frac{\text{tokens/s} \times C_{\text{token}}}{\text{peak FLOPS}}$$

It is two lines of code: a validation and a division. The only place you can slip is the `1e12`
that converts TeraFLOPS into FLOPS, so that numerator and denominator are in the same units.

### What actually comes out

The demo trains a few steps and times them. This is what was measured on the development machine
(M5, estimated peak 14.0 TFLOPS):

| batch | tokens/step | ms/step | tokens/s | MFU |
|---|---|---|---|---|
| 1 | 256 | 12.2 | 21.0k | 7.9% |
| 2 | 512 | 21.5 | 23.8k | 8.9% |
| 4 | 1,024 | 40.9 | 25.1k | 9.4% |
| 8 | 2,048 | 80.8 | 25.4k | 9.5% |
| 16 | 4,096 | 162.6 | 25.2k | 9.5% |

What is interesting about that table is not the final number, it is **the shape of the curve**:
it rises with the batch and then flattens out. That flattening point is where you stop being
limited by kernel launches and start being limited by computation. Raising the batch beyond it
no longer buys any efficiency — only memory.

### What value is reasonable

| situation | typical MFU |
|---|---|
| large, well-optimized models on A100/H100 | 0.4 – 0.5 |
| medium models | 0.2 – 0.3 |
| **our 9M model** | 0.1 – 0.2 |
| something is wrong | < 0.05 |

**Nobody reaches 1.** The theoretical peak is only hit with enormous, perfectly aligned matmuls
and absolutely nothing else in the way.

And with a small model a low MFU is **unavoidable**, not a failure of yours: 320×320 matrices are
not enough to saturate the tensor cores, and a significant share of the time goes into launching
kernels rather than computing. It is the same phenomenon you measured in module 01's demo, where
128-sized matrices gave less than 2 TFLOPS and 2048-sized ones ten times more.

### What it is really for

Not for its absolute value, but because it is **comparable**: it does not depend on the model or
the hardware. "25,000 tokens per second" tells you nothing; "9.5% MFU" does. You change the batch
size, enable `torch.compile`, move the dataloader to another thread, and watch whether the number
goes up. It is the thermometer for optimizations, and you will use it in module 13.

### Where the time goes when the MFU is low

Four suspects, in order of frequency:

1. **The dataloader.** If preparing the next batch takes longer than processing it, the GPU
   waits. You detect it by timing `get_batch` separately, which you already did in module 04 —
   there it came out at 0.04% of the step, so in our case it is not this.
2. **The batch is small.** Less work per kernel launch. It is the first thing to try, and the
   table above tells you exactly where it stops paying off.
3. **Accidental synchronizations.** Any `.item()`, `float(tensor)` or `print` of a tensor forces
   the CPU to wait for the GPU to finish. Inside the training loop that kills performance, and it
   is easy to sneak in without noticing while adding logging.
4. **Memory-bound operations.** Normalizations and activations do not show up in the FLOP count
   but do consume time. In a small model they are a significant fraction — the same business as
   module 07's RMSNorm timing table.

---

## Exercise 3: how to split the budget (`chinchilla_optimal_allocation`)

Now the design question. You have a fixed compute budget — one GPU and two weeks, say. You can
spend it on a **big model with little data** or on a **small model with a lot of data**. Which
ends up with less loss?

For years it was assumed you just had to make models bigger, full stop. GPT-3 had 175 billion
parameters trained on 300 billion tokens.

In 2022, Hoffmann et al. measured this properly: they trained over 400 models of different sizes
with different amounts of data and fitted a surface. Their conclusion:

> **Parameters and data should grow proportionally. About 20 tokens per parameter.**

GPT-3 had **1.7 tokens per parameter**, twelve times below the optimum. To prove it they trained
**Chinchilla**: 70 billion parameters and 1.4 trillion tokens, on the same compute budget as
Gopher, which had 280 billion. Chinchilla won on almost every benchmark **with a quarter of the
parameters**.

### The arithmetic, which is three lines

Starting from module 01's $C = 6ND$ and the constraint $D = kN$ (with $k$ the tokens per
parameter):

$$C = 6N(kN) = 6k\,N^2 \quad \Longrightarrow \quad N = \sqrt{\frac{C}{6k}}, \qquad D = kN$$

And from that comes a consequence worth internalizing: **if you double the budget, the optimal
model does not double, it grows by 41%** ($\sqrt{2}$). And the data, another 41%. Both at once,
never one alone. To quadruple the model you have to multiply the compute by sixteen.

### Check it against reality

This is the part that gives you confidence in the formula, and it is what the demo does. With
Chinchilla's real budget, $5.88 \times 10^{23}$ FLOPs:

```
   N = √(5.88·10²³ / 120) = 7.0·10¹⁰ = 70 billion parameters
```

The real model had 70 billion. Watching it land on a historical case gives rather more confidence
than reading the derivation.

And applied to the models you know:

| model | parameters | tokens | tok/param | Chinchilla optimum | verdict |
|---|---|---|---|---|---|
| GPT-3 | 1.75e11 | 3e11 | 2 | 5.12e10 | under-trained |
| Gopher | 2.8e11 | 3e11 | 1 | 6.48e10 | under-trained |
| Chinchilla | 7e10 | 1.4e12 | 20 | 7.0e10 | right on the mark |
| Llama-2 7B | 7e9 | 2e12 | 286 | 2.65e10 | over-trained on purpose |
| Llama-3 8B | 8e9 | 1.5e13 | 1875 | 7.75e10 | over-trained on purpose |
| **ours** | **7.62e6** | **5e8** | **66** | **1.38e7** | **over-trained on purpose** |

Notice Chinchilla's row: it is the only one landing exactly on its own optimum, which is precisely
what the paper set out to demonstrate.

### And why ours is three times over

```
   non-embedding parameters : 7.62 M
   tokens                   : 500 M
   tokens per parameter     : 66        (the "optimum" would be 20)
```

It is deliberate, for two reasons.

**The first: Chinchilla optimizes *training* compute, not usage compute.** If the model is going
to run many times afterwards, a smaller, more trained one is better: training is paid once and
inference every time. Llama-3 takes this to the extreme with ~1,800 tokens per parameter, ninety
times above Chinchilla, and it is not a mistake: their objective function is a different one. An
8B model that runs a billion times pays for itself handsomely even if training it cost more than
necessary.

**The second: at this scale, over-training is cheap.** Hours, not months. And it gives a
noticeably better model. Chinchilla optimality matters when compute is the scarce resource; here
the scarce resource is your patience.

So "over-trained" is not an insult and "under-trained" is not an automatic diagnosis of error:
they are positions on a trade-off, and which one suits you depends on what you are going to do
with the model.

---

## KV cache: why generating is different from training

A note that prepares module 14 and that makes sense right now, with exercise 1's breakdown fresh.

When training you process all 512 tokens at once and get the parallelism. When **generating**, you
produce one token at a time, and at each step the model recomputes the keys and values of *all*
the previous tokens, which have not changed since the step before.

Storing them turns a quadratic cost into a linear one. The price is memory:

$$\text{KV memory} = 2 \times n_{\text{layers}} \times T \times d_{\text{model}} \times \text{bytes}$$

For our model with 512 tokens in fp16: $2 \times 6 \times 512 \times 320 \times 2 = 3.9$ MB.
Nothing, compared with the 1007 MB of logits you counted in module 10. But for a 70B model with a
100,000-token context it would be tens of gigabytes, and that is why techniques like
*grouped-query attention* exist.

## Where the debate is

Scaling laws are **less well established than their name suggests**, and it is worth knowing that
before citing them too confidently.

Chinchilla's coefficients were fitted to a specific range of scales and a specific dataset, and
**extrapolating outside that is not justified**. In fact, in 2024 a group re-analysed the original
data and found the fit had methodological problems and that the confidence intervals were far
wider than reported. The qualitative conclusion — "you have to train on more data than people
thought" — holds; the exact numbers, with more caution. The 20 you use by default in exercise 3 is
a convenient round figure, not a constant of nature.

Also, scaling laws predict **loss**, not capabilities. The relationship between lowering the loss
and "reasoning better" is neither direct nor well understood, and it is one of the most important
open discussions in the field.

And there is something no scaling law captures: **data quality**. The TinyStories paper — the
dataset you are going to use — shows that a small, very clean corpus lets tiny models generate
coherent text, something you do not get with the same amount of internet text. No $N$ or $D$
captures that, and it is literally the reason a nine-million-parameter model is going to write
something readable at the end of module 13.

---

**Further reading:** Hoffmann et al. 2022,
[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
(Chinchilla) · Kaplan et al. 2020,
[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) · Chowdhery et al.
2022, [PaLM](https://arxiv.org/abs/2204.02311) (the MFU definition) · Besiroglu et al. 2024,
[Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102).
Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
