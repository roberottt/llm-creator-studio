# 12 — Efficiency and scaling laws

## Why this module matters

**Because "bigger is better" turned out to be false, and that changed the field.**

Two questions that look different and are the same: am I using my GPU? and how should I
split my budget between model size and amount of data?

The second has a concrete answer, and in 2022 it turned out the whole industry was getting it
wrong. GPT-3 had 175 billion parameters and was **twelve times under-trained**: with the same
compute budget, a model three times smaller trained on more data would have been better. They
proved it by training one.

In this module you are going to reproduce that formula and check that it predicts
Chinchilla's real size to four significant figures. And you are going to measure the
efficiency of your own training run with the field's standard metric.

### What you will know by the end

- What MFU is, what value is reasonable, and why yours will be low (and it is not your fault)
- Where to look when a training run is slower than it should be
- Chinchilla's formula, **verified against real historical models**
- Why our model is deliberately over-trained, and why Llama-3 is 90 times over

### What it costs

2 hours. Three functions of fewer than five lines each, but they decide how you design a
training run.

---

## Part 1: MFU, or how much of your GPU you are using

You already saw in module 01 what a token costs: about 65.4 million FLOPs for our model. And
you measured your hardware's peak. MFU puts the two together:

$$\text{MFU} = \frac{\text{tokens/s} \times C_{\text{token}}}{\text{peak FLOPS}}$$

An example with RTX 2060 numbers. If while training you see 3,000 tokens/s:

```
3,000 × 65.4·10⁶ = 1.96·10¹¹ real FLOPS
2060's peak      = 5.16·10¹³ FLOPS
MFU = 1.96·10¹¹ / 5.16·10¹³ = 0.004 = 0.4%
```

Less than 1%. That sounds like a disaster and you have to know how to read it.

### What MFU is reasonable

| situation | typical MFU |
|---|---|
| large, well-optimized models on A100/H100 | 0.4 – 0.5 |
| mid-sized models | 0.2 – 0.3 |
| **our 9M model** | 0.1 – 0.2 |
| something is wrong | < 0.05 |

**Nobody reaches 1.** The theoretical peak is only reached with enormous, perfectly aligned
matmuls and nothing else in the way.

With a small model a low MFU is unavoidable. 320×320 matrices are not enough to saturate the
tensor cores, and the time goes into launching kernels and moving memory. It is the same
phenomenon you measured in module 01's demo: 128-sized matrices gave less than 2 TFLOPS and
2048-sized ones ten times more.

**What matters about MFU is not its absolute value, it is that it is comparable.** It is
independent of the model and the hardware, so you can change the batch size, switch on
`torch.compile` or move the dataloader to another thread, and see whether the number goes up.

### Where the time goes when the MFU is low

Four suspects, in order of frequency:

1. **The dataloader.** If preparing the next batch takes longer than processing it, the GPU
   waits. You detect it by timing `get_batch` separately (you did that in module 04).
2. **The batch is small.** Less work per kernel launch. Raising `batch_size` until memory
   fills is usually the first thing to try.
3. **Accidental synchronizations.** Any `.item()`, `float(tensor)` or `print` of a tensor
   forces the CPU to wait for the GPU. Inside the training loop, that kills throughput.
4. **Memory-bound operations.** Normalizations and activations do not appear in the FLOP
   count but they do consume time. In a small model they are a significant fraction.

## Part 2: Chinchilla, or how to split the budget

Now the design question. You have a fixed compute budget. You can spend it on a **large model
with little data** or on a **small model with a lot of data**. Which gives the lower loss?

For years it was assumed you had to make the models bigger. GPT-3 had 175 billion parameters
trained on 300 billion tokens.

In 2022, Hoffmann et al. measured this properly: they trained more than 400 models of
different sizes with different amounts of data and fitted a surface. Their conclusion:

> **Parameters and data should grow proportionally. About 20 tokens per parameter.**

GPT-3 had **1.7 tokens per parameter**, twelve times below the optimum. It was massively
under-trained.

To prove it they trained **Chinchilla**: 70 billion parameters and 1.4 trillion tokens, with
the same compute budget as Gopher (280 billion parameters). Chinchilla won on almost every
benchmark **with a quarter of the parameters**.

### The arithmetic

Starting from $C = 6ND$ (module 01) and $D = 20N$:

$$C = 6N(20N) = 120N^2 \quad \Longrightarrow \quad N = \sqrt{\frac{C}{120}}, \qquad D = 20N$$

Check it with Chinchilla itself. Its budget was $5.76 \times 10^{23}$ FLOPs:

```
N = √(5.76·10²³ / 120) = 6.9·10¹⁰ = 69 billion parameters
```

The real model had 70 billion. The formula nails it.

### Our case

```
non-embedding parameters : 7.62 M
tokens                   : 500 M
tokens per parameter     : 65
```

**More than three times above Chinchilla's "optimum".** It is deliberate, for two reasons.

**The first: Chinchilla optimizes *training* compute, not usage compute.** If the model is
going to be run many times afterwards, a smaller and more heavily trained one is better:
training is paid once and inference every time. Llama-3 takes this to the extreme with ~1,800
tokens per parameter, ninety times above Chinchilla, and it is not a mistake: their objective
function is different.

**The second: at this scale over-training is cheap.** Hours, not months. And it gives a
noticeably better model. Chinchilla optimality matters when compute is the scarce resource;
here the scarce resource is your patience.

## KV cache: why generating is different from training

A note that sets up module 14.

During training you process all 512 tokens at once and take advantage of parallelization.
When **generating**, you produce one token at a time, and at each step the model recomputes
the keys and values of *all* the previous tokens, which have not changed.

Storing them turns a quadratic cost into a linear one. The price is memory:

$$\text{KV memory} = 2 \times n_{\text{layers}} \times T \times d_{\text{model}} \times \text{bytes}$$

For our model with 512 tokens in fp16: $2 \times 6 \times 512 \times 320 \times 2 = 3.9$ MB.
Nothing. For a 70B model with a 100,000-token context, it would be tens of gigabytes, which
is why techniques like *grouped-query attention* exist.

## Where the debate is

Scaling laws are **less well established than their name suggests**.

Chinchilla's coefficients were fitted over a particular range of scales and to a particular
dataset, and **extrapolating outside that is not justified**. In fact, in 2024 a group
reanalysed the original data and found the fit had methodological problems and that the
confidence intervals were much wider than reported. The qualitative conclusion — "you have to
train with more data than people thought" — holds; the exact numbers, with more caution.

On top of that, scaling laws predict **loss**, not capabilities. The relationship between
lowering the loss and "reasoning better" is neither direct nor well understood, and it is one
of the most important open discussions in the field.

And there is something no scaling law captures: **data quality**. The TinyStories paper shows
that a small, very clean dataset lets tiny models generate coherent text, something you do
not get with the same amount of internet text. No $N$ or $D$ captures that.

---

**Further reading:** Hoffmann et al. 2022,
[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
(Chinchilla) · Kaplan et al. 2020,
[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) · Chowdhery et
al. 2022, [PaLM](https://arxiv.org/abs/2204.02311) (definition of MFU) · Besiroglu et al.
2024, [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102).
Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
