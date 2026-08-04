# 17 — Extras and honest limits

## Why this module matters

**Because it is worth knowing where you stand.**

Two things. The first is practical: how to make the model four times smaller so you can serve
it, with a technique used in production everywhere.

The second is a frank conversation. You have built a model of 8.9 million parameters. A
frontier model has on the order of a trillion. The distance is not only one of size, and it is
worth understanding the **five** things that make it up, because four of them get mentioned far
less than the first.

And also the other side: what you have actually achieved, which is rather more than it looks
if you only look at the parameters.

### What you will know by the end

- How to store the model in a quarter of the space, and what exactly is lost
- Why 127 and not 128 (and why that detail matters more than it seems)
- What separates your model from GPT-4, with the five pieces broken down
- What you take away from the course that does not appear in the tutorials

### What it costs

2 hours. It is the last one.

---

## Quantization: the model in a quarter of the space

Your model takes 35.7 MB in fp32. Storing the weights as 8-bit integers it would take 8.9 MB.

The idea is simple: instead of storing each weight as a 4-byte float, you store a 1-byte
integer plus a **scale** that lets you recover the approximate value.

### With numbers

Take a row of weights:

```
W = [0.12, -0.45, 0.03, 0.28]
```

The largest in absolute value is 0.45. That range is mapped to `[-127, +127]`:

```
scale = 0.45 / 127 = 0.003543

W_int8 = round(W / scale) = [34, -127, 8, 79]
```

And to recover it:

```
W' = W_int8 × scale = [0.1204, -0.4500, 0.0283, 0.2799]
```

It is not exact. The error is on the order of half a unit of scale, and that is what you pay.

### Why 127 and not 128

`int8` goes from −128 to 127. Using 127 the range is **symmetric** and zero is represented
exactly. That matters more than it seems: in a matrix with many small values, zero being exact
avoids a systematic bias that would accumulate layer after layer.

### Per channel against per tensor

You can compute **one scale for the whole matrix** or **one per row**. Per row costs one extra
vector of scales —negligible— and reduces the error quite a lot, because a single row with
large values does not drag the rest along.

Measured on a real matrix from the model:

| method | relative error |
|---|---|
| per tensor | 1.07% |
| **per channel** | **0.71%** |

It is what every serious implementation does.

### What you gain and what you lose

You gain **4× in size**. On a GPU with little memory, that can be the difference between the
model fitting or not.

You lose precision. That a 0.7% error in the weights barely affects the model's quality is an
**empirical fact**, not a theorem. Nobody predicted that networks would be so robust to
quantization; it was discovered by trying.

And there is a nuance that is usually left out: **quantizing the weights does not speed
anything up on its own** if afterwards you convert to float to multiply. Real acceleration
requires kernels that operate natively in int8, and that depends on the hardware.

## Serving the model

With the model trained and quantized, serving it is an ordinary engineering problem: an HTTP
endpoint that receives a prompt and returns tokens. With FastAPI it is about 30 lines.

The only LLM-specific thing is that it is worth **streaming**: sending each token as it is
generated instead of waiting for the complete answer. At 30 tokens/s, a 200-token answer takes
7 seconds, and waiting 7 seconds staring at a blank screen feels like something broken.

## And now the honest part: what separates you from a frontier model

Your model has 8.9 million parameters and has seen 500 million tokens. A frontier model has on
the order of a trillion parameters and has seen tens of trillions of tokens. The difference is
not one of degree.

But **size is only one of five things**, and the other four get mentioned less.

### 1. The data

You use TinyStories: 2 GB of synthetic, clean, homogeneous text. A frontier model uses on the
order of 15 trillion tokens, filtered with classifiers trained for the purpose, deduplicated,
mixed in experimentally tuned proportions, and with enormous amounts of code and mathematics
because **they improve reasoning on tasks that are neither code nor mathematics** — an
empirical result nobody predicted and which still is not well explained.

The exact composition of those datasets is the worst-kept and best-protected secret in the
industry. No lab publishes its recipe.

### 2. The compute

```
your model   : ~2.3·10¹⁶ FLOPs      a few hours on an RTX 2060
GPT-4        : ~2·10²⁵ FLOPs        thousands of GPUs for months
```

That is **nine orders of magnitude**. And the cost is not only the GPUs: it is the data centre,
the network connecting them, and the engineers keeping all of that running for months without
a run falling over.

### 3. The architecture

Your model is dense: every parameter takes part in every token. Large models use **Mixture of
Experts**, where a router network activates only a fraction of the parameters per token. That
lets you have a trillion parameters with the compute cost of a hundred billion.

Add to that long-context attention, memory-efficiency techniques in attention, and a
considerable amount of work on making all of that train stably.

### 4. The post-training

You saw SFT in module 16. After that comes RLHF or DPO: collecting human preferences between
answers and adjusting the model towards the preferred ones. And after that, iterations of
red-teaming, evaluation and adjustment that last months.

**That phase is what turns a model that predicts text into something you would want to use**,
and in the large labs it employs more people than pretraining.

### 5. The infrastructure

Training on thousands of GPUs for months requires parallelism along several dimensions at
once, fault tolerance (with thousands of GPUs, one fails every few hours), monitoring, and the
ability to resume without losing days of work. It is a distributed systems problem as hard as
the machine learning problem.

## What you have achieved

And now the other side, because it is just as true.

**You have written all the pieces.** Attention, RoPE, SwiGLU, AdamW, the KV cache, the
tokenizer. All validated numerically against PyTorch. A frontier model uses exactly these
pieces: bigger, with more engineering around them, but the same ones.

**You know how to read an architecture paper.** When the next model comes out and they say it
uses grouped-query attention or RMSNorm or SwiGLU, you know what those are and why.

**You know how to debug a training run.** The step-0 loss against `ln(V)`, the overfit on a
batch, the gradient norm, the MFU. That is what separates someone who knows how to train models
from someone who copies scripts.

**And you know what is not known.** Throughout the course you have seen that SwiGLU works
without an explanation, that Adam dominates without anyone quite knowing why, that scaling laws
have wider confidence intervals than reported, and that benchmark evaluation is contaminated.
That part does not usually appear in tutorials, and it is the one that will serve you most for
reading with judgement.

## Where the debate is

**Whether scaling is enough.** The position that "scaling is all you need" has serious
defenders and serious detractors. High-quality data is running out, and models trained on
synthetic data generated by other models show degradation in some setups. Nobody knows whether
the curve continues.

**What a model builds inside.** Mechanistic interpretability has managed to explain specific
components —the *induction heads* of module 06 are the success story— but it is a long way from
accounting for a whole model. Whether these systems "understand" in any useful sense of the
word is an open question, and be suspicious of anyone who gives you a categorical answer in
either direction.

---

**Further reading:** Dettmers et al. 2022,
[LLM.int8()](https://arxiv.org/abs/2208.07339) · Shazeer et al. 2017,
[Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538) (MoE) ·
Elhage et al. 2021,
[Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).
Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
