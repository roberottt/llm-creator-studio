# 01 — Environment and hardware

## Why this module matters

**This module saves you a week.**

You are going to train a model on your own machine. Before writing a single line of neural
network it is worth knowing whether that will take two hours or two weeks, because the
answer changes every decision that comes afterwards: how many layers, what context, what
batch size.

And there is a second, less obvious reason. Almost everyone who trains models copies numbers
from a tutorial without knowing where they come from. By the end of this module you will be
able to **compute** what a model costs before building it, which is what separates designing
from copying.

### What you will know by the end

- How many operations it costs to process a token, and **where the formula comes from**
- How many TFLOPS your GPU **really** delivers, not what the spec sheet says
- Why your RTX 2060 is forced to use `float16` and what problem that brings
- How to estimate how long a training run will take before launching it

### What it costs

45 minutes. Three short functions, and the demo measures your hardware for real.

---

## The question: how do you measure "how much does it cost"?

A computer does not take the same time on two different tasks, so we need a common unit. The
one used is the **FLOP**: one operation on decimal numbers (an addition or a
multiplication). Training a model is a lot of FLOPs, and a GPU can do a few trillion per
second.

Two numbers, and one division:

```
time = total FLOPs to be done / FLOPs per second my GPU delivers
```

This whole module is about estimating those two numbers well.

## Where a network's FLOPs come from

Almost everything a neural network does is **multiply matrices**. Let us count exactly what
one costs.

Multiply a 2×3 matrix by a 3×2 one. The result is 2×2, that is, 4 numbers. Each of those 4
numbers comes from pairing 3 values with 3 others, multiplying them and adding them up: 3
multiplications and 2 additions, which we round to 6 operations (2 per pair). Total:

```
4 output numbers × 6 operations = 24 FLOPs
```

In general, multiplying an $m \times k$ matrix by a $k \times n$ one costs $2mnk$ FLOPs.

Now the step that turns this into a useful rule. A layer of the network stores its weights in
a matrix. If that matrix has $P$ numbers inside, processing **one token** through it costs
$2P$ FLOPs. It makes sense: each weight is used once, in one multiplication and one addition.

With that you can already estimate the forward pass of any network: count its parameters and
multiply by two.

### And the backward pass

Training is not just pushing the data forwards. You have to compute how to adjust each
weight, and that is the *backward* pass (module 02). It costs roughly **twice** the forward,
because it does two multiplications for every one the forward did: one to know how to change
the layer's input and one to know how to change its weights.

Adding forward + backward gives the number you will see cited everywhere:

$$C_{\text{token}} \approx 6N$$

where $N$ are the model's parameters. Six FLOPs per parameter per token. That is all.

### Attention is counted separately

There is one part of the Transformer that does not fit the rule, because it does not come
from multiplying by weights but from multiplying tokens **against each other**. It is
attention (module 06), and its cost depends on how many tokens are in the window:

$$C_{\text{token}} \approx 6N + 12 \cdot n_{\text{layers}} \cdot T \cdot d_{\text{model}}$$

With our numbers ($T=512$, 6 layers, $d_{\text{model}}=320$) the total comes to **65.4
million FLOPs per token**, of which attention is 18%. With a 4096-token window it would be
64%. That is why models with very long contexts are expensive: that term grows while the
other one stays put.

### What this calculation ignores

It does not count normalizations, activations or the softmax. It is not that they are free:
it is that their cost is not in computing, but in **moving data between memory and the
processor**. In a small model like ours that can be a significant part of the real time, and
that gap between "the FLOPs I count" and "the seconds I take" is exactly what MFU measures.

## MFU: how much of your GPU you are really using

$$\text{MFU} = \frac{\text{tokens/s} \times C_{\text{token}}}{\text{hardware peak FLOPS}}$$

If your GPU can do 50 trillion FLOPs per second and you are only getting 10 out of it, your
MFU is 0.2. **Nobody reaches 1.** A large, well-optimized model sits around 0.4-0.5. Ours,
at 9 million parameters, will stay at 0.1-0.2, and that is not your fault.

The reason is size. A GPU has thousands of compute units. To keep them all busy it needs
large matrices. Ours are 320×320, which is tiny: the GPU spends more time receiving
instructions and waiting for memory than multiplying. In the demo you will see the curve:
128-sized matrices give less than 2 TFLOPS and 2048-sized ones give ten times more, on the
same GPU and with the same data type.

The "peak" is not an honest number either. The RTX 2060's spec sheet says 51.6 TFLOPS, but
that is the ideal case. That is why exercise 1 makes you **measure** it instead of reading
it: the only number that counts is the one from your machine.

## Precision: why 16 bits and not 32

Decimal numbers are stored by splitting bits between the *exponent* (how large or small the
number can be) and the *mantissa* (how many significant digits it has):

| format | exponent | mantissa | range |
|---|---|---|---|
| fp32 | 8 bits | 23 bits | $10^{\pm 38}$ |
| fp16 | 5 bits | 10 bits | $6\times10^{-5}$ to $65504$ |
| bf16 | 8 bits | 7 bits | $10^{\pm 38}$ |

Using 16 bits instead of 32 takes half the memory and runs roughly twice as fast. The catch
is the range.

**fp16 has a tiny range.** During training, the gradients of the deep layers are very small
numbers, on the order of $10^{-7}$. In fp16 that is zero: the number cannot be represented
and it is lost. The result is that those layers stop learning, silently and with no error
message.

The fix has a name and it is simpler than it looks: **`GradScaler`**. Before computing the
gradients, it multiplies the loss by a large number (around 65,000). Since the gradient is a
derivative, every gradient ends up multiplied by that same number and rises into the
representable range. Right before updating the weights, it divides again. If some value goes
over the top and comes out infinite, that step is discarded and the factor is lowered.

**bf16 needs none of this**, because it keeps fp32's 8 exponent bits (at the cost of
precision, which in deep learning matters far less than range).

### Your particular hardware

**The RTX 2060 is Turing (`sm_75`) and has no bf16.** You are forced into fp16 +
GradScaler. There are also three traps that `llmfs/device.py` already dodges for you:

- `torch.cuda.is_bf16_supported()` **returns `True` on your GPU**, counting a software
  emulation that is correct and extremely slow. That is why the code looks directly at the
  *compute capability*: real bf16 starts at `sm_80`.
- **FlashAttention-2 does not work either** below `sm_80`. That is fine:
  `F.scaled_dot_product_attention` detects the GPU and uses another algorithm
  (*memory-efficient*) that does work and also avoids the naive method's memory cost.
- **`torch.compile` is off by default**, because on Turing it fails to compile fairly often
  and when it does compile it does not always win.

On the MacBook (MPS) the default is fp32. The memory is unified, so there is no PCIe traffic
to amortize and fp16 wins less than it does on a discrete GPU.

## Where the debate is

The $6N$ rule gets cited as if it were physics, and it is not: it is a cost model, with
debatable assumptions. It assumes the backward pass costs exactly twice the forward, which
depends on which activations you store and which you recompute (with *gradient
checkpointing* the factor rises to 4). It ignores everything that is memory-bound. And there
is an arbitrary decision in attention: since the causal mask only needs half the triangle,
you could divide by two, but the convention (nanoGPT, the papers) is not to. We follow the
convention so your MFU is comparable with everyone else's, not because it is more correct.

---

**Further reading:** Kaplan et al. 2020,
[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (appendix B) ·
Micikevicius et al. 2018, [Mixed Precision Training](https://arxiv.org/abs/1710.03740) ·
Chowdhery et al. 2022, [PaLM](https://arxiv.org/abs/2204.02311) (definition of MFU).
Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
