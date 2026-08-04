# 10 — The full GPT

## Why this module matters

**Because this is where everything comes together and produces an exact number.**

You have attention, normalization, the FFN and RoPE. This module assembles them into the
model you are going to train, and it ends with a check that either adds up or does not:
**8,933,440 parameters**. Not one more.

That number is not decoration. That the formula you derive by hand matches the model's real
count means you have understood where every weight is and why. If it does not add up,
something in your architecture is not what you think it is.

And there are three design decisions here that almost every tutorial skips and that are what
make the model train well: weight tying, depth-scaled initialization, and the final
normalization.

### What you will know by the end

- How a complete Transformer is assembled, from token ids to logits
- How to save 15% of the parameters by reusing a matrix you already have
- Why the initialization of the deep layers **cannot be the same** as everything else's
- How to verify your model is genuinely causal, with a check that gives exactly zero

### What it costs

3 hours. It closes Part II: when you finish it you have the model assembled and audited.

---

## The block

A Transformer block is this, and nothing more:

```
x = x + attention(norm1(x))
x = x + ffn(norm2(x))
```

Two sub-blocks, each with its pre-norm normalization and its residual. Attention **moves
information between tokens**; the FFN **processes each token separately**. They alternate.

The two residuals are independent on purpose: each sub-block can contribute a little or a
lot to the residual stream without constraining the other.

## The whole model

```
token ids
    ↓ embedding table
vectors
    ↓ block × 6
vectors
    ↓ final normalization
vectors
    ↓ projection to logits
scores over the 4096 tokens
```

With RoPE there is no positional embedding to add at the start: position is injected inside
attention, by rotating Q and K. That is why the first layer is only the token table.

Now the three decisions that make the model what it is.

## Decision 1: weight tying

The embedding table turns an id into a vector: it is a $4096 \times 320$ matrix. The output
layer turns a vector into scores over the vocabulary: it is a $320 \times 4096$ matrix.

**They are the same matrix, transposed.** Why not reuse it?

```python
self.lm_head.weight = self.token_embedding.weight
```

That does not copy: it makes both layers point at **the same tensor**. The saving:

```
without tying:  4096 × 320 × 2 = 2,621,440 parameters
with tying:     4096 × 320     = 1,310,720 parameters
saving:                          1,310,720   (15% of the model)
```

With a 9M model that is enormous. And it usually **improves quality** too, not just saves:
each weight receives gradient along two different paths — once as an input embedding, once
as an output projection — so it trains with twice the signal.

The conceptual justification is that a token should be "close" in embedding space to those
it can be confused with, and that notion of closeness serves both reading and writing.

A practical detail: `model.parameters()` deduplicates by identity, so the total comes out
right on its own. But if you break it down by component you have to track the `id()`s
already seen or you will count the matrix twice.

## Decision 2: depth-scaled initialization

This is the detail most people skip and it explains why deep models sometimes do not train
well.

Think about the residual stream. Each block **adds** its contribution to it:

```
x₀ → x₁ = x₀ + something₁ → x₂ = x₁ + something₂ → ... → x₆
```

If the contributions are independent and all have variance $\sigma^2$, the variance of the
sum grows **linearly with the number of terms**. With 6 layers and 2 sub-blocks each, that is
12 contributions: the output has 12 times the variance of the input.

GPT-2's fix, and the one we use: initialize with a smaller standard deviation **only the
projections that write into the residual stream**, which are attention's `out_proj` and the
FFN's `down_proj`:

$$\sigma = \frac{0.02}{\sqrt{2 \cdot n_{\text{layers}}}}$$

The 2 is because each block writes twice. With 6 layers: $0.02/\sqrt{12} = 0.0058$.

Everything else is initialized with plain $\sigma = 0.02$.

**And the 0.02 is not arbitrary either.** It is what makes the step-0 loss equal $\ln(V)$:
with almost identical logits, the softmax comes out almost uniform. If you initialized with
PyTorch's standard normal ($\sigma = 1$), the model would start with strong, random opinions
and the loss would come out above $\ln(V)$ — exactly what you saw in module 05's demo.

## Decision 3: the final normalization

In pre-norm, the residual stream **is never normalized along the way**. It reaches the output
at a scale that grows with depth. That is why there is a normalization right before the
projection to logits. It is not optional: without it, the logits come out at an arbitrary
scale and training is far more fragile.

## The exact count

And now the number. Derive the formula yourself before looking:

| component | formula | value |
|---|---|---|
| token embeddings | $V \cdot d$ | 4096 × 320 = **1,310,720** |
| attention per layer | $4d^2$ | 4 × 320² = 409,600 |
| SwiGLU per layer | $3 \cdot d \cdot d_{ff}$ | 3 × 320 × 896 = 860,160 |
| RMSNorm per layer | $2d$ | 2 × 320 = 640 |
| **per layer** | | **1,270,400** |
| × 6 layers | | **7,622,400** |
| final RMSNorm | $d$ | 320 |
| lm_head | tied | **0** |
| **TOTAL** | | **8,933,440** |

Three things worth noting in that table:

- **Attention has no biases**: they are four clean $d \times d$ matrices. Modern LLMs have
  been dropping them; they add little and they complicate module 11's weight decay.
- **RMSNorm has scale only** ($d$ parameters), not scale and bias ($2d$) like LayerNorm.
- **RoPE contributes zero parameters.** The cos and sin tables are computed from a formula
  and stored as *buffers*, not as parameters. That is why they are registered with
  `persistent=False`: they are recomputed when the model is built and there is no need to
  store them in the checkpoint.

**Non-embedding parameters: 7,622,720.** That is the number module 12 uses for the scaling
laws, because embeddings scale differently from the rest and Chinchilla treats them
separately.

## Where the debate is

It is worth separating what is settled from what is convention.

**Well founded:** attention's $\sqrt{d_k}$ scaling (there is a clear variance argument), the
need for residuals, the final normalization in pre-norm.

**Convention with empirical support:** pre-norm over post-norm, RMSNorm over LayerNorm,
SwiGLU over GELU, weight tying. They work better on the benchmarks; there is no theory.

**Practically arbitrary:** the 0.02 in the initialization (it comes from GPT-2 and nobody has
re-justified it), the FFN's 4x factor, RoPE's $\theta = 10000$, the ratio between depth and
width. Alternatives have been tried and the differences are small.

And an honest one about this particular model: **6 layers of 320 dimensions is not an
optimal choice derived from anything**. It is a reasonable point for fitting in an RTX 2060
and training in hours. With the same 9M parameters you could do 12 layers of 224, or 3 of
512, and they would work similarly. The relationship between depth and width is little
explored at this scale.

---

**Further reading:** Radford et al. 2019,
[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
(where the scaled init and the 0.02 come from) · Press & Wolf 2017,
[Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
(weight tying) · [nanoGPT](https://github.com/karpathy/nanoGPT). Stray terms are in
[GLOSSARY.md](../../GLOSSARY.md).
