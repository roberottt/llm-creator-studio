# 08 — FFN, GELU and SwiGLU

## Why this module matters

**Because two thirds of your model are here, and almost nobody knows it.**

When someone says a model has N parameters, most of them are not in the attention: they are
in this part, which gets far less attention in the explanations. In our model that is 5.16
million out of 8.93.

And there is a deeper reason. Attention is a weighted average, that is, a **linear
operation**, and stacking linear operations achieves nothing: a hundred layers are
equivalent to one. What stops the whole Transformer from collapsing into a single matrix
multiplication is precisely this module. The demo measures it: five linear layers with no
activation give exactly the same result as a single matrix.

### What you will know by the end

- Why without a nonlinearity a network's depth is an illusion
- What happens to a ReLU neuron when it drifts into the negative zone (it dies, literally)
- What SwiGLU is and **where the 896** in the final model's config comes from
- A case where the paper's own author writes that he does not know why it works

### What it costs

1.5 hours. The second exercise is pure arithmetic and produces a number from the config.

---

## The problem: attention alone is not enough

Look at what attention does: it mixes vectors with weights. A weighted average. And a
weighted average is a **linear operation**.

That is a serious problem, and you can see it with numbers. Imagine stacking two linear
layers with nothing in between:

```
layer 1:  y = W₁ · x
layer 2:  z = W₂ · y = W₂ · (W₁ · x) = (W₂ · W₁) · x
```

$W_2 W_1$ is **a single matrix**. A hundred stacked linear layers are exactly equivalent to
one linear layer. All the depth collapses.

For stacking to be worth anything you need something that is not linear between layers. That
is the FFN's job.

## The classic shape: expand, bend, contract

$$\text{FFN}(x) = W_2 \cdot \text{activation}(W_1 x)$$

With $W_1$ going $d \to 4d$ and $W_2$ going $4d \to d$. It expands to 4 times the size,
applies the nonlinearity, and compresses again.

**Why 4x?** Honestly: because the 2017 paper said so and it worked. There is no derivation.
Other factors have been tried and 4 is still a reasonable point, but it is convention, not
theorem. What does make sense is *expanding*: the nonlinearity has more room to operate, and
there is an interpretation — a debated one — that the FFN works as a key-value memory, where
each of the $4d$ intermediate neurons recognizes a specific pattern.

An important difference from attention: **the FFN processes each token separately**. It does
not mix information between positions. Attention moves information between tokens; the FFN
processes it. They alternate.

## ReLU, and why it is not enough

The simplest nonlinearity is ReLU: $\max(0, x)$. It works, but it has a flaw. Its derivative
is exactly **0** for every negative input. If a neuron ends up always producing negative
values, it stops receiving gradient forever. It is dead and there is no way to recover it.

## GELU: a soft cut

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the probability that a standard normal comes out below $x$.

The intuition: instead of deciding with a hard cut whether to let $x$ through, it multiplies
it by the probability that $x$ "stands out". With numbers:

```
x = -3   ->  Φ(-3) = 0.001   ->  GELU = -0.003    almost cancelled
x = -1   ->  Φ(-1) = 0.159   ->  GELU = -0.159    partially
x =  0   ->  Φ(0)  = 0.5     ->  GELU =  0
x =  1   ->  Φ(1)  = 0.841   ->  GELU =  0.841    almost whole
x =  3   ->  Φ(3)  = 0.999   ->  GELU =  2.996    whole
```

The transition is smooth, so the derivative is never exactly zero: a neuron in the negative
zone can recover.

In practice a tanh approximation is used, because `erf` was slow on 2016 GPUs:

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}\,(x + 0.044715x^3)\right]\right)$$

Today the speed difference is irrelevant, but GPT-2 was trained with the approximation and
it is still used for compatibility. It is what `F.gelu(x, approximate="tanh")` does.

## SwiGLU: adding a gate

Here comes the change our model uses, and every modern one.

The idea behind the **GLU** (*Gated Linear Unit*) variants is having **two** projections
instead of one. One of them acts as a **gate**: it multiplies the other element by element
and decides how much signal passes through each dimension.

$$\text{SwiGLU}(x) = \big(\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}}\big) W_{\text{down}}$$

with $\text{Swish}(z) = z \cdot \sigma(z)$, which is practically GELU under another formula.

The interesting part is that this filtering **depends on the input**. A normal activation
applies the same function to everything; a gate decides, for each dimension and each token,
how much it lets through.

### The 2/3 factor, with the arithmetic

SwiGLU has **three** matrices instead of two. With the same $d_{ff}$ that would be 50% more
parameters. To spend the same, $d_{ff}$ is cut to two thirds:

```
classic FFN:  2 matrices × d × 4d           = 8d²
SwiGLU:       3 matrices × d × (2/3 · 4d)   = 3 · d · (8/3)d = 8d²   ✓
```

With our $d_{\text{model}} = 320$:

```
(2/3) × 4 × 320 = 853.33
```

And then it is rounded **up to the next multiple of 64**: $853.33 \to 896$. That gives the
`d_ff: 896` in the config.

The rounding is not cosmetic. Dimensions aligned to powers of two let the tensor cores take
their fast paths; a matrix with 853 columns is noticeably slower than one with 896, extra
parameters and all.

## Where the parameters are

With the final config, per layer:

| component | parameters | % |
|---|---|---|
| attention ($4d^2$) | 409,600 | 32% |
| SwiGLU ($3 \cdot d \cdot d_{ff}$) | 860,160 | 68% |

**Two thirds of the model is FFN.** When you read that a model has N parameters, most of
them are here, not in the attention. It is also where interpretability research has found
the storage of concrete facts: there is work that locates and edits specific claims by
modifying the FFN weights of particular layers.

## Where the debate is

This module is probably where the "we do not know why" is most explicit, and it comes from
the author himself.

Shazeer (2020) systematically tried every GLU variant and SwiGLU came out best consistently.
His conclusion, quoted literally from the paper:

> *"We offer no explanation as to why these architectures seem to work; we attribute their
> success, as all else, to divine benevolence."*

It is not a joke: it is honesty about the state of the matter. SwiGLU is used today in
Llama, Mistral, PaLM and almost everything else, and the justification is that it works
better on the benchmarks. There is no theory.

The same goes for the 4x and for the interpretation of the FFN as a key-value memory: they
are reasonable observations and hypotheses, not established results. It is worth keeping in
mind when you read explanations that sound very sure of themselves.

---

**Further reading:** Hendrycks & Gimpel 2016,
[Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) · Shazeer 2020,
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (the paper with the
quote) · Geva et al. 2021,
[Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913).
Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
