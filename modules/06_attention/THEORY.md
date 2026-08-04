# 06 — Self-attention

## Why this module matters

**If you could only understand one module in the course, it would be this one.**

Everything before it — tokenizing, embeddings, Bengio's MLP — already existed in 2003 and
gave mediocre models. What changed in 2017 and eventually produced ChatGPT is exactly what
you are going to program here, and it fits in four lines of code.

The idea is simple to state: **let each word look at the previous ones and decide which to
pay attention to**. What is hard is believing that this is enough. By the end of the module
you will have seen it working in a model you trained yourself, with a heatmap that literally
shows what each letter is looking at.

### What you will know by the end

- Why a model can "remember" something it read 300 words earlier
- What Q, K and V are, and **why three things are needed and not one**
- Why we divide by `√d_k`, and what exactly breaks if you remove it (with numbers)
- What the causal mask is and why it is **the most expensive bug in the course** if you get
  it wrong
- You will have seen an attention heatmap from a model you trained

### What it costs

4 hours, tied with module 03 as the longest. It is the one most worth it.

---

## What problem does attention solve?

Sentence: *"the cat I saw yesterday was sleeping"*.

To get `sleeping` right you have to know that the subject is `cat`, five words back. Module
05's MLP cannot: it looks at a fixed window and treats every position the same, with no way
to say "of these tokens, the one that matters to me now is the first one".

Attention lets **each word look at the previous ones and decide which to pay attention to**.
Not with a fixed rule, but by computing it from the content.

## With real numbers

Let us do it by hand with 3 words and 2-dimensional vectors. Say that after the embeddings
we have:

```
cat      = [1.0, 0.2]
yesterday= [0.1, 0.9]
sleeping = [0.8, 0.3]
```

`sleeping` wants to know who to look at. It does so with a **dot product**: it measures how
similar two vectors are. The more aligned, the larger the number.

```
cat · sleeping       = 1.0×0.8 + 0.2×0.3 = 0.86      -> a lot
yesterday · sleeping = 0.1×0.8 + 0.9×0.3 = 0.35      -> a little
sleeping · sleeping  = 0.8×0.8 + 0.3×0.3 = 0.73      -> to itself
```

Those numbers are called **scores**. Now they have to be turned into weights that sum to 1,
and for that we use softmax (exponentiate and normalize, like module 00 but allowing
negatives):

```
softmax([0.86, 0.35, 0.73]) = [0.40, 0.24, 0.36]
```

And those weights are used to mix the vectors:

```
output = 0.40×cat + 0.24×yesterday + 0.36×sleeping
```

That is attention. **A weighted average where the weights are decided by the content
itself.** `sleeping`'s representation now carries 40% of `cat` inside it, which is exactly
the information it needed.

## Q, K, V: why three projections and not one

In the example I used the same vector for everything, and that is too rigid. A token needs
to do three different things:

- **ask** something ("I am looking for a singular subject")
- **advertise itself** to the others ("I am a singular noun")
- **contribute** content if it gets chosen ("the concept of a cat")

They are three different roles, so three different linear projections of the same input
vector are learned:

$$Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V$$

**Query** (question), **Key** (label) and **Value** (content). Similarity is computed
between queries and keys; what gets mixed are the values. That way the model can learn that
`cat` *answers well* to a question about subjects without that constraining *what
information it contributes* when it is chosen.

## The formula

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

It is exactly what we just did:

- $QK^\top$ is every dot product at once: a $T \times T$ matrix where cell $(i,j)$ says how
  interested $i$ is in token $j$.
- $\sqrt{d_k}$ is the scaling (we will get to it).
- $M$ is the causal mask.
- softmax turns each row into weights that sum to 1.
- multiplying by $V$ does the mixing.

## The scaling by √d_k: what happens if you remove it

This divisor looks like an arbitrary detail and it is not.

A dot product of two $d_k$-dimensional vectors with independent components of mean 0 and
variance 1 has **variance $d_k$**. With $d_k = 40$ (our case), the scores move in a range of
roughly $\pm 6$. With $d_k = 512$, in $\pm 22$.

So what? Softmax is exponential. If one score stands 20 units above the rest, $e^{20}$
against $e^{0}$ is 485 million to one: the softmax returns practically
`[0, 0, ..., 1, ..., 0]`. Attention becomes a hard selection of a single token.

And the real problem is not the forward pass, it is the **gradient**. The softmax's
derivative is $p(1-p)$; with $p$ pinned to 0 or 1, the derivative is practically zero. The
layer stops learning. Dividing by $\sqrt{d_k}$ returns the scores to variance 1, the softmax
stays in a soft region, and the gradient flows.

The module's demo shows you this by measuring the entropy of the distribution with and
without the scaling.

## The causal mask

During training we pass the whole sequence in at once and ask it to predict each token from
the previous ones. With nothing else, position 3 could look at position 4 — that is, see the
answer.

The mask puts $-\infty$ in the forbidden scores *before* the softmax. Since
$e^{-\infty} = 0$, those positions receive exactly zero weight:

```
[[ ✓  ·  ·  · ]      token 0 only sees itself
 [ ✓  ✓  ·  · ]      token 1 sees token 0 and itself
 [ ✓  ✓  ✓  · ]
 [ ✓  ✓  ✓  ✓ ]]
```

It goes before the softmax and not after for a specific reason: if you zeroed the weights
afterwards, the rows would no longer sum to 1. By putting $-\infty$ first, the softmax
normalizes only over what is allowed.

**This is the most expensive bug in the course.** If the mask is wrong, the loss drops
spectacularly, everything seems to be going wonderfully, and the trained model is useless
because at generation time that future does not exist. That is why module 05 insists on
comparing the step-0 loss against $\ln(V)$: if it comes out *lower*, look at the mask.

## Multi-head: several in parallel

A single attention has to resolve every relationship in the sentence with one pattern. The
fix is to do several in parallel, each with its own $W_Q, W_K, W_V$, and concatenate the
results.

With $d_{\text{model}} = 320$ and 8 heads, each works in $40$ dimensions ($320/8$). **It
costs no more**: instead of one 320-dimensional attention you do eight of 40, and the total
parameter count is identical.

The interesting part is that the heads specialize on their own. In trained models people
have identified heads that look at the previous token, heads that pair opening and closing
quotes, and the so-called *induction heads*, which detect the pattern "…A B … A" and predict
B. Nobody programmed them.

The implementation trick: you do not make 8 separate projections. You make one $320 \to 320$
projection and split the result into 8 chunks of 40. It is mathematically equivalent and
much faster, because it is one big matmul instead of eight small ones.

## Where the debate is

We know *what* attention computes. Why it works so well is another matter.

The intuitive explanation — "each token retrieves relevant information" — is a reasonable
story and it is not proven. There are results that complicate it: models with **fixed,
random** attention patterns work surprisingly well on some tasks, which suggests that part
of the credit belongs to the general architecture (residuals, normalization, depth) and not
only to the attention mechanism.

The most serious line of work in this direction is mechanistic interpretability, which tries
to read the circuits that form inside. It has managed to explain specific components — the
*induction heads* are the success story — but it is a long way from accounting for a whole
model.

And there is a structural limitation still unsolved: the cost grows with the **square** of
the context length. Dozens of subquadratic alternatives have been proposed (Linformer,
Performer, Mamba and family). None has displaced standard attention in general-purpose
models, and it is not clear whether that is because full attention is necessary or because
it has a twenty-year head start in kernel optimization.

---

**Further reading:** Vaswani et al. 2017,
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) · Elhage et al. 2021,
[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
(the *induction heads*). Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
