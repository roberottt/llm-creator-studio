# 14 — Inference and sampling

## Why this module matters

**Because a trained model is no use if you do not know how to get text out of it.**

And getting text out has more to it than it looks. If you always pick the most likely token
—which is the obvious thing— the model gets stuck in loops: *"the cat sat on the mat. the cat
sat on the mat."* The demo shows it. It turns out that **human text does not maximize
probability**, and understanding that is half the module.

The other half is speed. Naive generation recomputes the whole context for every token, which
makes generating N tokens cost N². The KV cache fixes it, and it is the most important
optimization there is in inference: without it, no chatbot would be usable.

### What you will know by the end

- Why always taking the most likely thing produces bad text
- What temperature, top-k and top-p do, and **which one to use when**
- How to generate N times faster without changing a single token of the output
- Why models with very long contexts consume so much memory at inference time

### What it costs

3 hours. The first three exercises are short; the cache is where the difficulty is.

---

## Part 1: how to pick the next token

The model gives you 4096 numbers, one per token of the vocabulary. Which one do you pick?

### Greedy: always the most likely

The obvious thing, and it works badly. It is deterministic —with the same prompt you always
get exactly the same thing— and above all **it gets stuck in loops**:

```
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
```

The reason is subtle and Holtzman et al. (2020) explain it well: human text **does not
maximize probability**. A person writes surprising things every now and then; always picking
the most likely thing produces flat, repetitive text, even though each individual token is
plausible.

### Temperature: flattening or sharpening the distribution

You divide the logits by a number before the softmax:

$$P_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

With numbers. Suppose logits `[3, 2, 1]`:

```
T = 1.0   →  [0.665, 0.245, 0.090]     the distribution as it is
T = 0.5   →  [0.867, 0.117, 0.016]     sharper: almost always the first one
T = 2.0   →  [0.506, 0.307, 0.186]     flatter: more variety
T → 0     →  [1, 0, 0]                 equivalent to greedy
```

Dividing by a small number **separates** the logits, and since the softmax is exponential,
that separation gets amplified. Dividing by a large one **brings them together**.

Typical values: 0.7–0.9 for coherent text, 1.0 for variety, above 1.2 it starts to ramble.

### Top-k: keeping the k best

The problem with temperature alone is that it **never eliminates** the bad tokens, it just
makes them less likely. With 4096 tokens, the long tail can hold 20% of the mass across
thousands of absurd options, and every so often one comes out.

Top-k cuts it dead: sort, keep the `k` largest, set the rest to $-\infty$.

Its flaw is that `k` is **fixed**. If the model is dead sure about the next token, k=40 lets
in 39 bad alternatives. If it is genuinely undecided between 100, it cuts off good options.

### Top-p (nucleus): keeping the ones that add up to p

The answer to that flaw. Instead of a fixed number, you accumulate probability until you reach
`p` and cut there:

```
probs = [0.60, 0.25, 0.10, 0.03, 0.02]
p = 0.9

accumulated without this token:
  0.60  →  0.00  ≤ 0.9  →  in
  0.25  →  0.60  ≤ 0.9  →  in
  0.10  →  0.85  ≤ 0.9  →  in    ← the one that CROSSES the threshold goes in too
  0.03  →  0.95  > 0.9  →  out
  0.02  →  0.98  > 0.9  →  out
```

It keeps 3 candidates, which add up to 0.95.

**Note the token that crosses the threshold: it goes in.** Holtzman's definition is *"the
smallest set whose cumulative probability **exceeds** p"*, and `[0.60, 0.25]` adds up to 0.85,
which does not exceed 0.9. You need the third one. If you cut earlier, the set would not reach
the requested mass.

It is an easy off-by-one to get wrong —I had it wrong writing this module— and that is why in
the code the comparison is against the cumulative **before** including each token.

And if the distribution were `[0.2, 0.2, 0.2, 0.2, 0.2]`, it would keep all 5. **The number of
candidates adapts to how sure the model is**, and that is exactly what you want.

An implementation detail: **the most likely token is always kept**, even if it alone already
exceeds `p`. Otherwise, with `p=0.5` and a token of probability 0.9 you would be left with no
candidates.

### Repetition penalty

A direct patch against loops: lower the logit of the tokens that have already come out.

Here there is a detail that almost everyone implements wrong. You have to **divide if the
logit is positive and multiply if it is negative**:

```
logit = +3  →  3 / 1.1 = 2.73    moves it towards zero
logit = -3  →  -3 * 1.1 = -3.3   moves it away from zero, downwards
```

If you always divided, a logit of −5 would become −4.5, that is, the token would become **more**
likely: exactly the opposite.

## Part 2: the KV cache

Now the speed part, and it is where the big win is.

### The problem

When generating token 100, the naive version passes all 100 tokens through the model. Again.
Even though the first 99 have not changed at all.

Generating N tokens costs $O(N^2)$ when it should cost $O(N)$.

### The solution

Save the keys and values of each layer. At every step, process **only the new token** and
concatenate its K and V to what is saved.

What you **cannot** cache are the queries: every new token needs its own question. What gets
reused are the answers (K) and the contents (V) of the previous ones. Hence the name.

The loop ends up in two phases:

1. **Prefill:** the whole prompt goes through at once and fills the cache.
2. **Decode:** at each step a single token goes in.

### The detail that breaks everything if you forget it

**RoPE has to rotate the new token with the angle of its real position.** When generating
token 50 you pass it a tensor of length 1, and if you apply RoPE as is, it will rotate it as if
it were position 0.

That is why `apply_rope` needs to know how many tokens are already in the cache:

```python
cos_t = cos[pos_offset : pos_offset + seq_len]
```

Without that slice, generation with the cache produces different —and worse— text than without
it, and the bug is hard to find because nothing fails: the model simply writes badly.

### The memory

$$\text{KV memory} = 2 \times n_{\text{layers}} \times T \times d_{\text{model}} \times \text{bytes}$$

Our model with 512 tokens in fp16: **3.9 MB**. Nothing.

A 70B model with a 100,000-token context: tens of gigabytes, more than the weights themselves.
That is why techniques like *grouped-query attention* exist, which share K and V across several
heads.

### A limitation worth knowing about

Generation with the cache **stops** when it reaches the maximum context, instead of cropping
the way the naive version does.

It is not laziness. Cropping with a cache would require discarding the old entries **and
remapping the RoPE positions** of everything that is left, because the surviving tokens would
end up in different positions from the ones they were rotated with. That is called *sliding
window attention*, and it is a topic in itself.

Stopping is the honest thing: the silent alternative would be generating incorrect text
without warning.

### The mandatory check

The cache has to give **exactly the same output**, not a similar one. With `temperature=0`
(greedy, deterministic) the two sequences must match token by token. If they do not, there is
a bug.

In the demo you will see both things: identical output and a 2–3× speedup that grows with the
length.

## Where the debate is

**Nobody knows what the right sampling parameters are.** The values in use —temperature 0.8,
top-p 0.9— are inherited folklore, tuned by eye on particular models. There is no theory that
derives them, and the optimum depends on the model, on the task and on who you ask.

More fundamentally: **why human text does not maximize probability** still has no satisfying
explanation. Holtzman et al. documented it empirically and proposed top-p as a remedy, but the
underlying question —what distribution actually generates human language and why maximizing
likelihood moves away from it— is open.

And there is a practical discussion going on about whether sampling should be replaced by
something better. Alternatives have been proposed (typical sampling, mirostat, min-p) with
reasonable arguments, and none has displaced top-p. Maybe because they are not better, or
maybe out of inertia.

---

**Further reading:** Holtzman et al. 2020,
[The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (top-p) ·
Fan et al. 2018, [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833)
(top-k). Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
