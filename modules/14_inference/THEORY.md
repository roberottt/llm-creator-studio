# 14 — Inference: getting text out of the model, and getting it out fast

## Why this module matters

**Because a trained model is useless if you cannot get text out of it.**

And getting text out has more to it than it seems. If you always pick the most probable token —
the obvious thing — the model gets stuck in loops: *"the cat sat on the mat. the cat sat on the
mat."* The demo shows it. It turns out that **human text does not maximize probability**, and
understanding that is half the module.

The other half is speed. Naive generation recomputes the whole context on every token, which
makes generating N tokens cost N². The KV cache fixes that, and it is the single most important
optimization in inference: without it, no chatbot would be usable.

### What you will know by the end

- Why always picking the most probable token produces bad text
- What temperature, top-k and top-p do, **and in what order they are applied**, which is what
  almost never gets said
- How to generate N times faster without changing a single token of the output
- Why models with very long contexts consume so much memory at inference time

### What you are going to write

Five exercises, and this theory follows them in order. The first three are filters over logits,
independent of each other; the last two are the cache and they chain:

| Exercise | What it does |
|---|---|
| 1. `apply_repetition_penalty` | Breaking the loops |
| 2. `top_k_filter` | Keeping the k best |
| 3. `top_p_filter` | Keeping the ones that add up to p |
| 4. `KVCache` | Storing what has already been computed |
| 5. `generate_with_cache` | The loop that puts it all together |

The first three are short and each fits in four or five lines. **Exercise 5 is where the
difficulty is**, and it has a merciless check: with the cache it has to produce *exactly* the same
text as without it. Not similar: identical, token by token.

### What it costs

3 hours.

---

## Part 1: how to choose the next token

The model gives you 4096 numbers, one per vocabulary token. Which one do you pick?

### Greedy: always the most probable

The obvious choice, and it works badly. It is deterministic — the same prompt always gives exactly
the same thing — and above all **it gets stuck in loops**:

```
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
```

The reason is subtle and Holtzman et al. (2020) explain it well: human text **does not maximize
probability**. A person writes surprising things now and then; always picking the most probable
produces flat, repetitive text, even though every individual token is plausible.

The demo measures it with the 4-gram variety of the generated text: greedy only gets 91%, while
the sampled variants reach 99-100%.

### Temperature: flattening or sharpening the distribution

You divide the logits by a number before the softmax:

$$P_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

With the demo's logits `[3, 2, 1, 0.5]`:

| T | tok 0 | tok 1 | tok 2 | tok 3 | effect |
|---|---|---|---|---|---|
| 0.5 | 0.862 | 0.117 | 0.016 | 0.006 | sharp: almost always the first |
| 1.0 | 0.623 | 0.229 | 0.084 | 0.051 | the distribution as it is |
| 2.0 | 0.409 | 0.248 | 0.150 | 0.117 | flat: more variety |

Dividing by a small number **separates** the logits, and since the softmax is exponential that
separation gets amplified. Dividing by a large one **brings them together**. With $T \to 0$ you
recover greedy.

Typical values: 0.7–0.9 for coherent text, 1.0 for variety, above 1.2 it starts rambling.

---

## Exercise 1: the repetition penalty (`apply_repetition_penalty`)

A direct patch against loops: lower the logit of tokens that have already appeared.

There is one detail here that almost everybody implements wrong, and it is the whole exercise. You
have to **divide if the logit is positive and multiply if it is negative**:

```
   token 0, logit +3.0  ->  3.0 / 1.1 = +1.50    moves it towards zero
   token 5, logit -3.0  -> -3.0 * 1.1 = -6.00    moves it away from zero, downwards
```

(The numbers are the demo's, with `penalty = 2.0`.)

If you always divided, the −3.0 would become −1.5 and the token would become **more** probable:
exactly the opposite of penalizing it. And it is not a rare case: in a 4096-token vocabulary most
logits are negative most of the time, so dividing indiscriminately would penalize a handful of
tokens and reward thousands.

The good thing about this filter is that it **rescues greedy without taking away its
determinism**: it stays reproducible, but it no longer gets stuck.

---

## Exercise 2: top-k (`top_k_filter`)

The problem with temperature alone is that it **never eliminates** the bad tokens, it only makes
them less probable. With 4096 tokens, the long tail can accumulate 20% of the mass across
thousands of absurd options, and every so often one comes out.

Top-k cuts it dead: sort, keep the `k` largest, set the rest to $-\infty$.

Its flaw is that `k` is **fixed**. If the model is dead sure about the next token, `k=40` brings in
39 bad alternatives. If it is genuinely torn between 100, it cuts good options. Hence the next
exercise.

---

## Exercise 3: top-p or nucleus (`top_p_filter`)

The answer to that flaw. Instead of a fixed number, you accumulate probability until you reach `p`
and cut there. With the demo's table and `p = 0.9`:

| token | logit | prob | cumulative | top-k=2 | top-p=0.9 |
|---|---|---|---|---|---|
| 0 | +3.0 | 0.623 | 0.623 | yes | yes |
| 1 | +2.0 | 0.229 | 0.852 | yes | yes |
| 2 | +1.0 | 0.084 | 0.936 | no | **yes** |
| 3 | +0.5 | 0.051 | 0.987 | no | no |
| 4 | −1.0 | 0.011 | 0.998 | no | no |
| 5 | −3.0 | 0.002 | 1.000 | no | no |

**Look at token 2, the one that crosses the threshold: it gets in.** Holtzman's definition is
*"the smallest set whose cumulative probability **exceeds** p"*, and `[0.623, 0.229]` adds up to
0.852, which does not exceed 0.9. The third one is needed.

It is an extremely easy off-by-one to get wrong, and that is why in the code the comparison is made
against the cumulative **before** including each token, not after.

And compare the last two columns of the table: with these logits, top-k=2 leaves 2 candidates and
top-p leaves 3. If the distribution were `[0.2, 0.2, 0.2, 0.2, 0.2]`, top-k would still leave 2 and
top-p would leave all 5. **The number of candidates adapts to how sure the model is**, and that is
exactly what you want.

**An implementation detail:** the most probable token is always kept, even if it alone already
exceeds `p`. Otherwise, with `p=0.5` and a token of probability 0.9 you would be left with no
candidates.

### The order they are applied in, which almost never gets said

The three filters and the temperature get combined, and **the order matters**. In exercise 5 you
write it like this:

```
   penalty  ->  temperature  ->  top-k  ->  top-p  ->  sample
```

- The **penalty goes first** because it operates on the raw logits: its divide-or-multiply rule
  depends on the sign, and temperature does not change signs but does change magnitudes.
- **Temperature goes before the filters** because it changes the probabilities, and top-p looks at
  cumulative probabilities. Applying top-p before the temperature would filter using a different
  distribution from the one later used to sample.

---

## Exercise 4: the KV cache (`KVCache`)

Now the speed part, and this is where the big win is.

**The problem.** When generating token 100, the naive version pushes all 100 tokens through the
model. Again. Even though the first 99 have not changed at all. Generating N tokens costs
$O(N^2)$ when it should cost $O(N)$.

**The fix.** Store each layer's keys and values. At each step, process **only the new token** and
concatenate its K and V to what is stored.

What you **cannot** cache are the queries: each new token needs its own question. What gets reused
are the labels (K) and contents (V) of the previous ones. Hence the name, and this is where module
06's split into three projections pays off.

The exercise itself is a simple data structure: a list of tensors per layer, a method to append and
another to report how much memory it takes. The difficulty comes in the next one.

---

## Exercise 5: the full loop (`generate_with_cache`)

This is where it all comes together. The loop ends up in two phases:

1. **Prefill:** the whole prompt goes through at once and fills the cache.
2. **Decode:** each step takes a single token in, reads the cache, and appends the new one.

### The detail that breaks everything if you forget it

**RoPE has to rotate the new token by the angle of its real position.** When generating token 50
you pass it a length-1 tensor, and if you apply RoPE as-is it will rotate it as if it were position
0.

That is why you need to know how many tokens are already in the cache:

```python
cos_t = cos[pos_offset : pos_offset + seq_len]
```

Without that slice, cached generation produces different — and worse — text than uncached, and the
bug is hard to locate because nothing fails: the model simply writes badly.

### The mandatory check

The cache has to give **exactly the same output**, not similar. With `temperature=0` generation is
greedy and therefore deterministic, so the two sequences must match token by token:

```
   without cache: [44, 1, 58, 46, 43, 1, 41, 53, 51, 51]
   with cache:    [44, 1, 58, 46, 43, 1, 41, 53, 51, 51]
```

If they do not match, the first thing to look at is RoPE's `pos_offset`. The second is the causal
mask during prefill: in prefill the whole prompt comes in, so a mask **is** needed; it can only be
skipped in decode, when a single token comes in and legitimately sees the whole past. If it is
skipped in both phases, the prompt tokens see each other forwards and that leak corrupts the K and
V left in the cache — so everything you generate afterwards comes out wrong. That bug was in this
course's reference until this very check caught it.

### The gain

Measured on a model with context 1024 (with the toy's context of 128 the long runs hit the limit
and the comparison flattens out right where it starts getting interesting):

| tokens | without cache | with cache | speedup | cache memory |
|---|---|---|---|---|
| 50 | 153 ms | 129 ms | 1.18× | 232 KB |
| 100 | 381 ms | 246 ms | 1.55× | 432 KB |
| 200 | 765 ms | 496 ms | 1.54× | 832 KB |
| 400 | 1566 ms | 996 ms | 1.57× | 1632 KB |
| 800 | 3585 ms | 1990 ms | 1.80× | 3232 KB |

**The speedup grows with length**, which is what to look at: without cache, generating N tokens
costs $O(N^2)$ and with cache $O(N)$. At the lengths of a real chatbot the difference stops being
1.8× and becomes orders of magnitude.

### The memory

$$\text{KV memory} = 2 \times n_{\text{layers}} \times T \times d_{\text{model}} \times \text{bytes}$$

Our model with 512 tokens in fp16: **3.9 MB**. Nothing. A 70B model with a 100,000-token context:
tens of gigabytes, more than the weights themselves. That is why techniques like *grouped-query
attention* exist, sharing K and V across several heads.

### A limitation worth knowing about

Cached generation **stops** when it reaches the maximum context, instead of cropping the way the
naive version does.

It is not laziness. Cropping with a cache would require discarding the old entries **and remapping
the RoPE positions** of everything left, because the surviving tokens would end up at different
positions from the ones they were rotated with. That is called *sliding window attention* and it is
a module in itself. Stopping is the honest option: the silent alternative would be generating
incorrect text without warning.

---

## Where the debate is

**Nobody knows what the right sampling parameters are.** The values people use — temperature 0.8,
top-p 0.9 — are inherited folklore, eyeballed on specific models. There is no theory that derives
them, and the optimum depends on the model, the task, and who you ask.

More fundamentally: **why human text does not maximize probability** still lacks a satisfying
explanation. Holtzman et al. documented it empirically and proposed top-p as a remedy, but the
underlying question — what distribution actually generates human language and why maximizing
likelihood drifts away from it — is open.

And there is an ongoing practical discussion about whether sampling should be replaced with
something better. Alternatives have been proposed (typical sampling, mirostat, min-p) with
reasonable arguments, and none has displaced top-p. Maybe because they are not better, or maybe out
of inertia.

---

**Further reading:** Holtzman et al. 2020,
[The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (top-p) ·
Fan et al. 2018, [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833)
(top-k). Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
