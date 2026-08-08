# 09 — Positional information and RoPE: telling the model what comes first

## Why this module matters

**Because attention does not know which word comes first.**

It sounds like a detail and it is a fatal flaw. Look again at module 06's formula: it is a
weighted sum, and a sum has no order. To the attention mechanism, *"the dog bites the man"* and
*"the man bites the dog"* produce exactly the same thing.

That is not a figure of speech; you checked it with numbers in module 06. Taking the three-token
example, swapping two of them and looking at the last position's output:

```
   original order:  [0.4708, 0.5040]
   swapped order:   [0.4708, 0.5040]     identical
```

This is where it gets fixed, and it gets fixed with a rather beautiful idea: instead of adding a
label to the vector saying "I am position 7", a **rotation** is applied whose angle depends on
the position. That makes the model learn relative distances — "the token two positions back" —
instead of absolute positions.

It is the technique used by Llama, Mistral and practically everything modern. And it has a
property that makes it even more striking: **it does not add a single parameter to the model.**
The tables you are going to compute are fixed, not trained. The classic alternative — a learned
table — would have cost 163,840 parameters, 1.8% of the model; RoPE costs zero.

### What you will know by the end

- Why without this your model cannot tell word order apart
- Three ways of solving it, in historical order, and what fails in each
- What RoPE is and **the mathematical property that justifies it**, checked with numbers you can
  reproduce
- **How the dimensions get paired up for rotating**, which is what makes exercise 2 look like
  magic if nobody shows you a table
- What really happens when you ask a model for a longer context than it trained on, measured

### What you are going to write

Three exercises. This theory is ordered so that you read them in this order, and **each one has
its own section with the matching numeric example**:

| Exercise | What it does | Where it is explained |
|---|---|---|
| 1. `sinusoidal_embeddings` | The 2017 paper's table (historical) | [§ Sines and cosines](#exercise-1-sines-and-cosines-sinusoidal_embeddings) |
| 2. `rope_frequencies` | Precomputing the rotation angles | [§ RoPE's tables](#exercise-2-the-angle-tables-rope_frequencies) |
| 3. `apply_rope` | Rotating Q and K | [§ Rotating](#exercise-3-actually-rotating-apply_rope) |

Exercise 2 is the hard one, and it is hard because of a single step: the one that duplicates the
frequencies. Exercise 3 is one line, but it only makes sense after understanding exercise 2.
Exercise 1 is not used by our model and is still worth doing — its section explains why.

### What it costs

2.5 hours.

---

## Which part of the LLM this is

Module 08 closed off the Transformer block. This one does not add a new box to the drawing: **it
goes inside one you already wrote.**

And in fact you have already called it. Go back to module 06's exercise 3, the
`MultiHeadAttention`. There is a step you copied without really knowing what it did:

```python
if cos is not None and sin is not None:
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
```

That `apply_rope` came ready-made from `llmfs.reference` with a comment saying it belonged to
module 09 and to ignore it for now. **Today you write it**, and the `cos` and `sin` arguments of
that signature are exactly what exercise 2 produces.

Where it fits, then:

```
   x  ──> q_proj ──> _split_heads ──>  q  (B, 8, T, 40)
                                        │
                                        ▼   apply_rope(q, cos, sin)      ← exercise 3
                                        │
                                     rotated q
                                        │
                                        ▼
                              q @ kᵀ / √d_k  ...  the rest of module 06
```

Three things from that diagram worth fixing in your head before going on:

- **It is applied inside each head**, over `head_dim` dimensions (40 in our case, that is 20
  pairs), not over `d_model`'s 320. That is why it goes *after* `_split_heads`.
- **Only to Q and K, never to V.** What should depend on position are the attention *scores*,
  not the content being transported. If you rotated the values too, you would be injecting
  position twice and muddying what the token contributes.
- **It is not a layer.** No `nn.Module`, no weights, nothing to train. The `cos` and `sin` tables
  are computed once when the model is built and stored as *buffers* — which is what PyTorch calls
  tensors that travel with the model but are not parameters. In module 10's GPT you will see them
  as `rope_cos` and `rope_sin`, with shape `(512, 40)`.

---

## The problem: attention does not know what comes first

Look again at module 06's attention formula. It is a weighted sum of the values, and the weights
come from dot products between queries and keys.

Nowhere does **position** appear.

The consequence is the one from the opening: if you shuffle the input tokens, the output gets
shuffled the same way and nothing else changes. That property is called **permutation
equivariance**, and in almost any other context it would be a virtue (processing a set of things
without the order mattering). Here it is a fatal flaw, because word order is half the meaning.

Position has to be injected somehow. We are going to look at three ways, in historical order,
because each one fixes a problem with the previous one and that is how RoPE comes about.

## Option 1: learn a table

The simplest one. A table with one row per position, trained like any other parameter, and
**added** to the token's embedding:

```
   input = token_embedding[id] + position_embedding[i]
```

That is what GPT-2 does. It works fine and there is no mystery to it: if row 7 ends up containing
whatever it takes for the model to know it is at position 7, problem solved.

It has two drawbacks, and both matter.

The first is a **hard ceiling**: if you trained with 1024 positions, there is no row to look up
for position 1025. The model cannot process it at all, not even badly. You will see this
literally in the demo, where this option's column says "cannot" as soon as the sequence exceeds
the trained context.

The second is subtler: the model learns **absolute** positions — "this is token number 7" —
when what usually matters is the relationship: "this is two words before the verb". And since
each row is trained separately, what it learns about position 7 is of no use for position 300,
even if the relationship they represent is the same.

## Option 2: sines and cosines

The 2017 paper proposed a **fixed** table, with no parameters, made of sines and cosines of
different frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

It is still something that gets **added** to the embedding, like option 1. What changes is where
the numbers come from: instead of being learned, they are computed.

### The intuition: a binary counter

Look at how you count in binary:

```
   0000    the rightmost bit changes at every step
   0001    the next one, every two
   0010    the next one, every four
   0011    ...
```

Each bit oscillates at a different rate, and the combination of all of them identifies a number
uniquely without any single bit having to do all the work. Sinusoidals do the same thing but with
continuous waves: the first pairs of dimensions oscillate fast and tell neighbouring positions
apart; the last ones oscillate extremely slowly and tell the beginning of the sequence from the
end.

Advantage over the learned table: it is defined for **any** position, there is no ceiling. Ask it
for row 100,000 and it computes it. In practice extrapolation does not work all that well either
— we will measure that at the end — but at least it exists.

---

## Exercise 1: sines and cosines (`sinusoidal_embeddings`)

### First, why you are writing something the model does not use

Worth saying up front so it does not nag at you: **our model does not use this function.** It
uses RoPE, which is exercises 2 and 3. This is option 2, the 2017 one, and it is here for three
reasons:

1. It is what introduces the **frequency ladder**, which is exactly the idea RoPE reuses. If you
   understand this table, exercise 2 stops being magic.
2. It shows up in a great deal of code and in every paper from that era; you will run into it.
3. The demo actually trains it and compares it with the other two, so the implementation has to
   exist for that comparison to happen.

### The example, with the whole table

With `seq_len = 5` and `d_model = 4`, your function has to return exactly this:

```
   position 0:  [ 0.0000,  1.0000,  0.0000,  1.0000]
   position 1:  [ 0.8415,  0.5403,  0.0100,  0.9999]
   position 2:  [ 0.9093, -0.4161,  0.0200,  0.9998]
   position 3:  [ 0.1411, -0.9900,  0.0300,  0.9996]
   position 4:  [-0.7568, -0.6536,  0.0400,  0.9992]
                 └───┬───┘└───┬────┘└───┬───┘└──┬──┘
                    sin      cos       sin     cos
                   fast     fast      slow    slow
```

Read it by columns and you will see the ladder. The first two columns are $\sin(pos)$ and
$\cos(pos)$: they oscillate fast, and from position 0 to 4 they have already covered more than
half a turn. The last two are $\sin(0.01 \cdot pos)$ and $\cos(0.01 \cdot pos)$: in five
positions they have barely moved from `[0, 1]`. With `d_model = 4` there are only two speeds;
with 320 there are 160, spread between those two extremes.

That `0.01` comes from `div_term`, which for `d_model=4` is `[1.0, 0.01]`. It is step 2 of the
docstring, and it is worth understanding why it is written with exponentials:

```python
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
```

`exp(-log(base) · 2i/d)` is **mathematically identical** to `base ** (-2i/d)`, but much more
numerically stable: raising 10000 to a large negative power loses floating-point precision, and
going through logarithms does not. It is a general rule that will serve you elsewhere: if you see
a power with a large exponent, `exp(log(...))` is usually better.

### The interleaving

The other step that can throw you is step 4:

```python
embeddings[:, 0::2] = torch.sin(position * div_term)
embeddings[:, 1::2] = torch.cos(position * div_term)
```

`[:, 0::2]` means "every row, columns from 0 in steps of two", that is, the **even** ones;
`[:, 1::2]`, the odd ones. That is how sine and cosine get interleaved without writing a loop:
each frequency occupies two adjacent columns, one with the sine and one with the cosine. You can
see it in the table above: columns 0 and 1 share a frequency, and 2 and 3 share the other.

And the `position * div_term` inside is a broadcast of `(T,1)` by `(d/2,)` giving `(T, d/2)`: all
the angles for all the positions at once, no loops. It is the same pattern you already used in
module 05 with the bigram probabilities.

---

## Option 3: RoPE, rotate instead of adding

Here is the idea our model uses, and Llama's, and almost everything modern.

**Instead of adding something to the vector, a rotation is applied whose angle depends on the
position.**

Take a 2-dimensional vector and rotate it by an angle $\theta$. The good old rotation matrix:

$$\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} =
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

RoPE splits each head's vector into pairs and rotates each pair by an angle proportional to the
position. As with the sinusoidals, each pair has its own rotation speed: the first ones spin
fast, the last ones extremely slowly. With `head_dim = 40` that is 20 pairs, so 20 speeds.

### Why this is such a good idea

And here comes the property that justifies everything. Rotations have a peculiarity: **the dot
product of two rotated vectors depends only on the difference of angles.**

$$\langle R(m)\,q,\; R(n)\,k \rangle = \langle q,\; R(n-m)\,k \rangle$$

Translated into what matters: the attention score between the token at position 5 and the one at
position 3 is **identical** to the one between 105 and 103. What the model learns is not "token
number 3" but **"the token two positions back"**, and it can apply that anywhere in the sequence.

You do not have to take my word for it. Taking one same pair of vectors `q` and `k`, placing them
at different positions and measuring the score between them:

| positions (q, k) | distance | score |
|---|---|---|
| (0, 3) | 3 | −5.9859375954 |
| (2, 5) | 3 | −5.9859371185 |
| (10, 13) | 3 | −5.9859361649 |
| (100, 103) | 3 | −5.9859242439 |
| (200, 203) | 3 | −5.9859414101 |
| (0, 7) | 7 | −0.7609109879 |
| (50, 57) | 7 | −0.7609119415 |

The first five rows are at distance 3 and give the same number to seven significant figures —
the differences at the end are floating-point rounding, not the mechanism — even though one is at
the start of the sequence and another at position 200. The last two, at distance 7, give a
different value but one that is also equal between them. When you finish exercise 3 you can
reproduce this table yourself; there is a test that checks it.

And there is a second, quieter but important benefit: **rotating does not change the vector's
length**. Adding a positional embedding does alter the magnitude, and since attention scores are
dot products — which depend on lengths — that alteration leaks into the scores without anyone
asking for it. A rotation only changes direction.

---

## Exercise 2: the angle tables (`rope_frequencies`)

This exercise rotates nothing: it precomputes, for each position and each pair of dimensions, the
cosine and sine of the angle that will apply. It runs once when the model is built and gets
reused across all six blocks and all the thousands of training steps.

Four steps, and only one of them is odd. Let us go with a tiny example, `head_dim = 4`, which is
two pairs.

### Step 2: the frequencies

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
```

With `head_dim=4` and `theta=10000`: exponents `[0/4, 2/4]`, that is, `inv_freq = [1.0, 0.01]`.
Two speeds, the same ladder as in exercise 1.

For the real model, with `head_dim=40`, the ladder looks like this (measured):

| pair | frequency (rad/position) | full turn every |
|---|---|---|
| 0 | 1.000000 | 6 positions |
| 4 | 0.316228 | 20 positions |
| 8 | 0.100000 | 63 positions |
| 16 | 0.010001 | 628 positions |
| 24 | 0.000977 | 6,434 positions |
| 31 | 0.000000 | never |

Keep that last row in mind: it is the origin of all of RoPE's limitations, and we come back to it
at the end.

### Step 3: all the angles

```python
angles = torch.outer(positions, inv_freq)      # (T, head_dim/2)
```

`torch.outer(a, b)[i,j] = a[i] * b[j]`, which is exactly what is needed: every combination of
position by frequency. In our example, position 1's row is `[1.0, 0.01]` and position 2's is
`[2.0, 0.02]`.

### Step 4: the confusing one, and why

```python
angles = torch.cat([angles, angles], dim=-1)   # (T, head_dim)
```

Duplicating the table by gluing it to itself. The first time you see this it looks like a bug:
why have 4 columns of angles if there are only 2 frequencies?

The answer is in **how the dimensions get paired up for rotating**, and there are two conventions:

```
   the original paper pairs CONSECUTIVE ones:  (x0,x1), (x2,x3), ...
   Llama and HuggingFace, by HALVES:           (x0,x2), (x1,x3), ...   ← we use this one
```

With the halves convention and `head_dim=4`, the pair that rotates together is `(x0, x2)` and the
other is `(x1, x3)`. Both components of a pair need **the same angle**, so column 0's angle has to
be repeated in column 2, and column 1's in column 3. And that is exactly what the `cat` does: it
turns `[a, b]` into `[a, b, a, b]`.

Look at it in the tables your function has to return with `head_dim=4`:

```
   cos = [[ 1.0000,  1.0000,  1.0000,  1.0000],     position 0: angle 0, no rotation
          [ 0.5403,  0.9999,  0.5403,  0.9999],     position 1: cos(1.0) and cos(0.01)
          [-0.4161,  0.9998, -0.4161,  0.9998],     position 2: cos(2.0) and cos(0.02)
          [-0.9900,  0.9996, -0.9900,  0.9996]]
                └── repeated ──┘

   sin = [[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.8415,  0.0100,  0.8415,  0.0100],
          [ 0.9093,  0.0200,  0.9093,  0.0200],
          [ 0.1411,  0.0300,  0.1411,  0.0300]]
```

Columns 0 and 2 are identical, and so are 1 and 3. If your function returns something like this,
step 4 is right. There is a dedicated test (`test_rope_duplicates_the_frequencies_by_halves`).

Notice the first row too: at position 0 the cosine is 1 and the sine is 0, that is, **no
rotation**. Which makes sense: the first token is the origin and does not move. There is another
test that checks it.

The two conventions are equivalent up to a permutation of the dimensions, which the network
learns without ever noticing. The halves one won because it makes exercise 3 a single line with
no reordering, and you are about to see that.

---

## Exercise 3: actually rotating (`apply_rope`)

Now for real. And it is one line:

```python
return x * cos + rotate_half(x) * sin
```

with a three-line helper:

```python
def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)
```

### Why that line is the rotation matrix

Rotating a pair `(x1, x2)` by an angle `t` is the usual thing:

```
   x1' = x1·cos(t) - x2·sin(t)
   x2' = x2·cos(t) + x1·sin(t)
```

And now check that the line produces that. With `head_dim=4`,
`rotate_half([x0, x1, x2, x3]) = [-x2, -x3, x0, x1]`. Component by component:

```
   output[0] = x0·cos[0] + (-x2)·sin[0]  =  x0·cos - x2·sin      pair (x0,x2), first component
   output[2] = x2·cos[2] + ( x0)·sin[2]  =  x2·cos + x0·sin      pair (x0,x2), second component
```

And because `cos[0] == cos[2]` thanks to step 4 of the previous exercise, both lines use **the
same angle**. There is the whole reason for the `cat`: without it, `output[2]` would have used the
wrong frequency. The two pairing conventions, the duplication and the `rotate_half` are three
pieces of one mechanism, and that is why exercise 2 has to be understood before this one.

### Check it with numbers

Take the vector `x = [1.0, 0.0, 0.0, 1.0]` and the `head_dim=4` tables from the previous section.
Your function has to return:

```
   position 0:  [ 1.0000,  0.0000,  0.0000,  1.0000]     untouched: angle 0
   position 1:  [ 0.5403, -0.0100,  0.8415,  0.9999]
   position 2:  [-0.4161, -0.0200,  0.9093,  0.9998]

   norm in all three:  1.4142   (that is √2, the original vector's)
```

Follow position 1 by hand and the whole mechanism shows up:

- **pair (x0, x2) = (1, 0)**, angle 1.0 rad → `(cos·1 − sin·0, sin·1 + cos·0)` = `(0.5403,
  0.8415)`, which are outputs 0 and 2. ✓
- **pair (x1, x3) = (0, 1)**, angle 0.01 rad → `(cos·0 − sin·1, sin·0 + cos·1)` = `(−0.0100,
  0.9999)`, which are outputs 1 and 3. ✓

And the norm is still √2 at all three positions, which is the property we talked about: rotating
does not change length.

### The two details that break if you skip them

**The `cos[:seq_len]` slice.** The tables are precomputed up to `max_seq_len` (512 in the final
model) and your sequence is almost never exactly that long. Without slicing, the broadcast either
fails or — worse — happens to succeed with the wrong shapes. There is a test that passes a
sequence shorter than the tables for precisely this reason.

**The `.to(dtype=x.dtype)`.** Under AMP the tables are in fp32 and `x` arrives in fp16. Mixing
them makes PyTorch promote to the wider type, and you end up computing in the precision you did
not want, slower and using more memory. It is the same kind of detail as RMSNorm's `.float()` in
module 07, here in the opposite direction. There is a test that runs in fp16.

**And no `unsqueeze` is needed.** This is the doubt everybody gets: `x` is
`(B, n_heads, T, head_dim)` and `cos` is `(T, head_dim)`. Shouldn't they be aligned? No:
PyTorch's broadcasting aligns **from the right**, matches `(T, head_dim)` against `x`'s last two
dimensions and repeats the rest on its own. And that is the correct thing anyway: position 5's
rotation is the same for every head and every sequence in the batch.

---

## What RoPE does not fix

It is often said that RoPE "extrapolates to longer contexts". That is half true and it is worth
knowing exactly where it stops, because it is one of the most repeated unqualified claims around.

The demo trains three identical models differing only in the positional encoding, all with
context 32, and evaluates them with contexts from 8 to 128:

| context | learned | sinusoidal | RoPE |
|---|---|---|---|
| 8 | 2.1924 | 2.1306 | 2.1168 |
| 16 | 2.1139 | 2.1088 | 2.0665 |
| **32 (trained)** | 2.1296 | 2.0823 | **2.0376** |
| 48 | *cannot* | 2.3490 | 2.1049 |
| 64 | *cannot* | 2.5117 | 2.2527 |
| 96 | *cannot* | 2.6723 | 2.4748 |
| 128 | *cannot* | 2.7601 | 2.6324 |

Three readings:

1. **The learned table has a literal hard ceiling.** It is not that it does badly: there is no
   row to look up and the model cannot process the sequence at all.
2. **Sinusoidal and RoPE do produce an answer** for any length, and RoPE wins in every row.
3. **And here is the honest part: "being able to process" is not the same as "working well".**
   Both degrade a lot. From context 32 to 128, sinusoidal gets 33% worse and RoPE 29%. It is an
   advantage, not a solution.

The reason is in exercise 2's frequency ladder, in that row I told you to keep in mind: the slow
frequencies barely complete a fraction of a turn within the trained range. If the model has only
ever seen angles between 0 and 0.03 in pair 24, the angles showing up at position 2000 are
**unseen territory**, and there is no reason for it to know what to do with them.

That is why a whole family of techniques exists for extending the context *after* training —
position interpolation, NTK-aware scaling, YaRN: because direct extrapolation is not enough. When
you read that "RoPE extrapolates", this is what is behind it.

## Where the debate is

Besides the extrapolation business, which is measured above, there is something deeper: **it is
not clear why relative positional encoding works better than absolute.** There are reasonable
arguments about generalization — what is learned in one part of the sequence carries over to
another — but no result that settles it.

And there is a finding that complicates matters further. Transformers with a causal mask **infer
some positional information on their own**, even with no explicit encoding at all, because the
mask itself breaks the symmetry: the token at position 0 sees one token, the one at position 5
sees six, and from that number of visible neighbours you can work out where you are. There is work
training causal models **with no positional encoding whatsoever** and they do surprisingly well.

So it is not even clear how necessary this whole module is. What is clear is that models come out
better with RoPE than without it, and that is why everybody uses it.

---

**Further reading:** Su et al. 2021, [RoFormer](https://arxiv.org/abs/2104.09864) (RoPE) ·
Press et al. 2021, [ALiBi](https://arxiv.org/abs/2108.12409) (another alternative, which biases
the scores instead of rotating) · Vaswani et al. 2017 (the original sinusoidals).
Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
