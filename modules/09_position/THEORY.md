# 09 — Positional information and RoPE

## Why this module matters

**Because attention does not know which word comes first.**

It sounds like a detail and it is a fatal flaw. Look again at module 06's formula: it is a
weighted sum, and a sum has no order. To the attention mechanism, *"the dog bites the man"*
and *"the man bites the dog"* produce exactly the same thing.

This is where it gets fixed, and it gets fixed with a rather beautiful idea: instead of
adding a label to the vector saying "I am position 7", a **rotation** is applied whose angle
depends on the position. That makes the model learn relative distances — "the token two
positions back" — instead of absolute positions.

It is the technique used by Llama, Mistral and practically everything modern.

### What you will know by the end

- Why without this your model cannot tell word order apart
- Three ways of solving it, in historical order, and what fails in each
- What RoPE is and **the mathematical property that justifies it**, checked with numbers
- What really happens when you ask a model for a longer context than it trained on

### What it costs

2.5 hours. The third exercise is one line of code, but only after understanding the second.

---

## The problem: attention does not know what comes first

Look again at module 06's attention formula. It is a weighted sum of the values, and the
weights come from dot products between queries and keys.

**Position** appears nowhere.

The consequence is brutal and worth seeing: to the attention mechanism, *"the dog bites the
man"* and *"the man bites the dog"* produce exactly the same set of output vectors, just
reordered. If you shuffle the input tokens, the output shuffles the same way and nothing
else changes. That property is called **permutation equivariance**, and here it is a fatal
flaw: word order is half the meaning.

Position has to be injected somehow. Let us look at three ways, in historical order.

## Option 1: learn a table

The simplest. A table with one row per position, trained like any other parameter, and
**added** to the token's embedding:

```
input = token_embedding[id] + position_embedding[i]
```

It is what GPT-2 does. It works well and there is no mystery to it.

It has two drawbacks. The first is a **hard ceiling**: if you trained with 1024 positions,
for position 1025 there is no row to look up. The model cannot process it at all, not even
badly. The second is that the model learns **absolute** positions — "this is token number 7"
— when what usually matters is the relationship: "this is two words before the verb".

## Option 2: sines and cosines

The 2017 paper proposed a fixed, parameter-free table made of sines and cosines at different
frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

The intuition is that of a **binary counter**. Look at how you count in binary:

```
0000    the rightmost bit changes at every step
0001    the next one, every two
0010    the next one, every four
0011    ...
```

Each bit oscillates at a different rate, and the combination of all of them identifies a
number uniquely. The sinusoidals do the same but with continuous waves: the first pairs of
dimensions oscillate fast and distinguish neighbouring positions; the last ones oscillate
extremely slowly and distinguish the start of the sequence from the end.

An advantage over the learned table: it is defined for any position, there is no ceiling. In
practice the extrapolation does not work very well either, but at least it exists.

## Option 3: RoPE — rotate instead of adding

Here is the idea our model uses, and Llama, and almost everything modern.

**Instead of adding something to the vector, a rotation is applied whose angle depends on
the position.**

Take a 2-dimensional vector and rotate it by an angle $\theta$. The good old rotation
matrix:

$$\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} =
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

RoPE splits each head's vector into pairs and rotates each pair by an angle proportional to
the position. As with the sinusoidals, each pair has its own rotation speed: the first ones
turn fast, the last ones extremely slowly.

### Why this is such a good idea

And here comes the property that justifies it all. Rotations have a peculiarity: **the dot
product of two rotated vectors depends only on the difference of angles.**

$$\langle R(m)\,q,\; R(n)\,k \rangle = \langle q,\; R(n-m)\,k \rangle$$

Translated into what matters: the attention score between the token at position 5 and the
one at position 3 is **identical** to the one there would be between 105 and 103. What the
model learns is not "token number 3" but **"the token two positions back"**.

You can check it yourself: the demo computes $\langle R(2)q, R(5)k \rangle$ and
$\langle R(4)q, R(7)k \rangle$ and they come out the same to the last decimal.

And there is a second benefit: rotating **does not change the vector's length**. Adding a
positional embedding does alter the magnitude, and that interferes with attention's dot
products. Rotating only changes direction.

### Two implementation details

**It is applied only to Q and K, never to V.** What has to depend on position is the
attention *scores*, not the content being transported. And since position is already encoded
in the scores, injecting it into the values too would be redundant and harmful.

**It is applied inside each head**, over `head_dim` dimensions (40 in our case, that is, 20
pairs), not over `d_model`'s 320.

On how to pair the dimensions there are two conventions. The original paper pairs
consecutive ones: $(x_0, x_1), (x_2, x_3)\ldots$. Llama and HuggingFace pair by halves:
$(x_0, x_{d/2}), (x_1, x_{d/2+1})\ldots$. **They are equivalent up to a permutation of the
dimensions**, which the network learns without noticing, and the halves one is implemented
with much cleaner vector operations. We use that one.

## Where the debate is

It is often said that RoPE "extrapolates to longer contexts". That is half true and it is
worth knowing where it ends.

RoPE does have the relative property, yes, but a model trained with a context of 512 and
evaluated at 4096 **degrades considerably**. The reason is that the slow frequencies barely
complete a fraction of a turn within the trained range, so large angles are literally unseen
territory. There is a whole family of techniques for extending the context after training —
position interpolation, NTK-aware scaling, YaRN — that exist precisely because direct
extrapolation is not enough.

More fundamentally: it is not clear *why* relative positional encoding works better than
absolute. There are reasonable arguments about generalization, and there is evidence that
causally masked transformers **infer some positional information on their own** even with no
explicit encoding, because the mask itself breaks the symmetry. There is work training
causal models with no positional encoding at all and they work surprisingly well. So it is
not even clear how necessary all this is.

---

**Further reading:** Su et al. 2021, [RoFormer](https://arxiv.org/abs/2104.09864) (RoPE) ·
Press et al. 2021, [ALiBi](https://arxiv.org/abs/2108.12409) (another alternative, which
biases the scores instead of rotating) · Vaswani et al. 2017 (the original sinusoidals).
Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
