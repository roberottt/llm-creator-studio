# 06 — Self-attention: letting each token choose what to pay attention to

## Why this module matters

**If you could only understand one module in the course, it would be this one.**

Everything before it — tokenizing, embeddings, Bengio's MLP — already existed in 2003 and
gave mediocre models. What changed in 2017 and eventually produced ChatGPT is exactly what
you are going to program here, and the core of it fits in four lines of code.

The idea is simple to state: **let each token look at the previous ones and decide which to
pay attention to**. What is hard is believing that this is enough. By the end of the module
you will have seen it working in a model you trained yourself, with a heatmap that literally
shows what each letter is looking at.

And it is worth saying up front what this module is *not*: it is not theory you will apply
later. The three functions you write here are, without a single line changed, the ones the
final model runs. 27.5% of the 8,933,440 parameters you are going to train live inside
exercise 3.

### What you will know by the end

- Why a model can "remember" something it read 300 words earlier
- What Q, K and V are, and **why three things are needed and not one** (with the numeric
  example where using a single one gives the wrong answer)
- Where the shapes `(B, T, d_k)` and `(B, n_heads, T, head_dim)` come from, which is where
  almost everybody gets stuck in exercise 3
- Why we divide by `√d_k`, and what exactly breaks if you remove it (with measured numbers)
- What the causal mask is and why it is **the most expensive bug in the course** if you get
  it wrong
- What the fourth projection, `out_proj`, does — it does not appear in the paper's formula
  and without it multi-head is useless
- What attention **cannot** do, and which module fixes it
- You will have seen an attention heatmap from a model you trained

### What you are going to write

Three exercises, and each fits inside the next. This theory is ordered so that you read them
in this order, and **each one has its own section with the matching numeric example**:

| Exercise | What it does | Where it is explained |
|---|---|---|
| 1. `causal_mask` | The triangular matrix that forbids looking into the future | [§ The causal mask](#exercise-1-the-causal-mask-causal_mask) |
| 2. `single_head_attention` | The whole formula, with one head | [§ Single-head attention](#exercise-2-single-head-attention-single_head_attention) |
| 3. `MultiHeadAttention` | Eight in parallel, which is what the model uses | [§ Multi-head](#exercise-3-eight-heads-in-parallel-multiheadattention) |

```
    causal_mask  ──────────┐
                           ▼
    q, k, v  ────>  single_head_attention  ────>  output, weights
                           │
                           │  the same computation, with one more dimension
                           ▼
                   MultiHeadAttention   ← this is what goes inside the GPT
```

Exercise 1 is one line. Exercise 2 is four, and each one has a trap. Exercise 3 is the same
computation as exercise 2 with an extra dimension, plus the plumbing of splitting the heads
and putting them back together: it is long, but it is not harder, and its section takes it
apart piece by piece.

### What it costs

4 hours, tied with module 03 as the longest. It is the one most worth it.

---

## Which part of the LLM this is

Building an LLM is five distinct jobs, and the course walks them in this order:

```
   0. FOUNDATIONS    what an LLM is, PyTorch, autograd        modules 00-02   ✔ done
   1. TOKENIZER      text  ->  numbers                        module 03       ✔ done
   2. DATA           numbers  ->  a learning task             module 04       ✔ done
   3. MODEL          the architecture that does the predicting modules 05-10  ← YOU ARE HERE
   4. TRAINING       adjust the weights until it gets it right modules 11-13
```

In module 05 the three baselines were dead ends on purpose. This one is not: the real model
starts here. And to place the piece, here is what a Transformer block — the thing you will
assemble in full in module 10 — looks like:

```
    x ──┬──> norm ──> ATTENTION (this module) ──┐
        │                                       ├──> +  ──┬──> norm ──> MLP ──┐
        └───────────────────────────────────────┘         │                   ├──> +
                                                          └───────────────────┘

         module 07: the norm and those two additions (residual connections)
         module 08: the MLP
         module 09: how the model is told which position each token is at
         module 10: stack six of these blocks and assemble the GPT
```

Six blocks like that, each with its own attention. In parameters:

```
   one attention layer:  4 projections of 320x320  =    409,600
   the six layers:                                     2,457,600
   the whole model:                                    8,933,440   -> 27.5%
```

That 27.5% is not the biggest share (module 08's MLP has more), but it is what lets the model
relate things that are far apart in the text. Without it you would have a network that
processes each position on its own, which is exactly Bengio's MLP from the previous module.

---

## The problem it solves

Sentence: *"the cat I saw yesterday was sleeping"*.

To get `sleeping` right you have to know that the subject is `cat`, five words back. Module
05's MLP cannot, and it is worth having fresh in your mind why, because attention is designed
precisely against those two limitations:

- It **concatenated** the previous `k` vectors and fed them into a linear layer. The weight
  of "the position four tokens back" was a fixed number in the matrix, the same for every
  sentence in the corpus. There was no way to say *"in this particular sentence, the one that
  matters is the first"*.
- And the size of that layer **grew with the context length**, so 512 tokens was
  unreachable.

Attention solves both at once: the weights are **computed from the content** in each
sentence, and the parameter count does not depend on the context length at all. The four
320×320 matrices in exercise 3 are the same whether you hand them 10 tokens or 512.

## The idea, with numbers you can follow by hand

Let us do it with 3 words and 2-dimensional vectors. After the embeddings we have:

```
   cat       = [1.0, 0.2]
   yesterday = [0.1, 0.9]
   sleeping  = [0.3, 0.4]
```

`sleeping` wants to know who to look at. It does so with a **dot product**, which measures how
similar two vectors are: multiply component by component and add. The more aligned, the larger
the number.

```
   sleeping · cat       = 0.3×1.0 + 0.4×0.2 = 0.38
   sleeping · yesterday = 0.3×0.1 + 0.4×0.9 = 0.39
   sleeping · sleeping  = 0.3×0.3 + 0.4×0.4 = 0.25
```

Those numbers are called **scores**. They get divided by `√d_k` — here `√2 = 1.414`, and the
scaling section explains why — and turned into weights that sum to 1 with the softmax you
already know from module 05: exponentiate and normalize.

```
   scaled:   [0.269, 0.276, 0.177]
   softmax:  [0.343, 0.345, 0.313]
```

And with those weights the vectors get mixed:

```
   output = 0.343×cat + 0.345×yesterday + 0.313×sleeping
```

That is attention: **a weighted average where the weights are decided by the content
itself.**

Now look closely at the result, because it is bad: the three weights are nearly equal and
`yesterday` beats `cat` by a hair. `sleeping` walks away with a mush of all three words in
which the one that mattered does not stand out. And with the raw embeddings there is nothing
to be done about it, because the similarity between two vectors is what it is. This leads
straight into the next section, which is where the mechanism becomes usable.

## Q, K, V: why three projections and not one

In the example I used the same vector for everything, and that is too rigid. A token needs to
do three different things, and with a single vector all three are tied together:

- **ask** something ("I am looking for a singular subject")
- **advertise itself** to the others ("I am a singular noun")
- **contribute** content if it gets chosen ("the concept of a cat")

They are three different roles, so **three different linear projections** of the same input
vector are learned:

$$Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V$$

**Query** (question), **Key** (label) and **Value** (content). Similarity is computed between
queries and keys; what gets mixed are the values.

### The same example, now with projections

Read the two dimensions of our vectors as "how much of a noun it is" and "how much of a time
reference it is":

```
   cat       = [1.0, 0.2]     very much a noun, hardly any time
   yesterday = [0.1, 0.9]     hardly a noun, very much time
   sleeping  = [0.3, 0.4]     a verb: neither one nor the other
```

What the model learns in $W_Q$ is, literally, "when you are a verb, ask for nouns". A matrix
that does that is:

$$W_Q = \begin{pmatrix} 0 & 0 \\ 2.5 & 0 \end{pmatrix}, \qquad W_K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

With $W_K$ leaving the keys as they were, the query for `sleeping` comes out as
$[0.3, 0.4] W_Q = [1.0,\ 0.0]$: a vector pointing purely along the "noun" direction. And the
scores change completely:

```
                          without projections    with learned Q and K
   sleeping -> cat              0.38                    1.00
   sleeping -> yesterday        0.39                    0.10
   sleeping -> sleeping         0.25                    0.30

   softmax (÷√2):        [0.343, 0.345, 0.313]   [0.468, 0.247, 0.285]
```

From a dead heat in which the wrong word was winning, to `cat` taking nearly twice as much as
anything else. **That is the entire job of $W_Q$ and $W_K$**: they do not change the
information, they change *who matches whom*. And they are not written by hand, they are
learned by gradient descent like any other weight.

The third one, $W_V$, exists to decouple one more thing: **what a token answers to a question
does not have to be the same as what it contributes when it gets chosen**. `cat` can advertise
itself as "singular noun" (its key) and contribute the concept of a feline (its value). With a
single vector, improving one would ruin the other.

## The formula

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

It is exactly what we just did, with every symbol in its place:

| symbol | what it is in the example |
|---|---|
| $QK^\top$ | all the dot products at once: cell $(i,j)$ is how interested $i$ is in token $j$ |
| $\sqrt{d_k}$ | the `1.414` we divided by |
| $M$ | the causal mask, which I left out of the example |
| softmax | the step that turned `[0.269, 0.276, 0.177]` into weights summing to 1 |
| $\cdots V$ | the final mix |

The mental leap you have to make is that $QK^\top$ asks **every question at once**. In the
example I only looked at the `sleeping` row, but the matrix has one row per token: `cat`
asking on its own account, `yesterday` on its own. One matrix multiplication, `T` questions
answered in parallel. That is the underlying reason the Transformer trains fast on a GPU and
the recurrent networks that dominated before 2017 did not: those had to walk the sentence
token by token.

## The shapes: what B, T, S and d_k are

Exercise 2's signature carries four different shapes, and it is worth having them clear before
opening it, because 90% of the module's errors are shape errors:

```
   q     (B, T, d_k)      the questions
   k     (B, S, d_k)      the labels
   v     (B, S, d_v)      the contents
   mask  (T, S)           what is allowed to be looked at
   ───────────────────────────────────────────────
   out   (B, T, d_v)      the output: one row per question
   w     (B, T, S)        the "who looks at whom" matrix
```

- **B** is the batch, the same one from module 04: several sequences processed at once.
  Attention never mixes them; each goes its own way.
- **T** is how many *questions* there are, and **S** how many *things there are to look at*.
  In this course they are always the same, because each token both asks and is available to
  be looked at: that is what the "self" in *self-attention* means. They get different letters
  because the computation does not require them to match — in a translator, the questions come
  from the Spanish text and the keys from the English one — and there is a test that checks
  your function survives that case.
- **d_k** is the size of the query and key vectors. It has to be the same for both, or the dot
  product is not defined. **d_v** may differ, though in practice it never does.

Notice a detail that nearly everything follows from: `d_k` disappears in the product
$QK^\top$ (it is summed over) and `S` disappears in the multiplication by $V$. What comes out
has `T` rows and `d_v` columns: **the output has the same shape as the input**, one vector per
token. That is what lets you stack six of these blocks one after another without the pieces
ceasing to fit.

---

## Exercise 1: the causal mask (`causal_mask`)

**The problem.** During training we hand the model the whole sequence at once and ask it to
predict each token from the previous ones — all 512 predictions of a window in a single pass,
which is what you saw being assembled in module 04. But $QK^\top$ computes *every* pair, so
position 3 can look at position 4, which is literally the answer it is supposed to give.

**The fix.** A lower-triangular boolean matrix saying what may be looked at. For
`seq_len = 4`, with the convention `True = CAN be looked at`:

```
        j=0    j=1    j=2    j=3
 i=0  [ True, False, False, False]     token 0 only sees itself
 i=1  [ True,  True, False, False]     token 1 sees 0 and itself
 i=2  [ True,  True,  True, False]
 i=3  [ True,  True,  True,  True]     the last one sees everything
```

Row `i` is "who token `i` may look at". The diagonal is included: a token may look at itself,
and in fact it does so a lot. That is the whole exercise, one line with
`torch.ones(...).tril()`.

**How it gets used, which is what really matters.** The mask does not erase weights: it puts
$-\infty$ in the forbidden scores **before** the softmax. Since $e^{-\infty} = 0$, those
positions get exactly zero weight. In exercise 2 you write it like this:

```python
scores = scores.masked_fill(~mask, float("-inf"))
```

The `~` flips the boolean: where the mask says `False` (forbidden), put `-inf`.

**And why before and not after.** This is the part that looks like a style detail and is not.
If you let the softmax normalize with the future included and erased those weights afterwards,
the rows would no longer sum to 1: each position would end up scaled by an arbitrary,
different factor, smaller the earlier it is in the sentence. By putting $-\infty$ first, the
softmax normalizes only over what is allowed and **each row still sums to exactly 1**. The
demo checks it:

```
   total weight on the future WITH mask:     0.000000
   total weight on the future WITHOUT mask:  3.901440
   sum of each row with the mask:            1.000000
```

**This is the most expensive bug in the course.** If the mask is wrong, the loss drops
spectacularly, everything looks like it is going beautifully, and the trained model is useless
because at generation time that future does not exist: when you ask it to write, there are no
later tokens to look at and the model is left without the crutch it learned with. That is why
module 05 insists on comparing the loss at step 0 against $\ln(V)$. **If it comes out lower,
look at the mask.**

**A warning you will be grateful for.** We use `True = allowed` because that is the convention
of `F.scaled_dot_product_attention`, which you will use in exercise 3. But PyTorch's
`nn.MultiheadAttention` uses the opposite one: in its boolean `attn_mask`, `True` marks what
must be **forbidden**. That is why the test comparing against it passes `~mask`. It is not a
mistake in the course: it is a real inconsistency inside the library itself, and knowing it
saves you an afternoon.

---

## Exercise 2: single-head attention (`single_head_attention`)

Here you write the formula. It is four lines and each is a chunk of the equation:

```
   1.  scores  = q @ k.transpose(-2, -1) / math.sqrt(d_k)     ->  QKᵀ/√d_k
   2.  scores  = scores.masked_fill(~mask, -inf)              ->  + M
   3.  weights = F.softmax(scores, dim=-1)                    ->  softmax(...)
   4.  out     = weights @ v                                  ->  ... V
```

The `transpose(-2, -1)` is what turns `k` from `(B, S, d_k)` into `(B, d_k, S)` so the matmul
lines up: `(B, T, d_k) @ (B, d_k, S)` gives `(B, T, S)`, the score matrix.

### Check it with numbers

If you hand it the three vectors from the example as `q`, `k` and `v` at once (which is what
self-attention does: all three come from the same place) together with the 3×3 mask, your
function has to return exactly this:

```python
X = torch.tensor([[[1.0, 0.2], [0.1, 0.9], [0.3, 0.4]]])
out, w = single_head_attention(X, X, X, causal_mask(3))
```

```
   weights = [[1.0000, 0.0000, 0.0000]      token 0 can only look at itself,
              [0.4057, 0.5943, 0.0000]      so it takes all the weight
              [0.3832, 0.2672, 0.3496]]

   output  = [[1.0000, 0.2000]              and that is why its output is itself, untouched
              [0.4651, 0.6160]
              [0.6896, 0.4220]]
```

Two things to look at there. The first row of weights is `[1, 0, 0]` **always**, whatever the
embeddings are: the first token has nobody to look at, so attention contributes absolutely
nothing to it. The second is that token 0's output is its own vector untouched, `[1.0, 0.2]`,
which is a good way to verify at a glance that your weighted average is put together right.

### The three traps

None of the three raises an error where you make it.

**`transpose(-2, -1)` with negative indices.** They count from the end, so they work the same
with `(B, T, d)` as with `(B, heads, T, d)`. If you write `transpose(1, 2)`, this exercise
passes and **exercise 3 breaks** with a shape error that is hard to connect back to the cause,
because there dimension 1 is the heads.

**`dim=-1` in the softmax.** You are normalizing over *who is being looked at*, so that each
row sums to 1. With `dim=-2` you would normalize over *who is looking*, which means nothing:
it would be splitting the attention a token *receives* among those looking at it, instead of
splitting the attention a token *gives* among those it looks at. And it raises no error: the
shapes are identical, the model trains, and it learns worse. There is a test that checks every
row sums to 1.

**The mask goes before the softmax**, for the reason in the previous section.

## The scaling by √d_k: what exactly breaks if you remove it

This divisor looks like an arbitrary detail and it is one of the few Transformer decisions
with a clean mathematical justification.

**Where the number comes from.** The dot product of two vectors of dimension $d_k$ with
independent components of mean 0 and variance 1 is a sum of $d_k$ independent terms, so it has
**variance $d_k$** and standard deviation $\sqrt{d_k}$. With $d_k = 40$ (our case) the scores
move in a range of about $\pm 6$; with $d_k = 512$, about $\pm 22$. Dividing by $\sqrt{d_k}$
brings them back to variance 1 regardless of the dimension, which is exactly what you want:
that the temperature of the softmax should not depend on an architecture decision.

**Why it matters.** Softmax is exponential. If one score stands 20 units above the rest,
$e^{20}$ against $e^{0}$ is 485 million to one: the softmax returns practically
`[0, 0, ..., 1, ..., 0]` and attention stops being a weighted average and becomes a hard
selection of a single token.

The demo measures it with the **entropy** of the attention distribution, which is the same as
module 05's perplexity but without exponentiating: high means "it spreads across many", near
zero means "it fixates on one". With 16 positions, the maximum possible is
$\ln(16) = 2.773$:

| d_k | entropy WITH scaling | entropy WITHOUT scaling | max weight without scaling |
|---|---|---|---|
| 8 | 2.502 | 1.629 | 0.8004 |
| 32 | 2.421 | 0.692 | 0.9984 |
| 128 | 2.318 | 0.085 | 1.0000 |
| 512 | 2.346 | 0.216 | 1.0000 |
| 2048 | 2.279 | 0.007 | 1.0000 |

With scaling the entropy stays high no matter what happens to `d_k`. Without it, from
`d_k = 128` on the maximum weight is 1.0000 to four decimals: the token fixates on one and
ignores everything else.

**And the real problem is not that one, it is the gradient.** The derivative of the softmax is
$p(1-p)$; with $p$ pinned to 0 or 1, the derivative is practically zero. The weights that
produce those scores stop receiving any signal and the layer stops learning. It is not that
the model attends badly: it is that **it cannot correct itself**, because the mechanism that
would tell it how has been switched off. Without the $\sqrt{d_k}$, a large Transformer simply
does not train.

---

## Exercise 3: eight heads in parallel (`MultiHeadAttention`)

**The problem.** A single attention has to resolve every relationship in the sentence with one
pattern. But in *"the cat I saw yesterday was sleeping"* there are several relationships at
once: the subject-verb agreement, the `I saw` clause that refers back to `cat`, the `yesterday`
that places the action. A single distribution of weights per token does not stretch that far:
if it gives weight to `cat` it is not giving it to `yesterday`.

**The fix.** Run several attentions in parallel, each with its own $W_Q, W_K, W_V$, and
concatenate the results. With $d_{\text{model}} = 320$ and 8 heads, each works in $40$
dimensions ($320/8$).

And here is the neat part: **it does not cost more**. Instead of one 320-dimensional attention
you do eight of 40, and the total parameter count is identical. What you gain is that eight
different attention distributions can coexist.

### The heads specialize on their own

In the model the demo trains — one layer, four heads, 400 steps on Shakespeare — you can
already see it, by measuring the average distance each head looks back:

```
   head 0: average distance 2.10 positions
   head 1: average distance 3.39 positions
   head 2: average distance 2.93 positions
   head 3: average distance 4.05 positions
```

Four different context ranges, and **nobody told them to**: it comes out of initializing each
head at random and letting the gradient do the rest. In large models this goes much further:
heads have been identified that always look at the previous token, heads that match opening and
closing quotes, and the so-called *induction heads*, which detect the pattern "…A B … A" and
predict B. None of them is programmed.

### The dance of the shapes, which is the hard part

This is the part of the module where people really get stuck, and it is pure dimension
bookkeeping. The full route, with `B=2`, `T=5` and the final model's configuration:

```
   x                 (2, 5, 320)      input
     │  q_proj / k_proj / v_proj      three Linear(320, 320)
     ▼
   q, k, v           (2, 5, 320)      not split yet
     │  _split_heads
     ▼
   q, k, v      (2, 8, 5, 40)         8 heads, each with 40 dimensions
     │  the computation from exercise 2, identical
     ▼
   scores       (2, 8, 5, 5)          one attention matrix PER HEAD
   out          (2, 8, 5, 40)
     │  _merge_heads
     ▼
   out               (2, 5, 320)      the heads glued back together
     │  out_proj                      one more Linear(320, 320)
     ▼
   output            (2, 5, 320)      same shape as the input
```

The input and the output having the same shape is no accident: it is what lets you stack
blocks and what makes module 07's residual connections possible.

**`_split_heads` specifically.** It is done in two steps, and the order matters:

```python
x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
```

The `view` splits each token's vector into chunks; the `transpose` moves the heads in front of
the positions. With a tiny example (`d_model=4`, 2 heads of 2, two tokens):

```
   input:  token 0 = [1, 2, 3, 4]        token 1 = [5, 6, 7, 8]

   RIGHT, view(B,T,2,2) and then transpose(1,2):
      head 0 = [[1, 2],      <- the first half of EACH token
                [5, 6]]
      head 1 = [[3, 4],      <- the second half of EACH token
                [7, 8]]

   WRONG, view(B,2,T,2) directly:
      head 0 = [[1, 2],      <- the whole of token 0
                [3, 4]]
      head 1 = [[5, 6],      <- the whole of token 1
                [7, 8]]
```

The wrong version has the right shape and scrambled data: it has handed out *positions* among
the heads instead of *dimensions*. It raises no error. There is a test that catches it by
checking that the heads do not produce identical patterns.

**`_merge_heads` undoes exactly that**, and it needs a `.contiguous()` in the middle:

```python
x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
```

`transpose` does not move data, it only changes how it is walked in memory (the *strides*), and
`view` requires contiguous memory. Without the `.contiguous()`, PyTorch raises an error that
talks about strides and does not clearly say what to do.

### The fourth projection: `out_proj`

The paper's formula does not mention it and without it multi-head is useless, so it is worth
understanding what it does.

After `_merge_heads` you have a vector of 320 numbers that are eight chunks of 40 glued one
after another. Each chunk is what one head produced **on its own, knowing nothing about the
others**. If that were the layer's output directly, the eight heads would be eight sealed
channels: head 3 could never combine what it found with what head 7 found.

`out_proj` is an `nn.Linear(320, 320)` whose job is precisely to **mix the heads' results**
together and decide how much each one weighs. It is what turns eight independent answers into
one. It is also, incidentally, what makes the layer's parameter count four matrices and not
three:

```
   q_proj    320 × 320 = 102,400
   k_proj    320 × 320 = 102,400
   v_proj    320 × 320 = 102,400
   out_proj  320 × 320 = 102,400
   ─────────────────────────────
                        409,600     per layer, without biases
```

Without biases because `ModelConfig` uses `bias=False`: modern LLMs drop them from the
projections because they contribute little and complicate weight decay. Explained in module 09.

### The three things in the signature you have not seen yet

Exercise 3 has three arguments that come from modules you have not done yet. You do not need to
understand them deeply to write it, but you do need to know what they are so you do not sit
there thinking you missed something:

- **`dropout`, in two places.** Dropout is switching off a random fraction of the numbers
  during training, so the model does not depend too much on any one of them; it shows up in
  module 11. Here it goes in two distinct spots: `attn_dropout` on the attention weights (it
  switches off some connections) and `resid_dropout` on the final output. In this course's
  model it is 0.
- **`cos` and `sin`: RoPE.** Module 09's way of encoding position. They go *after* splitting
  into heads, because the rotation depends on `head_dim` and not on `d_model`, and **only to q
  and k, never to v**: what should depend on position are the scores, not the content being
  transported. This module's tests pass `cos=None, sin=None`.
- **`use_sdpa`.** `F.scaled_dot_product_attention` is PyTorch's fused implementation, which
  does the four steps in a single kernel without materializing the full `(T, T)` matrix. It is
  the one real training uses. You write both branches: the explicit one, which is the one that
  teaches and the one that returns the weights for the heatmap, and the call to SDPA. A test
  checks they give the same result. Careful with `dropout_p`: you have to pass
  `self.dropout if self.training else 0.0`, because SDPA does not check the mode on its own and
  would apply dropout during evaluation too, which would make your samples noisy and
  irreproducible.

**And why one big projection instead of eight small ones.** `nn.Linear(320, 320)` followed by a
`view` is mathematically identical to eight `nn.Linear(320, 40)` whose results get concatenated.
But it is *one* big matmul instead of eight small ones, and as you measured in module 01, big
matrices make far better use of the GPU. It is the same reasoning as `nn.Embedding` versus
`nn.Linear` in module 05: same maths, different cost.

---

## What attention does not do

Just as important as knowing what it solves is knowing what it does not, because the next three
modules are largely the patches for this list.

**It knows nothing about order.** This is surprising and easy to check. Attention is a weighted
sum, and a sum does not distinguish the order of its terms: to the token asking, the previous
ones are a **bag**, not a sequence. Take the three-word example, swap `cat` and `yesterday`, and
look at the output of the last position:

```
   original order:  [0.4708, 0.5040]
   swapped order:   [0.4708, 0.5040]     identical
```

The causal mask imposes *who may look at whom*, but it does not tell the model how far away each
one is. Without fixing that, *"the cat bit the dog"* and *"the dog bit the cat"* would be
indistinguishable to the layer. That is module 09.

**It does no per-position processing at all.** Everything attention does is move information
between tokens; combining and transforming that information inside each token is the job of
module 08's MLP. In fact, without it, stacking attention layers would not help much: they are
linear operations apart from the softmax. That is why the two pieces alternate, and not out of
habit.

**It costs the square of the context length.** The score matrix is `(T, T)`: double the context
and you quadruple that cost. With our 512-token context and 8 heads, in fp32:

```
   T =  512:   512×512 × 8 heads × 4 bytes  =    8.4 MB per sequence
   T = 1024:  1024×1024 × 8 × 4             =   33.5 MB per sequence
```

Multiply that by the batch size and you will see why context length is the memory-costliest
decision in a Transformer. It is also the reason SDPA exists: it avoids materializing that whole
matrix.

---

## Where the debate is

We know *what* attention computes. Why it works so well is another matter entirely.

The intuitive explanation — "each token retrieves relevant information" — is the one I have just
given you, it is a reasonable story and it is not proven. There are results that complicate it:
models with **fixed, random** attention patterns work surprisingly well on some tasks, which
suggests part of the credit belongs to the general architecture (residuals, normalization,
depth) and not only to the attention mechanism. Worth keeping in mind when you read very
self-assured explanations, this one included.

The most serious line of work in this direction is mechanistic interpretability, which tries to
read the circuits that form inside. It has managed to explain specific components — *induction
heads* are the success story — but it is a long way from accounting for a whole model.

And there is a structural limitation that remains unsolved: the cost grows with the **square** of
the context length. Dozens of subquadratic alternatives have been proposed (Linformer, Performer,
Mamba and family). None has displaced standard attention in general-purpose models, and it is not
clear whether that is because full attention is necessary or because it has a twenty-year head
start in kernel optimization.

---

**Further reading:** Vaswani et al. 2017,
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) · Elhage et al. 2021,
[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
(the *induction heads*). Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
