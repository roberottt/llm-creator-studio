# 10 — The full GPT: putting it together and auditing it

## Why this module matters

**Because this is where everything comes together and produces an exact number.**

You have attention, normalization, the FFN and RoPE, each one tested on its own. This module
assembles them into the model you are going to train, and it ends with a check that either adds
up or does not: **8,933,440 parameters**. Not one more.

That number is not decoration. That the formula you derive by hand matches the model's real
count means you have understood where every weight is and why. If it does not add up, something
in your architecture is not what you think it is, and finding that out now is far cheaper than
discovering it halfway through a training run.

### What you will know by the end

- How a complete Transformer is assembled, from token ids to logits, with the shape at every step
- The three design decisions almost every tutorial skips — weight tying, depth-scaled
  initialization and the final normalization — which are what make the model train
- What a PyTorch *buffer* is and why RoPE lives in one
- The three checks you audit the model with before spending a single euro of GPU
- Where the memory actually goes, which is not where you would think

### What you are going to write

Four exercises:

| Exercise | What it does |
|---|---|
| 1. `expected_param_count` | The formula for how many parameters it will have |
| 2. `count_parameters` | Counting them for real, broken down |
| 3. `TransformerBlock` | One block: attention + FFN with their residuals |
| 4. `GPT` | The whole model |

They go in that order and this theory follows it. The first two are about counting, and you can
do them without having assembled anything: their tests use the reference model. In fact you
already have every number — module 06 gave you attention's 409,600, module 07 the norms' 4,160,
module 08 the FFN's 860,160 and module 09 told you RoPE costs not one parameter — so counting is
the payoff of the previous four modules more than a new exercise.

One thing is worth doing differently: **exercise 1, with paper, before typing**. If you go
straight to the code you will end up trying numbers until it adds up.

Exercise 3 is **two lines**. Exercise 4 is the longest in the course, and not because it is hard:
there are just many pieces and they have to go in order.

### What it costs

3 hours. It closes Part II: when you finish it you have the model assembled and audited, ready to
train in module 11.

---

## The whole model, at a glance

The previous four modules have been producing loose pieces. They all fit together here:

```
   module 06    MultiHeadAttention, causal_mask
   module 07    RMSNorm, the pre-norm + residual idea
   module 08    SwiGLU
   module 09    rope_frequencies, apply_rope
```

And some reassurance before starting: **you do not need all four finished.** At the top of
`exercises.py` the pieces are brought in with `resolve(...)`, which is the course's bridge: if
your `SwiGLU` still raises `NotImplementedError`, it uses the reference one and warns you on the
console. You can assemble the GPT today and go back to finish module 08 tomorrow.

Here is how it fits together, with the shape of the tensors at every point (`B` sequences of `T`
tokens, `d_model=320`, `vocab=4096`):

```
   idx                       (B, T)          integers: the token ids from module 04
     │
     │  token_embedding      nn.Embedding(4096, 320)
     ▼
   x                         (B, T, 320)     each id, turned into its vector
     │
     │  drop                 dropout (0 in this course)
     ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  block 1         x = x + attn(attn_norm(x), cos, sin, mask) │  ← exercise 3
   │                  x = x + ffn(ffn_norm(x))                   │
   │  ...             × 6                                        │
   └─────────────────────────────────────────────────────────────┘
     │
   x                         (B, T, 320)     same shape it came in, six times richer
     │
     │  norm_f               the final normalization
     ▼
   x                         (B, T, 320)
     │
     │  lm_head              nn.Linear(320, 4096, bias=False), tied to the embeddings
     ▼
   logits                    (B, T, 4096)    one score per vocabulary token,
                                             at EVERY position
```

Four things about that diagram deserve a moment before moving on, because the first two decide
the parameter count in the next section:

**The first and the last layer are the same matrix.** The embedding table turns an id into a
vector: it is $4096 \times 320$. The `lm_head` turns a vector into scores over the vocabulary: it
is $320 \times 4096$. **They are the same matrix, transposed**, so only one is kept: that is
*weight tying*, and it saves 1,310,720 parameters, 15% of the model. How it is written is in
exercise 4; for counting, it is enough to know that the `lm_head` costs nothing.

**There is no positional embedding at the start.** With RoPE, position is injected inside the
attention, by rotating Q and K (module 09). That is why the first layer is only the token table,
and why `cos` and `sin` travel as arguments all the way down: the GPT's `forward` passes them to
each block, and each block to its attention. And their tables are **not parameters**: nobody
trains them, so they do not count either.

**The shape does not change anywhere in the body of the model.** `(B, T, 320)` goes into block 1
and `(B, T, 320)` comes out of block 6. That is the property that lets you stack six, or sixty:
each block is a *refinement* of the representation, not a transformation into another space.

**The logits are `(B, T, 4096)`: one prediction per position**, not one per sequence. `T`
predictions in a single pass, which is module 04's idea. With `B=48` and `T=512` that is a hundred
million numbers in one tensor, and that will have consequences we will get to at the end.

---

## Exercises 1 and 2: the exact count

Now that you know what is inside, count it. And do it **twice, by independent routes**, which is
what these two exercises are about:

- **`expected_param_count(cfg)`** computes it with a formula, from the config, **without building
  the model**. It is for designing: you change `d_model` in the YAML and know instantly whether it
  fits in the GPU.
- **`count_parameters(model)`** counts it by walking the already-built model, broken down by
  component.

If the two do not give the same number, **either your formula or your model is lying**, and you
have to work out which. That is the whole point: it is a cross-audit, not an arithmetic exercise.

### The table, so you can derive it first

With paper. Then compare:

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

Three things worth noticing:

- **Attention has no biases**: four clean $d \times d$ matrices, because the config uses
  `bias=False`. Modern LLMs have been dropping them; they contribute little and complicate module
  11's weight decay.
- **RMSNorm only has a scale** ($d$ parameters), not scale and bias ($2d$) like LayerNorm. Hence
  the $2d$ per block: two norms of $d$ each.
- **RoPE contributes not one parameter.** If your count includes it, it is wrong.

Two mistakes you will recognize by the number they give:

```
   you get 10,244,160     ->  you forgot the weight tying (the difference is exact)
   you get slightly more  ->  you are counting biases that do not exist
```

### The breakdown from exercise 2

| component | parameters | % |
|---|---|---|
| embeddings | 1,310,720 | 14.7% |
| attention | 2,457,600 | 27.5% |
| FFN | 5,160,960 | **57.8%** |
| norms | 4,160 | 0.05% |
| lm_head | 0 | 0% (tied) |
| **TOTAL** | **8,933,440** | 100% |

**Non-embedding parameters: 7,622,720.** That is the number module 12 uses for the scaling laws,
because embeddings scale differently from the rest: they grow with the vocabulary, not with the
depth. That is why the exercise returns it separately.

Two implementation warnings:

**The `set` of `id()`.** This is where weight tying bites. `named_parameters()` deduplicates by
identity by default, so the **total** comes out right on its own; but if you add up each tensor
wherever you find it, you will count the shared matrix twice and be over by exactly 1,310,720. And
watch out for the Python trap: `if param in seen` **does not work**, because `in` uses `==`, which
on tensors is element-wise and blows up with *"Boolean value of Tensor is ambiguous"*. Compare by
`id()`.

**The order of the `if/elif`.** You classify by substrings of the name, and
`blocks.0.attn_norm.weight` contains both `attn` and `norm`: you want it counted as a norm. Order
from most specific to most general. And before writing anything:

```python
print([n for n, _ in model.named_parameters()])
```

In thirty seconds you see how the whole model is assembled and which substrings to look for. Do
not guess.

---

## Exercise 3: the block (`TransformerBlock`)

The entire `forward` is two lines:

```python
x = x + self.attn(self.attn_norm(x), cos=cos, sin=sin, mask=mask)
x = x + self.ffn(self.ffn_norm(x))
return x
```

That is the Transformer block, the unit that repeats six times and that everything you have read
about these models is made of.

Each line is a **pre-norm + residual** from module 07: normalize, compute something, and **add**
the result to what was already there. The `x` to the left of the `+` is the residual stream, the
highway that runs through the model from top to bottom without anything interrupting it. And the
division of labour is the one you know: attention **moves** information between tokens, the FFN
**processes** it token by token. They alternate, and that alternation is the whole Transformer.

The two residuals are independent on purpose: each sub-block can contribute a little or a lot to
the stream without constraining the other.

In `__init__` you create four things: two norms and two sub-blocks. The norms come from
`make_norm(cfg)` and the FFN from `make_ffn(cfg)`, two ready-made helpers that look at the config
to decide whether it is RMSNorm or LayerNorm, SwiGLU or the classic MLP. Do not reimplement them.

### The four places where people fail

All four are silent: the model builds, trains and gives plausible numbers.

**Normalizing the stream instead of the branch.** It is `x + attn(norm(x))`, not
`norm(x + attn(x))`. The second is post-norm and breaks module 07's property that gets the
gradient cleanly down to layer 1.

**Reusing the same norm for both sub-blocks.** `self.ffn_norm = self.attn_norm` compiles perfectly
and is wrong: they are two distinct objects with their own weights, and that is why the count says
`2 × d_model` per block and not `d_model`.

**Forgetting to pass `cos`, `sin` or `mask` to the attention.** Without `cos`/`sin` the model
loses all positional information and still trains, badly. Without `mask`, every token sees the
future and the loss drops suspiciously nicely.

**Passing `cos`/`sin`/`mask` to the FFN.** It does not accept them, and it does not need them: the
FFN does not look at other tokens, so there is nothing to mask and no position to inject.

---

## Exercise 4: the whole model (`GPT`)

The longest exercise in the course. Let us go part by part, in the same order the code is written,
with each design decision explained where it has to be written.

### `__init__`, part 1: the submodules

The embedding table, the dropout, the six blocks in an `nn.ModuleList`, the final norm and the
output layer. No mystery. The names matter twice over: the test copies weights by name and
exercise 2 classifies parameters by name.

### `__init__`, part 2: weight tying

```python
if cfg.tie_embeddings:
    self.lm_head.weight = self.token_embedding.weight
```

This is how you write the weight tying that the 1,310,720-parameter saving came from. One line,
and it is the first of the module's three decisions.

What matters is that this line **copies nothing**: it makes both modules point at the same tensor
in memory. You check it with `is`, and there is a test that does
(`test_the_tied_weights_are_the_same_tensor`).

And besides saving, it **usually improves quality**: each weight receives gradient by two
different routes — once as an input embedding, once as an output projection — so it trains with
twice the signal. The conceptual justification is that a token ought to be "close" in embedding
space to those it can be confused with, and that closeness is useful both for reading and for
writing.

### `__init__`, part 3: RoPE's tables, and what a buffer is

```python
cos, sin = rope_frequencies(cfg.head_dim, cfg.context_length, cfg.rope_theta)
self.register_buffer("rope_cos", cos, persistent=False)
self.register_buffer("rope_sin", sin, persistent=False)
```

An `nn.Module` holds two classes of tensor. **Parameters** (`nn.Parameter`, module 07's) are the
ones that get trained: they show up in `model.parameters()`, the optimizer updates them and they
receive gradient. **Buffers** are tensors that travel with the model but **are not trained**: they
move with `model.to(device)`, they appear in the `state_dict`, and the optimizer never sees them.

RoPE's tables are the textbook case. They are computed with a closed formula, they always have the
same value and nobody adjusts them. If you stored them as a parameter, the optimizer would try to
train them and you would break module 09's relative property.

The `persistent=False` adds one more thing: **they are not saved in the checkpoint either**. Since
they recompute themselves when the model is built, putting them in the weights file would waste
space. There is a test that checks it
(`test_rope_is_stored_as_a_non_persistent_buffer`).

### `__init__`, part 4: depth-scaled initialization

The module's second decision, and the detail most people skip.

```python
self.apply(self._init_weights)                    # 1. everything with std=0.02

scale = 0.02 / math.sqrt(2 * cfg.n_layers)        # 2. and then, OVERWRITING:
for name, param in self.named_parameters():
    if name.endswith(("out_proj.weight", "down_proj.weight", "fc_out.weight")):
        nn.init.normal_(param, mean=0.0, std=scale)
```

Two passes: first everything gets initialized the same, then the subset that needs something else
gets overwritten. That is simpler than trying to get it right in one pass.

**Why that subset.** Think about the residual stream: each sub-block **adds** its contribution to
it. If the contributions were independent and all of variance $\sigma^2$, the variance of the sum
would grow linearly with the number of summands. With 6 layers and 2 sub-blocks each that is 12
contributions, so the output would have 12 times the variance of the input.

GPT-2's solution, and ours: reduce the standard deviation **only in the projections that write
into the residual stream**, which are attention's `out_proj` (module 06) and the FFN's `down_proj`
(module 08):

$$\sigma = \frac{0.02}{\sqrt{2 \cdot n_{\text{layers}}}}$$

The 2 is because each block writes twice. With 6 layers: $0.02/\sqrt{12} = 0.0058$. You can verify
it on your own model:

```
   std of q_proj.weight    ≈ 0.0200      <- the rest of the model
   std of out_proj.weight  ≈ 0.0058      <- the ones writing into the stream
   ratio                   ≈ 3.47        <- which is √12
```

**And what happens if you do not.** Measuring the average norm of each token's vector on the way
out of each block, on the real model freshly built, with and without scaling those two
projections:

| | with scaling | without scaling |
|---|---|---|
| after the embeddings | 0.356 | 0.356 |
| after block 2 | 0.565 | 1.844 |
| after block 4 | 0.888 | 3.516 |
| **after block 6** | **1.194** (×3.4) | **4.782** (×13.4) |

Without scaling, the stream leaves the model thirteen times bigger than it went in, and that is
with six layers. Imagine with sixty.

A note of honesty about the argument above: it predicts growth of $\sqrt{12} \approx 3.46$ in the
unscaled case, and what is measured is 13.4. The contributions are **not independent** of the
stream — each block reads from it, so its output already scales with what is inside — and the
growth ends up multiplicative rather than additive. The argument explains the *direction* of the
problem well; it does not nail the magnitude.

**And the 0.02 is not arbitrary either.** It is what makes the loss at step 0 equal $\ln(V)$: with
nearly identical logits, the softmax comes out nearly uniform. With PyTorch's standard normal
($\sigma = 1$, `nn.Embedding`'s default) the model would start with strong random opinions and the
loss would come out above it — exactly what you saw measured in module 05, where the neural bigram
started half a nat above the floor.

### The `forward`

```python
B, T = idx.shape
if T > self.cfg.context_length:                   # 1. validate
    raise ValueError(...)

x = self.token_embedding(idx)                     # 2. ids -> vectors
x = self.drop(x)                                  # 3.
cos, sin = self.rope_cos, self.rope_sin           # 4. the tables from part 3
mask = causal_mask(T, device=idx.device)          # 5. ONCE, outside the loop
for block in self.blocks:                         # 6. the six blocks
    x = block(x, cos=cos, sin=sin, mask=mask)
x = self.norm_f(x)                                # 7. the final norm
logits = self.lm_head(x)                          # 8. -> (B, T, 4096)

loss = None                                       # 9. the loss, if there are targets
if targets is not None:
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )
return logits, loss
```

It is the diagram from the start, line by line. Four things to explain:

**Step 5, the mask, goes outside the loop.** It is exactly the same for all six layers, so
computing it inside works and wastes work on every one of the final training run's 10,172 steps.

**Step 7 is the module's third decision**, `norm_f`. In pre-norm the residual stream is never
normalized along the way: you have just seen in the table that it arrives at the end at ×3.4 scale
even with the initialization done right. That is why there is a normalization right before the
projection to logits. It is not optional: without it the logits come out at a scale that depends
on the depth, the loss at step 0 stops being $\ln(V)$ and training is more fragile. It is the
`norm_f` announced at the end of module 07, and now you know where it comes from.

**The `reshape(-1, vocab)` in step 9** is module 05's pattern for the third time:
`F.cross_entropy` wants `(N, V)` and `(N,)`, so batch and time get flattened into one dimension.

**The `ignore_index=-100`** does nothing right now, because no target equals −100. You will need it
in module 16: in instruction fine-tuning the prompt is marked with −100 so the model does not learn
to predict it, only to predict the answer. Leave it in and save yourself coming back.

And notice it returns `(logits, loss)` with `loss=None` if there are no targets, just like module
05's `NeuralBigram`. The reason is the same: when generating text (module 14) there is no correct
answer to be had, and you want the logits to sample from.

---

## The three checks

When you finish you have a nine-million-parameter model that has not trained a single step. These
are the three things you can verify **before** spending an hour of GPU, and the demo does all
three.

**1. The number.** Formula, count and target all have to give 8,933,440. That is the audit from
exercises 1 and 2.

**2. The loss at step 0.**

```
   loss of the untrained model : 8.3747
   ln(4096)                    : 8.3178
   deviation                   : +0.0569
```

Module 05's bug detector, applied to the real model. Notice it does **not come out exact, but a
hair above**, and that is correct: initializing with $\sigma = 0.02$ produces nearly identical
logits, not identical ones, and with a finite sample the average does not land exactly on the
theoretical mean either. A deviation of hundredths is normal; one of several nats is not. What
matters is the direction:

```
   ≈ ln(V)         correct, the model starts with no opinions
   noticeably MORE  the init is too aggressive
   LOWER            information leak: check the mask, then how the batch is built
```

That "then how the batch is built" is not rhetorical. Targets are **shifted by one token**
(`x = seq[:, :-1]`, `y = seq[:, 1:]`). If you passed `model(idx, idx)`, at position `t` the model
would see the token it has to predict and the loss would come out below $\ln(V)$. It looks like a
bug in the model and it is a bug in whoever builds the batch, with a symptom identical to a broken
mask.

**3. It is genuinely causal.** Change the token at position 6 and look at how much the logits move
at each position:

```
   position 0-5:  0.00e+00    <- EXACTLY zero
   position 6:    1.46e+00    <- the changed token
   position 7:    2.52e-01
   position 8:    2.32e-01
```

Exactly zero before position 6, not "very small": those predictions cannot see token 6 in any way.
From there on they do change. That is the causal mask working, and there is no cleaner way to
demonstrate there is no leak.

---

## Where the memory goes

A calculation you will need in module 13, when the RTX 2060 runs out of memory and you have to
decide where to cut:

| what | MB |
|---|---|
| model weights (fp32) | 35.7 |
| gradients (fp32) | 35.7 |
| AdamW state (two moments per parameter) | 71.5 |
| **logits** in fp16, batch 48 × ctx 512 × vocab 4096 | 201.3 |
| + its fp32 version (`cross_entropy` promotes) | 402.7 |
| + its gradient | 402.7 |

**The logits alone take seven times more than the model, the gradients and the optimizer combined**
(1007 MB against 143 MB). It is counterintuitive and it is the direct consequence of that
`(B, T, 4096)` in the diagram: one number per vocabulary token, at every position, of every
sequence in the batch.

When you run out of memory, this is the first place to look, not the model's activations. The usual
fix is computing the loss in chunks instead of materializing the whole tensor.

## Where the debate is

You have just assembled a model with a dozen design decisions, and they do not all have the same
backing. Worth separating them:

**Well founded:** attention's $\sqrt{d_k}$ scaling (there is a clear variance argument, module 06),
the need for residuals, the final normalization in pre-norm.

**Convention with empirical support:** pre-norm over post-norm, RMSNorm over LayerNorm, SwiGLU over
GELU, weight tying. They do better on the benchmarks; there is no theory.

**Practically arbitrary:** the 0.02 in the initialization (it comes from GPT-2 and nobody has
re-justified it), the FFN's 4x factor, RoPE's $\theta = 10000$, the depth-to-width ratio.

And an honest one about this specific model: **6 layers of 320 dimensions is not an optimal choice
derived from anything.** It is a reasonable point for fitting in an RTX 2060 and training in hours.
With the same 9M parameters you could do 12 layers of 224, or 3 of 512, and they would work
similarly. The depth-to-width relationship is poorly explored at this scale, and now you have
exactly the tools to try it: change the YAML, run `expected_param_count`, and train.

---

**Further reading:** Radford et al. 2019,
[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
(where the scaled init and the 0.02 come from) · Press & Wolf 2017,
[Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
(weight tying) · [nanoGPT](https://github.com/karpathy/nanoGPT). Stray terms are in
[GLOSSARY.md](../../GLOSSARY.md).
