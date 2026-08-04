# 05 — Annotated solution

## Exercise 1 — `uniform_baseline_loss`

```python
if vocab_size < 1:
    raise ValueError("vocab_size must be positive")
return math.log(vocab_size)
```

One line. What matters is not the code, it is **what you are going to use it for**.

This number is your cheapest bug detector. When you start module 11's training run, the loss
at step 0 has to be almost exactly `ln(vocab_size)`:

| what you see at step 0 | what it means |
|---|---|
| ≈ ln(V) | correct, the model starts with no opinions |
| noticeably higher | the initialization is too aggressive |
| lower | information leak: almost always the causal mask |

The "lower" case deserves a moment of attention, because it looks like good news and it is
the most expensive bug in the course. If the model can see the token it has to predict, the
loss drops to almost zero right away, everything seems to be going wonderfully, and the
trained model is useless because at generation time that future does not exist.

**The demo shows you the "higher" case live.** The `NeuralBigram` starts at ~4.64 when the
floor is 4.13, because `nn.Embedding` initializes by default with a normal of standard
deviation 1. Since those rows *are* the logits, the model starts with strong, random bets.
That half a nat of excess is exactly the price of having opinions with no information. That
is why the GPT in module 10 uses `std=0.02` everywhere.

## Exercise 2 — `bigram_counts`

```python
counts = torch.zeros(vocab_size, vocab_size, dtype=torch.int64)
tokens = torch.as_tensor(ids, dtype=torch.int64)
if tokens.numel() < 2:
    return counts
counts.index_put_(
    (tokens[:-1], tokens[1:]),
    torch.ones(tokens.numel() - 1, dtype=torch.int64),
    accumulate=True,
)
return counts
```

**`accumulate=True` is not optional.** Without it, `index_put_` *assigns* instead of adding:
each repeated pair overwrites the previous one and every count ends up at 1. The test
`test_repeats_accumulate_rather_than_overwrite` exists for exactly that: with `[0,0,0,0,0]`
the correct result is 4, not 1.

**Why vectorized.** A `for` loop works and is more readable, but with 500M tokens that is
500 million Python iterations. `tokens[:-1]` gives all the "from"s and `tokens[1:]` all the
"to"s; PyTorch processes them in one go.

## Exercise 3 — `bigram_nll`

```python
tokens = torch.as_tensor(ids, dtype=torch.int64)
if tokens.numel() < 2:
    raise ValueError("at least 2 tokens are needed")

smoothed = counts.double() + alpha
probs = smoothed / smoothed.sum(dim=1, keepdim=True)
selected = probs[tokens[:-1], tokens[1:]]
return float(-torch.log(selected).mean())
```

**The smoothing is the heart of the exercise.** Without it, a single unseen pair in
validation has probability 0, its logarithm is `-inf`, and since the loss is a **mean**, that
`-inf` contaminates the whole result. The perplexity of your entire validation set goes to
infinity because of one pair you did not see.

Adding `alpha` to everything before normalizing is admitting that "I have not seen it" is not
the same as "it is impossible".

**The detail about the denominator.** Adding `alpha` to the `V` entries of a row grows that
row's total by `alpha * V`, not by `alpha`. If you normalized by dividing by
`original_sum + alpha`, the probabilities would not sum to 1. Doing `smoothed.sum(dim=1)`
*after* adding alpha takes care of this by itself, without you having to write the term.

**The `keepdim=True`.** Without it, `sum(dim=1)` returns shape `(V,)` instead of `(V, 1)`,
and the broadcast would divide by *columns* instead of by rows. The result would be
numerically plausible and completely wrong. It is a classic silent bug; the test
`test_each_rows_probabilities_sum_to_one` catches it.

**`.double()` and not `.float()`.** With large corpora millions of counts get added; float32
has 24 bits of mantissa and starts losing precision sooner than you would expect.

## Exercise 4 — `NeuralBigram`

```python
def __init__(self, vocab_size):
    super().__init__()
    self.vocab_size = vocab_size
    self.token_embedding = nn.Embedding(vocab_size, vocab_size)

def forward(self, idx, targets=None):
    logits = self.token_embedding(idx)
    if targets is None:
        return logits, None
    loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
    return logits, loss
```

**An `nn.Embedding(V, V)` as a language model** looks like a trick and it is not: row `i` of
the table is literally the logits of the token that follows `i`. Training this with
cross-entropy converges to exercise 2's normalized counts. The demo checks it: 2.4916 by
counting and 2.4838 by learning.

**Why `nn.Embedding` and not `nn.Linear`.** They are the same operation: an Embedding is a
Linear whose input is a one-hot vector. The difference is that the Embedding *reads* the row
it needs instead of multiplying by a matrix full of zeros. With V=4096, reading 4096 numbers
versus doing 16 million multiplications.

**The `reshape(-1, V)`.** `F.cross_entropy` expects `(N, classes)` and `(N,)`, but you have
`(B, T, V)` and `(B, T)`. Flattening batch and time into a single dimension is the pattern
you will repeat in every model in the course, including the final GPT.

## Exercise 5 — `BengioMLP`

```python
def forward(self, idx, targets=None):
    batch = idx.shape[0]
    emb = self.embedding(idx)          # (B, block_size, d_embed)
    flat = emb.reshape(batch, -1)      # (B, block_size * d_embed)
    h = torch.tanh(self.hidden(flat))
    logits = self.output(h)            # (B, V)
    if targets is None:
        return logits, None
    return logits, F.cross_entropy(logits, targets)
```

**Concatenate, do not average.** The `reshape(batch, -1)` glues the tokens' embeddings one
after another. If instead you did `emb.mean(dim=1)`, the model would lose the order:
`[the, cat, eats]` and `[eats, cat, the]` would look the same to it. The test
`test_bengio_concatenates_instead_of_averaging` checks it by passing the context in reverse.

Watch the `-1`: it goes in the **second** dimension. A `reshape(-1, batch)` compiles and
produces garbage.

**Its limitation, which is why module 06 exists.** The `hidden` layer is
`Linear(block_size * d_embed, n_hidden)`, so its parameters grow **linearly with the context
length**. With a context of 512 and `d_embed=320`, that layer alone would have 163,840
inputs.

And there is a deeper problem than the size: the model treats each position as an
independent input. It has no way of saying "of these 512 tokens, the ones that matter to me
right now are 3 and 47". Attention solves both at once.

## What you should see in the demo

```
uniform (random)     4.1271   perplexity 62.0
bigram (counting)    2.4916   perplexity 12.1
bigram (neural)      2.4838   perplexity 12.0
Bengio MLP (ctx 4)   2.0939   perplexity  8.1
```

Two things the demo points at that are worth not skipping over.

The first, that the counted bigram and the learned one give **the same number**. They are the
same model by two routes.

The second, that the MLP with context 8 comes out *worse* than the one with context 4. That
is not an error: all three trained for the same 400 steps and the one with context 8 has
twice the parameters, so it is left half-trained. Comparing architectures at equal **steps**
is not comparing them at equal **compute**, and it systematically favours the small model. It
is exactly the mistake module 12's scaling laws come to correct.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
def uniform_baseline_loss(vocab_size: int) -> float:
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    return math.log(vocab_size)


def bigram_counts(ids: Sequence[int], vocab_size: int) -> torch.Tensor:
    counts = torch.zeros(vocab_size, vocab_size, dtype=torch.int64)
    tokens = torch.as_tensor(ids, dtype=torch.int64)
    if tokens.numel() < 2:
        return counts
    # index_put_ with accumulate adds at repeated positions instead of overwriting them.
    counts.index_put_((tokens[:-1], tokens[1:]), torch.ones(tokens.numel() - 1, dtype=torch.int64),
                      accumulate=True)
    return counts


def bigram_nll(counts: torch.Tensor, ids: Sequence[int], alpha: float = 1.0) -> float:
    tokens = torch.as_tensor(ids, dtype=torch.int64)
    if tokens.numel() < 2:
        raise ValueError("at least 2 tokens are needed to evaluate a bigram")

    vocab_size = counts.shape[0]
    smoothed = counts.double() + alpha
    probs = smoothed / smoothed.sum(dim=1, keepdim=True)

    selected = probs[tokens[:-1], tokens[1:]]
    return float(-torch.log(selected).mean())


class NeuralBigram(nn.Module):

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_embedding(idx)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss


class BengioMLP(nn.Module):

    def __init__(self, vocab_size: int, block_size: int, d_embed: int = 32, n_hidden: int = 128) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_embed = d_embed
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.hidden = nn.Linear(block_size * d_embed, n_hidden)
        self.output = nn.Linear(n_hidden, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch = idx.shape[0]
        emb = self.embedding(idx)                    # (B, block_size, d_embed)
        flat = emb.reshape(batch, -1)                # (B, block_size * d_embed)
        h = torch.tanh(self.hidden(flat))            # (B, n_hidden)
        logits = self.output(h)                      # (B, V)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits, targets)
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
