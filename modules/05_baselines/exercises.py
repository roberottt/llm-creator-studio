"""Module 05 - Baselines: what you have to compare against.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement -> `llmfs check 05` -> `llmfs hint 05 -e N`
-> `SOLUTION.md` has the complete code.

WHAT YOU ARE GOING TO BUILD
===========================

The way to measure, and three models to compare against:

    uniform_baseline_loss  (ex. 1)  THE FLOOR: what a model that knows nothing scores
    bigram_counts          (ex. 2)  count which token follows each token
    bigram_nll             (ex. 3)  measure how well that table predicts
    NeuralBigram           (ex. 4)  the same model, but learned by gradient
    BengioMLP              (ex. 5)  the grandparent of LLMs (2003)

Exercises 4 and 5 are your first two models in PyTorch.

Exercise 1 is one line and it is the most important: `ln(V)` will tell you, at step 0 of any
training run in the course, whether there is a bug.

`THEORY.md` is ordered just like this list: each exercise has its own section there with its
numeric example, and each docstring here tells you which one. Before exercise 4 there is a
separate section, "A stop along the way: what a model is in PyTorch", that translates what you
already did by hand in module 02 into the vocabulary of `torch.nn`: `nn.Module`, `forward`, the
`(B, T, V)` shapes and what `F.cross_entropy` does inside. If `nn.Embedding` sounds like a name
you have to copy without knowing what it does, that section is the missing piece.

VOCABULARY YOU ARE GOING TO NEED
================================

- **logit**: the raw score the model gives each token, before turning it into a probability.
  It can be any number, positive or negative.
- **cross-entropy**: the loss function of every language model. It is
  `-ln(probability you gave to the correct token)`.
- **perplexity**: `e` raised to the loss. It reads as "how many options the model is torn
  between".
- **nat**: the unit of the loss when using the natural logarithm. One nat is 1.44 bits.
- **Laplace smoothing**: adding a constant to every count so none of them is zero. Without
  it, a single unseen pair sends the loss to infinity.

    llmfs demo 05     trains the three baselines and compares them against the floor
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_baseline_loss(vocab_size: int) -> float:
    """The loss of a model that knows absolutely nothing.

    Context in `THEORY.md`: section "The floor: what a model that knows nothing scores".

    WHAT YOU HAVE TO WRITE
    ----------------------
    Two lines.

        1. If `vocab_size < 1`, raise `ValueError`.
        2. `return math.log(vocab_size)`

    WHERE IT COMES FROM
    -------------------
    A model that spreads probability equally across the V words of the vocabulary gives
    `1/V` to each one. The loss is `-ln(probability of the correct token)`, that is
    `-ln(1/V)`, which is `ln(V)`.

    WHAT IT IS FOR (this is the important part of the exercise)
    ----------------------------------------------------------
    It is the most useful number in the whole course. When you launch ANY training run, the
    loss at STEP 0 has to be almost exactly this:

        vocab 65    ->  ln(65)   = 4.174     (character-level shakespeare)
        vocab 4096  ->  ln(4096) = 8.317     (the final model)

    And then:

        - if it comes out MUCH HIGHER: the initialization is wrong. The model starts with
          strong, mistaken opinions instead of honest ignorance.
        - if it comes out LOWER: there is an information leak. It is almost always a badly
          placed causal mask and the model seeing the answer.

    The second case looks like good news and it is the most expensive bug in the course: the
    loss drops spectacularly, everything seems to be going wonderfully, and the trained
    model is useless because at generation time that future does not exist.

    It is the cheapest check there is and it catches the two most expensive bugs.

    Args:
        vocab_size: the vocabulary size. It has to be >= 1.

    Returns:
        `ln(vocab_size)`, in nats.

    Raises:
        ValueError: if `vocab_size` is not positive.
    """
    raise NotImplementedError("TODO: module 05, exercise 1 - uniform_baseline_loss")


def bigram_counts(ids: Sequence[int], vocab_size: int) -> torch.Tensor:
    """Counts how many times each token follows each token.

    Context in `THEORY.md`: section "Exercise 2: counting the pairs", with the matrix for this
    very example drawn cell by cell and why the table is V x V.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Four lines, no loops.

        1. Create the matrix of zeros:

               counts = torch.zeros(vocab_size, vocab_size, dtype=torch.int64)

        2. Turn the ids into a tensor:

               tokens = torch.as_tensor(ids, dtype=torch.int64)

        3. If there are fewer than 2 tokens, return the matrix of zeros as is (no pairs).

        4. Fill it in one go:

               counts.index_put_(
                   (tokens[:-1], tokens[1:]),
                   torch.ones(tokens.numel() - 1, dtype=torch.int64),
                   accumulate=True,
               )

        5. Return `counts`.

    HOW STEP 4 WORKS
    ----------------
    `tokens[:-1]` are all the "from"s and `tokens[1:]` all the "to"s. `index_put_` walks
    those two lists in parallel and adds 1 at each position `(from, to)`.

    EXAMPLE TO CHECK AGAINST
    ------------------------
        ids = [0, 1, 0, 1, 2]  with vocab_size = 3

        the pairs are (0,1), (1,0), (0,1), (1,2), so:

            counts[0][1] = 2      1 followed 0 twice
            counts[1][0] = 1
            counts[1][2] = 1
            everything else = 0

    THE `accumulate=True` IS NOT OPTIONAL
    -------------------------------------
    Without it, `index_put_` ASSIGNS instead of adding: each repeated pair overwrites the
    previous one and every count ends up at 1.

    Try it with `[0,0,0,0,0]`: the correct result is `counts[0][0] = 4`. Without
    `accumulate=True` it would come out 1. There is a test dedicated to this.

    WHY VECTORIZED AND NOT A LOOP
    -----------------------------
    A `for` works and is more readable, but with 500M tokens that is 500 million python
    iterations.

    Args:
        ids: the sequence of tokens.
        vocab_size: the vocabulary size.

    Returns:
        An `int64` tensor of shape `(vocab_size, vocab_size)`.
    """
    raise NotImplementedError("TODO: module 05, exercise 2 - bigram_counts")


def bigram_nll(counts: torch.Tensor, ids: Sequence[int], alpha: float = 1.0) -> float:
    """Measures how well a bigram table predicts.

    Context in `THEORY.md`: section "Exercise 3: measuring the table", with the example table
    smoothed row by row, the loss worked out by hand and the table of what each alpha does.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Five lines.

        1. Turn the ids into a tensor and validate that there are at least 2:

               tokens = torch.as_tensor(ids, dtype=torch.int64)
               if tokens.numel() < 2:
                   raise ValueError("at least 2 tokens are needed")

        2. Add alpha to every count, in double:

               smoothed = counts.double() + alpha

        3. Normalize each ROW:

               probs = smoothed / smoothed.sum(dim=1, keepdim=True)

        4. Select the probabilities of the pairs that actually appear:

               selected = probs[tokens[:-1], tokens[1:]]

        5. Return the mean of -ln:

               return float(-torch.log(selected).mean())

    THE FORMULA
    -----------
        P(b | a) = (C[a][b] + alpha) / (sum_b' C[a][b'] + alpha * V)

    Note that step 3 produces that denominator BY ITSELF: having added alpha to the V entries
    of each row in step 2, the row's sum has already grown by `alpha * V`. You do not have to
    write that term by hand.

    WHY THE SMOOTHING (the `alpha`)
    -------------------------------
    Without it, a pair that never appeared in training has probability 0, its logarithm is
    `-inf`, and since the loss is a MEAN, that `-inf` takes the whole result with it. The
    perplexity of your entire validation set goes to infinity because of a single pair you
    did not see.

    Adding alpha to everything before normalizing is admitting that "I have not seen it" is
    not the same as "it is impossible".

    TWO TRAPS
    ---------
    **The `keepdim=True` in step 3.** Without it, `sum(dim=1)` returns shape `(V,)` instead
    of `(V, 1)`, and the broadcast would divide by COLUMNS instead of by rows. The result
    comes out numerically plausible and is completely wrong. There is a test that catches it.

    **`.double()` and not `.float()`.** With large corpora millions of counts get added, and
    float32 has 24 bits of mantissa: it starts losing precision sooner than you would expect.

    Args:
        counts: the matrix from exercise 2, computed on TRAINING data.
        ids: the sequence to evaluate (usually validation).
        alpha: the smoothing constant.

    Returns:
        The mean loss in nats per token.

    Raises:
        ValueError: if `ids` has fewer than 2 tokens.
    """
    raise NotImplementedError("TODO: module 05, exercise 3 - bigram_nll")


class NeuralBigram(nn.Module):
    """The bigram from exercise 2, but learned by gradient.

    If this is your first model in PyTorch, read the section "A stop along the way: what a
    model is in PyTorch" in `THEORY.md` first: what `nn.Module` registers, why it is called as
    `model(x)` and not `model.forward(x)`, what the `(B, T, V)` shapes are and why
    `F.cross_entropy` wants flattened logits. Then the "Exercise 4" section.

    WHAT YOU HAVE TO WRITE
    ----------------------
    **In `__init__`** (two lines besides the `super()`):

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    The name `token_embedding` matters: the test copies weights by name.

    **In `forward`** (four lines):

        logits = self.token_embedding(idx)          # (B, T) -> (B, T, V)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )
        return logits, loss

    WHAT THIS MODEL IS
    ------------------
    A table with V rows and V columns where row `i` is DIRECTLY the logits of the token that
    follows token `i`.

    It looks like a trick and it is exactly the model from exercise 2: trained with
    cross-entropy, it converges to the normalized counts. The interesting part is seeing that
    COUNTING and LEARNING give the same thing when the model is this simple. From there on,
    learning scales and counting does not.

    WHY `nn.Embedding` AND NOT `nn.Linear`
    --------------------------------------
    They are the same operation: an Embedding is a Linear whose input is a one-hot vector.

    The difference is that the Embedding READS the row it needs instead of multiplying by a
    matrix full of zeros. With V=4096, that is reading 4096 numbers versus doing 16 million
    multiplications.

    THE `reshape(-1, V)`
    --------------------
    `F.cross_entropy` expects `(N, classes)` and `(N,)`, but you have `(B, T, V)` and
    `(B, T)`. Flattening batch and time into a single dimension is the pattern you will
    repeat in EVERY model in the course, including the final GPT.

    Return `(logits, None)` if there are no targets, not `(logits, 0)`.

    forward(idx, targets=None):
        Args:
            idx: `(B, T)` int64.
            targets: `(B, T)` int64, or `None`.
        Returns:
            `(logits, loss)` with logits `(B, T, V)`.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        raise NotImplementedError("TODO: module 05, exercise 4 - NeuralBigram.__init__")

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("TODO: module 05, exercise 4 - NeuralBigram.forward")


class BengioMLP(nn.Module):
    """The model from Bengio et al. 2003: the grandparent of modern LLMs.

    Context in `THEORY.md`: section "Exercise 5: looking further back", with the route through
    the shapes drawn step by step and the measured numbers for where this model breaks, which
    is the problem attention solves in module 06.

    WHAT YOU HAVE TO WRITE
    ----------------------
    **In `__init__`** (five lines besides the `super()`). Respect the names:

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_embed = d_embed
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.hidden = nn.Linear(block_size * d_embed, n_hidden)
        self.output = nn.Linear(n_hidden, vocab_size)

    **In `forward`** (five lines):

        batch = idx.shape[0]
        emb = self.embedding(idx)            # (B, block_size, d_embed)
        flat = emb.reshape(batch, -1)        # (B, block_size * d_embed)
        h = torch.tanh(self.hidden(flat))    # (B, n_hidden)
        logits = self.output(h)              # (B, V)

        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits, targets)

    Use `tanh`, which is what the original paper used.

    CONCATENATE, DO NOT AVERAGE
    ---------------------------
    The `reshape(batch, -1)` glues the tokens' embeddings one after another.

    If you did `emb.mean(dim=1)` you would be averaging, and the model would lose the ORDER:
    `[the, cat, eats]` and `[eats, cat, the]` would look the same to it. There is a test that
    checks this by passing the context in reverse.

    And watch where the `-1` goes: `reshape(batch, -1)`, not `reshape(-1, batch)`. The second
    compiles and produces garbage.

    WATCH THE SHAPE OF `targets`
    ----------------------------
    Here `targets` is `(B,)`, ONE token per sample, not a sequence. That is why
    `cross_entropy` is called without a `reshape`: the logits are already `(B, V)`.

    WHY THIS MODEL MATTERS
    ----------------------
    Two of its ideas are still alive twenty years later: representing each word as a LEARNED
    VECTOR (and not as a structureless id), and modelling the next token's probability with a
    network.

    And its limitation is exactly what attention comes to solve: the `hidden` layer is
    `Linear(block_size * d_embed, n_hidden)`, so its parameters grow LINEARLY with the
    context length. With a context of 512 and d_embed 320, that layer alone would have
    163,840 inputs.

    And there is a deeper problem than the size: the model treats each position as an
    independent input. It has no way of saying "of these 512 tokens, the ones that matter to
    me right now are 3 and 47".

    forward(idx, targets=None):
        Args:
            idx: `(B, block_size)` int64.
            targets: `(B,)` int64, ONE token per sample.
        Returns:
            `(logits, loss)` with logits `(B, V)`.
    """

    def __init__(
        self, vocab_size: int, block_size: int, d_embed: int = 32, n_hidden: int = 128
    ) -> None:
        super().__init__()
        raise NotImplementedError("TODO: module 05, exercise 5 - BengioMLP.__init__")

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("TODO: module 05, exercise 5 - BengioMLP.forward")
