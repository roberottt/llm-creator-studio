"""Reference for module 05: the models you have to beat.

Before writing a transformer it is worth knowing where the floor is. If your 9M-parameter
model does not beat a table of bigram counts, you have a bug, not a model.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_baseline_loss(vocab_size: int) -> float:
    """Loss of a model that spreads probability equally across every token.

    If `P(token) = 1/V` for all of them, the negative log-likelihood is
    `-ln(1/V) = ln(V)`. With `V = 4096` that gives 8.317 nats.

    It is the most useful number in training: at step 0, with freshly initialized weights,
    the loss has to be almost exactly this. If it comes out much higher, the initialization
    is wrong. If it comes out lower, there is an information leak.
    """
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    return math.log(vocab_size)


def bigram_counts(ids: Sequence[int], vocab_size: int) -> torch.Tensor:
    """A `V x V` matrix of how many times token `j` follows token `i`.

    Returns:
        An `int64` tensor of shape `(vocab_size, vocab_size)`.
    """
    counts = torch.zeros(vocab_size, vocab_size, dtype=torch.int64)
    tokens = torch.as_tensor(ids, dtype=torch.int64)
    if tokens.numel() < 2:
        return counts
    # index_put_ with accumulate adds at repeated positions instead of overwriting them.
    counts.index_put_((tokens[:-1], tokens[1:]), torch.ones(tokens.numel() - 1, dtype=torch.int64),
                      accumulate=True)
    return counts


def bigram_nll(counts: torch.Tensor, ids: Sequence[int], alpha: float = 1.0) -> float:
    """Mean negative log-likelihood of `ids` under the bigram in `counts`.

    With Laplace smoothing:

    $$P(b \\mid a) = \\frac{C_{ab} + \\alpha}{\\sum_{b'} C_{ab'} + \\alpha V}$$

    The smoothing is not a cosmetic detail: without it, any pair that does not appear in
    training has probability 0, its logarithm is `-inf`, and the perplexity of the whole
    validation set goes to infinity because of a single unseen pair.

    Returns:
        Mean loss in nats per token.
    """
    tokens = torch.as_tensor(ids, dtype=torch.int64)
    if tokens.numel() < 2:
        raise ValueError("at least 2 tokens are needed to evaluate a bigram")

    vocab_size = counts.shape[0]
    smoothed = counts.double() + alpha
    probs = smoothed / smoothed.sum(dim=1, keepdim=True)

    selected = probs[tokens[:-1], tokens[1:]]
    return float(-torch.log(selected).mean())


class NeuralBigram(nn.Module):
    """The count-based bigram, written as a neural network.

    An `nn.Embedding(V, V)` where row `i` is directly the logits of the token that follows
    `i`. Trained with cross-entropy it converges to the normalized counts: it is the same
    model, learned by gradient descent instead of by counting.

    Submodules (the tests copy weights by name):
        token_embedding: nn.Embedding(vocab_size, vocab_size)
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx: `(B, T)` int64.
            targets: `(B, T)` int64 or `None`.

        Returns:
            `(logits, loss)` with logits `(B, T, V)`. `loss` is `None` if there are no
            targets.
        """
        logits = self.token_embedding(idx)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss


class BengioMLP(nn.Module):
    """The model from Bengio et al. 2003, the grandparent of all this.

    It concatenates the embeddings of the previous `block_size` tokens and runs them
    through an MLP. Two ideas still standing twenty years later: representing words as
    learned dense vectors, and modelling the next-token probability with a network.

    Its limitation is exactly what attention comes to solve: the context has a fixed size
    and the concatenation makes the parameter count grow linearly with it.

    Submodules:
        embedding: nn.Embedding(vocab_size, d_embed)
        hidden:    nn.Linear(block_size * d_embed, n_hidden)
        output:    nn.Linear(n_hidden, vocab_size)
    """

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
        """
        Args:
            idx: `(B, block_size)` int64, the context window.
            targets: `(B,)` int64, the token to predict.

        Returns:
            `(logits, loss)` with logits `(B, V)`.
        """
        batch = idx.shape[0]
        emb = self.embedding(idx)                    # (B, block_size, d_embed)
        flat = emb.reshape(batch, -1)                # (B, block_size * d_embed)
        h = torch.tanh(self.hidden(flat))            # (B, n_hidden)
        logits = self.output(h)                      # (B, V)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits, targets)
