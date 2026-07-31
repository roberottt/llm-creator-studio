"""Referencia del modulo 04: los modelos contra los que hay que ganar.

Antes de escribir un transformer conviene saber cual es el suelo. Si tu modelo de 9M
parametros no bate a una tabla de conteos de bigramas, tienes un bug, no un modelo.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_baseline_loss(vocab_size: int) -> float:
    """Perdida de un modelo que reparte la probabilidad por igual entre todos los tokens.

    Si `P(token) = 1/V` para todos, la log-verosimilitud negativa es
    `-ln(1/V) = ln(V)`. Con `V = 4096` da 8,317 nats.

    Es el numero mas util del entrenamiento: en el paso 0, con los pesos recien
    inicializados, la perdida tiene que valer casi exactamente esto. Si sale mucho mas
    alta, la inicializacion esta mal. Si sale mas baja, hay fuga de informacion.
    """
    if vocab_size < 1:
        raise ValueError("vocab_size debe ser positivo")
    return math.log(vocab_size)


def bigram_counts(ids: Sequence[int], vocab_size: int) -> torch.Tensor:
    """Matriz `V x V` con cuantas veces el token `j` sigue al token `i`.

    Returns:
        Tensor `int64` de forma `(vocab_size, vocab_size)`.
    """
    counts = torch.zeros(vocab_size, vocab_size, dtype=torch.int64)
    tokens = torch.as_tensor(ids, dtype=torch.int64)
    if tokens.numel() < 2:
        return counts
    # index_put_ con accumulate suma en las posiciones repetidas en vez de pisarlas.
    counts.index_put_((tokens[:-1], tokens[1:]), torch.ones(tokens.numel() - 1, dtype=torch.int64),
                      accumulate=True)
    return counts


def bigram_nll(counts: torch.Tensor, ids: Sequence[int], alpha: float = 1.0) -> float:
    """Log-verosimilitud negativa media de `ids` bajo el bigrama de `counts`.

    Con suavizado de Laplace:

    $$P(b \\mid a) = \\frac{C_{ab} + \\alpha}{\\sum_{b'} C_{ab'} + \\alpha V}$$

    El suavizado no es un detalle cosmetico: sin el, cualquier par que no aparezca en
    entrenamiento tiene probabilidad 0, su logaritmo es `-inf` y la perplejidad de todo
    el conjunto de validacion se va a infinito por un solo par no visto.

    Returns:
        Perdida media en nats por token.
    """
    tokens = torch.as_tensor(ids, dtype=torch.int64)
    if tokens.numel() < 2:
        raise ValueError("hacen falta al menos 2 tokens para evaluar un bigrama")

    vocab_size = counts.shape[0]
    smoothed = counts.double() + alpha
    probs = smoothed / smoothed.sum(dim=1, keepdim=True)

    selected = probs[tokens[:-1], tokens[1:]]
    return float(-torch.log(selected).mean())


class NeuralBigram(nn.Module):
    """El bigrama por conteo, escrito como red neuronal.

    Una `nn.Embedding(V, V)` donde la fila `i` son directamente los logits del token que
    sigue a `i`. Entrenada con cross-entropy converge a los conteos normalizados: es el
    mismo modelo, aprendido por descenso de gradiente en vez de contando.

    Submodulos (los tests copian pesos por nombre):
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
            targets: `(B, T)` int64 o `None`.

        Returns:
            `(logits, loss)` con logits `(B, T, V)`. `loss` es `None` si no hay targets.
        """
        logits = self.token_embedding(idx)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss


class BengioMLP(nn.Module):
    """El modelo de Bengio et al. 2003, el abuelo de todo esto.

    Concatena los embeddings de los `block_size` tokens anteriores y los pasa por un MLP.
    Dos ideas que siguen vigentes veinte anyos despues: representar las palabras como
    vectores densos aprendidos, y modelar la probabilidad del siguiente token con una red.

    Su limitacion es justo lo que la atencion viene a resolver: el contexto es de tamanyo
    fijo y la concatenacion hace que el numero de parametros crezca linealmente con el.

    Submodulos:
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
            idx: `(B, block_size)` int64, la ventana de contexto.
            targets: `(B,)` int64, el token a predecir.

        Returns:
            `(logits, loss)` con logits `(B, V)`.
        """
        batch = idx.shape[0]
        emb = self.embedding(idx)                    # (B, block_size, d_embed)
        flat = emb.reshape(batch, -1)                # (B, block_size * d_embed)
        h = torch.tanh(self.hidden(flat))            # (B, n_hidden)
        logits = self.output(h)                      # (B, V)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits, targets)
