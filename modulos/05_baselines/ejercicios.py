"""Modulo 05 - Baselines: contra que hay que comparar.

Cinco ejercicios. Los tres primeros son la metrica; los dos ultimos son modelos de verdad
en PyTorch, los primeros del curso.

    llmfs check 05
    llmfs demo 05     entrena los tres baselines y compara sus perdidas con el suelo

Lee antes TEORIA.md. El numero ln(V) del ejercicio 1 lo vas a usar durante todo el resto
del curso para saber si un entrenamiento arranca bien.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_baseline_loss(vocab_size: int) -> float:
    """La perdida de un modelo que no sabe absolutamente nada.

    QUE ES ESTO
        El numero mas util de todo el entrenamiento, y se calcula en una linea.

        Un modelo que reparte la probabilidad por igual entre las V palabras del
        vocabulario le da 1/V a cada una. La perdida es -ln(probabilidad del correcto),
        o sea -ln(1/V), que es ln(V).

    PARA QUE SIRVE
        Cuando lances el entrenamiento del modulo 11, la perdida del PASO 0 tiene que
        valer casi exactamente esto:

            vocab 4096  ->  ln(4096) = 8.317
            vocab 65    ->  ln(65)   = 4.174

        - Si sale MUCHO MAS ALTA (12, 20): la inicializacion esta mal. El modelo empieza
          con opiniones fuertes y equivocadas en vez de con ignorancia.
        - Si sale MAS BAJA: hay fuga de informacion. Casi siempre es la mascara causal
          mal puesta y el modelo viendo la respuesta.

        Es la comprobacion mas barata que existe y caza los dos bugs mas caros.

    Args:
        vocab_size: tamanyo del vocabulario. Tiene que ser >= 1.

    Returns:
        `ln(vocab_size)`, en nats.

    Raises:
        ValueError: si `vocab_size` no es positivo.
    """
    raise NotImplementedError("TODO: modulo 05, ejercicio 1 - uniform_baseline_loss")


def bigram_counts(ids: Sequence[int], vocab_size: int) -> torch.Tensor:
    """Cuenta cuantas veces sigue cada token a cada token.

    QUE ES ESTO
        Una matriz V x V donde `counts[i][j]` es el numero de veces que el token j vino
        justo detras del token i. Es la tabla del modulo 00, ahora como tensor.

    EJEMPLO CONCRETO
        ids = [0, 1, 0, 1, 2]  con vocab_size = 3

        Los pares consecutivos son (0,1), (1,0), (0,1), (1,2). Asi que:

            counts[0][1] = 2      el 1 siguio al 0 dos veces
            counts[1][0] = 1
            counts[1][2] = 1
            el resto = 0

    COMO SE HACE SIN BUCLES
        Se puede con un `for` y funciona, pero con 500M tokens tardaria una eternidad.
        La version vectorizada:

            tokens = torch.as_tensor(ids, dtype=torch.int64)
            counts.index_put_((tokens[:-1], tokens[1:]),
                              torch.ones(len(tokens)-1, dtype=torch.int64),
                              accumulate=True)

        `tokens[:-1]` son todos los "desde" y `tokens[1:]` todos los "hasta". El
        `accumulate=True` es imprescindible: sin el, las posiciones repetidas se PISAN en
        vez de sumarse, y todos los conteos saldrian 1.

    CASO BORDE
        Con menos de 2 tokens no hay ningun par: devuelve la matriz de ceros.

    Args:
        ids: la secuencia de tokens.
        vocab_size: tamanyo del vocabulario.

    Returns:
        Tensor `int64` de forma `(vocab_size, vocab_size)`.
    """
    raise NotImplementedError("TODO: modulo 05, ejercicio 2 - bigram_counts")


def bigram_nll(counts: torch.Tensor, ids: Sequence[int], alpha: float = 1.0) -> float:
    """La perdida media de una secuencia bajo un modelo de bigramas.

    QUE ES ESTO
        Ya tienes los conteos (del texto de entrenamiento). Ahora mides como de bien
        predicen OTRA secuencia (la de validacion).

    LA FORMULA

        P(b | a) = (C[a][b] + alpha) / (suma_b' C[a][b'] + alpha * V)

        perdida = -media de ln( P(ids[i+1] | ids[i]) )

    POR QUE EL `alpha` (suavizado de Laplace)
        Sin el, un par que nunca aparecio en entrenamiento tiene probabilidad 0. Su
        logaritmo es -infinito. Y como la perdida es la MEDIA, un solo par no visto manda
        la perdida de todo el conjunto de validacion a infinito.

        Sumar alpha a todos los conteos antes de normalizar es admitir que "no lo he
        visto" no es lo mismo que "es imposible". Con alpha=1 se le da a cada par no visto
        la misma probabilidad que si lo hubieras visto una vez.

        Fijate en el denominador: hay que sumar `alpha * V`, no solo `alpha`. Si sumas
        alpha a las V entradas de una fila, el total de esa fila crece en alpha*V. Si no
        lo compensas, las probabilidades no suman 1.

    COMO
        1. `smoothed = counts.double() + alpha`
           (a double, no a float: con 500M tokens, float32 pierde precision al sumar)
        2. `probs = smoothed / smoothed.sum(dim=1, keepdim=True)`
           El `keepdim=True` mantiene la forma (V,1) para que el broadcast divida cada
           fila por su propia suma. Sin el, la forma seria (V,) y dividiria por columnas.
        3. Selecciona las probabilidades de los pares reales:
           `probs[tokens[:-1], tokens[1:]]`  <- indexacion avanzada, un tensor de (N-1,)
        4. `float(-torch.log(seleccionadas).mean())`

    Args:
        counts: la matriz del ejercicio 2, calculada sobre ENTRENAMIENTO.
        ids: la secuencia a evaluar (normalmente validacion).
        alpha: constante de suavizado.

    Returns:
        Perdida media en nats por token.

    Raises:
        ValueError: si `ids` tiene menos de 2 tokens.
    """
    raise NotImplementedError("TODO: modulo 05, ejercicio 3 - bigram_nll")


class NeuralBigram(nn.Module):
    """El mismo bigrama, pero aprendido por descenso de gradiente en vez de contando.

    QUE ES ESTO
        Una `nn.Embedding(V, V)`. La fila `i` de esa tabla son directamente los LOGITS del
        token que sigue al token `i`.

        Parece un truco y es exactamente el modelo del ejercicio 2: entrenado con
        cross-entropy, converge a los conteos normalizados. Lo interesante es ver que
        contar y aprender dan lo mismo cuando el modelo es asi de simple. A partir de ahi,
        aprender escala y contar no.

    POR QUE `nn.Embedding` Y NO `nn.Linear`
        Son lo mismo matematicamente: un Embedding es un Linear cuya entrada es un vector
        one-hot. Pero el Embedding solo LEE la fila que necesita en vez de multiplicar por
        una matriz de V x V llena de ceros. Con V=4096 eso es la diferencia entre leer 4096
        numeros y multiplicar 16 millones.

    SUBMODULOS (los tests copian pesos por nombre, respeta el nombre)
        token_embedding: nn.Embedding(vocab_size, vocab_size)

    forward(idx, targets=None):
        Args:
            idx: `(B, T)` int64.
            targets: `(B, T)` int64, o `None`.
        Returns:
            `(logits, loss)` con logits `(B, T, V)`. `loss` es None si no hay targets.

        Para la perdida:
            F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))

        El `reshape(-1, V)` aplana batch y tiempo en una sola dimension, porque
        `cross_entropy` espera `(N, clases)` y `(N,)`. Es el patron que vas a repetir en
        todos los modelos del curso.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        raise NotImplementedError("TODO: modulo 05, ejercicio 4 - NeuralBigram.__init__")

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("TODO: modulo 05, ejercicio 4 - NeuralBigram.forward")


class BengioMLP(nn.Module):
    """El modelo de Bengio et al. 2003: el abuelo de los LLM modernos.

    QUE ES ESTO
        En vez de mirar solo el token anterior, mira los `block_size` anteriores.
        Concatena sus embeddings en un vector largo y lo pasa por un MLP.

        Con block_size=4 y d_embed=32, el vector concatenado tiene 128 numeros, y de ahi
        salen los logits sobre todo el vocabulario.

    POR QUE IMPORTA
        Dos ideas suyas siguen vivas veinte anyos despues: representar cada palabra como un
        VECTOR APRENDIDO (y no como un id sin estructura), y modelar la probabilidad del
        siguiente token con una red.

        Y su limitacion es exactamente lo que la atencion viene a resolver: el contexto es
        de tamanyo fijo, y como se concatena, el numero de parametros de la primera capa
        crece linealmente con la longitud del contexto. Con contexto 512 esa capa sola
        seria enorme.

    SUBMODULOS (respeta los nombres)
        embedding: nn.Embedding(vocab_size, d_embed)
        hidden:    nn.Linear(block_size * d_embed, n_hidden)
        output:    nn.Linear(n_hidden, vocab_size)

    forward(idx, targets=None):
        Args:
            idx: `(B, block_size)` int64.
            targets: `(B,)` int64, UN solo token por muestra (no una secuencia).
        Returns:
            `(logits, loss)` con logits `(B, V)`.

        Los pasos:
            emb    = self.embedding(idx)        -> (B, block_size, d_embed)
            flat   = emb.reshape(B, -1)         -> (B, block_size * d_embed)
            h      = torch.tanh(self.hidden(flat))
            logits = self.output(h)             -> (B, V)

        Usa `tanh`, que es lo que usaba el paper original. Cuidado con el `reshape`: el
        -1 tiene que estar en la SEGUNDA dimension, no en la primera.
    """

    def __init__(
        self, vocab_size: int, block_size: int, d_embed: int = 32, n_hidden: int = 128
    ) -> None:
        super().__init__()
        raise NotImplementedError("TODO: modulo 05, ejercicio 5 - BengioMLP.__init__")

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("TODO: modulo 05, ejercicio 5 - BengioMLP.forward")
