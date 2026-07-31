"""Modulo 05 - Baselines: contra que hay que comparar.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa -> `llmfs check 05` -> `llmfs hint 05 -e N`
-> `SOLUCION.md` tiene el codigo completo.

QUÉ VAS A CONSTRUIR
===================

La forma de medir, y tres modelos contra los que comparar:

    uniform_baseline_loss  (ej. 1)  EL SUELO: lo que saca un modelo que no sabe nada
    bigram_counts          (ej. 2)  contar que token sigue a cada token
    bigram_nll             (ej. 3)  medir como de bien predice esa tabla
    NeuralBigram           (ej. 4)  el mismo modelo, pero aprendido por gradiente
    BengioMLP              (ej. 5)  el abuelo de los LLM (2003)

Los ejercicios 4 y 5 son tus dos primeros modelos en PyTorch.

El ejercicio 1 es una linea y es el mas importante: `ln(V)` te va a decir, en el paso 0 de
cualquier entrenamiento del curso, si hay un bug.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **logit**: la puntuacion en bruto que el modelo da a cada token, antes de convertirla en
  probabilidad. Puede ser cualquier numero, positivo o negativo.
- **cross-entropy**: la funcion de perdida de todos los modelos de lenguaje. Es
  `-ln(probabilidad que le diste al token correcto)`.
- **perplejidad**: `e` elevado a la perdida. Se lee como "entre cuantas opciones esta
  dudando el modelo".
- **nat**: la unidad de la perdida cuando usas logaritmo natural. Un nat son 1,44 bits.
- **suavizado de Laplace**: sumar una constante a todos los conteos para que ninguno sea
  cero. Sin eso, un solo par no visto manda la perdida a infinito.

    llmfs demo 05     entrena los tres baselines y los compara con el suelo
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_baseline_loss(vocab_size: int) -> float:
    """La perdida de un modelo que no sabe absolutamente nada.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Dos lineas.

        1. Si `vocab_size < 1`, lanza `ValueError`.
        2. `return math.log(vocab_size)`

    DE DÓNDE SALE
    -------------
    Un modelo que reparte la probabilidad por igual entre las V palabras del vocabulario le da
    `1/V` a cada una. La perdida es `-ln(probabilidad del token correcto)`, o sea `-ln(1/V)`,
    que es `ln(V)`.

    PARA QUÉ SIRVE (esto es lo importante del ejercicio)
    ----------------------------------------------------
    Es el numero mas util de todo el curso. Cuando lances CUALQUIER entrenamiento, la perdida
    del PASO 0 tiene que valer casi exactamente esto:

        vocab 65    ->  ln(65)   = 4.174     (shakespeare a nivel caracter)
        vocab 4096  ->  ln(4096) = 8.317     (el modelo final)

    Y entonces:

        - si sale MUCHO MAS ALTA: la inicializacion esta mal. El modelo arranca con opiniones
          fuertes y equivocadas en vez de con ignorancia honesta.
        - si sale MAS BAJA: hay fuga de informacion. Casi siempre es la mascara causal mal
          puesta y el modelo viendo la respuesta.

    El segundo caso parece una buena noticia y es el bug mas caro del curso: la perdida baja
    espectacularmente, todo parece ir de maravilla, y el modelo entrenado no sirve para nada
    porque en generacion ese futuro no existe.

    Es la comprobacion mas barata que existe y caza los dos bugs mas caros.

    Args:
        vocab_size: el tamanyo del vocabulario. Tiene que ser >= 1.

    Returns:
        `ln(vocab_size)`, en nats.

    Raises:
        ValueError: si `vocab_size` no es positivo.
    """
    raise NotImplementedError("TODO: modulo 05, ejercicio 1 - uniform_baseline_loss")


def bigram_counts(ids: Sequence[int], vocab_size: int) -> torch.Tensor:
    """Cuenta cuantas veces sigue cada token a cada token.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Cuatro lineas, sin bucles.

        1. Crea la matriz de ceros:

               counts = torch.zeros(vocab_size, vocab_size, dtype=torch.int64)

        2. Pasa los ids a tensor:

               tokens = torch.as_tensor(ids, dtype=torch.int64)

        3. Si hay menos de 2 tokens, devuelve la matriz de ceros tal cual (no hay pares).

        4. Rellena de golpe:

               counts.index_put_(
                   (tokens[:-1], tokens[1:]),
                   torch.ones(tokens.numel() - 1, dtype=torch.int64),
                   accumulate=True,
               )

        5. Devuelve `counts`.

    CÓMO FUNCIONA EL PASO 4
    -----------------------
    `tokens[:-1]` son todos los "desde" y `tokens[1:]` todos los "hasta". `index_put_` recorre
    esas dos listas en paralelo y suma 1 en cada posicion `(desde, hasta)`.

    EJEMPLO PARA COMPROBAR
    ----------------------
        ids = [0, 1, 0, 1, 2]  con vocab_size = 3

        los pares son (0,1), (1,0), (0,1), (1,2), asi que:

            counts[0][1] = 2      el 1 siguio al 0 dos veces
            counts[1][0] = 1
            counts[1][2] = 1
            el resto     = 0

    EL `accumulate=True` NO ES OPCIONAL
    -----------------------------------
    Sin el, `index_put_` ASIGNA en vez de sumar: cada par repetido pisa al anterior y todos los
    conteos acaban valiendo 1.

    Pruebalo con `[0,0,0,0,0]`: el resultado correcto es `counts[0][0] = 4`. Sin
    `accumulate=True` saldria 1. Hay un test dedicado a esto.

    POR QUÉ VECTORIZADO Y NO UN BUCLE
    ---------------------------------
    Un `for` funciona y es mas legible, pero con 500M tokens son 500 millones de iteraciones de
    python.

    Args:
        ids: la secuencia de tokens.
        vocab_size: el tamanyo del vocabulario.

    Returns:
        Tensor `int64` de forma `(vocab_size, vocab_size)`.
    """
    raise NotImplementedError("TODO: modulo 05, ejercicio 2 - bigram_counts")


def bigram_nll(counts: torch.Tensor, ids: Sequence[int], alpha: float = 1.0) -> float:
    """Mide como de bien predice una tabla de bigramas.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Cinco lineas.

        1. Pasa los ids a tensor y valida que haya al menos 2:

               tokens = torch.as_tensor(ids, dtype=torch.int64)
               if tokens.numel() < 2:
                   raise ValueError("hacen falta al menos 2 tokens")

        2. Suma alpha a todos los conteos, en double:

               smoothed = counts.double() + alpha

        3. Normaliza cada FILA:

               probs = smoothed / smoothed.sum(dim=1, keepdim=True)

        4. Selecciona las probabilidades de los pares que aparecen de verdad:

               selected = probs[tokens[:-1], tokens[1:]]

        5. Devuelve la media de -ln:

               return float(-torch.log(selected).mean())

    LA FORMULA
    ----------
        P(b | a) = (C[a][b] + alpha) / (suma_b' C[a][b'] + alpha * V)

    Fijate en que el paso 3 produce ese denominador SOLO: al haber sumado alpha a las V
    entradas de cada fila en el paso 2, la suma de la fila ya crecio en `alpha * V`. No tienes
    que escribir ese termino a mano.

    POR QUÉ EL SUAVIZADO (el `alpha`)
    ---------------------------------
    Sin el, un par que nunca aparecio en entrenamiento tiene probabilidad 0, su logaritmo es
    `-inf`, y como la perdida es una MEDIA, ese `-inf` se lleva por delante el resultado
    entero. La perplejidad de todo tu conjunto de validacion se va a infinito por un solo par
    que no viste.

    Sumar alpha a todo antes de normalizar es admitir que "no lo he visto" no es lo mismo que
    "es imposible".

    DOS TRAMPAS
    -----------
    **El `keepdim=True` del paso 3.** Sin el, `sum(dim=1)` devuelve forma `(V,)` en vez de
    `(V, 1)`, y el broadcast dividiria por COLUMNAS en lugar de por filas. El resultado sale
    numericamente plausible y es completamente incorrecto. Hay un test que lo caza.

    **`.double()` y no `.float()`.** Con corpus grandes se suman millones de conteos, y float32
    tiene 24 bits de mantisa: empieza a perder precision antes de lo que uno espera.

    Args:
        counts: la matriz del ejercicio 2, calculada sobre ENTRENAMIENTO.
        ids: la secuencia a evaluar (normalmente validacion).
        alpha: la constante de suavizado.

    Returns:
        La perdida media en nats por token.

    Raises:
        ValueError: si `ids` tiene menos de 2 tokens.
    """
    raise NotImplementedError("TODO: modulo 05, ejercicio 3 - bigram_nll")


class NeuralBigram(nn.Module):
    """El bigrama del ejercicio 2, pero aprendido por gradiente.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    **En `__init__`** (dos lineas ademas del `super()`):

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    El nombre `token_embedding` importa: el test copia pesos por nombre.

    **En `forward`** (cuatro lineas):

        logits = self.token_embedding(idx)          # (B, T) -> (B, T, V)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )
        return logits, loss

    QUÉ ES ESTE MODELO
    ------------------
    Una tabla de V filas y V columnas donde la fila `i` son DIRECTAMENTE los logits del token
    que sigue al token `i`.

    Parece un truco y es exactamente el modelo del ejercicio 2: entrenado con cross-entropy,
    converge a los conteos normalizados. Lo interesante es ver que CONTAR y APRENDER dan lo
    mismo cuando el modelo es asi de simple. A partir de ahi, aprender escala y contar no.

    POR QUÉ `nn.Embedding` Y NO `nn.Linear`
    ---------------------------------------
    Son la misma operacion: un Embedding es un Linear cuya entrada es un vector one-hot.

    La diferencia es que el Embedding LEE la fila que necesita en lugar de multiplicar por una
    matriz llena de ceros. Con V=4096, eso es leer 4096 numeros frente a hacer 16 millones de
    multiplicaciones.

    EL `reshape(-1, V)`
    -------------------
    `F.cross_entropy` espera `(N, clases)` y `(N,)`, pero tu tienes `(B, T, V)` y `(B, T)`.
    Aplanar batch y tiempo en una sola dimension es el patron que vas a repetir en TODOS los
    modelos del curso, incluido el GPT final.

    Devuelve `(logits, None)` si no hay targets, no `(logits, 0)`.

    forward(idx, targets=None):
        Args:
            idx: `(B, T)` int64.
            targets: `(B, T)` int64, o `None`.
        Returns:
            `(logits, loss)` con logits `(B, T, V)`.
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

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    **En `__init__`** (cinco lineas ademas del `super()`). Respeta los nombres:

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_embed = d_embed
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.hidden = nn.Linear(block_size * d_embed, n_hidden)
        self.output = nn.Linear(n_hidden, vocab_size)

    **En `forward`** (cinco lineas):

        batch = idx.shape[0]
        emb = self.embedding(idx)            # (B, block_size, d_embed)
        flat = emb.reshape(batch, -1)        # (B, block_size * d_embed)
        h = torch.tanh(self.hidden(flat))    # (B, n_hidden)
        logits = self.output(h)              # (B, V)

        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits, targets)

    Usa `tanh`, que es lo que usaba el paper original.

    CONCATENAR, NO PROMEDIAR
    ------------------------
    El `reshape(batch, -1)` pega los embeddings de los tokens uno detras de otro.

    Si hicieras `emb.mean(dim=1)` estarias promediando, y el modelo perderia el ORDEN: le
    daria lo mismo `[el, gato, come]` que `[come, gato, el]`. Hay un test que lo comprueba
    pasando el contexto al reves.

    Y cuidado con donde va el `-1`: `reshape(batch, -1)`, no `reshape(-1, batch)`. El segundo
    compila y produce basura.

    OJO CON LA FORMA DE `targets`
    -----------------------------
    Aqui `targets` es `(B,)`, UN token por muestra, no una secuencia. Por eso `cross_entropy`
    se llama sin `reshape`: los logits ya son `(B, V)`.

    POR QUÉ IMPORTA ESTE MODELO
    ---------------------------
    Dos ideas suyas siguen vivas veinte anyos despues: representar cada palabra como un VECTOR
    APRENDIDO (y no como un id sin estructura), y modelar la probabilidad del siguiente token
    con una red.

    Y su limitacion es exactamente lo que la atencion viene a resolver: la capa `hidden` es
    `Linear(block_size * d_embed, n_hidden)`, asi que sus parametros crecen LINEALMENTE con la
    longitud del contexto. Con contexto 512 y d_embed 320, esa capa sola tendria 163.840
    entradas.

    Y hay un problema mas profundo que el tamanyo: el modelo trata cada posicion como una
    entrada independiente. No tiene forma de decir "de estos 512 tokens, los que me importan
    ahora son el 3 y el 47".

    forward(idx, targets=None):
        Args:
            idx: `(B, block_size)` int64.
            targets: `(B,)` int64, UN token por muestra.
        Returns:
            `(logits, loss)` con logits `(B, V)`.
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
