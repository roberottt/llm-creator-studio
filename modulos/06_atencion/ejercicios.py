"""Modulo 06 - Self-attention.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa en orden -> `llmfs check 06` -> `llmfs hint 06 -e N`
-> `SOLUCION.md` tiene el codigo completo.

El ejemplo de "el gato que vi ayer dormia", hecho a mano con tres palabras y dos
dimensiones, es EXACTAMENTE lo que vas a programar. Tenlo delante.

QUÉ VAS A CONSTRUIR
===================

El corazon del Transformer. Tres ejercicios que encajan asi:

    causal_mask             (ej. 1)  impedir que un token mire al futuro
            |
            v
    single_head_attention   (ej. 2)  la formula, con una sola cabeza
            |
            v
    MultiHeadAttention      (ej. 3)  ocho en paralelo, que es lo que usa el modelo

El ejercicio 2 son cuatro lineas, y cada una tiene una trampa. El 3 es el mismo calculo con
una dimension mas.

LA IDEA, EN UNA FRASE
=====================

Cada token hace una PREGUNTA, todos los anteriores RESPONDEN, se mide cuanto encaja cada
respuesta, y se mezcla su CONTENIDO segun eso.

    salida = softmax( Q K^T / sqrt(d_k) + mascara ) V

VOCABULARIO QUE VAS A NECESITAR
===============================

- **Q, K, V** (query, key, value): las tres proyecciones. La query es la pregunta que lanza
  un token, la key es la etiqueta con la que cada token se anuncia, y el value es el
  contenido que aporta si resulta elegido.
- **softmax**: convierte una lista de numeros cualesquiera en probabilidades que suman 1.
  Exponencia cada uno y divide por la suma.
- **producto escalar**: multiplicar dos vectores componente a componente y sumar. Mide
  cuanto se parecen: cuanto mas alineados, mayor el numero.
- **cabeza** (head): una atencion independiente. El modelo tiene 8 en paralelo, cada una
  trabajando en 40 dimensiones.
- **mascara causal**: la que impide que la posicion 3 mire a la 4. Sin ella el modelo veria
  la respuesta.

    llmfs demo 06     entrena un modelo de atencion y dibuja a que mira cada letra
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# RoPE es del modulo 09; se da hecho para que MultiHeadAttention pueda usarlo. Si todavia
# no has llegado alli, ignoralo: los tests de este modulo pasan cos=None y sin=None.
from llmfs.reference import apply_rope


def causal_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """La mascara triangular que impide mirar hacia el futuro.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Una linea.

        return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()

    `tril` = *triangular lower*. Por defecto usa `diagonal=0`, que INCLUYE la diagonal, que es
    lo que quieres: un token si puede mirarse a si mismo.

    QUÉ TIENE QUE SALIR
    -------------------
    Para `seq_len = 4`, con el convenio `True = SI se puede mirar`:

        [[ True, False, False, False],     el token 0 solo se ve a si mismo
         [ True,  True, False, False],     el token 1 ve al 0 y a si mismo
         [ True,  True,  True, False],
         [ True,  True,  True,  True]]

    SI TE SALE MAL
    --------------
        - al reves        -> has usado `triu` en vez de `tril`
        - diagonal False  -> has pasado `diagonal=-1`

    POR QUÉ HACE FALTA
    ------------------
    Al entrenar le damos al modelo la frase entera de golpe y le pedimos que prediga cada token
    a partir de los anteriores. Sin una mascara, la posicion 3 podria mirar a la 4, que es
    literalmente la respuesta que tiene que dar.

    Ese es el bug mas caro del curso: la perdida baja espectacularmente, todo parece ir de
    maravilla, y el modelo no sirve para nada porque en generacion ese futuro no existe.

    UN AVISO PARA MÁS ADELANTE
    --------------------------
    Usamos `True = permitido` porque es el convenio de `F.scaled_dot_product_attention`. Pero
    `nn.MultiheadAttention` de PyTorch usa el CONTRARIO: su `attn_mask` booleana marca con
    `True` lo que hay que PROHIBIR. Por eso el test que compara contra ella pasa `~mask`. Es
    una inconsistencia real dentro de la propia libreria.

    Args:
        seq_len: la longitud de la secuencia.
        device: donde crear el tensor.

    Returns:
        Tensor booleano `(seq_len, seq_len)`.
    """
    raise NotImplementedError("TODO: modulo 06, ejercicio 1 - causal_mask")


def single_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Atencion de una cabeza. El corazon del Transformer, en cuatro lineas.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Cuatro pasos, y cada uno tiene una trampa.

        1. Coge la dimension y calcula las PUNTUACIONES:

               d_k = q.shape[-1]
               scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

        2. Si hay mascara, tapa lo prohibido:

               if mask is not None:
                   scores = scores.masked_fill(~mask, float("-inf"))

        3. Convierte en pesos que sumen 1:

               weights = F.softmax(scores, dim=-1)

        4. Mezcla los valores y devuelve las dos cosas:

               return weights @ v, weights

    QUÉ ESTÁ PASANDO EN CADA PASO
    -----------------------------
    **Paso 1.** `q @ k.transpose(-2,-1)` multiplica `(B, T, d_k)` por `(B, d_k, S)` y da
    `(B, T, S)`. La casilla `[b, i, j]` es el producto escalar de la query del token `i` con la
    key del token `j`: cuanto le interesa a `i` el token `j`.

    **Paso 2.** El `~` invierte el booleano: donde la mascara dice `False` (prohibido), pon
    `-inf`. Como `e^(-inf) = 0`, esas posiciones reciben peso exactamente cero.

    **Paso 3.** Softmax exponencia y normaliza, asi que cada fila acaba sumando 1.

    **Paso 4.** `weights @ v` es la media ponderada: cada token se lleva una mezcla de los
    valores, pesada por cuanto le interesa cada uno.

    LAS TRES TRAMPAS
    ----------------
    **`transpose(-2, -1)` con indices NEGATIVOS.** Cuentan desde el final, asi que funcionan
    igual con `(B, T, d)` que con `(B, heads, T, d)`. Si escribes `transpose(1, 2)`, este
    ejercicio pasa y el ejercicio 3 se rompe con un error de formas que cuesta relacionar con
    la causa.

    **`dim=-1` en el softmax.** Estas normalizando sobre A QUIEN se mira, de forma que cada
    fila sume 1. Con `dim=-2` normalizarias sobre quien mira, que no significa nada. Y no da
    error: las formas son identicas, el modelo entrena, y aprende peor. Hay un test que
    comprueba que cada fila suma 1.

    **La mascara va ANTES del softmax.** Si borraras los pesos despues, las filas dejarian de
    sumar 1 y estarias escalando la salida por un factor arbitrario distinto en cada posicion.

    POR QUÉ SE DIVIDE POR sqrt(d_k)
    -------------------------------
    El producto escalar de dos vectores de dimension `d_k` tiene varianza `d_k`. Sin dividir,
    con `d_k` grande las puntuaciones se disparan, y como softmax es exponencial, devuelve casi
    `[0,0,...,1,...,0]`: la atencion colapsa a elegir un solo token.

    Y el problema de verdad no es el forward, es el GRADIENTE: la derivada del softmax es
    `p(1-p)`, y con `p` pegado a 0 o a 1 vale practicamente cero. La capa deja de aprender. La
    demo del modulo lo mide.

    Args:
        q: `(B, T, d_k)` las preguntas.
        k: `(B, S, d_k)` las etiquetas.
        v: `(B, S, d_v)` los contenidos.
        mask: `(T, S)` o `(B, T, S)` booleana, `True` = permitido. `None` = sin mascara.

    Returns:
        `(salida, pesos)` con salida `(B, T, d_v)` y pesos `(B, T, S)`. Los pesos se devuelven
        porque son lo que dibuja el heatmap de la demo.
    """
    raise NotImplementedError("TODO: modulo 06, ejercicio 2 - single_head_attention")


class MultiHeadAttention(nn.Module):
    """Varias atenciones en paralelo, cada una con sus propias proyecciones.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    **En `__init__`:**

        1. Valida que `d_model` sea divisible por `n_heads`, y lanza `ValueError` si no.

        2. Guarda: `self.d_model`, `self.n_heads`, `self.head_dim = d_model // n_heads`,
           `self.dropout` y `self.use_sdpa`.

        3. Crea las cuatro proyecciones y los dos dropouts. Los nombres importan (el test copia
           pesos por nombre):

               self.q_proj = nn.Linear(d_model, d_model, bias=bias)
               self.k_proj = nn.Linear(d_model, d_model, bias=bias)
               self.v_proj = nn.Linear(d_model, d_model, bias=bias)
               self.out_proj = nn.Linear(d_model, d_model, bias=bias)
               self.attn_dropout = nn.Dropout(dropout)
               self.resid_dropout = nn.Dropout(dropout)

    **Dos ayudantes** (escribelos como metodos):

        def _split_heads(self, x):          # (B, T, d_model) -> (B, n_heads, T, head_dim)
            batch, seq_len, _ = x.shape
            return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        def _merge_heads(self, x):          # (B, n_heads, T, head_dim) -> (B, T, d_model)
            batch, _, seq_len, _ = x.shape
            return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    **En `forward`:**

        1. `seq_len = x.shape[1]`, y si `mask` es None, generala con `causal_mask`.

        2. Proyecta y parte en cabezas:

               q = self._split_heads(self.q_proj(x))
               k = self._split_heads(self.k_proj(x))
               v = self._split_heads(self.v_proj(x))

        3. Si `cos` y `sin` no son None, aplica RoPE a q y k (NUNCA a v):

               q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        4. La atencion. Si `self.use_sdpa` y no piden pesos:

               out = F.scaled_dot_product_attention(
                   q, k, v, attn_mask=mask,
                   dropout_p=self.dropout if self.training else 0.0,
               )
               weights = None

           Si no, el calculo explicito (el mismo del ejercicio 2, pero con 4 dimensiones):

               scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
               scores = scores.masked_fill(~mask, float("-inf"))
               weights = F.softmax(scores, dim=-1)
               out = self.attn_dropout(weights) @ v

        5. Junta las cabezas y proyecta:

               out = self.resid_dropout(self.out_proj(self._merge_heads(out)))
               return (out, weights) if return_weights else out

    CUATRO DETALLES QUE FALLAN SI NO LOS CUIDAS
    -------------------------------------------
    **El ORDEN del view en `_split_heads`.** Primero `view(B, T, n_heads, head_dim)` y LUEGO
    `transpose`. Si hicieras `view(B, n_heads, T, head_dim)` directamente estarias mezclando
    posiciones con cabezas: forma correcta, datos mal, cero errores. Hay un test que lo
    detecta comprobando que las cabezas no dan patrones identicos.

    **El `.contiguous()` en `_merge_heads`.** `transpose` no mueve datos, solo cambia como se
    recorren (los "strides"), y `view` exige memoria contigua. Sin el, PyTorch lanza un error
    que habla de strides y no dice claramente que hacer.

    **RoPE va DESPUES de partir en cabezas.** La rotacion depende de `head_dim`, no de
    `d_model`. Y solo a q y k: lo que debe depender de la posicion son las PUNTUACIONES, no el
    contenido que se transporta.

    **El `if self.training` del dropout de SDPA.** `F.scaled_dot_product_attention` no consulta
    el modo por su cuenta: si le pasas `dropout_p` fijo, aplicaria dropout tambien en
    evaluacion y tus muestras saldrian ruidosas y no reproducibles.

    POR QUÉ UNA PROYECCIÓN GRANDE Y NO OCHO PEQUEÑAS
    ------------------------------------------------
    `nn.Linear(320, 320)` seguido de un `view` es matematicamente identico a ocho
    `nn.Linear(320, 40)` cuyos resultados se concatenan. Pero es UN matmul grande en vez de
    ocho pequenyos, y como viste en el modulo 01, las matrices grandes aprovechan mucho mejor
    la GPU.

    forward(x, mask=None, cos=None, sin=None, return_weights=False):
        Args:
            x: `(B, T, d_model)`.
            mask: `(T, T)` booleana. Si es `None`, generala causal.
            cos, sin: tablas de RoPE (modulo 09), o `None`.
            return_weights: si `True`, devuelve `(salida, pesos)`.
        Returns:
            `(B, T, d_model)`, o la tupla si `return_weights`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_sdpa: bool = False,
    ) -> None:
        super().__init__()
        raise NotImplementedError("TODO: modulo 06, ejercicio 3 - MultiHeadAttention.__init__")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TODO: modulo 06, ejercicio 3 - MultiHeadAttention.forward")
