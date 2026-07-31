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

    QUE ES ESTO
        Al entrenar le damos al modelo la frase entera de golpe y le pedimos que prediga
        cada token a partir de los anteriores. Sin una mascara, la posicion 3 podria mirar
        a la 4, que es literalmente la respuesta que tiene que dar.

    CONVENIO: True = SI se puede mirar
        Es el mismo que usa `F.scaled_dot_product_attention` con `attn_mask` booleana, y
        el que espera el resto del curso.

        Para seq_len=4:

            [[ True, False, False, False],   el token 0 solo se ve a si mismo
             [ True,  True, False, False],   el 1 ve al 0 y a si mismo
             [ True,  True,  True, False],
             [ True,  True,  True,  True]]

        La diagonal va INCLUIDA: un token si puede mirarse a si mismo.

    COMO
        Una matriz de unos y `.tril()` (triangular inferior). Una linea.
        Cuidado con `.tril()` frente a `.triu()`, y con el argumento `diagonal`: por
        defecto es 0, que incluye la diagonal, que es lo que quieres.

    Args:
        seq_len: longitud de la secuencia.
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

    LA FORMULA
        salida = softmax( Q K^T / sqrt(d_k) + mascara ) V

    PASO A PASO

        1. PUNTUACIONES.  scores = q @ k.transpose(-2, -1)
           Multiplicar (B, T, d_k) por (B, d_k, S) da (B, T, S). La casilla [b, i, j] es
           el producto escalar de la query del token i con la key del token j: cuanto le
           interesa a i el token j.

           El `.transpose(-2, -1)` usa indices NEGATIVOS a proposito: asi funciona igual
           con tensores de 3 dimensiones (B, T, d) que de 4 (B, heads, T, d). Si escribes
           `.transpose(1, 2)` se te rompera en el ejercicio 3.

        2. ESCALADO.  scores = scores / sqrt(d_k)
           donde d_k = q.shape[-1].

           Por que: el producto escalar de dos vectores de dimension d_k tiene varianza
           d_k. Sin dividir, con d_k grande las puntuaciones se disparan, el softmax se
           satura (devuelve casi [0,0,...,1,...,0]) y su derivada p(1-p) se va a cero. La
           capa deja de aprender. La demo te lo ensenya midiendolo.

        3. MASCARA.  scores = scores.masked_fill(~mask, float("-inf"))
           El `~` invierte el booleano: donde la mascara dice False (prohibido), pon -inf.
           Como e^(-inf) = 0, esas posiciones reciben peso exactamente cero.

           TIENE que ir ANTES del softmax. Si borraras los pesos despues, las filas ya no
           sumarian 1.

        4. SOFTMAX Y MEZCLA.
           weights = F.softmax(scores, dim=-1)      <- dim=-1, sobre la ultima dimension
           salida  = weights @ v

           El `dim=-1` es crucial: normalizas sobre las posiciones a las que se MIRA (cada
           fila suma 1). Con `dim=-2` normalizarias sobre las que miran, que no significa
           nada y es un bug clasico que no da error.

    Args:
        q: `(B, T, d_k)` las preguntas.
        k: `(B, S, d_k)` las etiquetas.
        v: `(B, S, d_v)` los contenidos.
        mask: `(T, S)` o `(B, T, S)` booleana, `True` = permitido. `None` = sin mascara.

    Returns:
        `(salida, pesos)` con salida `(B, T, d_v)` y pesos `(B, T, S)`.
        Los pesos se devuelven porque son lo que dibuja el heatmap de la demo.
    """
    raise NotImplementedError("TODO: modulo 06, ejercicio 2 - single_head_attention")


class MultiHeadAttention(nn.Module):
    """Varias atenciones en paralelo, cada una con sus propias proyecciones.

    QUE ES ESTO
        Una sola cabeza tiene que resolver todas las relaciones de la frase con un unico
        patron de atencion. Con varias, cada una puede especializarse: en modelos
        entrenados se han encontrado cabezas que miran al token anterior, cabezas que
        emparejan comillas de apertura y cierre, y las llamadas induction heads, que
        detectan el patron "... A B ... A" y predicen B. Nadie las programo.

    NO CUESTA MAS
        Con d_model=320 y 8 cabezas, cada una trabaja en head_dim = 320/8 = 40
        dimensiones. En vez de una atencion de 320 haces ocho de 40, y el total de
        parametros es identico.

    EL TRUCO DE IMPLEMENTACION
        NO se hacen 8 proyecciones separadas. Se hace UNA proyeccion de d_model -> d_model
        y se parte el resultado en 8 trozos. Es matematicamente equivalente y mucho mas
        rapido: un matmul grande en vez de ocho pequenyos.

        Partir:  (B, T, d_model) -> (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
                 con .view(B, T, n_heads, head_dim).transpose(1, 2)

        Juntar:  el camino inverso, con .transpose(1, 2).contiguous().view(B, T, d_model)

        El `.contiguous()` no es opcional: `transpose` no mueve datos, solo cambia como se
        recorren (los "strides"), y `.view()` exige memoria contigua. Sin el, PyTorch
        lanza un error que menciona strides y no dice claramente que hacer.

    SUBMODULOS (respeta los nombres, los tests copian pesos por nombre)
        q_proj, k_proj, v_proj, out_proj:  nn.Linear(d_model, d_model, bias=bias)
        attn_dropout, resid_dropout:       nn.Dropout(dropout)

    __init__(self, d_model, n_heads, dropout=0.0, bias=False, use_sdpa=False)
        Guarda d_model, n_heads, head_dim = d_model // n_heads, dropout y use_sdpa.
        Valida que d_model sea divisible por n_heads y lanza ValueError si no.

    forward(self, x, mask=None, cos=None, sin=None, return_weights=False)
        Args:
            x: `(B, T, d_model)`.
            mask: `(T, T)` booleana. Si es `None`, generala con `causal_mask`.
            cos, sin: tablas de RoPE (modulo 09). Si NO son None, aplica
                `apply_rope(q, cos, sin)` y lo mismo a k, DESPUES de partir en cabezas.
                Solo a q y k, NUNCA a v.
            return_weights: si `True`, devuelve `(salida, pesos)` en vez de solo la salida.

        Los pasos:
            1. proyectar x con q_proj, k_proj, v_proj
            2. partir cada uno en cabezas -> (B, n_heads, T, head_dim)
            3. si cos/sin: aplicar RoPE a q y k
            4. atencion (la misma formula del ejercicio 2, pero ahora con 4 dimensiones;
               por eso el transpose del ejercicio 2 tiene que usar indices negativos)
            5. juntar las cabezas -> (B, T, d_model)
            6. out_proj y resid_dropout

        Si `self.use_sdpa` es True y no piden pesos, usa
        `F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=...)`
        en lugar del calculo explicito. Da el mismo resultado sin materializar la matriz
        T x T. Se activa en el modulo 12, donde se mide cuanto gana.
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
