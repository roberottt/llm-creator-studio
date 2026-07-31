"""Referencia del modulo 09: como se le dice al modelo en que posicion esta cada token.

La atencion es una suma ponderada, y una suma no tiene orden: sin informacion posicional,
"el perro muerde al hombre" y "el hombre muerde al perro" dan exactamente la misma salida.
"""

from __future__ import annotations

import math

import torch


def sinusoidal_embeddings(seq_len: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    """La tabla de senos y cosenos del paper original de 2017.

    $$PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{base^{2i/d}}\\right), \\qquad
      PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{base^{2i/d}}\\right)$$

    Cada par de dimensiones oscila a una frecuencia distinta: las primeras cambian rapido
    (distinguen posiciones vecinas), las ultimas cambian lentisimo (distinguen el principio
    del final de la secuencia). Juntas forman una firma unica por posicion, parecida a como
    un numero binario identifica un valor con bits que alternan a distinto ritmo.

    Se suman a los embeddings de token, no se concatenan.

    Returns:
        `(seq_len, d_model)`.
    """
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    # exp(-ln(base) * 2i/d) es numericamente mas estable que base ** (-2i/d)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(base) / d_model)
    )  # (d/2,)

    embeddings = torch.zeros(seq_len, d_model)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings


def rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precalcula las tablas de cosenos y senos de RoPE.

    RoPE no SUMA nada al embedding: ROTA pares de dimensiones un angulo proporcional a la
    posicion. El par `i` se rota un angulo `pos * theta^(-2i/d)`.

    Las frecuencias van de rapida a lenta igual que en las sinusoidales: los primeros pares
    giran deprisa y capturan posiciones relativas cortas, los ultimos giran lentisimo y
    capturan distancias largas.

    CONVENIO DE EMPAREJAMIENTO: se usa el de "mitades" (el de Llama y HuggingFace), no el
    de pares consecutivos del paper original. Es decir, la dimension `i` se empareja con
    la `i + head_dim/2`, no con la `i+1`. Las dos versiones son equivalentes salvo una
    permutacion de las dimensiones, que la red aprende sin enterarse, y la de mitades se
    implementa con operaciones vectoriales mucho mas limpias.

    Por eso las tablas devueltas tienen `head_dim` columnas y no `head_dim/2`: cada
    frecuencia aparece DUPLICADA, una vez para cada mitad. Asi `apply_rope` es una linea.

    Returns:
        `(cos, sin)`, ambos de forma `(max_seq_len, head_dim)`.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) tiene que ser par: RoPE rota pares.")

    # theta^(-2i/d) para i = 0, 1, ..., d/2-1
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)

    angles = torch.outer(positions, inv_freq)  # (T, head_dim/2)
    angles = torch.cat([angles, angles], dim=-1)  # (T, head_dim), duplicado por mitades

    cos, sin = angles.cos(), angles.sin()
    if device is not None:
        cos, sin = cos.to(device), sin.to(device)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """`[a, b]` -> `[-b, a]`, partiendo por la mitad la ultima dimension.

    Es la version vectorizada de rotar un vector 2-D 90 grados. Combinada con cos y sin
    reproduce la rotacion de cualquier angulo:

        x * cos + rotate_half(x) * sin

    que para un par `(x1, x2)` es exactamente

        x1' = x1*cos - x2*sin
        x2' = x2*cos + x1*sin

    o sea, la matriz de rotacion de toda la vida.
    """
    mitad = x.shape[-1] // 2
    x1, x2 = x[..., :mitad], x[..., mitad:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Aplica la rotacion posicional a Q o a K.

    Args:
        x: `(B, n_heads, T, head_dim)`.
        cos, sin: `(max_seq_len, head_dim)` de `rope_frequencies`. Se recortan a `T` y se
            emiten (broadcast) sobre batch y cabezas.

    Returns:
        Del mismo tamanyo que `x`.

    Por que esto funciona: el producto escalar de dos vectores rotados depende SOLO de la
    diferencia de angulos, es decir, de la distancia relativa entre las dos posiciones.

        <R(m)q, R(n)k> = <q, R(n-m)k>

    Asi, la puntuacion de atencion entre los tokens 5 y 3 es la misma que entre el 105 y el
    103. El modelo aprende "el token de dos posiciones atras", no "el token numero 3". Esa
    es la razon de que RoPE extrapole mejor a secuencias mas largas que las de
    entrenamiento, y es la propiedad que mide la demo del modulo.

    Se aplica solo a Q y K, nunca a V: lo que tiene que depender de la posicion son las
    PUNTUACIONES de atencion, no el contenido que se transporta.
    """
    seq_len = x.shape[-2]
    cos = cos[:seq_len].to(dtype=x.dtype, device=x.device)
    sin = sin[:seq_len].to(dtype=x.dtype, device=x.device)
    return x * cos + rotate_half(x) * sin


class LearnedPositionalEmbedding(torch.nn.Module):
    """Embeddings posicionales aprendidos, como en GPT-2. Para comparar en la demo.

    Una tabla `(max_seq_len, d_model)` que se entrena como cualquier otro parametro. Simple
    y funciona, pero tiene un techo duro: para posiciones mas alla de `max_seq_len` no hay
    fila que consultar, y el modelo no puede procesarlas de ninguna manera.
    """

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        return self.embedding(positions)
