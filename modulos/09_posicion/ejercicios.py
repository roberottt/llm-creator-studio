"""Modulo 09 - Informacion posicional y RoPE.

Tres ejercicios. El tercero es una linea, pero solo despues de haber entendido el segundo.

    llmfs check 09
    llmfs demo 09     dibuja las frecuencias y comprueba la invariancia relativa con numeros
"""

from __future__ import annotations

import math

import torch


def sinusoidal_embeddings(seq_len: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    """La tabla de senos y cosenos del paper original de 2017.

    QUE ES ESTO
        Una tabla (seq_len, d_model) que se SUMA a los embeddings de token para decirle al
        modelo en que posicion esta cada uno. No tiene parametros: es una formula fija.

    LA INTUICION: un contador binario
        Al contar en binario, cada bit oscila a un ritmo distinto:

            0000   el bit de la derecha cambia cada paso
            0001   el siguiente, cada dos
            0010   el siguiente, cada cuatro
            0011   ...

        La combinacion de todos identifica un numero de forma unica. Aqui es lo mismo pero
        con ondas continuas: los primeros pares de dimensiones oscilan rapido (distinguen
        posiciones vecinas) y los ultimos lentisimo (distinguen el principio del final).

    LA FORMULA
        PE[pos, 2i]   = sin( pos / base^(2i/d) )
        PE[pos, 2i+1] = cos( pos / base^(2i/d) )

        Las dimensiones PARES llevan seno y las IMPARES coseno, con la misma frecuencia
        cada par.

    COMO SE HACE SIN BUCLES
        1. `position = torch.arange(seq_len).unsqueeze(1)`  -> forma (T, 1)
        2. Las frecuencias, una por cada PAR de dimensiones (o sea d_model/2 de ellas):

               div_term = torch.exp(
                   torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(base) / d_model)
               )

           Ojo con esto: es matematicamente igual a `base ** (-2i/d)` pero mucho mas estable
           numericamente. Elevar 10000 a una potencia negativa grande pierde precision;
           hacerlo como `exp(-ln(base) * 2i/d)` no. Es el truco estandar y merece la pena
           conocerlo.

        3. `position * div_term` emite (T,1) x (d/2,) -> (T, d/2): todos los angulos.
        4. Rellenar el resultado:
               embeddings[:, 0::2] = torch.sin(angulos)     <- dimensiones pares
               embeddings[:, 1::2] = torch.cos(angulos)     <- dimensiones impares

           El `0::2` significa "desde 0, de dos en dos". Es la forma de intercalar sin
           bucles.

    Args:
        seq_len: cuantas posiciones generar.
        d_model: dimension del modelo. Se asume par.
        base: la constante 10000 del paper.

    Returns:
        Tensor `(seq_len, d_model)`.
    """
    raise NotImplementedError("TODO: modulo 09, ejercicio 1 - sinusoidal_embeddings")


def rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precalcula las tablas de cosenos y senos que usara RoPE.

    QUE ES ESTO
        RoPE no SUMA nada: ROTA pares de dimensiones un angulo proporcional a la posicion.
        Esta funcion calcula, de una vez y para todas las posiciones, el coseno y el seno
        de esos angulos.

        El par `i` en la posicion `pos` se rota un angulo:

            angulo = pos * theta^(-2i/head_dim)

        Igual que en las sinusoidales, las frecuencias van de rapida a lenta: los primeros
        pares giran deprisa y capturan relaciones cortas, los ultimos giran lentisimo y
        capturan distancias largas.

    LOS PASOS
        1. Validar que head_dim sea PAR y lanzar ValueError si no: RoPE rota pares, con
           dimension impar sobraria una.

        2. Las frecuencias inversas, una por par:

               inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

           Eso da head_dim/2 valores.

        3. Todos los angulos de golpe:

               positions = torch.arange(max_seq_len, dtype=torch.float32)
               angles = torch.outer(positions, inv_freq)      # (max_seq_len, head_dim/2)

           `torch.outer(a, b)[i,j] = a[i] * b[j]`. Es justo lo que hace falta.

        4. DUPLICAR por mitades:

               angles = torch.cat([angles, angles], dim=-1)   # (max_seq_len, head_dim)

           Este paso es el que mas confunde, asi que lee lo siguiente.

        5. Devolver `(angles.cos(), angles.sin())`, movidos a `device` si se pide.

    POR QUE SE DUPLICA (el convenio de MITADES)
        Hay dos formas de emparejar las dimensiones para rotarlas:

          - el paper original empareja consecutivas: (x0,x1), (x2,x3), ...
          - Llama y HuggingFace emparejan por mitades: (x0, x_{d/2}), (x1, x_{d/2+1}), ...

        Son EQUIVALENTES salvo una permutacion de las dimensiones, que la red aprende sin
        enterarse. Usamos el de mitades porque se implementa con operaciones vectoriales
        mucho mas limpias (lo veras en el ejercicio 3).

        Con ese convenio, la dimension `i` y la `i + head_dim/2` forman un par y necesitan
        EL MISMO angulo. Por eso las tablas tienen `head_dim` columnas en vez de
        `head_dim/2`: cada frecuencia aparece dos veces, una para cada mitad. Asi el
        ejercicio 3 es una sola linea sin reordenar nada.

    Args:
        head_dim: dimension de cada cabeza (40 en el modelo final). Tiene que ser par.
        max_seq_len: hasta que posicion precalcular.
        theta: la base, 10000 por defecto.
        device: donde dejar los tensores.

    Returns:
        `(cos, sin)`, ambos de forma `(max_seq_len, head_dim)`.

    Raises:
        ValueError: si `head_dim` es impar.
    """
    raise NotImplementedError("TODO: modulo 09, ejercicio 2 - rope_frequencies")


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Aplica la rotacion posicional a Q o a K.

    QUE ES ESTO
        Con las tablas del ejercicio 2 ya calculadas, esto es UNA LINEA. Pero conviene
        entender de donde sale.

    LA ROTACION, PRIMERO EN 2D
        Rotar un vector (x1, x2) un angulo t:

            x1' = x1*cos(t) - x2*sin(t)
            x2' = x2*cos(t) + x1*sin(t)

        Esa es la matriz de rotacion de toda la vida.

    LA VERSION VECTORIZADA
        Fijate en que las dos lineas de arriba se pueden escribir de golpe como:

            x' = x * cos + rotate_half(x) * sin

        donde `rotate_half([a, b]) = [-b, a]` partiendo por la mitad la ultima dimension.
        Comprueba que sale lo mismo:

            componente 1:  x1*cos + (-x2)*sin  = x1*cos - x2*sin    OK
            componente 2:  x2*cos + ( x1)*sin  = x2*cos + x1*sin    OK

        Y `rotate_half` es facil de escribir sin bucles:

            mitad = x.shape[-1] // 2
            x1, x2 = x[..., :mitad], x[..., mitad:]
            return torch.cat([-x2, x1], dim=-1)

        (Puedes escribir `rotate_half` como funcion auxiliar en el mismo fichero, o
        inlinearla. Lo que te resulte mas legible.)

    LOS DETALLES QUE HAY QUE CUIDAR
        - `cos` y `sin` vienen con `max_seq_len` filas y tu secuencia puede ser mas corta.
          RECORTALAS: `cos[:seq_len]`, con `seq_len = x.shape[-2]`.
        - Conviertelas al dtype y al device de `x`: `.to(dtype=x.dtype, device=x.device)`.
          Sin eso, con AMP en fp16 mezclarias tipos y torch se quejara (o peor, no).
        - Las formas: `x` es `(B, n_heads, T, head_dim)` y `cos` es `(T, head_dim)`. El
          broadcast se encarga solo de las dos primeras dimensiones; no hace falta
          `unsqueeze` de nada.

    POR QUE FUNCIONA (la propiedad que justifica RoPE)
        El producto escalar de dos vectores rotados depende SOLO de la diferencia de
        angulos:

            <R(m)q, R(n)k> = <q, R(n-m)k>

        O sea: la puntuacion de atencion entre los tokens 5 y 3 es la MISMA que entre el
        105 y el 103. El modelo aprende "el token de dos posiciones atras", no "el token
        numero 3". La demo lo comprueba con numeros.

        Y ademas rotar no cambia la longitud del vector, cosa que sumar un embedding
        posicional si hace.

    Args:
        x: `(B, n_heads, T, head_dim)`, normalmente Q o K.
        cos, sin: `(max_seq_len, head_dim)` de `rope_frequencies`.

    Returns:
        Del mismo tamanyo que `x`.
    """
    raise NotImplementedError("TODO: modulo 09, ejercicio 3 - apply_rope")
