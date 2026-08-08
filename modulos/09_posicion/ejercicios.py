"""Modulo 09 - Informacion posicional y RoPE.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa en orden -> `llmfs check 09` -> `llmfs hint 09 -e N`
-> `SOLUCION.md` tiene el codigo completo.

QUÉ VAS A CONSTRUIR
===================

La forma de decirle al modelo en que posicion esta cada token:

    sinusoidal_embeddings  (ej. 1)  la tabla del paper de 2017 (historico)
    rope_frequencies       (ej. 2)  precalcular los angulos de rotacion
            |
            v
    apply_rope             (ej. 3)  rotar Q y K. UNA LINEA, pero solo despues del ej. 2

El ejercicio 2 es el que cuesta, y cuesta por un solo paso: el que duplica las frecuencias. El
3 es una linea, pero solo tiene sentido despues de entender el 2.

Ojo con una cosa: el ejercicio 1 NO lo usa nuestro modelo. Es la opcion de 2017 y esta aqui
porque introduce la escalera de frecuencias que RoPE reutiliza, porque te la vas a encontrar en
mucho codigo, y porque la demo la entrena para compararla con las otras dos. La seccion
"Ejercicio 1" de `TEORIA.md` lo explica.

Y una cara conocida: el `apply_rope` del ejercicio 3 ya lo llamaste en el modulo 06, dentro de
`MultiHeadAttention`, importado de la referencia con un comentario que decia que lo ignoraras de
momento. Hoy lo escribes tu, y las tablas `cos` y `sin` de aquella firma salen del ejercicio 2.

`TEORIA.md` esta ordenada igual que esta lista y cada docstring de aqui te dice que seccion le
toca.

EL PROBLEMA QUE RESUELVE
========================

Vuelve a mirar la formula de la atencion (modulo 06): es una suma ponderada, y una suma no
tiene orden. Para el mecanismo de atencion, "el perro muerde al hombre" y "el hombre muerde
al perro" producen exactamente lo mismo.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **embedding posicional**: la informacion que le dice al modelo donde esta cada token.
- **posicion absoluta / relativa**: "soy el token 7" frente a "estoy dos posiciones detras
  de aquel". La relativa generaliza mejor.
- **RoPE** (Rotary Position Embedding): en vez de SUMAR algo al vector, lo ROTA un angulo
  proporcional a la posicion.
- **head_dim**: la dimension de cada cabeza de atencion. En nuestro modelo, 40. RoPE trabaja
  sobre esto, no sobre las 320 de d_model.
- **extrapolar**: usar el modelo con secuencias mas largas que las que vio al entrenar.

    llmfs demo 09     dibuja las frecuencias y mide la extrapolacion de verdad
"""

from __future__ import annotations

import math

import torch


def sinusoidal_embeddings(seq_len: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    """La tabla de senos y cosenos del paper de 2017.

    Contexto en `TEORIA.md`: seccion "Ejercicio 1: senos y cosenos", que empieza explicando por
    que escribes algo que el modelo final no usa, y trae la tabla entera de 5x4 que tiene que
    devolver tu funcion, leida por columnas para que se vea la escalera de frecuencias.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Cinco lineas, sin bucles.

        1. Las posiciones, como columna:

               position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)   # (T, 1)

        2. Las frecuencias, una por cada PAR de dimensiones:

               div_term = torch.exp(
                   torch.arange(0, d_model, 2, dtype=torch.float32)
                   * (-math.log(base) / d_model)
               )                                                                    # (d/2,)

        3. La tabla vacia:

               embeddings = torch.zeros(seq_len, d_model)

        4. Rellena intercalando:

               embeddings[:, 0::2] = torch.sin(position * div_term)
               embeddings[:, 1::2] = torch.cos(position * div_term)

        5. Devuelvela.

    CÓMO FUNCIONA
    -------------
    `position * div_term` emite `(T,1)` por `(d/2,)` y da `(T, d/2)`: todos los angulos de
    todas las posiciones de golpe, sin bucles.

    `[:, 0::2]` significa "todas las filas, columnas desde 0 de dos en dos", o sea las pares.
    Y `[:, 1::2]` las impares. Es la forma de intercalar seno y coseno sin escribir un `for`.

    EL TRUCO DEL PASO 2, QUE MERECE LA PENA CONOCER
    -----------------------------------------------
    `exp(-log(base) * 2i/d)` es matematicamente identico a `base ** (-2i/d)`, pero mucho mas
    ESTABLE. Elevar 10000 a una potencia negativa grande pierde precision en coma flotante;
    hacerlo pasando por logaritmos, no.

    Regla general que te servira en otros sitios: si ves una potencia con exponente grande,
    `exp(log(...))` suele ser mejor.

    LA IDEA: UN CONTADOR BINARIO
    ----------------------------
    Al contar en binario, cada bit oscila a un ritmo distinto:

        0000    el bit de la derecha cambia en cada paso
        0001    el siguiente, cada dos
        0010    el siguiente, cada cuatro

    La combinacion de todos identifica un numero de forma unica. Aqui es lo mismo pero con
    ondas continuas: los primeros pares de dimensiones oscilan rapido (distinguen posiciones
    vecinas) y los ultimos lentisimo (distinguen el principio del final).

    Args:
        seq_len: cuantas posiciones generar.
        d_model: la dimension del modelo. Se asume par.
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

    Contexto en `TEORIA.md`: seccion "Ejercicio 2: las tablas de angulos". Si el paso 4 (el
    `cat` que duplica) te parece un error, alli estan las tablas completas para head_dim=4 con
    las columnas repetidas senyaladas, y el porque: con el convenio de mitades, las dos
    componentes de un par necesitan EL MISMO angulo.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Cinco pasos.

        1. Valida que `head_dim` sea PAR y lanza `ValueError` si no. (RoPE rota pares: con
           dimension impar sobraria una.)

        2. Las frecuencias inversas, una por par:

               inv_freq = 1.0 / (theta ** (
                   torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
               ))                                                        # (head_dim/2,)

        3. Todos los angulos de golpe:

               positions = torch.arange(max_seq_len, dtype=torch.float32)
               angles = torch.outer(positions, inv_freq)                 # (T, head_dim/2)

        4. DUPLICA por mitades:

               angles = torch.cat([angles, angles], dim=-1)              # (T, head_dim)

        5. Devuelve el coseno y el seno, movidos a `device` si se pide:

               cos, sin = angles.cos(), angles.sin()
               if device is not None:
                   cos, sin = cos.to(device), sin.to(device)
               return cos, sin

    `torch.outer(a, b)[i,j] = a[i] * b[j]`, que es justo lo que hace falta: todas las
    combinaciones posicion x frecuencia.

    EL PASO 4 ES EL QUE CONFUNDE, Y ÉSTE ES EL PORQUÉ
    -------------------------------------------------
    Hay dos formas de emparejar las dimensiones para rotarlas:

        - el paper original empareja CONSECUTIVAS:  (x0,x1), (x2,x3), ...
        - Llama y HuggingFace emparejan por MITADES: (x0, x_{d/2}), (x1, x_{d/2+1}), ...

    Usamos el de mitades. Con ese convenio, la dimension `i` y la `i + head_dim/2` forman un par
    y necesitan EL MISMO angulo. Por eso cada frecuencia aparece DOS veces y las tablas tienen
    `head_dim` columnas en vez de `head_dim/2`.

    Los dos convenios son equivalentes salvo una permutacion de las dimensiones, que la red
    aprende sin enterarse. El de mitades gano porque hace que el ejercicio 3 sea UNA LINEA sin
    reordenar nada.

    QUÉ ESTÁ PASANDO
    ----------------
    RoPE no SUMA nada al vector: lo ROTA. El par `i` en la posicion `pos` se rota un angulo
    `pos * theta^(-2i/head_dim)`.

    Las frecuencias van de rapida a lenta: los primeros pares giran deprisa y capturan
    relaciones cortas, los ultimos giran lentisimo y capturan distancias largas.

    Args:
        head_dim: la dimension de cada cabeza (40 en el modelo final). Tiene que ser par.
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

    Contexto en `TEORIA.md`: seccion "Ejercicio 3: rotar de verdad", con la comprobacion de que
    esa linea ES la matriz de rotacion y un ejemplo con head_dim=4 que puedes seguir a mano: el
    vector [1, 0, 0, 1] rotado en las posiciones 0, 1 y 2, con las dos parejas por separado y la
    norma sin cambiar.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    **Un ayudante** (ponlo como funcion aparte en el mismo fichero):

        def rotate_half(x):
            mitad = x.shape[-1] // 2
            x1, x2 = x[..., :mitad], x[..., mitad:]
            return torch.cat([-x2, x1], dim=-1)

    **Y la funcion**, tres lineas:

        seq_len = x.shape[-2]
        cos = cos[:seq_len].to(dtype=x.dtype, device=x.device)
        sin = sin[:seq_len].to(dtype=x.dtype, device=x.device)
        return x * cos + rotate_half(x) * sin

    DE DÓNDE SALE ESA ÚLTIMA LÍNEA
    ------------------------------
    Rotar un par `(x1, x2)` un angulo `t` es la matriz de rotacion de siempre:

        x1' = x1*cos(t) - x2*sin(t)
        x2' = x2*cos(t) + x1*sin(t)

    Y ahora comprueba que `x * cos + rotate_half(x) * sin` produce exactamente eso, sabiendo
    que `rotate_half([a, b]) = [-b, a]`:

        componente 1:  x1*cos + (-x2)*sin  =  x1*cos - x2*sin    OK
        componente 2:  x2*cos + ( x1)*sin  =  x2*cos + x1*sin    OK

    DOS DETALLES QUE FALLAN SI LOS SALTAS
    -------------------------------------
    **El recorte `cos[:seq_len]`.** Las tablas se precalculan hasta `max_seq_len` (512 en el
    modelo final) y tu secuencia casi nunca mide eso exacto. Sin recortar, el broadcast falla
    o —peor— acierta por casualidad con las formas equivocadas.

    **El `.to(dtype=x.dtype)`.** Bajo AMP las tablas estan en fp32 y `x` llega en fp16.
    Mezclarlos hace que PyTorch promocione, y acabas calculando en la precision que no querias.

    NO HACE FALTA NINGÚN `unsqueeze`
    --------------------------------
    `x` es `(B, n_heads, T, head_dim)` y `cos` es `(T, head_dim)`. El broadcast alinea desde la
    derecha y se encarga solo de las dos primeras dimensiones.

    POR QUÉ FUNCIONA ESTO (la propiedad que justifica RoPE)
    -------------------------------------------------------
    El producto escalar de dos vectores rotados depende SOLO de la diferencia de angulos:

        <R(m)q, R(n)k> = <q, R(n-m)k>

    O sea: la puntuacion de atencion entre los tokens 5 y 3 es la MISMA que entre el 105 y el
    103. El modelo aprende "el token de dos posiciones atras", no "el token numero 3". La demo
    lo comprueba con numeros y sale igual hasta el ultimo decimal.

    Y ademas rotar NO cambia la longitud del vector, cosa que sumar un embedding posicional si
    hace.

    Args:
        x: `(B, n_heads, T, head_dim)`, normalmente Q o K.
        cos, sin: `(max_seq_len, head_dim)`, de `rope_frequencies`.

    Returns:
        Del mismo tamanyo que `x`.
    """
    raise NotImplementedError("TODO: modulo 09, ejercicio 3 - apply_rope")
