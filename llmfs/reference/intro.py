"""Referencia del modulo 00: un modelo de lenguaje de juguete, sin redes neuronales.

Aqui no hay torch, ni matrices, ni gradientes. Solo diccionarios y una division. El
objetivo es que veas con tus propios ojos que un modelo de lenguaje es, literalmente,
una cosa que dice "despues de esto, viene esto otro con esta probabilidad".

Todo lo que hagamos en los 17 modulos siguientes es la misma idea, con mejores formas de
estimar esas probabilidades.
"""

from __future__ import annotations

import random
from typing import Mapping, Sequence

#: Una tabla de conteos: para cada contexto, cuantas veces le siguio cada caracter.
#: Por ejemplo `{"a": {"b": 3, "c": 1}}` significa "tras la 'a' vino 'b' tres veces y
#: 'c' una vez".
CountTable = dict[str, dict[str, int]]


def build_count_table(text: str, context_size: int = 1) -> CountTable:
    """Recorre el texto y apunta que caracter sigue a cada contexto.

    No es un ejercicio: es el andamio para que puedas concentrarte en los tres que si
    lo son. Con `context_size=1` cuenta pares de caracteres; con 2, trios; etc.
    """
    table: CountTable = {}
    for i in range(len(text) - context_size):
        context = text[i : i + context_size]
        nxt = text[i + context_size]
        table.setdefault(context, {})
        table[context][nxt] = table[context].get(nxt, 0) + 1
    return table


def next_token_probs(counts: Mapping[str, int]) -> dict[str, float]:
    """Convierte conteos en probabilidades.

    Dividir cada conteo entre el total. Eso es todo. El resultado suma 1, que es la
    definicion de una distribucion de probabilidad.

    Ejemplo:
        {"b": 3, "c": 1}  ->  {"b": 0.75, "c": 0.25}
    """
    total = sum(counts.values())
    if total == 0:
        raise ValueError("no se puede normalizar una tabla de conteos vacia")
    return {token: count / total for token, count in counts.items()}


def sample_next_token(probs: Mapping[str, float], rng: random.Random | None = None) -> str:
    """Elige un token al azar, respetando sus probabilidades.

    El metodo es el de la ruleta: se saca un numero aleatorio entre 0 y 1, y se va
    acumulando probabilidad hasta pasarse. El token en el que te pasas es el elegido.

        probs = {"b": 0.75, "c": 0.25}
        r = 0.61  ->  acumulado tras "b" es 0.75 > 0.61  ->  sale "b"
        r = 0.92  ->  acumulado tras "b" es 0.75 < 0.92
                      acumulado tras "c" es 1.00 > 0.92  ->  sale "c"

    Se recorre `probs` en su orden de insercion, para que con la misma semilla siempre
    salga lo mismo.
    """
    rng = rng or random.Random()
    r = rng.random()
    acumulado = 0.0
    ultimo = ""
    for token, p in probs.items():
        acumulado += p
        ultimo = token
        if r < acumulado:
            return token
    # Solo se llega aqui por error de redondeo en coma flotante (acumulado = 0.9999...).
    return ultimo


def generate_naive(
    table: CountTable,
    start: str,
    length: int = 200,
    rng: random.Random | None = None,
) -> str:
    """Genera texto encadenando predicciones.

    El bucle es el mismo que usara tu GPT de 9M en el modulo 14:

        1. mirar el contexto actual
        2. obtener una distribucion sobre el siguiente token
        3. muestrear uno
        4. anyadirlo al texto y volver al paso 1

    A esto se le llama generacion **autorregresiva**: cada prediccion se convierte en
    parte de la entrada de la siguiente. Es la razon por la que generar es lento
    (no se puede paralelizar en el tiempo) y por la que un error temprano contamina
    todo lo que viene detras.

    Args:
        table: tabla de conteos de `build_count_table`.
        start: contexto inicial. Su longitud fija el `context_size`.
        length: cuantos caracteres generar en total, incluido `start`.
        rng: generador aleatorio, para reproducibilidad.

    Returns:
        El texto generado. Se corta antes si aparece un contexto nunca visto.
    """
    rng = rng or random.Random()
    context_size = len(start)
    salida = list(start)

    for _ in range(max(0, length - len(start))):
        contexto = "".join(salida[-context_size:])
        counts = table.get(contexto)
        if not counts:
            break  # contexto desconocido: el modelo no sabe seguir
        salida.append(sample_next_token(next_token_probs(counts), rng))

    return "".join(salida)
