"""Modulo 00 - Que es un LLM, en realidad.

CÓMO SE HACE ESTE MÓDULO
========================

1. Lee `TEORIA.md`. Son 10 minutos y sin eso estos ejercicios no tienen sentido.
2. Implementa las funciones de abajo, en orden. Cada una usa la anterior.
3. `llmfs check 00` para ver si van bien.
4. ¿Atascado? `llmfs hint 00 -e 1` (tres niveles, cada vez mas explicito).
5. ¿Sigues atascado? `SOLUCION.md` tiene el codigo completo. Copialo, mira como funciona,
   y despues vuelve y escribelo tu. No es hacer trampa.

QUÉ VAS A CONSTRUIR
===================

Un generador de texto que funciona. Sin torch, sin matrices, sin derivadas: diccionarios y
una division.

Las tres funciones encajan asi:

    build_count_table   (ya hecha)   texto -> tabla de conteos
            |
            v
    next_token_probs    (ejercicio 1) conteos -> probabilidades
            |
            v
    sample_next_token   (ejercicio 2) probabilidades -> UN caracter
            |
            v
    generate_naive      (ejercicio 3) todo lo anterior, en bucle -> texto

El ejercicio 3 es el interesante: ese bucle es EXACTAMENTE el mismo que ejecuta ChatGPT.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **token**: la unidad de texto que maneja el modelo. Aqui, un caracter.
- **contexto**: los caracteres anteriores que el modelo mira para decidir el siguiente.
- **distribucion de probabilidad**: una lista de numeros no negativos que suman 1.
- **muestrear**: elegir uno al azar respetando esas probabilidades.
"""

from __future__ import annotations

import random
from typing import Mapping

# La tabla de conteos ya esta hecha: no es lo que se aprende aqui. Recorre un texto y
# apunta que caracter siguio a cada caracter. Ejemplo con "banana":
#     build_count_table("banana") -> {'b': {'a': 1}, 'a': {'n': 2}, 'n': {'a': 2}}
from llmfs.reference import build_count_table


def next_token_probs(counts: Mapping[str, int]) -> dict[str, float]:
    """Convierte una tabla de conteos en probabilidades.

    QUE ES ESTO
        Tienes apuntado cuantas veces siguio cada caracter a un contexto. Eso son numeros
        enteros: 40, 25, 20, 15. Para poder muestrear necesitas probabilidades: numeros
        entre 0 y 1 que sumen exactamente 1.

    COMO SE HACE
        Dividir cada conteo entre la suma de todos. Ya esta, no hay mas.

            total = 40 + 25 + 20 + 15 = 100
            40/100 = 0.40   25/100 = 0.25   20/100 = 0.20   15/100 = 0.15

        A esto se le llama "normalizar", y lo vas a ver mil veces en el curso. La funcion
        softmax del modulo 06 hace exactamente esto mismo, solo que exponenciando antes
        para que funcione con numeros negativos.

    EJEMPLO CONCRETO
        >>> next_token_probs({"b": 3, "c": 1})
        {'b': 0.75, 'c': 0.25}

    Args:
        counts: `{caracter: veces_que_aparecio}`. Los conteos son enteros >= 0.

    Returns:
        `{caracter: probabilidad}` con las mismas claves y EN EL MISMO ORDEN. El orden
        importa: el ejercicio 2 recorre el diccionario y si cambias el orden, con la
        misma semilla saldra un texto distinto al de la referencia y el test fallara.

    Raises:
        ValueError: si el total es 0. No se puede repartir el 100% de nada entre nadie,
            y si no lo compruebas te llevas un ZeroDivisionError mucho mas adelante y en
            un sitio donde no vas a saber de donde viene.
    """
    raise NotImplementedError("TODO: modulo 00, ejercicio 1 - next_token_probs")


def sample_next_token(probs: Mapping[str, float], rng: random.Random | None = None) -> str:
    """Elige un caracter al azar, pero respetando sus probabilidades.

    QUE ES ESTO
        Tienes {'n': 0.40, 'r': 0.25, ' ': 0.20, 's': 0.15} y tienes que sacar UNO. No el
        mas probable siempre (eso genera texto repetitivo y con bucles), sino uno al azar
        de forma que 'n' salga el 40% de las veces si repitieras el proceso mil veces.

    COMO SE HACE: el metodo de la ruleta
        Imagina la recta del 0 al 1 partida en trozos, uno por caracter, de tamanyo
        proporcional a su probabilidad:

            |----'n'----|--'r'--|--' '--|-'s'-|
            0          0.40    0.65    0.85   1.0

        Sacas un numero aleatorio entre 0 y 1 y miras en que trozo cae.

        En codigo eso es: llevar un acumulador, ir sumandole la probabilidad de cada
        caracter, y devolver el primero con el que el acumulador SUPERE al numero
        aleatorio.

            r = 0.61
            'n' -> acumulado 0.40 ; 0.61 >= 0.40, sigo
            'r' -> acumulado 0.65 ; 0.61 <  0.65, DEVUELVO 'r'

    DETALLE QUE IMPORTA
        Recorre `probs` en su orden natural (`for token, p in probs.items()`), sin ordenar
        ni nada. La referencia hace lo mismo, y asi con la misma semilla ambos generan
        exactamente el mismo texto y el test los puede comparar.

        Y devuelve el ultimo caracter como red de seguridad al salir del bucle: por error
        de redondeo en coma flotante el acumulado puede quedarse en 0.9999999 y no llegar
        a superar un `r` de 0.99999995. Pasa poco, pero pasa, y devolver `None` ahi
        rompe la generacion entera.

    Args:
        probs: `{caracter: probabilidad}`, tal como lo devuelve el ejercicio 1.
        rng: generador aleatorio. Usa `rng.random()`, que da un float en [0, 1).
            Si es `None`, crea uno con `random.Random()`.

    Returns:
        Uno de los caracteres de `probs`.
    """
    raise NotImplementedError("TODO: modulo 00, ejercicio 2 - sample_next_token")


def generate_naive(
    table: dict[str, dict[str, int]],
    start: str,
    length: int = 200,
    rng: random.Random | None = None,
) -> str:
    """Genera texto encadenando predicciones. Aqui es donde aparece el modelo de lenguaje.

    QUE ES ESTO
        Ya sabes sacar una distribucion (ejercicio 1) y muestrear de ella (ejercicio 2).
        Generar texto es hacer eso en bucle, metiendo cada caracter que sales en la
        entrada del siguiente paso.

        Este bucle se llama generacion AUTORREGRESIVA y es EXACTAMENTE el mismo que vas a
        implementar en el modulo 14 con tu GPT de 9 millones de parametros. Lo unico que
        cambiara es de donde salen las probabilidades: aqui de una tabla de conteos, alli
        de una red neuronal.

    EL BUCLE
        salida = los caracteres de `start`

        repetir hasta tener `length` caracteres:
            1. contexto = los ultimos len(start) caracteres de la salida
            2. counts   = table.get(contexto)
            3. si no hay counts (contexto nunca visto): PARAR y devolver lo que haya
            4. probs    = next_token_probs(counts)
            5. elegido  = sample_next_token(probs, rng)
            6. anyadir `elegido` a la salida

        devolver "".join(salida)

    POR QUE EL PASO 3
        La tabla solo conoce los contextos que aparecieron en el texto de entrenamiento.
        Si generas un contexto que nunca se vio, no hay nada que consultar. Un modelo por
        conteo se queda mudo. Una red neuronal no tiene este problema, y esa es una de las
        razones por las que se usan redes: siempre dan una respuesta para cualquier
        entrada, aunque sea nueva.

    CUIDADO CON
        `length` es el total de caracteres devueltos, INCLUYENDO los de `start`. Si
        start="ab" y length=5, devuelves 5 caracteres, no 7.

    Args:
        table: `{contexto: {siguiente: veces}}`, de `build_count_table`.
        start: texto inicial. Su longitud define el tamanyo del contexto.
        length: longitud total del texto a devolver.
        rng: generador aleatorio, para que el test pueda reproducir el resultado.

    Returns:
        El texto generado, como una unica cadena.
    """
    raise NotImplementedError("TODO: modulo 00, ejercicio 3 - generate_naive")
