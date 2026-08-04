"""Modulo 00 - Que es un LLM, en realidad.

CÓMO SE HACE ESTE MÓDULO
========================

1. Lee `THEORY.md`. Son 10 minutos y sin eso estos ejercicios no tienen sentido.
2. Implementa las funciones de abajo, en orden. Cada una usa la anterior.
3. `llmfs check 00` para ver si van bien.
4. ¿Atascado? `llmfs hint 00 -e 1` (tres niveles, cada vez mas explicito).
5. ¿Sigues atascado? `SOLUTION.md` tiene el codigo completo. Copialo, mira como funciona,
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

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Tres lineas.

        1. Suma todos los conteos:

               total = sum(counts.values())

        2. Si el total es 0, lanza `ValueError`. (Ver mas abajo por que.)

        3. Devuelve un diccionario con cada conteo dividido entre el total:

               return {token: conteo / total for token, conteo in counts.items()}

    EJEMPLO PARA COMPROBAR
    ----------------------
        entrada:  {"b": 3, "c": 1}
        total  :  3 + 1 = 4
        salida :  {"b": 0.75, "c": 0.25}

    QUÉ ESTÁS HACIENDO Y POR QUÉ
    ----------------------------
    Tienes apuntado cuantas veces siguio cada caracter a un contexto: 40, 25, 20, 15. Esos
    numeros no significan nada por si solos, porque dependen de lo largo que fuera el texto.

    Lo que necesitas es la PROPORCION, y esas proporciones tienen que sumar 1 porque algo
    tuvo que venir siempre.

    A esto se le llama NORMALIZAR y lo vas a ver mil veces en el curso. La funcion `softmax`
    del modulo 06 hace exactamente esto, solo que exponenciando antes para que funcione con
    numeros negativos.

    DOS DETALLES QUE IMPORTAN
    -------------------------
    **El ValueError.** Si la tabla viene vacia, `sum()` da 0 y la division revienta con
    `ZeroDivisionError`. El problema no es que reviente: es DONDE. Sin la comprobacion, el
    error salta dentro de una comprension de diccionario, tres niveles por debajo de la causa
    real, y el mensaje no menciona en ningun sitio que el problema es una tabla vacia.

    **El orden de las claves.** Devuelvelas en el MISMO orden en que llegaron. Si recorres
    `counts.items()` para construir el resultado, el orden se conserva solo (en python 3.7+
    los diccionarios lo mantienen). No las ordenes alfabeticamente: el ejercicio 2 recorre
    este diccionario y el orden cambia que caracter sale.

    Args:
        counts: `{caracter: veces_que_aparecio}`, con conteos enteros >= 0.

    Returns:
        `{caracter: probabilidad}`, con las mismas claves y en el mismo orden.

    Raises:
        ValueError: si el total es 0.
    """
    raise NotImplementedError("TODO: modulo 00, ejercicio 1 - next_token_probs")


def sample_next_token(probs: Mapping[str, float], rng: random.Random | None = None) -> str:
    """Elige un caracter al azar, respetando sus probabilidades.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Un bucle con un acumulador. Cinco pasos.

        1. Si `rng` es None, crea uno: `rng = rng or random.Random()`

        2. Saca un numero aleatorio entre 0 y 1:

               r = rng.random()

        3. Prepara un acumulador a 0.0 y una variable para recordar el ultimo token visto.

        4. Recorre `probs.items()`. En cada vuelta:
             - suma la probabilidad al acumulador
             - guarda este token como "el ultimo visto"
             - si `r < acumulado`, DEVUELVE este token

        5. Si el bucle termina sin devolver nada, devuelve el ultimo token visto.
           (Ver mas abajo por que hace falta.)

    EL MÉTODO, DIBUJADO
    -------------------
    Imagina la recta del 0 al 1 partida en trozos, uno por caracter, de tamanyo proporcional
    a su probabilidad:

        |----'n'----|--'r'--|--' '--|-'s'-|
        0          0.40    0.65    0.85   1.0

    Sacas un numero al azar y miras en que trozo cae. Con r = 0.61:

        'n' -> acumulado 0.40 ;  0.61 < 0.40 ? NO, sigo
        'r' -> acumulado 0.65 ;  0.61 < 0.65 ? SI, devuelvo 'r'

    POR QUÉ NO COGER SIEMPRE EL MÁS PROBABLE
    ----------------------------------------
    Porque el texto sale repetitivo y con bucles. Lo veras medido en el modulo 14: coger
    siempre el mas probable produce cosas como "the cat sat on the mat. the cat sat on the
    mat."

    TRES DETALLES QUE IMPORTAN
    --------------------------
    **Usa `<` y no `<=`.** Con `{a: 0.5, b: 0.5}` y `r = 0.5` exacto: con `<`, tras 'a' el
    acumulado es 0.5 y `0.5 < 0.5` es falso, asi que sigue y devuelve 'b'. Eso es lo
    correcto: 'a' ocupa el intervalo [0, 0.5) y 'b' ocupa [0.5, 1). Como `rng.random()`
    devuelve un numero en [0, 1) —el 1 nunca sale, el 0 si— ese reparto da exactamente 50/50.

    **El paso 5 no es paranoia.** Los floats no suman exacto: prueba `sum([0.1] * 10)` en un
    interprete y da 0.9999999999999999. Si `rng.random()` devuelve 0.99999999999999995, el
    bucle termina sin devolver nada y la funcion devuelve `None`, lo que rompe el ejercicio 3
    con un error incomprensible varios pasos despues.

    **Recorre `probs` en su orden natural**, sin ordenar. La referencia hace lo mismo, y asi
    con la misma semilla ambos generan exactamente el mismo texto y el test los puede comparar.

    Args:
        probs: `{caracter: probabilidad}`, tal como lo devuelve el ejercicio 1.
        rng: generador aleatorio. Usa `rng.random()`, que da un float en [0, 1). Si es `None`,
            crea uno con `random.Random()`.

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
    """Genera texto encadenando predicciones. Aqui aparece el modelo de lenguaje.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Un bucle que usa las dos funciones anteriores.

        1. Si `rng` es None, crea uno.

        2. Guarda el tamanyo del contexto y arranca la salida:

               context_size = len(start)
               salida = list(start)

        3. Repite `max(0, length - len(start))` veces:

             a. Coge los ultimos `context_size` caracteres de lo que llevas:

                    contexto = "".join(salida[-context_size:])

             b. Buscalo en la tabla:

                    counts = table.get(contexto)

             c. Si no hay nada (`if not counts`), sal del bucle con `break`.

             d. Si lo hay, convierte a probabilidades, muestrea, y anyade:

                    salida.append(sample_next_token(next_token_probs(counts), rng))

        4. Devuelve `"".join(salida)`.

    QUÉ ESTÁS CONSTRUYENDO
    ----------------------
    Este bucle se llama generacion AUTORREGRESIVA («auto» = a si mismo, «regresivo» = se
    realimenta): cada caracter que sacas se convierte en parte de la entrada del paso
    siguiente.

    Y es EXACTAMENTE el mismo bucle que vas a implementar en el modulo 14 con tu GPT de 9
    millones de parametros. Lo unico que cambiara es de donde salen las probabilidades: aqui
    de una tabla de conteos, alli de una red neuronal.

    POR QUÉ EL `break` DEL PASO 3c
    ------------------------------
    La tabla solo conoce los contextos que aparecieron en el texto de entrenamiento. Si
    generas uno que nunca se vio, no hay nada que consultar: un modelo por conteo se queda
    literalmente mudo.

    Una red neuronal NUNCA tiene ese problema, porque no consulta una tabla: calcula. Le des
    lo que le des, produce una distribucion. Puede ser mala, pero existe. Esa es una de las
    razones profundas por las que se usan redes.

    DOS DETALLES QUE IMPORTAN
    -------------------------
    **El `length` cuenta el total devuelto, INCLUYENDO `start`.** Si `start` tiene 2
    caracteres y piden 5, generas 3, no 5. Por eso el bucle itera `length - len(start)` veces.

    **Acumula en una lista y une al final** con `"".join()`. Hacer `salida = salida + c` con
    cadenas crea una cadena nueva en cada vuelta. Aqui da igual, pero es una costumbre que en
    el modulo 14 sale cara.

    Args:
        table: `{contexto: {siguiente: veces}}`, de `build_count_table`.
        start: el texto inicial. Su longitud define el tamanyo del contexto.
        length: la longitud TOTAL del texto a devolver, contando `start`.
        rng: generador aleatorio, para que el test pueda reproducir el resultado.

    Returns:
        El texto generado, como una unica cadena.
    """
    raise NotImplementedError("TODO: modulo 00, ejercicio 3 - generate_naive")
