"""Modulo 03 - Tokenizacion y BPE.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `THEORY.md` -> implementa en orden -> `llmfs check 03` -> `llmfs hint 03 -e N`
-> `SOLUTION.md` tiene el codigo completo.

El ejemplo de "aaabdaaabac" del THEORY.md, hecho paso a paso a mano, es EXACTAMENTE lo que
vas a programar. Tenlo delante.

QUÉ VAS A CONSTRUIR
===================

El tokenizador del modelo final. Cinco funciones que encajan asi:

    get_stats    (ej. 1)  contar que pares de vecinos se repiten mas
    merge        (ej. 2)  sustituir un par por un token nuevo
        |
        +--> train_bpe   (ej. 3)  repetir 1 y 2 hasta tener 4096 tokens
                 |
                 +--> bpe_encode  (ej. 4)  texto -> ids
                 +--> bpe_decode  (ej. 5)  ids -> texto

Los dos primeros son cortos y mecanicos. El tercero es el central. Los dos ultimos usan lo
que aprendio el tercero.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **token**: la unidad de texto que maneja el modelo. Con BPE, un trozo de palabra.
- **vocabulario**: cuantos tokens distintos existen. El nuestro tendra 4096.
- **merge**: fusionar dos tokens adyacentes en uno nuevo. Es la operacion de BPE.
- **pre-tokenizador**: la expresion regular que trocea el texto ANTES de contar pares, para
  que ningun merge cruce de una palabra a la siguiente.
- **bytes fallback**: trabajar sobre bytes (0-255) en vez de caracteres, para que no exista
  el "caracter desconocido".

    llmfs demo 03     entrena vocabularios de varios tamanyos y compara con tiktoken
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import regex

# El patron de pre-tokenizacion de GPT-4. No hay que entenderlo entero: lo que hace es
# partir el texto en palabras, numeros, signos y espacios, para que los merges no crucen
# fronteras que no tienen sentido (ver THEORY.md).
#
#     regex.findall(GPT4_SPLIT_PATTERN, "Hola, mundo!")
#     -> ['Hola', ',', ' mundo', '!']
from llmfs.reference import GPT4_SPLIT_PATTERN

Pair = tuple[int, int]
Merges = dict[Pair, int]
Vocab = dict[int, bytes]


def get_stats(ids: Sequence[int], counts: dict[Pair, int] | None = None) -> dict[Pair, int]:
    """Cuenta cuantas veces aparece cada par de numeros CONSECUTIVOS.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Tres lineas.

        1. Si `counts` es None, empieza con un diccionario vacio:

               counts = {} if counts is None else counts

        2. Recorre los pares de vecinos y suma uno a cada uno:

               for pair in zip(ids, ids[1:]):
                   counts[pair] = counts.get(pair, 0) + 1

        3. Devuelve `counts`.

    `zip(ids, ids[1:])` produce todos los pares de vecinos —(ids[0],ids[1]), (ids[1],ids[2])…—
    sin tener que manejar indices a mano.

    EJEMPLO PARA COMPROBAR
    ----------------------
        get_stats([97, 97, 97, 98])  ->  {(97, 97): 2, (97, 98): 1}

    Fijate en que el par (97,97) sale DOS veces: en las posiciones 0-1 y en las 1-2. Al
    CONTAR, los pares SE SOLAPAN. (Al FUSIONAR, en el ejercicio 2, no. Son cosas distintas y
    conviene tenerlo claro desde ya.)

    PARA QUÉ SIRVE EL PARÁMETRO `counts`
    ------------------------------------
    Para acumular sobre un diccionario que ya existe, sin tener que concatenar listas.
    `train_bpe` lo necesita porque procesa el texto partido en trozos y quiere la suma de
    todos, pero SIN contar los pares que cruzan de un trozo al siguiente:

        stats = {}
        for chunk in chunks:
            get_stats(chunk, stats)     # va sumando sobre el mismo diccionario

    Devuelve el diccionario ademas de mutarlo: asi vale para los dos usos.

    UN DETALLE DE PYTHON
    --------------------
    El valor por defecto es `None` y no `{}` a proposito. Un `{}` como valor por defecto se
    crea UNA VEZ al definir la funcion y se comparte entre todas las llamadas: es el clasico
    bug de los argumentos mutables.

    Args:
        ids: la secuencia de numeros.
        counts: diccionario donde acumular, o `None` para crear uno nuevo.

    Returns:
        `{(a, b): veces}`. Es el mismo diccionario que se paso en `counts`, si se paso alguno.
    """
    raise NotImplementedError("TODO: modulo 03, ejercicio 1 - get_stats")


def merge(ids: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    """Sustituye cada aparicion de un par por un unico numero nuevo.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Un `while` con un indice que controlas tu.

        out, i, n = [], 0, len(ids)

        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                out.append(new_id)
                i += 2                      # <- consume DOS posiciones
            else:
                out.append(ids[i])
                i += 1                      # <- consume UNA

        return out

    POR QUÉ UN `while` Y NO UN `for`
    --------------------------------
    Un `for` avanza de uno en uno SIEMPRE. Aqui necesitas saltar dos posiciones cuando hay
    coincidencia, y esa es exactamente la diferencia entre contar pares (que si solapan) y
    fusionarlos (que no).

    Compruebalo con `[1, 1, 1]` fusionando `(1,1)`:

        - encuentras el par en la posicion 0, lo sustituyes y SALTAS a la posicion 2
        - en la 2 solo queda un 1 suelto, sin pareja
        - resultado: [256, 1], NO [256, 256]

    EJEMPLO PARA COMPROBAR
    ----------------------
        merge([97, 97, 97, 98, 97, 97], (97, 97), 256)  ->  [256, 97, 98, 256]

    EL `i < n - 1`
    --------------
    Evita mirar `ids[i+1]` cuando estas en el ultimo elemento. Sin el, te llevas un
    `IndexError` en cuanto la lista acabe justo en el primer elemento del par.

    Args:
        ids: la secuencia original.
        pair: el par a fusionar.
        new_id: el numero que lo sustituye.

    Returns:
        Una lista NUEVA. No modifiques `ids`.
    """
    raise NotImplementedError("TODO: modulo 03, ejercicio 2 - merge")


def train_bpe(
    text: str,
    vocab_size: int,
    pattern: str | None = None,
    verbose: bool = False,
) -> tuple[Merges, Vocab]:
    """Entrena el tokenizador: aprende que pares fusionar y en que orden.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Es el ejemplo de "aaabdaaabac" del THEORY.md, en bucle. Seis pasos.

        1. Valida que `vocab_size >= 256` y lanza `ValueError` si no. (Son los bytes: no puede
           haber un vocabulario mas pequenyo.)

        2. Trocea el texto:

               chunks = [text] if pattern is None else regex.findall(pattern, text)

        3. Pasa cada trozo a bytes y de ahi a enteros 0-255:

               ids = [list(chunk.encode("utf-8")) for chunk in chunks if chunk]

        4. Arranca las dos estructuras de salida:

               merges = {}
               vocab = {i: bytes([i]) for i in range(256)}

        5. Repite `vocab_size - 256` veces, con `i` como contador:

             a. Cuenta los pares de TODOS los trozos sobre un mismo diccionario:

                    stats = {}
                    for chunk_ids in ids:
                        get_stats(chunk_ids, stats)

             b. Si `stats` esta vacio, `break`. (Ya no quedan pares que fusionar.)

             c. Elige el ganador:

                    pair = max(stats, key=lambda p: (stats[p], p))

             d. El id nuevo es `256 + i`.

             e. Aplica el merge a cada trozo:

                    ids = [merge(c, pair, new_id) for c in ids]

             f. Apunta lo aprendido:

                    merges[pair] = new_id
                    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]     # bytes + bytes

        6. Devuelve `(merges, vocab)`.

    EL DESEMPATE DEL PASO 5c
    ------------------------
    `max(stats, key=lambda p: (stats[p], p))` compara TUPLAS: primero la frecuencia y, si
    empata, el par. Python compara tuplas elemento a elemento, asi que eso hace exactamente lo
    que quieres.

    Cual gane un empate da igual para la calidad del tokenizador, pero tiene que ser
    determinista y tiene que ser EL MISMO criterio que la referencia. Si usaras
    `max(stats, key=stats.get)`, el ganador dependeria del orden de insercion del diccionario
    y tus merges divergirian de los del test en cuanto hubiera un empate.

    EL `break` DEL PASO 5b NO ES OPCIONAL
    -------------------------------------
    Si pides 4096 merges sobre un texto de 20 caracteres, llega un momento en que cada trozo
    queda reducido a un solo token y no hay pares que contar. Sin el `break`, `max()` sobre un
    diccionario vacio lanza `ValueError`. Hay un test que cubre ese caso.

    POR QUÉ SE CUENTA TROZO A TROZO
    -------------------------------
    Para que ningun merge pueda unir el final de una palabra con el principio de la siguiente.
    Si concatenaras los trozos, BPE aprenderia tokens como "gato.El".

    SOBRE EL RENDIMIENTO
    --------------------
    Esta implementacion recorre el corpus entero en cada merge. Para 4096 merges sobre 2 GB
    serian dias. Es una decision consciente: el codigo esta escrito para entenderse. Por eso
    el modulo 04 entrena los merges sobre una muestra y luego codifica el corpus completo.

    Args:
        text: el texto de entrenamiento.
        vocab_size: tamanyo final del vocabulario. >= 256.
        pattern: expresion regular de pre-tokenizacion, o `None` para no trocear.
        verbose: si `True`, imprime cada merge segun lo aprende.

    Returns:
        `(merges, vocab)`:
          - `merges`: `{(a, b): id_nuevo}` en el ORDEN en que se aprendieron (los dicts de
            python conservan el orden de insercion, asi que no hay que hacer nada especial).
          - `vocab`: `{id: bytes}` con los 256 bytes iniciales mas uno por cada merge.

    Raises:
        ValueError: si `vocab_size` es menor que 256.
    """
    raise NotImplementedError("TODO: modulo 03, ejercicio 3 - train_bpe")


def bpe_encode(text: str, merges: Merges, pattern: str | None = None) -> list[int]:
    """Convierte texto en ids, aplicando los merges aprendidos.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Una funcion auxiliar que codifica UN trozo, y un bucle que la aplica a todos.

    **La auxiliar** (puedes llamarla `_encode_chunk` y ponerla donde quieras del fichero):

        def _encode_chunk(ids, merges):
            while len(ids) >= 2:
                stats = get_stats(ids)
                pair = min(stats, key=lambda p: merges.get(p, float("inf")))
                if pair not in merges:
                    break
                ids = merge(ids, pair, merges[pair])
            return ids

    **Y la funcion principal:**

        1. Trocea el texto con el MISMO patron con el que entrenaste.
        2. Para cada trozo: `list(chunk.encode("utf-8"))` y pasalo por `_encode_chunk`.
        3. Concatena los ids de todos los trozos en una sola lista y devuelvela.

    EL `min` CON `float("inf")` ES EL CORAZÓN DEL EJERCICIO
    ------------------------------------------------------
    Los ids de merge son 256, 257, 258... en ORDEN de aprendizaje. Asi que "el que se aprendio
    antes" es lo mismo que "el que tiene el id mas bajo".

    `merges.get(p, float("inf"))` da infinito a los pares que no estan en `merges`, de forma
    que nunca ganan el `min`. Y si el ganador resulta no estar en `merges`, significa que
    NINGUNO de los pares presentes es fusionable: hay que parar.

    POR QUÉ IMPORTA TANTO EL ORDEN
    ------------------------------
    El tokenizador no es "parte el texto en los trozos mas largos posibles": es "reproduce
    exactamente el proceso de entrenamiento".

    Dos tokenizaciones distintas del mismo texto pueden ser ambas validas como secuencias de
    ids, pero solo una es la que el modelo vio millones de veces al entrenar. La otra le
    resulta tan extranya como texto en otro idioma.

    UNA CONSECUENCIA QUE SORPRENDE
    ------------------------------
    Con merges `(a,a)->256` y `(256,a)->257`, la cadena "aaaa" da `[256, 256]`, NO `[257, a]`.

    El primer merge se aplica a TODA la secuencia de golpe y se lleva las cuatro 'a' de dos en
    dos, asi que el par `(256, a)` nunca llega a formarse. Con tres 'a' si sale `[257]`. No es
    un bug: es como funciona BPE, y lo mismo pasa en tiktoken. Hay un test que lo documenta.

    Args:
        text: el texto a codificar.
        merges: lo que devolvio `train_bpe`.
        pattern: el MISMO patron que usaste al entrenar. Si entrenaste con patron y codificas
            sin el (o al reves), los resultados no cuadran.

    Returns:
        La lista de ids.
    """
    raise NotImplementedError("TODO: modulo 03, ejercicio 4 - bpe_encode")


def bpe_decode(ids: Iterable[int], vocab: Vocab) -> str:
    """Convierte una lista de ids de vuelta en texto.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Dos lineas. Y el orden de esas dos lineas es todo el ejercicio.

        raw = b"".join(vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    PRIMERO JUNTAR, DESPUÉS DECODIFICAR
    -----------------------------------
    NO hagas esto:

        "".join(vocab[i].decode("utf-8") for i in ids)      # MAL

    Motivo: UTF-8 codifica los caracteres no-ASCII en varios bytes. Una 'n' es un byte, pero
    una 'ñ' son dos: 0xC3 0xB1.

    A BPE eso le da completamente igual —trabaja con bytes y no sabe nada de caracteres— asi
    que puede perfectamente haber aprendido un token que ACABA en 0xC3 y otro que EMPIEZA por
    0xB1. Decodificados por separado, ninguno de los dos es UTF-8 valido. Juntos, son una 'ñ'.

    Hay un test que construye exactamente ese caso.

    POR QUÉ `errors="replace"`
    --------------------------
    Es lo que se llama BYTES FALLBACK. Un modelo a medio entrenar genera secuencias de ids
    cualesquiera, y muchas no forman UTF-8 valido. Con `errors="replace"` sale un caracter de
    reemplazo donde no se pudo decodificar y la generacion continua; sin el, una excepcion
    tumbaria el bucle de generacion entero por un byte suelto.

    Cuando en el modulo 14 veas caracteres raros en las primeras muestras, ya sabes que son.

    Args:
        ids: los ids a decodificar.
        vocab: lo que devolvio `train_bpe`.

    Returns:
        El texto.
    """
    raise NotImplementedError("TODO: modulo 03, ejercicio 5 - bpe_decode")
