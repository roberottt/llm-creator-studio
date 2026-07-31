"""Modulo 03 - Tokenizacion y BPE.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa en orden -> `llmfs check 03` -> `llmfs hint 03 -e N`
-> `SOLUCION.md` tiene el codigo completo.

El ejemplo de "aaabdaaabac" del TEORIA.md, hecho paso a paso a mano, es EXACTAMENTE lo que
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
# fronteras que no tienen sentido (ver TEORIA.md).
#
#     regex.findall(GPT4_SPLIT_PATTERN, "Hola, mundo!")
#     -> ['Hola', ',', ' mundo', '!']
from llmfs.reference import GPT4_SPLIT_PATTERN

Pair = tuple[int, int]
Merges = dict[Pair, int]
Vocab = dict[int, bytes]


def get_stats(ids: Sequence[int], counts: dict[Pair, int] | None = None) -> dict[Pair, int]:
    """Cuenta cuantas veces aparece cada par de numeros CONSECUTIVOS.

    QUE ES ESTO
        El primer paso de BPE es saber que pareja de vecinos se repite mas. Esta funcion
        recorre la lista mirando de dos en dos y lleva la cuenta.

    EJEMPLO CONCRETO
        >>> get_stats([97, 97, 97, 98])
        {(97, 97): 2, (97, 98): 1}

        Fijate en que los pares SE SOLAPAN. En [97, 97, 97] el par (97,97) sale 2 veces:
        en las posiciones 0-1 y en las 1-2. No se saltan posiciones al contar (eso solo
        pasa al fusionar, en el ejercicio 2).

    EL PARAMETRO `counts`
        Sirve para acumular las cuentas de varios trozos sin tener que concatenarlos.
        `train_bpe` procesa el texto partido en chunks por el pre-tokenizador y necesita
        la suma de todos, pero sin contar los pares que cruzan de un chunk al siguiente.

            stats = {}
            for chunk in chunks:
                get_stats(chunk, stats)     # va sumando sobre el mismo diccionario

        Si es `None`, empiezas con un diccionario vacio.

    Args:
        ids: la secuencia de numeros.
        counts: diccionario donde acumular, o `None` para crear uno nuevo.

    Returns:
        `{(a, b): veces}`. Devuelve el mismo diccionario que se paso en `counts`, si se
        paso alguno.
    """
    raise NotImplementedError("TODO: modulo 03, ejercicio 1 - get_stats")


def merge(ids: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    """Sustituye cada aparicion de `pair` por un unico numero nuevo.

    QUE ES ESTO
        Ya sabes cual es el par mas frecuente. Ahora hay que reemplazarlo en toda la lista
        por el id nuevo. Esto es lo que acorta la secuencia y crea el token nuevo.

    EJEMPLO CONCRETO
        >>> merge([97, 97, 97, 98, 97, 97], (97, 97), 256)
        [256, 97, 98, 256]

    OJO CON EL SOLAPAMIENTO
        Al fusionar, las apariciones se consumen de izquierda a derecha y SIN solapar. En
        [97, 97, 97]:
            - encuentras (97,97) en la posicion 0, lo sustituyes y SALTAS DOS posiciones
            - te quedas en la posicion 2, donde solo hay un 97 suelto
            - resultado: [256, 97], no [256, 256]

        Este es el motivo de que un bucle `for` no valga bien aqui: necesitas controlar
        tu el indice para poder avanzar de dos en dos cuando hay coincidencia y de uno en
        uno cuando no. Un `while` con un indice manual es la forma natural.

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

    QUE ES ESTO
        El ejercicio central del modulo. Repites "cuenta los pares, fusiona el mas
        frecuente" hasta llegar al tamanyo de vocabulario que quieres. Es el ejemplo de
        `aaabdaaabac` del TEORIA.md, generalizado.

    EL ALGORITMO
        1. Trocear el texto:
             - si `pattern` es None -> un unico trozo con todo el texto
             - si no -> `regex.findall(pattern, text)`
        2. Pasar cada trozo a bytes UTF-8 y de ahi a una lista de enteros 0-255:
             `list(chunk.encode("utf-8"))`
        3. Arrancar el vocabulario con los 256 bytes: `{i: bytes([i]) for i in range(256)}`
        4. Repetir `vocab_size - 256` veces:
             a. contar los pares de TODOS los trozos sobre un mismo diccionario
             b. si no queda ningun par, salir del bucle (`break`)
             c. elegir el par ganador
             d. el id nuevo es 256 + numero_de_merge (0, 1, 2...)
             e. aplicar `merge` a cada trozo
             f. apuntar `merges[par] = id_nuevo`
             g. apuntar `vocab[id_nuevo] = vocab[par[0]] + vocab[par[1]]`  (concatenar bytes)

    COMO SE ELIGE EL GANADOR (importa para que el test pase)
        El mas frecuente. Y si hay EMPATE, gana el par mayor comparado como tupla:

            pair = max(stats, key=lambda p: (stats[p], p))

        Cual gane en un empate da igual para la calidad del tokenizador, pero tiene que ser
        determinista y tiene que ser el MISMO criterio que usa la referencia, o tus merges
        y los suyos divergiran en cuanto haya un empate y el test fallara con una
        diferencia dificil de interpretar.

    POR QUE SE CUENTA CHUNK A CHUNK Y NO SOBRE TODO JUNTO
        Para que un merge no pueda unir el final de una palabra con el principio de la
        siguiente. Si concatenaras los trozos, BPE aprenderia tokens como "gato.El".

    Args:
        text: el texto de entrenamiento.
        vocab_size: tamanyo final del vocabulario. Tiene que ser >= 256.
        pattern: expresion regular de pre-tokenizacion, o `None` para no trocear.
        verbose: si `True`, imprime cada merge segun lo aprende. Util para ver el proceso.

    Returns:
        `(merges, vocab)`:
          - `merges`: `{(a, b): id_nuevo}` en el ORDEN en que se aprendieron. Los dicts de
            python conservan el orden de insercion, asi que no hay que hacer nada especial.
          - `vocab`: `{id: bytes}` con los 256 bytes iniciales mas uno por cada merge.

    Raises:
        ValueError: si `vocab_size` es menor que 256. No se puede tener un vocabulario mas
            pequenyo que el numero de bytes que existen.
    """
    raise NotImplementedError("TODO: modulo 03, ejercicio 3 - train_bpe")


def bpe_encode(text: str, merges: Merges, pattern: str | None = None) -> list[int]:
    """Convierte texto en una lista de ids, aplicando los merges aprendidos.

    QUE ES ESTO
        Ya tienes los merges. Ahora hay que aplicarlos a texto nuevo. El detalle que lo
        hace no-trivial: hay que aplicarlos EN EL ORDEN EN QUE SE APRENDIERON, no en el
        orden en que aparecen en este texto concreto.

    POR QUE EL ORDEN
        Imagina que aprendiste primero `("a","a") -> 256` y despues `(256,"a") -> 257`.
        Si al codificar aplicaras el segundo antes que el primero, el 256 no existiria
        todavia y el resultado seria distinto. La tokenizacion saldria valida pero
        DIFERENTE de la que vio el modelo al entrenar, y el modelo no la entenderia.

    COMO SE CONSIGUE
        En cada vuelta, de todos los pares presentes en la secuencia, coge el que tenga el
        id de merge MAS BAJO (o sea, el que se aprendio antes):

            stats = get_stats(ids)
            pair = min(stats, key=lambda p: merges.get(p, float("inf")))

        El `float("inf")` es el truco: los pares que no estan en `merges` reciben infinito
        y por tanto nunca ganan el `min`. Si el par ganador resulta no estar en `merges`,
        significa que ya no queda nada que fusionar y hay que parar.

    ESTRUCTURA
        1. trocear el texto igual que en `train_bpe` (mismo `pattern`)
        2. para cada trozo: pasarlo a bytes y aplicar el bucle de arriba hasta que no
           queden merges aplicables
        3. concatenar los ids de todos los trozos en una sola lista

        Cuidado con el bucle: para si `len(ids) < 2` (no hay pares que contar) o si el par
        ganador no esta en `merges`.

    Args:
        text: el texto a codificar.
        merges: lo que devolvio `train_bpe`.
        pattern: el MISMO patron que usaste al entrenar. Si entrenaste con patron y
            codificas sin el (o al reves), los resultados no cuadran.

    Returns:
        Lista de ids.
    """
    raise NotImplementedError("TODO: modulo 03, ejercicio 4 - bpe_encode")


def bpe_decode(ids: Iterable[int], vocab: Vocab) -> str:
    """Convierte una lista de ids de vuelta en texto.

    QUE ES ESTO
        Lo contrario del ejercicio 4, y mucho mas corto. Cada id tiene asociada una
        secuencia de bytes en `vocab`. Las juntas todas y las decodificas.

    EL DETALLE QUE IMPORTA: junta primero, decodifica despues
        NO hagas esto:
            "".join(vocab[i].decode("utf-8") for i in ids)      # MAL

        Un token puede cortar un caracter multibyte por la mitad. Una 'n' es un byte pero
        una 'ñ' son dos (0xC3 0xB1), y a BPE eso le da igual: puede perfectamente haber
        aprendido un token que acaba en 0xC3 y otro que empieza por 0xB1. Decodificados
        por separado, ninguno de los dos es UTF-8 valido.

        Haz esto:
            raw = b"".join(vocab[i] for i in ids)     # primero todos los bytes juntos
            return raw.decode("utf-8", errors="replace")

    POR QUE `errors="replace"` Y NO DEJAR QUE FALLE
        A esto se le llama BYTES FALLBACK. Un modelo a medio entrenar genera secuencias de
        ids cualquiera, y muchas no forman UTF-8 valido. Con `errors="replace"` sale el
        caracter U+FFFD ('caja con interrogante') donde no se pudo decodificar, y la
        generacion continua. Sin el, una excepcion tumbaria el bucle de generacion entero
        por un byte suelto.

    Args:
        ids: los ids a decodificar.
        vocab: lo que devolvio `train_bpe`.

    Returns:
        El texto.
    """
    raise NotImplementedError("TODO: modulo 03, ejercicio 5 - bpe_decode")
