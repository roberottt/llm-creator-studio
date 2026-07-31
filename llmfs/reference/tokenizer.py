"""Referencia del modulo 02: Byte Pair Encoding desde cero.

BPE parte de los 256 bytes posibles y va fusionando el par adyacente mas frecuente en un
token nuevo, repitiendo hasta llegar al tamanyo de vocabulario deseado. Trabajar sobre
bytes y no sobre caracteres unicode garantiza que cualquier texto sea codificable: no
existe el "caracter desconocido".
"""

from __future__ import annotations

from typing import Iterable, Sequence

import regex

#: Patron de pre-tokenizacion de GPT-4 (tiktoken `cl100k_base`).
#:
#: Sirve para que los merges no crucen fronteras que no tienen sentido lingueistico: sin
#: el, BPE aprenderia tokens como "perro." o " el gato" que mezclan puntuacion y palabras
#: y desperdician vocabulario. Requiere el modulo `regex` y no el `re` de la stdlib, por
#: las clases unicode `\p{L}` / `\p{N}` y los cuantificadores posesivos `++` y `?+`.
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

Pair = tuple[int, int]
Merges = dict[Pair, int]
Vocab = dict[int, bytes]


def get_stats(ids: Sequence[int], counts: dict[Pair, int] | None = None) -> dict[Pair, int]:
    """Cuenta cuantas veces aparece cada par de ids consecutivos.

    Args:
        ids: secuencia de ids.
        counts: diccionario donde acumular. Permite sumar estadisticas de varios trozos
            sin concatenarlos, que es justo lo que hace `train_bpe` con los chunks del
            pre-tokenizador.

    Returns:
        `{(a, b): veces}`. Los pares solapados se cuentan todos: en `[1, 1, 1]` el par
        `(1, 1)` sale 2 veces.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    """Sustituye cada aparicion de `pair` por `new_id`.

    Las apariciones se consumen de izquierda a derecha y sin solapar: en `[1, 1, 1]`,
    fusionar `(1, 1)` da `[new_id, 1]`, no `[new_id, new_id]`.
    """
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(
    text: str,
    vocab_size: int,
    pattern: str | None = None,
    verbose: bool = False,
) -> tuple[Merges, Vocab]:
    """Entrena los merges de un BPE.

    Algoritmo:
        1. Trocear el texto con `pattern` (o dejarlo entero si es `None`).
        2. Convertir cada trozo a bytes UTF-8: eso da los ids 0-255 de partida.
        3. Repetir `vocab_size - 256` veces: contar pares, coger el mas frecuente,
           fusionarlo en un id nuevo empezando por 256.

    Desempate: si dos pares empatan en frecuencia gana el mayor en orden lexicografico
    (es decir, `max` sobre la tupla `(frecuencia, par)`). Da igual cual se elija mientras
    sea determinista; se fija por escrito para que tu implementacion y la referencia
    produzcan exactamente los mismos merges y el test pueda compararlos.

    Returns:
        `merges`: `{(a, b): nuevo_id}` en el orden en que se aprendieron.
        `vocab`: `{id: bytes}` con los 256 bytes iniciales mas un token por merge.
    """
    if vocab_size < 256:
        raise ValueError(f"vocab_size ({vocab_size}) no puede bajar de 256: son los bytes.")

    chunks = [text] if pattern is None else regex.findall(pattern, text)
    ids: list[list[int]] = [list(chunk.encode("utf-8")) for chunk in chunks if chunk]

    merges: Merges = {}
    vocab: Vocab = {i: bytes([i]) for i in range(256)}

    for i in range(vocab_size - 256):
        stats: dict[Pair, int] = {}
        for chunk_ids in ids:
            get_stats(chunk_ids, stats)
        if not stats:
            break  # ya no quedan pares que fusionar

        pair = max(stats, key=lambda p: (stats[p], p))
        new_id = 256 + i

        ids = [merge(chunk_ids, pair, new_id) for chunk_ids in ids]
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

        if verbose:
            print(f"merge {i + 1}/{vocab_size - 256}: {pair} -> {new_id} "
                  f"({vocab[new_id]!r}) x{stats[pair]}")

    return merges, vocab


def bpe_encode(text: str, merges: Merges, pattern: str | None = None) -> list[int]:
    """Texto -> ids.

    Los merges se aplican **en el orden en que se aprendieron**, no en orden de
    frecuencia en este texto concreto. Por eso el bucle busca en cada vuelta el par
    presente cuyo id de merge sea menor: reproduce el orden del entrenamiento.

    Aplicarlos en otro orden produce una tokenizacion distinta, valida sintacticamente
    pero incompatible con la que vio el modelo al entrenar.
    """
    chunks = [text] if pattern is None else regex.findall(pattern, text)
    out: list[int] = []
    for chunk in chunks:
        out.extend(_encode_chunk(list(chunk.encode("utf-8")), merges))
    return out


def _encode_chunk(ids: list[int], merges: Merges) -> list[int]:
    while len(ids) >= 2:
        stats = get_stats(ids)
        # El par cuyo merge se aprendio antes (id mas bajo). `inf` para los que no existen.
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def bpe_decode(ids: Iterable[int], vocab: Vocab) -> str:
    """ids -> texto.

    Se concatenan los bytes de cada token y se decodifica al final, no token a token.
    Un token puede cortar un caracter multibyte por la mitad (una 'ñ' son dos bytes y BPE
    no sabe nada de eso), asi que decodificar por separado fallaria.

    `errors="replace"` es el bytes-fallback: si la secuencia de bytes no es UTF-8 valido
    sale U+FFFD en vez de una excepcion. Un modelo a medio entrenar genera secuencias
    invalidas constantemente y no queremos que eso tumbe la generacion.
    """
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------- utilidades


def compression_ratio(text: str, ids: Sequence[int]) -> float:
    """Bytes de texto por token. Cuanto mas alto, mejor comprime el tokenizador."""
    n_bytes = len(text.encode("utf-8"))
    return n_bytes / len(ids) if ids else 0.0
