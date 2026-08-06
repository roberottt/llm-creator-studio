"""Preparacion de datasets: de texto a ficheros .bin listos para entrenar.

Se hace UNA vez y se cachea. Tokenizar TinyStories entero con un BPE en python puro tarda
del orden de una hora; hacerlo en cada arranque seria inaceptable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from llmfs.bridge import resolve
from llmfs.config import RunConfig
from llmfs.paths import data_dir


@dataclass
class Dataset:
    """Un dataset ya tokenizado y en disco."""

    train: np.ndarray
    val: np.ndarray
    vocab_size: int
    #: Para el tokenizador de caracteres: `{caracter: id}`. Vacio con BPE.
    stoi: dict[str, int]
    itos: dict[int, str]
    #: Para BPE: los merges y el vocabulario de bytes.
    merges: dict[Any, int] | None = None
    bpe_vocab: dict[int, bytes] | None = None

    def encode(self, texto: str) -> list[int]:
        if self.merges is not None:
            bpe_encode = resolve("03_tokenizacion", "bpe_encode")
            from llmfs.reference import GPT4_SPLIT_PATTERN

            return bpe_encode(texto, self.merges, GPT4_SPLIT_PATTERN)
        return [self.stoi[c] for c in texto if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        if self.bpe_vocab is not None:
            bpe_decode = resolve("03_tokenizacion", "bpe_decode")
            return bpe_decode(ids, self.bpe_vocab)
        return "".join(self.itos.get(i, "?") for i in ids)


def preparar_shakespeare(cfg: RunConfig, quiet: bool = False) -> Dataset:
    """Tiny-shakespeare a nivel caracter. Segundos, sin dependencias externas."""
    from llmfs.data.download import fetch_tinyshakespeare

    pack = resolve("04_datos", "pack_tokens_uint16")
    split = resolve("04_datos", "train_val_split")

    destino = data_dir() / "tinyshakespeare_char"
    destino.mkdir(parents=True, exist_ok=True)
    meta_path = destino / "meta.json"

    texto, _ = fetch_tinyshakespeare(quiet=quiet)
    vocab_chars = sorted(set(texto))
    stoi = {c: i for i, c in enumerate(vocab_chars)}
    itos = {i: c for c, i in stoi.items()}

    if (destino / "train.bin").exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("vocab_size") == len(vocab_chars):
            train = np.memmap(destino / "train.bin", dtype=np.uint16, mode="r")
            val = np.memmap(destino / "val.bin", dtype=np.uint16, mode="r")
            return Dataset(train, val, len(vocab_chars), stoi, itos)

    if not quiet:
        print(f"[llmfs] tokenizando shakespeare a nivel caracter ({len(texto):,} caracteres)...")

    tokens = pack([stoi[c] for c in texto], len(vocab_chars))
    train, val = split(tokens, cfg.data.val_fraction)
    np.asarray(train).tofile(destino / "train.bin")
    np.asarray(val).tofile(destino / "val.bin")
    meta_path.write_text(
        json.dumps({"vocab_size": len(vocab_chars), "chars": vocab_chars}, ensure_ascii=False),
        encoding="utf-8",
    )

    return Dataset(
        np.memmap(destino / "train.bin", dtype=np.uint16, mode="r"),
        np.memmap(destino / "val.bin", dtype=np.uint16, mode="r"),
        len(vocab_chars),
        stoi,
        itos,
    )


def preparar_tinystories(cfg: RunConfig, quiet: bool = False) -> Dataset:
    """TinyStories tokenizado con tu propio BPE (modulo 03).

    Tres pasos, todos cacheados en disco bajo `data/tinystories_bpe<vocab_size>/`:

    1. Descargar el corpus de ~2 GB (`fetch_tinystories`, cacheado aparte como un unico
       `data/tinystories.txt` para que volver a correr esto con otro `vocab_size` no
       vuelva a descargar nada).
    2. Entrenar los merges del BPE sobre una MUESTRA de `cfg.data.bpe_train_bytes`
       caracteres, no sobre el corpus entero: entrenar es la parte cara y superlineal
       (`train_bpe` recuenta todos los pares en cada merge), y una muestra de 150M de
       caracteres ya ve cada subpalabra comun muchas veces.
    3. Codificar el corpus COMPLETO con esos merges y empaquetarlo en `train.bin` /
       `val.bin`. Este paso es python puro y mas o menos lineal en el tamanyo del corpus,
       pero sobre ~2 GB de texto sigue siendo la parte lenta de esta funcion: cuenta con
       algo del orden de una hora, una unica vez.

    En cada llamada posterior a la primera, los ficheros `.bin` y `meta.json` ya existen
    y esto devuelve el resultado en milisegundos.
    """
    from llmfs.data.download import fetch_tinystories
    from llmfs.reference import GPT4_SPLIT_PATTERN

    train_bpe = resolve("03_tokenizacion", "train_bpe")
    bpe_encode = resolve("03_tokenizacion", "bpe_encode")
    pack = resolve("04_datos", "pack_tokens_uint16")
    split = resolve("04_datos", "train_val_split")

    vocab_size = cfg.model.vocab_size
    destino = data_dir() / f"tinystories_bpe{vocab_size}"
    destino.mkdir(parents=True, exist_ok=True)
    meta_path = destino / "meta.json"

    if (destino / "train.bin").exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("vocab_size") == vocab_size:
            merges = {tuple(par): new_id for par, new_id in meta["merges"]}
            bpe_vocab = {int(i): bytes(b) for i, b in meta["vocab"].items()}
            train = np.memmap(destino / "train.bin", dtype=np.uint16, mode="r")
            val = np.memmap(destino / "val.bin", dtype=np.uint16, mode="r")
            return Dataset(train, val, vocab_size, {}, {}, merges, bpe_vocab)

    texto = fetch_tinystories(quiet=quiet)

    muestra = texto[: cfg.data.bpe_train_bytes]
    if not quiet:
        print(
            f"[llmfs] entrenando BPE (vocab_size={vocab_size:,}) sobre "
            f"{len(muestra):,} de {len(texto):,} caracteres..."
        )
    merges, bpe_vocab = train_bpe(muestra, vocab_size, pattern=GPT4_SPLIT_PATTERN)

    if not quiet:
        print(f"[llmfs] codificando el corpus completo ({len(texto):,} caracteres)...")
    ids = bpe_encode(texto, merges, GPT4_SPLIT_PATTERN)

    tokens = pack(ids, vocab_size)
    train, val = split(tokens, cfg.data.val_fraction)
    np.asarray(train).tofile(destino / "train.bin")
    np.asarray(val).tofile(destino / "val.bin")
    meta_path.write_text(
        json.dumps(
            {
                "vocab_size": vocab_size,
                "merges": [[list(par), new_id] for par, new_id in merges.items()],
                "vocab": {str(i): list(b) for i, b in bpe_vocab.items()},
            }
        ),
        encoding="utf-8",
    )

    return Dataset(
        np.memmap(destino / "train.bin", dtype=np.uint16, mode="r"),
        np.memmap(destino / "val.bin", dtype=np.uint16, mode="r"),
        vocab_size,
        {},
        {},
        merges,
        bpe_vocab,
    )


def preparar(cfg: RunConfig, quiet: bool = False) -> Dataset:
    """Prepara el dataset que pida el config."""
    if cfg.data.dataset == "tinyshakespeare" or cfg.data.tokenizer == "char":
        return preparar_shakespeare(cfg, quiet=quiet)
    if cfg.data.dataset == "tinystories":
        return preparar_tinystories(cfg, quiet=quiet)
    raise NotImplementedError(
        f"El dataset '{cfg.data.dataset}' no tiene funcion de preparacion. "
        "Solo estan disponibles 'tinyshakespeare' (char) y 'tinystories' (bpe)."
    )


def hacer_get_batch(dataset: Dataset, cfg: RunConfig, device: Any) -> Any:
    """Devuelve la funcion `get_batch(split, batch_size)` que espera el Entrenador."""
    get_batch = resolve("04_datos", "get_batch")
    rng = np.random.default_rng(cfg.train.seed)
    datos = {"train": dataset.train, "val": dataset.val}

    def fn(split: str, batch_size: int):
        return get_batch(
            datos[split], batch_size, cfg.model.context_length, device=device, rng=rng
        )

    return fn
