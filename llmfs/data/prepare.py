"""Dataset preparation: from text to .bin files ready for training.

This happens ONCE and is cached. Tokenizing all of TinyStories with a pure-python BPE takes
on the order of an hour; doing it at every startup would be unacceptable.
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
    """A dataset already tokenized and on disk."""

    train: np.ndarray
    val: np.ndarray
    vocab_size: int
    #: For the character tokenizer: `{character: id}`. Empty with BPE.
    stoi: dict[str, int]
    itos: dict[int, str]
    #: For BPE: the merges and the byte vocabulary.
    merges: dict[Any, int] | None = None
    bpe_vocab: dict[int, bytes] | None = None

    def encode(self, text: str) -> list[int]:
        if self.merges is not None:
            bpe_encode = resolve("03_tokenization", "bpe_encode")
            from llmfs.reference import GPT4_SPLIT_PATTERN

            return bpe_encode(text, self.merges, GPT4_SPLIT_PATTERN)
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        if self.bpe_vocab is not None:
            bpe_decode = resolve("03_tokenization", "bpe_decode")
            return bpe_decode(ids, self.bpe_vocab)
        return "".join(self.itos.get(i, "?") for i in ids)


def prepare_shakespeare(cfg: RunConfig, quiet: bool = False) -> Dataset:
    """Tiny-shakespeare at character level. Seconds, with no external dependencies."""
    from llmfs.data.download import fetch_tinyshakespeare

    pack = resolve("04_data", "pack_tokens_uint16")
    split = resolve("04_data", "train_val_split")

    target = data_dir() / "tinyshakespeare_char"
    target.mkdir(parents=True, exist_ok=True)
    meta_path = target / "meta.json"

    text, _ = fetch_tinyshakespeare(quiet=quiet)
    vocab_chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab_chars)}
    itos = {i: c for c, i in stoi.items()}

    if (target / "train.bin").exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("vocab_size") == len(vocab_chars):
            train = np.memmap(target / "train.bin", dtype=np.uint16, mode="r")
            val = np.memmap(target / "val.bin", dtype=np.uint16, mode="r")
            return Dataset(train, val, len(vocab_chars), stoi, itos)

    if not quiet:
        print(f"[llmfs] tokenizing shakespeare at character level ({len(text):,} characters)...")

    tokens = pack([stoi[c] for c in text], len(vocab_chars))
    train, val = split(tokens, cfg.data.val_fraction)
    np.asarray(train).tofile(target / "train.bin")
    np.asarray(val).tofile(target / "val.bin")
    meta_path.write_text(
        json.dumps({"vocab_size": len(vocab_chars), "chars": vocab_chars}, ensure_ascii=False),
        encoding="utf-8",
    )

    return Dataset(
        np.memmap(target / "train.bin", dtype=np.uint16, mode="r"),
        np.memmap(target / "val.bin", dtype=np.uint16, mode="r"),
        len(vocab_chars),
        stoi,
        itos,
    )


def prepare(cfg: RunConfig, quiet: bool = False) -> Dataset:
    """Prepare whichever dataset the config asks for."""
    if cfg.data.dataset == "tinyshakespeare" or cfg.data.tokenizer == "char":
        return prepare_shakespeare(cfg, quiet=quiet)
    raise NotImplementedError(
        f"The '{cfg.data.dataset}' dataset is prepared in module 13. "
        "For now only character-level tinyshakespeare is available."
    )


def make_get_batch(dataset: Dataset, cfg: RunConfig, device: Any) -> Any:
    """Return the `get_batch(split, batch_size)` function the Trainer expects."""
    get_batch = resolve("04_data", "get_batch")
    rng = np.random.default_rng(cfg.train.seed)
    data = {"train": dataset.train, "val": dataset.val}

    def fn(split: str, batch_size: int):
        return get_batch(
            data[split], batch_size, cfg.model.context_length, device=device, rng=rng
        )

    return fn
