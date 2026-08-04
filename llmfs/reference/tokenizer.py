"""Reference for module 03: Byte Pair Encoding from scratch.

BPE starts from the 256 possible bytes and repeatedly merges the most frequent adjacent
pair into a new token, until it reaches the desired vocabulary size. Working over bytes
rather than unicode characters guarantees that any text is encodable: there is no such
thing as an "unknown character".
"""

from __future__ import annotations

from typing import Iterable, Sequence

import regex

#: GPT-4's pre-tokenization pattern (tiktoken `cl100k_base`).
#:
#: It stops merges from crossing boundaries that make no linguistic sense: without it, BPE
#: would learn tokens like "dog." or " the cat" that mix punctuation and words and waste
#: vocabulary. It requires the `regex` module rather than the stdlib `re`, because of the
#: unicode classes `\p{L}` / `\p{N}` and the possessive quantifiers `++` and `?+`.
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

Pair = tuple[int, int]
Merges = dict[Pair, int]
Vocab = dict[int, bytes]


def get_stats(ids: Sequence[int], counts: dict[Pair, int] | None = None) -> dict[Pair, int]:
    """Count how many times each pair of consecutive ids appears.

    Args:
        ids: a sequence of ids.
        counts: dictionary to accumulate into. It lets you add up statistics from several
            chunks without concatenating them, which is exactly what `train_bpe` does with
            the pre-tokenizer's chunks.

    Returns:
        `{(a, b): times}`. Overlapping pairs are all counted: in `[1, 1, 1]` the pair
        `(1, 1)` shows up twice.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every occurrence of `pair` with `new_id`.

    Occurrences are consumed left to right and without overlap: in `[1, 1, 1]`, merging
    `(1, 1)` gives `[new_id, 1]`, not `[new_id, new_id]`.
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
    """Train a BPE's merges.

    Algorithm:
        1. Split the text with `pattern` (or leave it whole if it is `None`).
        2. Convert each chunk to UTF-8 bytes: that gives the starting ids 0-255.
        3. Repeat `vocab_size - 256` times: count pairs, take the most frequent one, merge
           it into a new id starting at 256.

    Tie-breaking: if two pairs tie on frequency, the greater one in lexicographic order
    wins (that is, `max` over the tuple `(frequency, pair)`). It does not matter which one
    is chosen as long as it is deterministic; it is written down so that your
    implementation and the reference produce exactly the same merges and the test can
    compare them.

    Returns:
        `merges`: `{(a, b): new_id}` in the order they were learned.
        `vocab`: `{id: bytes}` with the 256 initial bytes plus one token per merge.
    """
    if vocab_size < 256:
        raise ValueError(f"vocab_size ({vocab_size}) cannot go below 256: those are the bytes.")

    chunks = [text] if pattern is None else regex.findall(pattern, text)
    ids: list[list[int]] = [list(chunk.encode("utf-8")) for chunk in chunks if chunk]

    merges: Merges = {}
    vocab: Vocab = {i: bytes([i]) for i in range(256)}

    for i in range(vocab_size - 256):
        stats: dict[Pair, int] = {}
        for chunk_ids in ids:
            get_stats(chunk_ids, stats)
        if not stats:
            break  # there are no pairs left to merge

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
    """Text -> ids.

    The merges are applied **in the order they were learned**, not in order of frequency in
    this particular text. That is why each pass through the loop looks for the present pair
    with the lowest merge id: it reproduces the training order.

    Applying them in another order produces a different tokenization, syntactically valid
    but incompatible with the one the model saw during training.
    """
    chunks = [text] if pattern is None else regex.findall(pattern, text)
    out: list[int] = []
    for chunk in chunks:
        out.extend(_encode_chunk(list(chunk.encode("utf-8")), merges))
    return out


def _encode_chunk(ids: list[int], merges: Merges) -> list[int]:
    while len(ids) >= 2:
        stats = get_stats(ids)
        # The pair whose merge was learned first (lowest id). `inf` for those that do not exist.
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def bpe_decode(ids: Iterable[int], vocab: Vocab) -> str:
    """ids -> text.

    The bytes of each token are concatenated and decoded at the end, not token by token.
    A token can cut a multi-byte character in half (an 'ñ' is two bytes and BPE knows
    nothing about that), so decoding separately would fail.

    `errors="replace"` is the bytes fallback: if the byte sequence is not valid UTF-8 you
    get U+FFFD instead of an exception. A half-trained model produces invalid sequences
    constantly and we do not want that to take generation down.
    """
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------- helpers


def compression_ratio(text: str, ids: Sequence[int]) -> float:
    """Bytes of text per token. The higher it is, the better the tokenizer compresses."""
    n_bytes = len(text.encode("utf-8"))
    return n_bytes / len(ids) if ids else 0.0
