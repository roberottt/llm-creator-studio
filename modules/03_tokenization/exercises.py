"""Module 03 - Tokenization and BPE.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement in order -> `llmfs check 03` -> `llmfs hint 03 -e N`
-> `SOLUTION.md` has the complete code.

The "aaabdaaabac" example from THEORY.md, worked through by hand, is EXACTLY what you are
going to program. Keep it in front of you.

WHAT YOU ARE GOING TO BUILD
===========================

The final model's tokenizer. Five functions that fit together like this:

    get_stats    (ex. 1)  count which pairs of neighbours repeat most
    merge        (ex. 2)  replace a pair with a new token
        |
        +--> train_bpe   (ex. 3)  repeat 1 and 2 until there are 4096 tokens
                 |
                 +--> bpe_encode  (ex. 4)  text -> ids
                 +--> bpe_decode  (ex. 5)  ids -> text

The first two are short and mechanical. The third is the central one. The last two use what
the third learned.

VOCABULARY YOU ARE GOING TO NEED
================================

- **token**: the unit of text the model handles. With BPE, a word fragment.
- **vocabulary**: how many distinct tokens exist. Ours will have 4096.
- **merge**: fusing two adjacent tokens into a new one. It is BPE's operation.
- **pre-tokenizer**: the regular expression that splits the text BEFORE counting pairs, so
  no merge crosses from one word to the next.
- **bytes fallback**: working over bytes (0-255) instead of characters, so the "unknown
  character" cannot exist.

    llmfs demo 03     trains vocabularies of several sizes and compares against tiktoken
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import regex

# GPT-4's pre-tokenization pattern. You do not have to understand all of it: what it does is
# split the text into words, numbers, punctuation and whitespace, so merges do not cross
# boundaries that make no sense (see THEORY.md).
#
#     regex.findall(GPT4_SPLIT_PATTERN, "Hello, world!")
#     -> ['Hello', ',', ' world', '!']
from llmfs.reference import GPT4_SPLIT_PATTERN

Pair = tuple[int, int]
Merges = dict[Pair, int]
Vocab = dict[int, bytes]


def get_stats(ids: Sequence[int], counts: dict[Pair, int] | None = None) -> dict[Pair, int]:
    """Counts how many times each pair of CONSECUTIVE numbers appears.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Three lines.

        1. If `counts` is None, start with an empty dictionary:

               counts = {} if counts is None else counts

        2. Walk the pairs of neighbours and add one to each:

               for pair in zip(ids, ids[1:]):
                   counts[pair] = counts.get(pair, 0) + 1

        3. Return `counts`.

    `zip(ids, ids[1:])` produces every pair of neighbours — (ids[0],ids[1]),
    (ids[1],ids[2])… — without handling indices by hand.

    EXAMPLE TO CHECK AGAINST
    ------------------------
        get_stats([97, 97, 97, 98])  ->  {(97, 97): 2, (97, 98): 1}

    Note that the pair (97,97) comes up TWICE: at positions 0-1 and at 1-2. When COUNTING,
    pairs DO overlap. (When MERGING, in exercise 2, they do not. They are different things
    and it is worth being clear about that from the start.)

    WHAT THE `counts` PARAMETER IS FOR
    ----------------------------------
    To accumulate into a dictionary that already exists, without having to concatenate
    lists. `train_bpe` needs it because it processes the text split into chunks and wants
    the sum over all of them, but WITHOUT counting pairs that cross from one chunk to the
    next:

        stats = {}
        for chunk in chunks:
            get_stats(chunk, stats)     # keeps adding into the same dictionary

    It returns the dictionary as well as mutating it: that way it works for both uses.

    A PYTHON DETAIL
    ---------------
    The default value is `None` and not `{}` on purpose. A `{}` as a default value is
    created ONCE when the function is defined and shared across every call: the classic
    mutable-argument bug.

    Args:
        ids: the sequence of numbers.
        counts: dictionary to accumulate into, or `None` to create a new one.

    Returns:
        `{(a, b): times}`. It is the same dictionary passed in as `counts`, if one was.
    """
    raise NotImplementedError("TODO: module 03, exercise 1 - get_stats")


def merge(ids: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    """Replaces every occurrence of a pair with a single new number.

    WHAT YOU HAVE TO WRITE
    ----------------------
    A `while` with an index you control yourself.

        out, i, n = [], 0, len(ids)

        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                out.append(new_id)
                i += 2                      # <- consumes TWO positions
            else:
                out.append(ids[i])
                i += 1                      # <- consumes ONE

        return out

    WHY A `while` AND NOT A `for`
    -----------------------------
    A `for` ALWAYS advances by one. Here you need to skip two positions on a match, and that
    is exactly the difference between counting pairs (which do overlap) and merging them
    (which do not).

    Check it with `[1, 1, 1]` merging `(1,1)`:

        - you find the pair at position 0, replace it and SKIP to position 2
        - at position 2 only a stray 1 is left, with no partner
        - result: [256, 1], NOT [256, 256]

    EXAMPLE TO CHECK AGAINST
    ------------------------
        merge([97, 97, 97, 98, 97, 97], (97, 97), 256)  ->  [256, 97, 98, 256]

    THE `i < n - 1`
    ---------------
    It stops you looking at `ids[i+1]` when you are on the last element. Without it you get
    an `IndexError` as soon as the list ends right on the first element of the pair.

    Args:
        ids: the original sequence.
        pair: the pair to merge.
        new_id: the number that replaces it.

    Returns:
        A NEW list. Do not modify `ids`.
    """
    raise NotImplementedError("TODO: module 03, exercise 2 - merge")


def train_bpe(
    text: str,
    vocab_size: int,
    pattern: str | None = None,
    verbose: bool = False,
) -> tuple[Merges, Vocab]:
    """Trains the tokenizer: learns which pairs to merge and in what order.

    WHAT YOU HAVE TO WRITE
    ----------------------
    It is the "aaabdaaabac" example from THEORY.md, in a loop. Six steps.

        1. Validate that `vocab_size >= 256` and raise `ValueError` if not. (Those are the
           bytes: there cannot be a smaller vocabulary.)

        2. Split the text:

               chunks = [text] if pattern is None else regex.findall(pattern, text)

        3. Turn each chunk into bytes and from there into integers 0-255:

               ids = [list(chunk.encode("utf-8")) for chunk in chunks if chunk]

        4. Start the two output structures:

               merges = {}
               vocab = {i: bytes([i]) for i in range(256)}

        5. Repeat `vocab_size - 256` times, with `i` as the counter:

             a. Count the pairs of ALL the chunks into one dictionary:

                    stats = {}
                    for chunk_ids in ids:
                        get_stats(chunk_ids, stats)

             b. If `stats` is empty, `break`. (There are no pairs left to merge.)

             c. Pick the winner:

                    pair = max(stats, key=lambda p: (stats[p], p))

             d. The new id is `256 + i`.

             e. Apply the merge to each chunk:

                    ids = [merge(c, pair, new_id) for c in ids]

             f. Record what was learned:

                    merges[pair] = new_id
                    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]     # bytes + bytes

        6. Return `(merges, vocab)`.

    THE TIE-BREAK IN STEP 5c
    ------------------------
    `max(stats, key=lambda p: (stats[p], p))` compares TUPLES: frequency first and, on a tie,
    the pair. Python compares tuples element by element, so that does exactly what you want.

    Which one wins a tie makes no difference to the tokenizer's quality, but it has to be
    deterministic and it has to be THE SAME criterion as the reference. If you used
    `max(stats, key=stats.get)`, the winner would depend on the dictionary's insertion order
    and your merges would diverge from the test's as soon as there was a tie.

    THE `break` IN STEP 5b IS NOT OPTIONAL
    --------------------------------------
    If you ask for 4096 merges over a 20-character text, at some point every chunk is reduced
    to a single token and there are no pairs to count. Without the `break`, `max()` over an
    empty dictionary raises `ValueError`. There is a test that covers that case.

    WHY WE COUNT CHUNK BY CHUNK
    ---------------------------
    So no merge can join the end of one word with the start of the next. If you concatenated
    the chunks, BPE would learn tokens like "cat.The".

    ABOUT PERFORMANCE
    -----------------
    This implementation walks the whole corpus on every merge. For 4096 merges over 2 GB that
    would be days. It is a conscious decision: the code is written to be understood. That is
    why module 04 trains the merges on a sample and then encodes the full corpus.

    Args:
        text: the training text.
        vocab_size: final vocabulary size. >= 256.
        pattern: the pre-tokenization regular expression, or `None` for no splitting.
        verbose: if `True`, prints each merge as it is learned.

    Returns:
        `(merges, vocab)`:
          - `merges`: `{(a, b): new_id}` in the ORDER they were learned (python dicts
            preserve insertion order, so nothing special is needed).
          - `vocab`: `{id: bytes}` with the 256 initial bytes plus one per merge.

    Raises:
        ValueError: if `vocab_size` is less than 256.
    """
    raise NotImplementedError("TODO: module 03, exercise 3 - train_bpe")


def bpe_encode(text: str, merges: Merges, pattern: str | None = None) -> list[int]:
    """Turns text into ids, applying the learned merges.

    WHAT YOU HAVE TO WRITE
    ----------------------
    A helper function that encodes ONE chunk, and a loop that applies it to all of them.

    **The helper** (you can call it `_encode_chunk` and put it anywhere in the file):

        def _encode_chunk(ids, merges):
            while len(ids) >= 2:
                stats = get_stats(ids)
                pair = min(stats, key=lambda p: merges.get(p, float("inf")))
                if pair not in merges:
                    break
                ids = merge(ids, pair, merges[pair])
            return ids

    **And the main function:**

        1. Split the text with the SAME pattern you trained with.
        2. For each chunk: `list(chunk.encode("utf-8"))` and run it through `_encode_chunk`.
        3. Concatenate the ids of every chunk into a single list and return it.

    THE `min` WITH `float("inf")` IS THE HEART OF THE EXERCISE
    ---------------------------------------------------------
    The merge ids are 256, 257, 258... in the ORDER they were learned. So "the one learned
    first" is the same as "the one with the lowest id".

    `merges.get(p, float("inf"))` gives infinity to pairs that are not in `merges`, so they
    never win the `min`. And if the winner turns out not to be in `merges`, that means NONE
    of the present pairs is mergeable: it is time to stop.

    WHY THE ORDER MATTERS SO MUCH
    -----------------------------
    The tokenizer is not "split the text into the longest possible pieces": it is "reproduce
    the training process exactly".

    Two different tokenizations of the same text can both be valid as sequences of ids, but
    only one is the one the model saw millions of times during training. The other one looks
    as foreign to it as text in another language.

    A CONSEQUENCE THAT SURPRISES PEOPLE
    -----------------------------------
    With merges `(a,a)->256` and `(256,a)->257`, the string "aaaa" gives `[256, 256]`, NOT
    `[257, a]`.

    The first merge is applied to the WHOLE sequence at once and takes the four 'a's two at a
    time, so the pair `(256, a)` never gets to form. With three 'a's you do get `[257]`. It
    is not a bug: it is how BPE works, and the same happens in tiktoken. There is a test that
    documents it.

    Args:
        text: the text to encode.
        merges: what `train_bpe` returned.
        pattern: the SAME pattern you trained with. If you trained with a pattern and encode
            without it (or the other way round), the results do not line up.

    Returns:
        The list of ids.
    """
    raise NotImplementedError("TODO: module 03, exercise 4 - bpe_encode")


def bpe_decode(ids: Iterable[int], vocab: Vocab) -> str:
    """Turns a list of ids back into text.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Two lines. And the order of those two lines is the whole exercise.

        raw = b"".join(vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    JOIN FIRST, DECODE AFTERWARDS
    -----------------------------
    Do NOT do this:

        "".join(vocab[i].decode("utf-8") for i in ids)      # WRONG

    Reason: UTF-8 encodes non-ASCII characters in several bytes. An 'n' is one byte, but an
    'ñ' is two: 0xC3 0xB1.

    BPE does not care at all — it works with bytes and knows nothing about characters — so it
    may perfectly well have learned a token that ENDS in 0xC3 and another that STARTS with
    0xB1. Decoded separately, neither is valid UTF-8. Together, they are an 'ñ'.

    There is a test that builds exactly that case.

    WHY `errors="replace"`
    ----------------------
    It is what is called the BYTES FALLBACK. A half-trained model generates arbitrary
    sequences of ids, and many of them do not form valid UTF-8. With `errors="replace"` you
    get a replacement character where decoding failed and generation continues; without it,
    one exception would take down the whole generation loop over a stray byte.

    When you see odd characters in the first samples in module 14, now you know what they
    are.

    Args:
        ids: the ids to decode.
        vocab: what `train_bpe` returned.

    Returns:
        The text.
    """
    raise NotImplementedError("TODO: module 03, exercise 5 - bpe_decode")
