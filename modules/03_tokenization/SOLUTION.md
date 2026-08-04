# 03 — Annotated solution

## Exercise 1 — `get_stats`

```python
counts = {} if counts is None else counts
for pair in zip(ids, ids[1:]):
    counts[pair] = counts.get(pair, 0) + 1
return counts
```

`zip(ids, ids[1:])` produces every pair of neighbours: `(ids[0],ids[1])`, `(ids[1],ids[2])`…
It is the idiomatic form and it avoids the `range(len(ids)-1)` with hand-managed indices.

**The mutable `counts` parameter** is what lets `train_bpe` accumulate statistics over all
the chunks without concatenating them. And returning it as well as mutating it makes the
function work for both uses: `stats = get_stats(ids)` and `get_stats(chunk, stats)`.

If a mutable default argument worries you: here the default value is `None`, not `{}`. That
distinction matters — a `{}` as a default value would be created **once** when the function
is defined and shared across every call, which is the classic Python bug. With `None` and
the check inside, each call gets its own dictionary.

## Exercise 2 — `merge`

```python
out, i, n = [], 0, len(ids)
while i < n:
    if i < n - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
        out.append(new_id)
        i += 2
    else:
        out.append(ids[i])
        i += 1
return out
```

**The `while` with a manual index is not optional.** A `for` always advances by one, and
here you need to skip two positions on a match. That is exactly the difference between
counting pairs (which do overlap) and merging them (which do not).

Check it with `[1,1,1]`:
- Counting: `(1,1)` comes up **twice** (positions 0-1 and 1-2).
- Merging: there is **one** substitution → `[256, 1]`. Having consumed positions 0 and 1,
  the `1` left at position 2 has nobody to pair with.

**The `i < n - 1`** stops you looking at `ids[i+1]` when you are on the last element.
Without it, an `IndexError` as soon as the list ends right on the first element of the pair.

## Exercise 3 — `train_bpe`

```python
if vocab_size < 256:
    raise ValueError(...)

chunks = [text] if pattern is None else regex.findall(pattern, text)
ids = [list(chunk.encode("utf-8")) for chunk in chunks if chunk]

merges, vocab = {}, {i: bytes([i]) for i in range(256)}

for i in range(vocab_size - 256):
    stats = {}
    for chunk_ids in ids:
        get_stats(chunk_ids, stats)      # accumulates into the same dict
    if not stats:
        break

    pair = max(stats, key=lambda p: (stats[p], p))
    new_id = 256 + i

    ids = [merge(c, pair, new_id) for c in ids]
    merges[pair] = new_id
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

return merges, vocab
```

**The `break` when there are no pairs left.** If you ask for 4096 merges over a 20-character
text, at some point each chunk is reduced to a single token and there are no pairs to count.
Without the `break`, `max()` over an empty dictionary raises `ValueError`. The test
`test_it_stops_if_it_runs_out_of_pairs_to_merge` covers this case, and it is not
theoretical: it happens as soon as you experiment with short texts.

**The tie-break `(stats[p], p)`.** Python compares tuples element by element: frequency
first and, on a tie, the pair. Which one wins makes no difference to the tokenizer's
quality, but it has to be **deterministic and the same as the reference's**. If you left
`max(stats, key=stats.get)`, the winner would depend on the dictionary's insertion order,
which in turn depends on the order you walked the chunks in. It would work, but any harmless
change would break reproducibility.

**`vocab[new_id] = vocab[a] + vocab[b]`** is concatenation of `bytes`, not of `str`. That is
why the vocabulary is built alongside the merges: each new token is defined in terms of the
two that form it, and recursively it ends up being a concrete byte sequence.

**The `if chunk` filter** discards the empty pieces the regex can produce. An empty chunk
gives an empty list that contributes nothing but gets walked on every one of the thousands
of merges.

### About performance

This implementation is $O(\text{merges} \times \text{text length})$: on every merge it walks
the whole corpus twice. For 4096 merges over 2 GB that would be days.

It is a conscious decision: the code is written to be understood. Serious implementations
keep incremental indices of where each pair appears and only update what changes. That is
why module 04 trains the merges on a **sample** (~150 MB) and then encodes the full corpus
with multiprocessing. Training on a sample barely changes the resulting merges: the relative
frequencies of the pairs stabilize long before you have seen all the text.

## Exercise 4 — `bpe_encode`

```python
def _encode_chunk(ids, merges):
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids
```

**The `min` with `float("inf")`** is the heart of the exercise. The merge ids are
`256, 257, 258…` in learning order, so "the lowest id" is equivalent to "the one learned
first". Pairs that are not in `merges` are assigned infinity, so they never win the minimum.

If the winner turns out not to be in `merges`, that means **none** of the present pairs is
mergeable, and it is time to stop.

**Why the order matters so much.** It is the hardest part to see. The tokenizer is not
"split the text into the longest possible pieces": it is "reproduce the training process
exactly". Two different tokenizations of the same text can both be valid as sequences of
ids, but only one is the one the model saw millions of times during training. The other one
looks as foreign to it as text in another language.

There is a surprising consequence that the test captures: with merges `(a,a)→256` and
`(256,a)→257`, the string `"aaaa"` gives `[256, 256]` and not `[257, a]`. The first merge is
applied **to the whole sequence at once** and takes the four `a`s two at a time, so the pair
`(256, a)` never gets to form. With three `a`s you do get `[257]`. It is not a bug: it is
how BPE works, and the same happens in tiktoken.

## Exercise 5 — `bpe_decode`

```python
raw = b"".join(vocab[i] for i in ids)
return raw.decode("utf-8", errors="replace")
```

Two lines, and both have their reason.

**Joining before decoding.** UTF-8 encodes non-ASCII characters in several bytes: `ñ` is
`0xC3 0xB1`. BPE does not care at all — it works with bytes and knows nothing about
characters — so it may have learned a token ending in `0xC3` and another starting with
`0xB1`. Decoded separately, neither is valid; together, they are an `ñ`. The test
`test_decoding_joins_the_bytes_before_decoding` builds exactly that case.

**`errors="replace"`, the bytes fallback.** A freshly initialized model generates random
ids, and many of those sequences do not form valid UTF-8. Without the `errors="replace"`, a
`UnicodeDecodeError` would take down the whole generation loop. With it you get a `�` and
generation continues. When you see odd characters in the first samples in module 14, now you
know what they are.

## What you should see in the demo

The same sentence, tokenized with growing vocabularies:

```
vocab 300  -> 35 tokens:  T | h | e  | k | ing |   | s | ha | ll  | ...
vocab 1024 -> 20 tokens:  The  | k | ing |  shall  | speak |   | to  | ...
tiktoken   -> 10 tokens:  The |  king |  shall |  speak |  to |  his | ...
```

And the most telling detail: note that tiktoken's tokens **start with a space** (`" king"`,
`" shall"`). That is not an accident, it is the pre-tokenizer: the pattern assigns the
preceding space to the word that follows, so that `"king"` at the start of a sentence and
`" king"` in the middle are different tokens. It is one of the reasons LLMs are sensitive to
whether or not your prompt ends with a space.
