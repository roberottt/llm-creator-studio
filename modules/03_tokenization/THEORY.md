# 03 — Tokenization and BPE

## Why this module matters

**Because a neural network cannot read.**

It only does arithmetic with numbers, so text has to be turned into a list of integers
before the model sees it. And how you chop it up is not a plumbing detail: it decides how
many parameters your model will have, how much text fits in its window, and what it will get
strangely wrong.

Here you build the tokenizer the final model uses: BPE from scratch, with a 4096-token
vocabulary trained on the corpus. It is not a library you call: it is 60 lines you write.

And along the way you will understand why LLMs fail at counting the letters in a word, why
they are bad at arithmetic, and why languages other than English come out more expensive.

### What you will know by the end

- Why neither characters nor words are used, but word fragments
- How BPE **learns on its own** which fragments deserve to be a token, without anyone
  telling it
- Why we work with bytes and not characters (spoiler: so the unknown token cannot exist)
- **Why 4096 and not 50,000**, with the numbers that justify it

### What you are going to write

Five functions. This theory is ordered so that you read them in this order, and each one has
its own section with the matching worked example:

| Exercise | What it does | Where it is explained |
|---|---|---|
| 1. `get_stats` | Count which pair of neighbours repeats most | [§ Counting the pairs](#exercise-1-counting-the-pairs-get_stats) |
| 2. `merge` | Replace a pair with a new token | [§ Merging the winning pair](#exercise-2-merging-the-winning-pair-merge) |
| 3. `train_bpe` | Repeat 1 and 2 until the vocabulary is full | [§ The training loop](#exercise-3-the-training-loop-train_bpe) |
| 4. `bpe_encode` | Text → ids, reproducing the training | [§ Encoding](#exercise-4-encoding-bpe_encode) |
| 5. `bpe_decode` | ids → text | [§ Decoding](#exercise-5-decoding-bpe_decode) |

The first two are short and mechanical, and they are the bricks everything else is built
from. The third is the central one. The last two use what the third learned.

### What it costs

4 hours. It is the longest module in Part I, and the tokenizer that comes out of it is the
one you will use for the rest of the course.

---

## The problem: a network cannot read

A neural network only does arithmetic with numbers. Text has to be turned into a list of
integers before the model sees it, and **how you chop it up conditions everything else**:
how many parameters the model will have, how much text fits in its window, and what it will
get strangely wrong.

The question is: what is the unit? Letters? Words? Something in between?

## Option A: by characters

You assign a number to each distinct character. With Shakespeare there are 65 symbols:

```
"cat"  ->  'c'=36, 'a'=34, 't'=58  ->  [36, 34, 58]
```

**In favour:** the vocabulary is tiny and there is never an unknown character.

**Against:** the sequences become extremely long. A 200-word story is about 1000 characters.
And here is the serious problem: attention's cost grows with the **square** of the window
length (module 06). Doubling the length quadruples the cost. Chopping finely is expensive.

You also force the model to spend capacity learning to spell before it can learn anything
about meaning.

## Option B: by words

One entry per dictionary word.

**In favour:** short sequences, and each token already means something.

**Against:** two big problems. First, the vocabulary explodes: 50,000 words minimum for
English, and many more for languages with rich inflection. The second is worse: **what do
you do with a word that is not there?** Proper nouns, typos, new words. The classic answer
was an `<UNK>` token that destroys information irrecoverably.

## Option C: word fragments, learned from the data

The idea behind **BPE** (*Byte Pair Encoding*): let frequent words be a single token and
rare ones be split into pieces. Neither characters nor words: whatever the data says.

And the interesting part is that nobody writes the list of fragments. **It is discovered by
counting.** The algorithm starts with the smallest possible units and keeps **merging the
most repeated pair of neighbours**, over and over, until it has as many tokens as you asked
for.

### The example, step by step

Text: `aaabdaaabac`. We start with the bytes (`a`=97, `b`=98, `c`=99, `d`=100):

```
[97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]
```

**Step 1.** We count each pair of neighbours:

```
(a,a) -> 4 times      (b,d) -> 1
(a,b) -> 2            (d,a) -> 1
                      (b,a) -> 1     (a,c) -> 1
```

`(a,a)` wins. We give it the number 256 (0 to 255 are already taken by the bytes) and
substitute, left to right and without overlapping:

```
aaabdaaabac  ->  [256] a b d [256] a b a c
```

**Step 2.** We count again **over the result**, not over the original text. This matters:
pairs are always counted over the sequence as it currently stands. Now `(256, a)` comes up
twice and `(a, b)` twice too. A tie, resolved by a fixed rule (here the greater pair wins).
`(256, a)` comes out, and it becomes 257, representing `"aaa"`:

```
[257] b d [257] b a c   ->   [257, 98, 100, 257, 98, 97, 99]
```

We have gone from 11 numbers to 7, and we have learned two "words" nobody told us about:
`"aa"` and `"aaa"`. You can check both steps by running the reference with `verbose=True`;
it prints exactly this:

```
merge 1/2: (97, 97) -> 256 (b'aa') x4
merge 2/2: (256, 97) -> 257 (b'aaa') x2
```

With real text and thousands of merges, what it learns are things like `" the"`, `"ing"` or
`" that"`. Here are the **first fifteen real merges** over 200,000 characters of
Shakespeare, straight out of running `train_bpe`:

| # | id | token | built from |
|---|---|---|---|
| 1 | 256 | `'e '` | `'e'` + `' '` |
| 2 | 257 | `'th'` | `'t'` + `'h'` |
| 3 | 258 | `'t '` | `'t'` + `' '` |
| 4 | 259 | `'s '` | `'s'` + `' '` |
| 5 | 260 | `'ou'` | `'o'` + `'u'` |
| 6 | 261 | `'d '` | `'d'` + `' '` |
| 7 | 262 | `', '` | `','` + `' '` |
| 8 | 263 | `'er'` | `'e'` + `'r'` |
| 9 | 264 | `'an'` | `'a'` + `'n'` |
| 10 | 265 | `'in'` | `'i'` + `'n'` |
| … | | | |
| 251 | 506 | `'rom'` | `'r'` + `'om'` |
| 252 | 507 | `"'ll "` | `"'"` + `'ll '` |
| 253 | 508 | `'itiz'` | `'iti'` + `'z'` |
| 254 | 509 | `'itizen'` | `'itiz'` + `'en'` |

(This training run goes without a pre-tokenizer —which is what `llmfs demo 03` does— and
that is why tokens come out with the space stuck **behind** them, like `'e '` or `'no '`.
With a pre-tokenizer the space stays attached to the word that follows; you will see that in
a moment.)

Notice the progression: the first ones are pairs of very frequent letters, and towards the
end you get long fragments like `'itizen'` — built out of earlier merges, not out of single
letters. Each new token can serve as material for the next one, and that is how you get from
letters to words without anyone explaining to the algorithm that words exist.

---

## From the algorithm to the five functions

Before you dive into the exercises it is worth being clear about **what data structures you
are working with**, because there are three of them and they show up in every signature. If
you sit down to code without this clear, the module is an uphill fight.

**1. The ids: `list[int]`.** The text, already turned into numbers. At the start they are
bytes (0-255) and as merges get applied, higher numbers start appearing:

```python
list("cat".encode("utf-8"))     # [99, 97, 116]
```

That is all. No classes, no tensors: in this module a text is a Python list of integers.

**2. `merges: dict[(int, int), int]`.** The learned rules, in the order they were learned.
The key is the pair being merged, the value is the new id:

```python
{(97, 97): 256, (256, 97): 257}
```

Read it as: "whenever you see a 97 followed by another 97, replace them with a 256". And
since the new ids are handed out in order (256, 257, 258…), **the id also tells you when the
rule was learned**. That detail, which looks like mere bookkeeping, is what makes exercise 4
work.

**3. `vocab: dict[int, bytes]`.** The meaning table: which bytes each id stands for. It
starts with the 256 bytes and grows by one token per merge:

```python
{0: b'\x00', ..., 97: b'a', ..., 256: b'aa', 257: b'aaa'}
```

`merges` is for **encoding** and `vocab` is for **decoding**. They are two views of the same
thing, and that is why `train_bpe` returns both.

With that, the five functions fit together like this:

```
    get_stats  (ex. 1) ─┐
                        ├─> train_bpe (ex. 3) ──> merges ──> bpe_encode (ex. 4)  text -> ids
    merge      (ex. 2) ─┘                    └──> vocab  ──> bpe_decode (ex. 5)  ids -> text
```

And there is an asymmetry worth keeping in mind from the start: **training happens once,
encoding happens millions of times**. `train_bpe` is allowed to take minutes because it only
runs when preparing the data; `bpe_encode` will be applied to every text that enters the
model.

---

## Exercise 1: counting the pairs (`get_stats`)

This is the "measure" half of the algorithm: walk the sequence and note how many times each
pair of neighbours shows up.

```python
get_stats([97, 97, 97, 98])  ->  {(97, 97): 2, (97, 98): 1}
```

**The detail you have to internalize: when counting, pairs overlap.** In `[97, 97, 97]` the
pair `(97, 97)` comes up **twice**, once at positions 0-1 and once at 1-2. In `[1, 1, 1, 1]`
it comes up three times. You are not splitting the list into couples: you are looking
through a two-element window that advances one step at a time.

```
[1, 1, 1, 1]
 └──┘             (0,1)
    └──┘          (1,2)
       └──┘       (2,3)      ->  {(1,1): 3}
```

Hold on to that picture, because in exercise 2 the window advances differently, and that
difference is the classic trap of this module.

**Why the function takes an optional `counts`.** Because `train_bpe` does not count over one
continuous text but over a list of chunks, and it wants the sum of all of them **without**
counting the pairs that cross from one chunk to the next (you will see why in exercise 3).
With an accumulator that is trivial:

```python
stats = {}
for chunk in chunks:
    get_stats(chunk, stats)     # keeps adding into the same dictionary
```

If `get_stats` could only return fresh dictionaries, you would have to merge them by hand
every round. It returns the dictionary *as well as* mutating it, so it works for both uses.

## Exercise 2: merging the winning pair (`merge`)

This is the "act" half: walk the sequence and replace every occurrence of the pair with a
single new number.

```python
merge([97, 97, 97, 98, 97, 97], (97, 97), 256)  ->  [256, 97, 98, 256]
```

**And here, unlike when counting, occurrences do NOT overlap.** When you find the pair you
consume it whole and jump two positions. Look at `[1, 1, 1]` merging `(1,1)`:

```
[1, 1, 1]
 └──┘        match -> you write 256 and jump to position 2
       ↑     a lone 1 is left, with no partner
                                              ->  [256, 1]   and NOT [256, 256]
```

That is where the only design decision of the exercise comes from: **a `while` with an index
of your own, not a `for`**. A `for` always advances one at a time; you need to advance one
*or* two depending on whether there was a match.

That behaviour is not an implementation quirk. It is what makes the next round's count
coherent: after merging, the sequence is shorter and there are new pairs that did not exist
before (the 256 now has neighbours). Counting again over that sequence is exactly step 2 of
the `aaabdaaabac` example.

## Exercise 3: the training loop (`train_bpe`)

This is where it all comes together. The body of the loop is literally the `aaabdaaabac`
example repeated `vocab_size - 256` times:

```
repeat until the vocabulary is full:
    1. count every pair                          (get_stats)
    2. keep the most frequent one
    3. give it the next free id
    4. substitute it throughout the sequence     (merge)
    5. record the rule in `merges` and the meaning in `vocab`
```

There are four things that sketch does not show, and they are the ones that cause trouble.

### Why the text gets split into chunks first

If you let BPE count freely over the whole text, it learns tokens that cross from one word
into the next. This is not a theoretical worry: this is what it actually learns from
`"the cat eats fish. the dog eats meat. the cat sleeps."` repeated eight times, **without** a
pre-tokenizer, looking at the first twenty merges:

```
'at', 'th', 'the', 'the ', 'eat', '. ', '. the ', 'eats', 'eats ', 'at ',
'cat ', '. the cat ', '. the cat s', '. the cat sl', '. the cat sle',
'. the cat slee', '. the cat sleep', '. the cat sleeps', 'eats m', 'eats meat'
```

Eight of the twenty tokens are prefixes of the same sentence, `'. the cat sleeps'`. The
algorithm is memorizing one concrete string from the corpus and spending the vocabulary on
it.

And this is what it learns from the same text **with** the pre-tokenizer:

```
'at', 'th', 'the', 'eat', ' the', 'eats', 'cat', ' cat', ' eats', 'sl',
'sle', 'slee', 'sleep', 'sleeps', 'sh', 'og', 'meat', 'ish', 'fish', 'dog'
```

Words. `' cat'`, `' eats'`, `'sleeps'`, `'fish'`, `'dog'`. Same algorithm, same text, same
number of merges: the only difference is that now pairs **are not counted across chunk
boundaries**.

That is why step 5a in the docstring counts chunk by chunk into a single dictionary instead
of concatenating everything. And it is why `get_stats` takes an accumulator.

One detail worth noticing in those examples: tokens start with a space (`' cat'`, not
`'cat'`). The pre-tokenizer leaves the space attached to the word that *follows*, so in the
vocabulary `' cat'` and `'cat'` are two different tokens. That is why you always see those
leading spaces in tokenizer demos.

### The tie-break has to be deterministic

```python
pair = max(stats, key=lambda p: (stats[p], p))
```

That `key` returns a **tuple**, and Python compares tuples element by element: first the
frequency and, on a tie, the pair itself. In the `aaabdaaabac` example there was a real tie
at 2 between `(256, 97)` and `(97, 98)`, and this rule is what decides that the first one
wins.

Which one wins does not matter for the quality of the tokenizer. What **does** matter is that
it is always the same one. If you used `max(stats, key=stats.get)`, the winner would depend
on the dictionary's insertion order, and the first time there was a tie your merges would
diverge from the reference's — and with them every id, and the test would fail for a reason
that has nothing to do with whether you understood BPE.

### The `break` when no pairs are left

If you ask for 4096 merges over a twenty-character text, at some point every chunk is
reduced to a single token and there is no pair left to count. `stats` comes out empty and
`max()` over an empty dictionary raises `ValueError`. There is a test covering exactly that
case.

### The cost

This implementation walks the **entire** corpus on every merge. Real timings, over 200,000
characters of Shakespeare:

| vocabulary | merges | time |
|---|---|---|
| 300 | 44 | 1 s |
| 512 | 256 | 3 s |
| 1,024 | 768 | 7 s |
| 2,048 | 1,792 | 14 s |

It grows linearly with the number of merges and linearly with the corpus size: 4096 merges
over the gigabytes of TinyStories would take days. This is a conscious decision —the code is
written to be understood, not to be fast— and it is why module 04 trains the merges on a
sample and then encodes the full corpus with them. Production tokenizers do the same, only
with the expensive part written in Rust.

## Exercise 4: encoding (`bpe_encode`)

You have the rules. Now, new text → ids.

The temptation is to think encoding means "find the longest fragments that are in the
vocabulary". **It does not.** Encoding is *reproducing the training process*: applying the
learned merges, in the same order they were learned.

### Why the order

The merges were learned in a chain: 257 may be built out of 256. If you apply 257 first,
without having created the 256s it needs, the result is not the same text badly encoded — it
is a *different* tokenization, perfectly valid as a list of ids, but one the model has never
seen. And to the model that is as foreign as text in another language: the same characters
cut into pieces it does not recognize.

That is why the loop looks, on every round, for **the pair present whose merge was learned
earliest**:

```python
pair = min(stats, key=lambda p: merges.get(p, float("inf")))
if pair not in merges:
    break
```

This is where that detail about ids being handed out in order pays off. "Learned earlier" is
the same as "lower id", so the learning order is recovered with a `min` and there is no need
to store any timestamp. And `merges.get(p, float("inf"))` gives infinity to the pairs that
are not mergeable, so they can never win the `min`. If one of them wins anyway, it means
**none** of the pairs present can be merged: we are done.

### A consequence that surprises people

With the merges `(a,a) -> 256` and `(256,a) -> 257`, how many tokens does `"aaaa"` give?

The intuitive answer is `[257, a]`: take the longest fragment you can, `"aaa"`, and leave the
leftover. But what comes out is `[256, 256]`, and you can check it:

```python
>>> bpe_encode("aaa",  {(97,97): 256, (256,97): 257})
[257]
>>> bpe_encode("aaaa", {(97,97): 256, (256,97): 257})
[256, 256]
```

The reason is that **each merge is applied to the whole sequence at once**. 256 is the oldest
one, so it goes first, and it takes the four `a`s two at a time. By the time `(256, a)` gets
its turn, there is no lone `a` left to form the pair with. With three `a`s you do get `[257]`,
because the first merge leaves one over.

This is not a bug in the course: it is how BPE works, and tiktoken does exactly the same.
There is a test documenting it.

### And do not forget the pattern

Encoding has to split the text with **the same** pattern you trained with. If you train with
a pre-tokenizer and encode without one (or the other way round), the pairs that form are not
the same and the ids do not line up. It does not raise an error: it gives worse results for
an invisible reason.

Here is a real sentence with a 512-token vocabulary trained on Shakespeare **without** a
pre-tokenizer (which is why the spaces trail behind, `'he '`, rather than leading):

```
The king shall speak to his people tomorrow morning.

'T' | 'he ' | 'k' | 'ing ' | 'shall ' | 'sp' | 'ea' | 'k ' | 'to ' | 'his ' |
'pe' | 'op' | 'l' | 'e ' | 't' | 'om' | 'or' | 'r' | 'ow' | ' m' | 'or' | 'n' | 'ing' | '.'
```

52 characters, **24 tokens**. Words that are frequent in the plays (`'shall '`, `'his '`)
come out whole; `'tomorrow'` is split into five pieces. With 4096 tokens trained on the real
corpus, most common words will be a single token.

## Exercise 5: decoding (`bpe_decode`)

Two lines, and the order of those two lines is the whole exercise:

```python
raw = b"".join(vocab[i] for i in ids)
return raw.decode("utf-8", errors="replace")
```

**First join all the bytes, then decode once.** What you must not do is this:

```python
"".join(vocab[i].decode("utf-8") for i in ids)      # WRONG
```

The reason: UTF-8 encodes non-ASCII characters in several bytes. An `n` is one byte, but an
`ï` is two. Look at the real bytes of `"naïve café"`:

```
[110, 97, 195, 175, 118, 101, 32, 99, 97, 102, 195, 169]
          └───┬──┘                             └───┬──┘
              ï                                    é
```

BPE could not care less: it works with bytes and knows nothing about characters. It may
perfectly well have learned a token that **ends** in 195 and another that **starts** with
175. Decoded separately, neither of them is valid UTF-8 and `.decode()` blows up. Together,
they are an `ï`. There is a test that builds exactly that case.

**And why `errors="replace"`.** This is what is called *bytes fallback*. A half-trained model
generates arbitrary id sequences, and many of them do not form valid UTF-8. With
`errors="replace"` you get a `�` where decoding failed and generation carries on; without it,
one stray byte would take down the entire generation loop with an exception. When you see odd
characters in your first samples in module 14, now you know what they are.

---

## Why bytes and not characters

You have seen the mechanics; what is left is the underlying reason for starting from the 256
bytes instead of from Unicode characters: **there is no such thing as unencodable text**.
Anything — an emoji, Chinese, a badly pasted binary — is a sequence of bytes, and all 256
bytes are in the vocabulary from the very first moment. The `<UNK>` token disappears from the
problem, and with it a whole family of failures.

If instead you started from Unicode characters, you would have to decide which ones make the
cut (there are more than 150,000) and what to do with the rest.

The price is the one you have already seen: a non-ASCII character costs several tokens in the
worst case, and that is why languages with accents, and even more so those that do not use
the Latin alphabet, come out more expensive.

## The pre-tokenizer, up close

The pattern we use is GPT-4's (`cl100k_base`), and it lives in
`llmfs/reference/tokenizer.py` as `GPT4_SPLIT_PATTERN`. You do not need to understand all of
it, but you do need to see what it does:

```python
regex.findall(GPT4_SPLIT_PATTERN, "Hello, world!")
['Hello', ',', ' world', '!']

regex.findall(GPT4_SPLIT_PATTERN, "The cat eats fish.")
['The', ' cat', ' eats', ' fish', '.']

regex.findall(GPT4_SPLIT_PATTERN, "in 2026 there were 1234 cats")
['in', ' ', '202', '6', ' there', ' were', ' ', '123', '4', ' cats']
```

It separates words (with their leading space), punctuation and stray whitespace. And numbers
in groups of **at most three digits**: notice that `2026` comes out as `'202'` and `'6'`. That
is deliberate — it stops the tokenizer from learning a token for every year or every frequent
quantity — and it is incidentally one of the reasons models are bad at arithmetic: `1234` is
not "one thousand two hundred and thirty-four" to the model, it is two pieces cut at a place
that means nothing.

It needs the `regex` module rather than the standard library's `re`, because it uses Unicode
classes (`\p{L}` = "any letter", `\p{N}` = "any digit") and possessive quantifiers (`++`,
`?+`) that `re` does not support.

## Why 4096 and not 50,000

This is where the tokenizer becomes an architecture decision. The embedding table —the one
that turns each id into a vector; it shows up in module 05 and you build the whole thing in
module 10— has `vocab_size × d_model` parameters. With our model, `d_model = 320`:

| vocabulary | parameters in embeddings | % of the model |
|---|---|---|
| 4,096 | 4096 × 320 = **1.31 M** | 15% of 8.9M |
| 32,000 | 32000 × 320 = **10.2 M** | more than the whole rest of the model |
| 50,257 (GPT-2) | 50257 × 320 = **16.1 M** | the model would be almost only embeddings |

With a small model, a large vocabulary is a disaster for two reasons. The obvious one: you
spend your parameters on a lookup table instead of on the layers that reason. And the one
that is easier to miss: each row of that table only gets trained when its token appears in
the text, so with 50,000 rows and a modest corpus there would be thousands of tokens seen
four times, with essentially random vectors.

The price of a small vocabulary is **compression**. Measured on Shakespeare, with our code:

| vocabulary | bytes per token |
|---|---|
| 300 | 1.42 |
| 512 | 2.05 |
| 1,024 | 2.74 |
| 2,048 | 3.48 |

The smaller the vocabulary, the more tokens you need for the same text, and therefore more
training steps and less real text fits in the 512-token window.

Notice too that the curve flattens fast, and this is what is most reassuring about picking
4096: over that same text, `cl100k_base` —GPT-4's tokenizer, with **100,277 tokens**— gets
3.67 bytes per token. Our 2,048 vocabulary gets 3.48. Fifty times the vocabulary for 5% more
compression. The diminishing returns are brutal, and the price in parameters is not.
`llmfs demo 03` plots both curves side by side.

It is a direct trade: **parameters in the table against sequence length**. With 9M
parameters, 4096 is a reasonable point; it is not the only defensible answer.

A practical warning you will see in module 04: since TinyStories tokenized with 4096
compresses worse than with GPT-2's 50k, the corpus will yield considerably more tokens than
the ~470M usually quoted. That number gets measured, not assumed.

## What real tokenizers do that you do not

What you write here is complete, correct BPE, and it is what the final model uses. The
differences from `tiktoken` or `sentencepiece` are engineering and finish:

| your code | a production tokenizer |
|---|---|
| `train_bpe` walks the corpus on every merge | incremental indexes; only recounts what changed |
| pure Python | the core in Rust or C++ |
| `bpe_encode` with no cache | per-word cache: most texts repeat the same words |
| no special tokens | `<|endoftext|>`, `<|im_start|>`… reserved outside the BPE |
| a 4096-token vocabulary | 100,000 or more |

The only one of those you will need in this course is the cache, and it shows up in module 04
when preparing TinyStories.

## Where the debate is

Tokenization is probably the ugliest part of modern LLMs, and quite a few people think it
should go away.

Many well-known oddities come from here. That models fail at counting the letters in a word:
they do not see letters, they see fragments, and `'itizen'` is as indivisible a symbol to the
model as `'a'` is to us. That they are bad at arithmetic: you have just seen what the
pre-tokenizer does to `1234`. That languages other than English are more expensive: the same
text needs more tokens, and at equal window size less of it fits.

There are active lines of research towards models that work directly over bytes, with no
tokenizer. They have not displaced BPE yet, partly because of attention's quadratic cost over
such long sequences. It is a genuinely open problem.

---

**Further reading:** Sennrich et al. 2016,
[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
(the paper that brought BPE to language) · Karpathy,
[minbpe](https://github.com/karpathy/minbpe) and his video, very much worth watching after
doing the exercises. Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
