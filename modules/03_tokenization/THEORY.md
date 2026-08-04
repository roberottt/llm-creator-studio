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
"cat"  ->  'c'=36, 'a'=34, 't'=58
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

The algorithm is surprisingly simple. You start with the smallest possible units and keep
**merging the most repeated pair**, over and over.

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

**Step 2.** We count again over the result. Now `(256, a)` comes up twice and `(a, b)` twice
too. A tie, resolved by a fixed rule (here the greater pair wins). `(256, a)` comes out, and
it becomes 257, representing `"aaa"`:

```
[257] b d [257] b a c   ->   [257, 98, 100, 257, 98, 97, 99]
```

We have gone from 11 numbers to 7, and we have learned two "words" nobody told us about:
`"aa"` and `"aaa"`. With real text and 4096 merges, what it learns are things like `" the"`,
`"ing"` or `" that"`.

### Encoding and decoding

To **encode** new text, you apply the learned merges **in the order they were learned**.
That detail matters: if you apply them in another order you get a different tokenization,
valid but incompatible with the one the model saw during training.

To **decode**, you concatenate each token's bytes and decode at the end. Not token by token:
a token can cut a multi-byte character in half (an `ñ` is two bytes and BPE knows nothing
about that), so decoding separately would fail.

## Why bytes and not characters

Starting from the 256 bytes instead of from Unicode characters has an enormous consequence:
**there is no such thing as unencodable text**. Anything — an emoji, Chinese, a badly pasted
binary — is a sequence of bytes, and all 256 bytes are in the vocabulary. The `<UNK>` token
disappears from the problem.

When decoding it can still happen that a byte sequence is not valid UTF-8 (a half-trained
model produces them constantly). That is what `errors="replace"` is for: you get a `�`
instead of an exception that takes generation down.

## The pre-tokenizer: why we do not count over the whole text

If you let BPE count freely over all the text, it learns tokens like `"dog."` or `" the
cat"`, which mix punctuation and words and waste vocabulary on combinations that mean
nothing.

The fix is to split the text first with a regular expression, and **count pairs only within
each chunk**, never across the boundaries. The pattern we use is GPT-4's: it separates
words, numbers (in groups of at most 3 digits), punctuation and whitespace. It needs the
`regex` module rather than the standard library's `re`, because it uses Unicode classes
(`\p{L}` = "any letter") and possessive quantifiers.

## Why 4096 and not 50,000

This is where the tokenizer becomes an architecture decision. The embedding table has
`vocab_size × d_model` parameters. With our model:

| vocabulary | parameters in embeddings | % of the model |
|---|---|---|
| 4,096 | 4096 × 320 = **1.31 M** | 15% of 8.9M |
| 32,000 | 32000 × 320 = **10.2 M** | more than the whole rest of the model |
| 50,257 (GPT-2) | 50257 × 320 = **16.1 M** | the model would be almost only embeddings |

With a small model, a large vocabulary is a disaster: you spend your parameters on a lookup
table instead of on the layers that reason. And on top of that each row would be seen very
few times during training, so it would learn badly.

The price is **compression**. Measured on Shakespeare, with our code:

| vocabulary | bytes per token |
|---|---|
| 300 | 1.42 |
| 512 | 2.05 |
| 1,024 | 2.74 |

The smaller the vocabulary, the more tokens you need for the same text, and therefore more
training steps and less real text fits in the 512-token window. It is a direct trade:
**parameters in the table against sequence length**. With 9M parameters, 4096 is a
reasonable point; it is not the only defensible answer.

A practical warning you will see in module 04: since TinyStories tokenized with 4096
compresses worse than with GPT-2's 50k, the corpus will yield considerably more tokens than
the ~470M usually quoted. That number gets measured, not assumed.

## Where the debate is

Tokenization is probably the ugliest part of modern LLMs, and quite a few people think it
should go away.

Many well-known oddities come from here. That models fail at counting the letters in a word:
they do not see letters, they see fragments. That they are bad at arithmetic: `327` may be
one token and `328` three. That languages other than English are more expensive: the same
text needs more tokens, and at equal window size less of it fits.

There are active lines of research towards models that work directly over bytes, with no
tokenizer. They have not displaced BPE yet, partly because of attention's quadratic cost
over such long sequences. It is a genuinely open problem.

---

**Further reading:** Sennrich et al. 2016,
[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
(the paper that brought BPE to language) · Karpathy,
[minbpe](https://github.com/karpathy/minbpe) and his video, very much worth watching after
doing the exercises. Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
