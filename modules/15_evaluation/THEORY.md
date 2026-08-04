# 15 — Evaluation

## Why this module matters

**Because "is my model any good?" is harder to answer than it looks.**

You have a trained model and one number: the loss. Now what? That number does not tell you
whether the model writes stories someone would want to read, and comparing your number with
another model's can be outright misleading.

Here you learn to use three different tools and, above all, **what each one measures and where
it fails**. Including a metric that *is* comparable across different models and that almost
nobody uses, and the part no automatic metric replaces: reading what it writes.

It is also the module where what you have built gets put in context, with concrete
expectations about what a 9M model can and cannot do.

### What you will know by the end

- Why comparing perplexities between models with different tokenizers **means nothing**, and
  it is done constantly
- A metric that *is* comparable, and its exact interpretation: your model is a compressor
- How to evaluate qualitatively with a fixed battery of prompts
- What to really expect from a 9M model, so you do not end up unfairly disappointed

### What it costs

2 hours. Three short functions and a generated report you can read at your leisure.

---

## Perplexity: the usual metric

You already know it from module 05. It is $e^L$, with $L$ the mean loss, and it is read as
**how many equiprobable options the model is hesitating between**:

```
loss 8.317  →  perplexity 4096   (untrained, hesitating over the whole vocabulary)
loss 1.60   →  perplexity 4.95   (hesitating between about 5 options)
```

It is the most used metric because it is cheap, automatic and correlates well with quality
*within the same setup*.

### And its problem, which is serious

**Perplexity depends on the tokenizer.** If your vocabulary splits words into smaller pieces,
each individual token is easier to predict and your perplexity comes out better without the
model being better.

An extreme example: a bit-level model would have a perplexity close to 2 and would be useless.
A word-level one, on the same text, would have a perplexity in the hundreds.

**Comparing perplexities between models with different tokenizers means nothing**, and it is
done constantly in papers and in blog posts.

## Bits per byte: the metric that *is* comparable

The solution: normalize by **bytes of original text** instead of by tokens. Bytes do not depend
on how you chop things up.

$$\text{bits/byte} = \frac{L_{\text{total}} / \ln 2}{n_{\text{bytes}}}$$

The $\ln 2$ converts nats into bits.

And it has an exact and rather nice interpretation: **it is how many bits you would need to
transmit the text using the model as a compressor**. A model at 1.0 bits/byte compresses to an
eighth. Reference points:

| | bits/byte |
|---|---|
| gzip on English text | ~2.5 |
| a good small model | ~1.2 |
| the best LLMs | 0.6–0.8 |
| the theoretical limit (Shannon) | ~0.6–1.3 (disputed) |

This equivalence between prediction and compression comes from Shannon (1948) and it is not an
analogy: it is an identity. A language model **is** a compressor.

## The qualitative battery

Neither of the two previous metrics tells you whether the model writes stories a human would
want to read. For that you have to read them.

The TinyStories paper proposes evaluating three things separately, with fixed prompts:

**Grammar.** Are the sentences well built? Do subject and verb agree?

**Coherence.** Does the story contradict itself? If in the first sentence the cat is black, is
it still black three sentences later?

**Creativity.** Does it contribute anything or does it repeat templates?

What is interesting about the paper is that these three capabilities **appear at different
scales**. A 1M-parameter model already does decent grammar; coherence needs more; creativity,
more still. It is not a single ladder: they are abilities that emerge separately.

In this module you will use a battery of six fixed prompts and read the continuations
yourself. Yes, by hand. There is no shortcut.

## Why there is no good automatic metric

Many have been tried and they all fail in the same place.

**BLEU, ROUGE** and company compare against a reference answer. For free generation there is no
correct answer: there are infinitely many, and all of them different from the reference one.

**Using another LLM as a judge** (LLM-as-a-judge) is what is done now, and it works reasonably
for large models. It has known and well-documented biases: it prefers long answers, it prefers
the judge model's own style, and it is sensitive to the order the options are presented in.

**Human evaluation** is the gold standard and it is expensive, slow and noisy: two annotators
disagree more than you would expect.

For your 9M model, reading six continuations is perfectly reasonable and probably more
informative than any number.

## What to expect from a 9M model on TinyStories

Be concrete about expectations, because the original paper trained similar models:

- **Correct grammar most of the time.** Well-formed sentences.
- **Local coherence, not global.** Two or three sentences in a row make sense; a ten-sentence
  story, probably not.
- **Limited vocabulary**, which is what you would expect: TinyStories is deliberately written
  with the vocabulary of a 4-year-old.
- **No reasoning at all.** No arithmetic, no world knowledge, no instruction following.

If your model does that, it worked. If you were expecting something like an assistant, the
difference is not one of training: it is three or four orders of magnitude in parameters and
data, and the post-training you will see in module 16.

## Where the debate is

**The relationship between perplexity and capabilities is looser than people assume.** It is
known that lowering the loss improves the model, but not by how much or in what. Two models
with the same perplexity can behave very differently on specific tasks, and perplexity can go
down through memorization without anything useful improving.

**Data contamination has ruined a good part of benchmark evaluation.** The test sets are on the
internet, and the models are trained on the internet. When a model scores well on a benchmark,
telling "it has learned" apart from "it has seen it" is technically hard and commercially
uncomfortable. It is one of the most serious methodological problems in the field right now.

**And emergent capabilities are under active discussion.** Sharp jumps in capability were
documented as scale increased, and in 2023 a convincing analysis was published arguing that
many of those jumps are **artefacts of discontinuous metrics**: if you measure with an
all-or-nothing metric, you see jumps where a continuous one would show a smooth curve. The
discussion continues.

---

**Further reading:** Eldan & Li 2023, [TinyStories](https://arxiv.org/abs/2305.07759) (the
qualitative battery) · Shannon 1948, *A Mathematical Theory of Communication* (prediction and
compression) · Schaeffer et al. 2023,
[Are Emergent Abilities a Mirage?](https://arxiv.org/abs/2304.15004). Stray terms are in
[GLOSSARY.md](../../GLOSSARY.md).
