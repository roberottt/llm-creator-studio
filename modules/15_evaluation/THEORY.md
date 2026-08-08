# 15 — Evaluation: is my model any good, and compared to what?

## Why this module matters

**Because "is my model any good?" is harder to answer than it looks.**

You have a trained model and a number: the loss. Now what? That number does not tell you whether
the model writes stories anyone would want to read, and comparing your number against another
model's can be outright misleading.

Here you learn to use three different tools and, above all, **what each one measures and where it
fails**. Including a metric that *is* comparable across different models and that almost nobody
uses, and the part no automatic metric replaces: reading what it writes.

It is also the module where what you have built gets put in context, with concrete expectations
about what a 9M model can and cannot do.

### What you will know by the end

- Why comparing perplexities across models with different tokenizers **means nothing**, and gets
  done constantly
- A metric that *is* comparable, and its exact interpretation: your model is a compressor
- How to evaluate qualitatively with a fixed prompt battery, and why it is done by hand
- What to really expect from a 9M model, so you do not end up unfairly disappointed

### What you are going to write

Three functions, and this theory follows them in order:

| Exercise | What it does |
|---|---|
| 1. `perplexity_from_loss` | The usual metric, and its limits |
| 2. `bits_per_byte` | The one that **can** be compared across models |
| 3. `run_prompt_battery` | The one no automatic metric replaces |

All three are short — two lines for the first two — and the third does not even compute anything:
it organizes the work so that the part that really matters, **reading what the model writes**, is
comfortable. That is the whole module: three tiny functions whose value is in knowing when to use
each one.

### What it costs

2 hours, and a good part goes into reading the report you generate.

---

## Exercise 1: perplexity (`perplexity_from_loss`)

You already know it from module 05. It is $e^L$, with $L$ the mean loss in nats, and it reads as
**how many equally likely options the model is hesitating between**. The function is two lines: a
guard for non-finite values and `math.exp(loss)`.

The three cases to recognize:

```
   loss 8.317  ->  perplexity 4096.0   untrained: hesitating across the WHOLE vocabulary
   loss 1.60   ->  perplexity    4.95  hesitating between about 5 options
   loss 0.0    ->  perplexity    1.0   perfect, no hesitation
```

The first deserves the check: `ln(4096) = 8.317`, so `exp(8.317) = 4096`. A freshly initialized
model spreads probability equally across the 4096 tokens and its perplexity is exactly the
vocabulary size. It is module 05's floor, seen from the other side.

And here is how it reads on the course's model, measured:

| set | loss | perplexity | |
|---|---|---|---|
| random (the floor) | 4.1744 | 65.0 | what an untrained model scores |
| train | 1.2746 | 3.58 | |
| val | 1.5497 | 4.71 | |

From hesitating between 65 characters to hesitating between 4.7. And the train/val gap of +0.275
is small, which is what you want to see: **if it were large, the model would be memorizing**
instead of learning. That is the useful reading of having both figures side by side.

### And its problem, which is serious

**Perplexity depends on the tokenizer.** If your vocabulary splits words into smaller pieces, each
individual token is easier to predict and your perplexity comes out better without the model being
better.

An extreme example to see it: a model predicting bit by bit would have perplexity close to 2 and be
useless. A word-level one, over the same text, would have perplexity in the hundreds.

**Comparing perplexities across models with different tokenizers means nothing**, and it gets done
constantly in papers and blog posts. Hence the next exercise.

---

## Exercise 2: bits per byte (`bits_per_byte`)

The fix: normalize by **bytes of original text** instead of by tokens. Bytes do not depend on how
you chop things up.

$$\text{bits/byte} = \frac{L_{\text{total}} / \ln 2}{n_{\text{bytes}}}$$

Also two lines. The $\ln 2$ is the only tricky bit: it converts nats to bits, because all the
course's loss is in nats — natural logarithm, module 05 — and this metric is expressed in bits by
convention.

Notice the input is the **total** loss, not the mean: if you passed the per-token mean you would be
mixing a per-token normalization with a per-byte one, which is exactly what this metric exists to
avoid.

### The nice part: your model is a compressor

This metric has an exact interpretation: **it is how many bits you would need to transmit the text
using the model as a compressor**. And it is not an analogy, it is an identity going back to
Shannon (1948).

Measured on the course's model:

| compressor | bits/byte |
|---|---|
| uncompressed | 8.00 |
| gzip (English text) | ~2.50 |
| **your model** | **2.236** |
| the best LLMs | 0.60 – 0.80 |

So **your toy model compresses better than gzip**. At 2.236 bits/byte it reduces the text to 1/3.6
of its size. It is not a presentational trick: if you hooked an arithmetic coder up to its
probabilities, it would genuinely compress at that ratio.

And unlike perplexity, this figure **can** be compared against any other model's, whatever its
tokenizer.

---

## Exercise 3: the qualitative battery (`run_prompt_battery`)

Neither of the previous two metrics tells you whether the model writes something a human would want
to read. For that you have to read it.

The exercise computes nothing: it takes a list of fixed prompts, generates a continuation for each
and returns them organized for you to read. **The prompts are fixed on purpose**: if you change
them between two evaluations you are not comparing models, you are comparing prompts. It is the
same reason the validation set does not get touched.

The TinyStories paper proposes looking at three things separately:

**Grammar.** Are the sentences well formed? Do subject and verb agree?

**Coherence.** Does the story contradict itself? If the cat is black in the first sentence, is it
still black three sentences later?

**Creativity.** Does it contribute anything or repeat templates?

What is interesting about the paper is that these three capabilities **appear at different
scales**. A 1M-parameter model already does decent grammar; coherence needs more; creativity, more
still. It is not a single ladder: they are abilities that emerge separately, and that is why they
get scored separately.

The battery's six prompts are not random: each tests something different — basic continuation,
causal coherence, object tracking, resolution, using an object, story closure. When you read the
continuations, read them against what each prompt was trying to test.

**A warning about the demo:** it runs with the character-level Shakespeare model, not one trained
on TinyStories. Modern-English prompts are completely out of distribution for it and the
continuations come out strange. The exercise of reading them is the same, and you actually learn
quite a lot from watching a model try to continue something it has never seen.

---

## Why there is no good automatic metric

Many have been tried and they all fail in the same place.

**BLEU, ROUGE** and friends compare against a reference answer. For free generation there is no
correct answer: there are infinitely many, and all of them different from the reference.

**Using another LLM as a judge** (*LLM-as-a-judge*) is what gets done now, and it works reasonably
for large models. It has well-documented biases: it prefers long answers, it prefers the judge
model's own style, and it is sensitive to the order the options are presented in.

**Human evaluation** is the gold standard and it is expensive, slow and noisy: two annotators
disagree more than you would expect.

For your 9M model, reading six continuations is perfectly reasonable and probably more informative
than any number.

## What to expect from a 9M model on TinyStories

It is worth being concrete about expectations, because the original paper trained similar models
and we know what comes out:

- **Correct grammar most of the time.** Well-formed sentences.
- **Local coherence, not global.** Two or three consecutive sentences make sense; a ten-sentence
  story, probably not.
- **Limited vocabulary**, which is expected: TinyStories is deliberately written with the
  vocabulary of a 4-year-old.
- **No reasoning at all.** No arithmetic, no world knowledge, no following instructions.

If your model does that, it worked. If you expected something like an assistant, the difference is
not about training: it is three or four orders of magnitude in parameters and data, plus module
16's post-training, which is literally what turns "a model that continues text" into "a model that
obeys".

## Where the debate is

**The relationship between perplexity and capabilities is looser than people assume.** Lowering the
loss is known to improve the model, but not by how much or in what. Two models with the same
perplexity can behave very differently on specific tasks, and perplexity can drop through
memorization without anything useful improving.

**Data contamination has ruined a good part of benchmark evaluation.** The test sets are on the
internet, and models are trained on the internet. When a model scores well on a benchmark,
distinguishing "it learned" from "it has seen it" is technically hard and commercially awkward. It
is one of the most serious methodological problems in the field right now.

**And emergent abilities are under active discussion.** Sudden jumps in capability with scale were
documented, and in 2023 a convincing analysis argued that many of those jumps are **artifacts of
discontinuous metrics**: if you measure with an all-or-nothing metric, you see jumps where a
continuous one would show a smooth curve. The discussion continues.

---

**Further reading:** Eldan & Li 2023, [TinyStories](https://arxiv.org/abs/2305.07759) (the
qualitative battery) · Shannon 1948, *A Mathematical Theory of Communication* (prediction and
compression) · Schaeffer et al. 2023,
[Are Emergent Abilities a Mirage?](https://arxiv.org/abs/2304.15004). Stray terms are in
[GLOSSARY.md](../../GLOSSARY.md).
