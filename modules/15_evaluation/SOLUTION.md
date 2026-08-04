# 15 — Annotated solution

## Exercise 1 — `perplexity_from_loss`

```python
if not math.isfinite(loss):
    return float("inf")
return math.exp(loss)
```

The `isfinite` guard is not decorative: `math.exp(inf)` raises `OverflowError`, and in the
middle of an evaluation that leaves you not knowing what happened. Returning `inf` is
informative.

**The useful check:** with loss $\ln(V)$, the perplexity is exactly $V$. With vocabulary 4096
and an untrained model, perplexity 4096. It is the same bug detector from module 05 seen from
the other side.

## Exercise 2 — `bits_per_byte`

```python
if n_bytes <= 0:
    raise ValueError("n_bytes has to be positive")
return total_loss_nats / math.log(2) / n_bytes
```

**The `/ math.log(2)`** converts nats into bits: one nat is $1/\ln 2 = 1.4427$ bits.

**The first argument is the TOTAL loss, not the mean.** If you pass it the mean, the result
comes out divided by the number of tokens and means nothing. The `n_tokens` parameter is not
used in the computation; it is in the signature precisely to make that clear.

**Why this metric and not perplexity.** Perplexity depends on the tokenizer: if your
vocabulary splits words into smaller pieces, each token is easier to predict and your number
comes out better without the model being better. Bits per byte normalizes by bytes of the
original text, which do not depend on how you chop things up.

And it has an exact interpretation: **how many bits you would need to transmit the text using
the model as a compressor**. It is not an analogy. A language model *is* a compressor, and the
equivalence between prediction and compression comes from Shannon (1948).

## Exercise 3 — `run_prompt_battery`

```python
prompts = prompts or PROMPTS_TINYSTORIES
return [
    {"prompt": prompt, "tests": label, "completion": generate_fn(prompt)}
    for prompt, label in prompts
]
```

Three lines. **The value of the exercise is not in the code**, it is in having a **fixed**
battery you can run again every time you change something, and compare.

**Why `generate_fn` is passed instead of the model.** It encapsulates the model *and* the
tokenizer, so the function knows nothing about either. It is the same pattern as `get_batch`
in module 04 and `optimizer_factory` in module 13: passing the capability as a function
instead of coupling to a concrete object. And it makes the exercise testable with a fake
generator.

## What you should see in the demo

**The metrics of the model trained in module 13:**

```
chance (the floor)    loss 4.1744    perplexity 65.0
train                 loss 1.3546    perplexity  3.88
val                   loss 1.5973    perplexity  4.94
```

From hesitating between 65 characters to hesitating between 5. The train/val gap of +0.24 is
incipient overfitting and at this scale it is normal.

**And bits per byte, which puts the model in context:**

| compressor | bits/byte |
|---|---|
| uncompressed | 8.00 |
| gzip (English) | ~2.50 |
| **your model** | **~2.30** |
| the best LLMs | 0.60–0.80 |

Your 0.8M-parameter model trained for 70 seconds compresses roughly as well as gzip. That is
not nothing: gzip is a very good algorithm refined over decades.

## About the battery, and an honest warning

The demo runs the TinyStories battery against a model trained on **Shakespeare at character
level**. Prompts in modern English are completely out of distribution for it, and the result
shows:

```
Once upon a time, there was a little girl named Lily. She is A my soul,
when thy should stay for thy true.  LUCENTIO: Your true
```

It starts by copying the prompt, and as soon as it can it goes back to Shakespeare. **This is
not a failure of the model: it is exactly what should happen.** A model only knows what it has
seen.

The exercise of reading the six continuations and judging them is the same with the TinyStories
model, and there you will see correct grammar and local coherence.

## What to expect from the final 9M model on TinyStories

So that expectations are concrete:

- **Correct grammar** most of the time.
- **Local coherence, not global.** Two or three sentences in a row make sense; a ten-sentence
  story, probably not.
- **Limited vocabulary**, which is what is wanted: TinyStories is deliberately written with
  the vocabulary of a 4-year-old.
- **No reasoning at all.** No arithmetic, no world knowledge, no instruction following.

If your model does that, it worked. The distance to an assistant is not one of training: it is
three or four orders of magnitude in parameters and data, plus all the post-training of
module 16.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
def perplexity_from_loss(loss: float) -> float:
    if not math.isfinite(loss):
        return float("inf")
    return math.exp(loss)


def bits_per_byte(total_loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    if n_bytes <= 0:
        raise ValueError("n_bytes has to be positive")
    return total_loss_nats / math.log(2) / n_bytes


def run_prompt_battery(
    generate_fn: Callable[[str], str],
    prompts: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    prompts = prompts or PROMPTS_TINYSTORIES
    return [
        {"prompt": prompt, "tests": label, "completion": generate_fn(prompt)}
        for prompt, label in prompts
    ]
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
