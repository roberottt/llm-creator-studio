# 00 — Annotated solution

## Exercise 1 — `next_token_probs`

One line of real work: add up and divide.

```
total = sum of every count
return {token: count/total for each pair}
```

**Why we check that the total is not zero.** If the table comes in empty, `sum()` gives 0
and the division blows up with `ZeroDivisionError`. The problem is not that it blows up: it
is *where* it blows up. Without the check, the error fires inside a dict comprehension,
three levels below where the real cause is, and the message never mentions that the problem
is an empty table. Raising a `ValueError` with a clear message turns half an hour of
debugging into five seconds. This is not a style quirk: it is the difference between an
error that helps you and one that gets in your way.

**Why the dictionary order matters.** In Python 3.7+ dictionaries preserve insertion order.
If you build the result by walking `counts.items()`, the order is kept. If you sorted it
alphabetically, your roulette wheel in exercise 2 would divide up the line [0,1] differently,
and with the same seed you would get different text from the reference. The test
`test_it_preserves_the_keys_and_their_order` is there for exactly that reason.

**Connection to what is coming.** This is normalizing. In module 06 you will see `softmax`,
which does the same thing but exponentiates first:

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

Why exponentiate? Because a neural network spits out arbitrary numbers, positive and
negative, and you cannot normalize `[-2.1, 0.5, 3.0]` by dividing by their sum: you would
get negative probabilities. The exponential turns any real number into a positive one
without changing the order. Here it is not needed because the counts are already positive.

## Exercise 2 — `sample_next_token`

```
r = rng.random()          # a float in [0, 1)
running = 0
for each (token, p) in probs:
    running += p
    if r < running:
        return token
return the last token seen
```

**The most common mistake is using `<=` instead of `<`, or the other way round.** Think it
through with `{'a': 0.5, 'b': 0.5}` and `r = 0.5` exactly. With `r < running`: after `'a'`
the running total is 0.5, and `0.5 < 0.5` is false, so it carries on and returns `'b'`. That
is correct: `'a'` occupies the interval $[0, 0.5)$ and `'b'` occupies $[0.5, 1)$. Since
`rng.random()` returns a number in $[0, 1)$ — 1 never comes out, 0 does — this split is the
one that gives exactly 50/50.

**The final `return` is not paranoia.** Floats do not add up exactly. Try `sum([0.1] * 10)`
in an interpreter: it gives `0.9999999999999999`, not `1.0`. If `rng.random()` returns
`0.99999999999999995`, the loop ends without having returned anything and the function
returns `None`. That breaks exercise 3 with an incomprehensible error several steps later.
The test `test_it_never_returns_none_even_if_the_probabilities_do_not_sum_exactly`
reproduces exactly that case.

**An alternative that also works.**
`random.choices(list(probs), weights=list(probs.values()))` does the same thing in one line.
You are asked to do it by hand because the goal is for you to understand the mechanism: in
module 14 you are going to manipulate this wheel directly (trimming it with top-k,
stretching or compressing it with temperature) and for that you have to know what is inside.

## Exercise 3 — `generate_naive`

```
out = list of the characters of start
repeat (length - len(start)) times:
    context = last len(start) characters of out
    counts  = table.get(context)
    if counts is None or empty: stop
    out.append(sample_next_token(next_token_probs(counts), rng))
return "".join(out)
```

**The important part of this exercise is not the code, it is what it represents.** This loop
is, literally, the same one ChatGPT runs. The only difference from module 14 is where the
probabilities come from: here from `table.get(context)`, there from a forward pass of the
network. The structure — look at the context, get a distribution, sample, append, repeat —
is identical.

**The `break` is the pedagogical point.** When you reach a context that was not in the
training text, a counting model goes literally mute: it has no row to look up. A neural
network *never* has this problem, because it does not look anything up: it computes.
Whatever you give it, it produces a distribution. It may be a bad one, but it exists. That is
one of the deep reasons networks are used.

**Working with a list and joining at the end.** Doing `out = out + character` with strings
creates a new string on every pass. For 200 characters it does not matter; for module 13's
500 million tokens it does. The habit of accumulating into a list and calling `"".join()` at
the end is free and always correct.

**The detail about `length`.** It counts the total returned, including `start`. If `start`
has 2 characters and you ask for 5, you generate 3. That is why the loop iterates
`length - len(start)` times and not `length`. The test checks it because it is an off-by-one
that slips in on its own.

## What you should see when you run the demo

On Shakespeare, the percentage of generated words that really exist:

| context | real words | what it looks like |
|---|---|---|
| 1 character | ~14% | `Wieisiopthote hashe hon ghou` |
| 2 characters | ~45% | `Fin tis fall mounto degiver he of` |
| 3 characters | ~62% | `First perange is ther, rumous the had to` |
| 4 characters | ~91% | `First Camiliar, And hear'd his now him in his way` |
| 6 characters | ~98% | `The senator: No more spices of my colour half way` |

With 6 characters of context the text already looks like Shakespeare from a distance. **And
yet this is not the way.** Look at the demo's second table: with a context of 6, the corpus
covers 0.00038% of the possible combinations. The model works by pure memorization, and what
it is doing with 283,313 contexts is essentially copying literal fragments.

That is the argument of the whole course. Counting gives decent results fast and smashes
into an exponential wall. Learning representations costs more, but it scales.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
def next_token_probs(counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        raise ValueError("cannot normalize an empty count table")
    return {token: count / total for token, count in counts.items()}


def sample_next_token(probs: Mapping[str, float], rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    r = rng.random()
    running = 0.0
    last = ""
    for token, p in probs.items():
        running += p
        last = token
        if r < running:
            return token
    # We only get here from floating-point rounding error (running = 0.9999...).
    return last


def generate_naive(
    table: dict[str, dict[str, int]],
    start: str,
    length: int = 200,
    rng: random.Random | None = None,
) -> str:
    rng = rng or random.Random()
    context_size = len(start)
    out = list(start)

    for _ in range(max(0, length - len(start))):
        context = "".join(out[-context_size:])
        counts = table.get(context)
        if not counts:
            break  # unknown context: the model does not know how to continue
        out.append(sample_next_token(next_token_probs(counts), rng))

    return "".join(out)
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
