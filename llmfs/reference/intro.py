"""Reference for module 00: a toy language model, with no neural networks.

There is no torch here, no matrices, no gradients. Just dictionaries and one division. The
goal is for you to see with your own eyes that a language model is, literally, a thing that
says "after this, this other thing comes with this probability".

Everything we do in the next 17 modules is the same idea, with better ways of estimating
those probabilities.
"""

from __future__ import annotations

import random
from typing import Mapping, Sequence

#: A count table: for each context, how many times each character followed it.
#: For example `{"a": {"b": 3, "c": 1}}` means "after 'a' came 'b' three times and
#: 'c' once".
CountTable = dict[str, dict[str, int]]


def build_count_table(text: str, context_size: int = 1) -> CountTable:
    """Walk the text and record which character follows each context.

    This is not an exercise: it is the scaffolding so you can focus on the three that are.
    With `context_size=1` it counts pairs of characters; with 2, triples; and so on.
    """
    table: CountTable = {}
    for i in range(len(text) - context_size):
        context = text[i : i + context_size]
        nxt = text[i + context_size]
        table.setdefault(context, {})
        table[context][nxt] = table[context].get(nxt, 0) + 1
    return table


def next_token_probs(counts: Mapping[str, int]) -> dict[str, float]:
    """Turn counts into probabilities.

    Divide each count by the total. That is all. The result sums to 1, which is the
    definition of a probability distribution.

    Example:
        {"b": 3, "c": 1}  ->  {"b": 0.75, "c": 0.25}
    """
    total = sum(counts.values())
    if total == 0:
        raise ValueError("cannot normalize an empty count table")
    return {token: count / total for token, count in counts.items()}


def sample_next_token(probs: Mapping[str, float], rng: random.Random | None = None) -> str:
    """Pick a token at random, respecting its probability.

    The method is the roulette wheel: draw a random number between 0 and 1, and accumulate
    probability until you go past it. The token you go past on is the chosen one.

        probs = {"b": 0.75, "c": 0.25}
        r = 0.61  ->  running total after "b" is 0.75 > 0.61  ->  "b" comes out
        r = 0.92  ->  running total after "b" is 0.75 < 0.92
                      running total after "c" is 1.00 > 0.92  ->  "c" comes out

    `probs` is walked in insertion order, so the same seed always gives the same result.
    """
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
    table: CountTable,
    start: str,
    length: int = 200,
    rng: random.Random | None = None,
) -> str:
    """Generate text by chaining predictions together.

    The loop is the same one your 9M GPT will use in module 14:

        1. look at the current context
        2. get a distribution over the next token
        3. sample one
        4. append it to the text and go back to step 1

    This is called **autoregressive** generation: each prediction becomes part of the input
    to the next one. It is the reason generation is slow (it cannot be parallelized over
    time) and the reason an early mistake contaminates everything that comes after it.

    Args:
        table: count table from `build_count_table`.
        start: initial context. Its length fixes the `context_size`.
        length: how many characters to generate in total, including `start`.
        rng: random generator, for reproducibility.

    Returns:
        The generated text. It stops early if an unseen context comes up.
    """
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
