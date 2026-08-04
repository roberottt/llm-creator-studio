"""Module 00 - What an LLM actually is.

HOW TO DO THIS MODULE
=====================

1. Read `THEORY.md`. It is 10 minutes and without it these exercises make no sense.
2. Implement the functions below, in order. Each one uses the previous.
3. `llmfs check 00` to see how they are doing.
4. Stuck? `llmfs hint 00 -e 1` (three levels, each more explicit).
5. Still stuck? `SOLUTION.md` has the complete code. Copy it, see how it works, and then
   come back and write it yourself. That is not cheating.

WHAT YOU ARE GOING TO BUILD
===========================

A text generator that works. No torch, no matrices, no derivatives: dictionaries and one
division.

The three functions fit together like this:

    build_count_table   (already done)  text -> count table
            |
            v
    next_token_probs    (exercise 1)    counts -> probabilities
            |
            v
    sample_next_token   (exercise 2)    probabilities -> ONE character
            |
            v
    generate_naive      (exercise 3)    all of the above, in a loop -> text

Exercise 3 is the interesting one: that loop is EXACTLY the same one ChatGPT runs.

VOCABULARY YOU ARE GOING TO NEED
================================

- **token**: the unit of text the model handles. Here, a character.
- **context**: the preceding characters the model looks at to decide the next one.
- **probability distribution**: a list of non-negative numbers that sum to 1.
- **sample**: pick one at random, respecting those probabilities.
"""

from __future__ import annotations

import random
from typing import Mapping

# The count table is already written: it is not what you learn here. It walks a text and
# notes which character followed each character. Example with "banana":
#     build_count_table("banana") -> {'b': {'a': 1}, 'a': {'n': 2}, 'n': {'a': 2}}
from llmfs.reference import build_count_table


def next_token_probs(counts: Mapping[str, int]) -> dict[str, float]:
    """Turns a count table into probabilities.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Three lines.

        1. Add up every count:

               total = sum(counts.values())

        2. If the total is 0, raise `ValueError`. (See below for why.)

        3. Return a dictionary with each count divided by the total:

               return {token: count / total for token, count in counts.items()}

    EXAMPLE TO CHECK AGAINST
    ------------------------
        input :  {"b": 3, "c": 1}
        total :  3 + 1 = 4
        output:  {"b": 0.75, "c": 0.25}

    WHAT YOU ARE DOING AND WHY
    --------------------------
    You have recorded how many times each character followed a context: 40, 25, 20, 15.
    Those numbers mean nothing on their own, because they depend on how long the text was.

    What you need is the PROPORTION, and those proportions have to sum to 1 because
    something always had to come next.

    This is called NORMALIZING and you are going to see it a thousand times in the course.
    The `softmax` function in module 06 does exactly this, only exponentiating first so it
    works with negative numbers.

    TWO DETAILS THAT MATTER
    -----------------------
    **The ValueError.** If the table comes in empty, `sum()` gives 0 and the division blows
    up with `ZeroDivisionError`. The problem is not that it blows up: it is WHERE. Without
    the check, the error fires inside a dict comprehension, three levels below the real
    cause, and the message never mentions that the problem is an empty table.

    **The key order.** Return them in the SAME order they arrived in. If you walk
    `counts.items()` to build the result, the order is preserved for free (in python 3.7+
    dictionaries keep it). Do not sort them alphabetically: exercise 2 walks this dictionary
    and the order changes which character comes out.

    Args:
        counts: `{character: times_it_appeared}`, with integer counts >= 0.

    Returns:
        `{character: probability}`, with the same keys and in the same order.

    Raises:
        ValueError: if the total is 0.
    """
    raise NotImplementedError("TODO: module 00, exercise 1 - next_token_probs")


def sample_next_token(probs: Mapping[str, float], rng: random.Random | None = None) -> str:
    """Picks a character at random, respecting its probability.

    WHAT YOU HAVE TO WRITE
    ----------------------
    A loop with a running total. Five steps.

        1. If `rng` is None, create one: `rng = rng or random.Random()`

        2. Draw a random number between 0 and 1:

               r = rng.random()

        3. Set up a running total at 0.0 and a variable to remember the last token seen.

        4. Walk `probs.items()`. On each pass:
             - add the probability to the running total
             - store this token as "the last one seen"
             - if `r < running`, RETURN this token

        5. If the loop ends without returning anything, return the last token seen.
           (See below for why that is needed.)

    THE METHOD, DRAWN OUT
    ---------------------
    Picture the line from 0 to 1 split into slices, one per character, of size proportional
    to its probability:

        |----'n'----|--'r'--|--' '--|-'s'-|
        0          0.40    0.65    0.85   1.0

    You draw a random number and see which slice it lands in. With r = 0.61:

        'n' -> running 0.40 ;  0.61 < 0.40 ? NO, keep going
        'r' -> running 0.65 ;  0.61 < 0.65 ? YES, return 'r'

    WHY NOT ALWAYS TAKE THE MOST LIKELY ONE
    ---------------------------------------
    Because the text comes out repetitive and loops. You will see it measured in module 14:
    always taking the most likely one produces things like "the cat sat on the mat. the cat
    sat on the mat."

    THREE DETAILS THAT MATTER
    -------------------------
    **Use `<` and not `<=`.** With `{a: 0.5, b: 0.5}` and `r = 0.5` exactly: with `<`, after
    'a' the running total is 0.5 and `0.5 < 0.5` is false, so it carries on and returns 'b'.
    That is correct: 'a' occupies the interval [0, 0.5) and 'b' occupies [0.5, 1). Since
    `rng.random()` returns a number in [0, 1) — 1 never comes out, 0 does — that split gives
    exactly 50/50.

    **Step 5 is not paranoia.** Floats do not add up exactly: try `sum([0.1] * 10)` in an
    interpreter and you get 0.9999999999999999. If `rng.random()` returns
    0.99999999999999995, the loop ends without returning anything and the function returns
    `None`, which breaks exercise 3 with an incomprehensible error several steps later.

    **Walk `probs` in its natural order**, without sorting. The reference does the same, so
    with the same seed both generate exactly the same text and the test can compare them.

    Args:
        probs: `{character: probability}`, as returned by exercise 1.
        rng: random generator. Use `rng.random()`, which gives a float in [0, 1). If it is
            `None`, create one with `random.Random()`.

    Returns:
        One of the characters in `probs`.
    """
    raise NotImplementedError("TODO: module 00, exercise 2 - sample_next_token")


def generate_naive(
    table: dict[str, dict[str, int]],
    start: str,
    length: int = 200,
    rng: random.Random | None = None,
) -> str:
    """Generates text by chaining predictions. This is where the language model appears.

    WHAT YOU HAVE TO WRITE
    ----------------------
    A loop that uses the two previous functions.

        1. If `rng` is None, create one.

        2. Store the context size and start the output:

               context_size = len(start)
               out = list(start)

        3. Repeat `max(0, length - len(start))` times:

             a. Take the last `context_size` characters of what you have so far:

                    context = "".join(out[-context_size:])

             b. Look it up in the table:

                    counts = table.get(context)

             c. If there is nothing (`if not counts`), leave the loop with `break`.

             d. If there is, convert to probabilities, sample, and append:

                    out.append(sample_next_token(next_token_probs(counts), rng))

        4. Return `"".join(out)`.

    WHAT YOU ARE BUILDING
    ---------------------
    This loop is called AUTOREGRESSIVE generation ("auto" = itself, "regressive" = it feeds
    back into itself): each character you draw becomes part of the input to the next step.

    And it is EXACTLY the same loop you are going to implement in module 14 with your 9
    million parameter GPT. The only thing that will change is where the probabilities come
    from: here from a count table, there from a neural network.

    WHY THE `break` IN STEP 3c
    --------------------------
    The table only knows the contexts that appeared in the training text. If you generate
    one that was never seen, there is nothing to look up: a counting model goes literally
    mute.

    A neural network NEVER has that problem, because it does not look anything up: it
    computes. Whatever you give it, it produces a distribution. It may be a bad one, but it
    exists. That is one of the deep reasons networks are used.

    TWO DETAILS THAT MATTER
    -----------------------
    **`length` counts the total returned, INCLUDING `start`.** If `start` has 2 characters
    and 5 are asked for, you generate 3, not 5. That is why the loop iterates
    `length - len(start)` times.

    **Accumulate into a list and join at the end** with `"".join()`. Doing `out = out + c`
    with strings creates a new string on every pass. Here it does not matter, but it is a
    habit that gets expensive in module 14.

    Args:
        table: `{context: {next: times}}`, from `build_count_table`.
        start: the initial text. Its length defines the context size.
        length: the TOTAL length of the text to return, counting `start`.
        rng: random generator, so the test can reproduce the result.

    Returns:
        The generated text, as a single string.
    """
    raise NotImplementedError("TODO: module 00, exercise 3 - generate_naive")
