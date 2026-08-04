"""Module 15 - Evaluation.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement -> `llmfs check 15` -> `llmfs hint 15 -e N`
-> `SOLUTION.md` has the complete code.

The three exercises are short. The third one does not even compute anything: it organizes
the work so that the part that really matters (reading what the model writes) is comfortable.

WHAT YOU ARE GOING TO BUILD
===========================

Three ways of answering "is my model any good?":

    perplexity_from_loss  (ex. 1)  the usual metric, and its limits
    bits_per_byte         (ex. 2)  the one that CAN be compared across models
    run_prompt_battery    (ex. 3)  the one no automatic metric replaces

VOCABULARY YOU ARE GOING TO NEED
================================

- **perplexity**: `e` raised to the loss. "How many options the model is hesitating
  between". It depends on the tokenizer, and that is why comparing perplexities between
  different models usually means nothing.
- **bits per byte**: the loss normalized by bytes of original text instead of by tokens. It
  IS comparable, and it is read as how much the model would compress the text.
- **nat / bit**: units of information. One nat is 1.4427 bits.
- **qualitative evaluation**: reading what the model writes and judging it. It is still
  indispensable.
- **contamination**: when the test set appears in the training data. It is one of the most
  serious methodological problems in the field right now.

    llmfs demo 15     evaluates your model and generates eval_report.md
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

# The six prompts of the TinyStories paper battery. Each one tests something different.
from llmfs.reference import PROMPTS_TINYSTORIES


def perplexity_from_loss(loss: float) -> float:
    """Perplexity from the mean loss in nats.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Two lines.

        1. The guard:

               if not math.isfinite(loss):
                   return float("inf")

        2. The formula:

               return math.exp(loss)

    WHAT SHOULD COME OUT
    --------------------
        loss 8.317  ->  perplexity 4096.0    untrained: hesitates over the WHOLE vocabulary
        loss 1.60   ->  perplexity    4.95   hesitates between about 5 options
        loss 0.0    ->  perplexity    1.0    perfect, no hesitation

    The first one is worth checking: `ln(4096) = 8.317`, so `exp(8.317) = 4096`. A freshly
    initialized model spreads the probability equally over the 4096 tokens, and the perplexity
    tells you so literally: "I am hesitating between 4096 options".

    HOW TO READ IT
    --------------
    The loss in nats does not read well: 1.60 says nothing on its own. Perplexity does, because
    it has understandable units: HOW MANY EQUIPROBABLE OPTIONS the model is hesitating between.

    A perplexity of 5 means "on average, the model is as undecided as if it had to choose at
    random between 5 words". It is a measure of surprise.

    WHY THE GUARD
    -------------
    Without it, `math.exp(float("inf"))` raises `OverflowError` and you are left not knowing
    what happened. And an `inf` or a `nan` in the loss DOES happen: it is exactly what you see
    when training diverges. Returning `inf` is the correct answer —the perplexity of a broken
    model IS infinite— and besides it prints without breaking anything.

    `math.isfinite(x)` is False for `inf`, `-inf` and `nan`.

    It is one line of code. What matters is knowing how to read it, and knowing it CANNOT be
    compared between models with different tokenizers, which is the next exercise.

    Args:
        loss: the MEAN loss per token, in nats (what `F.cross_entropy` returns).

    Returns:
        The perplexity. `inf` if the loss is not finite.
    """
    raise NotImplementedError("TODO: module 15, exercise 1 - perplexity_from_loss")


def bits_per_byte(total_loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    """Bits per byte: the metric that CAN be compared across different tokenizers.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Two lines.

        1. The validation:

               if n_bytes <= 0:
                   raise ValueError(f"n_bytes has to be positive: {n_bytes}")

        2. The formula:

               return (total_loss_nats / math.log(2)) / n_bytes

    THE PROBLEM WITH PERPLEXITY
    ---------------------------
    It depends on the tokenizer. If your vocabulary splits words into smaller pieces, each
    individual token is easier to predict and your perplexity comes out BETTER without the
    model being better.

    An extreme case to see it clearly: a model predicting bit by bit would have a perplexity
    close to 2 (there are only two options, 0 and 1) and would be completely useless. Its
    perplexity would be the best in the world.

    Comparing perplexities between models with different tokenizers means nothing, and it is
    done constantly in blogs and papers.

    THE SOLUTION
    ------------
    Normalize by BYTES of the original text instead of by tokens. Bytes do not depend on how
    you chop things up: the same text has the same bytes with any tokenizer.

    The `/ math.log(2)` converts nats into bits: one nat is `1/ln(2) = 1.4427` bits. It is just
    a change of units, like going from metres to feet.

    THE INTERPRETATION, WHICH IS RATHER NICE
    ----------------------------------------
    It is literally how many bits you would need to transmit the text using the model as a
    compressor. A model at 1.0 bits/byte compresses the text to an EIGHTH (a byte is 8 bits).

        gzip on English text          ~2.5 bits/byte
        a good small model            ~1.2
        the best LLMs                  0.6 - 0.8

    This equivalence between prediction and compression comes from Shannon (1948) and it is not
    an analogy: it is a mathematical identity. A language model IS a compressor, and a good
    compressor IS a language model. Predicting well and compressing well are the same thing.

    WATCH OUT FOR THE FIRST ARGUMENT
    --------------------------------
    It is the TOTAL loss (the sum over all the tokens), NOT the mean. If you pass it the mean,
    the result comes out divided by the number of tokens and means nothing, and besides it
    raises no error: you get a plausible number that is a thousand times too small.

    To get it: `mean_loss * n_tokens`, or accumulate `loss.item() * y.numel()` in the
    evaluation loop.

    And `n_tokens` is NOT used in the computation. It is in the signature precisely to make
    clear what unit the first argument is in, and so you can verify at a glance that the loss
    is total and not mean. It is executable documentation.

    Args:
        total_loss_nats: the TOTAL loss in nats, summed over all the tokens.
        n_tokens: how many tokens. Not used; it is there to make the unit clear.
        n_bytes: the bytes of the original text (`len(text.encode("utf-8"))`).

    Returns:
        Bits per byte.

    Raises:
        ValueError: if `n_bytes` is not positive.
    """
    raise NotImplementedError("TODO: module 15, exercise 2 - bits_per_byte")


def run_prompt_battery(
    generate_fn: Callable[[str], str],
    prompts: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Generates a completion for each prompt of the qualitative battery.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Three lines.

        1. The default prompts:

               if prompts is None:
                   prompts = PROMPTS_TINYSTORIES

        2. The walk, with the three EXACT keys (there is a test that checks them):

               return [
                   {
                       "prompt": prompt,
                       "tests": label,
                       "completion": generate_fn(prompt),
                   }
                   for prompt, label in prompts
               ]

    `PROMPTS_TINYSTORIES` is a tuple of `(prompt, label)` tuples, so it unpacks directly in the
    `for`.

    WHAT THIS IS
    ------------
    The part of evaluation no automatic metric replaces: asking the model always the SAME
    questions and READING what it answers.

    The six prompts of `PROMPTS_TINYSTORIES` come from the TinyStories paper and each one tests
    something different: narrative continuation, causal coherence, tracking an object mentioned
    earlier, resolving and closing the story. They are not random prompts.

    WHY `generate_fn` IS PASSED INSTEAD OF THE MODEL
    ------------------------------------------------
    `generate_fn(prompt) -> text` encapsulates the model AND the tokenizer. That way this
    function knows nothing about either of them, and you can use it just the same with your
    model, with one loaded from a checkpoint, with GPT-4 over an API, or with a fake function
    to test it without training anything.

    It is the same pattern as `get_batch` in module 04 or `optimizer_factory` in module 13:
    passing the capability as a function instead of coupling to a concrete object.

    WHERE THE VALUE OF THIS EXERCISE IS
    -----------------------------------
    It is not in the code, which is three lines. It is in having a FIXED battery you can run
    again every time you change something, and compare the answers side by side.

    Without a fixed battery, qualitative evaluation turns into writing prompts until you find
    one where the model looks good. With it, you see when a new run gets worse at something
    that used to come out well, which is the information you actually need.

    Args:
        generate_fn: `fn(prompt) -> generated text`.
        prompts: sequence of `(prompt, label)`, or None to use the TinyStories battery.

    Returns:
        A list of dicts with the keys `prompt`, `tests` and `completion`.
    """
    raise NotImplementedError("TODO: module 15, exercise 3 - run_prompt_battery")
