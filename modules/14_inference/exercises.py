"""Module 14 - Inference and sampling.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement in order -> `llmfs check 14` -> `llmfs hint 14 -e N`
-> `SOLUTION.md` has the complete code.

WHAT YOU ARE GOING TO BUILD
===========================

How you get text out of a trained model, and how to do it fast:

    apply_repetition_penalty  (ex. 1)  breaking the loops
    top_k_filter              (ex. 2)  keeping the k best
    top_p_filter              (ex. 3)  keeping the ones that add up to p
            |
    KVCache                   (ex. 4)  saving what has already been computed
            |
            v
    generate_with_cache       (ex. 5)  the loop that puts it all together

The first three are short. Number 5 is where the difficulty is, and it has a relentless
check: with the cache it has to produce EXACTLY the same text as without it.

VOCABULARY YOU ARE GOING TO NEED
================================

- **sample**: pick the next token at random respecting its probabilities, instead of always
  taking the most likely one.
- **greedy**: always taking the most likely one. It is deterministic and it gets stuck in
  loops.
- **temperature**: dividing the logits before the softmax. Below 1 sharpens the distribution,
  above 1 flattens it.
- **top-k / top-p**: two ways of discarding the bad tokens. Top-k takes a fixed number, top-p
  a variable number depending on how sure the model is.
- **KV cache**: saving the keys and values already computed so they are not recomputed for
  every token. It turns an O(N^2) cost into O(N).
- **prefill / decode**: the two phases of generation. Prefill processes the whole prompt;
  decode goes token by token.

    llmfs demo 14     compares sampling strategies and measures the cache speedup
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float = 1.1
) -> torch.Tensor:
    """Penalizes the tokens that have already come out, to break loops.

    WHAT YOU HAVE TO WRITE
    ----------------------
    A loop over the rows of the batch. Four lines inside.

        1. A copy, so you do not modify the input:

               out = logits.clone()

        2. For each row:

               for row in range(logits.shape[0]):
                   seen = torch.unique(generated[row])
                   values = out[row, seen]
                   out[row, seen] = torch.where(
                       values > 0, values / penalty, values * penalty
                   )

        3. `return out`

    If `penalty == 1.0` you can return `logits` as it is and save yourself everything:
    dividing and multiplying by 1 does nothing.

    THE DETAIL ALMOST EVERYONE IMPLEMENTS WRONG
    -------------------------------------------
        logit > 0  ->  logit / penalty      moves it towards zero
        logit < 0  ->  logit * penalty      moves it away from zero, DOWNWARDS

    With penalty=1.1:

        +3.0  ->   3.0 / 1.1 =  2.73    less likely   OK
        -3.0  ->  -3.0 * 1.1 = -3.30    less likely   OK

    If you ALWAYS divided, the -3.0 would become -2.73, that is, the token would become MORE
    likely: exactly the opposite of penalizing it. And since negative logits are the
    overwhelming majority, you would be rewarding almost everything that already came out. The
    bug is silent: the text comes out repetitive and it looks like the parameter does nothing.

    That is why the `torch.where` is needed and a single operation will not do.

    WHY `torch.unique`
    ------------------
    It avoids penalizing twice a token that appeared twice. If you wrote
    `out[row, generated[row]] = ...` with repeated indices, PyTorch applies the assignment only
    once, non-deterministically, so it would not even accumulate properly. There are
    implementations that do accumulate on purpose; ours does not, so the effect is predictable.

    WHAT THIS REALLY IS
    -------------------
    A patch. A well-trained model should not get stuck in loops, and ours, being 9M, does.
    This covers it up: if a token has already appeared, its logit is lowered.

    It is not a method with a theoretical basis —unlike top-p, which comes from a paper with
    experiments— but a practical trick that works. With high values (>1.3) it starts to show:
    the model avoids necessary words, like articles and prepositions, which by their nature
    have to repeat.

    Args:
        logits: `(B, vocab_size)`, the logits of the next token.
        generated: `(B, T)`, the tokens generated so far.
        penalty: 1.0 does nothing. Typical values: 1.05 to 1.2.

    Returns:
        The penalized logits. It does NOT modify the input.
    """
    raise NotImplementedError("TODO: module 14, exercise 1 - apply_repetition_penalty")


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Leaves only the `k` largest logits and sets the rest to -inf.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Three lines.

        1. The edge case:

               if k <= 0 or k >= logits.shape[-1]:
                   return logits

        2. The threshold, which is the k-th largest logit:

               threshold = torch.topk(logits, k, dim=-1).values[..., -1:]

        3. The filter:

               return logits.masked_fill(logits < threshold, float("-inf"))

    TWO THINGS TO LOOK AT CLOSELY
    -----------------------------
    **The `[..., -1:]` with the colon.** `torch.topk` returns the k values sorted from largest
    to smallest, so the last one is the threshold. With `[..., -1]` (without the colon) you
    would lose that dimension and the broadcast in step 3 would not line up. With `[..., -1:]`
    you get `(B, 1)`, which broadcasts properly against `(B, vocab_size)`.

    **`<` and not `<=`.** The threshold itself has to SURVIVE: it is the k-th best and it is
    one of the k you want. With `<=` you would end up with k-1 candidates (or fewer if there
    are ties).

    WHY -inf AND NOT 0
    ------------------
    Because these numbers are going through a softmax, and `exp(-inf) = 0` exactly. Putting 0
    would not discard anything: 0 is a perfectly normal logit, and `exp(0) = 1` is a QUITE high
    probability compared to a logit of -5.

    WHAT PROBLEM IT SOLVES
    ----------------------
    With a vocabulary of 4096 there are thousands of tokens with a tiny but non-zero
    probability. Added up, that long tail can take 20% of the total mass, and every so often
    one of those comes out and derails the whole sentence. Since the model is autoregressive,
    there is no going back: it keeps generating on top of the mistake.

    Top-k cuts the tail dead.

    ITS FLAW, WHICH THE NEXT EXERCISE FIXES
    ---------------------------------------
    `k` is FIXED. If the model is dead sure about the next token (after "Once upon a", "time"
    takes 99%), k=40 lets in 39 bad alternatives. If the model genuinely hesitates between 100
    valid continuations, k=40 cuts off good options.

    That is what top-p solves, with a variable number of candidates.

    Args:
        logits: `(B, vocab_size)`.
        k: how many candidates to keep. If it is <= 0 or >= vocab_size, it filters nothing.

    Returns:
        The logits with everything but the k largest set to -inf.
    """
    raise NotImplementedError("TODO: module 14, exercise 2 - top_k_filter")


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keeps the tokens that accumulate a mass of `p`.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Six lines, and number 5 is the one that is hard to see.

        1. The edge case:

               if p >= 1.0:
                   return logits

        2. Sort from largest to smallest, keeping where each one came from:

               sorted_logits, indices = torch.sort(logits, descending=True, dim=-1)

        3. The probabilities and their cumulative sum:

               probs = F.softmax(sorted_logits, dim=-1)
               cumulative = torch.cumsum(probs, dim=-1)

        4. Mark what is left over:

               remove = cumulative - probs > p

        5. The most likely one ALWAYS stays:

               remove[..., 0] = False

        6. Put the marks back in the original order and filter:

               to_remove = remove.scatter(-1, indices, remove)
               return logits.masked_fill(to_remove, float("-inf"))

    THE EXAMPLE, FOLLOWED BY HAND
    -----------------------------
    With probs = [0.60, 0.25, 0.10, 0.03, 0.02] and p = 0.9. The column that decides is the
    cumulative BEFORE including the current token, that is `cumulative - probs`:

        prob    cumulative    cumulative - prob    > 0.9?
        0.60      0.60             0.00             no    -> in
        0.25      0.85             0.60             no    -> in
        0.10      0.95             0.85             no    -> in   <- the one that CROSSES
        0.03      0.98             0.95             YES   -> out
        0.02      1.00             0.98             YES   -> out

    It keeps 3 candidates, which add up to 0.95.

    With probs = [0.2]*5 it would keep all 5. THE NUMBER OF CANDIDATES ADAPTS TO HOW SURE THE
    MODEL IS, and that is what makes it better than top-k in practice.

    THE `- probs` OF STEP 4 IS AN EASY OFF-BY-ONE TO GET WRONG
    ----------------------------------------------------------
    Holtzman's definition is "the smallest set whose cumulative probability EXCEEDS p". Look at
    the example: 0.60 + 0.25 = 0.85, which does NOT exceed 0.9. You need the third one.

    If you compared `cumulative > p` plainly, the token that crosses the threshold would be
    left out and you would cut one too few. It is a one-candidate error that breaks nothing and
    is only detected by counting.

    STEP 5 IS NOT OPTIONAL
    ----------------------
    With p=0.5 and a token of probability 0.9, the cumulative-before of the first token is
    0.00, which does not exceed 0.5... but the second one has a cumulative-before of 0.90, and
    so do all the others. Worse: if the first token had probability 0.95 and p were 0.5,
    without that line you could end up with NO candidates at all and `torch.multinomial` would
    blow up with an incomprehensible error. There is a test that checks it.

    STEP 6 IS THE HARDEST ONE TO SEE
    --------------------------------
    You sorted the logits, so your "remove" marks are in order of PROBABILITY, not in order of
    TOKEN. They have to be put back in place.

    `remove.scatter(-1, indices, remove)` does exactly that: for every position `j` of the
    sorted tensor, it writes `remove[j]` at position `indices[j]` of the result. And `indices`
    is precisely what `torch.sort` returned: where each element came from.

    If you skip it, you will be setting to -inf tokens chosen almost at random, and the result
    will be bad text with no visible error.

    Args:
        logits: `(B, vocab_size)`.
        p: the mass to keep. Typical: 0.9 or 0.95. If it is >= 1.0, it filters nothing.

    Returns:
        The logits with the tail tokens set to -inf.
    """
    raise NotImplementedError("TODO: module 14, exercise 3 - top_p_filter")


class KVCache:
    """Saves the keys and values already computed so they are not recomputed.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Five methods, all of one or two lines.

        def __init__(self, n_layers):
            self.n_layers = n_layers
            self.keys = [None] * n_layers
            self.values = [None] * n_layers

        def update(self, layer, k, v):
            if self.keys[layer] is None:
                self.keys[layer] = k
                self.values[layer] = v
            else:
                self.keys[layer] = torch.cat([self.keys[layer], k], dim=-2)
                self.values[layer] = torch.cat([self.values[layer], v], dim=-2)
            return self.keys[layer], self.values[layer]      # the COMPLETE ones

        @property
        def seq_len(self):
            return 0 if self.keys[0] is None else self.keys[0].shape[-2]

        def reset(self):
            self.keys = [None] * self.n_layers
            self.values = [None] * self.n_layers

        def memory_bytes(self):
            return sum(
                t.numel() * t.element_size()
                for t in (*self.keys, *self.values)
                if t is not None
            )

    THE `dim=-2`, WHICH IS NOT A WHIM
    ---------------------------------
    With the shape `(B, n_heads, T, head_dim)`, the time dimension is the second-to-last.
    `dim=-2` points at it counting from the end. `dim=2` would work exactly the same here, and
    breaks silently the day someone adds or removes a dimension. Count from where the shape is
    stable.

    THAT `update` RETURNS THE **COMPLETE** TABLES IS THE POINT
    ---------------------------------------------------------
    You pass it the K and V of the NEW token (T=1) and it gives you back those of ALL the
    tokens seen. Attention needs the complete ones: the new token's query has to be able to
    look at all the previous ones. The only thing you save is COMPUTING them again.

    THE PROBLEM IT SOLVES
    ---------------------
    When generating token 100, the naive version passes all 100 tokens through the whole model
    again, even though the first 99 have not changed a single bit. Generating N tokens costs
    O(N²) instead of O(N).

    WHY K AND V ARE CACHED BUT NOT Q
    --------------------------------
    The queries CANNOT be cached: every new token needs to ask its own question, and that
    question did not exist before. What gets reused are the answers (K) and the contents (V) of
    the previous tokens, which were already computed and do not change. Hence the name: KV
    cache, not QKV cache.

    THE MEMORY IT TAKES
    -------------------
        2 * n_layers * T * d_model * bytes_per_number

    Our model with 512 tokens in fp16: 3.9 MB. That is, nothing.

    A 70B model with a 100,000-token context: tens of gigabytes, MORE than the model weights
    themselves. That is why techniques like grouped-query attention exist, which share the K
    and V across several heads precisely so this table fits.

    Args (of `__init__`):
        n_layers: how many layers the model has.
    """

    def __init__(self, n_layers: int) -> None:
        raise NotImplementedError("TODO: module 14, exercise 4 - KVCache.__init__")

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TODO: module 14, exercise 4 - KVCache.update")

    @property
    def seq_len(self) -> int:
        raise NotImplementedError("TODO: module 14, exercise 4 - KVCache.seq_len")

    def reset(self) -> None:
        raise NotImplementedError("TODO: module 14, exercise 4 - KVCache.reset")

    def memory_bytes(self) -> int:
        raise NotImplementedError("TODO: module 14, exercise 4 - KVCache.memory_bytes")


@torch.no_grad()
def generate_with_cache(
    model: Any,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    eos_token: int | None = None,
) -> torch.Tensor:
    """The generation loop, with a cache and with all the filters.

    It is the same autoregressive loop from module 00 —context, distribution, sample, append,
    repeat— now with a real model.

    WHAT YOU HAVE TO WRITE
    ----------------------
        1. The checks and the setup:

               model.eval()
               ctx = model.cfg.context_length
               if idx.shape[1] >= ctx:
                   raise ValueError(
                       f"the prompt already takes {idx.shape[1]} of {ctx} context tokens"
                   )
               cache = KVCache(model.cfg.n_layers)

        2. PREFILL: the whole prompt at once, to fill the cache:

               logits, _ = model(idx, use_cache=True, cache=cache)

        3. The loop, `max_new_tokens` times:

               for _ in range(max_new_tokens):
                   if idx.shape[1] >= ctx:
                       break                          # see below why it STOPS

                   next_logits = logits[:, -1, :].float()   # only the last position

                   # the filters, IN THIS ORDER
                   if repetition_penalty != 1.0:
                       next_logits = apply_repetition_penalty(
                           next_logits, idx, repetition_penalty
                       )
                   if temperature != 1.0:
                       next_logits = next_logits / max(temperature, 1e-8)
                   if top_k is not None:
                       next_logits = top_k_filter(next_logits, top_k)
                   if top_p is not None:
                       next_logits = top_p_filter(next_logits, top_p)

                   # choose
                   if temperature == 0:
                       new = next_logits.argmax(dim=-1, keepdim=True)
                   else:
                       new = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)

                   idx = torch.cat([idx, new], dim=1)

                   if eos_token is not None and bool((new == eos_token).all()):
                       break

                   # DECODE: only the new token
                   logits, _ = model(new, use_cache=True, cache=cache)

        4. `return idx`

    THE TWO PHASES, WHICH IS WHAT YOU HAVE TO UNDERSTAND
    ----------------------------------------------------
    **Prefill**: the whole prompt goes through at once. It is a single forward over
    `(B, T_prompt)` and it fills the cache with the K and V of all those tokens.

    **Decode**: from then on, each turn passes ONLY the new token, `(B, 1)`. The cache provides
    everything before. This is the entire saving.

    Note the order inside the loop: the `logits` you start turn N with are the ones produced by
    the forward at the end of turn N-1 (or by the prefill, the first time). That is why the
    `model(...)` goes at the END and not at the beginning.

    THE ORDER OF THE FILTERS MATTERS
    --------------------------------
        1. repetition penalty   <- on the raw logits
        2. temperature          <- divide
        3. top-k
        4. top-p

    The temperature goes BEFORE the filters because it changes the probabilities top-p looks
    at. It does not change the RANKING —dividing by a positive constant does not reorder
    anything— but it does change the accumulated masses, and with them how many candidates
    survive. With temperature=0.7 the top-p nucleus is smaller than with 1.0.

    THE `.float()` ON THE LOGITS
    ----------------------------
    Under AMP the logits arrive in fp16, and `torch.multinomial` on fp16 can give odd results
    with very small probabilities (there is little resolution near zero). Converting to fp32
    before sampling is cheap and avoids the problem.

    THE CONTEXT LIMIT: HERE IT **STOPS**, IT DOES NOT CROP
    ------------------------------------------------------
    `model.generate` (the naive one) crops the context and carries on. This one cannot, and it
    is not laziness.

    Cropping with a cache would mean discarding the old entries AND REMAPPING the RoPE
    positions of everything that is left, because the surviving tokens would end up in
    different positions from the ones they had when their K were computed. That is called
    sliding window attention and it is worth a module of its own.

    Stopping is the honest thing: the silent alternative would be generating incorrect text
    without warning.

    And the `ValueError` above, for when the prompt ALREADY reaches the limit: better a clear
    error than an immediate `break` that returns the prompt untouched without explaining why.

    THE MANDATORY CHECK
    -------------------
    With `temperature=0` (greedy, deterministic), this function has to give EXACTLY the same
    output as generation without the cache. Not similar: identical, token by token. There is a
    test that verifies it, and it is the only way to know the cache is not corrupting anything.

    If it differs from a certain token on, look first at RoPE's `pos_offset`: the new token has
    to be rotated with the position it belongs to, not with 0.

    Args:
        model: the trained GPT. Its forward accepts `use_cache` and `cache`.
        idx: `(B, T)`, the prompt.
        max_new_tokens: how many tokens to generate at most.
        temperature: 0 for greedy. 1.0 changes nothing. <1 sharpens, >1 flattens.
        top_k, top_p: the filters, or None to skip them.
        repetition_penalty: 1.0 does nothing.
        eos_token: if it appears, generation stops.

    Returns:
        `(B, T + n)` with the prompt and what was generated, with n <= max_new_tokens.

    Raises:
        ValueError: if the prompt already fills the model's context.
    """
    raise NotImplementedError("TODO: module 14, exercise 5 - generate_with_cache")
