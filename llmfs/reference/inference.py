"""Reference for module 14: sampling and the KV cache."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float = 1.1
) -> torch.Tensor:
    """Penalize tokens that have already appeared, to break loops.

    The detail almost everyone gets wrong: you have to DIVIDE if the logit is positive and
    MULTIPLY if it is negative.

        logit > 0  ->  logit / penalty     moves it towards zero
        logit < 0  ->  logit * penalty     moves it away from zero, downwards

    If you always divided, a logit of -5 would become -4.5, which makes the token MORE
    likely: exactly the opposite of penalizing it.

    With `penalty=1.0` it does nothing. Typical values: 1.05 to 1.2.
    """
    if penalty == 1.0:
        return logits

    out = logits.clone()
    for row in range(logits.shape[0]):
        seen = torch.unique(generated[row])
        values = out[row, seen]
        out[row, seen] = torch.where(values > 0, values / penalty, values * penalty)
    return out


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the `k` largest logits and set the rest to -inf.

    This cuts off the long tail of the distribution. With a 4096-token vocabulary there are
    thousands of tokens with tiny but non-zero probability; added up, that tail can carry
    20% of the mass, and every so often one of them comes out and derails the sentence.

    `k <= 0` or `k >= vocab_size` filters nothing.
    """
    if k <= 0 or k >= logits.shape[-1]:
        return logits

    threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the tokens that accumulate a mass of `p`.

    Unlike top-k, the number of candidates is VARIABLE:

    - if the model is very sure, a single token can hold 90% and it is kept alone
    - if it is torn between many, many are kept

    That is what makes it better than top-k in practice (Holtzman et al. 2020): it adapts to
    how sure the model is at each position.

    The detail that matters: the first token is ALWAYS kept, even if on its own it already
    exceeds `p`. Otherwise, with p=0.5 and a token of probability 0.9 you would be left with
    no candidates at all.
    """
    if p >= 1.0:
        return logits

    sorted_logits, indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    drop = cumulative - F.softmax(sorted_logits, dim=-1) > p
    drop[..., 0] = False  # the most likely one is always kept

    to_drop = drop.scatter(-1, indices, drop)
    return logits.masked_fill(to_drop, float("-inf"))


class KVCache:
    """Stores the keys and values already computed, so they are not recomputed.

    THE PROBLEM. When generating token 100, the naive version runs all 100 tokens through
    the model again, even though the first 99 have not changed. Generating N tokens costs
    O(N^2) instead of O(N).

    THE SOLUTION. Store each layer's keys and values and, at every step, process ONLY the
    new token, concatenating its K and V onto what is stored.

    WHAT CANNOT BE CACHED: the queries. Each new token needs its own query to ask with; what
    gets reused are the answers (K) and the contents (V) of the previous ones.

    THE MEMORY:  2 * n_layers * T * d_model * bytes

    For our model with 512 tokens in fp16 that is 3.9 MB. For a 70B model with a
    100,000-token context, tens of gigabytes, which is why techniques like grouped-query
    attention exist.
    """

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self.keys: list[torch.Tensor | None] = [None] * n_layers
        self.values: list[torch.Tensor | None] = [None] * n_layers

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append the new token's K and V and return the full sequence.

        Args:
            layer: the layer index.
            k, v: `(B, n_heads, T_new, head_dim)`.

        Returns:
            The full `(K, V)`, `(B, n_heads, T_total, head_dim)`.
        """
        if self.keys[layer] is None:
            self.keys[layer] = k
            self.values[layer] = v
        else:
            # dim=-2 is the time dimension with the shape (B, heads, T, head_dim)
            self.keys[layer] = torch.cat([self.keys[layer], k], dim=-2)
            self.values[layer] = torch.cat([self.values[layer], v], dim=-2)
        return self.keys[layer], self.values[layer]

    @property
    def seq_len(self) -> int:
        """How many tokens are stored."""
        return 0 if self.keys[0] is None else self.keys[0].shape[-2]

    def reset(self) -> None:
        self.keys = [None] * self.n_layers
        self.values = [None] * self.n_layers

    def memory_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in [*self.keys, *self.values]
            if t is not None
        )


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
    """Generate text using the KV cache.

    The loop has two phases:

    1. **Prefill**: the whole prompt is passed in at once and the cache is filled.
    2. **Decode**: at each step ONLY the last token is passed in, the cache is read, and the
       new token is appended.

    The order of the filters matters, and it is this one: penalty -> temperature -> top-k ->
    top-p. The penalty goes first because it operates on the raw logits; the temperature
    goes before the filters because it changes the cumulative probabilities top-p looks at.

    CONTEXT LIMIT. This implementation STOPS when it reaches the model's maximum context,
    instead of truncating the way `model.generate` does.

    This is not laziness: truncating with a cache is genuinely more complicated. You would
    have to drop the old entries AND remap the RoPE positions of everything that remains,
    because the surviving tokens would end up at different positions. It is called sliding
    window attention and it is worth a whole module.

    Stopping is the honest option: the silent alternative would be generating incorrect text
    without any warning.
    """
    model.eval()
    cache = KVCache(model.cfg.n_layers)
    max_context = model.cfg.context_length

    if idx.shape[1] >= max_context:
        raise ValueError(
            f"the prompt already has {idx.shape[1]} tokens and the model's context is "
            f"{max_context}: there is no room left to generate"
        )

    logits, _ = model(idx, use_cache=True, cache=cache)

    for _ in range(max_new_tokens):
        if idx.shape[1] >= max_context:
            break
        next_logits = logits[:, -1, :].float()

        if repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(next_logits, idx, repetition_penalty)
        if temperature != 1.0:
            next_logits = next_logits / max(temperature, 1e-8)
        if top_k is not None:
            next_logits = top_k_filter(next_logits, top_k)
        if top_p is not None:
            next_logits = top_p_filter(next_logits, top_p)

        if temperature == 0.0:
            new_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            new_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)

        idx = torch.cat([idx, new_token], dim=1)
        if eos_token is not None and bool((new_token == eos_token).all()):
            break

        # Only the new token: the cache holds the rest.
        logits, _ = model(new_token, use_cache=True, cache=cache)

    return idx
