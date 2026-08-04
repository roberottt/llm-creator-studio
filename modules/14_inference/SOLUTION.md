# 14 — Annotated solution

## Exercise 1 — `apply_repetition_penalty`

```python
if penalty == 1.0:
    return logits

out = logits.clone()
for row in range(logits.shape[0]):
    seen = torch.unique(generated[row])
    values = out[row, seen]
    out[row, seen] = torch.where(values > 0, values / penalty, values * penalty)
return out
```

**The `torch.where` is the whole exercise.** Positives get divided, negatives get multiplied.

Check it with the demo numbers, with `penalty=2.0`:

```
logit +3.0  →  3.0 / 2.0 = +1.50    less likely  ✓
logit −3.0  →  −3.0 × 2.0 = −6.00   less likely  ✓
```

If you always divided, the −3.0 would become −1.5 and the token would become **more** likely.
And since negative logits are the majority of the vocabulary, you would be rewarding almost
everything that already came out: exactly the opposite of what you intend.

The `torch.unique` avoids penalizing twice a token that came out twice. There are
implementations that do accumulate; ours does not, so the effect is predictable.

## Exercise 2 — `top_k_filter`

```python
if k <= 0 or k >= logits.shape[-1]:
    return logits

threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
return logits.masked_fill(logits < threshold, float("-inf"))
```

**The `[..., -1:]` with the colon** keeps the dimension so the broadcast works. With
`[..., -1]` you would lose it and `masked_fill` would compare wrongly.

**`<` and not `<=`**: the threshold itself —the k-th logit— has to survive.

## Exercise 3 — `top_p_filter`

```python
if p >= 1.0:
    return logits

sorted_logits, indices = torch.sort(logits, descending=True, dim=-1)
probs = F.softmax(sorted_logits, dim=-1)
cumulative = torch.cumsum(probs, dim=-1)

drop = cumulative - probs > p
drop[..., 0] = False

to_drop = drop.scatter(-1, indices, drop)
return logits.masked_fill(to_drop, float("-inf"))
```

### The off-by-one I got wrong myself

Writing this module I put in the theory that `[0.60, 0.25, 0.10, 0.03, 0.02]` with `p=0.9`
left **2** candidates. The test said 3, and the test was right.

Holtzman's definition is *"the smallest set whose cumulative probability **exceeds** p"*. And
`0.60 + 0.25 = 0.85` does **not** exceed 0.9. The third one is needed, and it takes the mass
to 0.95.

That is why the comparison is `cumulative - probs > p`: you look at the cumulative **before**
including each token, so the one that crosses the threshold still gets in. If you compared
`cumulative > p` plainly, you would cut one too many.

It is the same criterion HuggingFace's implementation uses, which solves it by shifting the
mask one position to the right.

### The `drop[..., 0] = False`

With `p=0.5` and a token of probability 0.9, without that line no candidate would be left and
`torch.multinomial` would blow up. The most likely token **always** survives.

### The `scatter`, which is the hardest part to see

You sorted the logits, so the "drop" marks are in order of probability, not in order of token.
`scatter(-1, indices, drop)` puts them back in place: for every position `j` of the sorted
tensor, it writes its mark at position `indices[j]` of the result.

## Exercise 4 — `KVCache`

```python
def update(self, layer, k, v):
    if self.keys[layer] is None:
        self.keys[layer] = k
        self.values[layer] = v
    else:
        self.keys[layer] = torch.cat([self.keys[layer], k], dim=-2)
        self.values[layer] = torch.cat([self.values[layer], v], dim=-2)
    return self.keys[layer], self.values[layer]
```

**`dim=-2`** is the time dimension with the shape `(B, n_heads, T, head_dim)`. A negative
index: with `dim=2` it would work here and break the day the number of dimensions changes.

The rest is bookkeeping. The class is deliberately simple because the difficulty is not here,
it is in exercise 5.

## Exercise 5 — `generate_with_cache`

### The detail that breaks everything

**RoPE has to rotate the new token with the angle of its real position.**

When generating token 50 you pass it a tensor of length 1. If you apply RoPE as is, it rotates
it as if it were position 0. The result: generation with the cache produces **different and
worse** text than without it, and nothing fails — the model simply writes badly.

That is why attention receives `pos_offset` and slices the tables:

```python
cos_t = cos[pos_offset : pos_offset + seq_len]
```

The test `test_the_cache_gives_exactly_the_same_output` is what catches it, and its error
message points straight here.

### The order of the filters

```
penalty → temperature → top-k → top-p
```

The temperature goes before the filters because it changes the cumulative probabilities top-p
looks at. (It does not change the *ranking*: dividing by a positive constant does not reorder
anything.)

### The `.float()` on the logits

Under AMP the logits arrive in fp16, and `torch.multinomial` on fp16 can give odd results with
very small probabilities. Converting to fp32 before sampling is cheap.

### A bug I found writing the demo

The first version blew up when generating past the context: `model.generate` crops with
`idx[:, -context_length:]`, but with a cache that will not do.

Cropping with a cache would require discarding the old entries **and remapping the RoPE
positions** of everything that is left, because the surviving tokens would end up in different
positions from the ones they were rotated with. That is *sliding window attention* and it is
worth a module of its own.

The solution I adopted is to stop cleanly on reaching the limit, and to raise a clear
`ValueError` if the prompt already fills it. Stopping is the honest thing: the silent
alternative would be generating incorrect text without warning.

## What you should see in the demo

**Greedy's loop**, measured as the fraction of distinct 4-grams:

| strategy | variety | text |
|---|---|---|
| greedy (T=0) | **29%** | `The king of the sea of the sea That shall see the sea of the sea` |
| T=0.8 + top-k 40 | 96% | `The king; To bring what heart you but dead-look'd me to-morrow` |
| T=1.5 | 100% | `Tak't I am fan undooses our very looks, Were stewest, grounde;` |
| greedy + penalty 1.3 | 93% | `The king, As to my lady's brother with the prince.` |

Greedy gets stuck on `the sea of the sea` in a perfectly visible way. **Human text does not
maximize probability**, and that is Holtzman et al.'s central observation.

Note the last row too: the penalty rescues greedy without taking away its determinism. And
T=1.5, where 100% variety is a sign that it is already rambling.

**And the cache:**

```
without cache: [43, 1, 57, 43, 39, 0, 32, 46, 39, 58]
with cache:    [43, 1, 57, 43, 39, 0, 32, 46, 39, 58]   IDENTICAL
```

| tokens | without cache | with cache | speedup |
|---|---|---|---|
| 50 | 159 ms | 133 ms | 1.20x |
| 200 | 1115 ms | 833 ms | 1.34x |
| 800 | 4330 ms | 1962 ms | **2.21x** |

**The speedup grows with the length**, which is exactly what the analysis predicts: without a
cache it is $O(N^2)$ and with one $O(N)$. With the short sequences in this example the gain is
modest; with contexts of thousands of tokens, the difference is of another order.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
def apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float = 1.1
) -> torch.Tensor:
    if penalty == 1.0:
        return logits

    out = logits.clone()
    for row in range(logits.shape[0]):
        seen = torch.unique(generated[row])
        values = out[row, seen]
        out[row, seen] = torch.where(values > 0, values / penalty, values * penalty)
    return out


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits

    threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits

    sorted_logits, indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    drop = cumulative - F.softmax(sorted_logits, dim=-1) > p
    drop[..., 0] = False  # the most likely one is always kept

    to_drop = drop.scatter(-1, indices, drop)
    return logits.masked_fill(to_drop, float("-inf"))


class KVCache:

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self.keys: list[torch.Tensor | None] = [None] * n_layers
        self.values: list[torch.Tensor | None] = [None] * n_layers

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
