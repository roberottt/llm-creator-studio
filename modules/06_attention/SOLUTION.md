# 06 — Annotated solution

## Exercise 1 — `causal_mask`

```python
return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
```

`tril` = *triangular lower*. By default it uses `diagonal=0`, which **includes** the
diagonal, and that is what you want: a token can look at itself.

If it comes out inverted, you used `triu`. If the diagonal comes out `False`, you passed
`diagonal=-1`.

**About the `True = allowed` convention.** It is the one
`F.scaled_dot_product_attention` uses with boolean masks, which is why we follow it. But
**`nn.MultiheadAttention` uses the opposite**: its boolean `attn_mask` marks with `True`
what has to be *forbidden*. That is why the test comparing against PyTorch passes `~mask`.
It is a real inconsistency inside the library itself and a classic source of bugs.

## Exercise 2 — `single_head_attention`

```python
d_k = q.shape[-1]
scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))
weights = F.softmax(scores, dim=-1)
return weights @ v, weights
```

Four lines, and each one has a trap.

**`transpose(-2, -1)` and not `transpose(1, 2)`.** Negative indices count from the end, so
they work the same with `(B, T, d)` as with `(B, heads, T, d)`. If you write positive
indices, exercise 2 passes and exercise 3 fails with a shape error that is hard to connect
back to the cause.

**`dim=-1` in the softmax.** You are normalizing over *who is being looked at*, so each row
sums to 1. With `dim=-2` you would normalize over *who is looking*, which means nothing. And
it gives no error: the shapes are identical, the model trains, and it learns worse. The test
`test_each_row_of_weights_sums_to_one` is what catches it.

**`masked_fill(~mask, -inf)` before the softmax.** The `~` inverts the boolean: where the
mask says `False` (forbidden), put `-inf`. Since $e^{-\infty} = 0$, the softmax assigns it
exactly zero weight.

It has to go **before**. If you zeroed the weights after the softmax, the rows would stop
summing to 1 and you would be scaling the output by an arbitrary factor that differs at
every position.

**The `/ math.sqrt(d_k)`.** The demo measures it: with $d_k = 2048$ and no scaling, the
attention entropy falls to 0.007 nats (out of a maximum of 2.77) and the maximum weight
reaches 1.00. The token fixates on one and ignores everything else.

But the serious part is not the forward pass, it is the gradient. The softmax's derivative
is $p(1-p)$; with $p$ pinned to 0 or 1 it is practically zero and the layer stops learning.
Without the $\sqrt{d_k}$, a large Transformer simply does not train.

## Exercise 3 — `MultiHeadAttention`

### Splitting and merging heads

```python
def _split_heads(self, x):                     # (B, T, d_model) -> (B, H, T, head_dim)
    B, T, _ = x.shape
    return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

def _merge_heads(self, x):                     # (B, H, T, head_dim) -> (B, T, d_model)
    B, _, T, _ = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, self.d_model)
```

The order matters. `view(B, T, H, head_dim)` splits the last dimension into heads while
**keeping** the correspondence with the positions; the `transpose(1, 2)` moves the heads to
the front so attention operates over them in parallel.

If you did `view(B, H, T, head_dim)` directly, you would be mixing positions with heads: the
result has the right shape and the wrong data. It is a bug that never raises an error. The
test `test_mha_does_not_mix_information_between_heads` detects it by checking the heads do
not give identical patterns.

**The `.contiguous()`.** `transpose` does not move data: it only changes the *strides*, that
is, how memory is walked. `view` demands contiguous memory. Without the `.contiguous()`,
PyTorch raises an error that talks about strides and does not clearly say what to do.
(`reshape` would do it for you, but it is worth seeing the distinction once.)

### The forward

```python
seq_len = x.shape[1]
if mask is None:
    mask = causal_mask(seq_len, device=x.device)

q = self._split_heads(self.q_proj(x))
k = self._split_heads(self.k_proj(x))
v = self._split_heads(self.v_proj(x))

if cos is not None and sin is not None:
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

if self.use_sdpa and not return_weights:
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                         dropout_p=self.dropout if self.training else 0.0)
    weights = None
else:
    scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    out = self.attn_dropout(weights) @ v

out = self.resid_dropout(self.out_proj(self._merge_heads(out)))
return (out, weights) if return_weights else out
```

**RoPE is applied after splitting into heads and only to Q and K.** The rotation depends on
`head_dim`, not on `d_model`, so it has to come after the split. And never to V: what should
depend on position is the *scores*, not the content being transported.

**Dropout goes in two different places.** `attn_dropout` over the attention weights (before
multiplying by V) and `resid_dropout` over `out_proj`'s output. With `dropout=0.0` — which
is the final model's config — both are the identity, but the structure has to be there for
the `tiny_char` config, which does use dropout.

**SDPA's `dropout_p` only during training.** `F.scaled_dot_product_attention` does not check
`self.training` on its own: if you pass it a fixed `dropout_p`, it will apply dropout at
evaluation time too and your samples will come out noisy and non-reproducible.

**Why one big projection and not 8 small ones.** `nn.Linear(320, 320)` followed by a `view`
is mathematically identical to eight `nn.Linear(320, 40)` whose results get concatenated. But
it is one big matmul instead of eight small ones, and as you saw in module 01, large matrices
use the GPU far better.

## What you should see in the demo

**The scaling experiment**, with the mean attention entropy (maximum 2.77 nats with 16
positions):

| d_k | with scaling | unscaled |
|---|---|---|
| 8 | 2.51 | 1.63 |
| 128 | 2.32 | 0.08 |
| 2048 | 2.28 | 0.007 |

With scaling the entropy stays high whatever happens. Unscaled, it collapses.

**The heatmaps.** Four heads from a model trained for 400 steps on Shakespeare. Three things
to look at:

1. The upper triangle is always black. That is the causal mask, and seeing it is the best
   check that it is correctly in place.
2. The diagonal is bright: almost every token pays a lot of attention to itself.
3. Each head has a different pattern. In the reference run, the mean distances each head
   looks back come out as 2.10 / 3.39 / 2.93 / 4.05 positions. **Nobody told them to
   specialize.**

With a one-layer model trained for 2 seconds you can already see it. In large models this
goes much further: there are heads that pair opening and closing quotes, and the *induction
heads*, which detect the pattern "…A B … A" and predict B — the mechanism believed to be
responsible for in-context learning.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
from typing import Any

def causal_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def single_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = q.shape[-1]

    # (B, T, d_k) @ (B, d_k, S) -> (B, T, S). scores[b,i,j] = how interested i is in j.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        # -inf before the softmax becomes probability 0 after it.
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_sdpa: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) is not divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        #: If `True`, use PyTorch's fused kernel. It is switched on in module 12, where
        #: the gain is measured. The output is the same; memory and speed change.
        self.use_sdpa = use_sdpa

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        # contiguous() is needed because transpose leaves the tensor with odd strides and
        # view() demands contiguous memory.
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        return_weights: bool = False,
        cache: Any = None,
        layer_idx: int = 0,
        pos_offset: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if cos is not None and sin is not None:
            # To Q and K only, never to V: what should depend on position is the attention
            # scores, not the content being transported.
            #
            # The slice by `pos_offset` is what makes the cache correct: when generating
            # token 50 only that token is passed in, but it has to be rotated by the angle
            # of position 50, not position 0.
            from llmfs.reference.position import apply_rope

            cos_t = cos[pos_offset : pos_offset + seq_len]
            sin_t = sin[pos_offset : pos_offset + seq_len]
            q, k = apply_rope(q, cos_t, sin_t), apply_rope(k, cos_t, sin_t)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        total_len = k.shape[-2]
        if mask is None:
            if cache is not None:
                # With a cache, the new token can look at EVERYTHING before it plus
                # itself. Nothing needs triangulating: the cache only holds the past.
                mask = torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device)
            else:
                mask = causal_mask(seq_len, device=x.device)

        if self.use_sdpa and not return_weights:
            # Fused kernel: it does not materialize the T x T matrix. On Turing it uses
            # the memory_efficient backend because FlashAttention-2 needs sm_80+.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
            )
            weights = None
        else:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            out = self.attn_dropout(weights) @ v

        out = self.resid_dropout(self.out_proj(self._merge_heads(out)))
        if return_weights:
            return out, weights  # type: ignore[return-value]
        return out
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
