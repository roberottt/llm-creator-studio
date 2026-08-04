# 10 — Annotated solution

## Exercise 1 — `expected_param_count`

```python
d, v, ff = cfg.d_model, cfg.vocab_size, cfg.d_ff

total = v * d                                    # token embeddings
if cfg.pos == "learned":
    total += cfg.context_length * d

attention = 4 * d * d + (4 * d if cfg.bias else 0)
ffn_matrices = 3 if cfg.activation == "swiglu" else 2
ffn = ffn_matrices * d * ff
per_norm = d if cfg.norm == "rmsnorm" else (2 * d if cfg.bias else d)

total += cfg.n_layers * (attention + ffn + 2 * per_norm)
total += per_norm                                # the final norm

if not cfg.tie_embeddings:
    total += v * d

return total
```

Arithmetic. What matters is having derived it by hand before writing it.

**RoPE contributes not one parameter.** Its tables come from a formula and are stored as
*buffers*. If your count includes anything from RoPE, you have one term too many.

**RMSNorm has $d$ parameters, not $2d$.** Scale only, no bias. With 6 layers × 2 norms + 1
final that is 13 × 320 = 4,160 parameters: a pittance, but if you forget them the total does
not add up and the test says so.

## Exercise 2 — `count_parameters`

```python
breakdown = {"embeddings": 0, "attention": 0, "ffn": 0, "norms": 0, "lm_head": 0, "other": 0}
seen = set()

for name, param in model.named_parameters():
    if id(param) in seen:
        continue
    seen.add(id(param))
    n = param.numel()

    if "token_embedding" in name or "pos_embedding" in name:
        breakdown["embeddings"] += n
    elif "attn." in name:
        breakdown["attention"] += n
    elif any(k in name for k in ("gate_proj", "up_proj", "down_proj", "fc_in", "fc_out")):
        breakdown["ffn"] += n
    elif "norm" in name:
        breakdown["norms"] += n
    elif "lm_head" in name:
        breakdown["lm_head"] += n
    else:
        breakdown["other"] += n

breakdown["total"] = sum(breakdown.values())
breakdown["non_embedding"] = breakdown["total"] - breakdown["embeddings"]
return breakdown
```

**A correction to what is usually said about weight tying.** When writing this module I
assumed `named_parameters()` returned the tied tensor twice, and the test proved otherwise:
**both `parameters()` and `named_parameters()` deduplicate by identity by default**
(`remove_duplicate=True`, since PyTorch 1.13). The total comes out right without doing
anything.

The `set` of `id()` is still worth it for two reasons: it makes explicit that you know there
are shared weights, and it protects the breakdown if one day you walk the parameters with
`remove_duplicate=False`. With `named_parameters(remove_duplicate=False)` over the final
model you would count 1,310,720 parameters too many.

**The order of the checks matters.** The names look like `blocks.3.attn.out_proj.weight`, and
`"norm"` also appears in `attn_norm` and `ffn_norm`. If you checked `"norm"` before
`"attn."`, the attention's normalization would end up in the wrong category. Check from more
specific to more general.

## Exercise 3 — `TransformerBlock`

```python
def __init__(self, cfg):
    super().__init__()
    self.attn_norm = make_norm(cfg)
    self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads,
                                   dropout=cfg.dropout, bias=cfg.bias)
    self.ffn_norm = make_norm(cfg)
    self.ffn = make_ffn(cfg)

def forward(self, x, cos=None, sin=None, mask=None):
    x = x + self.attn(self.attn_norm(x), mask=mask, cos=cos, sin=sin)
    x = x + self.ffn(self.ffn_norm(x))
    return x
```

Two lines of forward. **Two independent residuals**, not one around the whole block: each
sub-block decides on its own how much it contributes to the residual stream.

The test `test_the_block_uses_residuals` checks it in the most direct way: it zeroes the
output weights of both branches and verifies that the output is **exactly** the input. If
your block had no residuals, it would return zero.

The FFN does not receive `cos`, `sin` or `mask`: it does not look at other tokens, so it does
not need them.

## Exercise 4 — `GPT`

### The tying

```python
if cfg.tie_embeddings:
    self.lm_head.weight = self.token_embedding.weight
```

Assigning the attribute makes both layers point at **the same object**. The test checks `is`,
not `==`: they have to be the same tensor, not two copies with the same values. If you did
`self.lm_head.weight.data = self.token_embedding.weight.data.clone()` you would have two
tensors with the same numbers that would diverge as soon as training started.

It goes after creating `lm_head` and before the initialization.

### RoPE's buffers

```python
cos, sin = rope_frequencies(cfg.head_dim, cfg.context_length, cfg.rope_theta)
self.register_buffer("rope_cos", cos, persistent=False)
self.register_buffer("rope_sin", sin, persistent=False)
```

A *buffer* is a tensor that travels with the model — it moves with `.to(device)`, it appears
in `.eval()` — but is not a parameter and receives no gradient.

`persistent=False` also means it is **not stored in the checkpoint**. Since they are
recomputed from a formula when the model is built, storing them would waste space and create
a problem if one day you changed `rope_theta`.

### The initialization, in two passes

```python
self.apply(self._init_weights)                    # everything with std=0.02

scale = 0.02 / math.sqrt(2 * cfg.n_layers)        # and then the right ones get overridden
for name, param in self.named_parameters():
    if name.endswith(("out_proj.weight", "down_proj.weight", "fc_out.weight")):
        nn.init.normal_(param, mean=0.0, std=scale)
```

**The order matters**: everything first, then the override. If you did it the other way
round, the `apply` would overwrite the scaled initialization.

**Why.** Each block *adds* its contribution to the residual stream. With independent
contributions of variance $\sigma^2$, the variance of the sum grows linearly with the number
of terms. With 6 layers × 2 sub-blocks that is 12 contributions: the output would have 12
times the variance of the input. Reducing $\sigma$ by $\sqrt{2 n_{\text{layers}}}$
compensates for it exactly.

**And the 0.02 is not magic either**: it is what makes the step-0 loss equal $\ln(V)$. With
`std=1` (PyTorch's default) the model would start opinionated and random, and the loss would
come out above — exactly what you see in module 05's demo.

### The forward

The only subtlety: **the mask is computed once**, before the block loop, and passed to all of
them. Computing it inside each block would work, but that would be 6 identical tensors per
forward.

## A bug I made writing these tests, and that can happen to you

The test checking the initial loss was failing, giving **4.94 when it should give 5.55**.
Lower than $\ln(V)$, which is the classic symptom of an information leak.

There was no leak in the model. There was one in the test: I was passing `model(idx, idx)`,
that is, **unshifted targets**. At position $t$ the model sees token `idx[t]` and is asked to
predict `idx[t]`: it can read it straight off its own input. With weight tying it is even
more direct, because the logits are $x W_{\text{emb}}^\top$ and the dot product of an
embedding with itself is large.

The correct thing is `x = seq[:, :-1]`, `y = seq[:, 1:]`.

It is worth keeping in mind because the symptom — a suspiciously low loss — is identical
whether the bug is in the model or in how you assemble the batch, and the second is more
common.

## What you should see in the demo

**The breakdown**, which closes Part II:

```
embeddings   1,310,720   14.7%
attention    2,457,600   27.5%
ffn          5,160,960   57.8%
norms            4,160    0.0%
lm_head              0    0.0%   (tied)
TOTAL        8,933,440
```

Formula, count and target all match.

**The causality check** is the prettiest thing in the module. The token at position 6 is
changed and you look at how much the logits move:

```
position 0-5:  0.00e+00     exactly zero
position 6:    1.46e+00
position 7-11: ~2.5e-01
```

**Exactly** zero, not small. The earlier predictions cannot see that token at all. It is the
most direct verification there is that the causal mask is correct.

**And the memory**, which sets up module 13:

```
weights + gradients + AdamW :  143 MB
logits (fp16 + fp32 + grad) : 1007 MB
```

The logits tensor (`48 × 512 × 4096`) takes **seven times more** than the model, its
gradients and the optimizer combined. When you run out of memory on the RTX 2060, that is the
first place to look, not the model's activations.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
from llmfs.reference import make_ffn, make_norm, sinusoidal_embeddings

def expected_param_count(cfg: ModelConfig) -> int:
    d, v, ff = cfg.d_model, cfg.vocab_size, cfg.d_ff

    total = v * d  # token embeddings
    if cfg.pos == "learned":
        total += cfg.context_length * d

    attention = 4 * d * d + (4 * d if cfg.bias else 0)
    ffn_matrices = 3 if cfg.activation == "swiglu" else 2
    ffn = ffn_matrices * d * ff
    if cfg.bias:
        ffn += 2 * ff + d if ffn_matrices == 3 else ff + d

    # RMSNorm has scale only; LayerNorm has scale and (optionally) bias.
    per_norm = d if cfg.norm == "rmsnorm" else (2 * d if cfg.bias else d)

    total += cfg.n_layers * (attention + ffn + 2 * per_norm)
    total += per_norm  # the final norm

    if not cfg.tie_embeddings:
        total += v * d

    return total


def count_parameters(model: nn.Module) -> dict[str, int]:
    breakdown = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "norms": 0,
        "lm_head": 0,
        "other": 0,
    }
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if id(param) in seen:
            continue  # tying: already counted
        seen.add(id(param))
        n = param.numel()

        if "token_embedding" in name or "pos_embedding" in name:
            breakdown["embeddings"] += n
        elif "attn." in name or "attention" in name:
            breakdown["attention"] += n
        elif any(k in name for k in ("gate_proj", "up_proj", "down_proj", "fc_in", "fc_out")):
            breakdown["ffn"] += n
        elif "norm" in name:
            breakdown["norms"] += n
        elif "lm_head" in name:
            breakdown["lm_head"] += n
        else:
            breakdown["other"] += n

    breakdown["total"] = sum(breakdown.values())
    breakdown["non_embedding"] = breakdown["total"] - breakdown["embeddings"]
    return breakdown


class TransformerBlock(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = make_norm(cfg)
        self.attn = MultiHeadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, bias=cfg.bias
        )
        self.ffn_norm = make_norm(cfg)
        self.ffn = make_ffn(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: object = None,
        layer_idx: int = 0,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            mask=mask,
            cos=cos,
            sin=sin,
            cache=cache,
            layer_idx=layer_idx,
            pos_offset=pos_offset,
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPT(nn.Module):

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embedding: nn.Embedding | None = None
        if cfg.pos == "learned":
            self.pos_embedding = nn.Embedding(cfg.context_length, cfg.d_model)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        if cfg.pos == "rope":
            cos, sin = rope_frequencies(cfg.head_dim, cfg.context_length, cfg.rope_theta)
            # persistent=False: they are recomputed on construction, so there is no need
            # to store them in the checkpoint or have them take up space in the file.
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        elif cfg.pos == "sinusoidal":
            self.register_buffer(
                "pos_table", sinusoidal_embeddings(cfg.context_length, cfg.d_model),
                persistent=False,
            )

        self.apply(self._init_weights)
        # The scaled init is applied AFTER the general apply, to override it.
        scale = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, param in self.named_parameters():
            if name.endswith(("out_proj.weight", "down_proj.weight", "fc_out.weight")):
                nn.init.normal_(param, mean=0.0, std=scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_cache: bool = False,
        cache: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len = idx.shape
        pos_offset = cache.seq_len if (use_cache and cache is not None) else 0

        if seq_len + pos_offset > self.cfg.context_length:
            raise ValueError(
                f"sequence of {seq_len + pos_offset} tokens, but the model's context "
                f"is {self.cfg.context_length}"
            )

        x = self.token_embedding(idx)
        positions = torch.arange(pos_offset, pos_offset + seq_len, device=idx.device)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(positions)
        elif self.cfg.pos == "sinusoidal":
            x = x + self.pos_table[positions]
        x = self.drop(x)

        cos = sin = None
        if self.cfg.pos == "rope":
            cos, sin = self.rope_cos, self.rope_sin

        mask = None if (use_cache and cache is not None) else causal_mask(seq_len, device=idx.device)
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                cos=cos,
                sin=sin,
                mask=mask,
                cache=cache if use_cache else None,
                layer_idx=i,
                pos_offset=pos_offset,
            )

        x = self.norm_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            window = idx[:, -self.cfg.context_length :]
            logits, _ = self(window)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
