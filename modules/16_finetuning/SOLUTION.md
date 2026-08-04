# 16 — Annotated solution

## Exercise 1 — `build_chat_template`

```python
parts = []
for message in messages:
    role = message["role"]
    if role not in CHAT_MARKERS:
        raise ValueError(f"unknown role: {role!r}")
    parts.append(f"{CHAT_MARKERS[role]}{message['content']}{CHAT_MARKERS['end']}")

if add_generation_prompt:
    parts.append(CHAT_MARKERS["assistant"])

return "".join(parts)
```

**The `add_generation_prompt` leaves the string open**, without `<|end|>`. It is what you use
at inference time: the model continues right there and what it writes is the answer. Without
that opening, the model does not know it is its turn to speak.

**The `<|end|>` is what teaches it when to stop.** Without an end marker, a model would
generate indefinitely. At inference time it is used as a stop token.

**The markers are not magical:** they are text the model learns to recognize during SFT. Every
model family uses its own and they are incompatible with each other. Using the wrong template
degrades quality quite a lot, and it is a surprisingly frequent mistake.

## Exercise 2 — `mask_prompt_tokens`

```python
targets = [ignore_index] * len(input_ids)
for i in range(prompt_len - 1, len(input_ids) - 1):
    targets[i] = input_ids[i + 1]
return targets
```

### The off-by-one, which is the whole exercise

```
input_ids = [10, 11, 12, 20, 21, 22]      prompt_len = 3
targets   = [-100, -100, 20, 21, 22, -100]
```

**Two ignored positions at the start, not three.** The loop starts at `prompt_len - 1`.

The reason: the targets are shifted one token. At position 2 —the **last** token of the
prompt— the target is already `input_ids[3] = 20`, which is the first token of the answer. And
that one does matter: it is precisely the *"the question is over, my turn to answer"*
transition, which is the most important thing the model has to learn from SFT.

If you started at `prompt_len`, you would skip exactly that transition. And it gives no signal:
you just waste the most informative position of the example.

**The last position is also ignored** because there is no `input_ids[i+1]` to predict.

## Exercise 3 — `LoRALinear`

```python
def __init__(self, base_layer, r=8, alpha=16.0, dropout=0.0):
    super().__init__()
    if r <= 0:
        raise ValueError(f"the rank r has to be positive: {r}")

    self.base = base_layer
    self.r, self.alpha = r, alpha
    self.scaling = alpha / r

    for p in self.base.parameters():
        p.requires_grad = False

    d_in, d_out = base_layer.in_features, base_layer.out_features
    self.lora_A = nn.Parameter(torch.empty(r, d_in))
    self.lora_B = nn.Parameter(torch.zeros(d_out, r))
    self.lora_dropout = nn.Dropout(dropout)

    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

def forward(self, x):
    base = self.base(x)
    adaptation = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
    return base + adaptation * self.scaling
```

### The asymmetric initialization is the important part

```
A ~ Kaiming
B = zeros
```

With `B = 0`, the product `BA` is zero at the start and **the layer is exactly the original
one**. Fine-tuning starts without perturbing anything. The test
`test_on_initialization_the_output_is_identical_to_the_base` checks it directly.

If you initialized both at random, the model would start degraded and would have to recover
before starting to improve.

**And why not both at zero?** Because then the gradient of both would be zero —each one
multiplies the other— and they would never learn anything. There is a test for that too.

### Freezing the base is not optional

```python
for p in self.base.parameters():
    p.requires_grad = False
```

It is the whole point of LoRA. Without that you would be training everything *and* the
adapters on top.

### The transposes

`lora_A` is `(r, d_in)`, so `x @ lora_A.T` gives `(..., r)`. Then `@ lora_B.T` with `lora_B` of
`(d_out, r)` gives `(..., d_out)`. Correct.

## Exercise 4 — `merge_lora_weights`

```python
merged = nn.Linear(d_in, d_out, bias=layer.base.bias is not None)
with torch.no_grad():
    delta = (layer.lora_B @ layer.lora_A) * layer.scaling
    merged.weight.copy_(layer.base.weight + delta)
    if layer.base.bias is not None:
        merged.bias.copy_(layer.base.bias)
return merged
```

**The `B @ A` order** and not the other way round: `B` is `(d_out, r)` and `A` is `(r, d_in)`,
so the product gives `(d_out, d_in)`, which is exactly the shape of `weight` in an `nn.Linear`.

**Why merging matters.** During training LoRA adds two matmuls per layer, and that shows at
inference time. Merged, the model is indistinguishable from a normal one: same cost, same
shapes, and it can be served with no LoRA dependency.

It is an advantage of LoRA over other efficient fine-tuning methods: the adaptation is
**exactly** a sum of matrices, so it is absorbed without approximating anything. The demo
verifies it: an error of $1.3 \times 10^{-6}$ between the LoRA layer and the merged one.

## A bug I found writing the reference

`apply_lora_to_model` gave **86% trainable parameters** when LoRA should give ~1%.

The cause: `LoRALinear` freezes its own base layer, but that does not touch the embeddings, the
FFNs or the normalizations, which stayed trainable. Freeze the **whole model first** and add
the adapters afterwards:

```python
for p in model.parameters():
    p.requires_grad = False
# ... and now yes, replace the target layers
```

With that the numbers come out as they should:

| r | trainable | % of the model |
|---|---|---|
| 4 | 30,720 | **0.34%** |
| 8 | 61,440 | **0.68%** |
| 16 | 122,880 | **1.36%** |

It is an easy mistake to make and a silent one: training works, you are simply not doing LoRA.

## What you should see in the demo

The SFT on the Shakespeare model, with 96 examples and 150 steps:

```
BEFORE:   Q: Who is the king?
          A:
          I have the courtesy? I do hear thee a dealth,
          Company, but

AFTER:    Q: Who is the king?
          A:
          I say we must go.

          MARIANA:
          The castle.
```

**What you have to look at is not whether the answer is correct.** With 0.8M parameters and 96
examples, it is not going to be: *"I say we must go"* is literally one of the answers from the
training set, memorized.

What you have to look at is the **format**: before, it rambled on in Shakespeare indefinitely;
afterwards, it produces a short answer. The post-training has done its job.

And that is the module's lesson: **post-training does not add knowledge**. It brings to the
surface a behaviour that was already latent. A model that does not know something after
pretraining does not learn it from a thousand conversation examples.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
def build_chat_template(
    messages: Sequence[dict[str, str]], add_generation_prompt: bool = False
) -> str:
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        if role not in CHAT_MARKERS:
            raise ValueError(f"unknown role: {role!r}. Valid ones: system, user, assistant")
        parts.append(f"{CHAT_MARKERS[role]}{message['content']}{CHAT_MARKERS['end']}")

    if add_generation_prompt:
        parts.append(CHAT_MARKERS["assistant"])

    return "".join(parts)


def mask_prompt_tokens(
    input_ids: Sequence[int], prompt_len: int, ignore_index: int = -100
) -> list[int]:
    if prompt_len < 1:
        raise ValueError("prompt_len has to be at least 1")
    if prompt_len > len(input_ids):
        raise ValueError(
            f"prompt_len ({prompt_len}) is larger than the sequence ({len(input_ids)})"
        )

    targets = [ignore_index] * len(input_ids)
    for i in range(prompt_len - 1, len(input_ids) - 1):
        targets[i] = input_ids[i + 1]
    return targets


class LoRALinear(nn.Module):

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"the rank r has to be positive: {r}")

        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # The base layer is frozen: that is the whole point of LoRA.
        for p in self.base.parameters():
            p.requires_grad = False

        d_in, d_out = base_layer.in_features, base_layer.out_features
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        self.lora_dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays at zeros: at the start, the layer is identical to the original.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        adaptation = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + adaptation * self.scaling


def merge_lora_weights(layer: LoRALinear) -> nn.Linear:
    d_in = layer.base.in_features
    d_out = layer.base.out_features
    merged = nn.Linear(d_in, d_out, bias=layer.base.bias is not None)

    with torch.no_grad():
        delta = (layer.lora_B @ layer.lora_A) * layer.scaling
        merged.weight.copy_(layer.base.weight + delta)
        if layer.base.bias is not None:
            merged.bias.copy_(layer.base.bias)

    return merged
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
