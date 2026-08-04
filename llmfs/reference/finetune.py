"""Reference for module 16: post-training (SFT and LoRA)."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

#: The chat template markers. They delimit the turns so the model learns WHERE each
#: intervention starts and ends. Without them it cannot know when to stop.
CHAT_MARKERS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "end": "<|end|>",
}


def build_chat_template(
    messages: Sequence[dict[str, str]], add_generation_prompt: bool = False
) -> str:
    """Serialize a conversation to plain text with markers.

    A pretrained model only knows how to continue text. For it to answer instead of just
    carrying on writing, you have to teach it a FORMAT: markers that delimit the turns.

    Example:

        [{"role": "user", "content": "Hello"},
         {"role": "assistant", "content": "How are you"}]

        ->  <|user|>Hello<|end|><|assistant|>How are you<|end|>

    `add_generation_prompt=True` leaves the string open at `<|assistant|>` so the model can
    continue. That is what is used at inference time.

    There is nothing magic about the markers: they are text the model learns to recognize
    during SFT. Each model family uses its own, and they are incompatible with each other.
    """
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
    """Build the targets ignoring the prompt: only the ANSWER is learned from.

    THE IDEA. In SFT you do not want the model to learn to generate the user's questions:
    you want it to learn to answer them. By putting `-100` at the prompt's positions,
    `F.cross_entropy(..., ignore_index=-100)` skips them.

    Example with `prompt_len=3`:

        input_ids = [10, 11, 12, 20, 21, 22]
        targets   = [-100, -100, 20, 21, 22, -100]

    Note TWO things:

    1. The targets are SHIFTED by one token, as always: at position `i` the target is
       `input_ids[i+1]`.
    2. That is why there are `prompt_len - 1` ignored positions at the start, not
       `prompt_len`: at position `prompt_len - 1` (the last prompt token) the target is
       already the first token of the answer, and THAT one does matter.

    That off-by-one is the classic mistake in this exercise, and it produces no visible
    error: it merely wastes (or over-uses) one position.
    """
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
    """A linear layer with low-rank adapters.

    THE IDEA (Hu et al. 2021). Instead of training the whole d_in x d_out matrix `W`, it is
    frozen and the product of TWO thin matrices is added to it:

        output = x @ W^T  +  (alpha/r) * x @ A^T @ B^T

    with `A` of size r x d_in and `B` of size d_out x r, and `r` small (4, 8, 16).

    THE ARITHMETIC that justifies it. With d_in = d_out = 320 and r = 8:

        full W  :  320 x 320       = 102,400 parameters
        A and B :  8x320 + 320x8   =   5,120 parameters    (5%)

    You train 5% of the parameters, and since the gradient only has to be computed for those
    two matrices, the optimizer state is 5% too. On large models that is the difference
    between needing 8 GPUs and needing one.

    THE INITIALIZATION MATTERS AND IT IS NOT SYMMETRIC:
        A ~ normal (Kaiming)
        B = ZEROS

    With `B = 0`, the product `BA` is zero at the start and the layer is EXACTLY the
    original. Fine-tuning begins without perturbing anything. If you initialized both at
    random, the model would start out degraded and would have to recover before improving.

    The `alpha/r` scale exists so that changing `r` does not force you to retune the
    learning rate.
    """

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
    """Fold the adapters into the base matrix and return a plain `nn.Linear`.

        W_new = W + (alpha/r) * B @ A

    WHY IT MATTERS. During training, LoRA adds two matmuls per layer, and that shows up at
    inference time. By merging the weights, the resulting model is indistinguishable from a
    normal one: same cost, same shapes, and it can be served with no LoRA dependency at all.

    It is one of LoRA's advantages over other parameter-efficient fine-tuning methods: the
    adaptation is EXACTLY a sum of matrices, so it can be absorbed.

    The output has to match the LoRA layer's up to floating-point error.
    """
    d_in = layer.base.in_features
    d_out = layer.base.out_features
    merged = nn.Linear(d_in, d_out, bias=layer.base.bias is not None)

    with torch.no_grad():
        delta = (layer.lora_B @ layer.lora_A) * layer.scaling
        merged.weight.copy_(layer.base.weight + delta)
        if layer.base.bias is not None:
            merged.bias.copy_(layer.base.bias)

    return merged


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_names: Sequence[str] = ("q_proj", "v_proj"),
) -> nn.Module:
    """Replace the target layers with LoRA versions. Not an exercise.

    By default only `q_proj` and `v_proj`, which is what the original paper does: they give
    the best ratio between trained parameters and quality.

    WATCH THE ORDER, which is where the only non-obvious detail lies: the WHOLE model is
    frozen first, and the adapters are added afterwards. `LoRALinear` freezes its own base
    layer, but that does not touch the embeddings, the FFNs or the normalizations, which
    would stay trainable and defeat the purpose: with the 9M GPT you would go from training
    1% to training 86%.
    """
    for p in model.parameters():
        p.requires_grad = False

    for _name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in target_names and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha))
    return model


def count_trainable(model: nn.Module) -> dict[str, int]:
    """How many parameters are actually trained, against the total."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "percent": round(100 * trainable / max(1, total), 4),
    }
