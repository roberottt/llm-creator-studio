"""Module 16 - Post-training: SFT and LoRA.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement -> `llmfs check 16` -> `llmfs hint 16 -e N`
-> `SOLUTION.md` has the complete code.

WHAT YOU ARE GOING TO BUILD
===========================

How to turn a model that continues text into one that answers:

    build_chat_template  (ex. 1)  the format that teaches it where each turn starts
    mask_prompt_tokens   (ex. 2)  so it learns to ANSWER, not to ask
    LoRALinear           (ex. 3)  training 0.7% of the parameters
    merge_lora_weights   (ex. 4)  merging the changes without a trace

The first two are about format and they are short. The last two are LoRA.

THE PROBLEM
===========

Write "What is the capital of France?" to your trained model and the most likely thing is
that it answers with MORE questions. It is not broken: it is doing exactly what you taught
it, which is continuing plausible text.

VOCABULARY YOU ARE GOING TO NEED
================================

- **pretraining**: the long phase, learning language by predicting the next token.
- **post-training / SFT**: carrying on training on instruction-and-answer examples.
- **chat template**: the markers (`<|user|>`, `<|end|>`) that delimit the turns.
- **ignore_index**: the value (-100) that makes `cross_entropy` skip a position without
  counting it in the loss.
- **LoRA**: training two small matrices added to the model instead of all its weights.
- **rank** (r) of LoRA: the internal dimension of those matrices. Typically 4, 8 or 16.
- **freezing** a parameter: setting `requires_grad = False` so it is not trained.

    llmfs demo 16     does real SFT and compares the before and the after
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

# The chat template markers. They delimit the turns so the model learns WHERE each
# intervention starts and ends.
from llmfs.reference import CHAT_MARKERS


def build_chat_template(
    messages: Sequence[dict[str, str]], add_generation_prompt: bool = False
) -> str:
    """Serializes a conversation into plain text with markers.

    WHAT YOU HAVE TO WRITE
    ----------------------
    A loop and a `join`.

        1. The list to accumulate into:

               parts = []

        2. One message at a time, validating the role:

               for message in messages:
                   role = message["role"]
                   if role not in CHAT_MARKERS:
                       raise ValueError(
                           f"unknown role: {role!r}. Valid ones: system, user, assistant"
                       )
                   parts.append(
                       f"{CHAT_MARKERS[role]}{message['content']}{CHAT_MARKERS['end']}"
                   )

        3. The opening for the answer, if asked for:

               if add_generation_prompt:
                   parts.append(CHAT_MARKERS["assistant"])

        4. `return "".join(parts)`

    Note that it joins with `""` and not with spaces or newlines: the markers already
    separate, and any extra character would be one more the model has to learn to predict.

    WHAT SHOULD COME OUT
    --------------------
        [{"role": "user", "content": "Hello"},
         {"role": "assistant", "content": "How are you"}]

        ->  <|user|>Hello<|end|><|assistant|>How are you<|end|>

    And with `add_generation_prompt=True` on only the first message:

        ->  <|user|>Hello<|end|><|assistant|>

    The string is left OPEN on purpose.

    WHAT PROBLEM IT SOLVES
    ----------------------
    A pretrained model only knows how to continue text. If you write "What is the capital of
    France?" the most likely thing is that it answers with MORE questions: a document starting
    like that usually carries on like that. It is not broken, it is doing exactly what you
    taught it.

    For it to ANSWER you have to teach it a FORMAT, and that is what the markers are.

    THE `add_generation_prompt`
    ---------------------------
    It is what you use at INFERENCE time. You leave the string open at `<|assistant|>`, the
    model continues right there, and what it writes is the answer. Without that opening the
    model does not know it is its turn to speak, and it is quite likely to generate another
    `<|user|>` and start making up your next question.

    In TRAINING it goes to False: there the assistant's answer is already in the data.

    WHY THE `<|end|>` MATTERS
    -------------------------
    It is what teaches the model WHEN TO STOP. Without an end marker, the model would generate
    indefinitely. At inference time it is used as a stop token (module 14's `eos_token`).

    THE MARKERS ARE NOT MAGICAL
    ---------------------------
    They are ordinary text the model learns to recognize during SFT. Every model family uses
    its own and they are incompatible with each other: using the wrong template with a model
    degrades its quality quite a lot, and it is a surprisingly frequent mistake because it
    raises no warning, only worse answers.

    Args:
        messages: list of `{"role": ..., "content": ...}`. Valid roles: those of
            `CHAT_MARKERS` minus "end".
        add_generation_prompt: leave the string open at `<|assistant|>`.

    Returns:
        The serialized conversation.

    Raises:
        ValueError: if some role is not in `CHAT_MARKERS`. Better a clear error than
            generating text with a made-up marker the model has never seen.
    """
    raise NotImplementedError("TODO: module 16, exercise 1 - build_chat_template")


def mask_prompt_tokens(
    input_ids: Sequence[int], prompt_len: int, ignore_index: int = -100
) -> list[int]:
    """Builds the targets ignoring the prompt: you only learn from the ANSWER.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Four lines, and the range of the loop is the WHOLE exercise.

        1. The validations:

               if prompt_len < 1:
                   raise ValueError("prompt_len has to be at least 1")
               if prompt_len > len(input_ids):
                   raise ValueError(
                       f"prompt_len ({prompt_len}) larger than the sequence ({len(input_ids)})"
                   )

        2. Everything ignored to start with:

               targets = [ignore_index] * len(input_ids)

        3. Fill in only the stretch that does count:

               for i in range(prompt_len - 1, len(input_ids) - 1):
                   targets[i] = input_ids[i + 1]

        4. `return targets`

    THE EXAMPLE, AND READ IT CAREFULLY
    ----------------------------------
        input_ids = [10, 11, 12, 20, 21, 22]     with prompt_len = 3
        targets   = [-100, -100, 20, 21, 22, -100]

    There are TWO ignored positions at the start, not three. And one at the end.

    WHY TWO AND NOT THREE
    ---------------------
    The targets are SHIFTED one token, as always in this course: at position `i` the target is
    `input_ids[i+1]`.

    So at position 2 —the LAST token of the prompt— the target is already `input_ids[3] = 20`,
    which is the FIRST token of the answer. And that one matters enormously: it is precisely
    the transition from "the question is over" to "my answer begins", the most important thing
    the model has to learn in the whole of SFT.

    Hence the `prompt_len - 1` in the `range`. That off-by-one is THE mistake of the exercise,
    and it gives no signal: it just wastes the most valuable position there is.

    WHY THE LAST ONE IS ALSO IGNORED
    --------------------------------
    At the last position there is no `input_ids[i+1]`: there is nothing to predict. Hence the
    `len(input_ids) - 1` as the end of the `range`.

    WHAT THE -100 DOES
    ------------------
    `F.cross_entropy(..., ignore_index=-100)` SKIPS those positions: they do not contribute to
    the loss and they generate no gradient. The -100 is a PyTorch convention, not a magic
    number; it could be any impossible token value.

    And that is why your GPT's `forward` (module 10) has had `ignore_index=-100` set since
    then. This is the module where it finally serves a purpose.

    THE IDEA
    --------
    In SFT you do not want the model to learn to GENERATE the user's questions: you want it to
    learn to ANSWER them. If you counted the loss over the prompt, you would be spending
    capacity on learning to imitate the user, which is exactly the opposite of what you want.

    Args:
        input_ids: the complete sequence, prompt and answer together.
        prompt_len: how many tokens the prompt takes.
        ignore_index: the value `cross_entropy` ignores. -100 is the PyTorch convention.

    Returns:
        The list of targets, of the SAME length as `input_ids`.

    Raises:
        ValueError: if `prompt_len` is less than 1 or larger than the sequence.
    """
    raise NotImplementedError("TODO: module 16, exercise 2 - mask_prompt_tokens")


class LoRALinear(nn.Module):
    """A linear layer with low-rank adapters.

    WHAT YOU HAVE TO WRITE
    ----------------------
    **In `__init__`**, six steps:

        1. Validate the rank:

               if r <= 0:
                   raise ValueError(f"r has to be positive: {r}")

        2. Keep the base layer and FREEZE it. This is the whole point of LoRA:

               self.base = base_layer
               for p in self.base.parameters():
                   p.requires_grad = False

        3. The dimensions come from the layer itself:

               d_in = base_layer.in_features
               d_out = base_layer.out_features

        4. The two adapters:

               self.lora_A = nn.Parameter(torch.empty(r, d_in))
               self.lora_B = nn.Parameter(torch.zeros(d_out, r))

        5. The initialization, which is NOT symmetric (see below):

               nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
               # lora_B stays at ZEROS

        6. The scale and the dropout:

               self.r = r
               self.alpha = alpha
               self.scaling = alpha / r
               self.lora_dropout = nn.Dropout(dropout)

    **In `forward`**, one line:

        return (
            self.base(x)
            + self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        )

    FOLLOW THE SHAPES OF THE `forward` LINE
    ---------------------------------------
        x            (..., d_in)
        @ lora_A.T   with lora_A of (r, d_in), its T is (d_in, r)   ->  (..., r)
        @ lora_B.T   with lora_B of (d_out, r), its T is (r, d_out) ->  (..., d_out)

    It lines up with the output of `self.base(x)`, which is `(..., d_out)`. If you get tangled
    up with the transposes, follow the shapes like this: it is faster than thinking it through.

    THE ASYMMETRIC INITIALIZATION IS THE IMPORTANT PART OF THE EXERCISE
    ------------------------------------------------------------------
    `lora_B` starts at ZEROS. Therefore `B @ A = 0` when the layer is built, and the output is
    EXACTLY that of the original layer. Fine-tuning starts without perturbing anything at all.

    If you initialized both at random, the model would start degraded and would have to spend
    the first steps recovering what it already knew before starting to improve.

    And why not BOTH at zero? Because then the gradient of both would be zero forever (A's
    gradient goes through B and vice versa) and they would never learn anything. One at zero
    breaks the symmetry, both at zero freeze it.

    There is a test that builds the layer and checks its output is identical to the base one's.

    THE IDEA (Hu et al. 2021)
    -------------------------
    Full fine-tuning of a large model needs memory for the weights, the gradients AND Adam's
    two states: about 12 bytes per parameter. With 7B parameters that is 84 GB, and it does not
    fit in any consumer GPU.

    LoRA starts from an observation: the changes fine-tuning makes are LOW RANK. You do not
    need to be able to modify the matrix in every possible direction; a few directions are
    enough. So W is FROZEN and the product of two skinny matrices is added to it:

        output = x @ W^T  +  (alpha/r) * x @ A^T @ B^T

    with A of `(r, d_in)`, B of `(d_out, r)` and r small (4, 8, 16).

    THE ARITHMETIC, WITH OUR NUMBERS
    --------------------------------
    With d_in = d_out = 320 and r = 8:

        the whole W:  320 x 320        = 102,400 parameters
        A and B:      8x320 + 320x8    =   5,120 parameters     (5%)

    Applied to the 9M GPT only in `q_proj` and `v_proj`, with r=8: 61,440 trainable parameters,
    that is 0.68% of the model.

    THE `alpha/r` SCALE
    -------------------
    It exists so that changing `r` does not force you to retune the learning rate. With a
    higher r, `B @ A` has more terms and its magnitude grows; dividing by r compensates. Keep
    it in `self.scaling` because exercise 4 needs it.

    FREEZING THE BASE IS NOT OPTIONAL
    ---------------------------------
    Without the `requires_grad = False` of step 2 you would be training the whole model AND the
    adapters on top: the worst of both worlds. And it raises no error, it just consumes the
    memory you wanted to save. If on applying LoRA to your model you see that 86% of the
    parameters are trainable instead of 0.7%, this is what happened.

    SUBMODULES (respect the names, exercise 4 uses them)
        base:         the original layer, frozen
        lora_A:       `nn.Parameter` of shape `(r, d_in)`
        lora_B:       `nn.Parameter` of shape `(d_out, r)`, at zeros
        lora_dropout: `nn.Dropout(dropout)`
        scaling:      the float `alpha / r`

    __init__(self, base_layer, r=8, alpha=16.0, dropout=0.0)
        Raises:
            ValueError: if `r` is not positive.

    forward(self, x):
        Args:
            x: `(..., d_in)`.
        Returns:
            `(..., d_out)`, the same shape the base layer would give.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        raise NotImplementedError("TODO: module 16, exercise 3 - LoRALinear.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: module 16, exercise 3 - LoRALinear.forward")


def merge_lora_weights(layer: LoRALinear) -> nn.Linear:
    """Merges the adapters into the base matrix and returns a normal `nn.Linear`.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Four lines.

        1. The empty layer, with the same dimensions and the same bias (or its absence):

               d_in = layer.base.in_features
               d_out = layer.base.out_features
               merged = nn.Linear(d_in, d_out, bias=layer.base.bias is not None)

        2. The merge, inside `no_grad`:

               with torch.no_grad():
                   delta = layer.lora_B @ layer.lora_A * layer.scaling
                   merged.weight.copy_(layer.base.weight + delta)
                   if layer.base.bias is not None:
                       merged.bias.copy_(layer.base.bias)

        3. `return merged`

    THE `B @ A` ORDER IS NOT INTERCHANGEABLE
    ----------------------------------------
        B is (d_out, r)  and  A is (r, d_in)
        B @ A            ->   (d_out, d_in)

    Which is exactly the shape of `weight` in an `nn.Linear`. The other way round (`A @ B`) the
    shapes do not even line up unless d_in == d_out, and in that case they would line up giving
    the wrong result, which is worse.

    WHY `copy_` AND NOT AN ASSIGNMENT
    ---------------------------------
    `merged.weight.copy_(...)` writes INSIDE the tensor `nn.Linear` already created, preserving
    its identity and its metadata (that it is registered as a parameter, its `requires_grad`,
    its place in the `state_dict`).

    `merged.weight = ...` with a normal tensor would fail: `nn.Module` only accepts
    `nn.Parameter` in that attribute. It is the same kind of distinction you already saw with
    the weight tying of module 10, where there you did want to reassign.

    THE `no_grad`
    -------------
    You are writing over parameters. Without it, each `copy_` would build an autograd graph
    that serves no purpose and would additionally keep the original tensors alive in memory.

    WHY THIS MATTERS
    ----------------
    During training, LoRA adds two matrix multiplications per layer, and that shows at
    inference time: more kernels to launch, more latency per token.

    By merging the weights, the resulting model is INDISTINGUISHABLE from a normal one: same
    cost, same shapes, and it can be served with no dependency on the LoRA code.

    It is an advantage of LoRA over other efficient fine-tuning methods: the adaptation is
    EXACTLY a sum of matrices, so it is absorbed without approximating anything. You do not
    lose a single decimal.

    And it has a nice practical consequence: you can keep several 60 KB adapters for different
    tasks over a single base model, and merge whichever one you need at any moment.

    THE CHECK
    ---------
    The merged layer has to give the SAME output as the LoRA layer, up to floating point error.
    There is a test that verifies it with `torch.allclose`.

    (With `dropout > 0` compare in `eval()` mode: in `train()` the dropout randomizes the LoRA
    layer's output and there would be nothing to compare.)

    Args:
        layer: the already trained LoRA layer.

    Returns:
        A normal `nn.Linear` with the weights already merged.
    """
    raise NotImplementedError("TODO: module 16, exercise 4 - merge_lora_weights")
