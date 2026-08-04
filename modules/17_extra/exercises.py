"""Module 17 - Extras and honest limits.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement -> `llmfs check 17` -> `llmfs hint 17 -e N`
-> `SOLUTION.md` has the complete code.

These are the last three exercises of the course, and they are short.

WHAT YOU ARE GOING TO BUILD
===========================

    quantize_int8_symmetric  (ex. 1)  storing the weights in 1 byte instead of 4
    dequantize_int8          (ex. 2)  recovering them (approximately)
    quantization_error       (ex. 3)  measuring how much was lost

With that the model goes from 35.7 MB to 9.0 MB.

VOCABULARY YOU ARE GOING TO NEED
================================

- **quantize**: storing the weights with fewer bits. From float32 (4 bytes) to int8 (1 byte).
- **scale**: the number you have to multiply the integers by to recover the original values.
  It is stored alongside them.
- **symmetric / asymmetric**: whether the range is centred on zero or whether it also carries
  an offset.
- **per channel / per tensor**: one scale per row of the matrix, or a single one for all of it.
- **relative error**: the error divided by the magnitude of the original. It is the metric
  worth looking at, because it does not depend on the scale of the data.

    llmfs demo 17     quantizes your model, measures the damage, and closes the course
"""

from __future__ import annotations

import torch


def quantize_int8_symmetric(
    weight: torch.Tensor, per_channel: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts a weight matrix into int8 with a scale.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Three lines.

        1. The largest absolute value, per row or over the whole matrix:

               if per_channel:
                   max_abs = weight.abs().amax(dim=-1, keepdim=True)
               else:
                   max_abs = weight.abs().amax()

        2. The scale:

               scale = (max_abs / 127.0).clamp_min(1e-12)

        3. The integers:

               quantized = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
               return quantized, scale

    WITH NUMBERS, FOLLOWED BY HAND
    ------------------------------
        W = [0.12, -0.45, 0.03, 0.28]

    The largest in absolute value is 0.45. That range is mapped to [-127, +127]:

        scale = 0.45 / 127 = 0.003543
        W_int8 = round(W / scale) = [34, -127, 8, 79]

    And to recover it: `W' = W_int8 * scale = [0.1204, -0.4500, 0.0283, 0.2799]`.

    It is not exact: the error is on the order of HALF a unit of scale, which is what you lose
    by rounding. Note that the maximum value (-0.45) is recovered EXACTLY, and the small ones
    are the ones that accumulate the most relative error.

    THE IDEA
    --------
    A float32 takes 4 bytes and an int8 only one: the model takes a quarter of the space, from
    35.7 MB to 9.0 MB. The trick is to store, alongside the integers, a SCALE that lets you
    recover the approximate values.

    WHY 127 AND NOT 128
    -------------------
    int8 goes from -128 to 127. Using 127 the range is SYMMETRIC and zero is represented
    EXACTLY (the integer 0 comes back as the float 0).

    That matters more than it seems: in a matrix with many small values, zero being exact
    avoids a systematic bias that would accumulate layer after layer. A random rounding error
    cancels out; a constant bias does not.

    WHAT "SYMMETRIC" MEANS
    ----------------------
    That the range is centred on zero. The alternative (asymmetric) also uses an OFFSET —a
    "zero point"— and makes better use of the range when the data is skewed to one side, but it
    is more expensive to apply because you have to add and subtract that offset in every
    operation. A network's weights are usually fairly centred, so the symmetric one pays off.

    PER CHANNEL AGAINST PER TENSOR
    ------------------------------
    With `per_channel=True` one scale per ROW is computed instead of a single one for the whole
    matrix. It costs one extra vector of scales (negligible: one per row) and reduces the error
    quite a lot, because a single row with large values no longer drags all the others into
    using a coarse scale.

    Measured on a real matrix from our model: **0.71% error per channel against 1.07% per
    tensor**.

    THE TWO PROTECTIONS, WHICH ARE NOT DECORATIVE
    ---------------------------------------------
    **`clamp_min(1e-12)`**: it avoids dividing by zero if a whole row is zeros. It happens more
    than you would think with pruned weights or with layers that have learned nothing.

    **`clamp(-127, 127)`**: it protects against rounding at the edge. Without it, a value right
    at the maximum could round to 128, which does not fit in int8 and WRAPS to -128. That is,
    the largest weight of the row would become the most negative one. It is a spectacular and
    silent bug.

    Args:
        weight: the weight matrix, in float.
        per_channel: one scale per row (True) or one for the whole matrix (False).

    Returns:
        `(quantized, scale)`, with `quantized` of dtype `int8`.
    """
    raise NotImplementedError("TODO: module 17, exercise 1 - quantize_int8_symmetric")


def dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Goes back to float by multiplying by the scale.

    WHAT YOU HAVE TO WRITE
    ----------------------
    One line:

        return quantized.to(torch.float32) * scale

    THE `.to(torch.float32)` IS MANDATORY, AND IT GOES FIRST
    --------------------------------------------------------
    If you multiplied the int8 directly, PyTorch would do the operation in integers and the
    result would be garbage: the scale is 0.0035, rounded to an integer it is 0, and the whole
    matrix would come out as zero.

    Converting first and multiplying afterwards is the right order. It is an easy mistake to
    make and the symptom —a model that suddenly always generates the same token— does not point
    here.

    WHAT HAPPENS WITH THE BROADCAST
    -------------------------------
    With `per_channel=True`, `scale` has shape `(rows, 1)` thanks to the `keepdim=True` of the
    previous exercise, and it broadcasts on its own against `(rows, columns)`. Each row is
    multiplied by its scale. That is why the `keepdim` was not optional.

    With `per_channel=False`, `scale` is a scalar and multiplies everything equally.

    THE RESULT IS NOT EQUAL TO THE ORIGINAL
    ---------------------------------------
    Information was lost in the rounding and there is no way of recovering it. What comes out is
    the APPROXIMATE value, with an error that depends on how much had to be rounded. Exercise 3
    measures it.

    Args:
        quantized: the int8 tensor.
        scale: the scale `quantize_int8_symmetric` returned.

    Returns:
        The tensor in float32, approximately equal to the original.
    """
    raise NotImplementedError("TODO: module 17, exercise 2 - dequantize_int8")


def quantization_error(original: torch.Tensor, per_channel: bool = True) -> dict[str, float]:
    """Measures how much the quantization damages.

    WHAT YOU HAVE TO WRITE
    ----------------------
    There and back, and compare.

        1. The complete cycle:

               q, scale = quantize_int8_symmetric(original, per_channel=per_channel)
               recovered = dequantize_int8(q, scale)

        2. The error, element by element:

               error = (original - recovered).abs()
               original_norm = float(original.norm())

        3. The dict with the six metrics:

               return {
                   "relative_error": float(error.norm()) / max(original_norm, 1e-12),
                   "max_error": float(error.max()),
                   "mean_error": float(error.mean()),
                   "compression": original.element_size() / q.element_size(),
                   "original_bytes": original.numel() * original.element_size(),
                   "quantized_bytes": (
                       q.numel() * q.element_size() + scale.numel() * scale.element_size()
                   ),
               }

    WHAT SHOULD COME OUT
    --------------------
    On the `q_proj` matrix of the first block of our model, with `per_channel=True`:

        relative_error    0.00714     <- 0.71%
        max_error         0.000368
        mean_error        0.000123
        compression       4.0
        original_bytes    409,600
        quantized_bytes   103,680     <- 102,400 of int8 + 1,280 of the 320 scales

    And with `per_channel=False`, the relative error rises to 0.01068. That comparison is the
    result of the module: per channel is clearly better and practically free.

    WHICH OF THE SIX TO LOOK AT
    ---------------------------
    **`relative_error`**, without a doubt. It is the only one independent of the scale of the
    data, so you can compare different layers, different models and different configurations
    with the same number. With the weights of a trained network, int8 per channel is around
    0.5-1%.

    `max_error` and `mean_error` are in the units of the weights, so they only serve within one
    matrix.

    `error.norm()` is the L2 norm of the error vector, the same idea as in `clip_grad_norm` in
    module 11: treating every element as a single giant vector.

    THE HONEST DETAIL OF `quantized_bytes`
    --------------------------------------
    The bytes of the scales are ALSO added, not just those of the int8. With one scale per row
    they are negligible (320 floats against 102,400 int8), but counting them is the correct
    thing: if someone used one scale for every 8 elements, the real saving would be a fair bit
    less than the nominal 4x, and this metric would show it.

    Note that `compression` does NOT use those bytes: it is the nominal per-element ratio
    (`element_size()` of 4 against 1). Both things are useful and they measure different things:
    one is the theoretical limit and the other the real size on disk.

    `tensor.element_size()` gives the bytes per element (4 for float32, 1 for int8). Using it
    instead of writing a 4 and a 1 keeps the code correct if one day you quantize from float16.

    WHAT IT IS FOR
    --------------
    To know whether it is worth it BEFORE quantizing the whole model and discovering it
    generates garbage. You quantize one matrix, look at the relative error, and decide.

    Args:
        original: the weight matrix in float.
        per_channel: which quantization mode to measure.

    Returns:
        A dict with `relative_error`, `max_error`, `mean_error`, `compression`,
        `original_bytes` and `quantized_bytes`.
    """
    raise NotImplementedError("TODO: module 17, exercise 3 - quantization_error")
