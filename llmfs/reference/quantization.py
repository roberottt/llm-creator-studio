"""Reference for module 17: int8 quantization."""

from __future__ import annotations

import torch


def quantize_int8_symmetric(
    weight: torch.Tensor, per_channel: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a weight matrix to int8 with a scale.

    THE IDEA. A float32 takes 4 bytes and an int8 only one: the model takes a quarter of the
    space. The trick is storing, alongside the integers, a SCALE that lets you recover the
    approximate values.

    SYMMETRIC means the range is centered on zero: `[-max, +max]` maps to `[-127, +127]`.
    The alternative (asymmetric) also uses an offset and makes better use of the range when
    the data is not centered, but it is more expensive to apply. A network's weights are
    usually fairly centered, so the symmetric version does fine.

        scale  = max(|W|) / 127
        W_int8 = round(W / scale)  clamped to [-127, 127]

    WHY 127 AND NOT 128. int8 runs from -128 to 127. Using 127 keeps the range symmetric and
    represents zero exactly, which matters more than it seems: in a matrix with many small
    values, having zero be exact avoids a systematic bias.

    PER CHANNEL VERSUS PER TENSOR. With `per_channel=True` one scale is computed per ROW
    instead of one for the whole matrix. It costs one extra vector of scales (negligible)
    and reduces the error considerably, because a single row with large values does not drag
    the others down with it. It is what every serious implementation does.

    Returns:
        `(quantized, scales)` with `quantized` of dtype `int8` and `scales` of shape
        `(rows, 1)` if per-channel, or a scalar otherwise.
    """
    if per_channel and weight.dim() >= 2:
        max_abs = weight.abs().amax(dim=-1, keepdim=True)
    else:
        max_abs = weight.abs().amax()

    # clamp_min avoids dividing by zero if a row is all zeros.
    scale = (max_abs / 127.0).clamp_min(1e-12)
    quantized = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Back to float by multiplying by the scale.

    The result is NOT equal to the original: information has been lost. What you recover is
    the approximate value, with an error that depends on how much rounding was needed.
    """
    return quantized.to(torch.float32) * scale


def quantization_error(original: torch.Tensor, per_channel: bool = True) -> dict[str, float]:
    """Measure how much quantization hurts.

    Three numbers:

    - `relative_error`: the norm of the error divided by the norm of the original. This is
      the metric worth looking at: independent of the scale of the data.
    - `max_error`: the worst case.
    - `compression`: how many times smaller it is (4x going from fp32 to int8).

    With the weights of a trained network, the relative error of per-channel int8 is around
    0.5-1%. That this barely affects the model's quality is an empirical fact, not a theorem.
    """
    q, scale = quantize_int8_symmetric(original, per_channel=per_channel)
    recovered = dequantize_int8(q, scale)

    error = (original - recovered).abs()
    original_norm = float(original.norm())

    return {
        "relative_error": float(error.norm()) / max(original_norm, 1e-12),
        "max_error": float(error.max()),
        "mean_error": float(error.mean()),
        "compression": original.element_size() / q.element_size(),
        "original_bytes": original.numel() * original.element_size(),
        "quantized_bytes": q.numel() * q.element_size() + scale.numel() * scale.element_size(),
    }
