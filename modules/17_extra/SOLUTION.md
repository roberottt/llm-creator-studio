# 17 — Annotated solution

## Exercise 1 — `quantize_int8_symmetric`

```python
if per_channel and weight.dim() >= 2:
    max_abs = weight.abs().amax(dim=-1, keepdim=True)
else:
    max_abs = weight.abs().amax()

scale = (max_abs / 127.0).clamp_min(1e-12)
quantized = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
return quantized, scale
```

**The `clamp_min(1e-12)`** avoids dividing by zero if a row is all zeros. It happens more than
you would expect with sparse matrices.

**The `clamp(-127, 127)`** protects against rounding at the edge. Without it, a value right at
the maximum could round to 128, which does not fit in `int8` and would *wrap around* to −128:
the largest weight would become the most negative one. Silent and devastating.

**Why 127 and not 128.** `int8` goes from −128 to 127. Using 127 the range is symmetric and
zero is represented exactly. In a matrix with many small values, zero being exact avoids a
systematic bias that would accumulate layer after layer.

## Exercise 2 — `dequantize_int8`

```python
return quantized.to(torch.float32) * scale
```

**The `.to(torch.float32)` goes before the multiplication.** If you multiplied the `int8`
directly, PyTorch would do the operation in integers and the result would be garbage.

## Exercise 3 — `quantization_error`

```python
q, scale = quantize_int8_symmetric(original, per_channel=per_channel)
recovered = dequantize_int8(q, scale)
error = (original - recovered).abs()

return {
    "relative_error": float(error.norm()) / max(float(original.norm()), 1e-12),
    "max_error": float(error.max()),
    "mean_error": float(error.mean()),
    "compression": original.element_size() / q.element_size(),
    "original_bytes": original.numel() * original.element_size(),
    "quantized_bytes": q.numel() * q.element_size() + scale.numel() * scale.element_size(),
}
```

**`element_size()`** gives the bytes per element (4 for float32, 1 for int8). With that the
compression comes out on its own, with no magic numbers.

**The quantized bytes include the scales.** With one scale per row they are negligible, but
counting them is the honest thing. There is a test that checks it.

**The relative error is the metric worth looking at**, because it is independent of the scale
of the data: you can compare different layers. The test
`test_the_relative_error_is_independent_of_the_scale` multiplies the weights by 1000 and
verifies it does not change.

## What you should see in the demo

**The example by hand:**

| original | int8 | recovered | error |
|---|---|---|---|
| +0.1200 | 34 | +0.1205 | 0.0005 |
| **−0.4500** | **−127** | **−0.4500** | **0.0000** |
| +0.0300 | 8 | +0.0283 | 0.0017 |
| +0.2800 | 79 | +0.2799 | 0.0001 |

The −0.45 is recovered **exactly** because it is the maximum and maps right onto −127. The rest
lose up to half a unit of scale.

**On the real model:**

| matrix | per channel | per tensor |
|---|---|---|
| token_embedding | 0.711% | 1.108% |
| q_proj | 0.714% | 1.067% |
| down_proj | 0.779% | 1.116% |

**Per channel always wins**, and by a consistent margin: a single row with large values does
not drag the others along. It costs one extra vector of scales, which is negligible.

And the result that matters: **35.7 MB → 9.0 MB, 4× smaller**, with a 0.7% error in the
weights.

## Two nuances that are usually left out

**That a 0.7% error barely affects the model's quality is an empirical fact, not a theorem.**
Nobody predicted that networks would be so robust to quantization; it was discovered by trying.
And it is not universal: there are layers and architectures where int8 does degrade
appreciably, and that is why mixed schemes exist that keep some layers at higher precision.

**Quantizing the weights does not speed anything up on its own** if afterwards you convert to
float to multiply, which is what this exercise does. Real acceleration requires kernels that
operate natively in int8, and that depends on the hardware. What you always gain is memory,
and on a GPU with 6 GB that can be the difference between the model fitting or not.

---

## End of the course

You have written **all** the pieces: attention, RoPE, SwiGLU, RMSNorm, AdamW, the KV cache, the
BPE tokenizer, the training loop. All validated numerically against PyTorch or against the
original papers.

A frontier model uses exactly these pieces. Bigger, with vastly more engineering around them,
with data nobody publishes and compute that costs a hundred million. But the same ones.

What you take away that does not appear in the tutorials: **you know what is not known**. That
SwiGLU works without an explanation and its own author says so. That Adam dominates without
anyone quite understanding why. That scaling laws have wider confidence intervals than
reported. That the benchmarks are contaminated. That interpretability has explained a few
circuits and nowhere near a whole model.

That is what separates reading a paper with judgement from reading it with faith.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
def quantize_int8_symmetric(
    weight: torch.Tensor, per_channel: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    if per_channel and weight.dim() >= 2:
        max_abs = weight.abs().amax(dim=-1, keepdim=True)
    else:
        max_abs = weight.abs().amax()

    # clamp_min avoids dividing by zero if a row is all zeros.
    scale = (max_abs / 127.0).clamp_min(1e-12)
    quantized = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return quantized.to(torch.float32) * scale


def quantization_error(original: torch.Tensor, per_channel: bool = True) -> dict[str, float]:
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
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
