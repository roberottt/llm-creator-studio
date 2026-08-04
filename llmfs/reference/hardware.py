"""Reference for module 01: measuring the hardware and estimating the speed ceiling."""

from __future__ import annotations

import time

import torch

from llmfs.device import DeviceConfig, get_device


def matmul_flops(size: int) -> int:
    """FLOPs of multiplying two square matrices of side `size`.

    Each element of the output is a dot product of length `size`: `size` multiplications
    and `size - 1` additions. By convention this is counted as `2 * size` (one addition of
    difference is irrelevant, and this way the number is comparable with everyone else's).
    There are `size^2` output elements.
    """
    return 2 * size**3


def measure_matmul_tflops(
    cfg: DeviceConfig | None = None,
    size: int = 2048,
    dtype: torch.dtype | None = None,
    warmup: int = 3,
    iters: int = 10,
) -> float:
    """Measure the effective TFLOPS of a large matmul on this device.

    The warmup is not optional: the first call pays for cuBLAS/Metal kernel selection and
    memory allocation, and comes out 10 to 100 times slower. And you have to synchronize
    before looking at the clock, because GPU calls are asynchronous: without
    `synchronize()` you would be measuring the time to enqueue, not the time to compute.
    """
    cfg = cfg or get_device()
    if dtype is None:
        dtype = cfg.amp_dtype or torch.float32

    a = torch.randn(size, size, device=cfg.device, dtype=dtype)
    b = torch.randn(size, size, device=cfg.device, dtype=dtype)

    for _ in range(warmup):
        a @ b
    cfg.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    cfg.synchronize()
    elapsed = time.perf_counter() - start

    return (matmul_flops(size) * iters) / elapsed / 1e12


def transformer_flops_per_token(
    n_layers: int,
    d_model: int,
    d_ff: int,
    context_length: int,
    vocab_size: int,
    n_ffn_matrices: int = 3,
    include_backward: bool = True,
) -> int:
    """Training FLOPs per token of a decoder transformer.

    This uses the standard approximation (Kaplan 2020, appendix B):

    - A matmul with `P` parameters costs `2P` FLOPs per token in the forward pass.
    - The backward pass costs roughly twice the forward (one pass for the gradient with
      respect to the input and another with respect to the weights). Total: `3x` forward.

    That is where the `6N` you see everywhere comes from: `2 * P * 3 = 6P`.

    The quadratic attention term is counted separately because it does not come from
    parameters: `Q K^T` costs `2 * T * d_model` per token and layer, and `softmax @ V` the
    same again, hence `4 * n_layers * T * d_model` in the forward pass.

    A deliberate simplification: we do not divide by two even though the causal mask only
    computes half of the matrix. That is the convention nanoGPT and the papers use, so the
    MFU numbers you compute will be comparable with theirs. Softmax, normalizations and
    activations are also ignored; they are memory-bound and add little to the count.

    Args:
        n_ffn_matrices: 3 for SwiGLU (gate, up, down), 2 for a classic FFN.

    Returns:
        FLOPs per token, as an integer.
    """
    attention_params = 4 * d_model**2
    ffn_params = n_ffn_matrices * d_model * d_ff
    matmul_params = n_layers * (attention_params + ffn_params)
    matmul_params += d_model * vocab_size  # the final projection to logits

    forward = 2 * matmul_params + 4 * n_layers * context_length * d_model
    return int(3 * forward if include_backward else forward)


def estimate_tokens_per_second(tflops: float, flops_per_token: int, mfu: float = 0.4) -> float:
    """Achievable tokens per second, given the measured peak and an assumed MFU.

    `mfu` (Model FLOPs Utilization) is the fraction of the theoretical peak you actually
    get. Realistic values: 0.4-0.5 for large, well-optimized models; 0.1-0.2 for a 9M
    model, where kernel launches and memory bandwidth weigh more than the arithmetic.
    """
    if flops_per_token <= 0:
        raise ValueError("flops_per_token must be positive")
    return tflops * 1e12 * mfu / flops_per_token
