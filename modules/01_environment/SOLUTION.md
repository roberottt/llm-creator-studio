# 01 — Annotated solution

## Exercise 1 — `measure_matmul_tflops`

The structure is: allocate tensors, warm up, synchronize, time N iterations, synchronize,
divide.

The two mistakes people make here are always the same ones.

**Not synchronizing.** `a @ b` on CUDA or MPS queues work and returns control immediately.
If you write `t0 = perf_counter(); a @ b; t1 = perf_counter()`, you measure how long the CPU
takes to enqueue an order — on the order of 20 µs — and you get figures in the thousands of
TFLOPS. The test `test_the_measurement_returns_a_plausible_number` bounds the result
precisely to catch this. You have to call `cfg.synchronize()` before each `perf_counter()`.

**Not warming up.** The first invocation at a given size triggers cuBLAS's kernel-selection
heuristic (or the shader compilation in Metal) and allocates memory in the caching
allocator. It can be two orders of magnitude slower. With `iters=10` and no warmup, that
first call dominates the average.

A minor detail: you do not need to store the matmul's result, but do not optimize it away to
`None` either — PyTorch does not eliminate dead code, so a bare `a @ b` runs all the same.

The default `dtype` comes from `cfg.amp_dtype`, which on CUDA is fp16 and on MPS is `None`;
that is why the fallback is `torch.float32`.

## Exercise 2 — `transformer_flops_per_token`

Pure arithmetic; the interesting part is understanding where each term comes from.

```
params_matmul = n_layers * (4·d² + n_ffn·d·d_ff) + d·V
forward       = 2·params_matmul + 4·n_layers·T·d
total         = 3·forward
```

The `2·params` is the rule of thumb: *each parameter contributes one multiplication and one
addition per token*. The `4·n_layers·T·d` is attention, and it is worth being clear that it
is **not a parameter term**: it is the products $QK^\top$ and $\text{softmax}\cdot V$, whose
cost depends on how many tokens are in the context, not on how many weights the model has.

The backward's `3×` deserves a comment. The forward computes $y = Wx$. The backward needs
two things: $\partial L/\partial x = W^\top \, \partial L/\partial y$ to keep propagating,
and $\partial L/\partial W = \partial L/\partial y \, x^\top$ to update the weights. Those
are two matmuls the same size as the forward's, hence the extra factor of 2.

The `lm_head` counts even if `tie_embeddings` is `true`. Tying the weights saves memory and
parameters, not computation: the matmul $(B \cdot T, d) \times (d, V)$ runs all the same.

On not dividing by two for causal attention: it is true that only the lower triangle is
computed, and with an ideal kernel it would cost half. In practice, dense kernels compute
the whole matrix and mask it, and Flash-style kernels skip whole blocks but not exactly
half. The convention of counting it in full comes from nanoGPT and is what the papers use
when reporting MFU. If you divided by two, your MFU would come out ~9% lower than everyone
else's on the same hardware.

## Exercise 3 — `estimate_tokens_per_second`

$$\text{tokens/s} = \frac{\text{TFLOPS} \times 10^{12} \times \text{MFU}}{C_{\text{token}}}$$

The `flops_per_token > 0` check is not decorative: it is what prevents a silent division by
zero that would produce `inf` and absurd estimates.

## What to expect when you run the demo

On the RTX 2060 you will see somewhere between 15 and 30 TFLOPS in fp16 with 2048–4096
matrices, and less than 2 TFLOPS with 128×128. That drop is the module's message: **the
matrix size determines whether you use the hardware**, and our 320-dimension model lives in
the bad part of that curve. It is a conscious decision: we want a model that trains in
hours, not one that maximizes MFU.

On the M5 with MPS the numbers are flatter across dtypes, because the memory is unified and
there is no PCIe traffic to amortize; fp16 wins less than it would on a discrete GPU.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
def matmul_flops(size: int) -> int:
    return 2 * size**3


def measure_matmul_tflops(
    cfg: DeviceConfig | None = None,
    size: int = 2048,
    dtype: torch.dtype | None = None,
    warmup: int = 3,
    iters: int = 10,
) -> float:
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
    attention_params = 4 * d_model**2
    ffn_params = n_ffn_matrices * d_model * d_ff
    matmul_params = n_layers * (attention_params + ffn_params)
    matmul_params += d_model * vocab_size  # the final projection to logits

    forward = 2 * matmul_params + 4 * n_layers * context_length * d_model
    return int(3 * forward if include_backward else forward)


def estimate_tokens_per_second(tflops: float, flops_per_token: int, mfu: float = 0.4) -> float:
    if flops_per_token <= 0:
        raise ValueError("flops_per_token must be positive")
    return tflops * 1e12 * mfu / flops_per_token
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
