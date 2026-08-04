"""Module 01 - Environment and hardware.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement -> `llmfs check 01` -> if you get stuck, `llmfs hint 01 -e N`
-> and if you are still stuck, `SOLUTION.md` has the complete code to copy.

WHAT YOU ARE GOING TO BUILD
===========================

A calculator for how long your training run will take. Three functions:

    measure_matmul_tflops        how many operations per second your GPU REALLY delivers
    transformer_flops_per_token  how many operations it costs to process a token
    estimate_tokens_per_second   dividing one by the other, the speed

With that you can answer "is this two hours or two weeks?" before writing the model.

VOCABULARY YOU ARE GOING TO NEED
================================

- **FLOP**: one operation on decimal numbers (an addition or a multiplication). The unit
  used to measure what training something costs.
- **TFLOPS**: trillions of FLOPs per second. What your GPU delivers.
- **MFU** (Model FLOPs Utilization): what fraction of the theoretical peak you really get.
  Nobody reaches 1; with a small model, 0.1-0.2 is already good.
- **forward / backward**: pushing the data through the network, and computing how to adjust
  the weights. The backward pass costs roughly twice the forward.

    llmfs demo 01     measures your GPU and estimates the time of the final run
"""

from __future__ import annotations

import time

import torch

from llmfs.device import DeviceConfig, get_device


def measure_matmul_tflops(
    cfg: DeviceConfig | None = None,
    size: int = 2048,
    dtype: torch.dtype | None = None,
    warmup: int = 3,
    iters: int = 10,
) -> float:
    """Measures how many operations per second your GPU REALLY does.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Seven steps. None of them is tricky on its own; the order is what matters.

        1. If `cfg` is None, get it with `get_device()`.
           If `dtype` is None, use `cfg.amp_dtype`, and if that is None too,
           `torch.float32`.

        2. Create the two matrices you are going to multiply:

               a = torch.randn(size, size, device=cfg.device, dtype=dtype)
               b = torch.randn(size, size, device=cfg.device, dtype=dtype)

        3. WARM UP: repeat the operation `a @ b` `warmup` times. Do NOT time this.
           (It does not matter that you discard the result: PyTorch does not eliminate
           dead code.)

        4. cfg.synchronize()          <- wait for the GPU to actually finish

        5. Start the clock and repeat the multiplication `iters` times:

               t0 = time.perf_counter()
               for _ in range(iters):
                   a @ b

        6. Synchronize AGAIN, and only then stop the clock:

               cfg.synchronize()
               seconds = time.perf_counter() - t0

        7. Return the TFLOPS:

               return (2 * size**3 * iters) / seconds / 1e12

    WHY THAT FORMULA
    ----------------
    Multiplying two matrices of side `size` produces `size**2` numbers, and each one comes
    from a dot product of length `size`: `size` multiplications and `size-1` additions,
    which by convention are counted as `2 * size`. Total: `2 * size**3` operations.

    Dividing by the seconds gives FLOPS, and by `1e12` gives TeraFLOPS.

    WATCH OUT FOR STEPS 3, 4 AND 6
    ------------------------------
    They are what make the number mean anything, and all three are easy to skip.

    **Without warming up (step 3)**, the first multiplication at a given size is between 10
    and 100 times slower: the GPU is choosing which kernel to use and allocating memory.
    With `iters=10` and no warmup, that first call dominates the average.

    **Without synchronizing (steps 4 and 6)**, `a @ b` on a GPU waits for nothing: it queues
    the work and returns control. You would be measuring how long the CPU takes to enqueue
    an order (about 20 microseconds) and you would get thousands of TFLOPS. There is a test
    that bounds the result specifically to catch this.

    Args:
        cfg: the device. If it is `None`, it is fetched with `get_device()`.
        size: the side of the matrices.
        dtype: the tensors' type. If it is `None`, look at `cfg.amp_dtype`, and if there is
            none either, use `torch.float32`.
        warmup: how many warmup iterations (they are not timed).
        iters: how many timed iterations.

    Returns:
        The effective TFLOPS, as a positive float.
    """
    raise NotImplementedError("TODO: module 01, exercise 1 - measure_matmul_tflops")


def transformer_flops_per_token(
    n_layers: int,
    d_model: int,
    d_ff: int,
    context_length: int,
    vocab_size: int,
    n_ffn_matrices: int = 3,
    include_backward: bool = True,
) -> int:
    """Computes how many operations it costs to process ONE token.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Three lines of arithmetic. No loops and no odd conditions.

        1. Count the parameters that take part in matrix multiplications:

               params = n_layers * (4 * d_model**2 + n_ffn_matrices * d_model * d_ff)
               params += d_model * vocab_size

        2. The forward cost:

               forward = 2 * params + 4 * n_layers * context_length * d_model

        3. Return, multiplying by 3 if there is a backward pass:

               return int(3 * forward if include_backward else forward)

    WHERE EACH TERM COMES FROM
    --------------------------
    - `4 * d_model**2` are attention's four projections (Wq, Wk, Wv, Wo), each a
      `d_model x d_model` matrix with no bias.
    - `n_ffn_matrices * d_model * d_ff` is the FFN: 3 matrices with SwiGLU (gate, up, down),
      2 with a classic FFN.
    - `d_model * vocab_size` is the final projection to logits.
    - The `2 *` comes from each parameter taking part in one multiplication and one addition.
    - `4 * n_layers * context_length * d_model` is attention itself (`Q @ K^T` and
      `softmax @ V`). That term does NOT come from parameters: it grows with the CONTEXT,
      not with the model size.
    - The `3 *` is because the backward pass costs twice the forward: it does two
      multiplications for each one in the forward, one for the gradient with respect to the
      input and one with respect to the weights.

    TWO THINGS THAT GET FORGOTTEN
    -----------------------------
    1. The final projection counts EVEN if you use weight tying. Tying the weights saves
       memory, not computation: the matmul happens all the same.
    2. Do NOT divide by two even though the causal mask only computes half the triangle. It
       is the nanoGPT and paper convention; if you divide, your MFU will not be comparable
       with anyone's.

    CHECK IT
    --------
    With the final model's config (6 layers, d_model 320, d_ff 896, context 512, vocab 4096)
    it has to give exactly **65,372,160**.

    Args:
        n_layers: number of layers.
        d_model: the model dimension.
        d_ff: the FFN's inner dimension.
        context_length: length of the context window.
        vocab_size: vocabulary size.
        n_ffn_matrices: 3 for SwiGLU, 2 for a classic FFN.
        include_backward: if `True`, multiply the total by 3.

    Returns:
        The FLOPs per token, as an integer.
    """
    raise NotImplementedError("TODO: module 01, exercise 2 - transformer_flops_per_token")


def estimate_tokens_per_second(tflops: float, flops_per_token: int, mfu: float = 0.4) -> float:
    """Estimates how many tokens per second you are going to process.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Two lines.

        1. If `flops_per_token` is not positive, raise `ValueError`.

        2. Return:

               return tflops * 1e12 * mfu / flops_per_token

    The `1e12` converts TeraFLOPS into FLOPS.

    WHY THE `mfu`
    -------------
    You never use 100% of a GPU's power. MFU (Model FLOPs Utilization) is the fraction you
    really get, and you have to multiply by it or the estimate comes out absurdly
    optimistic.

    Realistic values:
        0.4 - 0.5   billion-parameter models, well optimized
        0.1 - 0.2   our 9M model

    WHY VALIDATE
    ------------
    A division by zero here silently produces `inf` and meaningless estimates, which you
    would discover much later. A `ValueError` with a clear message costs one line.

    Args:
        tflops: the measured peak, in TFLOPS (what exercise 1 returns).
        flops_per_token: what exercise 2 returns.
        mfu: the fraction of the peak you expect to reach.

    Returns:
        Tokens per second, as a float.

    Raises:
        ValueError: if `flops_per_token` is not positive.
    """
    raise NotImplementedError("TODO: module 01, exercise 3 - estimate_tokens_per_second")
