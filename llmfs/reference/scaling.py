"""Reference for module 12: efficiency and scaling laws."""

from __future__ import annotations

from llmfs.config import ModelConfig


def model_flops_per_token(cfg: ModelConfig, include_backward: bool = True) -> dict[str, int]:
    """FLOPs per token, split into matmuls and attention.

    It is the same calculation as module 01, but returning the breakdown instead of a single
    number. Knowing how much each part weighs is what lets you decide whether it is better
    to change the context length or the model size.

    Returns:
        A dict with `matmul`, `attention`, `total` and `params_matmul`.
    """
    d, ff, v = cfg.d_model, cfg.d_ff, cfg.vocab_size
    n_ffn = 3 if cfg.activation == "swiglu" else 2

    params_matmul = cfg.n_layers * (4 * d * d + n_ffn * d * ff) + d * v

    matmul = 2 * params_matmul
    attention = 4 * cfg.n_layers * cfg.context_length * d

    factor = 3 if include_backward else 1
    return {
        "matmul": matmul * factor,
        "attention": attention * factor,
        "total": (matmul + attention) * factor,
        "params_matmul": params_matmul,
    }


def compute_mfu(
    tokens_per_second: float, flops_per_token: int, peak_tflops: float
) -> float:
    """Model FLOPs Utilization: what fraction of the hardware peak you are getting.

    $$\\text{MFU} = \\frac{\\text{tokens/s} \\times C_{\\text{token}}}{\\text{FLOPS pico}}$$

    It is THE metric for knowing whether your training run is well optimized, and it is
    independent of the model and the hardware, so it can be compared across configurations.

    Reference values for what is reasonable:
        0.4 - 0.5   large, well-optimized models on A100/H100
        0.2 - 0.3   mid-sized models
        0.1 - 0.2   our 9M model
        < 0.05      something is wrong: look at the dataloader or the batch size

    With a small model a low MFU is expected: 320x320 matrices are not enough to saturate
    the tensor cores, and the time goes into launching kernels and moving memory.
    """
    if peak_tflops <= 0:
        raise ValueError("peak_tflops has to be positive")
    return tokens_per_second * flops_per_token / (peak_tflops * 1e12)


def chinchilla_optimal_allocation(
    compute_budget: float, tokens_per_param: float = 20.0
) -> dict[str, float]:
    """Split a compute budget between parameters and tokens, following Chinchilla.

    THE PROBLEM. You have a fixed budget of FLOPs. You can spend it on a large model with
    little data, or on a small one with a lot. Which gives the lower loss?

    THE ANSWER (Hoffmann et al. 2022). Both should grow PROPORTIONALLY. Their empirical
    result: about **20 tokens per parameter**.

    It was an important result because it contradicted the practice of the time. GPT-3 had
    175 billion parameters trained on 300 billion tokens: 1.7 tokens per parameter, almost
    twelve times below the optimum. Chinchilla (70 billion parameters, 1.4 trillion tokens)
    beat it on almost every benchmark with less than half the parameters.

    THE ARITHMETIC. With `C = 6ND`:

        C = 6 * N * (20N) = 120 N^2      ->     N = sqrt(C / 120)
        D = 20 * N

    OUR CASE. The model has 7.62M non-embedding parameters and will see 500M tokens: 65
    tokens per parameter, more than three times above the "optimum". That is deliberate, for
    two reasons:

    1. Chinchilla optimizes TRAINING compute. If the model is going to be used a lot
       afterwards, a smaller and more heavily trained model is better: inference is paid
       every time. Llama-3 takes this to the extreme with ~1,800 tokens per parameter.
    2. At this scale, over-training is cheap (hours) and gives a noticeably better model.

    Returns:
        `{"params", "tokens", "tokens_per_param", "compute"}`.
    """
    if compute_budget <= 0:
        raise ValueError("the compute budget has to be positive")

    params = (compute_budget / (6 * tokens_per_param)) ** 0.5
    tokens = tokens_per_param * params

    return {
        "params": params,
        "tokens": tokens,
        "tokens_per_param": tokens_per_param,
        "compute": compute_budget,
    }
