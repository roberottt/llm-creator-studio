"""Module 12 - Efficiency and scaling laws.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> implement -> `llmfs check 12` -> `llmfs hint 12 -e N`
-> `SOLUTION.md` has the complete code.

None of the three functions is more than five lines of code. The difficulty is in
understanding what the numbers mean.

WHAT YOU ARE GOING TO BUILD
===========================

    model_flops_per_token          (ex. 1)  what a token costs, broken down
    compute_mfu                    (ex. 2)  what fraction of your GPU you are using
    chinchilla_optimal_allocation  (ex. 3)  how to split the compute budget

Exercise 3 reproduces a result that in 2022 showed the entire industry was training its
models wrong. And you are going to check it against real historical models.

VOCABULARY YOU ARE GOING TO NEED
================================

- **MFU** (Model FLOPs Utilization): tokens/s x FLOPs per token, divided by your hardware's
  peak. Nobody reaches 1.
- **compute budget**: how many total FLOPs you can afford to spend on training.
- **Chinchilla**: the 2022 paper that measured how to split that budget between model size
  and amount of data. Answer: ~20 tokens per parameter.
- **non-embedding parameters**: the total minus the embedding table. It is what the scaling
  laws use, because embeddings scale differently.
- **over-trained / under-trained**: above or below those 20 tokens per parameter. Neither is
  necessarily bad: it depends on your objective.

    llmfs demo 12     measures your real MFU and reproduces the Chinchilla result
"""

from __future__ import annotations

from llmfs.config import ModelConfig


def model_flops_per_token(cfg: ModelConfig, include_backward: bool = True) -> dict[str, int]:
    """FLOPs per token, split into matmuls and attention.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Four steps, all arithmetic on the config fields.

        1. How many matrices the FFN has:

               n_ffn = 3 if cfg.activation == "swiglu" else 2

        2. The parameters that take part in matrix multiplications:

               params_matmul = cfg.n_layers * (
                   4 * cfg.d_model**2 + n_ffn * cfg.d_model * cfg.d_ff
               ) + cfg.d_model * cfg.vocab_size

        3. The two terms:

               matmul = 2 * params_matmul
               attention = 4 * cfg.n_layers * cfg.context_length * cfg.d_model

        4. The backward, if applicable, and the output dict:

               if include_backward:
                   matmul *= 3
                   attention *= 3

               return {
                   "matmul": matmul,
                   "attention": attention,
                   "total": matmul + attention,
                   "params_matmul": params_matmul,
               }

    WHERE EACH CONSTANT COMES FROM
    ------------------------------
    **The 2 in `matmul`.** Multiplying a matrix by a vector does, for each weight, one
    multiplication and one addition: 2 operations per parameter. Hence the `2 * params`.

    **The 3 of the backward.** The backward pass costs roughly TWICE what the forward does
    (you have to compute the gradient with respect to the input and with respect to the
    weights, two matmuls where the forward did one). Forward + backward = 1 + 2 = 3.

    **The `d*vocab_size` counts EVEN IF there is weight tying.** Tying the weights saves
    MEMORY, not COMPUTE: the final multiplication happens all the same, with the same matrix.

    **The 4 in `attention`.** Those are the two matmuls that do NOT involve parameters:
    `Q@K^T` and `attn@V`. Each one costs `2 * T * d` per token, and there are two of them:
    `4 * T * d` per layer.

    WHY THEY ARE RETURNED SEPARATELY
    --------------------------------
    Because the two terms grow with different things:

        matmul     grows with the SIZE of the model (d_model, n_layers, d_ff)
        attention  grows with the CONTEXT (context_length)

    With our config (T=512) attention is 18% of the total. With T=4096 it would be 64%. That
    breakdown tells you instantly whether lengthening the context is going to be expensive,
    without having to try it.

    Args:
        cfg: the model configuration.
        include_backward: whether to include the cost of the backward pass (True for
            training, False for pure inference).

    Returns:
        A dict with `matmul`, `attention`, `total` and `params_matmul`.
        `total` = `matmul` + `attention`.
    """
    raise NotImplementedError("TODO: module 12, exercise 1 - model_flops_per_token")


def compute_mfu(tokens_per_second: float, flops_per_token: int, peak_tflops: float) -> float:
    """Model FLOPs Utilization: what fraction of the hardware peak you are using.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Two lines.

        1. The validation (otherwise a silent division by zero gives you `inf` and you never
           find out):

               if peak_tflops <= 0:
                   raise ValueError(f"peak_tflops has to be positive: {peak_tflops}")

        2. The formula:

               return tokens_per_second * flops_per_token / (peak_tflops * 1e12)

    The `1e12` converts TeraFLOPS into FLOPS, so that the top and the bottom are in the same
    units. It is the only place where you can get it wrong.

    AN EXAMPLE WITH REAL NUMBERS
    ----------------------------
    The ones from the `tiny_char` run on the MacBook:

        tokens_per_second = 112,000
        flops_per_token   = 2.6e7        (from exercise 1)
        peak_tflops       = 14.2         (M5, fp32)

        MFU = 112000 * 2.6e7 / 1.42e13 = 0.205,  that is 20.5%

    HOW TO READ IT
    --------------
        0.4 - 0.5   large, well-optimized models on A100/H100
        0.2 - 0.3   mid-sized models
        0.1 - 0.2   our 9M model
        < 0.05      something is wrong: look at the dataloader or raise the batch size

    NOBODY REACHES 1. The theoretical peak is only reached with enormous, perfectly aligned
    matmuls and absolutely nothing else in the way. With a small model a low MFU is
    unavoidable: 320x320 matrices are not enough to saturate the tensor cores, and a
    significant part of the time goes into launching kernels instead of computing. It is not
    your fault.

    WHAT IT IS REALLY FOR
    ---------------------
    Not for its absolute value, but because it is COMPARABLE: it does not depend on the model
    or on the hardware. "1200 tokens per second" tells you nothing; "18% MFU" does. You change
    the batch size, switch on `torch.compile`, move the dataloader to another thread, and see
    whether the number goes up. It is the thermometer for optimizations.

    Args:
        tokens_per_second: the measured throughput, not the theoretical one.
        flops_per_token: from `model_flops_per_token(...)["total"]`.
        peak_tflops: the hardware peak, from `llmfs device`.

    Returns:
        A fraction between 0 and 1.

    Raises:
        ValueError: if `peak_tflops` is not positive.
    """
    raise NotImplementedError("TODO: module 12, exercise 2 - compute_mfu")


def chinchilla_optimal_allocation(
    compute_budget: float, tokens_per_param: float = 20.0
) -> dict[str, float]:
    """Splits a compute budget between parameters and tokens.

    WHAT YOU HAVE TO WRITE
    ----------------------
    Three lines and a dict.

        1. The validation:

               if compute_budget <= 0:
                   raise ValueError(f"the budget has to be positive: {compute_budget}")

        2. Solve for N and get D:

               params = (compute_budget / (6 * tokens_per_param)) ** 0.5
               tokens = tokens_per_param * params

           (`** 0.5` is the square root. It is written that way so you do not have to import
           `math` in this file, which does not import it.)

        3. Return the four things:

               return {
                   "params": params,
                   "tokens": tokens,
                   "tokens_per_param": tokens_per_param,
                   "compute": compute_budget,
               }

    WHERE THAT SQUARE ROOT COMES FROM
    ---------------------------------
    From the `C = 6ND` of module 01 (total cost = 6 x parameters x tokens) plus the constraint
    that you want `D = k*N`, with k the tokens per parameter:

        C = 6 * N * (k*N) = 6k * N²

        N = sqrt( C / (6k) )
        D = k * N

    That is: if you double the budget, the optimal model does NOT double, it grows by 41%
    (sqrt(2)). And the data by 41% too. Both at the same time.

    THE PROBLEM IT SOLVES
    ---------------------
    You have a fixed budget of FLOPs -one GPU and two weeks, say-. You can spend it on a large
    model with little data or on a small one with a lot of data. Which one ends up with the
    lower loss?

    THE ANSWER (Hoffmann et al. 2022, "Chinchilla")
    -----------------------------------------------
    Both have to grow PROPORTIONALLY: about 20 tokens per parameter.

    It was an important result because it contradicted the practice of the time. GPT-3 had 175
    billion parameters and was trained on 300 billion tokens: 1.7 tokens per parameter, TWELVE
    TIMES below the optimum. Chinchilla trained a model four times smaller with four times
    more data, with the same compute, and won on almost every evaluation.

    CHECK IT WITH CHINCHILLA ITSELF
    -------------------------------
    Its real budget was 5.76e23 FLOPs. With k=20:

        N = sqrt(5.76e23 / 120) = 6.9e10 = 69 billion

    The real model had 70 billion. The formula nails it. Seeing it work on a historical case
    gives a lot more confidence than reading it.

    AND NOW LOOK AT OUR MODEL
    -------------------------
    Our model has 7.62M non-embedding parameters, so the Chinchilla optimum would be about
    152M tokens. We are going to train it with 500M: 65 tokens per parameter, more than three
    times above the "optimum". It is deliberate, it is not a mistake.

    Chinchilla optimizes loss per TRAINING FLOP. If the model is going to be used a lot
    afterwards, it pays off to over-train a small model: every inference is cheaper forever.
    That is exactly what Llama does, and it is why Llama-7B was trained on 1 trillion tokens
    (143 per parameter) instead of on 140 billion.

    Args:
        compute_budget: available FLOPs. Has to be positive.
        tokens_per_param: the Chinchilla constant. 20 by default.

    Returns:
        `{"params", "tokens", "tokens_per_param", "compute"}`, all floats.

    Raises:
        ValueError: if the budget is not positive.
    """
    raise NotImplementedError("TODO: module 12, exercise 3 - chinchilla_optimal_allocation")
