# 12 — Annotated solution

## Exercise 1 — `model_flops_per_token`

```python
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
```

It is the computation from module 01, but returning the breakdown. And the breakdown is what
makes the function useful:

| context | matmul | attention | % attention |
|---|---|---|---|
| 128 | 160.8M | 2.9M | 2% |
| **512** | **160.8M** | **11.8M** | **7%** |
| 2048 | 160.8M | 47.2M | 23% |
| 8192 | 160.8M | 188.7M | **54%** |

The matmul term **does not move**: it only depends on the size of the model. The attention
one grows linearly with the context, and from 2048 on it starts to dominate. With that table
in front of you, the "should I lengthen the context?" decision stops being blind.

**The final projection counts even with weight tying.** Tying the weights saves memory, not
compute: the matmul $(B\cdot T, d) \times (d, V)$ runs all the same. There is a test that
checks it.

## Exercise 2 — `compute_mfu`

```python
if peak_tflops <= 0:
    raise ValueError("peak_tflops has to be positive")
return tokens_per_second * flops_per_token / (peak_tflops * 1e12)
```

One line. What is interesting is how to read it.

In the demo, really measuring on this hardware, the MFU rises with the batch size and then
flattens out. That flattening point is where you stop being limited by kernel launches and
start being limited by the computation.

**Nobody reaches 1.** And with a 9M model, 0.1–0.2 is already good: 320×320 matrices are not
enough to saturate the tensor cores. It is the same phenomenon you measured in module 01.

**The value of the MFU is not its absolute number, it is that it is comparable.** It does not
depend on the model or on the hardware, so you can change the batch size, switch on
`torch.compile` or move the dataloader to another thread, and see whether it goes up.

## Exercise 3 — `chinchilla_optimal_allocation`

```python
if compute_budget <= 0:
    raise ValueError("the compute budget has to be positive")

params = (compute_budget / (6 * tokens_per_param)) ** 0.5
tokens = tokens_per_param * params

return {"params": params, "tokens": tokens,
        "tokens_per_param": tokens_per_param, "compute": compute_budget}
```

The derivation, starting from $C = 6ND$ and $D = kN$:

$$C = 6N(kN) = 6kN^2 \quad \Longrightarrow \quad N = \sqrt{\frac{C}{6k}}$$

### The check that gives confidence

This is not an abstract arithmetic exercise. Feed it Chinchilla's real budget,
$5.88 \times 10^{23}$ FLOPs:

```
N = √(5.88·10²³ / 120) = 7.0·10¹⁰ = 70.0 billion
```

**The real model had 70 billion parameters.** The formula nails it.

### The table that made the paper famous

| model | params | tokens | tok/param | optimum |
|---|---|---|---|---|
| GPT-3 | 175 B | 300 B | **1.7** | 51 B |
| Gopher | 280 B | 300 B | **1.1** | 65 B |
| Chinchilla | 70 B | 1.4 T | 20 | 70 B ✓ |
| Llama-3 8B | 8 B | 15 T | **1875** | 78 B |
| **ours** | 7.6 M | 500 M | **66** | 14 M |

GPT-3 was **twelve times under-trained**. With its compute budget, the optimum would have
been a model of 51 billion parameters —a third of the size— trained on more than three times
the data.

And Llama-3 goes **ninety times above** Chinchilla, which is not a mistake: its objective
function is simply a different one. Chinchilla optimizes **training** compute; if the model is
going to be run millions of times afterwards, a smaller and more heavily trained one is
better, because inference is paid every time.

Our model is at 66 tokens per parameter, three times above. Deliberate, for the same reason
and because at this scale over-training costs hours.

## About what the formula does not say

It is worth keeping the debate section of `THEORY.md` in mind: Chinchilla's coefficients were
fitted to a particular range of scales, a 2024 reanalysis found the confidence intervals were
much wider than reported, and scaling laws predict **loss**, not capabilities.

And above all: no scaling law captures **data quality**. The TinyStories paper shows that a
small, very clean dataset lets tiny models generate coherent text, something you do not get
with the same amount of internet text. No $N$ or $D$ captures that.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
def model_flops_per_token(cfg: ModelConfig, include_backward: bool = True) -> dict[str, int]:
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
    if peak_tflops <= 0:
        raise ValueError("peak_tflops has to be positive")
    return tokens_per_second * flops_per_token / (peak_tflops * 1e12)


def chinchilla_optimal_allocation(
    compute_budget: float, tokens_per_param: float = 20.0
) -> dict[str, float]:
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
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
