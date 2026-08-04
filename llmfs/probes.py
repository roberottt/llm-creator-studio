"""The smoke tests the bridge runs to decide whether your implementation is usable.

A probe does NOT check that your code is correct: it checks that it can be called and that
it returns something of the expected shape. Correctness is verified by the module tests,
comparing against the reference with `torch.allclose`.

A probe's contract:

    @probe("symbol_name")
    def _(impl):
        ...   # call impl with tiny data
        ...   # raise any exception if it is not usable

If a symbol has no registered probe, the bridge only checks that it exists. That is enough
for functions whose unimplemented version raises `NotImplementedError` on entry, which is
how every `exercises.py` is written.

Probes are registered phase by phase, as the modules get written. `tests/` includes a test
that checks no exercise with its own test is left without its corresponding reference piece.
"""

from __future__ import annotations

from typing import Any, Callable

PROBES: dict[str, Callable[[Any], None]] = {}


def probe(name: str) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
    """Register the smoke test of a curriculum symbol."""

    def decorator(fn: Callable[[Any], None]) -> Callable[[Any], None]:
        if name in PROBES:
            raise RuntimeError(f"Duplicate probe for {name!r}")
        PROBES[name] = fn
        return fn

    return decorator


def expect_shape(tensor: Any, shape: tuple[int, ...], what: str = "the output") -> None:
    """Helper for the probes: fails with a readable message if the shape does not match."""
    actual = tuple(getattr(tensor, "shape", ()))
    if actual != shape:
        raise ValueError(f"{what} has shape {actual}, {shape} was expected")


# ----------------------------------------------------------------- module 00


@probe("next_token_probs")
def _(impl: Any) -> None:
    probs = impl({"a": 3, "b": 1})
    if set(probs) != {"a", "b"}:
        raise ValueError(f"should return the keys 'a' and 'b', returns {set(probs)}")
    if abs(sum(probs.values()) - 1.0) > 1e-9:
        raise ValueError(f"the probabilities sum to {sum(probs.values())}, they should sum to 1")


@probe("sample_next_token")
def _(impl: Any) -> None:
    import random

    chosen = impl({"a": 0.5, "b": 0.5}, random.Random(0))
    if chosen not in {"a", "b"}:
        raise ValueError(f"returned {chosen!r}, which is not a key of the distribution")


@probe("generate_naive")
def _(impl: Any) -> None:
    import random

    out = impl({"a": {"b": 1}, "b": {"a": 1}}, "a", 6, random.Random(0))
    if not isinstance(out, str):
        raise ValueError(f"should return str, returns {type(out).__name__}")
    if len(out) != 6:
        raise ValueError(f"with length=6 it should return 6 characters, returns {len(out)}")


# ----------------------------------------------------------------- module 01


@probe("transformer_flops_per_token")
def _(impl: Any) -> None:
    value = impl(
        n_layers=2, d_model=64, d_ff=128, context_length=32, vocab_size=100
    )
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"should return a positive integer, returns {value!r}")


@probe("estimate_tokens_per_second")
def _(impl: Any) -> None:
    value = impl(10.0, 1_000_000, mfu=0.5)
    if not (value > 0):
        raise ValueError(f"should return a positive number, returns {value!r}")


# ----------------------------------------------------------------- module 03


@probe("get_stats")
def _(impl: Any) -> None:
    if impl([97, 97, 97, 98]) != {(97, 97): 2, (97, 98): 1}:
        raise ValueError("it does not count overlapping pairs correctly")


@probe("merge")
def _(impl: Any) -> None:
    if list(impl([1, 1, 1], (1, 1), 256)) != [256, 1]:
        raise ValueError("merging must not overlap: [1,1,1] -> [256,1]")


@probe("train_bpe")
def _(impl: Any) -> None:
    merges, vocab = impl("aaabdaaabac", 258)
    if len(merges) != 2 or vocab.get(256) != b"aa":
        raise ValueError(f"unexpected merges or vocab: {merges}")


@probe("bpe_encode")
def _(impl: Any) -> None:
    ids = impl("aaabdaaabac", {(97, 97): 256, (256, 97): 257})
    if not isinstance(ids, list) or ids[0] != 257:
        raise ValueError(f"unexpected encoding: {ids}")


@probe("bpe_decode")
def _(impl: Any) -> None:
    vocab = {i: bytes([i]) for i in range(256)}
    if impl([104, 101, 108, 108, 111], vocab) != "hello":
        raise ValueError("it does not decode ASCII bytes correctly")


# ----------------------------------------------------------------- module 04


@probe("pack_tokens_uint16")
def _(impl: Any) -> None:
    import numpy as np

    out = impl([0, 1, 4095], 4096)
    if getattr(out, "dtype", None) != np.uint16:
        raise ValueError(f"should return dtype uint16, returns {getattr(out, 'dtype', '?')}")


@probe("train_val_split")
def _(impl: Any) -> None:
    import numpy as np

    train, val = impl(np.arange(1000, dtype=np.uint16), 0.1)
    if len(train) != 900 or len(val) != 100:
        raise ValueError(f"unexpected sizes: train={len(train)}, val={len(val)}")


@probe("get_batch")
def _(impl: Any) -> None:
    import numpy as np
    import torch

    x, y = impl(np.arange(1000, dtype=np.uint16), 2, 8, rng=np.random.default_rng(0))
    expect_shape(x, (2, 8), "x")
    expect_shape(y, (2, 8), "y")
    if x.dtype != torch.int64:
        raise ValueError(f"x should be int64, it is {x.dtype}")
    if not torch.equal(x[:, 1:], y[:, :-1]):
        raise ValueError("y has to be x shifted by one token")


# ----------------------------------------------------------------- module 05


@probe("uniform_baseline_loss")
def _(impl: Any) -> None:
    import math

    if abs(impl(4096) - math.log(4096)) > 1e-6:
        raise ValueError(f"ln(4096) should be {math.log(4096):.4f}, returns {impl(4096)}")


@probe("bigram_counts")
def _(impl: Any) -> None:
    counts = impl([0, 1, 0, 1, 2], 3)
    if tuple(counts.shape) != (3, 3):
        raise ValueError(f"shape {tuple(counts.shape)}, (3, 3) was expected")
    if int(counts[0][1]) != 2:
        raise ValueError("repeated counts must accumulate (accumulate=True)")


@probe("bigram_nll")
def _(impl: Any) -> None:
    import math

    import llmfs.reference as _ref

    value = impl(_ref.bigram_counts([0, 1, 0, 1], 2), [0, 1, 0, 1])
    if not math.isfinite(float(value)):
        raise ValueError("the loss has to be finite")


@probe("NeuralBigram")
def _(impl: Any) -> None:
    import torch

    model = impl(7)
    logits, loss = model(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long))
    expect_shape(logits, (2, 3, 7), "the logits")
    if loss is None:
        raise ValueError("with targets it has to return a loss")


@probe("BengioMLP")
def _(impl: Any) -> None:
    import torch

    model = impl(7, 4, d_embed=8, n_hidden=16)
    logits, _ = model(torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, dtype=torch.long))
    expect_shape(logits, (2, 7), "the logits")


# ----------------------------------------------------------------- module 06


@probe("causal_mask")
def _(impl: Any) -> None:
    import torch

    m = impl(4)
    if m.dtype != torch.bool:
        raise ValueError(f"it should be boolean, it is {m.dtype}")
    if bool(m.triu(diagonal=1).any()):
        raise ValueError("there are positions looking into the future: the mask is inverted")
    if not bool(m.diagonal().all()):
        raise ValueError("the diagonal should be allowed")


@probe("single_head_attention")
def _(impl: Any) -> None:
    import torch

    q = k = v = torch.randn(2, 5, 8)
    out, weights = impl(q, k, v)
    expect_shape(out, (2, 5, 8), "the output")
    expect_shape(weights, (2, 5, 5), "the weights")
    if not torch.allclose(weights.sum(-1), torch.ones(2, 5), atol=1e-5):
        raise ValueError("each row of weights must sum to 1 (check the softmax dim)")


@probe("MultiHeadAttention")
def _(impl: Any) -> None:
    import torch

    mha = impl(32, 4)
    out = mha(torch.randn(2, 6, 32))
    if isinstance(out, tuple):
        out = out[0]
    expect_shape(out, (2, 6, 32), "the output")


# ----------------------------------------------------------------- module 07


@probe("layer_norm")
def _(impl: Any) -> None:
    import torch
    import torch.nn.functional as F

    x = torch.randn(2, 4, 16)
    if not torch.allclose(impl(x), F.layer_norm(x, (16,)), atol=1e-4):
        raise ValueError("does not match F.layer_norm (check unbiased=False)")


@probe("RMSNorm")
def _(impl: Any) -> None:
    import torch

    norm = impl(16)
    out = norm(torch.randn(2, 4, 16))
    expect_shape(out, (2, 4, 16), "the output")
    if not torch.allclose(norm.weight, torch.ones(16)):
        raise ValueError("weight has to start at ones")


@probe("prenorm_residual")
def _(impl: Any) -> None:
    import torch

    x = torch.randn(2, 4, 8)
    out = impl(x, lambda z: torch.zeros_like(z), lambda z: z)
    if not torch.allclose(out, x, atol=1e-6):
        raise ValueError("with a null block the input has to pass through untouched")


# ----------------------------------------------------------------- module 08


@probe("gelu")
def _(impl: Any) -> None:
    import torch
    import torch.nn.functional as F

    x = torch.linspace(-3, 3, 50)
    if not torch.allclose(impl(x), F.gelu(x, approximate="tanh"), atol=1e-5):
        raise ValueError("does not match F.gelu(approximate='tanh')")


@probe("swiglu_hidden_dim")
def _(impl: Any) -> None:
    if impl(320) != 896:
        raise ValueError(f"swiglu_hidden_dim(320) should give 896, gives {impl(320)}")


@probe("SwiGLU")
def _(impl: Any) -> None:
    import torch

    ffn = impl(32, 64)
    expect_shape(ffn(torch.randn(2, 5, 32)), (2, 5, 32), "the output")


# ----------------------------------------------------------------- module 09


@probe("sinusoidal_embeddings")
def _(impl: Any) -> None:
    import torch

    table = impl(32, 16)
    expect_shape(table, (32, 16), "the table")
    if not torch.allclose(table[0, 1::2], torch.ones(8), atol=1e-5):
        raise ValueError("at position 0 the odd dimensions (cosine) should be 1")


@probe("rope_frequencies")
def _(impl: Any) -> None:
    import torch

    cos, sin = impl(16, 32)
    expect_shape(cos, (32, 16), "the cosine table")
    if not torch.allclose(cos[:, :8], cos[:, 8:], atol=1e-6):
        raise ValueError("the frequencies have to be duplicated by halves")


@probe("apply_rope")
def _(impl: Any) -> None:
    import torch

    import llmfs.reference as _ref

    cos, sin = _ref.rope_frequencies(16, 32)
    x = torch.randn(1, 2, 8, 16)
    out = impl(x, cos, sin)
    expect_shape(out, (1, 2, 8, 16), "the output")
    if not torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-4):
        raise ValueError("rotating cannot change the vector's norm")


# ----------------------------------------------------------------- module 10


@probe("expected_param_count")
def _(impl: Any) -> None:
    from llmfs.config import ModelConfig

    value = impl(ModelConfig())
    if value != 8_933_440:
        raise ValueError(f"the final model has 8,933,440 parameters, your formula gives {value:,}")


@probe("count_parameters")
def _(impl: Any) -> None:
    from llmfs.config import ModelConfig
    from llmfs.reference import GPT as _GPT

    cfg = ModelConfig(vocab_size=32, n_layers=1, d_model=16, n_heads=2, d_ff=32, context_length=8)
    model = _GPT(cfg)
    counts = impl(model)
    if counts.get("total") != sum(p.numel() for p in model.parameters()):
        raise ValueError("the total does not match sum(p.numel() for p in parameters())")


@probe("TransformerBlock")
def _(impl: Any) -> None:
    import torch

    from llmfs.config import ModelConfig

    cfg = ModelConfig(vocab_size=32, n_layers=1, d_model=16, n_heads=2, d_ff=32, context_length=8)
    block = impl(cfg)
    expect_shape(block(torch.randn(2, 4, 16)), (2, 4, 16), "the block's output")


@probe("GPT")
def _(impl: Any) -> None:
    import torch

    from llmfs.config import ModelConfig

    cfg = ModelConfig(vocab_size=32, n_layers=1, d_model=16, n_heads=2, d_ff=32, context_length=8)
    model = impl(cfg)
    logits, loss = model(torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long))
    expect_shape(logits, (2, 4, 32), "the logits")
    if loss is None:
        raise ValueError("with targets it has to return a loss")


# --------------------------------------------------------------- modules 11, 12 and 13


@probe("AdamWScratch")
def _(impl: Any) -> None:
    import torch
    import torch.nn as nn

    p = nn.Parameter(torch.zeros(10))
    opt = impl([p], lr=0.1, betas=(0.9, 0.95))
    p.grad = torch.ones(10)
    opt.step()
    jump = float(p.detach().abs().mean())
    if abs(jump - 0.1) > 0.02:
        raise ValueError(
            f"the first step moves {jump:.4f} and it should move ~0.1 (the bias correction "
            "is missing)"
        )


@probe("lr_at_step")
def _(impl: Any) -> None:
    if abs(impl(500, 10000, 1e-3, 500, 0.1) - 1e-3) > 1e-9:
        raise ValueError("right after the warmup ends the lr should be at its maximum")
    if abs(impl(10000, 10000, 1e-3, 500, 0.1) - 1e-4) > 1e-9:
        raise ValueError("at the end the lr should be at the floor (10%)")


@probe("clip_grad_norm")
def _(impl: Any) -> None:
    import torch
    import torch.nn as nn

    p = nn.Parameter(torch.ones(4))
    p.grad = torch.full((4,), 3.0)  # norm 6
    norm = impl([p], 1.0)
    if abs(float(norm) - 6.0) > 1e-4:
        raise ValueError(f"should return the norm BEFORE clipping (6.0), returns {norm}")


@probe("build_param_groups")
def _(impl: Any) -> None:
    import torch.nn as nn

    groups = impl(nn.Linear(4, 4), 0.1)
    if len(groups) != 2:
        raise ValueError(f"should return 2 groups, returns {len(groups)}")
    if groups[1].get("weight_decay") != 0.0:
        raise ValueError("the second group has to go without weight decay")


@probe("model_flops_per_token")
def _(impl: Any) -> None:
    from llmfs.config import ModelConfig

    f = impl(ModelConfig())
    if f.get("total") != 65_372_160:
        raise ValueError(f"with the final config the total should be 65,372,160, gives {f.get('total')}")


@probe("compute_mfu")
def _(impl: Any) -> None:
    if abs(impl(1e6, 1e7, 10.0) - 1.0) > 1e-6:
        raise ValueError("the maximum MFU should give 1.0")


@probe("chinchilla_optimal_allocation")
def _(impl: Any) -> None:
    a = impl(5.76e23)
    if not 6.5e10 < a["params"] < 7.5e10:
        raise ValueError(
            f"with Chinchilla's budget it should give ~7e10 parameters, gives {a['params']:.2e}"
        )


@probe("overfit_single_batch")
def _(impl: Any) -> None:
    import torch

    from llmfs.config import ModelConfig
    from llmfs.reference import GPT as _GPT

    cfg = ModelConfig(vocab_size=32, n_layers=1, d_model=16, n_heads=2, d_ff=32, context_length=8)
    seq = torch.randint(0, 32, (2, 5))
    hist = impl(_GPT(cfg), seq[:, :-1], seq[:, 1:], steps=5, lr=1e-3)
    if len(hist) != 5:
        raise ValueError(f"should return 5 losses, returns {len(hist)}")


@probe("format_eta")
def _(impl: Any) -> None:
    if impl(3725) != "1h 2m":
        raise ValueError(f"format_eta(3725) should give '1h 2m', gives {impl(3725)!r}")
    if impl(-1) != "?":
        raise ValueError("negative values should give '?'")


# ----------------------------------------------------------- modules 14, 15, 16 and 17


@probe("apply_repetition_penalty")
def _(impl: Any) -> None:
    import torch

    out = impl(torch.tensor([[-3.0, 2.0]]), torch.tensor([[0]]), 2.0)
    if abs(float(out[0, 0]) - (-6.0)) > 1e-4:
        raise ValueError("a negative logit has to be MULTIPLIED by the penalty")


@probe("top_k_filter")
def _(impl: Any) -> None:
    import torch

    out = impl(torch.tensor([[3.0, 2.0, 1.0]]), 2)
    if int(torch.isfinite(out).sum()) != 2:
        raise ValueError("with k=2 exactly 2 logits have to survive")


@probe("top_p_filter")
def _(impl: Any) -> None:
    import torch

    out = impl(torch.log(torch.tensor([[0.9, 0.05, 0.05]])), 0.5)
    if not torch.isfinite(out[0, 0]):
        raise ValueError("the most likely token can never be filtered out")


@probe("KVCache")
def _(impl: Any) -> None:
    import torch

    cache = impl(2)
    if cache.seq_len != 0:
        raise ValueError("the cache has to start empty")
    K, _ = cache.update(0, torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
    if K.shape[-2] != 3:
        raise ValueError(f"unexpected shape after update: {tuple(K.shape)}")


@probe("generate_with_cache")
def _(impl: Any) -> None:
    import torch

    from llmfs.config import ModelConfig
    from llmfs.reference import GPT as _GPT

    cfg = ModelConfig(vocab_size=32, n_layers=1, d_model=16, n_heads=2, d_ff=32, context_length=32)
    model = _GPT(cfg).eval()
    prompt = torch.zeros(1, 3, dtype=torch.long)
    out = impl(model, prompt, 5, temperature=0.0)
    if out.shape[1] != 8:
        raise ValueError(f"should return 8 tokens, returns {out.shape[1]}")


@probe("perplexity_from_loss")
def _(impl: Any) -> None:
    import math

    if abs(impl(math.log(4096)) - 4096) > 1:
        raise ValueError("with a loss of ln(V) the perplexity has to be V")


@probe("bits_per_byte")
def _(impl: Any) -> None:
    import math

    if abs(impl(math.log(2), 1, 1) - 1.0) > 1e-6:
        raise ValueError("ln(2) nats over 1 byte has to give 1.0 bits/byte")


@probe("run_prompt_battery")
def _(impl: Any) -> None:
    results = impl(lambda p: p + "!")
    if not results or set(results[0]) != {"prompt", "tests", "completion"}:
        raise ValueError("each result needs the keys prompt, tests and completion")


@probe("build_chat_template")
def _(impl: Any) -> None:
    out = impl([{"role": "user", "content": "Hello"}])
    if out != "<|user|>Hello<|end|>":
        raise ValueError(f"unexpected format: {out!r}")


@probe("mask_prompt_tokens")
def _(impl: Any) -> None:
    if impl([10, 11, 12, 20, 21, 22], 3) != [-100, -100, 20, 21, 22, -100]:
        raise ValueError("check the off-by-one: it is TWO ignored positions, not three")


@probe("LoRALinear")
def _(impl: Any) -> None:
    import torch
    import torch.nn as nn

    base = nn.Linear(32, 32)
    lora = impl(base, r=4)
    x = torch.randn(2, 32)
    if not torch.allclose(lora(x), base(x), atol=1e-5):
        raise ValueError("at initialization the output has to be identical to the base (lora_B=0)")


@probe("merge_lora_weights")
def _(impl: Any) -> None:
    import torch
    import torch.nn as nn

    from llmfs.reference import LoRALinear as _LoRA

    lora = _LoRA(nn.Linear(32, 32), r=4)
    merged = impl(lora)
    if not isinstance(merged, nn.Linear):
        raise ValueError("it has to return a plain nn.Linear")


@probe("quantize_int8_symmetric")
def _(impl: Any) -> None:
    import torch

    q, _ = impl(torch.randn(4, 8))
    if q.dtype != torch.int8:
        raise ValueError(f"the dtype has to be int8, it is {q.dtype}")


@probe("dequantize_int8")
def _(impl: Any) -> None:
    import torch

    from llmfs.reference import quantize_int8_symmetric as _q

    w = torch.randn(4, 8)
    q, scale = _q(w)
    if not torch.allclose(impl(q, scale), w, atol=0.05):
        raise ValueError("the roundtrip should come close to the original")


@probe("quantization_error")
def _(impl: Any) -> None:
    import torch

    e = impl(torch.randn(4, 8))
    if e.get("compression") != 4.0:
        raise ValueError(f"from fp32 to int8 the compression is 4x, gives {e.get('compression')}")
