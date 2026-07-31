"""Smoke tests que decide el bridge para saber si tu implementacion es usable.

Un probe NO comprueba que tu codigo sea correcto: comprueba que se puede llamar y que
devuelve algo con la forma esperada. La correccion la verifican los tests del modulo
comparando contra la referencia con `torch.allclose`.

Contrato de un probe:

    @probe("nombre_del_simbolo")
    def _(impl):
        ...   # llama a impl con datos minusculos
        ...   # lanza cualquier excepcion si no es usable

Si un simbolo no tiene probe registrado, el bridge solo comprueba que exista. Eso basta
para funciones cuya version sin implementar lanza `NotImplementedError` nada mas entrar,
que es como estan escritos todos los `ejercicios.py`.

Los probes se registran por fase, segun se van escribiendo los modulos. `tests/` incluye
un test que verifica que no queda ningun ejercicio con test propio sin su pieza de
referencia correspondiente.
"""

from __future__ import annotations

from typing import Any, Callable

PROBES: dict[str, Callable[[Any], None]] = {}


def probe(name: str) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
    """Registra el smoke test de un simbolo del curriculo."""

    def decorator(fn: Callable[[Any], None]) -> Callable[[Any], None]:
        if name in PROBES:
            raise RuntimeError(f"Probe duplicado para {name!r}")
        PROBES[name] = fn
        return fn

    return decorator


def expect_shape(tensor: Any, shape: tuple[int, ...], what: str = "salida") -> None:
    """Ayuda para los probes: falla con un mensaje legible si la forma no cuadra."""
    actual = tuple(getattr(tensor, "shape", ()))
    if actual != shape:
        raise ValueError(f"{what} con forma {actual}, se esperaba {shape}")


# ---------------------------------------------------------------------------- modulo 00


@probe("next_token_probs")
def _(impl: Any) -> None:
    probs = impl({"a": 3, "b": 1})
    if set(probs) != {"a", "b"}:
        raise ValueError(f"deberia devolver las claves 'a' y 'b', devuelve {set(probs)}")
    if abs(sum(probs.values()) - 1.0) > 1e-9:
        raise ValueError(f"las probabilidades suman {sum(probs.values())}, deberian sumar 1")


@probe("sample_next_token")
def _(impl: Any) -> None:
    import random

    elegido = impl({"a": 0.5, "b": 0.5}, random.Random(0))
    if elegido not in {"a", "b"}:
        raise ValueError(f"ha devuelto {elegido!r}, que no es una clave de la distribucion")


@probe("generate_naive")
def _(impl: Any) -> None:
    import random

    salida = impl({"a": {"b": 1}, "b": {"a": 1}}, "a", 6, random.Random(0))
    if not isinstance(salida, str):
        raise ValueError(f"deberia devolver str, devuelve {type(salida).__name__}")
    if len(salida) != 6:
        raise ValueError(f"con length=6 deberia devolver 6 caracteres, devuelve {len(salida)}")


# ---------------------------------------------------------------------------- modulo 01


@probe("transformer_flops_per_token")
def _(impl: Any) -> None:
    valor = impl(
        n_layers=2, d_model=64, d_ff=128, context_length=32, vocab_size=100
    )
    if not isinstance(valor, int) or valor <= 0:
        raise ValueError(f"deberia devolver un entero positivo, devuelve {valor!r}")


@probe("estimate_tokens_per_second")
def _(impl: Any) -> None:
    valor = impl(10.0, 1_000_000, mfu=0.5)
    if not (valor > 0):
        raise ValueError(f"deberia devolver un numero positivo, devuelve {valor!r}")


# ---------------------------------------------------------------------------- modulo 03


@probe("get_stats")
def _(impl: Any) -> None:
    if impl([97, 97, 97, 98]) != {(97, 97): 2, (97, 98): 1}:
        raise ValueError("no cuenta bien los pares solapados")


@probe("merge")
def _(impl: Any) -> None:
    if list(impl([1, 1, 1], (1, 1), 256)) != [256, 1]:
        raise ValueError("al fusionar no debe solapar: [1,1,1] -> [256,1]")


@probe("train_bpe")
def _(impl: Any) -> None:
    merges, vocab = impl("aaabdaaabac", 258)
    if len(merges) != 2 or vocab.get(256) != b"aa":
        raise ValueError(f"merges o vocab inesperados: {merges}")


@probe("bpe_encode")
def _(impl: Any) -> None:
    ids = impl("aaabdaaabac", {(97, 97): 256, (256, 97): 257})
    if not isinstance(ids, list) or ids[0] != 257:
        raise ValueError(f"codificacion inesperada: {ids}")


@probe("bpe_decode")
def _(impl: Any) -> None:
    vocab = {i: bytes([i]) for i in range(256)}
    if impl([104, 111, 108, 97], vocab) != "hola":
        raise ValueError("no decodifica los bytes ASCII correctamente")


# ---------------------------------------------------------------------------- modulo 04


@probe("pack_tokens_uint16")
def _(impl: Any) -> None:
    import numpy as np

    salida = impl([0, 1, 4095], 4096)
    if getattr(salida, "dtype", None) != np.uint16:
        raise ValueError(f"deberia devolver dtype uint16, devuelve {getattr(salida, 'dtype', '?')}")


@probe("train_val_split")
def _(impl: Any) -> None:
    import numpy as np

    train, val = impl(np.arange(1000, dtype=np.uint16), 0.1)
    if len(train) != 900 or len(val) != 100:
        raise ValueError(f"tamanyos inesperados: train={len(train)}, val={len(val)}")


@probe("get_batch")
def _(impl: Any) -> None:
    import numpy as np
    import torch

    x, y = impl(np.arange(1000, dtype=np.uint16), 2, 8, rng=np.random.default_rng(0))
    expect_shape(x, (2, 8), "x")
    expect_shape(y, (2, 8), "y")
    if x.dtype != torch.int64:
        raise ValueError(f"x deberia ser int64, es {x.dtype}")
    if not torch.equal(x[:, 1:], y[:, :-1]):
        raise ValueError("y tiene que ser x desplazado un token")


# ---------------------------------------------------------------------------- modulo 05


@probe("uniform_baseline_loss")
def _(impl: Any) -> None:
    import math

    if abs(impl(4096) - math.log(4096)) > 1e-6:
        raise ValueError(f"ln(4096) deberia ser {math.log(4096):.4f}, devuelve {impl(4096)}")


@probe("bigram_counts")
def _(impl: Any) -> None:
    counts = impl([0, 1, 0, 1, 2], 3)
    if tuple(counts.shape) != (3, 3):
        raise ValueError(f"forma {tuple(counts.shape)}, se esperaba (3, 3)")
    if int(counts[0][1]) != 2:
        raise ValueError("los conteos repetidos deben acumularse (accumulate=True)")


@probe("bigram_nll")
def _(impl: Any) -> None:
    import math

    import llmfs.reference as _ref

    valor = impl(_ref.bigram_counts([0, 1, 0, 1], 2), [0, 1, 0, 1])
    if not math.isfinite(float(valor)):
        raise ValueError("la perdida tiene que ser finita")


@probe("NeuralBigram")
def _(impl: Any) -> None:
    import torch

    modelo = impl(7)
    logits, perdida = modelo(torch.zeros(2, 3, dtype=torch.long), torch.zeros(2, 3, dtype=torch.long))
    expect_shape(logits, (2, 3, 7), "los logits")
    if perdida is None:
        raise ValueError("con targets tiene que devolver perdida")


@probe("BengioMLP")
def _(impl: Any) -> None:
    import torch

    modelo = impl(7, 4, d_embed=8, n_hidden=16)
    logits, _ = modelo(torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, dtype=torch.long))
    expect_shape(logits, (2, 7), "los logits")


# ---------------------------------------------------------------------------- modulo 06


@probe("causal_mask")
def _(impl: Any) -> None:
    import torch

    m = impl(4)
    if m.dtype != torch.bool:
        raise ValueError(f"deberia ser booleana, es {m.dtype}")
    if bool(m.triu(diagonal=1).any()):
        raise ValueError("hay posiciones que miran al futuro: la mascara esta al reves")
    if not bool(m.diagonal().all()):
        raise ValueError("la diagonal deberia estar permitida")


@probe("single_head_attention")
def _(impl: Any) -> None:
    import torch

    q = k = v = torch.randn(2, 5, 8)
    salida, pesos = impl(q, k, v)
    expect_shape(salida, (2, 5, 8), "la salida")
    expect_shape(pesos, (2, 5, 5), "los pesos")
    if not torch.allclose(pesos.sum(-1), torch.ones(2, 5), atol=1e-5):
        raise ValueError("cada fila de pesos debe sumar 1 (revisa el dim del softmax)")


@probe("MultiHeadAttention")
def _(impl: Any) -> None:
    import torch

    mha = impl(32, 4)
    salida = mha(torch.randn(2, 6, 32))
    if isinstance(salida, tuple):
        salida = salida[0]
    expect_shape(salida, (2, 6, 32), "la salida")


# ---------------------------------------------------------------------------- modulo 07


@probe("layer_norm")
def _(impl: Any) -> None:
    import torch
    import torch.nn.functional as F

    x = torch.randn(2, 4, 16)
    if not torch.allclose(impl(x), F.layer_norm(x, (16,)), atol=1e-4):
        raise ValueError("no coincide con F.layer_norm (revisa unbiased=False)")


@probe("RMSNorm")
def _(impl: Any) -> None:
    import torch

    norm = impl(16)
    salida = norm(torch.randn(2, 4, 16))
    expect_shape(salida, (2, 4, 16), "la salida")
    if not torch.allclose(norm.weight, torch.ones(16)):
        raise ValueError("weight tiene que arrancar a unos")


@probe("prenorm_residual")
def _(impl: Any) -> None:
    import torch

    x = torch.randn(2, 4, 8)
    salida = impl(x, lambda z: torch.zeros_like(z), lambda z: z)
    if not torch.allclose(salida, x, atol=1e-6):
        raise ValueError("con un bloque nulo la entrada tiene que pasar intacta")


# ---------------------------------------------------------------------------- modulo 08


@probe("gelu")
def _(impl: Any) -> None:
    import torch
    import torch.nn.functional as F

    x = torch.linspace(-3, 3, 50)
    if not torch.allclose(impl(x), F.gelu(x, approximate="tanh"), atol=1e-5):
        raise ValueError("no coincide con F.gelu(approximate='tanh')")


@probe("swiglu_hidden_dim")
def _(impl: Any) -> None:
    if impl(320) != 896:
        raise ValueError(f"swiglu_hidden_dim(320) deberia dar 896, da {impl(320)}")


@probe("SwiGLU")
def _(impl: Any) -> None:
    import torch

    ffn = impl(32, 64)
    expect_shape(ffn(torch.randn(2, 5, 32)), (2, 5, 32), "la salida")


# ---------------------------------------------------------------------------- modulo 09


@probe("sinusoidal_embeddings")
def _(impl: Any) -> None:
    import torch

    tabla = impl(32, 16)
    expect_shape(tabla, (32, 16), "la tabla")
    if not torch.allclose(tabla[0, 1::2], torch.ones(8), atol=1e-5):
        raise ValueError("en la posicion 0 las dimensiones impares (coseno) deberian valer 1")


@probe("rope_frequencies")
def _(impl: Any) -> None:
    import torch

    cos, sin = impl(16, 32)
    expect_shape(cos, (32, 16), "la tabla de cosenos")
    if not torch.allclose(cos[:, :8], cos[:, 8:], atol=1e-6):
        raise ValueError("las frecuencias tienen que estar duplicadas por mitades")


@probe("apply_rope")
def _(impl: Any) -> None:
    import torch

    import llmfs.reference as _ref

    cos, sin = _ref.rope_frequencies(16, 32)
    x = torch.randn(1, 2, 8, 16)
    salida = impl(x, cos, sin)
    expect_shape(salida, (1, 2, 8, 16), "la salida")
    if not torch.allclose(salida.norm(dim=-1), x.norm(dim=-1), atol=1e-4):
        raise ValueError("rotar no puede cambiar la norma del vector")


# ---------------------------------------------------------------------------- modulo 10


@probe("expected_param_count")
def _(impl: Any) -> None:
    from llmfs.config import ModelConfig

    valor = impl(ModelConfig())
    if valor != 8_933_440:
        raise ValueError(f"el modelo final tiene 8.933.440 parametros, tu formula da {valor:,}")


@probe("count_parameters")
def _(impl: Any) -> None:
    from llmfs.config import ModelConfig
    from llmfs.reference import GPT as _GPT

    cfg = ModelConfig(vocab_size=32, n_layers=1, d_model=16, n_heads=2, d_ff=32, context_length=8)
    modelo = _GPT(cfg)
    conteo = impl(modelo)
    if conteo.get("total") != sum(p.numel() for p in modelo.parameters()):
        raise ValueError("el total no coincide con sum(p.numel() for p in parameters())")


@probe("TransformerBlock")
def _(impl: Any) -> None:
    import torch

    from llmfs.config import ModelConfig

    cfg = ModelConfig(vocab_size=32, n_layers=1, d_model=16, n_heads=2, d_ff=32, context_length=8)
    bloque = impl(cfg)
    expect_shape(bloque(torch.randn(2, 4, 16)), (2, 4, 16), "la salida del bloque")


@probe("GPT")
def _(impl: Any) -> None:
    import torch

    from llmfs.config import ModelConfig

    cfg = ModelConfig(vocab_size=32, n_layers=1, d_model=16, n_heads=2, d_ff=32, context_length=8)
    modelo = impl(cfg)
    logits, perdida = modelo(torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long))
    expect_shape(logits, (2, 4, 32), "los logits")
    if perdida is None:
        raise ValueError("con targets tiene que devolver perdida")


# ------------------------------------------------------------------- modulos 11, 12 y 13


@probe("AdamWScratch")
def _(impl: Any) -> None:
    import torch
    import torch.nn as nn

    p = nn.Parameter(torch.zeros(10))
    opt = impl([p], lr=0.1, betas=(0.9, 0.95))
    p.grad = torch.ones(10)
    opt.step()
    salto = float(p.detach().abs().mean())
    if abs(salto - 0.1) > 0.02:
        raise ValueError(
            f"el primer paso mueve {salto:.4f} y deberia mover ~0.1 (falta la correccion "
            "de sesgo)"
        )


@probe("lr_at_step")
def _(impl: Any) -> None:
    if abs(impl(500, 10000, 1e-3, 500, 0.1) - 1e-3) > 1e-9:
        raise ValueError("justo al acabar el warmup el lr deberia estar en su maximo")
    if abs(impl(10000, 10000, 1e-3, 500, 0.1) - 1e-4) > 1e-9:
        raise ValueError("al final el lr deberia estar en el suelo (10%)")


@probe("clip_grad_norm")
def _(impl: Any) -> None:
    import torch
    import torch.nn as nn

    p = nn.Parameter(torch.ones(4))
    p.grad = torch.full((4,), 3.0)  # norma 6
    norma = impl([p], 1.0)
    if abs(float(norma) - 6.0) > 1e-4:
        raise ValueError(f"deberia devolver la norma ANTES de recortar (6.0), devuelve {norma}")


@probe("build_param_groups")
def _(impl: Any) -> None:
    import torch.nn as nn

    grupos = impl(nn.Linear(4, 4), 0.1)
    if len(grupos) != 2:
        raise ValueError(f"deberia devolver 2 grupos, devuelve {len(grupos)}")
    if grupos[1].get("weight_decay") != 0.0:
        raise ValueError("el segundo grupo tiene que ir sin weight decay")


@probe("model_flops_per_token")
def _(impl: Any) -> None:
    from llmfs.config import ModelConfig

    f = impl(ModelConfig())
    if f.get("total") != 65_372_160:
        raise ValueError(f"con la config final el total deberia ser 65.372.160, da {f.get('total')}")


@probe("compute_mfu")
def _(impl: Any) -> None:
    if abs(impl(1e6, 1e7, 10.0) - 1.0) > 1e-6:
        raise ValueError("la MFU maxima deberia dar 1.0")


@probe("chinchilla_optimal_allocation")
def _(impl: Any) -> None:
    a = impl(5.76e23)
    if not 6.5e10 < a["params"] < 7.5e10:
        raise ValueError(
            f"con el presupuesto de Chinchilla deberia dar ~7e10 parametros, da {a['params']:.2e}"
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
        raise ValueError(f"deberia devolver 5 perdidas, devuelve {len(hist)}")


@probe("format_eta")
def _(impl: Any) -> None:
    if impl(3725) != "1h 2m":
        raise ValueError(f"format_eta(3725) deberia dar '1h 2m', da {impl(3725)!r}")
    if impl(-1) != "?":
        raise ValueError("los valores negativos deberian dar '?'")


# ------------------------------------------------------------- modulos 14, 15, 16 y 17


@probe("apply_repetition_penalty")
def _(impl: Any) -> None:
    import torch

    salida = impl(torch.tensor([[-3.0, 2.0]]), torch.tensor([[0]]), 2.0)
    if abs(float(salida[0, 0]) - (-6.0)) > 1e-4:
        raise ValueError("un logit negativo tiene que MULTIPLICARSE por la penalizacion")


@probe("top_k_filter")
def _(impl: Any) -> None:
    import torch

    salida = impl(torch.tensor([[3.0, 2.0, 1.0]]), 2)
    if int(torch.isfinite(salida).sum()) != 2:
        raise ValueError("con k=2 tienen que sobrevivir exactamente 2 logits")


@probe("top_p_filter")
def _(impl: Any) -> None:
    import torch

    salida = impl(torch.log(torch.tensor([[0.9, 0.05, 0.05]])), 0.5)
    if not torch.isfinite(salida[0, 0]):
        raise ValueError("el token mas probable no puede filtrarse nunca")


@probe("KVCache")
def _(impl: Any) -> None:
    import torch

    cache = impl(2)
    if cache.seq_len != 0:
        raise ValueError("la cache tiene que arrancar vacia")
    K, _ = cache.update(0, torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
    if K.shape[-2] != 3:
        raise ValueError(f"forma inesperada tras update: {tuple(K.shape)}")


@probe("generate_with_cache")
def _(impl: Any) -> None:
    import torch

    from llmfs.config import ModelConfig
    from llmfs.reference import GPT as _GPT

    cfg = ModelConfig(vocab_size=32, n_layers=1, d_model=16, n_heads=2, d_ff=32, context_length=32)
    modelo = _GPT(cfg).eval()
    prompt = torch.zeros(1, 3, dtype=torch.long)
    salida = impl(modelo, prompt, 5, temperature=0.0)
    if salida.shape[1] != 8:
        raise ValueError(f"deberia devolver 8 tokens, devuelve {salida.shape[1]}")


@probe("perplexity_from_loss")
def _(impl: Any) -> None:
    import math

    if abs(impl(math.log(4096)) - 4096) > 1:
        raise ValueError("con perdida ln(V) la perplejidad tiene que ser V")


@probe("bits_per_byte")
def _(impl: Any) -> None:
    import math

    if abs(impl(math.log(2), 1, 1) - 1.0) > 1e-6:
        raise ValueError("ln(2) nats sobre 1 byte tiene que dar 1.0 bits/byte")


@probe("run_prompt_battery")
def _(impl: Any) -> None:
    resultados = impl(lambda p: p + "!")
    if not resultados or set(resultados[0]) != {"prompt", "que_prueba", "completion"}:
        raise ValueError("cada resultado necesita las claves prompt, que_prueba y completion")


@probe("build_chat_template")
def _(impl: Any) -> None:
    salida = impl([{"role": "user", "content": "Hola"}])
    if salida != "<|user|>Hola<|end|>":
        raise ValueError(f"formato inesperado: {salida!r}")


@probe("mask_prompt_tokens")
def _(impl: Any) -> None:
    if impl([10, 11, 12, 20, 21, 22], 3) != [-100, -100, 20, 21, 22, -100]:
        raise ValueError("revisa el off-by-one: son DOS posiciones ignoradas, no tres")


@probe("LoRALinear")
def _(impl: Any) -> None:
    import torch
    import torch.nn as nn

    base = nn.Linear(32, 32)
    lora = impl(base, r=4)
    x = torch.randn(2, 32)
    if not torch.allclose(lora(x), base(x), atol=1e-5):
        raise ValueError("al inicializar, la salida tiene que ser identica a la base (lora_B=0)")


@probe("merge_lora_weights")
def _(impl: Any) -> None:
    import torch
    import torch.nn as nn

    from llmfs.reference import LoRALinear as _LoRA

    lora = _LoRA(nn.Linear(32, 32), r=4)
    fundida = impl(lora)
    if not isinstance(fundida, nn.Linear):
        raise ValueError("tiene que devolver un nn.Linear normal")


@probe("quantize_int8_symmetric")
def _(impl: Any) -> None:
    import torch

    q, _ = impl(torch.randn(4, 8))
    if q.dtype != torch.int8:
        raise ValueError(f"el dtype tiene que ser int8, es {q.dtype}")


@probe("dequantize_int8")
def _(impl: Any) -> None:
    import torch

    from llmfs.reference import quantize_int8_symmetric as _q

    w = torch.randn(4, 8)
    q, escala = _q(w)
    if not torch.allclose(impl(q, escala), w, atol=0.05):
        raise ValueError("el roundtrip deberia acercarse al original")


@probe("quantization_error")
def _(impl: Any) -> None:
    import torch

    e = impl(torch.randn(4, 8))
    if e.get("compresion") != 4.0:
        raise ValueError(f"de fp32 a int8 la compresion es 4x, da {e.get('compresion')}")
