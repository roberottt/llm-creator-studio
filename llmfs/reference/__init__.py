"""Implementaciones de referencia: la red de seguridad del curso.

Aqui esta TODO resuelto y correcto. Dos usos:

1. **Red de seguridad.** `llmfs.bridge` tira de este paquete cuando tu ejercicio del
   modulo N todavia no esta, para que los modulos N+1 en adelante sigan funcionando.
2. **Oraculo de los tests.** Los `test_NN.py` comparan tu implementacion contra estas
   con `torch.allclose`, no contra "no ha petado".

Estilo: se prioriza la legibilidad sobre la velocidad. Son implementaciones directas de
la formula del TEORIA.md correspondiente, con las mismas variables. No busques trucos de
rendimiento aqui; los que importan estan en `llmfs/model/` y `llmfs/train/`.

Si vas a mirar este codigo antes de intentar el ejercicio, mira antes `llmfs hint NN`.
Las pistas estan escritas para desbloquearte sin quemarte el ejercicio.

Todos los simbolos del curriculo se reexportan aqui con su nombre, que es la clave que
usa el bridge. El fichero en el que vivan es un detalle de organizacion.
"""

from __future__ import annotations

# --- modulo 00: que es un LLM --------------------------------------------------------
from llmfs.reference.intro import (
    build_count_table,
    generate_naive,
    next_token_probs,
    sample_next_token,
)

# --- modulo 01: entorno y hardware ---------------------------------------------------
from llmfs.reference.hardware import (
    estimate_tokens_per_second,
    matmul_flops,
    measure_matmul_tflops,
    transformer_flops_per_token,
)

# --- modulo 01: autograd -------------------------------------------------------------
from llmfs.reference.autograd import (
    MLP,
    Layer,
    Neuron,
    Value,
    topological_order,
    train_scalar_mlp,
)

# --- modulo 02: tokenizacion ---------------------------------------------------------
from llmfs.reference.tokenizer import (
    GPT4_SPLIT_PATTERN,
    bpe_decode,
    bpe_encode,
    compression_ratio,
    get_stats,
    merge,
    train_bpe,
)

# --- modulo 03: datos ----------------------------------------------------------------
from llmfs.reference.data import get_batch, pack_tokens_uint16, train_val_split

# --- modulo 06: atencion -------------------------------------------------------------
from llmfs.reference.attention import MultiHeadAttention, causal_mask, single_head_attention

# --- modulo 07: normalizacion --------------------------------------------------------
from llmfs.reference.norm import RMSNorm, layer_norm, postnorm_residual, prenorm_residual

# --- modulo 08: mlp y activaciones ---------------------------------------------------
from llmfs.reference.mlp import MLP as FeedForwardMLP
from llmfs.reference.mlp import SwiGLU, gelu, swiglu_hidden_dim

# --- modulo 09: posicion -------------------------------------------------------------
from llmfs.reference.position import (
    LearnedPositionalEmbedding,
    apply_rope,
    rope_frequencies,
    rotate_half,
    sinusoidal_embeddings,
)

# --- modulo 10: el gpt completo ------------------------------------------------------
from llmfs.reference.gpt import (
    GPT,
    TransformerBlock,
    count_parameters,
    expected_param_count,
    make_ffn,
    make_norm,
)

# --- modulo 11: bucle de entrenamiento ------------------------------------------------
from llmfs.reference.training import (
    AdamWScratch,
    build_param_groups,
    clip_grad_norm,
    lr_at_step,
)

# --- modulo 12: eficiencia y escalado -------------------------------------------------
from llmfs.reference.scaling import (
    chinchilla_optimal_allocation,
    compute_mfu,
    model_flops_per_token,
)

# --- modulo 13: la tirada real --------------------------------------------------------
from llmfs.reference.final_run import estimate_remaining, format_eta, overfit_single_batch

# --- modulo 14: inferencia ------------------------------------------------------------
from llmfs.reference.inference import (
    KVCache,
    apply_repetition_penalty,
    generate_with_cache,
    top_k_filter,
    top_p_filter,
)

# --- modulo 15: evaluacion ------------------------------------------------------------
from llmfs.reference.evaluation import (
    PROMPTS_TINYSTORIES,
    bits_per_byte,
    evaluate_perplexity,
    perplexity_from_loss,
    run_prompt_battery,
    write_eval_report,
)

# --- modulo 16: post-training ---------------------------------------------------------
from llmfs.reference.finetune import (
    CHAT_MARKERS,
    LoRALinear,
    apply_lora_to_model,
    build_chat_template,
    count_trainable,
    mask_prompt_tokens,
    merge_lora_weights,
)

# --- modulo 17: extras ----------------------------------------------------------------
from llmfs.reference.quantization import (
    dequantize_int8,
    quantization_error,
    quantize_int8_symmetric,
)

# --- modulo 05: baselines ------------------------------------------------------------
from llmfs.reference.baselines import (
    BengioMLP,
    NeuralBigram,
    bigram_counts,
    bigram_nll,
    uniform_baseline_loss,
)

__all__: list[str] = [
    # 00
    "build_count_table",
    "next_token_probs",
    "sample_next_token",
    "generate_naive",
    # 01
    "matmul_flops",
    "measure_matmul_tflops",
    "transformer_flops_per_token",
    "estimate_tokens_per_second",
    # 01
    "Value",
    "topological_order",
    "train_scalar_mlp",
    "Neuron",
    "Layer",
    "MLP",
    # 02
    "GPT4_SPLIT_PATTERN",
    "get_stats",
    "merge",
    "train_bpe",
    "bpe_encode",
    "bpe_decode",
    "compression_ratio",
    # 03
    "pack_tokens_uint16",
    "train_val_split",
    "get_batch",
    # 04
    "uniform_baseline_loss",
    "bigram_counts",
    "bigram_nll",
    "NeuralBigram",
    "BengioMLP",
    # 06
    "causal_mask",
    "single_head_attention",
    "MultiHeadAttention",
    # 07
    "layer_norm",
    "RMSNorm",
    "prenorm_residual",
    "postnorm_residual",
    # 08
    "gelu",
    "swiglu_hidden_dim",
    "SwiGLU",
    "FeedForwardMLP",
    # 09
    "sinusoidal_embeddings",
    "rope_frequencies",
    "rotate_half",
    "apply_rope",
    "LearnedPositionalEmbedding",
    # 10
    "expected_param_count",
    "count_parameters",
    "TransformerBlock",
    "GPT",
    "make_norm",
    "make_ffn",
    # 11
    "AdamWScratch",
    "lr_at_step",
    "clip_grad_norm",
    "build_param_groups",
    # 12
    "model_flops_per_token",
    "compute_mfu",
    "chinchilla_optimal_allocation",
    # 13
    "overfit_single_batch",
    "format_eta",
    "estimate_remaining",
    # 14
    "apply_repetition_penalty",
    "top_k_filter",
    "top_p_filter",
    "KVCache",
    "generate_with_cache",
    # 15
    "PROMPTS_TINYSTORIES",
    "perplexity_from_loss",
    "bits_per_byte",
    "evaluate_perplexity",
    "run_prompt_battery",
    "write_eval_report",
    # 16
    "CHAT_MARKERS",
    "build_chat_template",
    "mask_prompt_tokens",
    "LoRALinear",
    "merge_lora_weights",
    "apply_lora_to_model",
    "count_trainable",
    # 17
    "quantize_int8_symmetric",
    "dequantize_int8",
    "quantization_error",
]
