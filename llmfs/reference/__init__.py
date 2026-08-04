"""Reference implementations: the safety net of the course.

Everything here is solved and correct. Two uses:

1. **Safety net.** `llmfs.bridge` pulls from this package when your exercise from module N
   is not there yet, so that modules N+1 onwards keep working.
2. **Oracle for the tests.** The `test_NN.py` files compare your implementation against
   these with `torch.allclose`, not against "it did not crash".

Style: readability wins over speed. These are direct implementations of the formula in the
corresponding THEORY.md, using the same variable names. Do not look for performance tricks
here; the ones that matter live in `llmfs/model/` and `llmfs/train/`.

If you are about to look at this code before attempting the exercise, look at
`llmfs hint NN` first. The hints are written to unblock you without burning the exercise.

Every symbol in the curriculum is re-exported here under its own name, which is the key the
bridge uses. Which file it lives in is an organizational detail.
"""

from __future__ import annotations

# --- module 00: what an LLM is --------------------------------------------------------
from llmfs.reference.intro import (
    build_count_table,
    generate_naive,
    next_token_probs,
    sample_next_token,
)

# --- module 01: environment and hardware ----------------------------------------------
from llmfs.reference.hardware import (
    estimate_tokens_per_second,
    matmul_flops,
    measure_matmul_tflops,
    transformer_flops_per_token,
)

# --- module 02: autograd --------------------------------------------------------------
from llmfs.reference.autograd import (
    MLP,
    Layer,
    Neuron,
    Value,
    topological_order,
    train_scalar_mlp,
)

# --- module 03: tokenization ----------------------------------------------------------
from llmfs.reference.tokenizer import (
    GPT4_SPLIT_PATTERN,
    bpe_decode,
    bpe_encode,
    compression_ratio,
    get_stats,
    merge,
    train_bpe,
)

# --- module 04: data ------------------------------------------------------------------
from llmfs.reference.data import get_batch, pack_tokens_uint16, train_val_split

# --- module 06: attention -------------------------------------------------------------
from llmfs.reference.attention import MultiHeadAttention, causal_mask, single_head_attention

# --- module 07: normalization ---------------------------------------------------------
from llmfs.reference.norm import RMSNorm, layer_norm, postnorm_residual, prenorm_residual

# --- module 08: mlp and activations ---------------------------------------------------
from llmfs.reference.mlp import MLP as FeedForwardMLP
from llmfs.reference.mlp import SwiGLU, gelu, swiglu_hidden_dim

# --- module 09: position --------------------------------------------------------------
from llmfs.reference.position import (
    LearnedPositionalEmbedding,
    apply_rope,
    rope_frequencies,
    rotate_half,
    sinusoidal_embeddings,
)

# --- module 10: the full gpt ----------------------------------------------------------
from llmfs.reference.gpt import (
    GPT,
    TransformerBlock,
    count_parameters,
    expected_param_count,
    make_ffn,
    make_norm,
)

# --- module 11: training loop ---------------------------------------------------------
from llmfs.reference.training import (
    AdamWScratch,
    build_param_groups,
    clip_grad_norm,
    lr_at_step,
)

# --- module 12: efficiency and scaling ------------------------------------------------
from llmfs.reference.scaling import (
    chinchilla_optimal_allocation,
    compute_mfu,
    model_flops_per_token,
)

# --- module 13: the real run ----------------------------------------------------------
from llmfs.reference.final_run import estimate_remaining, format_eta, overfit_single_batch

# --- module 14: inference -------------------------------------------------------------
from llmfs.reference.inference import (
    KVCache,
    apply_repetition_penalty,
    generate_with_cache,
    top_k_filter,
    top_p_filter,
)

# --- module 15: evaluation ------------------------------------------------------------
from llmfs.reference.evaluation import (
    PROMPTS_TINYSTORIES,
    bits_per_byte,
    evaluate_perplexity,
    perplexity_from_loss,
    run_prompt_battery,
    write_eval_report,
)

# --- module 16: post-training ---------------------------------------------------------
from llmfs.reference.finetune import (
    CHAT_MARKERS,
    LoRALinear,
    apply_lora_to_model,
    build_chat_template,
    count_trainable,
    mask_prompt_tokens,
    merge_lora_weights,
)

# --- module 17: extras ----------------------------------------------------------------
from llmfs.reference.quantization import (
    dequantize_int8,
    quantization_error,
    quantize_int8_symmetric,
)

# --- module 05: baselines -------------------------------------------------------------
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
