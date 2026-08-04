"""The curriculum: which modules exist, what exercises each one has, and in what order.

This is the single source of truth for the course. The CLI (`status`, `next`, `check`,
`demo`, `hint`) and the bridge (the safety net) read from here. If you add an exercise,
you add it in this file and everything else finds out on its own.

The times (`est_minutes`) are estimates of hands-on-keyboard work for someone who already
programs, NOT counting GPU time. They do not include reading the theory twice because you
did not get it the first time, which happens and is normal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from llmfs.paths import modules_dir


@dataclass(frozen=True)
class Exercise:
    """An exercise: a function or class you have to implement in `exercises.py`."""

    #: Exact name of the symbol in `exercises.py`. It is the key the bridge uses.
    name: str
    #: What it does, in one line.
    title: str
    #: `True` if it is a class (`nn.Module` or otherwise) instead of a function.
    is_class: bool = False


@dataclass(frozen=True)
class Module:
    """A course module."""

    number: int
    slug: str
    title: str
    part: str
    summary: str
    est_minutes: int
    exercises: tuple[Exercise, ...] = field(default_factory=tuple)
    has_demo: bool = True
    #: Papers and links cited in THEORY.md.
    references: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def id(self) -> str:
        """Directory identifier, e.g. `06_attention`."""
        return f"{self.number:02d}_{self.slug}"

    @property
    def path(self) -> Path:
        return modules_dir() / self.id

    @property
    def test_file(self) -> Path:
        return self.path / f"test_{self.number:02d}.py"

    @property
    def exercises_file(self) -> Path:
        return self.path / "exercises.py"

    @property
    def demo_file(self) -> Path:
        return self.path / "demo.py"

    @property
    def theory_file(self) -> Path:
        return self.path / "THEORY.md"

    @property
    def solution_file(self) -> Path:
        return self.path / "SOLUTION.md"

    def exercise(self, name_or_index: str | int) -> Exercise:
        """Look up an exercise by name or by position (1-indexed, as displayed)."""
        if isinstance(name_or_index, int):
            if not 1 <= name_or_index <= len(self.exercises):
                raise KeyError(
                    f"Module {self.id} has {len(self.exercises)} exercises; "
                    f"you asked for number {name_or_index}."
                )
            return self.exercises[name_or_index - 1]
        for ex in self.exercises:
            if ex.name == name_or_index:
                return ex
        raise KeyError(f"There is no exercise {name_or_index!r} in {self.id}.")


PART_0 = "0 - Before you start"
PART_I = "I - Foundations"
PART_II = "II - Architecture"
PART_III = "III - Training"
PART_IV = "IV - Use and evaluation"


CURRICULUM: tuple[Module, ...] = (
    # ------------------------------------------------------------------ Part 0
    Module(
        number=0,
        slug="what_is_an_llm",
        title="What an LLM actually is",
        part=PART_0,
        summary="No jargon, no transformers: you build a text generator in 20 lines.",
        est_minutes=60,
        exercises=(
            Exercise("next_token_probs", "Turn counts into probabilities that sum to 1"),
            Exercise("sample_next_token", "Pick the next character according to those probabilities"),
            Exercise("generate_naive", "Chain predictions together to generate text"),
        ),
        references=(
            ("Shannon 1948, A Mathematical Theory of Communication", "https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf"),
        ),
    ),
    # ------------------------------------------------------------------ Part I
    Module(
        number=1,
        slug="environment",
        title="Environment and hardware",
        part=PART_I,
        summary="uv, PyTorch, CUDA/MPS, and how many tokens/s your GPU really delivers.",
        est_minutes=45,
        exercises=(
            Exercise("measure_matmul_tflops", "Measure the real TFLOPS of a large matmul"),
            Exercise("transformer_flops_per_token", "FLOPs per token of a transformer (6N + attention)"),
            Exercise("estimate_tokens_per_second", "Theoretical tokens/s from TFLOPS and MFU"),
        ),
        references=(
            ("Kaplan et al. 2020 (FLOPs appendix)", "https://arxiv.org/abs/2001.08361"),
        ),
    ),
    Module(
        number=2,
        slug="autograd",
        title="Autodifferentiation from scratch",
        part=PART_I,
        summary="A scalar engine in the style of micrograd. Backprop stops being magic.",
        est_minutes=180,
        exercises=(
            Exercise("Value", "A scalar with a compute graph and the chain rule", is_class=True),
            Exercise("topological_order", "Topological order of the graph for the backward pass"),
            Exercise("train_scalar_mlp", "Train an MLP using only your engine"),
        ),
        references=(
            ("Karpathy, micrograd", "https://github.com/karpathy/micrograd"),
            ("Baydin et al. 2018, Automatic Differentiation in ML", "https://arxiv.org/abs/1502.05767"),
        ),
    ),
    Module(
        number=3,
        slug="tokenization",
        title="Tokenization and BPE",
        part=PART_I,
        summary="From characters to your own 4096-token BPE trained on TinyStories.",
        est_minutes=240,
        exercises=(
            Exercise("get_stats", "Count adjacent pairs"),
            Exercise("merge", "Merge a pair into a new id"),
            Exercise("train_bpe", "Train the merges up to vocab_size"),
            Exercise("bpe_encode", "Text -> ids by applying merges in order"),
            Exercise("bpe_decode", "ids -> text with a bytes fallback"),
        ),
        references=(
            ("Sennrich et al. 2016, Neural MT of Rare Words with Subword Units", "https://arxiv.org/abs/1508.07909"),
            ("Karpathy, minbpe", "https://github.com/karpathy/minbpe"),
        ),
    ),
    Module(
        number=4,
        slug="data",
        title="Data: from text to memmap",
        part=PART_I,
        summary="Download, parallel tokenization, uint16 on disk and a windowed dataloader.",
        est_minutes=120,
        exercises=(
            Exercise("pack_tokens_uint16", "Pack ids into a validated uint16 array"),
            Exercise("train_val_split", "Split the corpus without leaking information"),
            Exercise("get_batch", "Sample (x, y) windows shifted by one token"),
        ),
    ),
    # ------------------------------------------------------------------ Part II
    Module(
        number=5,
        slug="baselines",
        title="Baselines: bigram and Bengio's MLP",
        part=PART_II,
        summary="Cross-entropy, perplexity, and what the floor of a trivial model looks like.",
        est_minutes=120,
        exercises=(
            Exercise("uniform_baseline_loss", "The loss of guessing at random: ln(V)"),
            Exercise("bigram_counts", "A V x V count matrix"),
            Exercise("bigram_nll", "Negative log-likelihood with smoothing"),
            Exercise("NeuralBigram", "The bigram as a single embedding layer", is_class=True),
            Exercise("BengioMLP", "MLP with a context window (Bengio 2003)", is_class=True),
        ),
        references=(
            ("Bengio et al. 2003, A Neural Probabilistic Language Model", "https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf"),
        ),
    ),
    Module(
        number=6,
        slug="attention",
        title="Self-attention",
        part=PART_II,
        summary="Q/K/V, causal mask, scaling by sqrt(d_k), and multi-head.",
        est_minutes=240,
        exercises=(
            Exercise("causal_mask", "Triangular mask that blocks looking into the future"),
            Exercise("single_head_attention", "softmax(QK^T/sqrt(d_k) + mask) V"),
            Exercise("MultiHeadAttention", "Multi-head with an output projection", is_class=True),
        ),
        references=(
            ("Vaswani et al. 2017, Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
            ("Elhage et al. 2021, A Mathematical Framework for Transformer Circuits", "https://transformer-circuits.pub/2021/framework/index.html"),
        ),
    ),
    Module(
        number=7,
        slug="normalization",
        title="Normalization and residuals",
        part=PART_II,
        summary="LayerNorm -> RMSNorm, pre-norm vs post-norm, and why the gradient survives.",
        est_minutes=90,
        exercises=(
            Exercise("layer_norm", "Normalize by mean and variance, with gamma and beta"),
            Exercise("RMSNorm", "Scale only by the root mean square", is_class=True),
            Exercise("prenorm_residual", "x + f(norm(x)) versus norm(x + f(x))"),
        ),
        references=(
            ("Ba et al. 2016, Layer Normalization", "https://arxiv.org/abs/1607.06450"),
            ("Zhang & Sennrich 2019, Root Mean Square Layer Normalization", "https://arxiv.org/abs/1910.07467"),
            ("Xiong et al. 2020, On Layer Normalization in the Transformer Architecture", "https://arxiv.org/abs/2002.04745"),
        ),
    ),
    Module(
        number=8,
        slug="mlp_and_activations",
        title="FFN, GELU y SwiGLU",
        part=PART_II,
        summary="The 4x ratio, why SwiGLU uses 2/3 of the hidden size, and where the parameters live.",
        est_minutes=90,
        exercises=(
            Exercise("gelu", "GELU with the tanh approximation"),
            Exercise("swiglu_hidden_dim", "Compute d_ff = round(2/3 * 4d) to the nearest multiple of 64"),
            Exercise("SwiGLU", "Gated FFN: (Swish(xW1) * xW3) W2", is_class=True),
        ),
        references=(
            ("Hendrycks & Gimpel 2016, Gaussian Error Linear Units", "https://arxiv.org/abs/1606.08415"),
            ("Shazeer 2020, GLU Variants Improve Transformer", "https://arxiv.org/abs/2002.05202"),
        ),
    ),
    Module(
        number=9,
        slug="position",
        title="Positional information and RoPE",
        part=PART_II,
        summary="Learned -> sinusoidal -> RoPE, and what happens when you extrapolate the context.",
        est_minutes=150,
        exercises=(
            Exercise("sinusoidal_embeddings", "The sine/cosine table from the original paper"),
            Exercise("rope_frequencies", "cos/sin tables for theta^(-2i/d)"),
            Exercise("apply_rope", "Rotate pairs of dimensions of Q and K"),
        ),
        references=(
            ("Su et al. 2021, RoFormer: Rotary Position Embedding", "https://arxiv.org/abs/2104.09864"),
            ("Press et al. 2021, ALiBi", "https://arxiv.org/abs/2108.12409"),
        ),
    ),
    Module(
        number=10,
        slug="the_full_gpt",
        title="The full GPT",
        part=PART_II,
        summary="Block, GPT, weight tying, depth-scaled init, and the exact parameter count.",
        est_minutes=180,
        exercises=(
            Exercise("expected_param_count", "The parameter formula derived by hand"),
            Exercise("count_parameters", "Real count broken down by component"),
            Exercise("TransformerBlock", "Attention + FFN with pre-norm and residuals", is_class=True),
            Exercise("GPT", "The whole model, with tying and scaled init", is_class=True),
        ),
        references=(
            ("Radford et al. 2019, Language Models are Unsupervised Multitask Learners (GPT-2)", "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"),
            ("Press & Wolf 2017, Using the Output Embedding to Improve Language Models", "https://arxiv.org/abs/1608.05859"),
        ),
    ),
    # ------------------------------------------------------------------ Part III
    Module(
        number=11,
        slug="training_loop",
        title="The training loop",
        part=PART_III,
        summary="Your own AdamW, warmup+cosine, clipping, accumulation, AMP and checkpoints.",
        est_minutes=240,
        exercises=(
            Exercise("AdamWScratch", "AdamW from scratch, with bias correction", is_class=True),
            Exercise("lr_at_step", "Linear warmup + cosine decay down to 10%"),
            Exercise("clip_grad_norm", "Clipping by global norm"),
            Exercise("build_param_groups", "No weight decay on norms or biases"),
        ),
        references=(
            ("Loshchilov & Hutter 2019, Decoupled Weight Decay Regularization", "https://arxiv.org/abs/1711.05101"),
            ("Micikevicius et al. 2018, Mixed Precision Training", "https://arxiv.org/abs/1710.03740"),
        ),
    ),
    Module(
        number=12,
        slug="efficiency_and_scaling",
        title="Efficiency and scaling laws",
        part=PART_III,
        summary="MFU, memory, KV cache, Chinchilla, and why 9M parameters and 500M tokens.",
        est_minutes=120,
        exercises=(
            Exercise("model_flops_per_token", "FLOPs per token, separating matmuls from attention"),
            Exercise("compute_mfu", "Model FLOPs Utilization against the hardware peak"),
            Exercise("chinchilla_optimal_allocation", "Optimal split of a compute budget"),
        ),
        references=(
            ("Kaplan et al. 2020, Scaling Laws for Neural Language Models", "https://arxiv.org/abs/2001.08361"),
            ("Hoffmann et al. 2022, Training Compute-Optimal LLMs (Chinchilla)", "https://arxiv.org/abs/2203.15556"),
            ("Chowdhery et al. 2022, PaLM (definition of MFU)", "https://arxiv.org/abs/2204.02311"),
        ),
    ),
    Module(
        number=13,
        slug="final_training",
        title="The real run",
        part=PART_III,
        summary="TinyStories end to end: resumable, with periodic samples and an ETA.",
        est_minutes=60,
        exercises=(
            Exercise("overfit_single_batch", "Memorize one batch down to ~0 loss: the test that catches bugs"),
            Exercise("format_eta", "An honest ETA from the measured throughput"),
        ),
        references=(
            ("Eldan & Li 2023, TinyStories", "https://arxiv.org/abs/2305.07759"),
        ),
    ),
    # ------------------------------------------------------------------ Part IV
    Module(
        number=14,
        slug="inference",
        title="Inference and sampling",
        part=PART_IV,
        summary="Temperature, top-k, top-p, repetition penalty and KV cache.",
        est_minutes=180,
        exercises=(
            Exercise("apply_repetition_penalty", "Penalize already generated tokens"),
            Exercise("top_k_filter", "Keep the k largest logits"),
            Exercise("top_p_filter", "Nucleus of cumulative mass p"),
            Exercise("KVCache", "Per-layer cache of keys and values", is_class=True),
            Exercise("generate_with_cache", "Incremental generation with the cache"),
        ),
        references=(
            ("Holtzman et al. 2020, The Curious Case of Neural Text Degeneration", "https://arxiv.org/abs/1904.09751"),
        ),
    ),
    Module(
        number=15,
        slug="evaluation",
        title="Evaluation",
        part=PART_IV,
        summary="Perplexity, bits per byte, and the qualitative battery from the TinyStories paper.",
        est_minutes=120,
        exercises=(
            Exercise("perplexity_from_loss", "Perplexity from the mean loss"),
            Exercise("bits_per_byte", "A metric comparable across different tokenizers"),
            Exercise("run_prompt_battery", "Generate completions for the fixed prompts"),
        ),
        references=(
            ("Eldan & Li 2023, TinyStories (evaluation section)", "https://arxiv.org/abs/2305.07759"),
        ),
    ),
    Module(
        number=16,
        slug="finetuning",
        title="Post-training: SFT y LoRA",
        part=PART_IV,
        summary="Chat template, loss on the answer only, and LoRA implemented by hand.",
        est_minutes=180,
        exercises=(
            Exercise("build_chat_template", "Serialize messages to text with markers"),
            Exercise("mask_prompt_tokens", "Ignore the prompt in the loss (-100)"),
            Exercise("LoRALinear", "W + (alpha/r) B A with low-rank A, B", is_class=True),
            Exercise("merge_lora_weights", "Fold the adapters into the base weights"),
        ),
        references=(
            ("Hu et al. 2021, LoRA", "https://arxiv.org/abs/2106.09685"),
            ("Ouyang et al. 2022, InstructGPT", "https://arxiv.org/abs/2203.02155"),
        ),
    ),
    Module(
        number=17,
        slug="extra",
        title="Extras and honest limits",
        part=PART_IV,
        summary="int8 quantization, serving with FastAPI, and what separates this from a frontier model.",
        est_minutes=120,
        exercises=(
            Exercise("quantize_int8_symmetric", "Weights to int8 with a per-channel scale"),
            Exercise("dequantize_int8", "Back to float, and the reconstruction error"),
            Exercise("quantization_error", "Measure the damage: relative error and perplexity"),
        ),
    ),
)


# ---------------------------------------------------------------------------- lookup

_BY_ID = {m.id: m for m in CURRICULUM}
_BY_NUMBER = {m.number: m for m in CURRICULUM}


def all_modules() -> tuple[Module, ...]:
    return CURRICULUM


def get_module(ref: str | int) -> Module:
    """Resolve a module from `5`, `"5"`, `"05"` or `"06_attention"`.

    Raises:
        KeyError: with the list of valid modules, so the error is actually useful.
    """
    if isinstance(ref, int):
        if ref in _BY_NUMBER:
            return _BY_NUMBER[ref]
        raise KeyError(f"There is no module {ref}. They run from 0 to {len(CURRICULUM) - 1}.")

    ref = str(ref).strip()
    if ref in _BY_ID:
        return _BY_ID[ref]
    if ref.isdigit() and int(ref) in _BY_NUMBER:
        return _BY_NUMBER[int(ref)]
    # Prefix search on the slug, so you can type `llmfs check atten`.
    matches = [m for m in CURRICULUM if m.slug.startswith(ref.lower())]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        f"Module {ref!r} not recognized. Valid ones: " + ", ".join(m.id for m in CURRICULUM)
    )


def parts() -> dict[str, list[Module]]:
    """Modules grouped by part, in order."""
    grouped: dict[str, list[Module]] = {}
    for module in CURRICULUM:
        grouped.setdefault(module.part, []).append(module)
    return grouped


def total_exercises() -> int:
    return sum(len(m.exercises) for m in CURRICULUM)


def total_minutes() -> int:
    return sum(m.est_minutes for m in CURRICULUM)
