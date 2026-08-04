# LLM from scratch

A **course-repository** for building a GPT of 8,933,440 parameters by programming in PyTorch,
and training it on your own hardware until it writes coherent short stories.

It is not a tutorial to read. You open the repo in VSCode, read the theory, implement functions
marked with `NotImplementedError`, and run tests until they pass.

```bash
make install
uv run python -m llmfs next
```

## How it works

Every module has five files and the loop is always the same:

```
THEORY.md      →  you read (10-15 min)
exercises.py   →  you implement
llmfs check NN →  red → llmfs hint NN -e 1   →  green → next
llmfs demo NN  →  you see the concept in charts and numbers
SOLUTION.md    →  the explanation, AND the complete code to copy if you get stuck
```

**If you really get stuck, every `SOLUTION.md` ends with the whole code**, ready to copy and
paste. A test verifies that this code compiles, uses only names you have available, and passes
the module's tests. Copying it is not cheating: cheating would be copying it without having
tried.

**The tests compare against a reference, not against "it did not blow up".** Your
`MultiHeadAttention` is validated with `torch.allclose` against `nn.MultiheadAttention`; your
`AdamW` against `torch.optim.AdamW`; your `layer_norm` against `F.layer_norm`.

**You never get blocked.** There is an `llmfs/reference/` with everything implemented. If your
module 6 exercise is half done, modules 7 onwards use the reference automatically and warn you
on stderr. And when your exercise is right, **the final model trains with your code**.

## The curriculum

**18 modules, 62 exercises, ~42 h of work** (not counting GPU time).

### Part 0 — Before you start

| | module | what you build | time |
|---|---|---|---|
| 00 | **What is an LLM** | a text generator by counting, without torch | 1 h |

### Part I — Foundations

| | module | what you build | time |
|---|---|---|---|
| 01 | Environment and hardware | you measure your GPU's real TFLOPS | 45 min |
| 02 | Autodifferentiation | a scalar autograd engine, micrograd style | 3 h |
| 03 | Tokenization | BPE from scratch, a 4096-token vocabulary | 4 h |
| 04 | Data | uint16 memmap and the window dataloader | 2 h |

### Part II — Architecture

| | module | what you build | time |
|---|---|---|---|
| 05 | Baselines | bigram, Bengio's MLP, cross-entropy | 2 h |
| 06 | **Self-attention** | Q/K/V, causal mask, multi-head | 4 h |
| 07 | Normalization | LayerNorm → RMSNorm, pre-norm vs post-norm | 1.5 h |
| 08 | FFN and activations | GELU, SwiGLU, the 2/3 factor | 1.5 h |
| 09 | Position and RoPE | sinusoidal → RoPE, extrapolation | 2.5 h |
| 10 | **The full GPT** | weight tying, scaled init, **8,933,440 params** | 3 h |

### Part III — Training

| | module | what you build | time |
|---|---|---|---|
| 11 | The loop | AdamW from scratch, warmup+cosine, clipping | 4 h |
| 12 | Efficiency and scaling | MFU, Chinchilla | 2 h |
| 13 | **The real run** | overfit on a batch, and you train for real | 1 h |

### Part IV — Use and evaluation

| | module | what you build | time |
|---|---|---|---|
| 14 | Inference | temperature, top-k, top-p, **KV cache** | 3 h |
| 15 | Evaluation | perplexity, bits/byte, TinyStories battery | 2 h |
| 16 | Post-training | chat template, SFT, **LoRA from scratch** | 3 h |
| 17 | Extras and limits | int8, and what separates you from a frontier model | 2 h |

## The final model

```yaml
vocab_size: 4096      n_layers: 6       d_model: 320
n_heads: 8            d_ff: 896         context_length: 512
norm: rmsnorm         pos: rope         activation: swiglu
tie_embeddings: true  dropout: 0.0
```

| component | parameters |
|---|---|
| embeddings (4096 × 320) | 1,310,720 |
| attention (6 × 4 × 320²) | 2,457,600 |
| SwiGLU (6 × 3 × 320 × 896) | 5,160,960 |
| RMSNorm (13 × 320) | 4,160 |
| lm_head (tied) | 0 |
| **TOTAL** | **8,933,440** |

## Measured times

Everything below is **really measured**, not estimated. On a MacBook Pro M5 (MPS):

| | time |
|---|---|
| complete test suite | **4.6 s** |
| `llmfs demo 06` (trains attention and produces heatmaps) | 15 s |
| `llmfs demo 13` (complete training run) | 40 s |
| **`llmfs train --config tiny_char`** (1500 steps) | **70 s** |
| `llmfs demo 16` (real SFT) | 30 s |

The `tiny_char` training runs at **112k tokens/s** and takes the loss from 3.2 to 1.60.

### The final TinyStories run

On the RTX 2060 (Turing, 51.6 TFLOPS fp16), with 500M tokens:

```
FLOPs = 6 × 7.62M × 500M ≈ 2.3·10¹⁶
```

With a realistic MFU of 10-15%: **between 2 and 5 hours**. The real number will come from your
own measurement in the first few minutes — the trainer prints tokens/s, MFU and ETA every few
steps.

Before launching it, two things: the overfit on a batch (30 s) and `--max-steps 100` to see the
real pace.

## Hardware

Everything runs on **CUDA, MPS and CPU without changes**. Detection and the precision policy
live in `llmfs/device.py` and nowhere else.

**RTX 2060 (Turing, sm_75):** no bfloat16 in hardware, so fp16 + `GradScaler`. Careful:
`torch.cuda.is_bf16_supported()` returns `True` on Turing counting software emulation, which is
why the code looks at the compute capability directly. No FlashAttention-2 (it needs sm_80),
but `F.scaled_dot_product_attention` falls back on its own to the *memory-efficient* backend.
`torch.compile` off by default: on Turing it frequently fails to compile.

**Apple Silicon (MPS):** `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before importing torch. fp32 by
default. Some ops fall back to CPU silently, and that is the most common cause of unexplained
slowness on a Mac.

## Commands

```bash
make install                    # uv sync --extra compare
make test                       # your progress (red until you implement: that is normal)
make test-reference             # course health (always green)
make test-solutions             # checks that the solutions' code can be copied

llmfs status                    # progress table, computed by running the tests
llmfs next                      # which module comes next and which exercise
llmfs check 06                  # module 06 tests
llmfs hint 06 -e 2              # progressive hint (repeat for a deeper level)
llmfs demo 06                   # the module's experiment
llmfs device                    # detected hardware
llmfs train --config tiny_char  # trains for real
```

**The curriculum state is not declared anywhere**: it is computed by running the tests.

## About intellectual honesty

Every `THEORY.md` closes with a **"Where the debate is"** section, and it is not decoration.
Throughout the course you are going to read that:

- SwiGLU works better and **its own author writes** that there is no explanation for it.
- Adam dominates without anyone quite knowing why; the usual justification does not survive
  analysis.
- Chinchilla's coefficients have much wider confidence intervals than was reported, according
  to a 2024 reanalysis.
- Benchmark evaluation is contaminated, and separating "it has learned" from "it has seen it"
  is technically hard.
- Normalization alone already rescues the gradient almost as much as the residuals — the usual
  argument is half true, and module 07's demo measures it.

That part does not usually appear in tutorials, and it is the one that helps most for reading
papers with judgement.

## Dependencies

`torch`, `numpy`, `datasets`, `matplotlib`, `pytest`, `tqdm`, `pyyaml`, `rich`, `regex`.

`tiktoken` goes in the `[compare]` extra and is only used in the module 03 comparison.
**No `transformers` or HuggingFace for the model**: `datasets` only downloads TinyStories.

## Getting started

```bash
make install
uv run python -m llmfs next
```

Module 00 has no torch, no matrices, no derivatives: you build a text generator with
dictionaries and one division. And you see the autoregressive loop working before knowing what
a transformer is.

## License

MIT. See [LICENSE](LICENSE).
