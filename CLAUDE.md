# CLAUDE.md — instructions for Claude Code sessions in this repo

## What this is

A **course-repository** for learning to build an LLM from scratch by programming in PyTorch.
It is not a library and not a tutorial to read: the user opens VSCode, reads the theory,
implements functions marked with `NotImplementedError` and runs tests until they pass. The
final goal is a GPT of **8,933,440 parameters** trained by them on TinyStories until it
generates coherent short stories in English.

**This branch (`english`) is the English version of the course. `main` holds the Spanish
one.** Everything here — prose, identifiers, file names, test names, CLI messages — is in
English. Nothing is bilingual: do not reintroduce Spanish on this branch, and do not port
translations back to `main` unless asked.

## THE MOST IMPORTANT POINT: who you are talking to

The user is an **experienced software engineer** (Python, git, CLI, architecture) but their
**background in LLMs and machine learning is low**, and the course is aimed at people in that
same situation. Keep the two apart or you will get one of them wrong:

- **Do not explain programming to them.** What a loop is, what a class is, what `dict.get`
  does. That wastes their time.
- **Do explain everything about ML from scratch.** What a logit is, why things are normalized,
  what it means for a gradient to "go to zero". Nothing is assumed.

The first version of this repo was written far too technically and had to be rewritten. The
original brief asked for theory that was "dense, no analogies, 400-900 words"; that
instruction was **repealed** on 2026-07-30 after seeing the material. What governs is what
follows.

### Mandatory structure of every explanation: intuition → example → formula

Every ML concept enters **three times and in this order**:

1. **What problem it solves**, in plain language. No formulas, no undefined jargon.
2. **An example with small numbers** the reader can follow by hand. 2×3 matrices, three words,
   four counts. Concrete numbers, not symbols.
3. **The formal formula**, connected explicitly to the previous example.

The mathematics is **neither removed nor hidden**: it just stops being the first thing you
read. A `THEORY.md` that opens with `C_token ≈ 6N + 12·n_layers·T·d_model` is badly written
even if the formula is correct.

### Mandatory structure of every exercise docstring

It is the same idea applied to code, and it was established on 2026-07-31 after the user said
"I read this and I do not know what I have to do". The diagnosis: the docstrings explained
**what** the function was and **why** it was needed, but never **what to type**.

The order is fixed, and **`WHAT YOU HAVE TO WRITE` goes first, before any theory**:

```
<One sentence: what the function does>

WHAT YOU HAVE TO WRITE
----------------------
<Numbered steps with the concrete code to type, indented. Not pseudocode:
 the real lines, with the real variable names.>

WHAT SHOULD COME OUT / CHECK IT WITH...
---------------------------------------
<Concrete numbers the reader can verify by hand>

WHY / WHERE THAT FORMULA COMES FROM
-----------------------------------
<The intuition, now that they know what they are writing>

WATCH OUT FOR... / THE MISTAKES TO AVOID
----------------------------------------
<The traps, and above all the silent ones: the ones that raise no error and degrade the
 result>

Args: / Returns: / Raises:
```

All 62 docstrings follow this format. When adding a new one, the sweep to check is:

```python
# walk CURRICULUM and verify that each symbol has the section
"WHAT YOU HAVE TO WRITE" in ast.get_docstring(node)
```

Two rules that came out of writing them:

- **The code in the steps has to compile with what the student has imported.** If your step
  uses `math.sqrt` and `exercises.py` does not import `math`, the step is wrong. It really
  happened in module 12 (changed to `** 0.5`) and in 01 (the file was missing `import time`).
- **The example numbers are measured, not estimated.** Module 17's were invented to look
  plausible and were off by a factor of 5; you have to run the reference and copy.

## Writing rules, non-negotiable

- **All the prose in English**: theory, docstrings, comments, CLI messages, test names. The
  code identifiers are in English too — on this branch there is no mixing, everything is one
  language.
- **Plain ASCII in the `.py` files' prose.** No smart quotes, no em dashes in code comments,
  no accented characters: the CLI is read in terminals with all sorts of encodings. The
  section titles in capitals of the docstrings (`WHAT YOU HAVE TO WRITE`, `WHY THAT FORMULA`)
  are the anchor points for skim reading and must stay exactly as written. The `.md` files may
  use normal typographic characters (–, ×, →, …) freely.
- **Every `THEORY.md` opens with `## Why this module matters`**, BEFORE any concept: what
  problem it solves, what you will know at the end, and what it costs. There is a test that
  verifies it. Someone who does not know about LLMs cannot judge whether four hours on
  attention are worth it unless you first tell them it is THE piece that separates a mediocre
  model from ChatGPT.
- **At least 900 words per `THEORY.md`, no ceiling.** The upper limit was removed on
  2026-07-31: every concept gets explained for as long as it needs, with concrete examples.
- **Every `SOLUTION.md` ends with `## The complete code`**: the entire implementation of every
  exercise in the module, copy-pasteable. There is a test that verifies it. The original brief
  said "not the bare code"; that instruction was repealed on 2026-07-31. Whoever gets stuck
  needs to see the code, not read about it.
- **The same register in the `exercises.py` docstrings, in `SOLUTION.md`, in the `hints.py`
  hints and in the CLI messages.**
- **Every exercise docstring opens with `WHAT YOU HAVE TO WRITE`**, and after that come the
  example, the why and the traps. See the section above: it is non-negotiable and there is a
  sweep that checks it.
- **Every ML term that appears must be in `GLOSSARY.md`**, and every `THEORY.md` links there
  at the end.
- Analogies: allowed if they are **mechanical and verifiable** (the roulette wheel for
  sampling, splitting up the [0,1] line). Mystical ones are forbidden ("it is like a brain",
  "it understands").
- **Intellectual honesty is mandatory.** Where there is open debate (pre-norm vs post-norm,
  whether the scaling laws hold, what a model really builds inside) you say so, in a
  `## Where the debate is` section at the end of every `THEORY.md`. The original papers are
  cited with a link, in the `references` tuple of `curriculum.py` and in the `THEORY.md`.
- **Run what you write.** Every `demo.py` and every test has to actually run before you call it
  done. No code that "should work".
- **Validate the tests against the reference** with `make test-reference` before calling a
  module finished. If a test fails in that mode, either the test is badly written or the
  reference is wrong: in both cases it is a course bug, not a student bug.

## Architecture: three layers not to be confused

```
modules/NN_*/exercises.py   ← what THE USER WRITES (templates with NotImplementedError)
llmfs/reference/            ← canonical implementations, complete and correct
llmfs/{model,train,infer}/  ← the code that really trains, built on top of the bridge
```

`llmfs/bridge.py` is the key piece. When the production code needs `MultiHeadAttention`, it
calls `bridge.resolve("06_attention", "MultiHeadAttention")`, which:

1. loads `modules/06_attention/exercises.py`,
2. checks with AST whether the symbol is still the template (`raise NotImplementedError` or
   `pass` as the only body) — see `bridge.looks_unimplemented`,
3. runs the probe from `llmfs/probes.py` if one is registered,
4. returns the user's implementation if it passes, and otherwise the one from `llmfs.reference`,
   warning **once** on stderr.

Consequence: **the user never gets blocked**, and when their exercise is right, the final model
trains with THEIR code. The stderr warning is deliberate and must not be silenced.

Useful environment variables: `LLMFS_FORCE_REFERENCE=1` (ignores the exercises),
`LLMFS_BRIDGE_VERBOSE=1` (also warns when it uses the user's code),
`LLMFS_DEVICE=cpu|cuda|mps`, `LLMFS_AMP=0|1`, `LLMFS_ROOT`.

## Target hardware — respect it in every decision

| | |
|---|---|
| Main PC | Intel i7 7700K, 16 GB RAM, **RTX 2060 6 GB (Turing, sm_75)** |
| Laptop | MacBook Pro M5, 16 GB RAM, **MPS** backend |

Everything hardware-specific lives in `llmfs/device.py` and nowhere else. What has to be kept
in mind:

- **sm_75 has no bfloat16 in hardware.** `float16` + `GradScaler` is used. And careful:
  `torch.cuda.is_bf16_supported()` returns `True` on Turing counting software emulation, so it
  **is not used**; the compute capability is checked directly.
- **No FlashAttention-2** (it requires sm_80+). `F.scaled_dot_product_attention` falls back on
  its own to the `memory_efficient` backend, which does work on Turing.
- **`torch.compile` off by default.** On Turing it frequently fails to compile. An optional
  flag (`compile: true` in the YAML), never by default.
- **MPS**: `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by `llmfs/__init__.py` *before* importing
  torch. fp32 by default, fp16 optional. Some ops fall back to CPU silently.
- The logits tensor (`batch × ctx × 4096`) is the biggest memory consumer of the final run,
  above the activations. If there is an OOM on the 2060, that is where to look.

## Commands

```bash
make install          # uv sync --extra compare
make test             # complete suite (tests/ + modules/)
make test-fast        # without the ones marked @pytest.mark.slow
uv run pytest tests/  # only the package infrastructure

uv run python -m llmfs status        # progress table (runs the tests)
uv run python -m llmfs status --cached
uv run python -m llmfs next          # which module comes next and which exercise
uv run python -m llmfs check 05      # module 05 tests with hints
uv run python -m llmfs demo 05       # module 05 experiment
uv run python -m llmfs hint 05 -e 2  # progressive hint (repeat for a deeper level)
uv run python -m llmfs device        # detected hardware
```

The curriculum state **is not declared anywhere**: it is computed by running the tests.
`.llmfs_progress.json` is only a cache (and it is in `.gitignore`).

## How to add a module

1. Register it in `llmfs/curriculum.py`: `Module(...)` with its `Exercise(...)`, an honest
   `est_minutes` and `references` with links to the papers.
2. Create `modules/NN_name/` with **the five files**: `THEORY.md`, `exercises.py`, `demo.py`,
   `test_NN.py`, `SOLUTION.md`. There is a test that verifies none is missing. The `THEORY.md`
   follows the intuition → numeric example → formula structure, closes with
   `## Where the debate is` and links to `GLOSSARY.md`.
3. Implement the pieces in `llmfs/reference/` and re-export them in
   `llmfs/reference/__init__.py`. **Exercise names are unique across the whole course** (there
   is a test); the bridge resolves by plain name.
4. Register probes in `llmfs/probes.py` for the exercises where "written but returns garbage"
   is a real risk (tensor shapes, above all).
5. Write the three hints in `llmfs/hints.py`: conceptual → technical → structural. The third
   one does not give the written solution. Add the new terms to `GLOSSARY.md`.
6. `exercises.py`: the module docstring opens with **HOW TO DO THIS MODULE** (the 5 steps),
   **WHAT YOU ARE GOING TO BUILD** (a diagram of how the exercises fit together) and
   **VOCABULARY YOU ARE GOING TO NEED** (every ML term that appears, defined in one line). The
   docstrings of each exercise carry the input/output shapes and the formula, with a
   `raise NotImplementedError(...)` body.
7. `test_NN.py`: validates against `llmfs.reference` or against the PyTorch equivalent
   (`nn.MultiheadAttention`, `F.layer_norm`...) with `torch.allclose`. **Checking only that it
   does not blow up is not enough.** The tests import with
   `from llmfs.testing import load_exercises` and `ex = load_exercises(__file__)`, never with
   `sys.path`.
8. `demo.py`: an executable experiment that visualizes the concept. It saves figures to
   `runs/figures/` via `llmfs.paths.figures_dir()`. It has to run on cuda, mps and cpu.
9. Regenerate the `SOLUTION.md` code block with
   `uv run python scripts/regenerate_solutions.py` (it extracts it from `llmfs/reference/`, so
   it never diverges). If the exercise needs a helper function or an import that is not in
   `exercises.py`, add it to that script's `HELPERS` or `IMPORTS`.
10. Run `make test` and `make test-solutions` before calling the phase finished. The second one
   pastes each solution over its `exercises.py` and runs the tests: it is the only way to
   guarantee the solutions' code can be copied and works.

## Dependencies

`torch`, `numpy`, `datasets`, `matplotlib`, `pytest`, `tqdm`, `pyyaml`, `rich`, `regex`.

- `regex` (not the stdlib's `re`) because module 02's GPT-4-style pre-tokenizer uses `\p{L}`.
- `tiktoken` is in the `[compare]` extra and is used **only** in module 02's compression
  comparison.
- **No `transformers` and no HuggingFace for the model.** `datasets` is used solely to download
  TinyStories.

## Curriculum state

**18 modules, numbered 00-17, 62 exercises, ~42 h of estimated work.**

`00_what_is_an_llm` was added on 2026-07-30 (it was not in the original brief) and renumbered
everything else: what the brief called module NN is now NN+1. It is a conceptual module without
torch where a text generator by counting is built, and it works as an anchor for everything
that follows: module 14's autoregressive loop is literally the same one.

| Phase | Content | State |
|---|---|---|
| 1 | Skeleton, `llmfs`, CLI, bridge, infrastructure tests | ✅ done |
| 2 | Modules 00-04 (foundations) | ✅ done |
| 3 | Modules 05-10 (baselines + architecture) | ✅ done |
| 4 | Modules 11-13 (training) | ✅ done |
| 5 | Modules 14-17 (use and evaluation) | ✅ done |

When picking this up again: `make test` and `make test-reference` have to be green before
touching anything. `llmfs status` is always the source of truth for which modules exist.

**Parts 0, I and II are finished: modules 00-10.** The model is assembled and audited;
`GPT(ModelConfig())` gives exactly 8,933,440 parameters and `expected_param_count` matches
`count_parameters`.

**THE COURSE IS COMPLETE: all 18 modules, 62 exercises.** 98 infrastructure tests and 518
course tests against the reference, all green. 17 figures generated from real runs.

`uv run python -m llmfs train --config tiny_char` really trains (~70 s on MPS, loss from 3.2 to
1.60, 112k tokens/s) and generates recognizable Shakespeare.

What is left, if this gets picked up again:
- The TinyStories pipeline with real BPE (`llmfs/data/prepare.py` only has
  `prepare_shakespeare`; `prepare()` raises `NotImplementedError` for the rest).
- The `llmfs sample` command is still a stub (`_LATER` in `llmfs/cli.py`).
- The module 17 FastAPI server is described in the theory but not implemented.
- The real 500M-token run on the RTX 2060 has not been executed (there is no CUDA hardware
  here). The README times for that run are marked as an estimate.

Findings worth not repeating:
- `named_parameters()` **deduplicates by default** (`remove_duplicate=True`). The common belief
  that it returns tied weights twice is false.
- A test that passes `model(idx, idx)` (unshifted targets) produces an information leak and a
  loss below `ln(V)`. The symptom is identical to a broken causal mask, so faced with that
  symptom you have to look first at how the batch is assembled.
- `x in list_of_tensors` blows up with "Boolean value of Tensor is ambiguous", because `in`
  uses `==` and on tensors that returns an elementwise comparison. Compare by `id()`.
- The guard `test_no_test_file_defines_the_same_test_twice` has already caught **nine**
  duplicates of `test_matches_the_reference`. When writing a `test_NN.py` with several
  exercises, name each test after the exercise it belongs to
  (`test_adamw_matches_the_reference`), never generically.

The `llmfs/` directories (`model/`, `train/`, `infer/`, `eval/`, `viz/`, `tokenizer/`, `data/`)
are created in the phase that fills them, not before: in this repo nothing exists that does not
work. `llmfs status` is always the source of truth for what is done.

The `llmfs train`, `llmfs sample` and `llmfs data` subcommands exist as stubs that explain in
which module they are built. When implementing them, remove the corresponding entry from
`_LATER` in `llmfs/cli.py`.
