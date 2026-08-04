"""Reference for module 15: evaluation."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import torch

#: The prompts from the TinyStories paper's qualitative battery. They are chosen to test
#: different things: narrative continuation, causal coherence, use of an object mentioned
#: earlier, and story closure.
PROMPTS_TINYSTORIES: tuple[tuple[str, str], ...] = (
    ("Once upon a time, there was a little girl named Lily. She", "basic continuation"),
    ("Tom and Jane went to the park. They saw a big dog. The dog", "causal coherence"),
    ("Sara had a red ball. She threw the ball and it", "object tracking"),
    ("The cat was very hungry. It looked for food and finally", "resolution"),
    ("One day, a boy found a shiny key. He used the key to", "use of an object"),
    ("The sun was going down. The children said goodbye and", "story closure"),
)


def perplexity_from_loss(loss: float) -> float:
    """Perplexity from the mean loss in nats: `exp(loss)`.

    Interpretation: how many equally likely options the model is effectively torn between.
    A perplexity of 10 means that on average it is as undecided as if it were choosing at
    random among 10 words.

    With a 4096-token vocabulary and an untrained model, the perplexity is 4096.
    """
    if not math.isfinite(loss):
        return float("inf")
    return math.exp(loss)


def bits_per_byte(total_loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    """Bits per byte: the metric you CAN compare across different tokenizers.

    THE PROBLEM WITH PERPLEXITY. It depends on the tokenizer. If your vocabulary splits
    words into smaller pieces, each individual token is easier to predict and your
    perplexity comes out better without the model being better. Comparing perplexities
    across models with different tokenizers means absolutely nothing, and it is done
    constantly.

    THE SOLUTION. Normalize by BYTES of original text instead of by tokens. Bytes do not
    depend on how you chop things up.

        bits_per_byte = (total_loss_in_nats / ln(2)) / n_bytes

    The `/ ln(2)` converts nats to bits.

    INTERPRETATION. It is literally how many bits you would need to transmit the text using
    the model as a compressor. A model at 1.0 bits/byte compresses the text to an eighth.
    The best language models are around 0.6-0.8 bits/byte on English text; gzip is around
    2.5.

    Args:
        total_loss_nats: the SUM of the losses, not the mean.
        n_tokens: how many tokens were evaluated (not used in the computation, but asked for
            to make it clear the loss is a total and not a mean).
        n_bytes: how many bytes the original text had.
    """
    if n_bytes <= 0:
        raise ValueError("n_bytes has to be positive")
    return total_loss_nats / math.log(2) / n_bytes


@torch.no_grad()
def evaluate_perplexity(
    model: Any,
    data: Any,
    batch_size: int,
    context_length: int,
    iters: int = 100,
    device: Any = None,
    get_batch: Callable[..., Any] | None = None,
) -> dict[str, float]:
    """Mean loss and perplexity over a dataset.

    Not an exercise; it is used by module 15's report.
    """
    from llmfs.bridge import resolve

    get_batch = get_batch or resolve("04_data", "get_batch")
    model.eval()

    total, n = 0.0, 0
    for _ in range(iters):
        x, y = get_batch(data, batch_size, context_length, device=device)
        _, loss = model(x, y)
        total += float(loss.detach())
        n += 1

    mean = total / max(1, n)
    return {"loss": mean, "perplexity": perplexity_from_loss(mean), "iters": n}


def run_prompt_battery(
    generate_fn: Callable[[str], str],
    prompts: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Generate one completion for each prompt in the battery.

    `generate_fn(prompt) -> text` wraps the model and the tokenizer, so this function knows
    nothing about either of them.

    Returns:
        A list of dicts with `prompt`, `tests` and `completion`.
    """
    prompts = prompts or PROMPTS_TINYSTORIES
    return [
        {"prompt": prompt, "tests": label, "completion": generate_fn(prompt)}
        for prompt, label in prompts
    ]


def write_eval_report(
    path: Any,
    metrics: dict[str, Any],
    battery: list[dict[str, str]],
    config_summary: str = "",
) -> Any:
    """Write the evaluation report in markdown.

    Not an exercise: it is the scaffolding that makes the interesting part (reading the
    completions) comfortable.
    """
    from datetime import datetime
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Evaluation report",
        "",
        f"Generated on {datetime.now():%Y-%m-%d %H:%M}.",
        "",
    ]
    if config_summary:
        lines += ["```", config_summary, "```", ""]

    lines += ["## Metrics", "", "| metric | value |", "|---|---|"]
    for key, value in metrics.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {formatted} |")

    lines += [
        "",
        "## Qualitative battery",
        "",
        "The prompts come from the TinyStories paper. The evaluation is yours: read them",
        "and judge grammar, coherence, and whether the story makes sense.",
        "",
    ]
    for i, case in enumerate(battery, start=1):
        lines += [
            f"### {i}. {case['tests']}",
            "",
            f"**Prompt:** `{case['prompt']}`",
            "",
            "```",
            case["completion"],
            "```",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
