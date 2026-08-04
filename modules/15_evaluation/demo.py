"""Module 15 demo: evaluates your model and generates the report.

    llmfs demo 15

It uses the model trained in module 13 if it exists. It produces `eval_report.md` with the
metrics and the six continuations of the battery, ready to read.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.config import RunConfig
from llmfs.data import make_get_batch, prepare
from llmfs.device import get_device, set_seed
from llmfs.paths import configs_dir, figures_dir
from llmfs.reference import GPT, write_eval_report

console = Console()

perplexity_from_loss = resolve("15_evaluation", "perplexity_from_loss")
bits_per_byte = resolve("15_evaluation", "bits_per_byte")
run_prompt_battery = resolve("15_evaluation", "run_prompt_battery")
generate_with_cache = resolve("14_inference", "generate_with_cache")


def main() -> None:
    cfg_dev = get_device()
    cfg = RunConfig.from_yaml(configs_dir() / "tiny_char.yaml")
    dataset = prepare(cfg, quiet=True)
    cfg.model.vocab_size = dataset.vocab_size

    from llmfs.train import load_checkpoint

    model = GPT(cfg.model)
    path = cfg.run_dir / "best.pt"
    trained = path.exists()
    if trained:
        load_checkpoint(path, model, map_location="cpu")
        console.print(f"[green]Model loaded from {path}[/green]\n")
    else:
        console.print(
            "[yellow]There is no trained model. Train one with "
            "`llmfs train --config tiny_char` and come back.[/yellow]\n"
        )
    model = model.to(cfg_dev.device).eval()
    get_batch = make_get_batch(dataset, cfg, cfg_dev.device)

    # ------------------------------------------------------------- 1. metrics
    console.rule("[bold]1. Perplexity[/bold]")
    losses = {}
    with torch.no_grad():
        for split in ("train", "val"):
            total, n = 0.0, 0
            for _ in range(50):
                x, y = get_batch(split, 32)
                total += float(model(x, y)[1])
                n += 1
            losses[split] = total / n

    floor = math.log(cfg.model.vocab_size)
    table = Table(header_style="bold")
    table.add_column("split")
    table.add_column("loss", justify="right")
    table.add_column("perplexity", justify="right")
    table.add_column("")
    table.add_row(
        "chance (the floor)",
        f"{floor:.4f}",
        f"{perplexity_from_loss(floor):.1f}",
        "[dim]what an untrained model gets[/dim]",
    )
    for split, loss in losses.items():
        table.add_row(split, f"{loss:.4f}", f"{perplexity_from_loss(loss):.2f}", "")
    console.print(table)

    gap = losses["val"] - losses["train"]
    console.print(
        f"\n  train/val gap: [bold]{gap:+.4f}[/bold] "
        + (
            "[green](small: it is not memorizing)[/green]"
            if gap < 0.3
            else "[yellow](it is starting to overfit)[/yellow]"
        )
        + f"\n  improvement over chance: from perplexity {perplexity_from_loss(floor):.0f} to "
        f"[bold]{perplexity_from_loss(losses['val']):.1f}[/bold]"
    )

    # ------------------------------------------------------------- 2. bits per byte
    console.rule("[bold]2. Bits per byte[/bold]")
    n_tokens = 50 * 32 * cfg.model.context_length
    # At character level, a token is roughly a byte
    bpb = bits_per_byte(losses["val"] * n_tokens, n_tokens, n_tokens)

    table2 = Table(header_style="bold")
    table2.add_column("compressor")
    table2.add_column("bits/byte", justify="right")
    table2.add_row("uncompressed", "8.00")
    table2.add_row("gzip (English text)", "~2.50")
    table2.add_row("[bold]your model[/bold]", f"[bold]{bpb:.3f}[/bold]")
    table2.add_row("the best LLMs", "0.60 - 0.80")
    console.print(table2)
    console.print(
        f"\nYour model would compress the text to [bold]1/{8 / bpb:.1f}[/bold] of its size.\n"
        "[dim]It is not an analogy: a language model IS a compressor, and this equivalence\n"
        "between prediction and compression comes from Shannon (1948).\n\n"
        "Unlike perplexity, this metric CAN be compared between models with different\n"
        "tokenizers, because it normalizes by bytes and not by tokens.[/dim]"
    )

    # ------------------------------------------------------------- 3. the battery
    console.rule("[bold]3. The qualitative battery[/bold]")

    def generate(prompt: str) -> str:
        set_seed(1234)
        ids = torch.tensor([dataset.encode(prompt)], device=cfg_dev.device)
        if ids.shape[1] == 0:
            ids = torch.zeros(1, 1, dtype=torch.long, device=cfg_dev.device)
        out = generate_with_cache(model, ids, 120, temperature=0.8, top_k=40)
        return dataset.decode(out[0].tolist())

    console.print(
        "[dim]Careful: the model is trained on Shakespeare at character level, not on\n"
        "TinyStories. Prompts in modern English are out of distribution for it, so the\n"
        "continuations will be odd. The exercise of READING them is the same.[/dim]\n"
    )
    battery = run_prompt_battery(generate)
    for case in battery:
        console.print(
            Panel(
                case["completion"].replace("\n", " ")[:200],
                title=f"{case['tests']}  |  prompt: {case['prompt'][:40]}...",
                border_style="cyan",
            )
        )

    console.print(
        "\n[bold]Read the six and judge three things separately:[/bold]\n"
        "  1. GRAMMAR:    are the sentences well built?\n"
        "  2. COHERENCE:  does it contradict itself? does it keep what it said before?\n"
        "  3. CREATIVITY: does it contribute anything or does it repeat templates?\n\n"
        "[dim]The TinyStories paper separates these three capabilities because they appear\n"
        "at different scales: a 1M model already does decent grammar, coherence needs more,\n"
        "and creativity more still. It is not a single ladder.[/dim]"
    )

    # ------------------------------------------------------------- the report
    metrics = {
        "loss (train)": losses["train"],
        "loss (val)": losses["val"],
        "perplexity (val)": perplexity_from_loss(losses["val"]),
        "bits per byte": bpb,
        "chance floor": floor,
        "vocabulary": cfg.model.vocab_size,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    target = write_eval_report(cfg.run_dir / "eval_report.md", metrics, battery, cfg.summary())
    console.print(f"\n[green]report saved to {target}[/green]")

    # ------------------------------------------------------------- figure
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    names = ["chance", "train", "val"]
    values = [
        perplexity_from_loss(floor),
        perplexity_from_loss(losses["train"]),
        perplexity_from_loss(losses["val"]),
    ]
    colors = ["tab:red", "tab:blue", "tab:green"]
    left.bar(names, values, color=colors)
    for i, v in enumerate(values):
        left.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    left.set_yscale("log")
    left.set_ylabel("perplexity (log scale)")
    left.set_title("How much the model hesitates")
    left.grid(alpha=0.3, axis="y")

    comps = ["uncom-\npressed", "gzip", "your\nmodel", "best\nLLMs"]
    bits = [8.0, 2.5, bpb, 0.7]
    right.bar(comps, bits, color=["gray", "tab:orange", "tab:green", "tab:blue"])
    for i, v in enumerate(bits):
        right.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    right.set_ylabel("bits per byte")
    right.set_title("A language model is a compressor")
    right.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig_path = figures_dir() / "15_evaluation.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    console.print(f"[green]figure saved to {fig_path}[/green]")


if __name__ == "__main__":
    main()
