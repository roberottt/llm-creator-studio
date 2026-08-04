"""Module 13 demo: the 30-second check, and a complete training run.

    llmfs demo 13              overfit a batch + a short real training run
    llmfs demo 13 --only-test  only the overfit, without training

This demo really trains: it takes around a minute on MPS.
"""

from __future__ import annotations

import math
import sys
import time

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
from llmfs.reference import GPT

console = Console()

overfit_single_batch = resolve("13_final_training", "overfit_single_batch")
format_eta = resolve("13_final_training", "format_eta")


def overfit_experiment(cfg_dev, cfg, dataset, get_batch):
    console.rule("[bold]1. The 30-second check[/bold]")
    console.print(
        "ONE batch is taken and given to the model over and over. A healthy model memorizes\n"
        "it and the loss drops practically to zero. If it does not drop, there is a bug, and\n"
        "you know it now instead of in four hours.\n"
    )

    set_seed(0)
    model = GPT(cfg.model).to(cfg_dev.device)
    x, y = get_batch("train", 4)

    start = time.perf_counter()
    history = overfit_single_batch(model, x, y, steps=300, lr=3e-3)
    elapsed = time.perf_counter() - start

    floor = math.log(cfg.model.vocab_size)
    table = Table(header_style="bold")
    table.add_column("step", justify="right")
    table.add_column("loss", justify="right")
    table.add_column("")
    for step in (0, 10, 50, 100, 200, 299):
        note = ""
        if step == 0:
            note = f"<- should be around ln({cfg.model.vocab_size}) = {floor:.3f}"
        elif step == 299:
            note = "<- should be almost at zero"
        table.add_row(str(step), f"{history[step]:.4f}", note)
    console.print(table)

    ok_start = abs(history[0] - floor) < 0.3
    ok_end = history[-1] < 0.1
    console.print(
        f"\n  starts at ln(V): {'[green]yes[/green]' if ok_start else '[red]NO[/red]'}   "
        f"memorizes the batch: {'[green]yes[/green]' if ok_end else '[red]NO[/red]'}   "
        f"[dim]({elapsed:.1f} s)[/dim]\n"
    )
    if ok_start and ok_end:
        console.print(
            "[green]The model is healthy.[/green] The long training run can be launched.\n"
            "[dim]If the loss had dropped to zero in five steps, you would have to suspect an\n"
            "information leak: targets not shifted relative to the input.[/dim]"
        )
    else:
        console.print("[red]Something is wrong. Do NOT launch the long training run yet.[/red]")
    return history


def eta_experiment():
    console.rule("[bold]2. The ETA[/bold]")
    table = Table(header_style="bold")
    table.add_column("seconds", justify="right")
    table.add_column("formatted")
    table.add_column("when you will see it")
    for s, context in [
        (45, "the end of a short run"),
        (125, "a module's demo"),
        (3725, "the full tiny_char on CPU"),
        (14400, "TinyStories on the 2060"),
        (float("inf"), "the first steps, with no data yet"),
    ]:
        table.add_row(f"{s:,.0f}" if math.isfinite(s) else "inf", format_eta(s), context)
    console.print(table)
    console.print(
        "[dim]Non-finite values give '?' on purpose: it is more honest than making up a\n"
        "number when there is not yet enough data to estimate.[/dim]"
    )


def training_experiment(cfg_dev, cfg, dataset, get_batch):
    console.rule("[bold]3. A complete training run, for real[/bold]")

    from llmfs.train import Trainer

    cfg.train.max_tokens = 600 * cfg.tokens_per_step
    cfg.train.eval_interval = 200
    cfg.train.log_interval = 100
    cfg.train.sample_interval = 300
    cfg.name = "demo13"

    set_seed(1337)
    model = GPT(cfg.model)

    samples: list[tuple[int, str]] = []

    def sample(step: int) -> str:
        model.eval()
        start = torch.tensor([dataset.encode("\n")], dtype=torch.long, device=cfg_dev.device)
        with torch.no_grad():
            out = model.generate(start, max_new_tokens=150, temperature=0.8, top_k=40)
        model.train()
        text = dataset.decode(out[0].tolist())
        samples.append((step, text))
        return text

    trainer = Trainer(cfg, model, get_batch, device=cfg_dev, on_sample=sample)
    sample(0)  # before training anything
    state = trainer.train(console=console)

    console.print("\n[bold]How the text has been changing:[/bold]")
    for step, text in samples:
        console.print(
            Panel(
                text.replace("\n", " ")[:200],
                title=f"step {step}",
                border_style="red" if step == 0 else "green",
            )
        )
    console.print(
        "\n[dim]That samples file, read top to bottom when a long run finishes, is the model\n"
        "learning to write. It is more informative than the loss curve: a jump from 1.6 to\n"
        "1.5 does not say much, but seeing that it has started closing its parentheses\n"
        "does.[/dim]"
    )
    return state, trainer


def main() -> None:
    only_test = "--only-test" in sys.argv
    cfg_dev = get_device()
    cfg = RunConfig.from_yaml(configs_dir() / "tiny_char.yaml")

    dataset = prepare(cfg, quiet=True)
    cfg.model.vocab_size = dataset.vocab_size
    get_batch = make_get_batch(dataset, cfg, cfg_dev.device)

    history = overfit_experiment(cfg_dev, cfg, dataset, get_batch)
    eta_experiment()

    curves = None
    if not only_test:
        state, trainer = training_experiment(cfg_dev, cfg, dataset, get_batch)
        from llmfs.train import TrainingLogger

        curves = TrainingLogger(cfg.run_dir, resuming=True).read_csv()
        console.print(
            f"\n[bold]Result:[/bold] {state.step:,} steps, "
            f"best validation loss {state.best_val_loss:.4f}\n"
            f"[dim]The floor was ln({cfg.model.vocab_size}) = "
            f"{math.log(cfg.model.vocab_size):.4f}.[/dim]"
        )

    fig, axes = plt.subplots(
        1, 2 if curves else 1, figsize=(12 if curves else 6, 4.5), squeeze=False
    )
    axes = axes[0]

    axes[0].plot(history, color="tab:red")
    axes[0].axhline(0.1, color="gray", ls="--", lw=1, label="'memorized' threshold")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss (log scale)")
    axes[0].set_title("Overfit on a single batch")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    if curves:
        tr = [(f["step"], f["train_loss"]) for f in curves if f.get("train_loss")]
        val = [(f["step"], f["val_loss"]) for f in curves if f.get("val_loss")]
        if tr:
            axes[1].plot([s for s, _ in tr], [v for _, v in tr], lw=1, alpha=0.7, label="training")
        if val:
            axes[1].plot(
                [s for s, _ in val], [v for _, v in val], marker="o", ms=4, label="validation"
            )
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("loss (nats)")
        axes[1].set_title("Real training")
        axes[1].grid(alpha=0.3)
        axes[1].legend(fontsize=8)

    fig.tight_layout()
    target = figures_dir() / "13_final_training.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")


if __name__ == "__main__":
    main()
