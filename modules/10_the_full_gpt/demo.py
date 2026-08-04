"""Demo for module 10: the complete model, assembled and audited.

    llmfs demo 10

Five checks on the 8,933,440-parameter model:
  1. The parameter breakdown, component by component.
  2. The formula against the real count.
  3. The step-0 loss against ln(V): the bug detector.
  4. That it is genuinely causal, checked by changing a token.
  5. How much memory it takes and where the bulk is (spoiler: the logits).
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")  # no window: this has to run over SSH and in CI

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.config import ModelConfig, RunConfig
from llmfs.device import get_device, set_seed
from llmfs.paths import configs_dir, figures_dir

console = Console()

expected_param_count = resolve("10_the_full_gpt", "expected_param_count")
count_parameters = resolve("10_the_full_gpt", "count_parameters")
GPT = resolve("10_the_full_gpt", "GPT")


def main() -> None:
    dev_cfg = get_device()
    set_seed(1337)

    run = RunConfig.from_yaml(configs_dir() / "tinystories_9m.yaml")
    cfg = run.model

    console.rule("[bold]The model[/bold]")
    console.print(run.summary())

    # --------------------------------------------------------------- 1 and 2
    console.rule("[bold]1. Parameter breakdown[/bold]")
    model = GPT(cfg)
    counts = count_parameters(model)
    total = counts["total"]

    table = Table(header_style="bold")
    table.add_column("component")
    table.add_column("parameters", justify="right")
    table.add_column("%", justify="right")
    table.add_column("where they come from")

    details = {
        "embeddings": f"{cfg.vocab_size} x {cfg.d_model}",
        "attention": f"{cfg.n_layers} layers x 4 x {cfg.d_model}^2",
        "ffn": f"{cfg.n_layers} layers x 3 x {cfg.d_model} x {cfg.d_ff}",
        "norms": f"{2 * cfg.n_layers + 1} RMSNorm x {cfg.d_model}",
        "lm_head": "tied to the embeddings" if cfg.tie_embeddings else "untied",
        "other": "-",
    }
    for key in ("embeddings", "attention", "ffn", "norms", "lm_head", "other"):
        if counts[key] == 0 and key == "other":
            continue
        table.add_row(
            key, f"{counts[key]:,}", f"{100 * counts[key] / total:.1f}%", details[key]
        )
    table.add_section()
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total:,}[/bold]", "100%", "")
    table.add_row("non-embedding", f"{counts['non_embedding']:,}", "", "what Chinchilla uses")
    console.print(table)

    formula = expected_param_count(cfg)
    matches = formula == total == 8_933_440
    console.print(
        f"\n  formula : {formula:,}\n"
        f"  count   : {total:,}\n"
        f"  target  : 8,933,440\n"
        + (
            "  [bold green]All three match.[/bold green]"
            if matches
            else "  [bold red]They do NOT match.[/bold red]"
        )
    )

    if cfg.tie_embeddings:
        console.print(
            f"\n[dim]Without weight tying the model would have "
            f"{expected_param_count(ModelConfig(**{**cfg.__dict__, 'tie_embeddings': False})):,} "
            f"parameters: 15% more, just for not reusing a matrix you already have.[/dim]"
        )

    # --------------------------------------------------------------- 3
    console.rule("[bold]2. The step-0 loss[/bold]")
    model = model.to(dev_cfg.device).eval()
    sequence = torch.randint(0, cfg.vocab_size, (4, 65), device=dev_cfg.device)
    x, y = sequence[:, :-1], sequence[:, 1:]
    with torch.no_grad():
        _, loss = model(x, y)

    floor = math.log(cfg.vocab_size)
    drift = float(loss) - floor
    console.print(
        f"  loss of the untrained model : [bold]{float(loss):.4f}[/bold]\n"
        f"  ln({cfg.vocab_size})                    : {floor:.4f}\n"
        f"  drift                       : {drift:+.4f}\n"
    )
    if abs(drift) < 0.1:
        console.print(
            "[green]Correct.[/green] The model starts with no opinions, which is what it "
            "should do.\n"
            "[dim]This is the number you have to see at step 0 in module 11. If it comes out\n"
            "higher, the init is too aggressive. If it comes out lower, there is an\n"
            "information leak and the first thing to look at is the causal mask.[/dim]"
        )
    else:
        console.print("[red]Out of range: check the initialization or the mask.[/red]")

    console.print(
        "\n[dim]A note on method: the targets are SHIFTED by one token (x = seq[:, :-1],\n"
        "y = seq[:, 1:]). If you passed `model(idx, idx)`, at position t the model would see\n"
        "the token it has to predict and the loss would come out below ln(V). It looks like a\n"
        "model bug and it is a bug in whoever assembles the batch.[/dim]"
    )

    # --------------------------------------------------------------- 4
    console.rule("[bold]3. It is genuinely causal[/bold]")
    idx = torch.randint(0, cfg.vocab_size, (1, 12), device=dev_cfg.device)
    with torch.no_grad():
        original = model(idx)[0]
        modified = idx.clone()
        modified[0, 6] = (modified[0, 6] + 1) % cfg.vocab_size
        altered = model(modified)[0]

    diffs = (altered - original).abs().max(dim=-1).values[0]
    table2 = Table(header_style="bold")
    table2.add_column("position", justify="right")
    table2.add_column("maximum change in the logits", justify="right")
    table2.add_column("")
    for pos in range(12):
        mark = " <- changed token" if pos == 6 else ""
        colour = "green" if pos < 6 else "yellow"
        table2.add_row(str(pos), f"[{colour}]{float(diffs[pos]):.2e}[/{colour}]", mark)
    console.print(table2)
    console.print(
        "[dim]Before position 6 the change is exactly zero: those predictions cannot see\n"
        "token 6. From there on they do change. That is the causal mask working, and it is\n"
        "the most direct check there is that there is no leak.[/dim]"
    )

    # --------------------------------------------------------------- 5
    console.rule("[bold]4. Memory[/bold]")
    bytes_per_param = 4
    model_weights = total * bytes_per_param
    # AdamW stores two moments per parameter, both in fp32
    optimizer_states = total * 2 * bytes_per_param
    gradients = total * bytes_per_param

    B, T, V = run.train.batch_size, cfg.context_length, cfg.vocab_size
    logits_fp16 = B * T * V * 2
    logits_fp32 = B * T * V * 4

    table3 = Table(header_style="bold")
    table3.add_column("what")
    table3.add_column("MB", justify="right")
    table3.add_column("note")
    table3.add_row("model weights (fp32)", f"{model_weights / 1e6:.1f}", "")
    table3.add_row("gradients (fp32)", f"{gradients / 1e6:.1f}", "")
    table3.add_row("AdamW states", f"{optimizer_states / 1e6:.1f}", "two moments per parameter")
    table3.add_section()
    table3.add_row(
        "logits in fp16", f"{logits_fp16 / 1e6:.1f}", f"batch {B} x ctx {T} x vocab {V}"
    )
    table3.add_row(
        "+ its fp32 version", f"{logits_fp32 / 1e6:.1f}", "cross_entropy promotes"
    )
    table3.add_row(
        "+ its gradient", f"{logits_fp32 / 1e6:.1f}", "", style="bold"
    )
    console.print(table3)
    console.print(
        f"[bold]The logits alone take more than the model, the gradients and the optimizer "
        f"combined.[/bold]\n"
        f"({(logits_fp16 + 2 * logits_fp32) / 1e6:.0f} MB against "
        f"{(model_weights + gradients + optimizer_states) / 1e6:.0f} MB.)\n\n"
        "[dim]When you run out of memory on the 2060 in module 13, this is the first place\n"
        "to look, not the model's activations. The usual fix is computing the loss in chunks\n"
        "instead of materializing the whole tensor.[/dim]"
    )

    # --------------------------------------------------------------- plot
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    components = ["embeddings", "attention", "ffn", "norms"]
    values = [counts[c] for c in components]
    left.pie(
        values,
        labels=[f"{c}\n{v:,}" for c, v in zip(components, values)],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 8},
    )
    left.set_title(f"The {total:,} parameters")

    labels = ["weights", "gradients", "AdamW", "logits\n(fp16+fp32+grad)"]
    memories = [
        model_weights / 1e6,
        gradients / 1e6,
        optimizer_states / 1e6,
        (logits_fp16 + 2 * logits_fp32) / 1e6,
    ]
    colours = ["tab:blue"] * 3 + ["tab:red"]
    right.bar(labels, memories, color=colours)
    for i, v in enumerate(memories):
        right.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    right.set_ylabel("MB")
    right.set_title(f"Memory with batch {B} x ctx {T}")
    right.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    target = figures_dir() / "10_the_full_gpt.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")
    console.print(
        "\n[bold]This is where Part II ends.[/bold] You have the model assembled and "
        "audited.\nIn Part III it gets trained."
    )


if __name__ == "__main__":
    main()
