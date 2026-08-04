"""Demo for module 08: why a nonlinearity is needed, and which one.

    llmfs demo 08

Four experiments:
  1. The collapse: a network with no activations is equivalent to ONE matrix, measured.
  2. The curves of ReLU, GELU and Swish, and their derivatives. That is where ReLU's problem
     shows.
  3. The parameter split: two thirds of the model is FFN.
  4. SwiGLU against the classic FFN, training both with the same parameter budget.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no window: this has to run over SSH and in CI

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.device import get_device, set_seed
from llmfs.paths import figures_dir
from llmfs.reference import FeedForwardMLP

console = Console()

gelu = resolve("08_mlp_and_activations", "gelu")
swiglu_hidden_dim = resolve("08_mlp_and_activations", "swiglu_hidden_dim")
SwiGLU = resolve("08_mlp_and_activations", "SwiGLU")


def collapse_experiment() -> None:
    console.rule("[bold]1. A network with no activations is ONE layer[/bold]")
    console.print(
        "Five linear layers are stacked with nothing in between and compared against the\n"
        "product of their five matrices, which is a single matrix.\n"
    )

    set_seed(0)
    d = 32
    layers = [nn.Linear(d, d, bias=False) for _ in range(5)]
    x = torch.randn(4, d)

    with_layers = x
    for layer in layers:
        with_layers = layer(with_layers)

    # The product of the five matrices, as one
    equivalent = layers[0].weight
    for layer in layers[1:]:
        equivalent = layer.weight @ equivalent
    with_one = x @ equivalent.T

    error = float((with_layers - with_one).abs().max())
    console.print(
        f"  5 stacked layers vs 1 single matrix  ->  maximum difference: "
        f"[bold]{error:.2e}[/bold]\n"
        "  [dim](that is, zero apart from floating-point rounding)[/dim]\n"
    )

    # And now with an activation
    with_gelu = x
    for layer in layers:
        with_gelu = gelu(layer(with_gelu))
    gelu_error = float((with_gelu - with_one).abs().max())
    console.print(
        f"  The same 5 layers WITH GELU vs 1 matrix  ->  difference: "
        f"[bold]{gelu_error:.3f}[/bold]\n\n"
        "[bold]That is the whole argument.[/bold] Without a nonlinearity, a hundred layers\n"
        "are equivalent to one and all the depth collapses. Attention is a weighted average,\n"
        "that is, linear, so the FFN is the only thing stopping the whole Transformer from\n"
        "collapsing."
    )


def activations_experiment() -> tuple:
    console.rule("[bold]2. ReLU, GELU and Swish[/bold]")

    x = torch.linspace(-4, 4, 400, requires_grad=True)
    curves = {
        "ReLU": F.relu(x),
        "GELU": gelu(x),
        "Swish/SiLU": F.silu(x),
    }
    derivatives = {}
    for name, y in curves.items():
        g = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
        derivatives[name] = g.detach().clone()

    table = Table(header_style="bold")
    table.add_column("x", justify="right")
    for name in curves:
        table.add_column(f"{name}", justify="right")
        table.add_column(f"d{name}/dx", justify="right")

    for value in (-3.0, -1.0, 0.0, 1.0, 3.0):
        idx = int((value + 4) / 8 * 399)
        row = [f"{value:+.1f}"]
        for name in curves:
            row.append(f"{float(curves[name][idx]):+.4f}")
            row.append(f"{float(derivatives[name][idx]):+.4f}")
        table.add_row(*row)
    console.print(table)

    console.print(
        "[bold]Look at the dReLU/dx column on the negative rows: it is exactly 0.[/bold]\n"
        "A neuron that ends up always producing negative values stops receiving gradient\n"
        "forever. It is dead and there is no way to recover it.\n\n"
        "GELU and Swish have a small but NOT zero derivative in that zone, so a neuron in\n"
        "the negative region can still come back.\n\n"
        "[dim]Note too that GELU and Swish are almost the same curve. Swish comes from an\n"
        "automatic architecture search and GELU from a probabilistic argument, and they\n"
        "ended up in practically the same place.[/dim]"
    )
    return x.detach(), curves, derivatives


def parameters_experiment() -> None:
    console.rule("[bold]3. Where the parameters are[/bold]")

    table = Table(header_style="bold")
    table.add_column("d_model", justify="right")
    table.add_column("d_ff (SwiGLU)", justify="right")
    table.add_column("attention (4d²)", justify="right")
    table.add_column("FFN (3·d·d_ff)", justify="right")
    table.add_column("% FFN", justify="right")

    for d in (128, 320, 768, 4096):
        d_ff = swiglu_hidden_dim(d)
        attn = 4 * d * d
        ffn = 3 * d * d_ff
        table.add_row(
            str(d), str(d_ff), f"{attn:,}", f"{ffn:,}", f"{100 * ffn / (attn + ffn):.0f}%"
        )
    console.print(table)
    console.print(
        "[bold]Two thirds of the model is FFN, not attention.[/bold] When you read that a\n"
        "model has N parameters, most of them are here.\n\n"
        "[dim]And the 896 in the d_model=320 row is the final config's d_ff: "
        "ceil_64(2/3 · 4 · 320) = ceil_64(853.3) = 896.[/dim]"
    )

    console.print("\n[bold]SwiGLU's budget against the classic FFN:[/bold]")
    d = 320
    classic = 2 * d * 4 * d
    swiglu_equal = 3 * d * swiglu_hidden_dim(d)
    swiglu_unadjusted = 3 * d * 4 * d
    console.print(
        f"  classic FFN (2 matrices, d_ff=4d=1280)   : {classic:,}\n"
        f"  SwiGLU unadjusted (3 matrices, d_ff=1280): {swiglu_unadjusted:,}  "
        f"[red](+{100 * swiglu_unadjusted / classic - 100:.0f}%)[/red]\n"
        f"  SwiGLU with the 2/3 (d_ff=896)           : {swiglu_equal:,}  "
        f"[green]({100 * swiglu_equal / classic - 100:+.0f}%)[/green]\n\n"
        "[dim]That is the reason for the 2/3 factor: three matrices instead of two would\n"
        "cost 50% more, so the hidden size is cut to spend the same and compare fairly.[/dim]"
    )


def training_experiment(cfg) -> tuple[list[float], list[float]]:
    console.rule("[bold]4. SwiGLU against the classic FFN, at equal parameters[/bold]")

    set_seed(0)
    d, n = 256, 4096
    # A nonlinear function that is hard to approximate, so the two do not solve it instantly
    # and the comparison says something.
    x = torch.randn(n, d, device=cfg.device)
    target = (
        (3 * x[:, :1]).sin() * (2 * x[:, 1:2]).cos()
        + (x[:, 2:3] * x[:, 3:4]).tanh()
        + 0.5 * x[:, 4:5].abs()
    )

    results = {}
    parameters = {}
    for name, model in [
        ("classic FFN + GELU", FeedForwardMLP(d, 4 * d)),
        ("SwiGLU", SwiGLU(d, swiglu_hidden_dim(d))),
    ]:
        set_seed(1)
        model = nn.Sequential(model, nn.Linear(d, 1)).to(cfg.device)
        parameters[name] = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        history = []
        for _ in range(300):
            loss = F.mse_loss(model(x), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))
        results[name] = history
        console.print(
            f"  {name:<22} {parameters[name]:>9,} parameters  ->  "
            f"final loss {history[-1]:.4e}"
        )

    classic, swiglu = list(results.values())
    drift = 100 * abs(parameters["SwiGLU"] / parameters["classic FFN + GELU"] - 1)
    console.print(
        f"\n[dim]The parameters are not exactly equal: SwiGLU carries {drift:.0f}% more,\n"
        "because rounding d_ff to a multiple of 64 does not land exactly. The 2/3 equalizes\n"
        "the budgets asymptotically, not exactly.[/dim]\n"
    )

    difference = abs(swiglu[-1] - classic[-1]) / max(classic[-1], 1e-12)
    if difference < 0.1:
        console.print(
            "[bold yellow]The two losses end up within 10% of each other.[/bold yellow]\n"
            "With that difference and a single seed, this experiment does NOT distinguish\n"
            "between the two architectures. Saying one wins would be reading noise."
        )
    else:
        better = "SwiGLU" if swiglu[-1] < classic[-1] else "the classic FFN"
        console.print(
            f"On this particular task [bold]{better}[/bold] wins, by {100 * difference:.0f}%."
        )
    console.print(
        "\n[dim]And in any case: a toy experiment on an invented task proves NOTHING about\n"
        "language models. Shazeer (2020) tried every GLU variant training real transformers,\n"
        "and SwiGLU came out better consistently. His explanation of why, quoted literally\n"
        "from the paper: 'We offer no explanation as to why these architectures seem to\n"
        "work; we attribute their success, as all else, to divine benevolence.' It is one of\n"
        "the most used and least understood architecture decisions in the field.[/dim]"
    )
    return classic, swiglu


def main() -> None:
    cfg = get_device()
    collapse_experiment()
    x, curves, derivatives = activations_experiment()
    parameters_experiment()
    classic, swiglu = training_experiment(cfg)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for name, y in curves.items():
        axes[0].plot(x.numpy(), y.detach().numpy(), label=name)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].axvline(0, color="black", lw=0.6)
    axes[0].set_title("The activations")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    for name, g in derivatives.items():
        axes[1].plot(x.numpy(), g.numpy(), label=name)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_title("Their derivatives (ReLU is 0 in the negative zone)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].plot(classic, label="classic FFN + GELU")
    axes[2].plot(swiglu, label="SwiGLU")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("loss (MSE, log scale)")
    axes[2].set_title("At equal parameters")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    target = figures_dir() / "08_mlp_and_activations.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")


if __name__ == "__main__":
    main()
