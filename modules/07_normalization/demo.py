"""Demo for module 07: why pre-norm and not post-norm, measured.

    llmfs demo 07

Three experiments:
  1. Without residuals or normalization, the gradient vanishes with depth. With residuals,
     it does not. The norm of the gradient reaching the input is measured.
  2. Pre-norm against post-norm, with networks of 4 to 64 layers.
  3. LayerNorm against RMSNorm: what each costs and how similar they are.
"""

from __future__ import annotations

import time

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
from llmfs.reference import postnorm_residual

console = Console()

layer_norm = resolve("07_normalization", "layer_norm")
RMSNorm = resolve("07_normalization", "RMSNorm")
prenorm_residual = resolve("07_normalization", "prenorm_residual")

D = 64
DEPTHS = [4, 8, 16, 32, 64]


def block(d: int, scale: float = 1.0) -> nn.Module:
    """Any old block. All that matters is that it shrinks its output a little."""
    layer = nn.Linear(d, d)
    with torch.no_grad():
        layer.weight.mul_(scale)
        layer.bias.zero_()
    return layer


def gradient_at_the_input(mode: str, n_layers: int, seed: int = 0) -> float:
    """Norm of the gradient reaching the input after crossing `n_layers` blocks."""
    set_seed(seed)
    x = torch.randn(4, 16, D, requires_grad=True)
    layers = [block(D, scale=0.5) for _ in range(n_layers)]
    norms = [RMSNorm(D) for _ in range(n_layers)]

    h = x
    for layer, norm in zip(layers, norms):
        if mode == "nothing":
            h = layer(h)
        elif mode == "norm only":
            h = layer(norm(h))
        elif mode == "pre-norm":
            h = prenorm_residual(h, layer, norm)
        elif mode == "post-norm":
            h = postnorm_residual(h, layer, norm)
    h.sum().backward()
    return float(x.grad.norm())


def residuals_experiment() -> dict[str, list[float]]:
    console.rule("[bold]1 and 2. What reaches the gradient, by architecture[/bold]")
    console.print(
        "N identical blocks get stacked and the NORM OF THE GRADIENT reaching the input is\n"
        "measured. If it approaches zero, the first layers receive no signal and do not\n"
        "learn.\n"
    )

    results: dict[str, list[float]] = {}
    table = Table(header_style="bold")
    table.add_column("layers", justify="right")
    modes = ("nothing", "norm only", "post-norm", "pre-norm")
    for mode in modes:
        table.add_column(mode, justify="right")

    for mode in modes:
        results[mode] = [gradient_at_the_input(mode, n) for n in DEPTHS]

    for i, n in enumerate(DEPTHS):
        table.add_row(str(n), *[f"{results[m][i]:.3e}" for m in results])
    console.print(table)

    n = DEPTHS[-1]
    console.print(
        f"\n[bold]With {n} layers, read left to right:[/bold]\n\n"
        f"  [bold]nothing[/bold] ({results['nothing'][-1]:.2e}): linear layers that shrink "
        "their output, one after another.\n"
        "  The gradient is multiplied by a factor below 1 at every layer and vanishes\n"
        "  exponentially. The first layers receive no signal at all.\n\n"
        f"  [bold]norm only[/bold] ({results['norm only'][-1]:.2e}): the same network with "
        "RMSNorm in front of each layer.\n"
        "  Here the comparison cannot even be expressed as a ratio, because the previous\n"
        "  case reached exactly ZERO through floating-point underflow.\n"
        "  [bold]Normalization alone already rescues much of the problem[/bold],\n"
        "  because it returns the scale to 1 at every step and cuts the chain of factors.\n\n"
        f"  [bold]post-norm[/bold] ({results['post-norm'][-1]:.2e}) and "
        f"[bold]pre-norm[/bold] ({results['pre-norm'][-1]:.2e}): with residuals.\n"
        "  Pre-norm is the only one that GROWS with depth instead of shrinking, because the\n"
        "  path x -> x is completely clear and each layer adds its contribution.\n\n"
        "[dim]An honest conclusion: normalization and residuals attack the same problem by\n"
        "different routes, and they are not alternatives but complements. What sets\n"
        "pre-norm apart is not avoiding the vanishing (the norm already does that) but\n"
        "leaving the residual path with no toll at all.[/dim]"
    )
    return results


def norms_experiment(cfg) -> None:
    console.rule("[bold]3. LayerNorm against RMSNorm[/bold]")

    torch.manual_seed(0)
    x = torch.randn(8, 512, 320, device=cfg.device)
    rms = RMSNorm(320).to(cfg.device)

    def time_it(fn, times: int = 50) -> float:
        for _ in range(5):
            fn()
        cfg.synchronize()
        t0 = time.perf_counter()
        for _ in range(times):
            fn()
        cfg.synchronize()
        return (time.perf_counter() - t0) / times * 1000

    t_ln = time_it(lambda: F.layer_norm(x, (320,)))
    t_rms = time_it(lambda: rms(x))
    t_mine = time_it(lambda: layer_norm(x))

    table = Table(header_style="bold")
    table.add_column("implementation")
    table.add_column("ms per call", justify="right")
    table.add_column("parameters for d=320", justify="right")
    table.add_row("F.layer_norm (PyTorch)", f"{t_ln:.3f}", "640 (scale + bias)")
    table.add_row("your layer_norm", f"{t_mine:.3f}", "640")
    table.add_row("RMSNorm", f"{t_rms:.3f}", "320 (scale only)")
    console.print(table)

    console.print(
        "[dim]The timings here do not say much: these are memory-bound operations and at\n"
        "this scale the kernel launch cost dominates. The real RMSNorm gain Zhang and\n"
        "Sennrich report (between 7% and 64%) is measured over the whole training run, not\n"
        "over the isolated layer. What is a fact here is half the parameters.[/dim]\n"
    )

    # Where they really differ. Correlation is NO use here: it is invariant to affine
    # transformations, so it gives ~1 even if one of them leaves an offset the other
    # removes. What to look at is the MEAN of the output.
    console.print("What each does with already-centered data and with shifted data:\n")
    table2 = Table(header_style="bold")
    table2.add_column("input")
    table2.add_column("mean after LayerNorm", justify="right")
    table2.add_column("mean after RMSNorm", justify="right")
    table2.add_column("maximum difference", justify="right")

    torch.manual_seed(0)
    for label, inputs in [
        ("mean 0", torch.randn(4, 16, 320)),
        ("mean +5", torch.randn(4, 16, 320) + 5.0),
        ("mean +50", torch.randn(4, 16, 320) + 50.0),
    ]:
        with_ln = layer_norm(inputs).detach()
        with_rms = RMSNorm(320)(inputs).detach()
        table2.add_row(
            label,
            f"{float(with_ln.mean()):+.4f}",
            f"{float(with_rms.mean()):+.4f}",
            f"{float((with_ln - with_rms).abs().max()):.3f}",
        )
    console.print(table2)
    console.print(
        "[dim]With already-centered data the two do practically the same thing, which is why\n"
        "you can do without subtracting the mean: inside a network, activations are usually\n"
        "centered.\n\n"
        "With a large offset they do diverge: LayerNorm removes it and RMSNorm keeps it.\n"
        "That this does not hurt in practice is an EMPIRICAL result, not a theorem. Zhang\n"
        "and Sennrich checked it by training, not by proving it.[/dim]"
    )


def main() -> None:
    cfg = get_device()
    results = residuals_experiment()
    norms_experiment(cfg)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    for mode, values in results.items():
        left.plot(DEPTHS, values, marker="o", label=mode)
    left.set_yscale("log")
    left.set_xlabel("number of stacked layers")
    left.set_ylabel("gradient norm at the input")
    left.set_title("The gradient survives thanks to the residual")
    left.grid(alpha=0.3)
    left.legend()

    inputs = torch.tensor([[2.0, 8.0, 4.0, 6.0]])
    labels = ["original x", "LayerNorm", "RMSNorm"]
    values = [
        inputs[0].tolist(),
        layer_norm(inputs)[0].tolist(),
        RMSNorm(4)(inputs)[0].detach().tolist(),
    ]
    width = 0.25
    positions = torch.arange(4).float()
    for i, (label, v) in enumerate(zip(labels, values)):
        right.bar(positions + i * width, v, width, label=label)
    right.axhline(0, color="black", lw=0.8)
    right.set_xticks(positions + width)
    right.set_xticklabels([f"dim {i}" for i in range(4)])
    right.set_title("The THEORY.md example, dimension by dimension")
    right.grid(alpha=0.3, axis="y")
    right.legend(fontsize=8)

    fig.tight_layout()
    target = figures_dir() / "07_normalization.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")


if __name__ == "__main__":
    main()
