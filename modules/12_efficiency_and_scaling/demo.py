"""Module 12 demo: measures your real MFU and reproduces the Chinchilla result.

    llmfs demo 12

Three experiments:
  1. The MFU really measured, training a few steps and timing them.
  2. How the MFU changes with the batch size: where the bottleneck is.
  3. Chinchilla: the formula against the real historical models.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.config import ModelConfig
from llmfs.device import get_device, set_seed
from llmfs.paths import figures_dir
from llmfs.reference import GPT

console = Console()

model_flops_per_token = resolve("12_efficiency_and_scaling", "model_flops_per_token")
compute_mfu = resolve("12_efficiency_and_scaling", "compute_mfu")
chinchilla = resolve("12_efficiency_and_scaling", "chinchilla_optimal_allocation")


def peak_tflops(cfg) -> float:
    if cfg.kind == "cuda":
        n = cfg.name.lower()
        return 51.6 if "2060" in n else 71.0 if "3090" in n else 312.0 if "a100" in n else 50.0
    return 14.0 if cfg.kind == "mps" else 1.0


def breakdown_experiment():
    console.rule("[bold]1. Where the FLOPs go[/bold]")
    table = Table(header_style="bold")
    table.add_column("context", justify="right")
    table.add_column("matmul", justify="right")
    table.add_column("attention", justify="right")
    table.add_column("total", justify="right")
    table.add_column("% attention", justify="right")

    contexts = [128, 512, 1024, 2048, 4096, 8192]
    percentages = []
    for T in contexts:
        f = model_flops_per_token(ModelConfig(context_length=T))
        pct = 100 * f["attention"] / f["total"]
        percentages.append(pct)
        table.add_row(
            str(T),
            f"{f['matmul'] / 1e6:.1f}M",
            f"{f['attention'] / 1e6:.1f}M",
            f"{f['total'] / 1e6:.1f}M",
            f"{pct:.0f}%",
            style="bold" if T == 512 else "",
        )
    console.print(table)
    console.print(
        "[dim]The bold row is our config. The matmul term does not move: it only depends on\n"
        "the size of the model. The attention one grows linearly with the context, and from\n"
        "2048 on it already dominates. That is why lengthening the context is expensive.[/dim]"
    )
    return contexts, percentages


def mfu_experiment(cfg):
    console.rule("[bold]2. The real MFU of your machine[/bold]")
    peak = peak_tflops(cfg)
    console.print(
        f"Really training a few steps and timing them.\n"
        f"Estimated peak of your hardware: [bold]{peak} TFLOPS[/bold]\n"
    )

    table = Table(header_style="bold")
    table.add_column("batch", justify="right")
    table.add_column("tokens/step", justify="right")
    table.add_column("ms/step", justify="right")
    table.add_column("tokens/s", justify="right")
    table.add_column("MFU", justify="right")

    batches = [1, 2, 4, 8, 16]
    mfus, tps_list = [], []
    m_cfg = ModelConfig(vocab_size=512, context_length=256)
    fpt = model_flops_per_token(m_cfg)["total"]

    set_seed(0)
    model = GPT(m_cfg).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for b in batches:
        x = torch.randint(0, 512, (b, 256), device=cfg.device)
        y = torch.randint(0, 512, (b, 256), device=cfg.device)
        try:
            for _ in range(3):  # warm up
                _, p = model(x, y)
                opt.zero_grad(); p.backward(); opt.step()
            cfg.synchronize()
            t0 = time.perf_counter()
            n = 8
            for _ in range(n):
                _, p = model(x, y)
                opt.zero_grad(); p.backward(); opt.step()
            cfg.synchronize()
            dt = (time.perf_counter() - t0) / n
        except RuntimeError as exc:
            console.print(f"  batch {b}: {exc.__class__.__name__} (out of memory)")
            break

        tps = b * 256 / dt
        mfu = compute_mfu(tps, fpt, peak)
        mfus.append(mfu); tps_list.append(tps)
        table.add_row(str(b), f"{b * 256:,}", f"{dt * 1000:.1f}", f"{tps / 1e3:.1f}k", f"{mfu:.1%}")

    console.print(table)
    console.print(
        "\n[bold]The MFU rises with the batch[/bold] and then flattens out. That flattening\n"
        "point is where you stop being limited by kernel launches and start being limited by\n"
        "the computation itself.\n\n"
        "[dim]Reference values: 0.4-0.5 on large, well-optimized models, 0.1-0.2 on models of\n"
        "this size, and below 0.05 it is worth looking at the dataloader. With a small model a\n"
        "low MFU is unavoidable: the matrices are not enough to saturate the tensor cores.[/dim]"
    )
    return batches[: len(mfus)], mfus


def chinchilla_experiment():
    console.rule("[bold]3. Chinchilla against the real models[/bold]")

    models = [
        ("GPT-3", 175e9, 300e9),
        ("Gopher", 280e9, 300e9),
        ("Chinchilla", 70e9, 1.4e12),
        ("Llama-2 7B", 7e9, 2e12),
        ("Llama-3 8B", 8e9, 15e12),
        ("ours", 7.62e6, 500e6),
    ]

    table = Table(header_style="bold")
    table.add_column("model")
    table.add_column("parameters", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("tok/param", justify="right")
    table.add_column("Chinchilla optimum", justify="right")
    table.add_column("verdict")

    for name, N, D in models:
        C = 6 * N * D
        opt = chinchilla(C)
        ratio = D / N
        if ratio < 10:
            verdict = "[red]under-trained[/red]"
        elif ratio < 40:
            verdict = "[green]on the spot[/green]"
        else:
            verdict = "[yellow]over-trained on purpose[/yellow]"
        table.add_row(
            name,
            f"{N:.3g}",
            f"{D:.3g}",
            f"{ratio:.0f}",
            f"{opt['params']:.3g}",
            verdict,
        )
    console.print(table)

    chin = chinchilla(6 * 70e9 * 1.4e12)
    console.print(
        f"\n[bold]The check that gives confidence in the formula:[/bold]\n"
        f"  with Chinchilla's real budget (5.88e23 FLOPs), the formula predicts\n"
        f"  [bold]{chin['params'] / 1e9:.1f}B parameters[/bold] and the real model had "
        f"[bold]70B[/bold].\n"
    )
    console.print(
        "[bold]And the result that made the paper famous:[/bold] GPT-3 had 1.7 tokens per\n"
        "parameter when the optimum was 20. It was twelve times under-trained. Chinchilla,\n"
        "with the SAME compute as Gopher and a quarter of the parameters, won on almost every\n"
        "benchmark.\n\n"
        "[bold]Our model is at 65 tok/param, three times above the optimum.[/bold]\n"
        "[dim]It is deliberate, and for two reasons. Chinchilla optimizes TRAINING compute; if\n"
        "the model is going to be used a lot, a smaller and more heavily trained one is better,\n"
        "because inference is paid every time. Llama-3 takes this to the extreme with ~1,800\n"
        "tok/param. And besides, at this scale over-training costs hours, not months.[/dim]"
    )
    return models


def main() -> None:
    cfg = get_device()
    contexts, percentages = breakdown_experiment()
    batches, mfus = mfu_experiment(cfg)
    models = chinchilla_experiment()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].plot(contexts, percentages, marker="o")
    axes[0].axvline(512, color="gray", ls="--", lw=1, label="our config")
    axes[0].axhline(50, color="red", ls=":", lw=1, label="attention dominates")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("context length")
    axes[0].set_ylabel("% of the cost that is attention")
    axes[0].set_title("The context is expensive")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    if mfus:
        axes[1].plot(batches, [m * 100 for m in mfus], marker="o", color="tab:green")
    axes[1].set_xlabel("batch size")
    axes[1].set_ylabel("MFU (%)")
    axes[1].set_title(f"MFU measured on {cfg.kind}")
    axes[1].grid(alpha=0.3)

    for name, N, D in models:
        axes[2].scatter(N, D, s=60)
        axes[2].annotate(name, (N, D), fontsize=7, xytext=(4, 4), textcoords="offset points")
    xs = [1e6, 1e12]
    axes[2].plot(xs, [20 * x for x in xs], "k--", lw=1, label="Chinchilla optimum (20x)")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("parameters")
    axes[2].set_ylabel("training tokens")
    axes[2].set_title("Real models against Chinchilla")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    target = figures_dir() / "12_efficiency_and_scaling.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")


if __name__ == "__main__":
    main()
