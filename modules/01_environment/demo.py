"""Demo for module 01: measure your hardware and estimate how long the final run will take.

    llmfs demo 01

It does three things:
  1. Sweeps matrix sizes and dtypes measuring real TFLOPS -> plot.
  2. Computes the FLOPs/token of the 9M model and of the toy.
  3. With those two numbers, estimates the training time at several MFUs.

The pedagogical point: seeing with your own eyes that the spec-sheet peak is almost never
reached, and that small matrices leave the GPU twiddling its thumbs.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no window: this has to run over SSH and in CI

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.config import RunConfig
from llmfs.device import get_device
from llmfs.paths import configs_dir, figures_dir

console = Console()

measure_matmul_tflops = resolve("01_environment", "measure_matmul_tflops")
transformer_flops_per_token = resolve("01_environment", "transformer_flops_per_token")
estimate_tokens_per_second = resolve("01_environment", "estimate_tokens_per_second")

SIZES = [128, 256, 512, 1024, 2048, 4096]


def size_sweep(cfg) -> dict[str, list[float]]:
    """Measures TFLOPS for several sizes and dtypes."""
    dtypes: list[tuple[str, torch.dtype]] = [("fp32", torch.float32)]
    if cfg.kind == "cuda":
        dtypes.append(("fp16", torch.float16))
        if cfg.supports_bf16:
            dtypes.append(("bf16", torch.bfloat16))
    elif cfg.kind == "mps":
        dtypes.append(("fp16", torch.float16))

    results: dict[str, list[float]] = {}
    for name, dtype in dtypes:
        series: list[float] = []
        for size in SIZES:
            try:
                series.append(measure_matmul_tflops(cfg=cfg, size=size, dtype=dtype))
            except RuntimeError as exc:  # OOM on the 2060 with 4096 in fp32, for example
                console.print(f"[dim]{name} {size}x{size}: {exc.__class__.__name__}[/dim]")
                series.append(float("nan"))
        results[name] = series
        console.print(
            f"  {name}: " + "  ".join(f"{s}={v:.1f}" for s, v in zip(SIZES, series) if v == v)
        )
    return results


def plot(results: dict[str, list[float]], cfg) -> str:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, series in results.items():
        left.plot(SIZES, series, marker="o", label=name)
    left.set_xscale("log", base=2)
    left.set_xlabel("matrix side")
    left.set_ylabel("effective TFLOPS")
    left.set_title(f"Matmul throughput - {cfg.name}")
    left.grid(alpha=0.3)
    left.legend()

    # Efficiency relative to the best result: where the hardware is lost.
    best = max(v for series in results.values() for v in series if v == v)
    for name, series in results.items():
        right.plot(SIZES, [100 * v / best for v in series], marker="o", label=name)
    right.axhline(100, color="gray", ls="--", lw=1)
    right.set_xscale("log", base=2)
    right.set_xlabel("matrix side")
    right.set_ylabel("% of the best result")
    right.set_title("Small matrices waste the GPU")
    right.grid(alpha=0.3)
    right.legend()

    fig.tight_layout()
    target = figures_dir() / "01_hardware.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    return str(target)


def estimates(cfg, peak: float) -> None:
    table = Table(title="How long each config would take", header_style="bold")
    table.add_column("config")
    table.add_column("FLOPs/token", justify="right")
    table.add_column("tokens", justify="right")
    for mfu in (0.10, 0.20, 0.40):
        table.add_column(f"MFU {mfu:.0%}", justify="right")

    for name in ("tiny_char", "tinystories_9m"):
        run_cfg = RunConfig.from_yaml(configs_dir() / f"{name}.yaml")
        m = run_cfg.model
        fpt = transformer_flops_per_token(
            n_layers=m.n_layers,
            d_model=m.d_model,
            d_ff=m.d_ff,
            context_length=m.context_length,
            vocab_size=m.vocab_size,
            n_ffn_matrices=3 if m.activation == "swiglu" else 2,
        )
        row = [name, f"{fpt / 1e6:.1f}M", f"{run_cfg.train.max_tokens / 1e6:.0f}M"]
        for mfu in (0.10, 0.20, 0.40):
            tps = estimate_tokens_per_second(peak, fpt, mfu=mfu)
            hours = run_cfg.train.max_tokens / tps / 3600
            row.append(f"{hours:.2f} h" if hours >= 0.05 else f"{hours * 60:.1f} min")
        table.add_row(*row)

    console.print(table)
    console.print(
        "\n[dim]Remember: a 9M model rarely goes above MFU 0.2. 320x320 matrices are too\n"
        "small to saturate the tensor cores. The 40% column is what you would see with a\n"
        "model of several billion parameters.[/dim]"
    )


def main() -> None:
    cfg = get_device()
    console.rule("[bold]detected hardware[/bold]")
    console.print(cfg.summary())

    console.rule("[bold]measuring matmul[/bold]")
    console.print("[dim](warming up and timing, a few seconds)[/dim]")
    results = size_sweep(cfg)

    peak = max(v for series in results.values() for v in series if v == v)
    console.print(f"\n[bold]Measured peak: {peak:.1f} TFLOPS[/bold]")
    if cfg.kind == "cuda" and "2060" in cfg.name:
        console.print(
            "[dim]The RTX 2060's spec sheet says 51.6 TFLOPS fp16. The difference with\n"
            "what you just measured is real life.[/dim]"
        )

    target = plot(results, cfg)
    console.print(f"[green]figure saved to {target}[/green]")

    console.rule("[bold]time estimates[/bold]")
    estimates(cfg, peak)


if __name__ == "__main__":
    main()
