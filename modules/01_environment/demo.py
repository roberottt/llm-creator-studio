"""Demo del modulo 01: mide tu hardware y estima cuanto va a durar la tirada final.

    llmfs demo 01

Hace tres cosas:
  1. Barre tamanyos de matriz y dtypes midiendo TFLOPS reales -> grafica.
  2. Calcula los FLOPs/token del modelo de 9M y del juguete.
  3. Con esos dos numeros, estima el tiempo de entrenamiento a varias MFU.

El punto pedagogico: ver con tus ojos que el pico de la ficha tecnica no se alcanza casi
nunca, y que las matrices pequenyas dejan la GPU muerta de risa.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # sin ventana: esto tiene que correr por SSH y en CI

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


def barrido_de_tamanyos(cfg) -> dict[str, list[float]]:
    """Mide TFLOPS para varios tamanyos y dtypes."""
    dtypes: list[tuple[str, torch.dtype]] = [("fp32", torch.float32)]
    if cfg.kind == "cuda":
        dtypes.append(("fp16", torch.float16))
        if cfg.supports_bf16:
            dtypes.append(("bf16", torch.bfloat16))
    elif cfg.kind == "mps":
        dtypes.append(("fp16", torch.float16))

    resultados: dict[str, list[float]] = {}
    for nombre, dtype in dtypes:
        serie: list[float] = []
        for size in SIZES:
            try:
                serie.append(measure_matmul_tflops(cfg=cfg, size=size, dtype=dtype))
            except RuntimeError as exc:  # OOM en la 2060 con 4096 en fp32, por ejemplo
                console.print(f"[dim]{nombre} {size}x{size}: {exc.__class__.__name__}[/dim]")
                serie.append(float("nan"))
        resultados[nombre] = serie
        console.print(
            f"  {nombre}: " + "  ".join(f"{s}={v:.1f}" for s, v in zip(SIZES, serie) if v == v)
        )
    return resultados


def grafica(resultados: dict[str, list[float]], cfg) -> str:
    fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))

    for nombre, serie in resultados.items():
        izq.plot(SIZES, serie, marker="o", label=nombre)
    izq.set_xscale("log", base=2)
    izq.set_xlabel("lado de la matriz")
    izq.set_ylabel("TFLOPS efectivos")
    izq.set_title(f"Rendimiento de matmul - {cfg.name}")
    izq.grid(alpha=0.3)
    izq.legend()

    # Eficiencia relativa al mejor resultado: donde se pierde el hardware.
    mejor = max(v for serie in resultados.values() for v in serie if v == v)
    for nombre, serie in resultados.items():
        der.plot(SIZES, [100 * v / mejor for v in serie], marker="o", label=nombre)
    der.axhline(100, color="gray", ls="--", lw=1)
    der.set_xscale("log", base=2)
    der.set_xlabel("lado de la matriz")
    der.set_ylabel("% del mejor resultado")
    der.set_title("Las matrices pequenyas desaprovechan la GPU")
    der.grid(alpha=0.3)
    der.legend()

    fig.tight_layout()
    destino = figures_dir() / "01_hardware.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    return str(destino)


def estimaciones(cfg, pico: float) -> None:
    tabla = Table(title="Cuanto tardaria cada config", header_style="bold")
    tabla.add_column("config")
    tabla.add_column("FLOPs/token", justify="right")
    tabla.add_column("tokens", justify="right")
    for mfu in (0.10, 0.20, 0.40):
        tabla.add_column(f"MFU {mfu:.0%}", justify="right")

    for nombre in ("tiny_char", "tinystories_9m"):
        cfg_run = RunConfig.from_yaml(configs_dir() / f"{nombre}.yaml")
        m = cfg_run.model
        fpt = transformer_flops_per_token(
            n_layers=m.n_layers,
            d_model=m.d_model,
            d_ff=m.d_ff,
            context_length=m.context_length,
            vocab_size=m.vocab_size,
            n_ffn_matrices=3 if m.activation == "swiglu" else 2,
        )
        fila = [nombre, f"{fpt / 1e6:.1f}M", f"{cfg_run.train.max_tokens / 1e6:.0f}M"]
        for mfu in (0.10, 0.20, 0.40):
            tps = estimate_tokens_per_second(pico, fpt, mfu=mfu)
            horas = cfg_run.train.max_tokens / tps / 3600
            fila.append(f"{horas:.2f} h" if horas >= 0.05 else f"{horas * 60:.1f} min")
        tabla.add_row(*fila)

    console.print(tabla)
    console.print(
        "\n[dim]Recuerda: un modelo de 9M rara vez pasa de MFU 0,2. Las matrices de\n"
        "320x320 son demasiado pequenyas para saturar los tensor cores. La columna del\n"
        "40% es lo que verias con un modelo de varios miles de millones.[/dim]"
    )


def main() -> None:
    cfg = get_device()
    console.rule("[bold]hardware detectado[/bold]")
    console.print(cfg.summary())

    console.rule("[bold]midiendo matmul[/bold]")
    console.print("[dim](calentando y cronometrando, unos segundos)[/dim]")
    resultados = barrido_de_tamanyos(cfg)

    pico = max(v for serie in resultados.values() for v in serie if v == v)
    console.print(f"\n[bold]Pico medido: {pico:.1f} TFLOPS[/bold]")
    if cfg.kind == "cuda" and "2060" in cfg.name:
        console.print(
            "[dim]La ficha tecnica de la RTX 2060 dice 51,6 TFLOPS fp16. La diferencia\n"
            "con lo que acabas de medir es la vida real.[/dim]"
        )

    destino = grafica(resultados, cfg)
    console.print(f"[green]figura guardada en {destino}[/green]")

    console.rule("[bold]estimacion de tiempos[/bold]")
    estimaciones(cfg, pico)


if __name__ == "__main__":
    main()
