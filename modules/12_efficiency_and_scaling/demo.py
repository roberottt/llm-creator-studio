"""Demo del modulo 12: mide tu MFU real y reproduce el resultado de Chinchilla.

    llmfs demo 12

Tres experimentos:
  1. La MFU medida de verdad, entrenando unos pasos y cronometrando.
  2. Como cambia la MFU con el tamanyo del batch: donde esta el cuello de botella.
  3. Chinchilla: la formula frente a los modelos historicos reales.
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


def pico_tflops(cfg) -> float:
    if cfg.kind == "cuda":
        n = cfg.name.lower()
        return 51.6 if "2060" in n else 71.0 if "3090" in n else 312.0 if "a100" in n else 50.0
    return 14.0 if cfg.kind == "mps" else 1.0


def experimento_desglose():
    console.rule("[bold]1. Donde se van los FLOPs[/bold]")
    tabla = Table(header_style="bold")
    tabla.add_column("contexto", justify="right")
    tabla.add_column("matmul", justify="right")
    tabla.add_column("atencion", justify="right")
    tabla.add_column("total", justify="right")
    tabla.add_column("% atencion", justify="right")

    contextos = [128, 512, 1024, 2048, 4096, 8192]
    porcentajes = []
    for T in contextos:
        f = model_flops_per_token(ModelConfig(context_length=T))
        pct = 100 * f["attention"] / f["total"]
        porcentajes.append(pct)
        tabla.add_row(
            str(T),
            f"{f['matmul'] / 1e6:.1f}M",
            f"{f['attention'] / 1e6:.1f}M",
            f"{f['total'] / 1e6:.1f}M",
            f"{pct:.0f}%",
            style="bold" if T == 512 else "",
        )
    console.print(tabla)
    console.print(
        "[dim]La fila en negrita es nuestra config. El termino de matmul no se mueve: solo\n"
        "depende del tamanyo del modelo. El de atencion crece linealmente con el contexto,\n"
        "y a partir de 2048 ya domina. Por eso alargar el contexto sale caro.[/dim]"
    )
    return contextos, porcentajes


def experimento_mfu(cfg):
    console.rule("[bold]2. La MFU real de tu maquina[/bold]")
    pico = pico_tflops(cfg)
    console.print(
        f"Entrenando de verdad unos pasos y cronometrando.\n"
        f"Pico estimado de tu hardware: [bold]{pico} TFLOPS[/bold]\n"
    )

    tabla = Table(header_style="bold")
    tabla.add_column("batch", justify="right")
    tabla.add_column("tokens/paso", justify="right")
    tabla.add_column("ms/paso", justify="right")
    tabla.add_column("tokens/s", justify="right")
    tabla.add_column("MFU", justify="right")

    batches = [1, 2, 4, 8, 16]
    mfus, tps_list = [], []
    m_cfg = ModelConfig(vocab_size=512, context_length=256)
    fpt = model_flops_per_token(m_cfg)["total"]

    set_seed(0)
    modelo = GPT(m_cfg).to(cfg.device)
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-4)

    for b in batches:
        x = torch.randint(0, 512, (b, 256), device=cfg.device)
        y = torch.randint(0, 512, (b, 256), device=cfg.device)
        try:
            for _ in range(3):  # calentar
                _, p = modelo(x, y)
                opt.zero_grad(); p.backward(); opt.step()
            cfg.synchronize()
            t0 = time.perf_counter()
            n = 8
            for _ in range(n):
                _, p = modelo(x, y)
                opt.zero_grad(); p.backward(); opt.step()
            cfg.synchronize()
            dt = (time.perf_counter() - t0) / n
        except RuntimeError as exc:
            console.print(f"  batch {b}: {exc.__class__.__name__} (sin memoria)")
            break

        tps = b * 256 / dt
        mfu = compute_mfu(tps, fpt, pico)
        mfus.append(mfu); tps_list.append(tps)
        tabla.add_row(str(b), f"{b * 256:,}", f"{dt * 1000:.1f}", f"{tps / 1e3:.1f}k", f"{mfu:.1%}")

    console.print(tabla)
    console.print(
        f"\n[bold]La MFU sube con el batch[/bold] y luego se estanca. Ese punto de\n"
        "estancamiento es donde dejas de estar limitado por el lanzamiento de kernels y\n"
        "pasas a estarlo por el calculo.\n\n"
        "[dim]Valores de referencia: 0.4-0.5 en modelos grandes bien optimizados, 0.1-0.2 en\n"
        "modelos de este tamanyo, y por debajo de 0.05 conviene mirar el dataloader. Con un\n"
        "modelo pequenyo la MFU baja es inevitable: las matrices no dan para saturar los\n"
        "tensor cores.[/dim]"
    )
    return batches[: len(mfus)], mfus


def experimento_chinchilla():
    console.rule("[bold]3. Chinchilla frente a los modelos reales[/bold]")

    modelos = [
        ("GPT-3", 175e9, 300e9),
        ("Gopher", 280e9, 300e9),
        ("Chinchilla", 70e9, 1.4e12),
        ("Llama-2 7B", 7e9, 2e12),
        ("Llama-3 8B", 8e9, 15e12),
        ("el nuestro", 7.62e6, 500e6),
    ]

    tabla = Table(header_style="bold")
    tabla.add_column("modelo")
    tabla.add_column("parametros", justify="right")
    tabla.add_column("tokens", justify="right")
    tabla.add_column("tok/param", justify="right")
    tabla.add_column("optimo Chinchilla", justify="right")
    tabla.add_column("veredicto")

    for nombre, N, D in modelos:
        C = 6 * N * D
        opt = chinchilla(C)
        ratio = D / N
        if ratio < 10:
            veredicto = "[red]infraentrenado[/red]"
        elif ratio < 40:
            veredicto = "[green]en el punto[/green]"
        else:
            veredicto = "[yellow]sobreentrenado a proposito[/yellow]"
        tabla.add_row(
            nombre,
            f"{N:.3g}",
            f"{D:.3g}",
            f"{ratio:.0f}",
            f"{opt['params']:.3g}",
            veredicto,
        )
    console.print(tabla)

    chin = chinchilla(6 * 70e9 * 1.4e12)
    console.print(
        f"\n[bold]La comprobacion que da confianza en la formula:[/bold]\n"
        f"  con el presupuesto real de Chinchilla (5.88e23 FLOPs), la formula predice\n"
        f"  [bold]{chin['params'] / 1e9:.1f}B parametros[/bold] y el modelo real tenia "
        f"[bold]70B[/bold].\n"
    )
    console.print(
        "[bold]Y el resultado que hizo famoso al paper:[/bold] GPT-3 tenia 1,7 tokens por\n"
        "parametro cuando lo optimo eran 20. Estaba doce veces infraentrenado. Chinchilla,\n"
        "con el MISMO computo que Gopher y la cuarta parte de parametros, gano en casi\n"
        "todos los benchmarks.\n\n"
        "[bold]Nuestro modelo esta a 65 tok/param, tres veces por encima del optimo.[/bold]\n"
        "[dim]Es deliberado, y por dos razones. Chinchilla optimiza el computo de\n"
        "ENTRENAMIENTO; si el modelo se va a usar mucho, conviene uno mas pequenyo y mas\n"
        "entrenado, porque la inferencia se paga cada vez. Llama-3 lleva esto al extremo con\n"
        "~1.800 tok/param. Y ademas, a esta escala entrenar de mas cuesta horas, no meses.[/dim]"
    )
    return modelos


def main() -> None:
    cfg = get_device()
    contextos, porcentajes = experimento_desglose()
    batches, mfus = experimento_mfu(cfg)
    modelos = experimento_chinchilla()

    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.2))

    ejes[0].plot(contextos, porcentajes, marker="o")
    ejes[0].axvline(512, color="gray", ls="--", lw=1, label="nuestra config")
    ejes[0].axhline(50, color="red", ls=":", lw=1, label="la atencion domina")
    ejes[0].set_xscale("log", base=2)
    ejes[0].set_xlabel("longitud del contexto")
    ejes[0].set_ylabel("% del coste que es atencion")
    ejes[0].set_title("El contexto sale caro")
    ejes[0].grid(alpha=0.3)
    ejes[0].legend(fontsize=8)

    if mfus:
        ejes[1].plot(batches, [m * 100 for m in mfus], marker="o", color="tab:green")
    ejes[1].set_xlabel("tamanyo de batch")
    ejes[1].set_ylabel("MFU (%)")
    ejes[1].set_title(f"MFU medida en {cfg.kind}")
    ejes[1].grid(alpha=0.3)

    for nombre, N, D in modelos:
        ejes[2].scatter(N, D, s=60)
        ejes[2].annotate(nombre, (N, D), fontsize=7, xytext=(4, 4), textcoords="offset points")
    xs = [1e6, 1e12]
    ejes[2].plot(xs, [20 * x for x in xs], "k--", lw=1, label="optimo Chinchilla (20x)")
    ejes[2].set_xscale("log")
    ejes[2].set_yscale("log")
    ejes[2].set_xlabel("parametros")
    ejes[2].set_ylabel("tokens de entrenamiento")
    ejes[2].set_title("Modelos reales frente a Chinchilla")
    ejes[2].grid(alpha=0.3)
    ejes[2].legend(fontsize=8)

    fig.tight_layout()
    destino = figures_dir() / "12_efficiency_and_scaling.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")


if __name__ == "__main__":
    main()
