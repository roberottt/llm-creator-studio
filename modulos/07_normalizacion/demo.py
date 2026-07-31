"""Demo del modulo 07: por que pre-norm y no post-norm, medido.

    llmfs demo 07

Tres experimentos:
  1. Sin residuales ni normalizacion, el gradiente se desvanece con la profundidad.
     Con residuales, no. Se mide la norma del gradiente que llega a la entrada.
  2. Pre-norm frente a post-norm, con redes de 4 a 64 capas.
  3. LayerNorm frente a RMSNorm: cuanto cuesta cada una y cuanto se parecen.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")  # sin ventana: esto tiene que correr por SSH y en CI

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

layer_norm = resolve("07_normalizacion", "layer_norm")
RMSNorm = resolve("07_normalizacion", "RMSNorm")
prenorm_residual = resolve("07_normalizacion", "prenorm_residual")

D = 64
PROFUNDIDADES = [4, 8, 16, 32, 64]


def bloque(d: int, escala: float = 1.0) -> nn.Module:
    """Un bloque cualquiera. Lo unico que importa es que encoja un poco su salida."""
    capa = nn.Linear(d, d)
    with torch.no_grad():
        capa.weight.mul_(escala)
        capa.bias.zero_()
    return capa


def gradiente_en_la_entrada(modo: str, n_capas: int, semilla: int = 0) -> float:
    """Norma del gradiente que llega a la entrada tras atravesar `n_capas` bloques."""
    set_seed(semilla)
    x = torch.randn(4, 16, D, requires_grad=True)
    capas = [bloque(D, escala=0.5) for _ in range(n_capas)]
    normas = [RMSNorm(D) for _ in range(n_capas)]

    h = x
    for capa, norm in zip(capas, normas):
        if modo == "nada":
            h = capa(h)
        elif modo == "solo norma":
            h = capa(norm(h))
        elif modo == "pre-norm":
            h = prenorm_residual(h, capa, norm)
        elif modo == "post-norm":
            h = postnorm_residual(h, capa, norm)
    h.sum().backward()
    return float(x.grad.norm())


def experimento_residuales() -> dict[str, list[float]]:
    console.rule("[bold]1 y 2. Que le llega al gradiente segun la arquitectura[/bold]")
    console.print(
        "Se apilan N bloques identicos y se mide la NORMA DEL GRADIENTE que llega a la\n"
        "entrada. Si se acerca a cero, las primeras capas no reciben senyal y no aprenden.\n"
    )

    resultados: dict[str, list[float]] = {}
    tabla = Table(header_style="bold")
    tabla.add_column("capas", justify="right")
    modos = ("nada", "solo norma", "post-norm", "pre-norm")
    for modo in modos:
        tabla.add_column(modo, justify="right")

    for modo in modos:
        resultados[modo] = [gradiente_en_la_entrada(modo, n) for n in PROFUNDIDADES]

    for i, n in enumerate(PROFUNDIDADES):
        tabla.add_row(str(n), *[f"{resultados[m][i]:.3e}" for m in resultados])
    console.print(tabla)

    n = PROFUNDIDADES[-1]
    console.print(
        f"\n[bold]Con {n} capas, leido de izquierda a derecha:[/bold]\n\n"
        f"  [bold]nada[/bold] ({resultados['nada'][-1]:.2e}): capas lineales que encogen su "
        "salida, una detras de otra.\n"
        "  El gradiente se multiplica por un factor menor que 1 en cada capa y se desvanece\n"
        "  exponencialmente. Las primeras capas no reciben ninguna senyal.\n\n"
        f"  [bold]solo norma[/bold] ({resultados['solo norma'][-1]:.2e}): la misma red con "
        "RMSNorm delante de cada capa.\n"
        "  Aqui la comparacion ni siquiera se puede expresar como un cociente, porque el\n"
        "  caso anterior ha llegado a CERO exacto por underflow de la coma flotante.\n"
        "  [bold]La normalizacion sola ya rescata buena parte del problema[/bold],\n"
        "  porque devuelve la escala a 1 en cada paso y corta la cadena de factores.\n\n"
        f"  [bold]post-norm[/bold] ({resultados['post-norm'][-1]:.2e}) y "
        f"[bold]pre-norm[/bold] ({resultados['pre-norm'][-1]:.2e}): con residuales.\n"
        "  Pre-norm es el unico que CRECE con la profundidad en vez de decrecer, porque el\n"
        "  camino x -> x esta completamente libre y cada capa suma su contribucion.\n\n"
        "[dim]Conclusion honesta: normalizacion y residuales atacan el mismo problema por\n"
        "caminos distintos, y no son alternativas sino complementos. Lo que distingue a\n"
        "pre-norm no es evitar el desvanecimiento (eso ya lo hace la norma) sino dejar el\n"
        "camino residual sin ningun peaje.[/dim]"
    )
    return resultados


def experimento_normas(cfg) -> None:
    console.rule("[bold]3. LayerNorm frente a RMSNorm[/bold]")

    torch.manual_seed(0)
    x = torch.randn(8, 512, 320, device=cfg.device)
    rms = RMSNorm(320).to(cfg.device)

    def cronometra(fn, veces: int = 50) -> float:
        for _ in range(5):
            fn()
        cfg.synchronize()
        t0 = time.perf_counter()
        for _ in range(veces):
            fn()
        cfg.synchronize()
        return (time.perf_counter() - t0) / veces * 1000

    t_ln = cronometra(lambda: F.layer_norm(x, (320,)))
    t_rms = cronometra(lambda: rms(x))
    t_mio = cronometra(lambda: layer_norm(x))

    tabla = Table(header_style="bold")
    tabla.add_column("implementacion")
    tabla.add_column("ms por llamada", justify="right")
    tabla.add_column("parametros para d=320", justify="right")
    tabla.add_row("F.layer_norm (PyTorch)", f"{t_ln:.3f}", "640 (escala + sesgo)")
    tabla.add_row("tu layer_norm", f"{t_mio:.3f}", "640")
    tabla.add_row("RMSNorm", f"{t_rms:.3f}", "320 (solo escala)")
    console.print(tabla)

    console.print(
        "[dim]Los tiempos de aqui no dicen gran cosa: son operaciones limitadas por memoria\n"
        "y a esta escala domina el coste de lanzar el kernel. La ganancia real de RMSNorm\n"
        "que reportan Zhang y Sennrich (entre 7% y 64%) se mide sobre el entrenamiento\n"
        "entero, no sobre la capa aislada. Lo que si es un hecho aqui es la mitad de\n"
        "parametros.[/dim]\n"
    )

    # En que se diferencian de verdad. La correlacion NO sirve aqui: es invariante a
    # transformaciones afines, asi que da ~1 aunque una de las dos deje un desplazamiento
    # que la otra quita. Lo que hay que mirar es la MEDIA de la salida.
    console.print("Que hace cada una con datos ya centrados y con datos desplazados:\n")
    tabla2 = Table(header_style="bold")
    tabla2.add_column("entrada")
    tabla2.add_column("media tras LayerNorm", justify="right")
    tabla2.add_column("media tras RMSNorm", justify="right")
    tabla2.add_column("diferencia maxima", justify="right")

    torch.manual_seed(0)
    for etiqueta, entrada in [
        ("media 0", torch.randn(4, 16, 320)),
        ("media +5", torch.randn(4, 16, 320) + 5.0),
        ("media +50", torch.randn(4, 16, 320) + 50.0),
    ]:
        con_ln = layer_norm(entrada).detach()
        con_rms = RMSNorm(320)(entrada).detach()
        tabla2.add_row(
            etiqueta,
            f"{float(con_ln.mean()):+.4f}",
            f"{float(con_rms.mean()):+.4f}",
            f"{float((con_ln - con_rms).abs().max()):.3f}",
        )
    console.print(tabla2)
    console.print(
        "[dim]Con los datos ya centrados las dos hacen practicamente lo mismo, y por eso se\n"
        "puede prescindir de restar la media: dentro de una red, las activaciones suelen\n"
        "estar centradas.\n\n"
        "Con un desplazamiento grande si divergen: LayerNorm lo elimina y RMSNorm lo\n"
        "conserva. Que en la practica eso no perjudique es un resultado EMPIRICO, no un\n"
        "teorema. Zhang y Sennrich lo comprobaron entrenando, no demostrandolo.[/dim]"
    )


def main() -> None:
    cfg = get_device()
    resultados = experimento_residuales()
    experimento_normas(cfg)

    fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))

    for modo, valores in resultados.items():
        izq.plot(PROFUNDIDADES, valores, marker="o", label=modo)
    izq.set_yscale("log")
    izq.set_xlabel("numero de capas apiladas")
    izq.set_ylabel("norma del gradiente en la entrada")
    izq.set_title("El gradiente sobrevive gracias al residual")
    izq.grid(alpha=0.3)
    izq.legend()

    x = torch.linspace(-4, 4, 200).unsqueeze(0).repeat(1, 1)
    entrada = torch.tensor([[2.0, 8.0, 4.0, 6.0]])
    etiquetas = ["x original", "LayerNorm", "RMSNorm"]
    valores = [
        entrada[0].tolist(),
        layer_norm(entrada)[0].tolist(),
        RMSNorm(4)(entrada)[0].detach().tolist(),
    ]
    ancho = 0.25
    posiciones = torch.arange(4).float()
    for i, (etiqueta, v) in enumerate(zip(etiquetas, valores)):
        der.bar(posiciones + i * ancho, v, ancho, label=etiqueta)
    der.axhline(0, color="black", lw=0.8)
    der.set_xticks(posiciones + ancho)
    der.set_xticklabels([f"dim {i}" for i in range(4)])
    der.set_title("El ejemplo de TEORIA.md, dimension a dimension")
    der.grid(alpha=0.3, axis="y")
    der.legend(fontsize=8)

    fig.tight_layout()
    destino = figures_dir() / "07_normalizacion.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")


if __name__ == "__main__":
    main()
