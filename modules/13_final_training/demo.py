"""Demo del modulo 13: la comprobacion de los 30 segundos, y un entrenamiento completo.

    llmfs demo 13              overfit a un batch + entrenamiento corto de verdad
    llmfs demo 13 --solo-test  solo el overfit, sin entrenar

Este demo entrena de verdad: tarda alrededor de un minuto en MPS.
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


def experimento_overfit(cfg_dev, cfg, dataset, get_batch):
    console.rule("[bold]1. La comprobacion de los 30 segundos[/bold]")
    console.print(
        "Se coge UN batch y se le da al modelo una y otra vez. Un modelo sano lo memoriza\n"
        "y la perdida baja practicamente a cero. Si no baja, hay un bug, y lo sabes ahora\n"
        "en vez de dentro de cuatro horas.\n"
    )

    set_seed(0)
    modelo = GPT(cfg.model).to(cfg_dev.device)
    x, y = get_batch("train", 4)

    inicio = time.perf_counter()
    historial = overfit_single_batch(modelo, x, y, steps=300, lr=3e-3)
    transcurrido = time.perf_counter() - inicio

    suelo = math.log(cfg.model.vocab_size)
    tabla = Table(header_style="bold")
    tabla.add_column("paso", justify="right")
    tabla.add_column("perdida", justify="right")
    tabla.add_column("")
    for paso in (0, 10, 50, 100, 200, 299):
        nota = ""
        if paso == 0:
            nota = f"<- deberia rondar ln({cfg.model.vocab_size}) = {suelo:.3f}"
        elif paso == 299:
            nota = "<- deberia estar casi en cero"
        tabla.add_row(str(paso), f"{historial[paso]:.4f}", nota)
    console.print(tabla)

    ok_inicio = abs(historial[0] - suelo) < 0.3
    ok_final = historial[-1] < 0.1
    console.print(
        f"\n  arranca en ln(V): {'[green]si[/green]' if ok_inicio else '[red]NO[/red]'}   "
        f"memoriza el batch: {'[green]si[/green]' if ok_final else '[red]NO[/red]'}   "
        f"[dim]({transcurrido:.1f} s)[/dim]\n"
    )
    if ok_inicio and ok_final:
        console.print(
            "[green]El modelo esta sano.[/green] Se puede lanzar el entrenamiento largo.\n"
            "[dim]Si la perdida hubiera bajado a cero en cinco pasos, habria que sospechar\n"
            "de una fuga de informacion: targets sin desplazar respecto a la entrada.[/dim]"
        )
    else:
        console.print("[red]Algo va mal. NO lances el entrenamiento largo todavia.[/red]")
    return historial


def experimento_eta():
    console.rule("[bold]2. El ETA[/bold]")
    tabla = Table(header_style="bold")
    tabla.add_column("segundos", justify="right")
    tabla.add_column("formateado")
    tabla.add_column("cuando lo veras")
    for s, contexto in [
        (45, "el final de una tirada corta"),
        (125, "el demo de un modulo"),
        (3725, "tiny_char completo en CPU"),
        (14400, "TinyStories en la 2060"),
        (float("inf"), "los primeros pasos, sin datos aun"),
    ]:
        tabla.add_row(f"{s:,.0f}" if math.isfinite(s) else "inf", format_eta(s), contexto)
    console.print(tabla)
    console.print(
        "[dim]Los valores no finitos dan '?' a proposito: es mas honesto que inventarse un\n"
        "numero cuando todavia no hay datos suficientes para estimar.[/dim]"
    )


def experimento_entrenar(cfg_dev, cfg, dataset, get_batch):
    console.rule("[bold]3. Un entrenamiento completo, de verdad[/bold]")

    from llmfs.train import Trainer

    cfg.train.max_tokens = 600 * cfg.tokens_per_step
    cfg.train.eval_interval = 200
    cfg.train.log_interval = 100
    cfg.train.sample_interval = 300
    cfg.name = "demo13"

    set_seed(1337)
    modelo = GPT(cfg.model)

    muestras: list[tuple[int, str]] = []

    def muestra(step: int) -> str:
        modelo.eval()
        inicio = torch.tensor([dataset.encode("\n")], dtype=torch.long, device=cfg_dev.device)
        with torch.no_grad():
            salida = modelo.generate(inicio, max_new_tokens=150, temperature=0.8, top_k=40)
        modelo.train()
        texto = dataset.decode(salida[0].tolist())
        muestras.append((step, texto))
        return texto

    trainer = Trainer(cfg, modelo, get_batch, device=cfg_dev, on_sample=muestra)
    muestra(0)  # antes de entrenar nada
    estado = trainer.entrenar(console=console)

    console.print("\n[bold]Como ha ido cambiando el texto:[/bold]")
    for step, texto in muestras:
        console.print(
            Panel(
                texto.replace("\n", " ")[:200],
                title=f"paso {step}",
                border_style="red" if step == 0 else "green",
            )
        )
    console.print(
        "\n[dim]Ese fichero de muestras, leido de arriba abajo cuando termina una tirada\n"
        "larga, es el modelo aprendiendo a escribir. Es mas informativo que la curva de\n"
        "perdida: un salto de 1.6 a 1.5 no dice mucho, pero ver que ha empezado a cerrar\n"
        "los parentesis, si.[/dim]"
    )
    return estado, trainer


def main() -> None:
    solo_test = "--solo-test" in sys.argv
    cfg_dev = get_device()
    cfg = RunConfig.from_yaml(configs_dir() / "tiny_char.yaml")

    dataset = prepare(cfg, quiet=True)
    cfg.model.vocab_size = dataset.vocab_size
    get_batch = make_get_batch(dataset, cfg, cfg_dev.device)

    historial = experimento_overfit(cfg_dev, cfg, dataset, get_batch)
    experimento_eta()

    curvas = None
    if not solo_test:
        estado, trainer = experimento_entrenar(cfg_dev, cfg, dataset, get_batch)
        from llmfs.train import TrainingLogger

        curvas = TrainingLogger(cfg.run_dir, reanudando=True).read_csv()
        console.print(
            f"\n[bold]Resultado:[/bold] {estado.step:,} pasos, "
            f"mejor perdida de validacion {estado.best_val_loss:.4f}\n"
            f"[dim]El suelo era ln({cfg.model.vocab_size}) = "
            f"{math.log(cfg.model.vocab_size):.4f}.[/dim]"
        )

    fig, ejes = plt.subplots(1, 2 if curvas else 1, figsize=(12 if curvas else 6, 4.5), squeeze=False)
    ejes = ejes[0]

    ejes[0].plot(historial, color="tab:red")
    ejes[0].axhline(0.1, color="gray", ls="--", lw=1, label="umbral de 'memorizado'")
    ejes[0].set_yscale("log")
    ejes[0].set_xlabel("paso")
    ejes[0].set_ylabel("perdida (escala log)")
    ejes[0].set_title("Overfit a un solo batch")
    ejes[0].grid(alpha=0.3)
    ejes[0].legend(fontsize=8)

    if curvas:
        tr = [(f["step"], f["train_loss"]) for f in curvas if f.get("train_loss")]
        val = [(f["step"], f["val_loss"]) for f in curvas if f.get("val_loss")]
        if tr:
            ejes[1].plot([s for s, _ in tr], [v for _, v in tr], lw=1, alpha=0.7, label="entrenamiento")
        if val:
            ejes[1].plot([s for s, _ in val], [v for _, v in val], marker="o", ms=4, label="validacion")
        ejes[1].set_xlabel("paso")
        ejes[1].set_ylabel("perdida (nats)")
        ejes[1].set_title("Entrenamiento de verdad")
        ejes[1].grid(alpha=0.3)
        ejes[1].legend(fontsize=8)

    fig.tight_layout()
    destino = figures_dir() / "13_final_training.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")


if __name__ == "__main__":
    main()
