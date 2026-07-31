"""Demo del modulo 15: evalua tu modelo y genera el informe.

    llmfs demo 15

Usa el modelo entrenado en el modulo 13 si existe. Produce `eval_report.md` con las
metricas y las seis continuaciones de la bateria, listo para leer.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.config import RunConfig
from llmfs.data import hacer_get_batch, preparar
from llmfs.device import get_device, set_seed
from llmfs.paths import configs_dir, figures_dir
from llmfs.reference import GPT, write_eval_report

console = Console()

perplexity_from_loss = resolve("15_evaluacion", "perplexity_from_loss")
bits_per_byte = resolve("15_evaluacion", "bits_per_byte")
run_prompt_battery = resolve("15_evaluacion", "run_prompt_battery")
generate_with_cache = resolve("14_inferencia", "generate_with_cache")


def main() -> None:
    cfg_dev = get_device()
    cfg = RunConfig.from_yaml(configs_dir() / "tiny_char.yaml")
    dataset = preparar(cfg, quiet=True)
    cfg.model.vocab_size = dataset.vocab_size

    from llmfs.train import cargar_checkpoint

    modelo = GPT(cfg.model)
    ruta = cfg.run_dir / "best.pt"
    entrenado = ruta.exists()
    if entrenado:
        cargar_checkpoint(ruta, modelo, map_location="cpu")
        console.print(f"[green]Modelo cargado de {ruta}[/green]\n")
    else:
        console.print(
            "[yellow]No hay modelo entrenado. Entrena uno con "
            "`llmfs train --config tiny_char` y vuelve.[/yellow]\n"
        )
    modelo = modelo.to(cfg_dev.device).eval()
    get_batch = hacer_get_batch(dataset, cfg, cfg_dev.device)

    # ------------------------------------------------------------- 1. metricas
    console.rule("[bold]1. Perplejidad[/bold]")
    perdidas = {}
    with torch.no_grad():
        for split in ("train", "val"):
            total, n = 0.0, 0
            for _ in range(50):
                x, y = get_batch(split, 32)
                total += float(modelo(x, y)[1])
                n += 1
            perdidas[split] = total / n

    suelo = math.log(cfg.model.vocab_size)
    tabla = Table(header_style="bold")
    tabla.add_column("conjunto")
    tabla.add_column("perdida", justify="right")
    tabla.add_column("perplejidad", justify="right")
    tabla.add_column("")
    tabla.add_row("azar (el suelo)", f"{suelo:.4f}", f"{perplexity_from_loss(suelo):.1f}",
                  "[dim]lo que saca un modelo sin entrenar[/dim]")
    for split, perdida in perdidas.items():
        tabla.add_row(split, f"{perdida:.4f}", f"{perplexity_from_loss(perdida):.2f}", "")
    console.print(tabla)

    brecha = perdidas["val"] - perdidas["train"]
    console.print(
        f"\n  brecha train/val: [bold]{brecha:+.4f}[/bold] "
        + ("[green](poca: no esta memorizando)[/green]" if brecha < 0.3
           else "[yellow](empieza a sobreajustar)[/yellow]")
        + f"\n  mejora sobre el azar: de perplejidad {perplexity_from_loss(suelo):.0f} a "
        f"[bold]{perplexity_from_loss(perdidas['val']):.1f}[/bold]"
    )

    # ------------------------------------------------------------- 2. bits por byte
    console.rule("[bold]2. Bits por byte[/bold]")
    n_tokens = 50 * 32 * cfg.model.context_length
    # A nivel de caracter, un token es aproximadamente un byte
    bpb = bits_per_byte(perdidas["val"] * n_tokens, n_tokens, n_tokens)

    tabla2 = Table(header_style="bold")
    tabla2.add_column("compresor")
    tabla2.add_column("bits/byte", justify="right")
    tabla2.add_row("sin comprimir", "8.00")
    tabla2.add_row("gzip (texto en ingles)", "~2.50")
    tabla2.add_row("[bold]tu modelo[/bold]", f"[bold]{bpb:.3f}[/bold]")
    tabla2.add_row("los mejores LLM", "0.60 - 0.80")
    console.print(tabla2)
    console.print(
        f"\nTu modelo comprimiria el texto a [bold]1/{8/bpb:.1f}[/bold] de su tamanyo.\n"
        "[dim]No es una analogia: un modelo de lenguaje ES un compresor, y esta\n"
        "equivalencia entre prediccion y compresion viene de Shannon (1948).\n\n"
        "A diferencia de la perplejidad, esta metrica SI se puede comparar entre modelos\n"
        "con tokenizadores distintos, porque normaliza por bytes y no por tokens.[/dim]"
    )

    # ------------------------------------------------------------- 3. la bateria
    console.rule("[bold]3. La bateria cualitativa[/bold]")

    def generar(prompt: str) -> str:
        set_seed(1234)
        ids = torch.tensor([dataset.encode(prompt)], device=cfg_dev.device)
        if ids.shape[1] == 0:
            ids = torch.zeros(1, 1, dtype=torch.long, device=cfg_dev.device)
        salida = generate_with_cache(modelo, ids, 120, temperature=0.8, top_k=40)
        return dataset.decode(salida[0].tolist())

    console.print(
        "[dim]Ojo: el modelo esta entrenado sobre Shakespeare a nivel caracter, no sobre\n"
        "TinyStories. Los prompts en ingles moderno le quedan fuera de distribucion, asi que\n"
        "las continuaciones seran raras. El ejercicio de LEERLAS es el mismo.[/dim]\n"
    )
    bateria = run_prompt_battery(generar)
    for caso in bateria:
        console.print(
            Panel(
                caso["completion"].replace("\n", " ")[:200],
                title=f"{caso['que_prueba']}  |  prompt: {caso['prompt'][:40]}...",
                border_style="cyan",
            )
        )

    console.print(
        "\n[bold]Lee las seis y juzga tres cosas por separado:[/bold]\n"
        "  1. GRAMATICA:  ¿las frases estan bien construidas?\n"
        "  2. COHERENCIA: ¿se contradice? ¿mantiene lo que dijo antes?\n"
        "  3. CREATIVIDAD: ¿aporta algo o repite plantillas?\n\n"
        "[dim]El paper de TinyStories separa estas tres capacidades porque aparecen a\n"
        "escalas distintas: un modelo de 1M ya hace gramatica decente, la coherencia\n"
        "necesita mas, y la creatividad todavia mas. No es una escalera unica.[/dim]"
    )

    # ------------------------------------------------------------- el informe
    metricas = {
        "perdida (train)": perdidas["train"],
        "perdida (val)": perdidas["val"],
        "perplejidad (val)": perplexity_from_loss(perdidas["val"]),
        "bits por byte": bpb,
        "suelo del azar": suelo,
        "vocabulario": cfg.model.vocab_size,
        "parametros": sum(p.numel() for p in modelo.parameters()),
    }
    destino = write_eval_report(
        cfg.run_dir / "eval_report.md", metricas, bateria, cfg.summary()
    )
    console.print(f"\n[green]informe guardado en {destino}[/green]")

    # ------------------------------------------------------------- grafica
    fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))

    nombres = ["azar", "train", "val"]
    valores = [perplexity_from_loss(suelo), perplexity_from_loss(perdidas["train"]),
               perplexity_from_loss(perdidas["val"])]
    colores = ["tab:red", "tab:blue", "tab:green"]
    izq.bar(nombres, valores, color=colores)
    for i, v in enumerate(valores):
        izq.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    izq.set_yscale("log")
    izq.set_ylabel("perplejidad (escala log)")
    izq.set_title("Cuanto duda el modelo")
    izq.grid(alpha=0.3, axis="y")

    comps = ["sin\ncomprimir", "gzip", "tu\nmodelo", "mejores\nLLM"]
    bits = [8.0, 2.5, bpb, 0.7]
    der.bar(comps, bits, color=["gray", "tab:orange", "tab:green", "tab:blue"])
    for i, v in enumerate(bits):
        der.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    der.set_ylabel("bits por byte")
    der.set_title("Un modelo de lenguaje es un compresor")
    der.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    ruta_fig = figures_dir() / "15_evaluacion.png"
    fig.savefig(ruta_fig, dpi=120)
    plt.close(fig)
    console.print(f"[green]figura guardada en {ruta_fig}[/green]")


if __name__ == "__main__":
    main()
