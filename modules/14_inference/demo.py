"""Demo del modulo 14: las estrategias de muestreo, y la cache que hace todo mas rapido.

    llmfs demo 14

Usa el modelo entrenado en el modulo 13 si existe (`checkpoints/tiny_char/best.pt`); si no,
uno sin entrenar (que sirve igual para medir velocidad, aunque el texto sea ruido).

Tres experimentos:
  1. Los filtros, con numeros pequenyos que puedes seguir a mano.
  2. El mismo prompt con distintas estrategias de muestreo. Se ve el bucle de greedy.
  3. La KV cache: salida identica y speedup medido.
"""

from __future__ import annotations

import math
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.config import RunConfig
from llmfs.device import get_device, set_seed
from llmfs.paths import configs_dir, figures_dir
from llmfs.reference import GPT

console = Console()

apply_repetition_penalty = resolve("14_inference", "apply_repetition_penalty")
top_k_filter = resolve("14_inference", "top_k_filter")
top_p_filter = resolve("14_inference", "top_p_filter")
KVCache = resolve("14_inference", "KVCache")
generate_with_cache = resolve("14_inference", "generate_with_cache")


def cargar_modelo(cfg_dev):
    """El modelo entrenado del modulo 13, o uno nuevo si no existe."""
    from llmfs.data import prepare
    from llmfs.train import load_checkpoint

    cfg = RunConfig.from_yaml(configs_dir() / "tiny_char.yaml")
    dataset = prepare(cfg, quiet=True)
    cfg.model.vocab_size = dataset.vocab_size

    modelo = GPT(cfg.model)
    ruta = cfg.run_dir / "best.pt"
    entrenado = False
    if ruta.exists():
        try:
            load_checkpoint(ruta, modelo, map_location="cpu")
            entrenado = True
        except Exception:
            pass

    return modelo.to(cfg_dev.device).eval(), dataset, cfg, entrenado


def experimento_filtros():
    console.rule("[bold]1. Los filtros, con numeros que puedes seguir a mano[/bold]")

    logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, -1.0, -3.0]])
    probs = F.softmax(logits, dim=-1)[0]

    tabla = Table(header_style="bold")
    tabla.add_column("token", justify="right")
    tabla.add_column("logit", justify="right")
    tabla.add_column("prob", justify="right")
    tabla.add_column("acumulada", justify="right")
    tabla.add_column("top-k=2")
    tabla.add_column("top-p=0.9")

    tk = top_k_filter(logits.clone(), 2)[0]
    tp = top_p_filter(logits.clone(), 0.9)[0]
    acum = 0.0
    for i in range(6):
        acum += float(probs[i])
        tabla.add_row(
            str(i),
            f"{float(logits[0, i]):+.1f}",
            f"{float(probs[i]):.3f}",
            f"{acum:.3f}",
            "[green]si[/green]" if torch.isfinite(tk[i]) else "[dim]no[/dim]",
            "[green]si[/green]" if torch.isfinite(tp[i]) else "[dim]no[/dim]",
        )
    console.print(tabla)
    console.print(
        f"[dim]top-p=0.9 deja {int(torch.isfinite(tp).sum())} candidatos: el conjunto mas "
        "pequenyo cuya masa EXCEDE 0.9.\nFijate en que el token que cruza el umbral entra: "
        "sin el, la masa se quedaria por debajo.[/dim]\n"
    )

    console.print("[bold]La temperatura, sobre los mismos logits:[/bold]")
    tabla2 = Table(header_style="bold")
    tabla2.add_column("T", justify="right")
    for i in range(4):
        tabla2.add_column(f"tok {i}", justify="right")
    tabla2.add_column("efecto")
    for T, efecto in [(0.5, "afilada: casi siempre el primero"),
                      (1.0, "la distribucion tal cual"),
                      (2.0, "plana: mas variedad")]:
        p = F.softmax(logits / T, dim=-1)[0]
        tabla2.add_row(f"{T}", *[f"{float(p[i]):.3f}" for i in range(4)], efecto)
    console.print(tabla2)

    console.print("\n[bold]La penalizacion de repeticion:[/bold]")
    pen = apply_repetition_penalty(logits.clone(), torch.tensor([[0, 5]]), 2.0)[0]
    console.print(
        f"  token 0 (logit [green]+3.0[/green]) -> {float(pen[0]):+.2f}   "
        "[dim]positivo: se DIVIDE[/dim]\n"
        f"  token 5 (logit [red]-3.0[/red]) -> {float(pen[5]):+.2f}   "
        "[dim]negativo: se MULTIPLICA[/dim]\n\n"
        "[dim]Si dividieras siempre, el -3.0 pasaria a -1.5 y el token se volveria MAS\n"
        "probable: justo lo contrario de penalizarlo. Y como los logits negativos son la\n"
        "mayoria, estarias premiando casi todo lo que ya salio.[/dim]"
    )


def experimento_muestreo(modelo, dataset, cfg_dev, entrenado):
    console.rule("[bold]2. La misma frase con distintas estrategias[/bold]")
    if not entrenado:
        console.print(
            "[yellow]No hay modelo entrenado en checkpoints/tiny_char/best.pt.[/yellow]\n"
            "[dim]Entrena uno con `llmfs train --config tiny_char` y vuelve a ejecutar esto:\n"
            "el texto sera legible y la comparacion tendra sentido.[/dim]\n"
        )

    prompt = "The king"
    ids = torch.tensor([dataset.encode(prompt)], device=cfg_dev.device)

    estrategias = [
        ("greedy (T=0)", dict(temperature=0.0)),
        ("T=0.5", dict(temperature=0.5)),
        ("T=0.8 + top-k 40", dict(temperature=0.8, top_k=40)),
        ("T=0.8 + top-p 0.9", dict(temperature=0.8, top_p=0.9)),
        ("T=1.5 (demasiado)", dict(temperature=1.5)),
        ("greedy + penalty 1.3", dict(temperature=0.0, repetition_penalty=1.3)),
    ]

    for nombre, kwargs in estrategias:
        set_seed(1234)
        salida = generate_with_cache(modelo, ids.clone(), 150, **kwargs)
        texto = dataset.decode(salida[0].tolist()).replace("\n", " ")
        # Cuanta repeticion hay: fraccion de 4-gramas distintos
        gramas = [texto[i : i + 4] for i in range(len(texto) - 4)]
        variedad = len(set(gramas)) / max(1, len(gramas))
        color = "red" if variedad < 0.5 else "yellow" if variedad < 0.8 else "green"
        console.print(
            Panel(
                texto[:180],
                title=f"{nombre}  |  variedad de 4-gramas: [{color}]{variedad:.0%}[/{color}]",
                border_style=color,
            )
        )

    console.print(
        "[dim]Mira la variedad de greedy frente a las demas. Greedy siempre elige el token\n"
        "mas probable, es determinista, y se mete en bucles: el texto humano NO maximiza la\n"
        "probabilidad, y esa es la observacion central de Holtzman et al. (2020).\n\n"
        "La penalizacion de repeticion rescata a greedy sin dejar de ser determinista.[/dim]"
    )


def experimento_cache(modelo, cfg_dev, cfg):
    console.rule("[bold]3. La KV cache[/bold]")

    prompt = torch.randint(0, cfg.model.vocab_size, (1, 8), device=cfg_dev.device)

    # Para medir la velocidad hace falta un modelo con contexto largo: con el juguete
    # (contexto 128) las tiradas de 100+ tokens topan con el limite y la comparacion se
    # aplana justo donde empezaba a ser interesante.
    from llmfs.config import ModelConfig

    set_seed(0)
    cfg_largo = ModelConfig(
        vocab_size=cfg.model.vocab_size, n_layers=4, d_model=128, n_heads=4,
        d_ff=384, context_length=1024,
    )
    modelo_largo = GPT(cfg_largo).to(cfg_dev.device).eval()
    prompt_largo = torch.randint(0, cfg.model.vocab_size, (1, 8), device=cfg_dev.device)

    console.print("[bold]Primero lo importante: ¿da la misma salida?[/bold]\n")
    sin = modelo.generate(prompt.clone(), 40, temperature=0.0)
    con = generate_with_cache(modelo, prompt.clone(), 40, temperature=0.0)
    iguales = torch.equal(sin, con)
    console.print(
        f"  sin cache: {sin[0, -10:].tolist()}\n"
        f"  con cache: {con[0, -10:].tolist()}\n"
        + (
            "  [bold green]IDENTICOS.[/bold green] La cache es una optimizacion pura: no "
            "cambia el resultado.\n"
            if iguales
            else "  [bold red]DIVERGEN.[/bold red] Mira el pos_offset de RoPE.\n"
        )
    )

    console.print("[bold]Y ahora la velocidad:[/bold]\n")
    tabla = Table(header_style="bold")
    tabla.add_column("tokens", justify="right")
    tabla.add_column("sin cache", justify="right")
    tabla.add_column("con cache", justify="right")
    tabla.add_column("speedup", justify="right")
    tabla.add_column("memoria de la cache", justify="right")

    console.print(
        f"[dim](midiendo sobre un modelo de contexto {cfg_largo.context_length}: con el\n"
        f"juguete de contexto {cfg.model.context_length} las tiradas largas toparian con el\n"
        f"limite y la comparacion se aplanaria justo donde empieza a ser interesante)[/dim]\n"
    )

    longitudes, speedups = [], []
    for n in (50, 100, 200, 400, 800):
        t0 = time.perf_counter()
        modelo_largo.generate(prompt_largo.clone(), n, temperature=0.0)
        cfg_dev.synchronize()
        t_sin = time.perf_counter() - t0

        t0 = time.perf_counter()
        generate_with_cache(modelo_largo, prompt_largo.clone(), n, temperature=0.0)
        cfg_dev.synchronize()
        t_con = time.perf_counter() - t0

        cache = KVCache(cfg_largo.n_layers)
        modelo_largo(prompt_largo, use_cache=True, cache=cache)
        for _ in range(n):
            modelo_largo(prompt_largo[:, -1:], use_cache=True, cache=cache)

        longitudes.append(n)
        speedups.append(t_sin / max(t_con, 1e-9))
        tabla.add_row(
            str(n),
            f"{t_sin * 1000:.0f} ms",
            f"{t_con * 1000:.0f} ms",
            f"[bold]{t_sin / max(t_con, 1e-9):.2f}x[/bold]",
            f"{cache.memory_bytes() / 1024:.0f} KB",
        )
    console.print(tabla)

    bytes_full = 2 * 6 * 512 * 320 * 2   # el modelo final: 6 capas, ctx 512, d 320, fp16
    console.print(
        f"\n[dim]El speedup crece con la longitud: sin cache generar N tokens cuesta O(N^2)\n"
        f"y con cache O(N). Con secuencias mas largas la diferencia se dispara.\n\n"
        f"La memoria de la cache es 2 * n_layers * T * d_model * bytes. Para el modelo FINAL\n"
        f"de 9M con contexto 512 en fp16 son {bytes_full / 1e6:.1f} MB: nada. Para un modelo de\n"
        f"70B con contexto 100.000, decenas de gigabytes, mas que los propios pesos. De ahi\n"
        f"que existan tecnicas como grouped-query attention.[/dim]"
    )
    return longitudes, speedups


def main() -> None:
    cfg_dev = get_device()
    modelo, dataset, cfg, entrenado = cargar_modelo(cfg_dev)

    if entrenado:
        console.print("[green]Usando el modelo entrenado de checkpoints/tiny_char/best.pt[/green]\n")

    experimento_filtros()
    experimento_muestreo(modelo, dataset, cfg_dev, entrenado)
    longitudes, speedups = experimento_cache(modelo, cfg_dev, cfg)

    fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))

    logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, -1.0, -3.0]])
    for T in (0.5, 1.0, 2.0):
        izq.plot(F.softmax(logits / T, dim=-1)[0].numpy(), marker="o", label=f"T = {T}")
    izq.set_xlabel("token")
    izq.set_ylabel("probabilidad")
    izq.set_title("El efecto de la temperatura")
    izq.grid(alpha=0.3)
    izq.legend()

    der.plot(longitudes, speedups, marker="o", color="tab:green")
    der.axhline(1.0, color="gray", ls="--", lw=1, label="sin ganancia")
    der.set_xlabel("tokens generados")
    der.set_ylabel("speedup de la KV cache")
    der.set_title("La ganancia crece con la longitud")
    der.grid(alpha=0.3)
    der.legend(fontsize=8)

    fig.tight_layout()
    destino = figures_dir() / "14_inference.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")


if __name__ == "__main__":
    main()
