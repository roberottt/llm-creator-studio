"""Demo del modulo 05: los tres baselines compitiendo, y donde esta el suelo.

    llmfs demo 05

Entrena de verdad (unos segundos) un bigrama por conteo, un bigrama neuronal y un MLP de
Bengio sobre Shakespeare a nivel caracter, y compara sus perdidas con el baseline uniforme.

Lo que hay que llevarse: el suelo `ln(V)`, que el bigrama neuronal converge exactamente al
mismo sitio que el de conteo, y que mirar mas contexto ayuda... con un coste en parametros
que crece linealmente. Esa es la puerta de entrada al modulo 06.
"""

from __future__ import annotations

import math
import time

import matplotlib

matplotlib.use("Agg")  # sin ventana: esto tiene que correr por SSH y en CI

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.data import fetch_tinyshakespeare
from llmfs.device import get_device, set_seed
from llmfs.paths import figures_dir

console = Console()

uniform_baseline_loss = resolve("05_baselines", "uniform_baseline_loss")
bigram_counts = resolve("05_baselines", "bigram_counts")
bigram_nll = resolve("05_baselines", "bigram_nll")
NeuralBigram = resolve("05_baselines", "NeuralBigram")
BengioMLP = resolve("05_baselines", "BengioMLP")


def entrena(modelo, x, y, pasos: int, lr: float, cfg) -> list[float]:
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr)
    historial: list[float] = []
    for _ in range(pasos):
        _, perdida = modelo(x, y)
        opt.zero_grad(set_to_none=True)
        perdida.backward()
        opt.step()
        historial.append(float(perdida.detach()))
    return historial


def main() -> None:
    cfg = get_device()
    set_seed(1337)

    texto, _ = fetch_tinyshakespeare()
    texto = texto[:200_000]
    vocab_chars = sorted(set(texto))
    V = len(vocab_chars)
    stoi = {c: i for i, c in enumerate(vocab_chars)}
    ids = [stoi[c] for c in texto]

    corte = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:corte], ids[corte:]

    suelo = uniform_baseline_loss(V)
    console.rule("[bold]0. El suelo[/bold]")
    console.print(
        f"Vocabulario: {V} caracteres.\n"
        f"Un modelo que adivina al azar da una perdida de [bold]ln({V}) = {suelo:.4f}[/bold] "
        f"nats,\nque es una perplejidad de {math.exp(suelo):.1f} (o sea, dudando entre "
        f"{V} opciones).\n\n"
        "[dim]Cualquier modelo que no baje de este numero no ha aprendido nada. Y cuando\n"
        "entrenes el modelo final, la perdida del paso 0 tiene que valer casi exactamente\n"
        "esto: mas alta significa init mal, mas baja significa fuga de informacion.[/dim]"
    )

    resultados: list[tuple[str, float, int, str]] = []
    resultados.append(("uniforme (azar)", suelo, 0, "-"))

    # ------------------------------------------------------------- bigrama por conteo
    console.rule("[bold]1. Bigrama por conteo[/bold]")
    inicio = time.perf_counter()
    counts = bigram_counts(train_ids, V)
    perdida_conteo = bigram_nll(counts, val_ids, alpha=1.0)
    console.print(
        f"  matriz {V}x{V} rellenada en {time.perf_counter() - inicio:.2f} s\n"
        f"  perdida en validacion: [bold]{perdida_conteo:.4f}[/bold] "
        f"(perplejidad {math.exp(perdida_conteo):.1f})"
    )
    resultados.append(("bigrama (conteo)", perdida_conteo, V * V, "0.0 s"))

    console.print("\n[bold]Efecto del suavizado:[/bold]")
    for alpha in (0.0001, 0.01, 1.0, 100.0, 10000.0):
        p = bigram_nll(counts, val_ids, alpha=alpha)
        marca = "  <- sin suavizar casi, y aun asi finito porque el corpus es denso" if alpha == 0.0001 else ""
        console.print(f"    alpha={alpha:<8} -> {p:.4f}{marca}")
    console.print(
        "[dim]Con alpha enorme el modelo se vuelve uniforme y la perdida sube al suelo.\n"
        "Con alpha diminuto y un par no visto en validacion, se iria a infinito.[/dim]"
    )

    # ------------------------------------------------------------- bigrama neuronal
    console.rule("[bold]2. El mismo bigrama, aprendido por gradiente[/bold]")
    x_tr = torch.tensor(train_ids[:-1], device=cfg.device).unsqueeze(0)
    y_tr = torch.tensor(train_ids[1:], device=cfg.device).unsqueeze(0)

    modelo_nb = NeuralBigram(V).to(cfg.device)
    inicio = time.perf_counter()
    hist_nb = entrena(modelo_nb, x_tr, y_tr, pasos=400, lr=0.5, cfg=cfg)
    t_nb = time.perf_counter() - inicio

    x_val = torch.tensor(val_ids[:-1], device=cfg.device).unsqueeze(0)
    y_val = torch.tensor(val_ids[1:], device=cfg.device).unsqueeze(0)
    with torch.no_grad():
        val_nb = float(modelo_nb(x_val, y_val)[1])

    exceso = hist_nb[0] - suelo
    console.print(
        f"  perdida inicial : {hist_nb[0]:.4f}   [dim](el suelo es {suelo:.4f})[/dim]\n"
        f"  perdida final   : {hist_nb[-1]:.4f} en entrenamiento\n"
        f"  en validacion   : [bold]{val_nb:.4f}[/bold]\n"
        f"  parametros      : {sum(p.numel() for p in modelo_nb.parameters()):,}\n\n"
        f"[bold]Contando: {perdida_conteo:.4f}. Aprendiendo: {val_nb:.4f}.[/bold]\n"
        "[dim]Convergen al mismo sitio, porque son el mismo modelo. La diferencia es que\n"
        "contar no escala mas alla de esto y aprender si.[/dim]"
    )

    console.print(
        f"\n[bold yellow]Fijate en la perdida inicial: {hist_nb[0]:.4f}, que esta "
        f"{exceso:.2f} nats POR ENCIMA\ndel suelo {suelo:.4f}.[/bold yellow] Eso no deberia "
        "pasar, y es un ejemplo perfecto del\nsintoma que describe THEORY.md.\n\n"
        "La causa: `nn.Embedding` inicializa por defecto con una normal N(0,1). Como esas\n"
        "filas SON los logits, el modelo arranca con opiniones fuertes y aleatorias en vez\n"
        "de con ignorancia. Un logit de +2 frente a otro de -2 es una apuesta de 55 a 1\n"
        "hecha antes de ver un solo dato, y acertar por azar es improbable: de ahi el exceso.\n\n"
        "Por eso el GPT del modulo 10 inicializa TODO con std=0.02 en vez de 1.0. Con logits\n"
        "casi identicos, el softmax sale casi uniforme y la perdida del paso 0 cae justo\n"
        "sobre ln(V). Puedes comprobarlo:\n"
        "  [cyan]torch.nn.init.normal_(modelo.token_embedding.weight, std=0.02)[/cyan]"
    )
    resultados.append(
        ("bigrama (neuronal)", val_nb, sum(p.numel() for p in modelo_nb.parameters()), f"{t_nb:.1f} s")
    )

    # ------------------------------------------------------------- MLP de Bengio
    console.rule("[bold]3. MLP de Bengio: mirar mas atras[/bold]")
    hist_bengio: dict[int, list[float]] = {}
    val_por_bloque: dict[int, float] = {}
    for bloque in (2, 4, 8):
        datos_tr = torch.tensor(train_ids, dtype=torch.long)
        xb = torch.stack([datos_tr[i : i + bloque] for i in range(0, len(datos_tr) - bloque, 3)])
        yb = torch.stack([datos_tr[i + bloque] for i in range(0, len(datos_tr) - bloque, 3)])
        xb, yb = xb.to(cfg.device), yb.to(cfg.device)

        datos_val = torch.tensor(val_ids, dtype=torch.long)
        xv = torch.stack([datos_val[i : i + bloque] for i in range(0, len(datos_val) - bloque, 7)])
        yv = torch.stack([datos_val[i + bloque] for i in range(0, len(datos_val) - bloque, 7)])
        xv, yv = xv.to(cfg.device), yv.to(cfg.device)

        modelo = BengioMLP(V, bloque, d_embed=24, n_hidden=128).to(cfg.device)
        n_params = sum(p.numel() for p in modelo.parameters())

        inicio = time.perf_counter()
        historial: list[float] = []
        opt = torch.optim.AdamW(modelo.parameters(), lr=0.01)
        for paso in range(400):
            sel = torch.randint(0, len(xb), (512,), device=cfg.device)
            _, perdida = modelo(xb[sel], yb[sel])
            opt.zero_grad(set_to_none=True)
            perdida.backward()
            opt.step()
            historial.append(float(perdida.detach()))
        transcurrido = time.perf_counter() - inicio

        with torch.no_grad():
            val = float(modelo(xv, yv)[1])
        hist_bengio[bloque] = historial
        val_por_bloque[bloque] = val
        console.print(
            f"  contexto {bloque} caracteres: validacion [bold]{val:.4f}[/bold]  "
            f"({n_params:,} parametros, {transcurrido:.1f} s)"
        )
        resultados.append((f"Bengio MLP (ctx {bloque})", val, n_params, f"{transcurrido:.1f} s"))

    mejor_bloque = min(val_por_bloque, key=val_por_bloque.get)
    monotono = list(val_por_bloque.values()) == sorted(val_por_bloque.values(), reverse=True)

    console.print(
        f"\nTodos baten al bigrama ({perdida_conteo:.4f}), asi que mirar mas de un token "
        "atras ayuda.\n"
        f"El mejor de los tres es el de contexto {mejor_bloque} "
        f"({val_por_bloque[mejor_bloque]:.4f})."
    )
    if not monotono:
        console.print(
            "\n[bold yellow]Pero la mejora NO es monotona: el contexto mas largo no gana."
            "[/bold yellow]\n"
            "No es un error de la demo, es un resultado honesto y conviene entenderlo. Los\n"
            "tres modelos han entrenado los MISMOS 400 pasos, y el de contexto 8 tiene mas\n"
            "del doble de parametros que el de contexto 2. Con el mismo presupuesto de\n"
            "computo, el modelo grande se queda a medio entrenar.\n\n"
            "[dim]Leccion transferible: comparar arquitecturas a igualdad de PASOS no es\n"
            "comparar a igualdad de computo, y casi siempre favorece al modelo pequenyo.\n"
            "Es el mismo error que las leyes de escala del modulo 12 vienen a corregir.[/dim]"
        )
    console.print(
        "\n[dim]Y en cualquier caso, mira los parametros: la primera capa es\n"
        "Linear(block_size * d_embed, n_hidden), asi que crece LINEALMENTE con el contexto.\n"
        "Con contexto 512 esa capa sola seria enorme. Ademas el modelo trata cada posicion\n"
        "como una entrada independiente, sin saber que estan relacionadas ni poder decidir\n"
        "a cual hacer caso. Resolver las dos cosas a la vez es lo que hace la atencion,\n"
        "en el modulo 06.[/dim]"
    )

    # ------------------------------------------------------------- resumen
    console.rule("[bold]Resumen[/bold]")
    tabla = Table(header_style="bold")
    tabla.add_column("modelo")
    tabla.add_column("perdida (val)", justify="right")
    tabla.add_column("perplejidad", justify="right")
    tabla.add_column("parametros", justify="right")
    tabla.add_column("tiempo", justify="right")
    for nombre, perdida, params, tiempo in resultados:
        tabla.add_row(
            nombre,
            f"{perdida:.4f}",
            f"{math.exp(perdida):.1f}",
            f"{params:,}" if params else "-",
            tiempo,
        )
    console.print(tabla)

    # ------------------------------------------------------------- grafica
    fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))

    izq.plot(hist_nb, label="bigrama neuronal")
    for bloque, hist in hist_bengio.items():
        izq.plot(hist, label=f"Bengio MLP (ctx {bloque})")
    izq.axhline(suelo, color="red", ls="--", lw=1.5, label=f"suelo: ln({V}) = {suelo:.2f}")
    izq.axhline(perdida_conteo, color="gray", ls=":", lw=1.5, label="bigrama por conteo")
    izq.set_xlabel("paso de entrenamiento")
    izq.set_ylabel("perdida (nats)")
    izq.set_title("Todos los baselines")
    izq.grid(alpha=0.3)
    izq.legend(fontsize=8)

    nombres = [r[0] for r in resultados]
    perdidas = [r[1] for r in resultados]
    colores = ["tab:red"] + ["tab:gray"] * 2 + ["tab:blue"] * (len(resultados) - 3)
    der.barh(range(len(nombres)), perdidas, color=colores)
    der.set_yticks(range(len(nombres)))
    der.set_yticklabels(nombres, fontsize=8)
    der.axvline(suelo, color="red", ls="--", lw=1.5)
    der.set_xlabel("perdida en validacion (nats)")
    der.set_title("Cuanto baja cada uno del suelo")
    der.grid(alpha=0.3, axis="x")
    der.invert_yaxis()

    fig.tight_layout()
    destino = figures_dir() / "05_baselines.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")


if __name__ == "__main__":
    main()
