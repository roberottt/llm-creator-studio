"""Demo del modulo 03: que aprende de verdad un BPE, y cuanto comprime.

    llmfs demo 03
    llmfs demo 03 --rapido      muestra mas pequenya, tarda la mitad

Cuatro cosas:
  1. Los primeros merges que aprende, en orden. Se ve como pasa de letras a silabas y de
     silabas a palabras enteras.
  2. La misma frase tokenizada con vocabularios de distinto tamanyo.
  3. La curva de compresion, y al lado el coste en parametros de la tabla de embeddings.
     Ese es el intercambio que decide el vocab_size del modelo final.
  4. Comparacion con tiktoken (el tokenizador de GPT-4), si esta instalado.

Entrenar BPE en python puro no es rapido: cuenta con uno o dos minutos.
"""

from __future__ import annotations

import sys
import time

import matplotlib

matplotlib.use("Agg")  # sin ventana: esto tiene que correr por SSH y en CI

import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.data import fetch_tinyshakespeare
from llmfs.paths import figures_dir

console = Console()

train_bpe = resolve("03_tokenizacion", "train_bpe")
bpe_encode = resolve("03_tokenizacion", "bpe_encode")
bpe_decode = resolve("03_tokenizacion", "bpe_decode")

FRASE = "The king shall speak to his people tomorrow morning."
D_MODEL = 320  # el del modelo final, para calcular el coste de la tabla de embeddings


def main() -> None:
    rapido = "--rapido" in sys.argv
    texto, origen = fetch_tinyshakespeare()
    muestra = texto[: 60_000 if rapido else 200_000]
    vocabularios = [300, 512, 1024] if rapido else [300, 512, 1024, 2048]

    console.print(
        f"Entrenando sobre {len(muestra):,} caracteres "
        f"({'muestra reducida' if rapido else 'muestra normal'}).\n"
    )

    # ------------------------------------------------------------- 1. que aprende
    console.rule("[bold]1. Los primeros merges, en orden[/bold]")
    inicio = time.perf_counter()
    merges_512, vocab_512 = train_bpe(muestra, 512)
    console.print(f"[dim]512 tokens entrenados en {time.perf_counter() - inicio:.1f} s[/dim]\n")

    tabla = Table(header_style="bold")
    tabla.add_column("#", justify="right", style="dim")
    tabla.add_column("id", justify="right")
    tabla.add_column("token nuevo")
    tabla.add_column("se forma con")

    items = list(merges_512.items())
    for n, ((a, b), nuevo) in enumerate(items[:12] + items[-6:]):
        if n == 12:
            tabla.add_row("...", "...", "...", "...")
        pa = vocab_512[a].decode("utf-8", errors="replace")
        pb = vocab_512[b].decode("utf-8", errors="replace")
        tabla.add_row(
            str(items.index(((a, b), nuevo)) + 1),
            str(nuevo),
            repr(vocab_512[nuevo].decode("utf-8", errors="replace")),
            f"{pa!r} + {pb!r}",
        )
    console.print(tabla)
    console.print(
        "[dim]Los primeros merges son pares de letras muy frecuentes. Los ultimos ya son\n"
        "palabras enteras o terminaciones completas. Nadie le ha dicho al algoritmo que\n"
        "existen las palabras: lo ha deducido de las frecuencias.[/dim]"
    )

    # ------------------------------------------------------------- 2. la misma frase
    console.rule("[bold]2. La misma frase, con vocabularios distintos[/bold]")
    console.print(f"Frase: [cyan]{FRASE}[/cyan]\n")

    entrenados: dict[int, tuple[dict, dict]] = {}
    ratios: list[float] = []
    for vs in vocabularios:
        merges, vocab = train_bpe(muestra, vs)
        entrenados[vs] = (merges, vocab)

        ids = bpe_encode(FRASE, merges)
        piezas = [bpe_decode([i], vocab) for i in ids]
        console.print(f"[bold]vocab {vs}[/bold] -> {len(ids)} tokens")
        console.print("  " + " | ".join(p.replace("\n", "\\n") for p in piezas) + "\n")

        ids_muestra = bpe_encode(muestra[:20_000], merges)
        ratios.append(len(muestra[:20_000].encode("utf-8")) / len(ids_muestra))

    # ------------------------------------------------------------- 3. el intercambio
    console.rule("[bold]3. El intercambio que decide el vocab_size[/bold]")
    tabla2 = Table(header_style="bold")
    tabla2.add_column("vocabulario", justify="right")
    tabla2.add_column("bytes/token", justify="right")
    tabla2.add_column("tokens para 1 MB", justify="right")
    tabla2.add_column(f"params de embeddings (d={D_MODEL})", justify="right")
    tabla2.add_column("% de un modelo de 8,9M", justify="right")

    for vs, ratio in zip(vocabularios, ratios):
        params = vs * D_MODEL
        tabla2.add_row(
            f"{vs:,}",
            f"{ratio:.2f}",
            f"{1_000_000 / ratio / 1e3:.0f}k",
            f"{params:,}",
            f"{100 * params / 8_933_440:.0f}%",
        )
    for vs in (4096, 32000, 50257):
        params = vs * D_MODEL
        tabla2.add_row(
            f"{vs:,}",
            "[dim]?[/dim]",
            "[dim]?[/dim]",
            f"{params:,}",
            f"{100 * params / 8_933_440:.0f}%",
            style="dim" if vs != 4096 else "bold",
        )
    console.print(tabla2)
    console.print(
        "\nUn vocabulario mas grande comprime mejor (menos tokens por texto, secuencias\n"
        "mas cortas, menos pasos de entrenamiento) pero se come el presupuesto de\n"
        "parametros en una tabla de consulta. Con 50.257 tokens como GPT-2, la tabla de\n"
        "embeddings sola seria [red]casi el doble[/red] que nuestro modelo entero.\n"
        "[bold]Por eso el modelo final usa 4.096.[/bold]"
    )

    # ------------------------------------------------------------- 4. tiktoken
    console.rule("[bold]4. Comparacion con tiktoken (GPT-4)[/bold]")
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        ids_gpt4 = enc.encode(muestra[:20_000])
        ratio_gpt4 = len(muestra[:20_000].encode("utf-8")) / len(ids_gpt4)
        piezas_gpt4 = [enc.decode([i]) for i in enc.encode(FRASE)]

        console.print(f"cl100k_base tiene [bold]{enc.n_vocab:,}[/bold] tokens.")
        console.print(f"  compresion : {ratio_gpt4:.2f} bytes/token")
        console.print(f"  la frase   : {len(piezas_gpt4)} tokens")
        console.print("  " + " | ".join(piezas_gpt4))
        mejor = ratio_gpt4 / ratios[-1]
        console.print(
            f"\nComprime [bold]{mejor:.1f}x[/bold] mejor que nuestro vocab de "
            f"{vocabularios[-1]}, con {enc.n_vocab / vocabularios[-1]:.0f}x mas tokens.\n"
            "[dim]Rendimientos decrecientes: multiplicar el vocabulario por 50 no "
            "multiplica la compresion por 50.[/dim]"
        )
    except ImportError:
        console.print(
            "[yellow]tiktoken no esta instalado. Instalalo con `uv sync --extra compare` "
            "para ver esta comparacion.[/yellow]"
        )

    # ------------------------------------------------------------- grafica
    fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))

    izq.plot(vocabularios, ratios, marker="o", color="tab:blue")
    izq.set_xscale("log", base=2)
    izq.set_xlabel("tamanyo del vocabulario")
    izq.set_ylabel("bytes por token")
    izq.set_title("Mas vocabulario comprime mejor...")
    izq.grid(alpha=0.3)

    todos = [*vocabularios, 4096, 32000, 50257]
    der.plot(todos, [v * D_MODEL / 1e6 for v in todos], marker="o", color="tab:red")
    der.axhline(8.93, color="gray", ls="--", lw=1, label="modelo entero (8,9M)")
    der.axvline(4096, color="tab:green", ls=":", lw=2, label="nuestra eleccion")
    der.set_xscale("log", base=2)
    der.set_xlabel("tamanyo del vocabulario")
    der.set_ylabel("millones de parametros solo en embeddings")
    der.set_title("...pero se come el modelo")
    der.grid(alpha=0.3)
    der.legend()

    fig.tight_layout()
    destino = figures_dir() / "03_tokenizacion.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")


if __name__ == "__main__":
    main()
