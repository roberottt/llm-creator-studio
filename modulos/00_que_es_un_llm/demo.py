"""Demo del modulo 00: tu primer modelo de lenguaje, generando texto de verdad.

    llmfs demo 00

Entrena modelos por conteo con contextos de 1, 2, 3, 4 y 6 caracteres sobre Shakespeare
y te ensenya lo que escribe cada uno. El salto de calidad entre el de 1 y el de 4 es el
argumento entero del curso: mirar mas atras ayuda muchisimo... hasta que la tabla no cabe
en el disco. La grafica final ensenya justo eso.
"""

from __future__ import annotations

import random

import matplotlib

matplotlib.use("Agg")  # sin ventana: esto tiene que correr por SSH y en CI

import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.data import fetch_tinyshakespeare
from llmfs.paths import figures_dir
from llmfs.reference import build_count_table

console = Console()

next_token_probs = resolve("00_que_es_un_llm", "next_token_probs")
sample_next_token = resolve("00_que_es_un_llm", "sample_next_token")
generate_naive = resolve("00_que_es_un_llm", "generate_naive")

CONTEXTOS = [1, 2, 3, 4, 6]


def tamanyo_de_la_tabla(vocab: int, contexto: int) -> float:
    """Cuantas entradas TENDRIA la tabla si estuvieran todas las combinaciones."""
    return float(vocab) ** contexto


def main() -> None:
    texto, origen = fetch_tinyshakespeare()
    if origen == "reserva":
        console.print("[yellow]Sin conexion: usando el texto de reserva (mucho mas corto).[/yellow]")

    vocab = sorted(set(texto))
    console.print(
        Panel(
            f"corpus     : {len(texto):,} caracteres\n"
            f"vocabulario: {len(vocab)} caracteres distintos\n"
            f"muestra    : {texto[:80]!r}",
            title="el texto de entrenamiento",
            border_style="cyan",
        )
    )

    # ------------------------------------------------------------------ la distribucion
    tabla_1 = build_count_table(texto, context_size=1)
    console.rule("[bold]1. Que hay dentro del modelo[/bold]")
    console.print("Despues de la letra 'q', esto es lo que el modelo cree que puede venir:\n")

    probs_q = next_token_probs(tabla_1["q"])
    tabla_probs = Table(header_style="bold")
    tabla_probs.add_column("caracter")
    tabla_probs.add_column("probabilidad", justify="right")
    tabla_probs.add_column("", width=30)
    for char, p in sorted(probs_q.items(), key=lambda kv: -kv[1])[:6]:
        tabla_probs.add_row(repr(char), f"{p:.4f}", "#" * int(p * 30))
    console.print(tabla_probs)
    console.print(
        f"[dim]Suman {sum(probs_q.values()):.6f}. Eso es una distribucion de "
        f"probabilidad, y es LA salida de cualquier modelo de lenguaje.[/dim]"
    )

    # ------------------------------------------------------------------ generacion
    console.rule("[bold]2. Lo que escribe cada modelo[/bold]")
    realismo: list[float] = []
    palabras_reales = set(texto.lower().split())

    for n in CONTEXTOS:
        tabla = build_count_table(texto, context_size=n)
        inicio = texto[:n]
        salida = generate_naive(tabla, inicio, length=240, rng=random.Random(1234))

        # Fraccion de palabras generadas que existen de verdad en el corpus. Es la
        # medida honesta: contar trigramas daria 100% siempre que el contexto sea >= 2,
        # porque por construccion cada trigrama generado ya existia.
        generadas = salida.lower().split()
        pct = 100 * sum(w in palabras_reales for w in generadas) / max(1, len(generadas))
        realismo.append(pct)

        console.print(
            Panel(
                salida.replace("\n", " "),
                title=f"contexto de {n} caracter{'es' if n > 1 else ''}  "
                f"| {len(tabla):,} contextos vistos | {pct:.0f}% de palabras reales",
                border_style="green" if pct > 80 else "yellow" if pct > 50 else "red",
            )
        )

    # ------------------------------------------------------------------ el muro
    console.rule("[bold]3. Por que esto no escala[/bold]")
    tabla_muro = Table(header_style="bold")
    tabla_muro.add_column("contexto")
    tabla_muro.add_column("contextos vistos", justify="right")
    tabla_muro.add_column("combinaciones posibles", justify="right")
    tabla_muro.add_column("% cubierto", justify="right")

    for n in CONTEXTOS:
        vistos = len(build_count_table(texto, context_size=n))
        posibles = tamanyo_de_la_tabla(len(vocab), n)
        tabla_muro.add_row(f"{n} car.", f"{vistos:,}", f"{posibles:.3g}", f"{100 * vistos / posibles:.2g}%")
    console.print(tabla_muro)

    console.print(
        "\n[bold]Mira la ultima columna.[/bold] Con contexto de 6 caracteres, el corpus\n"
        "cubre una fraccion ridicula de las combinaciones posibles. Y esto es a nivel de\n"
        "CARACTER. Nuestro modelo final usara 4096 tokens y un contexto de 512:\n"
        f"[red]4096^512[/red] combinaciones, un numero con mas de 1800 cifras.\n\n"
        "Contar no puede resolver esto. Por eso hacen falta las redes neuronales: en vez\n"
        "de memorizar cada contexto, aprenden que 'gato' y 'perro' se parecen y comparten\n"
        "lo aprendido entre ellos. Eso es el modulo 05 en adelante."
    )

    # ------------------------------------------------------------------ grafica
    fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))

    izq.plot(CONTEXTOS, realismo, marker="o", color="tab:green")
    izq.set_xlabel("caracteres de contexto")
    izq.set_ylabel("% de palabras que existen en el original")
    izq.set_title("Mas contexto -> texto mas realista")
    izq.set_ylim(0, 105)
    izq.grid(alpha=0.3)

    vistos = [len(build_count_table(texto, context_size=n)) for n in CONTEXTOS]
    posibles = [tamanyo_de_la_tabla(len(vocab), n) for n in CONTEXTOS]
    der.plot(CONTEXTOS, posibles, marker="o", label="combinaciones posibles")
    der.plot(CONTEXTOS, vistos, marker="s", label="contextos vistos en el corpus")
    der.set_yscale("log")
    der.set_xlabel("caracteres de contexto")
    der.set_ylabel("numero de contextos (escala log)")
    der.set_title("...pero la tabla explota")
    der.grid(alpha=0.3)
    der.legend()

    fig.tight_layout()
    destino = figures_dir() / "00_conteo_vs_generalizacion.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")


if __name__ == "__main__":
    main()
