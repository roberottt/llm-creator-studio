"""Demo del modulo 04: el pipeline de datos completo, con ficheros de verdad.

    llmfs demo 04

Hace el recorrido entero: texto -> tokens -> fichero binario -> memmap -> batch en la GPU.
Y por el camino ensenya:
  1. Cuanto ocupa el corpus en cada formato, y por que uint16.
  2. Un batch de verdad, con la correspondencia x/y token a token.
  3. Cuantos batches por segundo da tu disco, comparado con lo que tarda un paso de
     entrenamiento. Si el dato va mas lento que el calculo, la GPU se queda parada.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")  # sin ventana: esto tiene que correr por SSH y en CI

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.data import fetch_tinyshakespeare
from llmfs.device import get_device
from llmfs.paths import data_dir, figures_dir

console = Console()

pack_tokens_uint16 = resolve("04_data", "pack_tokens_uint16")
train_val_split = resolve("04_data", "train_val_split")
get_batch = resolve("04_data", "get_batch")

CONTEXT = 64
BATCH = 8


def main() -> None:
    cfg = get_device()
    texto, _ = fetch_tinyshakespeare()

    # ------------------------------------------------------------- 1. tokenizar
    console.rule("[bold]1. De texto a tokens[/bold]")
    vocab_chars = sorted(set(texto))
    stoi = {c: i for i, c in enumerate(vocab_chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = [stoi[c] for c in texto]

    console.print(
        f"Tokenizador de caracteres (el mas simple posible, para que la demo sea rapida).\n"
        f"  texto      : {len(texto):,} caracteres\n"
        f"  vocabulario: {len(vocab_chars)} tokens\n"
        f"  tokens     : {len(ids):,}\n"
    )

    tokens = pack_tokens_uint16(ids, len(vocab_chars))

    tabla = Table(title="Lo que ocupa el corpus en cada formato", header_style="bold")
    tabla.add_column("formato")
    tabla.add_column("bytes/token", justify="right")
    tabla.add_column("este corpus", justify="right")
    tabla.add_column("500M tokens", justify="right")
    for nombre, ancho in [("int64 (python)", 8), ("uint32", 4), ("uint16", 2)]:
        tabla.add_row(
            nombre,
            str(ancho),
            f"{len(ids) * ancho / 1e6:.1f} MB",
            f"{500e6 * ancho / 1e9:.1f} GB",
            style="bold green" if ancho == 2 else "",
        )
    console.print(tabla)

    # ------------------------------------------------------------- 2. a disco
    console.rule("[bold]2. A disco, y de vuelta con memmap[/bold]")
    train, val = train_val_split(tokens, 0.05)
    ruta = data_dir() / "demo_shakespeare_char.bin"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    train.tofile(ruta)

    mb = ruta.stat().st_size / 1e6

    # Se mide dos veces cada una y se coge la mejor, para que el orden no decida el
    # resultado: la primera lectura calienta la cache de paginas del sistema operativo y
    # la segunda medicion sale artificialmente rapida.
    def cronometra(fn) -> float:
        mejor = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            mejor = min(mejor, time.perf_counter() - t0)
        return mejor

    t_memmap = cronometra(lambda: np.memmap(ruta, dtype=np.uint16, mode="r"))
    t_ram = cronometra(lambda: np.fromfile(ruta, dtype=np.uint16))
    mapeado = np.memmap(ruta, dtype=np.uint16, mode="r")

    console.print(
        f"  entrenamiento: {len(train):,} tokens  ({mb:.1f} MB en disco)\n"
        f"  validacion   : {len(val):,} tokens (el 5% final, contiguo)\n\n"
        f"  abrir con memmap    : {t_memmap * 1000:6.2f} ms\n"
        f"  cargar entero a RAM : {t_ram * 1000:6.2f} ms\n"
    )
    console.print(
        f"[bold yellow]No leas nada en estos dos numeros.[/bold yellow] Con {mb:.0f} MB, el "
        "fichero entra entero en la\ncache del sistema operativo y lo que estas midiendo es "
        "ruido y coste fijo de\nabrir un descriptor. Cual salga mas rapido depende del dia.\n"
    )
    console.print(
        "Y tampoco extrapolo a 1 GB a partir de ellos, porque seria cometer el mismo\n"
        "error: la velocidad que acabas de medir es la de la cache, no la del disco.\n\n"
        "El argumento a favor de memmap no es de velocidad de lectura, es este:\n\n"
        "  - [bold]arranque[/bold]: cargar el fichero entero cuesta lo que cuesta, y lo "
        "pagas cada vez\n    que lanzas el script. memmap no lee nada hasta que tocas los "
        "datos.\n"
        "  - [bold]gestion[/bold]: como accedes a posiciones aleatorias una y otra vez, la "
        "cache del\n    sistema operativo acaba reteniendo lo que mas usas. Gratis, y "
        "mejor de lo\n    que lo harias tu.\n"
        "  - [bold]escala[/bold]: el mismo codigo vale igual si manyana el corpus son 50 GB.\n\n"
        "[dim]Con 16 GB de RAM, el corpus real de 1 GB tambien cabria cargado entero. Si\n"
        "prefieres np.fromfile por simplicidad, es una opcion perfectamente defendible a\n"
        "esta escala. Aqui no hay magia.[/dim]"
    )

    # ------------------------------------------------------------- 3. un batch
    console.rule("[bold]3. Un batch de verdad[/bold]")
    x, y = get_batch(mapeado, BATCH, CONTEXT, device=cfg.device, rng=np.random.default_rng(0))
    console.print(f"x: {tuple(x.shape)} {x.dtype} en {x.device}")
    console.print(f"y: {tuple(y.shape)} {y.dtype} en {y.device}\n")

    fila = x[0, :12].tolist()
    objetivo = y[0, :12].tolist()
    console.print("Los primeros 12 tokens de la primera muestra:\n")
    console.print("  x (entrada) : " + " ".join(f"{repr(itos[i])[1:-1]:>4}" for i in fila))
    console.print("  y (objetivo): " + " ".join(f"{repr(itos[i])[1:-1]:>4}" for i in objetivo))
    console.print(
        f"\n  como texto  : x = {''.join(itos[i] for i in fila)!r}"
        f"\n                y = {''.join(itos[i] for i in objetivo)!r}\n"
    )

    console.print("Y esto es lo que aprende el modelo de esta UNICA muestra:\n")
    for k in range(1, 6):
        contexto = "".join(itos[i] for i in fila[:k])
        console.print(f"  viendo {contexto!r:<12} -> debe predecir {itos[objetivo[k - 1]]!r}")
    console.print(
        f"  [dim]... y asi hasta {CONTEXT} predicciones, de una sola ventana.[/dim]\n"
        f"[bold]Un batch de {BATCH}x{CONTEXT} son {BATCH * CONTEXT:,} predicciones.[/bold] "
        "Por eso los modelos de\nlenguaje aprovechan tanto cada pasada por los datos."
    )

    # ------------------------------------------------------------- 4. velocidad
    console.rule("[bold]4. Velocidad del pipeline[/bold]")
    tamanyos = [(8, 64), (16, 128), (32, 256), (48, 512)]
    tokens_por_seg: list[float] = []
    etiquetas: list[str] = []

    rng = np.random.default_rng(0)
    for b, t in tamanyos:
        if len(mapeado) < t + 2:
            continue
        for _ in range(3):  # calentar
            get_batch(mapeado, b, t, device=cfg.device, rng=rng)
        cfg.synchronize()
        inicio = time.perf_counter()
        n = 20
        for _ in range(n):
            get_batch(mapeado, b, t, device=cfg.device, rng=rng)
        cfg.synchronize()
        transcurrido = time.perf_counter() - inicio
        tps = n * b * t / transcurrido
        tokens_por_seg.append(tps)
        etiquetas.append(f"{b}x{t}")
        console.print(
            f"  batch {b:>2} x {t:>3} = {b * t:>6,} tokens  ->  "
            f"{transcurrido / n * 1000:6.2f} ms/batch   {tps / 1e6:6.2f} M tokens/s"
        )

    console.print(
        "\n[dim]Compara el ms/batch de la ultima fila (48x512, el del modelo final) con lo\n"
        "que tarda un paso de entrenamiento. Lo mediras en el modulo 12: si el dato tarda\n"
        "mas que el calculo, la GPU se pasa el rato esperando y hay que paralelizar la\n"
        "carga.[/dim]"
    )

    # ------------------------------------------------------------- grafica
    fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))

    anchos = [8, 4, 2]
    nombres = ["int64", "uint32", "uint16"]
    colores = ["tab:red", "tab:orange", "tab:green"]
    izq.bar(nombres, [500e6 * a / 1e9 for a in anchos], color=colores)
    for i, a in enumerate(anchos):
        izq.text(i, 500e6 * a / 1e9, f"{500e6 * a / 1e9:.1f} GB", ha="center", va="bottom")
    izq.set_ylabel("GB en disco")
    izq.set_title("500M tokens, segun el tipo elegido")
    izq.grid(alpha=0.3, axis="y")

    der.bar(etiquetas, [t / 1e6 for t in tokens_por_seg], color="tab:blue")
    der.set_ylabel("millones de tokens/s")
    der.set_xlabel("batch x contexto")
    der.set_title(f"Velocidad de get_batch ({cfg.kind})")
    der.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    destino = figures_dir() / "04_data.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")


if __name__ == "__main__":
    main()
