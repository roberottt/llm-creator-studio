"""Demo del modulo 02: mira el grafo de computo por dentro.

    llmfs demo 02

Tres experimentos:
  1. Un grafo pequenyo, nodo a nodo, con el gradiente que recibe cada uno. Al lado, lo
     que dice torch.autograd para la misma expresion.
  2. El experimento del `+=`: que pasa exactamente si acumulas mal los gradientes.
  3. Entrenar un MLP con tu motor y dibujar la curva de perdida.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # sin ventana: esto tiene que correr por SSH y en CI

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.paths import figures_dir

console = Console()

Value = resolve("02_autograd", "Value")
topological_order = resolve("02_autograd", "topological_order")
train_scalar_mlp = resolve("02_autograd", "train_scalar_mlp")


def experimento_1_el_grafo() -> None:
    console.rule("[bold]1. Un grafo de computo, por dentro[/bold]")
    console.print("Expresion:  [cyan]L = (a * b + a.tanh()) * 2[/cyan]   con a=2.0, b=-3.0\n")

    a, b = Value(2.0), Value(-3.0)
    a.label, b.label = "a", "b"
    dos = Value(2.0)
    dos.label = "2 (constante)"
    producto = a * b
    activacion = a.tanh()
    suma = producto + activacion
    L = suma * dos
    L.backward()

    # Lo mismo con torch, para tener una segunda opinion.
    ta = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    ((ta * tb + ta.tanh()) * 2.0).backward()

    orden = topological_order(L)
    tabla = Table(header_style="bold")
    tabla.add_column("#", justify="right", style="dim")
    tabla.add_column("op")
    tabla.add_column("valor", justify="right")
    tabla.add_column("gradiente", justify="right")
    tabla.add_column("que significa")

    significados = {
        "": "hoja: un numero de entrada, no viene de ninguna operacion",
        "*": "producto",
        "+": "suma",
        "tanh": "activacion no lineal",
    }
    for i, nodo in enumerate(orden):
        tabla.add_row(
            str(i),
            nodo._op or (nodo.label or "hoja"),
            f"{nodo.data:.4f}",
            f"{nodo.grad:+.4f}",
            significados.get(nodo._op, ""),
        )
    console.print(tabla)

    console.print(
        f"\nOrden topologico: {len(orden)} nodos. El backward los recorre AL REVES,\n"
        f"del ultimo (la salida, gradiente 1) al primero.\n"
    )
    console.print("[bold]Comprobacion contra torch.autograd:[/bold]")
    console.print(f"  dL/da  ->  tuyo {a.grad:+.10f}   torch {ta.grad.item():+.10f}")
    console.print(f"  dL/db  ->  tuyo {b.grad:+.10f}   torch {tb.grad.item():+.10f}")
    coincide = abs(a.grad - ta.grad.item()) < 1e-9 and abs(b.grad - tb.grad.item()) < 1e-9
    console.print(
        "[green]  Identicos. Tu motor calcula lo mismo que PyTorch.[/green]"
        if coincide
        else "[red]  No coinciden. Revisa las derivadas locales.[/red]"
    )


def experimento_2_la_acumulacion() -> None:
    console.rule("[bold]2. Por que el gradiente se ACUMULA[/bold]")
    console.print(
        "El caso minimo que distingue `+=` de `=`:  [cyan]y = x + x[/cyan]  con x = 3.\n"
        "Como y = 2x, la derivada correcta es 2.\n"
    )

    x = Value(3.0)
    (x + x).backward()
    console.print(f"  con acumulacion (`+=`) :  dy/dx = {x.grad:.1f}   [green](correcto)[/green]")
    console.print("  con asignacion (`=`)   :  dy/dx = 1.0   [red](mal: la segunda rama pisa a la primera)[/red]")

    console.print("\nY con un nodo usado 10 veces:")
    z = Value(1.5)
    total = z
    for _ in range(9):
        total = total + z
    total.backward()
    console.print(f"  x aparece 10 veces  ->  dL/dx = {z.grad:.1f}   [green](correcto: 10)[/green]")

    console.print(
        "\n[dim]Esto es exactamente por lo que existe optimizer.zero_grad(). Si no limpias\n"
        "los gradientes entre pasos, el paso N usa la suma de los gradientes de los pasos\n"
        "1 a N. No da ningun error: simplemente el modelo no aprende.[/dim]"
    )


def experimento_3_entrenar() -> tuple[list[float], list[float]]:
    console.rule("[bold]3. Entrenar un MLP con tu motor[/bold]")

    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    console.print("Cuatro puntos, dos clases (+1 / -1). MLP de 3 -> 8 -> 8 -> 1.\n")

    rapido = train_scalar_mlp(xs, ys, hidden=(8, 8), steps=100, lr=0.05, seed=0)
    lento = train_scalar_mlp(xs, ys, hidden=(8, 8), steps=100, lr=0.005, seed=0)

    console.print(f"  lr=0.05  :  perdida {rapido[0]:.4f}  ->  {rapido[-1]:.6f}")
    console.print(f"  lr=0.005 :  perdida {lento[0]:.4f}  ->  {lento[-1]:.6f}")
    console.print(
        "\n[dim]Mismo modelo, misma inicializacion, solo cambia el learning rate. Diez veces\n"
        "mas pequenyo y no le da tiempo a converger. Este hiperparametro es el que mas\n"
        "entrenamientos arruina, y lo veras otra vez en el modulo 11.[/dim]"
    )
    return rapido, lento


def main() -> None:
    experimento_1_el_grafo()
    experimento_2_la_acumulacion()
    rapido, lento = experimento_3_entrenar()

    fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))

    izq.plot(rapido, label="lr = 0.05")
    izq.plot(lento, label="lr = 0.005")
    izq.set_xlabel("paso")
    izq.set_ylabel("perdida (MSE)")
    izq.set_title("Entrenamiento con un motor de autodiff escrito a mano")
    izq.grid(alpha=0.3)
    izq.legend()

    der.plot(rapido, label="lr = 0.05")
    der.plot(lento, label="lr = 0.005")
    der.set_yscale("log")
    der.set_xlabel("paso")
    der.set_ylabel("perdida (escala log)")
    der.set_title("La misma curva en log: se ve la convergencia")
    der.grid(alpha=0.3)
    der.legend()

    fig.tight_layout()
    destino = figures_dir() / "02_autograd.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")


if __name__ == "__main__":
    main()
