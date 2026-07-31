"""Demo del modulo 08: por que hace falta una no-linealidad, y cual.

    llmfs demo 08

Cuatro experimentos:
  1. El colapso: una red sin activaciones equivale a UNA sola matriz, medido.
  2. Las curvas de ReLU, GELU y Swish, y sus derivadas. Ahi se ve el problema de ReLU.
  3. El reparto de parametros: dos tercios del modelo son FFN.
  4. SwiGLU frente al FFN clasico, entrenando los dos con los mismos parametros.
"""

from __future__ import annotations

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
from llmfs.reference import FeedForwardMLP

console = Console()

gelu = resolve("08_mlp_y_activaciones", "gelu")
swiglu_hidden_dim = resolve("08_mlp_y_activaciones", "swiglu_hidden_dim")
SwiGLU = resolve("08_mlp_y_activaciones", "SwiGLU")


def experimento_colapso() -> None:
    console.rule("[bold]1. Una red sin activaciones es UNA capa[/bold]")
    console.print(
        "Se apilan 5 capas lineales sin nada en medio y se comparan con el producto de sus\n"
        "cinco matrices, que es una sola matriz.\n"
    )

    set_seed(0)
    d = 32
    capas = [nn.Linear(d, d, bias=False) for _ in range(5)]
    x = torch.randn(4, d)

    con_capas = x
    for capa in capas:
        con_capas = capa(con_capas)

    # El producto de las cinco matrices, en una sola
    equivalente = capas[0].weight
    for capa in capas[1:]:
        equivalente = capa.weight @ equivalente
    con_una = x @ equivalente.T

    error = float((con_capas - con_una).abs().max())
    console.print(
        f"  5 capas apiladas vs 1 sola matriz  ->  diferencia maxima: [bold]{error:.2e}[/bold]\n"
        "  [dim](o sea, cero salvo redondeo de coma flotante)[/dim]\n"
    )

    # Y ahora con activacion
    con_gelu = x
    for capa in capas:
        con_gelu = gelu(capa(con_gelu))
    error_gelu = float((con_gelu - con_una).abs().max())
    console.print(
        f"  Las mismas 5 capas CON GELU vs 1 matriz  ->  diferencia: [bold]{error_gelu:.3f}[/bold]\n\n"
        "[bold]Eso es todo el argumento.[/bold] Sin no-linealidad, cien capas equivalen a una\n"
        "y toda la profundidad se derrumba. La atencion es una media ponderada, o sea lineal,\n"
        "asi que el FFN es lo unico que impide que el Transformer entero colapse."
    )


def experimento_activaciones() -> tuple:
    console.rule("[bold]2. ReLU, GELU y Swish[/bold]")

    x = torch.linspace(-4, 4, 400, requires_grad=True)
    curvas = {
        "ReLU": F.relu(x),
        "GELU": gelu(x),
        "Swish/SiLU": F.silu(x),
    }
    derivadas = {}
    for nombre, y in curvas.items():
        g = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
        derivadas[nombre] = g.detach().clone()

    tabla = Table(header_style="bold")
    tabla.add_column("x", justify="right")
    for nombre in curvas:
        tabla.add_column(f"{nombre}", justify="right")
        tabla.add_column(f"d{nombre}/dx", justify="right")

    for valor in (-3.0, -1.0, 0.0, 1.0, 3.0):
        idx = int((valor + 4) / 8 * 399)
        fila = [f"{valor:+.1f}"]
        for nombre in curvas:
            fila.append(f"{float(curvas[nombre][idx]):+.4f}")
            fila.append(f"{float(derivadas[nombre][idx]):+.4f}")
        tabla.add_row(*fila)
    console.print(tabla)

    console.print(
        "[bold]Mira la columna dReLU/dx en las filas negativas: vale 0 exacto.[/bold]\n"
        "Una neurona que acabe dando siempre valores negativos deja de recibir gradiente\n"
        "para siempre. Esta muerta y no hay forma de recuperarla.\n\n"
        "GELU y Swish tienen derivada pequenya pero NO nula en esa zona, asi que una\n"
        "neurona en negativo todavia puede volver.\n\n"
        "[dim]Fijate tambien en que GELU y Swish son casi la misma curva. Swish sale de una\n"
        "busqueda automatica de arquitecturas y GELU de un argumento probabilistico, y\n"
        "acabaron practicamente en el mismo sitio.[/dim]"
    )
    return x.detach(), curvas, derivadas


def experimento_parametros() -> None:
    console.rule("[bold]3. Donde estan los parametros[/bold]")

    tabla = Table(header_style="bold")
    tabla.add_column("d_model", justify="right")
    tabla.add_column("d_ff (SwiGLU)", justify="right")
    tabla.add_column("atencion (4d²)", justify="right")
    tabla.add_column("FFN (3·d·d_ff)", justify="right")
    tabla.add_column("% FFN", justify="right")

    for d in (128, 320, 768, 4096):
        d_ff = swiglu_hidden_dim(d)
        atn = 4 * d * d
        ffn = 3 * d * d_ff
        tabla.add_row(
            str(d), str(d_ff), f"{atn:,}", f"{ffn:,}", f"{100 * ffn / (atn + ffn):.0f}%"
        )
    console.print(tabla)
    console.print(
        "[bold]Dos tercios del modelo son FFN, no atencion.[/bold] Cuando leas que un modelo\n"
        "tiene N parametros, la mayoria estan aqui.\n\n"
        "[dim]Y el 896 de la fila de d_model=320 es el d_ff del config final: "
        "ceil_64(2/3 · 4 · 320) = ceil_64(853,3) = 896.[/dim]"
    )

    console.print("\n[bold]El presupuesto de SwiGLU frente al FFN clasico:[/bold]")
    d = 320
    clasico = 2 * d * 4 * d
    swiglu_igual = 3 * d * swiglu_hidden_dim(d)
    swiglu_sin_ajuste = 3 * d * 4 * d
    console.print(
        f"  FFN clasico (2 matrices, d_ff=4d=1280) : {clasico:,}\n"
        f"  SwiGLU sin ajustar (3 matrices, d_ff=1280): {swiglu_sin_ajuste:,}  "
        f"[red](+{100 * swiglu_sin_ajuste / clasico - 100:.0f}%)[/red]\n"
        f"  SwiGLU con el 2/3 (d_ff=896)            : {swiglu_igual:,}  "
        f"[green]({100 * swiglu_igual / clasico - 100:+.0f}%)[/green]\n\n"
        "[dim]Ese es el motivo del factor 2/3: tres matrices en vez de dos costarian un 50%\n"
        "mas, asi que se reduce el hidden para gastar lo mismo y comparar de forma justa.[/dim]"
    )


def experimento_entrenamiento(cfg) -> tuple[list[float], list[float]]:
    console.rule("[bold]4. SwiGLU frente al FFN clasico, a igualdad de parametros[/bold]")

    set_seed(0)
    d, n = 256, 4096
    # Una funcion no lineal dificil de aproximar, para que no la resuelvan las dos al
    # instante y la comparacion diga algo.
    x = torch.randn(n, d, device=cfg.device)
    objetivo = (
        (3 * x[:, :1]).sin() * (2 * x[:, 1:2]).cos()
        + (x[:, 2:3] * x[:, 3:4]).tanh()
        + 0.5 * x[:, 4:5].abs()
    )

    resultados = {}
    parametros = {}
    for nombre, modelo in [
        ("FFN clasico + GELU", FeedForwardMLP(d, 4 * d)),
        ("SwiGLU", SwiGLU(d, swiglu_hidden_dim(d))),
    ]:
        set_seed(1)
        modelo = nn.Sequential(modelo, nn.Linear(d, 1)).to(cfg.device)
        parametros[nombre] = sum(p.numel() for p in modelo.parameters())
        opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3)
        historial = []
        for _ in range(300):
            perdida = F.mse_loss(modelo(x), objetivo)
            opt.zero_grad(set_to_none=True)
            perdida.backward()
            opt.step()
            historial.append(float(perdida.detach()))
        resultados[nombre] = historial
        console.print(
            f"  {nombre:<22} {parametros[nombre]:>9,} parametros  ->  "
            f"perdida final {historial[-1]:.4e}"
        )

    clasico, swiglu = list(resultados.values())
    desvio = 100 * abs(parametros["SwiGLU"] / parametros["FFN clasico + GELU"] - 1)
    console.print(
        f"\n[dim]Los parametros no son exactamente iguales: SwiGLU lleva un {desvio:.0f}% mas,\n"
        "porque el redondeo de d_ff al multiplo de 64 no cae justo. El 2/3 iguala los\n"
        "presupuestos de forma asintotica, no exacta.[/dim]\n"
    )

    diferencia = abs(swiglu[-1] - clasico[-1]) / max(clasico[-1], 1e-12)
    if diferencia < 0.1:
        console.print(
            "[bold yellow]Las dos perdidas quedan a menos de un 10% la una de la otra.[/bold yellow]\n"
            "Con esa diferencia y una sola semilla, este experimento NO distingue entre las\n"
            "dos arquitecturas. Decir que gana una seria leer ruido."
        )
    else:
        mejor = "SwiGLU" if swiglu[-1] < clasico[-1] else "el FFN clasico"
        console.print(f"En esta tarea concreta gana [bold]{mejor}[/bold], por un {100 * diferencia:.0f}%.")
    console.print(
        "\n[dim]Y en cualquier caso: un experimento de juguete con una tarea inventada NO\n"
        "demuestra nada sobre modelos de lenguaje. Shazeer (2020) probo todas las variantes\n"
        "GLU entrenando transformers de verdad, y SwiGLU salio mejor de forma consistente.\n"
        "Su explicacion de por que, citada literalmente del paper: 'We offer no explanation\n"
        "as to why these architectures seem to work; we attribute their success, as all\n"
        "else, to divine benevolence.' Es una de las decisiones de arquitectura mas usadas\n"
        "y peor entendidas del campo.[/dim]"
    )
    return clasico, swiglu


def main() -> None:
    cfg = get_device()
    experimento_colapso()
    x, curvas, derivadas = experimento_activaciones()
    experimento_parametros()
    clasico, swiglu = experimento_entrenamiento(cfg)

    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.2))

    for nombre, y in curvas.items():
        ejes[0].plot(x.numpy(), y.detach().numpy(), label=nombre)
    ejes[0].axhline(0, color="black", lw=0.6)
    ejes[0].axvline(0, color="black", lw=0.6)
    ejes[0].set_title("Las activaciones")
    ejes[0].grid(alpha=0.3)
    ejes[0].legend(fontsize=8)

    for nombre, g in derivadas.items():
        ejes[1].plot(x.numpy(), g.numpy(), label=nombre)
    ejes[1].axhline(0, color="black", lw=0.6)
    ejes[1].set_title("Sus derivadas (ReLU vale 0 en negativo)")
    ejes[1].grid(alpha=0.3)
    ejes[1].legend(fontsize=8)

    ejes[2].plot(clasico, label="FFN clasico + GELU")
    ejes[2].plot(swiglu, label="SwiGLU")
    ejes[2].set_yscale("log")
    ejes[2].set_xlabel("paso")
    ejes[2].set_ylabel("perdida (MSE, escala log)")
    ejes[2].set_title("A igualdad de parametros")
    ejes[2].grid(alpha=0.3)
    ejes[2].legend(fontsize=8)

    fig.tight_layout()
    destino = figures_dir() / "08_mlp_y_activaciones.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")


if __name__ == "__main__":
    main()
