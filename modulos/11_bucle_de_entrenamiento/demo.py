"""Demo del modulo 11: las cuatro piezas del bucle, comparadas y medidas.

    llmfs demo 11

Cuatro experimentos:
  1. Tu AdamW frente al de PyTorch: mismos pesos hasta el ultimo decimal.
  2. Que pasa sin correccion de sesgo, sin momento y sin escalado. Con curvas.
  3. El planificador, dibujado, y por que hace falta el warmup.
  4. El recorte de gradientes ante un batch toxico.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")  # sin ventana: esto tiene que correr por SSH y en CI

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.config import ModelConfig
from llmfs.device import set_seed
from llmfs.paths import figures_dir
from llmfs.reference import GPT

console = Console()

AdamWScratch = resolve("11_bucle_de_entrenamiento", "AdamWScratch")
lr_at_step = resolve("11_bucle_de_entrenamiento", "lr_at_step")
clip_grad_norm = resolve("11_bucle_de_entrenamiento", "clip_grad_norm")
build_param_groups = resolve("11_bucle_de_entrenamiento", "build_param_groups")


def tarea():
    """Una regresion no lineal cualquiera, para tener algo que optimizar."""
    set_seed(0)
    x = torch.randn(256, 16)
    y = (x[:, :1] * x[:, 1:2]).sin() + 0.5 * x[:, 2:3].abs()
    return x, y


def entrena(hacer_opt, pasos=200, lr=1e-2):
    set_seed(0)
    modelo = nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 1))
    opt = hacer_opt(modelo.parameters(), lr)
    x, y = tarea()
    historial = []
    for _ in range(pasos):
        perdida = ((modelo(x) - y) ** 2).mean()
        opt.zero_grad()
        perdida.backward()
        opt.step()
        historial.append(float(perdida.detach()))
    return historial, [p.detach().clone() for p in modelo.parameters()]


# ------------------------------------------------------------- 1. contra torch


def experimento_vs_torch():
    console.rule("[bold]1. Tu AdamW frente al de PyTorch[/bold]")

    h_mio, p_mio = entrena(
        lambda p, lr: AdamWScratch(p, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    )
    h_torch, p_torch = entrena(
        lambda p, lr: torch.optim.AdamW(p, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    )

    err = max(float((a - b).abs().max()) for a, b in zip(p_mio, p_torch))
    console.print(
        f"  perdida final:  tuya {h_mio[-1]:.8f}   torch {h_torch[-1]:.8f}\n"
        f"  error maximo en los pesos tras 200 pasos: [bold]{err:.2e}[/bold]\n"
    )
    console.print(
        "[green]Identicos salvo redondeo de fp32.[/green] Estas haciendo exactamente las "
        "mismas\noperaciones en el mismo orden."
        if err < 1e-4
        else "[red]Divergen. Revisa la correccion de sesgo o el weight decay.[/red]"
    )
    return h_mio, h_torch


# ------------------------------------------------------------- 2. las piezas


class SGDSimple(torch.optim.Optimizer):
    """Descenso de gradiente pelado, para comparar."""

    def __init__(self, params, lr=1e-2):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-g["lr"])


class AdamSinCorreccion(torch.optim.Optimizer):
    """Adam sin correccion de sesgo, para ver el destrozo de los primeros pasos."""

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.95), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                # SIN dividir por (1 - beta^t)
                p.addcdiv_(st["m"], st["v"].sqrt().add(g["eps"]), value=-g["lr"])


def experimento_piezas():
    console.rule("[bold]2. Que aporta cada pieza[/bold]")

    variantes = {
        "SGD (sin momento ni escalado)": lambda p, lr: SGDSimple(p, lr=lr),
        "Adam sin correccion de sesgo": lambda p, lr: AdamSinCorreccion(p, lr=lr),
        "AdamW completo (el tuyo)": lambda p, lr: AdamWScratch(p, lr=lr, betas=(0.9, 0.95)),
    }

    resultados = {}
    tabla = Table(header_style="bold")
    tabla.add_column("variante")
    tabla.add_column("perdida paso 1", justify="right")
    tabla.add_column("perdida paso 10", justify="right")
    tabla.add_column("perdida final", justify="right")

    for nombre, hacer in variantes.items():
        h, _ = entrena(hacer, pasos=200)
        resultados[nombre] = h
        tabla.add_row(nombre, f"{h[0]:.4f}", f"{h[9]:.4f}", f"{h[-1]:.6f}")
    console.print(tabla)

    sgd = resultados["SGD (sin momento ni escalado)"]
    completo = resultados["AdamW completo (el tuyo)"]
    console.print(
        f"\nAdamW llega a {completo[-1]:.6f} y SGD se queda en {sgd[-1]:.6f} con el mismo lr\n"
        "y los mismos pasos. Esa diferencia es el escalado por dimension: cada parametro\n"
        "acaba con su propio learning rate efectivo.\n\n"
        "[dim]Fijate en la columna del paso 10 de la version sin correccion de sesgo. Ahi es\n"
        "donde se nota: los primeros pasos dan saltos mayores de lo debido, porque m y v\n"
        "arrancan en cero y subestiman las magnitudes reales.[/dim]"
    )
    return resultados


# ------------------------------------------------------------- 3. el planificador


def experimento_scheduler():
    console.rule("[bold]3. El planificador del learning rate[/bold]")

    max_steps, lr, warmup = 10_172, 1e-3, 500
    pasos = list(range(0, max_steps + 500, 20))
    curva = [lr_at_step(s, max_steps, lr, warmup, 0.1) for s in pasos]

    tabla = Table(header_style="bold")
    tabla.add_column("paso", justify="right")
    tabla.add_column("lr", justify="right")
    tabla.add_column("tramo")
    for s, tramo in [
        (0, "warmup (arranca casi en cero)"),
        (250, "warmup (a mitad)"),
        (500, "fin del warmup: el maximo"),
        (2500, "coseno"),
        (5086, "coseno (a mitad de entrenamiento)"),
        (10_172, "fin: el suelo del 10%"),
        (12_000, "pasado el final: se queda en el suelo"),
    ]:
        tabla.add_row(f"{s:,}", f"{lr_at_step(s, max_steps, lr, warmup, 0.1):.3e}", tramo)
    console.print(tabla)

    console.print(
        "\n[bold]Por que el warmup.[/bold] En los primeros pasos los momentos de Adam estan\n"
        "casi vacios y sus estimaciones son ruidosas, y ademas los pesos recien\n"
        "inicializados dan gradientes grandes. Arrancar a lr completo suele producir un\n"
        "pico de perdida del que a veces el modelo no se recupera.\n\n"
        "[bold]Por que un coseno.[/bold] Baja despacio al principio (todavia interesa moverse\n"
        "rapido), deprisa en medio, y despacio al final (afinando).\n\n"
        "[bold]Por que no hasta cero.[/bold] Por debajo de cierto punto el modelo deja de\n"
        "aprender del todo y se desperdicia computo. El 10% es la convencion."
    )
    return pasos, curva


# ------------------------------------------------------------- 4. el recorte


def experimento_clip():
    console.rule("[bold]4. El recorte de gradientes[/bold]")

    set_seed(0)
    modelo = nn.Linear(64, 64)
    # Un "batch toxico": produce gradientes enormes
    ((modelo(torch.randn(8, 64)) ** 2).sum() * 500).backward()

    antes = torch.cat([p.grad.flatten().clone() for p in modelo.parameters()])
    norma = clip_grad_norm(modelo.parameters(), 1.0)
    despues = torch.cat([p.grad.flatten() for p in modelo.parameters()])

    coseno = float(torch.dot(antes, despues) / (antes.norm() * despues.norm()))
    console.print(
        f"  norma ANTES de recortar : [red]{norma:,.1f}[/red]\n"
        f"  norma DESPUES           : [green]{float(despues.norm()):.4f}[/green]\n"
        f"  coseno entre las dos direcciones: [bold]{coseno:.8f}[/bold]\n"
    )
    console.print(
        "Ese coseno de 1.0 es lo importante: [bold]la direccion no ha cambiado[/bold], solo\n"
        "la magnitud. El gradiente sigue apuntando exactamente a donde apuntaba.\n\n"
        "[dim]Si recortaras cada tensor por separado, cada uno se escalaria por un factor\n"
        "distinto y la direccion conjunta SI cambiaria. Por eso se usa la norma global.[/dim]"
    )

    # Y el efecto sobre un entrenamiento
    console.print("\n[bold]Sobre un entrenamiento con un batch envenenado en el paso 50:[/bold]\n")
    resultados = {}
    for etiqueta, umbral in (("sin recortar", 0.0), ("con grad_clip=1.0", 1.0)):
        set_seed(0)
        m = nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 1))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        x, y = tarea()
        hist = []
        for paso in range(150):
            xb, yb = (x, y) if paso != 50 else (x * 50, y * 50)  # el batch toxico
            perdida = ((m(xb) - yb) ** 2).mean()
            opt.zero_grad()
            perdida.backward()
            if umbral > 0:
                clip_grad_norm(m.parameters(), umbral)
            opt.step()
            hist.append(float(((m(x) - y) ** 2).mean().detach()))
        resultados[etiqueta] = hist
        console.print(
            f"  {etiqueta:<20} perdida en el paso 49: {hist[49]:.4f}  ->  "
            f"paso 55: {hist[55]:.4f}  ->  final: {hist[-1]:.4f}"
        )

    sin, con = resultados["sin recortar"], resultados["con grad_clip=1.0"]
    danyo_sin = sin[55] / max(sin[49], 1e-9)
    danyo_con = con[55] / max(con[49], 1e-9)
    veredicto = (
        f"  sin recortar     : la perdida SUBE {danyo_sin:.1f}x tras el batch toxico\n"
        + (
            f"  con grad_clip=1.0: ni se entera ({danyo_con:.1f}x, sigue bajando)\n"
            if danyo_con < 1.0
            else f"  con grad_clip=1.0: sube solo {danyo_con:.1f}x\n"
        )
    )
    console.print(
        f"\n{veredicto}"
        "[dim]Con grad_clip el danyo maximo de un batch cualquiera esta acotado. En una\n"
        "tirada de miles de pasos, un solo batch raro puede costarte el entrenamiento\n"
        "entero.[/dim]"
    )
    return resultados


# ------------------------------------------------------------- 5. param groups


def experimento_grupos():
    console.rule("[bold]5. Que parametros decaen[/bold]")
    modelo = GPT(ModelConfig())
    grupos = build_param_groups(modelo, 0.1)

    tabla = Table(header_style="bold")
    tabla.add_column("grupo")
    tabla.add_column("weight_decay", justify="right")
    tabla.add_column("tensores", justify="right")
    tabla.add_column("parametros", justify="right")
    tabla.add_column("que hay dentro")
    tabla.add_row(
        "con decay",
        f"{grupos[0]['weight_decay']}",
        str(len(grupos[0]["params"])),
        f"{sum(p.numel() for p in grupos[0]['params']):,}",
        "matrices: embeddings y proyecciones",
    )
    tabla.add_row(
        "sin decay",
        f"{grupos[1]['weight_decay']}",
        str(len(grupos[1]["params"])),
        f"{sum(p.numel() for p in grupos[1]['params']):,}",
        "escalas de RMSNorm",
    )
    console.print(tabla)
    console.print(
        "\nLa escala de un RMSNorm arranca en 1 y su trabajo es reescalar la salida de la\n"
        "capa. Empujarla hacia cero es empujar la salida hacia cero: exactamente lo\n"
        "contrario de lo que hace falta.\n\n"
        "[dim]Aplicar decay a todo es un error frecuente, no da ningun error visible, y\n"
        "degrada el resultado. Solo se detecta comparando dos entrenamientos completos.[/dim]"
    )


def main() -> None:
    h_mio, h_torch = experimento_vs_torch()
    variantes = experimento_piezas()
    pasos, curva = experimento_scheduler()
    clip = experimento_clip()
    experimento_grupos()

    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.2))

    for nombre, hist in variantes.items():
        ejes[0].plot(hist, label=nombre, lw=1.2)
    ejes[0].set_yscale("log")
    ejes[0].set_xlabel("paso")
    ejes[0].set_ylabel("perdida (escala log)")
    ejes[0].set_title("Que aporta cada pieza de Adam")
    ejes[0].grid(alpha=0.3)
    ejes[0].legend(fontsize=7)

    ejes[1].plot(pasos, curva, color="tab:orange")
    ejes[1].axvline(500, color="gray", ls="--", lw=1, label="fin del warmup")
    ejes[1].axhline(1e-4, color="red", ls=":", lw=1, label="suelo (10%)")
    ejes[1].set_xlabel("paso")
    ejes[1].set_ylabel("learning rate")
    ejes[1].set_title("Warmup + coseno")
    ejes[1].grid(alpha=0.3)
    ejes[1].legend(fontsize=8)

    for nombre, hist in clip.items():
        ejes[2].plot(hist, label=nombre, lw=1.2)
    ejes[2].axvline(50, color="red", ls="--", lw=1, label="batch toxico")
    ejes[2].set_yscale("log")
    ejes[2].set_xlabel("paso")
    ejes[2].set_ylabel("perdida (escala log)")
    ejes[2].set_title("Un batch raro, con y sin recorte")
    ejes[2].grid(alpha=0.3)
    ejes[2].legend(fontsize=8)

    fig.tight_layout()
    destino = figures_dir() / "11_bucle_de_entrenamiento.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")


if __name__ == "__main__":
    main()
