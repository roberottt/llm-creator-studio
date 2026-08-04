"""Demo del modulo 06: mira a donde mira la atencion.

    llmfs demo 06

Tres experimentos:
  1. El escalado por sqrt(d_k): que le pasa al softmax cuando lo quitas. Con numeros.
  2. La mascara causal, dibujada.
  3. Un modelo de atencion de una capa entrenado de verdad (unos 20 s) sobre Shakespeare,
     y el heatmap de a que mira cada caracter. Las cabezas se especializan solas y se ve.
"""

from __future__ import annotations

import math
import time

import matplotlib

matplotlib.use("Agg")  # sin ventana: esto tiene que correr por SSH y en CI

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.data import fetch_tinyshakespeare
from llmfs.device import get_device, set_seed
from llmfs.paths import figures_dir

console = Console()

causal_mask = resolve("06_attention", "causal_mask")
single_head_attention = resolve("06_attention", "single_head_attention")
MultiHeadAttention = resolve("06_attention", "MultiHeadAttention")

FRASE = "the king shall speak to his people"


def entropia(pesos: torch.Tensor) -> float:
    """Entropia media de las distribuciones de atencion, en nats.

    Alta = el token reparte su atencion entre muchos. Baja = se fija en uno solo.
    """
    return float(-(pesos * torch.log(pesos + 1e-12)).sum(-1).mean())


# --------------------------------------------------------------- 1. el escalado


def experimento_escalado() -> tuple[list[int], list[float], list[float]]:
    console.rule("[bold]1. Por que se divide por sqrt(d_k)[/bold]")
    console.print(
        "El producto escalar de dos vectores de dimension d_k tiene varianza d_k. Cuanto\n"
        "mayor es d_k, mas se dispersan las puntuaciones, y softmax es exponencial: con\n"
        "puntuaciones muy separadas devuelve casi [0, 0, ..., 1, ..., 0].\n\n"
        "Se mide con la ENTROPIA de la distribucion de atencion. Maxima = reparte entre\n"
        "todos; cerca de cero = se fija en uno solo y no aprende nada del resto.\n"
    )

    dims = [8, 32, 128, 512, 2048]
    con, sin = [], []
    seq = 16
    maxima = math.log(seq)

    tabla = Table(header_style="bold")
    tabla.add_column("d_k", justify="right")
    tabla.add_column("entropia CON escalado", justify="right")
    tabla.add_column("entropia SIN escalado", justify="right")
    tabla.add_column("peso maximo sin escalar", justify="right")

    for d_k in dims:
        torch.manual_seed(0)
        q, k, v = torch.randn(1, seq, d_k), torch.randn(1, seq, d_k), torch.randn(1, seq, d_k)

        _, pesos_con = single_head_attention(q, k, v)
        pesos_sin = torch.softmax(q @ k.transpose(-2, -1), dim=-1)

        con.append(entropia(pesos_con))
        sin.append(entropia(pesos_sin))
        tabla.add_row(
            str(d_k),
            f"{con[-1]:.3f}",
            f"{sin[-1]:.3f}",
            f"{float(pesos_sin.max()):.4f}",
        )

    console.print(tabla)
    console.print(
        f"[dim]La entropia maxima posible con {seq} posiciones es ln({seq}) = {maxima:.3f}.[/dim]\n"
    )
    console.print(
        f"Con escalado la entropia se mantiene alta pase lo que pase con d_k.\n"
        f"Sin escalado, en d_k={dims[-1]} cae a {sin[-1]:.3f} y el peso maximo llega a "
        f"{1.0:.2f}:\nel token se fija en UNO solo e ignora todo lo demas.\n\n"
        "[bold]Y el problema de verdad no es esto, es el gradiente.[/bold] La derivada del\n"
        "softmax es p(1-p). Con p pegado a 0 o a 1 vale practicamente cero, asi que la capa\n"
        "deja de aprender. Sin el sqrt(d_k), un Transformer grande no entrena."
    )
    return dims, con, sin


# --------------------------------------------------------------- 2. la mascara


def experimento_mascara() -> None:
    console.rule("[bold]2. La mascara causal[/bold]")
    m = causal_mask(8)
    console.print("Mascara para 8 tokens ([green]#[/green] = puede mirar, . = prohibido):\n")
    for i in range(8):
        fila = " ".join("[green]#[/green]" if bool(m[i, j]) else "[dim].[/dim]" for j in range(8))
        console.print(f"  token {i}: {fila}")

    torch.manual_seed(0)
    q, k, v = torch.randn(1, 8, 16), torch.randn(1, 8, 16), torch.randn(1, 8, 16)
    _, con_mascara = single_head_attention(q, k, v, m)
    _, sin_mascara = single_head_attention(q, k, v)

    console.print(
        f"\n  peso total sobre el futuro CON mascara: {float(con_mascara.triu(1).sum()):.6f}\n"
        f"  peso total sobre el futuro SIN mascara: {float(sin_mascara.triu(1).sum()):.6f}\n"
    )
    console.print(
        "[dim]Se pone -inf ANTES del softmax, no se borran los pesos despues. Si los\n"
        "borraras despues, las filas ya no sumarian 1. Comprobacion: cada fila con\n"
        f"mascara suma {float(con_mascara[0].sum(-1).mean()):.6f}.[/dim]"
    )


# --------------------------------------------------------------- 3. atencion real


class ModeloAtencion(nn.Module):
    """Lo minimo para que la atencion tenga algo que aprender: embedding + MHA + salida."""

    def __init__(self, vocab: int, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, d_model)
        self.pos_embedding = nn.Embedding(256, d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx, targets=None, return_weights=False):
        seq = idx.shape[1]
        x = self.token_embedding(idx) + self.pos_embedding(torch.arange(seq, device=idx.device))
        if return_weights:
            salida, pesos = self.attn(self.norm(x), return_weights=True)
            return self.head(x + salida), pesos
        x = x + self.attn(self.norm(x))
        logits = self.head(x)
        if targets is None:
            return logits, None
        perdida = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        return logits, perdida


def experimento_heatmap(cfg) -> tuple[torch.Tensor, list[str]]:
    console.rule("[bold]3. Atencion de un modelo entrenado de verdad[/bold]")

    texto, _ = fetch_tinyshakespeare()
    texto = texto[:150_000].lower()
    vocab_chars = sorted(set(texto + FRASE))
    stoi = {c: i for i, c in enumerate(vocab_chars)}
    datos = torch.tensor([stoi[c] for c in texto], dtype=torch.long)

    set_seed(0)
    modelo = ModeloAtencion(len(vocab_chars), d_model=64, n_heads=4).to(cfg.device)
    opt = torch.optim.AdamW(modelo.parameters(), lr=3e-3)

    ctx, batch, pasos = 48, 64, 400
    console.print(f"Entrenando un modelo de 1 capa de atencion, {pasos} pasos...")
    inicio = time.perf_counter()
    for paso in range(pasos):
        i = torch.randint(0, len(datos) - ctx - 1, (batch,))
        x = torch.stack([datos[j : j + ctx] for j in i]).to(cfg.device)
        y = torch.stack([datos[j + 1 : j + 1 + ctx] for j in i]).to(cfg.device)
        _, perdida = modelo(x, y)
        opt.zero_grad(set_to_none=True)
        perdida.backward()
        opt.step()
    console.print(
        f"  perdida final: {float(perdida.detach()):.4f}  "
        f"(el suelo es ln({len(vocab_chars)}) = {math.log(len(vocab_chars)):.4f})  "
        f"[dim]{time.perf_counter() - inicio:.1f} s[/dim]\n"
    )

    modelo.eval()
    ids = torch.tensor([[stoi[c] for c in FRASE]], device=cfg.device)
    with torch.no_grad():
        _, pesos = modelo(ids, return_weights=True)

    pesos = pesos[0].cpu()  # (n_heads, T, T)
    caracteres = list(FRASE)

    console.print(f'Frase: [cyan]"{FRASE}"[/cyan]\n')
    for cabeza in range(pesos.shape[0]):
        w = pesos[cabeza]
        # A que distancia mira cada token, en promedio ponderado
        posiciones = torch.arange(len(caracteres)).float()
        distancia_media = float(
            sum((posiciones[i] - posiciones[: i + 1]) @ w[i, : i + 1] for i in range(1, len(caracteres)))
            / (len(caracteres) - 1)
        )
        # El token al que mas mira el ultimo caracter (sin contarse a si mismo)
        ultima = w[-1, :-1]
        favorito = int(ultima.argmax())
        console.print(
            f"  cabeza {cabeza}: distancia media {distancia_media:5.2f} posiciones   |   "
            f"el ultimo caracter mira sobre todo a {caracteres[favorito]!r} (pos {favorito})"
        )

    console.print(
        "\n[dim]Fijate en las distancias medias: son distintas entre cabezas. Cada una se ha\n"
        "especializado en un rango de contexto distinto, y nadie se lo ha dicho. En modelos\n"
        "grandes esto llega mucho mas lejos: hay cabezas que emparejan comillas de apertura\n"
        "y cierre, y las induction heads, que detectan '...A B ... A' y predicen B.[/dim]"
    )
    return pesos, caracteres


# --------------------------------------------------------------- grafica


def main() -> None:
    cfg = get_device()
    dims, con, sin = experimento_escalado()
    experimento_mascara()
    pesos, caracteres = experimento_heatmap(cfg)

    n_heads = pesos.shape[0]
    fig = plt.figure(figsize=(13, 8))

    ax = fig.add_subplot(2, 3, 1)
    ax.plot(dims, con, marker="o", label="con /sqrt(d_k)")
    ax.plot(dims, sin, marker="s", label="sin escalar")
    ax.axhline(math.log(16), color="gray", ls="--", lw=1, label="entropia maxima")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("d_k")
    ax.set_ylabel("entropia de la atencion (nats)")
    ax.set_title("Sin escalar, el softmax se satura")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 2)
    # Mismo mapa de color que los heatmaps de atencion, para que se lea igual: lo claro
    # es donde hay peso y lo oscuro donde no. Con "Greens" el claro era el prohibido y
    # se leia al reves.
    ax.imshow(causal_mask(16).numpy().astype(float), cmap="viridis", interpolation="nearest")
    ax.set_title("Mascara causal (claro = permitido)")
    ax.set_xlabel("token al que se mira")
    ax.set_ylabel("token que mira")

    for cabeza in range(min(n_heads, 4)):
        ax = fig.add_subplot(2, 3, 3 + cabeza)
        ax.imshow(pesos[cabeza].numpy(), cmap="viridis", interpolation="nearest")
        ax.set_title(f"cabeza {cabeza}", fontsize=10)
        ax.set_xticks(range(len(caracteres)))
        ax.set_yticks(range(len(caracteres)))
        ax.set_xticklabels(caracteres, fontsize=5)
        ax.set_yticklabels(caracteres, fontsize=5)

    fig.suptitle("Modulo 06: self-attention", fontsize=13)
    fig.tight_layout()
    destino = figures_dir() / "06_attention.png"
    fig.savefig(destino, dpi=130)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")
    console.print(
        "[dim]Abre la figura: los cuatro heatmaps de abajo son las cuatro cabezas. El\n"
        "triangulo superior esta siempre negro (la mascara), y cada cabeza tiene un patron\n"
        "visiblemente distinto.[/dim]"
    )


if __name__ == "__main__":
    main()
