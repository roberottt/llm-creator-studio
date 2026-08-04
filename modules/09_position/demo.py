"""Demo del modulo 09: las frecuencias de RoPE y la extrapolacion a contextos largos.

    llmfs demo 09

Tres experimentos:
  1. Las frecuencias, dibujadas. Se ve la escalera de rapida a lenta.
  2. La invariancia relativa, comprobada con numeros: <R(m)q, R(n)k> solo depende de n-m.
  3. Entrenar tres modelos identicos salvo por la codificacion posicional (aprendida,
     sinusoidal y RoPE) con contexto 32, y evaluarlos con contextos de hasta 128.
     Tarda unos 30 s.
"""

from __future__ import annotations

import math
import time

import matplotlib

matplotlib.use("Agg")  # sin ventana: esto tiene que correr por SSH y en CI

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.data import fetch_tinyshakespeare
from llmfs.device import get_device, set_seed
from llmfs.paths import figures_dir
from llmfs.reference import MultiHeadAttention, RMSNorm

console = Console()

sinusoidal_embeddings = resolve("09_position", "sinusoidal_embeddings")
rope_frequencies = resolve("09_position", "rope_frequencies")
apply_rope = resolve("09_position", "apply_rope")

CTX_ENTRENAMIENTO = 32
CONTEXTOS_EVAL = [8, 16, 32, 48, 64, 96, 128]


def experimento_frecuencias() -> tuple[torch.Tensor, torch.Tensor]:
    console.rule("[bold]1. La escalera de frecuencias[/bold]")
    cos, sin = rope_frequencies(64, 256)

    tabla = Table(header_style="bold")
    tabla.add_column("par de dimensiones", justify="right")
    tabla.add_column("frecuencia (rad/posicion)", justify="right")
    tabla.add_column("da una vuelta completa cada", justify="right")

    for i in (0, 4, 8, 16, 24, 31):
        # El angulo en la posicion 1 es exactamente la frecuencia
        freq = float(torch.acos(cos[1, i].clamp(-1, 1)))
        periodo = 2 * math.pi / freq if freq > 1e-12 else float("inf")
        tabla.add_row(
            f"{i}", f"{freq:.6f}", f"{periodo:,.0f} posiciones" if periodo < 1e9 else "nunca"
        )
    console.print(tabla)
    console.print(
        "[dim]Los primeros pares giran deprisa y distinguen posiciones vecinas; los ultimos\n"
        "giran tan despacio que dentro del contexto entrenado apenas completan una fraccion\n"
        "de vuelta, y capturan distancias largas.\n\n"
        "Esa ultima fila es la clave de las limitaciones de RoPE: si el modelo solo ha visto\n"
        "512 posiciones, los angulos grandes de los pares lentos son territorio no visto.[/dim]"
    )
    return cos, sin


def experimento_invariancia(cos: torch.Tensor, sin: torch.Tensor) -> None:
    console.rule("[bold]2. La invariancia relativa[/bold]")
    console.print(
        "La propiedad que justifica RoPE:  <R(m)q, R(n)k>  depende SOLO de n-m.\n\n"
        "Se coge el MISMO par de vectores q y k, se colocan en posiciones distintas y se\n"
        "mide la puntuacion de atencion entre ellos.\n"
    )

    set_seed(0)
    head_dim = 64
    q_base, k_base = torch.randn(head_dim), torch.randn(head_dim)

    def puntuacion(pos_q: int, pos_k: int) -> float:
        q = torch.zeros(1, 1, 256, head_dim)
        k = torch.zeros(1, 1, 256, head_dim)
        q[0, 0, pos_q] = q_base
        k[0, 0, pos_k] = k_base
        return float(apply_rope(q, cos, sin)[0, 0, pos_q] @ apply_rope(k, cos, sin)[0, 0, pos_k])

    tabla = Table(header_style="bold")
    tabla.add_column("posiciones (q, k)")
    tabla.add_column("distancia", justify="right")
    tabla.add_column("puntuacion", justify="right")

    for pq, pk in [(0, 3), (2, 5), (10, 13), (100, 103), (200, 203)]:
        tabla.add_row(f"({pq}, {pk})", str(pk - pq), f"{puntuacion(pq, pk):.10f}")
    tabla.add_section()
    for pq, pk in [(0, 7), (50, 57)]:
        tabla.add_row(f"({pq}, {pk})", str(pk - pq), f"{puntuacion(pq, pk):.10f}")
    console.print(tabla)

    console.print(
        "[bold]Las cinco primeras filas dan el mismo numero hasta el ultimo decimal.[/bold]\n"
        "Todas estan a distancia 3, aunque una este al principio de la secuencia y otra en\n"
        "la posicion 200.\n\n"
        "Eso significa que el modelo no aprende 'el token numero 3': aprende 'el token de\n"
        "tres posiciones atras'. Y por eso puede aplicar lo aprendido en cualquier parte de\n"
        "la secuencia.\n\n"
        "[dim]Las dos ultimas filas, a distancia 7, dan un valor distinto pero tambien igual\n"
        "entre si.[/dim]"
    )


# ------------------------------------------------------------- 3. extrapolacion


class ModeloPosicional(nn.Module):
    """Un transformer de una capa. Lo unico que cambia entre variantes es la posicion."""

    def __init__(self, vocab: int, d_model: int, n_heads: int, tipo: str, max_ctx: int) -> None:
        super().__init__()
        self.tipo = tipo
        self.token_embedding = nn.Embedding(vocab, d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

        if tipo == "aprendida":
            self.pos_embedding = nn.Embedding(max_ctx, d_model)
        elif tipo == "sinusoidal":
            self.register_buffer("tabla", sinusoidal_embeddings(4096, d_model), persistent=False)
        elif tipo == "rope":
            cos, sin = rope_frequencies(d_model // n_heads, 4096)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        seq = idx.shape[1]
        x = self.token_embedding(idx)

        cos = sin = None
        if self.tipo == "aprendida":
            if seq > self.pos_embedding.num_embeddings:
                raise IndexError("posicion fuera de la tabla aprendida")
            x = x + self.pos_embedding(torch.arange(seq, device=idx.device))
        elif self.tipo == "sinusoidal":
            x = x + self.tabla[:seq].to(x.device)
        elif self.tipo == "rope":
            cos, sin = self.rope_cos, self.rope_sin

        x = x + self.attn(self.norm(x), cos=cos, sin=sin)
        logits = self.head(x)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def experimento_extrapolacion(cfg) -> dict[str, list[float | None]]:
    console.rule("[bold]3. Extrapolacion a contextos mas largos que el entrenado[/bold]")

    texto, _ = fetch_tinyshakespeare()
    texto = texto[:120_000].lower()
    vocab_chars = sorted(set(texto))
    stoi = {c: i for i, c in enumerate(vocab_chars)}
    datos = torch.tensor([stoi[c] for c in texto], dtype=torch.long)
    corte = int(len(datos) * 0.9)
    train, val = datos[:corte], datos[corte:]

    console.print(
        f"Tres modelos identicos salvo por la codificacion posicional.\n"
        f"Entrenados con contexto {CTX_ENTRENAMIENTO}, evaluados con contextos de "
        f"{CONTEXTOS_EVAL[0]} a {CONTEXTOS_EVAL[-1]}.\n"
    )

    resultados: dict[str, list[float | None]] = {}
    for tipo in ("aprendida", "sinusoidal", "rope"):
        set_seed(0)
        modelo = ModeloPosicional(
            len(vocab_chars), d_model=64, n_heads=4, tipo=tipo, max_ctx=CTX_ENTRENAMIENTO
        ).to(cfg.device)
        opt = torch.optim.AdamW(modelo.parameters(), lr=3e-3)

        inicio = time.perf_counter()
        for _ in range(400):
            i = torch.randint(0, len(train) - CTX_ENTRENAMIENTO - 1, (64,))
            x = torch.stack([train[j : j + CTX_ENTRENAMIENTO] for j in i]).to(cfg.device)
            y = torch.stack([train[j + 1 : j + 1 + CTX_ENTRENAMIENTO] for j in i]).to(cfg.device)
            _, perdida = modelo(x, y)
            opt.zero_grad(set_to_none=True)
            perdida.backward()
            opt.step()
        transcurrido = time.perf_counter() - inicio

        modelo.eval()
        perdidas: list[float | None] = []
        for ctx in CONTEXTOS_EVAL:
            try:
                with torch.no_grad():
                    total, n = 0.0, 0
                    for _ in range(20):
                        i = torch.randint(0, len(val) - ctx - 1, (16,))
                        x = torch.stack([val[j : j + ctx] for j in i]).to(cfg.device)
                        y = torch.stack([val[j + 1 : j + 1 + ctx] for j in i]).to(cfg.device)
                        total += float(modelo(x, y)[1])
                        n += 1
                    perdidas.append(total / n)
            except (IndexError, RuntimeError):
                perdidas.append(None)  # no puede procesar esa longitud
        resultados[tipo] = perdidas
        console.print(f"  [dim]{tipo} entrenado en {transcurrido:.1f} s[/dim]")

    tabla = Table(header_style="bold")
    tabla.add_column("contexto", justify="right")
    for tipo in resultados:
        tabla.add_column(tipo, justify="right")
    for i, ctx in enumerate(CONTEXTOS_EVAL):
        marca = " (entrenado)" if ctx == CTX_ENTRENAMIENTO else ""
        fila = [f"{ctx}{marca}"]
        for tipo in resultados:
            v = resultados[tipo][i]
            fila.append("[red]no puede[/red]" if v is None else f"{v:.4f}")
        tabla.add_row(*fila, style="bold" if ctx == CTX_ENTRENAMIENTO else "")
    console.print(tabla)

    idx_entrenado = CONTEXTOS_EVAL.index(CTX_ENTRENAMIENTO)
    console.print(
        "\n[bold]Lo que hay que mirar:[/bold]\n\n"
        "1. La codificacion [bold]aprendida[/bold] literalmente [red]no puede[/red] procesar "
        "secuencias mas\n   largas que las entrenadas: no hay fila en la tabla que "
        "consultar. Es un techo duro.\n\n"
        "2. [bold]sinusoidal[/bold] y [bold]rope[/bold] si producen una respuesta para "
        "cualquier longitud.\n"
    )

    for tipo in ("sinusoidal", "rope"):
        base = resultados[tipo][idx_entrenado]
        largo = resultados[tipo][-1]
        if base is not None and largo is not None:
            console.print(
                f"   {tipo}: de {base:.4f} en contexto {CTX_ENTRENAMIENTO} a {largo:.4f} en "
                f"{CONTEXTOS_EVAL[-1]}  ([bold]{100 * (largo / base - 1):+.0f}%[/bold])"
            )

    console.print(
        "\n[bold yellow]Y aqui la parte honesta:[/bold yellow] 'poder procesar' no es lo "
        "mismo que 'funcionar bien'.\nSe degradan los dos. RoPE tiene la propiedad relativa, "
        "pero las frecuencias lentas\napenas completan una fraccion de vuelta dentro del "
        "rango entrenado, asi que los\nangulos grandes son territorio no visto.\n\n"
        "[dim]Por eso existe toda una familia de tecnicas para extender el contexto DESPUES\n"
        "de entrenar (interpolacion de posiciones, NTK-aware scaling, YaRN): porque la\n"
        "extrapolacion directa no basta. Cuando leas que 'RoPE extrapola', esto es lo que\n"
        "hay detras.[/dim]"
    )
    return resultados


def main() -> None:
    cfg = get_device()
    cos, sin = experimento_frecuencias()
    experimento_invariancia(cos, sin)
    resultados = experimento_extrapolacion(cfg)

    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.2))

    for i, par in enumerate((0, 4, 8, 16, 31)):
        ejes[0].plot(cos[:128, par].numpy(), label=f"par {par}", lw=1.2)
    ejes[0].set_xlabel("posicion")
    ejes[0].set_ylabel("cos(angulo)")
    ejes[0].set_title("Las frecuencias de RoPE")
    ejes[0].grid(alpha=0.3)
    ejes[0].legend(fontsize=7)

    im = ejes[1].imshow(cos[:64, :32].numpy(), aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    ejes[1].set_xlabel("dimension (primera mitad)")
    ejes[1].set_ylabel("posicion")
    ejes[1].set_title("La tabla de cosenos")
    fig.colorbar(im, ax=ejes[1], fraction=0.046)

    for tipo, perdidas in resultados.items():
        xs = [c for c, v in zip(CONTEXTOS_EVAL, perdidas) if v is not None]
        ys = [v for v in perdidas if v is not None]
        ejes[2].plot(xs, ys, marker="o", label=tipo)
    ejes[2].axvline(
        CTX_ENTRENAMIENTO, color="gray", ls="--", lw=1.5, label="contexto de entrenamiento"
    )
    ejes[2].set_xlabel("contexto en evaluacion")
    ejes[2].set_ylabel("perdida en validacion")
    ejes[2].set_title("Extrapolacion mas alla de lo entrenado")
    ejes[2].grid(alpha=0.3)
    ejes[2].legend(fontsize=8)

    fig.tight_layout()
    destino = figures_dir() / "09_position.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")


if __name__ == "__main__":
    main()
