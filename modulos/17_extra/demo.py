"""Demo del modulo 17: cuantiza tu modelo y mide el danyo. Y el cierre del curso.

    llmfs demo 17
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.config import ModelConfig, RunConfig
from llmfs.device import get_device, set_seed
from llmfs.paths import configs_dir, figures_dir
from llmfs.reference import GPT

console = Console()

quantize_int8_symmetric = resolve("17_extra", "quantize_int8_symmetric")
dequantize_int8 = resolve("17_extra", "dequantize_int8")
quantization_error = resolve("17_extra", "quantization_error")


def experimento_ejemplo():
    console.rule("[bold]1. Cuantizar, con numeros que puedes seguir[/bold]")
    w = torch.tensor([[0.12, -0.45, 0.03, 0.28]])
    q, escala = quantize_int8_symmetric(w)
    rec = dequantize_int8(q, escala)

    tabla = Table(header_style="bold")
    tabla.add_column("original", justify="right")
    tabla.add_column("int8", justify="right")
    tabla.add_column("recuperado", justify="right")
    tabla.add_column("error", justify="right")
    for i in range(4):
        tabla.add_row(
            f"{float(w[0, i]):+.4f}",
            f"{int(q[0, i]):>4}",
            f"{float(rec[0, i]):+.4f}",
            f"{abs(float(w[0, i]) - float(rec[0, i])):.4f}",
        )
    console.print(tabla)
    console.print(
        f"  escala = max(|W|) / 127 = {float(w.abs().max()):.2f} / 127 = "
        f"[bold]{float(escala):.6f}[/bold]\n\n"
        "[dim]El -0.45 se recupera exacto porque es el maximo y se mapea justo a -127. Los\n"
        "demas pierden hasta media unidad de escala. Y el cero, si lo hubiera, se\n"
        "representaria exactamente: por eso se usa 127 y no 128, para que el rango quede\n"
        "simetrico.[/dim]"
    )


def experimento_modelo():
    console.rule("[bold]2. Sobre el modelo de verdad[/bold]")
    set_seed(0)
    modelo = GPT(ModelConfig())

    tabla = Table(header_style="bold")
    tabla.add_column("matriz")
    tabla.add_column("forma", justify="right")
    tabla.add_column("error por canal", justify="right")
    tabla.add_column("error por tensor", justify="right")

    ejemplos = [
        ("token_embedding", modelo.token_embedding.weight),
        ("q_proj (capa 0)", modelo.blocks[0].attn.q_proj.weight),
        ("gate_proj (capa 0)", modelo.blocks[0].ffn.gate_proj.weight),
        ("down_proj (capa 5)", modelo.blocks[5].ffn.down_proj.weight),
    ]
    for nombre, w in ejemplos:
        e_c = quantization_error(w.data, per_channel=True)["error_relativo"]
        e_t = quantization_error(w.data, per_channel=False)["error_relativo"]
        tabla.add_row(nombre, str(tuple(w.shape)), f"{e_c * 100:.3f}%", f"{e_t * 100:.3f}%")
    console.print(tabla)

    total_fp32 = sum(p.numel() * 4 for p in modelo.parameters())
    matrices = [p for p in modelo.parameters() if p.dim() >= 2]
    total_int8 = sum(
        quantization_error(p.data)["bytes_cuantizado"] for p in matrices
    ) + sum(p.numel() * 4 for p in modelo.parameters() if p.dim() < 2)

    console.print(
        f"\n  modelo en fp32 : [bold]{total_fp32 / 1e6:.1f} MB[/bold]\n"
        f"  modelo en int8 : [bold]{total_int8 / 1e6:.1f} MB[/bold]  "
        f"([green]{total_fp32 / total_int8:.1f}x mas pequenyo[/green])\n\n"
        "[dim]Por canal siempre es mejor que por tensor: una sola fila con valores grandes\n"
        "no arrastra a las demas. Cuesta un vector de escalas mas, que es despreciable.\n\n"
        "Que un error del 0,7% en los pesos apenas afecte a la calidad del modelo es un\n"
        "HECHO EMPIRICO, no un teorema. Nadie predijo que las redes fueran tan robustas a\n"
        "la cuantizacion: se descubrio probando.\n\n"
        "Y un matiz que se suele omitir: cuantizar los pesos NO acelera nada por si solo si\n"
        "despues conviertes a float para multiplicar. La aceleracion de verdad requiere\n"
        "kernels que operen en int8 nativamente.[/dim]"
    )
    return ejemplos


def experimento_cierre():
    console.rule("[bold]3. Que te separa de un modelo frontier[/bold]")

    tabla = Table(header_style="bold")
    tabla.add_column("")
    tabla.add_column("el tuyo", justify="right")
    tabla.add_column("GPT-4 (estimado)", justify="right")
    tabla.add_column("factor", justify="right")

    filas = [
        ("parametros", 8.93e6, 1.8e12, None),
        ("tokens de entrenamiento", 5e8, 1.3e13, None),
        ("FLOPs de entrenamiento", 2.3e16, 2e25, None),
        ("coste aproximado", 0.5, 1e8, None),
    ]
    unidades = ["", "", " FLOPs", " EUR"]
    for (nombre, mio, suyo, _), unidad in zip(filas, unidades):
        tabla.add_row(nombre, f"{mio:.3g}{unidad}", f"{suyo:.3g}{unidad}", f"{suyo/mio:.0e}x")
    console.print(tabla)

    console.print(
        "\n[bold]Pero el tamanyo es solo una de cinco cosas:[/bold]\n\n"
        "  1. [bold]DATOS[/bold]: 15 billones de tokens filtrados con clasificadores,\n"
        "     deduplicados, y con mucho codigo y matematicas porque mejoran el razonamiento\n"
        "     en tareas que NO son ni codigo ni matematicas. La receta exacta no la publica\n"
        "     nadie.\n\n"
        "  2. [bold]COMPUTO[/bold]: nueve ordenes de magnitud, y no solo en GPUs: el centro\n"
        "     de datos, la red, y los ingenieros que mantienen eso meses sin que se caiga.\n\n"
        "  3. [bold]ARQUITECTURA[/bold]: Mixture of Experts, que activa solo una fraccion de\n"
        "     los parametros por token. Un billon de parametros con el coste de cien mil\n"
        "     millones.\n\n"
        "  4. [bold]POST-ENTRENAMIENTO[/bold]: RLHF/DPO despues del SFT, mas meses de\n"
        "     red-teaming y ajuste. Es la fase que convierte un predictor de texto en algo\n"
        "     que quieras usar, y emplea a mas gente que el pretraining.\n\n"
        "  5. [bold]INFRAESTRUCTURA[/bold]: paralelismo en varias dimensiones, tolerancia a\n"
        "     fallos (con miles de GPUs alguna falla cada pocas horas), y poder reanudar sin\n"
        "     perder dias.\n"
    )

    console.print(
        Panel(
            "[bold]Y esto es lo que si has conseguido:[/bold]\n\n"
            "Has escrito TODAS las piezas: atencion, RoPE, SwiGLU, AdamW, KV cache, el\n"
            "tokenizador. Todas validadas numericamente contra PyTorch. Un modelo frontier\n"
            "usa exactamente estas piezas, mas grandes y con mas ingenieria alrededor.\n\n"
            "Sabes leer un paper de arquitectura. Cuando salga el siguiente modelo y digan\n"
            "que usa grouped-query attention o RMSNorm, sabes que son y por que.\n\n"
            "Sabes depurar un entrenamiento: la perdida del paso 0 frente a ln(V), el\n"
            "overfit a un batch, la norma del gradiente, la MFU. Eso es lo que distingue a\n"
            "quien sabe entrenar modelos de quien copia scripts.\n\n"
            "Y sabes QUE NO SE SABE: que SwiGLU funciona sin explicacion, que Adam domina\n"
            "sin que nadie sepa bien por que, que las leyes de escala tienen intervalos mas\n"
            "amplios de lo que se reporta, y que los benchmarks estan contaminados. Esa\n"
            "parte no suele aparecer en los tutoriales, y es la que mas sirve para leer con\n"
            "criterio.",
            title="[bold green]Fin del curso[/bold green]",
            border_style="green",
        )
    )


def main() -> None:
    get_device()
    experimento_ejemplo()
    ejemplos = experimento_modelo()
    experimento_cierre()

    fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))

    nombres = [n.split(" ")[0] for n, _ in ejemplos]
    e_canal = [quantization_error(w.data, True)["error_relativo"] * 100 for _, w in ejemplos]
    e_tensor = [quantization_error(w.data, False)["error_relativo"] * 100 for _, w in ejemplos]
    x = range(len(nombres))
    izq.bar([i - 0.2 for i in x], e_canal, 0.4, label="por canal")
    izq.bar([i + 0.2 for i in x], e_tensor, 0.4, label="por tensor")
    izq.set_xticks(list(x))
    izq.set_xticklabels(nombres, fontsize=7, rotation=15)
    izq.set_ylabel("error relativo (%)")
    izq.set_title("El danyo de int8, por matriz")
    izq.grid(alpha=0.3, axis="y")
    izq.legend(fontsize=8)

    escalas = ["el tuyo\n8.9M", "GPT-2\n1.5B", "Llama-3\n70B", "GPT-4\n~1.8T"]
    params = [8.93e6, 1.5e9, 7e10, 1.8e12]
    der.bar(escalas, params, color=["tab:green", "tab:blue", "tab:orange", "tab:red"])
    der.set_yscale("log")
    der.set_ylabel("parametros (escala log)")
    der.set_title("Donde esta tu modelo")
    der.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    destino = figures_dir() / "17_extra.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")


if __name__ == "__main__":
    main()
