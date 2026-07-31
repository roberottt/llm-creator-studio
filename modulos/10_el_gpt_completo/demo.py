"""Demo del modulo 10: el modelo completo, montado y auditado.

    llmfs demo 10

Cinco comprobaciones sobre el modelo de 8.933.440 parametros:
  1. El desglose de parametros, componente a componente.
  2. La formula frente al conteo real.
  3. La perdida del paso 0 frente a ln(V): el detector de bugs.
  4. Que es causal de verdad, comprobado cambiando un token.
  5. Cuanta memoria ocupa y donde esta el grueso (spoiler: los logits).
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")  # sin ventana: esto tiene que correr por SSH y en CI

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.config import ModelConfig, RunConfig
from llmfs.device import get_device, set_seed
from llmfs.paths import configs_dir, figures_dir

console = Console()

expected_param_count = resolve("10_el_gpt_completo", "expected_param_count")
count_parameters = resolve("10_el_gpt_completo", "count_parameters")
GPT = resolve("10_el_gpt_completo", "GPT")


def main() -> None:
    cfg_dev = get_device()
    set_seed(1337)

    run = RunConfig.from_yaml(configs_dir() / "tinystories_9m.yaml")
    cfg = run.model

    console.rule("[bold]El modelo[/bold]")
    console.print(run.summary())

    # --------------------------------------------------------------- 1 y 2
    console.rule("[bold]1. Desglose de parametros[/bold]")
    modelo = GPT(cfg)
    conteo = count_parameters(modelo)
    total = conteo["total"]

    tabla = Table(header_style="bold")
    tabla.add_column("componente")
    tabla.add_column("parametros", justify="right")
    tabla.add_column("%", justify="right")
    tabla.add_column("de donde salen")

    detalles = {
        "embeddings": f"{cfg.vocab_size} x {cfg.d_model}",
        "attention": f"{cfg.n_layers} capas x 4 x {cfg.d_model}^2",
        "ffn": f"{cfg.n_layers} capas x 3 x {cfg.d_model} x {cfg.d_ff}",
        "norms": f"{2 * cfg.n_layers + 1} RMSNorm x {cfg.d_model}",
        "lm_head": "atada a los embeddings" if cfg.tie_embeddings else "sin atar",
        "other": "-",
    }
    for clave in ("embeddings", "attention", "ffn", "norms", "lm_head", "other"):
        if conteo[clave] == 0 and clave == "other":
            continue
        tabla.add_row(
            clave, f"{conteo[clave]:,}", f"{100 * conteo[clave] / total:.1f}%", detalles[clave]
        )
    tabla.add_section()
    tabla.add_row("[bold]TOTAL[/bold]", f"[bold]{total:,}[/bold]", "100%", "")
    tabla.add_row("no-embedding", f"{conteo['non_embedding']:,}", "", "lo que usa Chinchilla")
    console.print(tabla)

    formula = expected_param_count(cfg)
    coincide = formula == total == 8_933_440
    console.print(
        f"\n  formula  : {formula:,}\n"
        f"  conteo   : {total:,}\n"
        f"  objetivo : 8,933,440\n"
        + (
            "  [bold green]Los tres coinciden.[/bold green]"
            if coincide
            else "  [bold red]NO coinciden.[/bold red]"
        )
    )

    if cfg.tie_embeddings:
        console.print(
            f"\n[dim]Sin weight tying el modelo tendria "
            f"{expected_param_count(ModelConfig(**{**cfg.__dict__, 'tie_embeddings': False})):,} "
            f"parametros: un 15% mas, solo por no reutilizar una matriz que ya tienes.[/dim]"
        )

    # --------------------------------------------------------------- 3
    console.rule("[bold]2. La perdida del paso 0[/bold]")
    modelo = modelo.to(cfg_dev.device).eval()
    secuencia = torch.randint(0, cfg.vocab_size, (4, 65), device=cfg_dev.device)
    x, y = secuencia[:, :-1], secuencia[:, 1:]
    with torch.no_grad():
        _, perdida = modelo(x, y)

    suelo = math.log(cfg.vocab_size)
    desvio = float(perdida) - suelo
    console.print(
        f"  perdida del modelo sin entrenar : [bold]{float(perdida):.4f}[/bold]\n"
        f"  ln({cfg.vocab_size})                        : {suelo:.4f}\n"
        f"  desvio                          : {desvio:+.4f}\n"
    )
    if abs(desvio) < 0.1:
        console.print(
            "[green]Correcto.[/green] El modelo arranca sin opiniones, que es lo que debe.\n"
            "[dim]Este es el numero que tienes que ver en el paso 0 del modulo 11. Si sale\n"
            "mas alto, la init es demasiado agresiva. Si sale mas bajo, hay fuga de\n"
            "informacion y lo primero que hay que mirar es la mascara causal.[/dim]"
        )
    else:
        console.print("[red]Fuera de rango: revisa la inicializacion o la mascara.[/red]")

    console.print(
        "\n[dim]Nota de metodo: los targets van DESPLAZADOS un token (x = seq[:, :-1],\n"
        "y = seq[:, 1:]). Si pasaras `modelo(idx, idx)`, en la posicion t el modelo veria el\n"
        "token que tiene que predecir y la perdida saldria por debajo de ln(V). Parece un\n"
        "bug del modelo y es un bug del que monta el batch.[/dim]"
    )

    # --------------------------------------------------------------- 4
    console.rule("[bold]3. Es causal de verdad[/bold]")
    idx = torch.randint(0, cfg.vocab_size, (1, 12), device=cfg_dev.device)
    with torch.no_grad():
        original = modelo(idx)[0]
        modificado = idx.clone()
        modificado[0, 6] = (modificado[0, 6] + 1) % cfg.vocab_size
        alterado = modelo(modificado)[0]

    diffs = (alterado - original).abs().max(dim=-1).values[0]
    tabla2 = Table(header_style="bold")
    tabla2.add_column("posicion", justify="right")
    tabla2.add_column("cambio maximo en los logits", justify="right")
    tabla2.add_column("")
    for pos in range(12):
        marca = " <- token cambiado" if pos == 6 else ""
        color = "green" if pos < 6 else "yellow"
        tabla2.add_row(str(pos), f"[{color}]{float(diffs[pos]):.2e}[/{color}]", marca)
    console.print(tabla2)
    console.print(
        "[dim]Antes de la posicion 6 el cambio es cero exacto: esas predicciones no pueden\n"
        "ver el token 6. A partir de ahi si cambian. Eso es la mascara causal funcionando, y\n"
        "es la comprobacion mas directa que existe de que no hay fuga.[/dim]"
    )

    # --------------------------------------------------------------- 5
    console.rule("[bold]4. Memoria[/bold]")
    bytes_por_param = 4
    peso_modelo = total * bytes_por_param
    # AdamW guarda dos momentos por parametro, ambos en fp32
    estados_opt = total * 2 * bytes_por_param
    gradientes = total * bytes_por_param

    B, T, V = run.train.batch_size, cfg.context_length, cfg.vocab_size
    logits_fp16 = B * T * V * 2
    logits_fp32 = B * T * V * 4

    tabla3 = Table(header_style="bold")
    tabla3.add_column("que")
    tabla3.add_column("MB", justify="right")
    tabla3.add_column("nota")
    tabla3.add_row("pesos del modelo (fp32)", f"{peso_modelo / 1e6:.1f}", "")
    tabla3.add_row("gradientes (fp32)", f"{gradientes / 1e6:.1f}", "")
    tabla3.add_row("estados de AdamW", f"{estados_opt / 1e6:.1f}", "dos momentos por parametro")
    tabla3.add_section()
    tabla3.add_row(
        "logits en fp16", f"{logits_fp16 / 1e6:.1f}", f"batch {B} x ctx {T} x vocab {V}"
    )
    tabla3.add_row(
        "+ su version fp32", f"{logits_fp32 / 1e6:.1f}", "cross_entropy promociona"
    )
    tabla3.add_row(
        "+ su gradiente", f"{logits_fp32 / 1e6:.1f}", "", style="bold"
    )
    console.print(tabla3)
    console.print(
        f"[bold]Los logits solos ocupan mas que el modelo, los gradientes y el optimizador "
        f"juntos.[/bold]\n"
        f"({(logits_fp16 + 2 * logits_fp32) / 1e6:.0f} MB frente a "
        f"{(peso_modelo + gradientes + estados_opt) / 1e6:.0f} MB.)\n\n"
        "[dim]Cuando en el modulo 13 te quedes sin memoria en la 2060, este es el primer\n"
        "sitio donde mirar, no las activaciones del modelo. La solucion habitual es calcular\n"
        "la perdida por trozos en vez de materializar el tensor entero.[/dim]"
    )

    # --------------------------------------------------------------- grafica
    fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))

    componentes = ["embeddings", "attention", "ffn", "norms"]
    valores = [conteo[c] for c in componentes]
    izq.pie(
        valores,
        labels=[f"{c}\n{v:,}" for c, v in zip(componentes, valores)],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 8},
    )
    izq.set_title(f"Los {total:,} parametros")

    etiquetas = ["pesos", "gradientes", "AdamW", "logits\n(fp16+fp32+grad)"]
    memorias = [
        peso_modelo / 1e6,
        gradientes / 1e6,
        estados_opt / 1e6,
        (logits_fp16 + 2 * logits_fp32) / 1e6,
    ]
    colores = ["tab:blue"] * 3 + ["tab:red"]
    der.bar(etiquetas, memorias, color=colores)
    for i, v in enumerate(memorias):
        der.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    der.set_ylabel("MB")
    der.set_title(f"Memoria con batch {B} x ctx {T}")
    der.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    destino = figures_dir() / "10_el_gpt_completo.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")
    console.print(
        "\n[bold]Con esto termina la Parte II.[/bold] Tienes el modelo montado y auditado.\n"
        "En la Parte III se entrena."
    )


if __name__ == "__main__":
    main()
