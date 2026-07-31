"""Demo del modulo 16: SFT de verdad sobre el modelo del modulo 13.

    llmfs demo 16

Entrena el modelo con un pequenyo conjunto de instruccion-respuesta y compara el
comportamiento antes y despues. Tarda unos 30 segundos.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.config import RunConfig
from llmfs.data import preparar
from llmfs.device import get_device, set_seed
from llmfs.paths import configs_dir, figures_dir
from llmfs.reference import GPT, apply_lora_to_model, count_trainable

console = Console()

build_chat_template = resolve("16_finetuning", "build_chat_template")
mask_prompt_tokens = resolve("16_finetuning", "mask_prompt_tokens")
LoRALinear = resolve("16_finetuning", "LoRALinear")
merge_lora_weights = resolve("16_finetuning", "merge_lora_weights")
generate_with_cache = resolve("14_inferencia", "generate_with_cache")

# Un conjunto de SFT diminuto, en el estilo del corpus (Shakespeare) para que el modelo
# tenga alguna posibilidad. Con 9M de parametros y un dataset asi, lo que se aprende es el
# FORMATO, no el contenido.
EJEMPLOS = [
    ("Who is the king?", "The king is Richard."),
    ("What did the lady say?", "She said farewell."),
    ("Where is the lord?", "The lord is in the castle."),
    ("Who loves Juliet?", "Romeo loves Juliet."),
    ("What is the news?", "The news is grave."),
    ("Who speaks now?", "The duke speaks now."),
    ("Where is the queen?", "The queen is in her chamber."),
    ("What say you?", "I say we must go."),
] * 12


def experimento_formato():
    console.rule("[bold]1. El chat template[/bold]")
    msgs = [
        {"role": "user", "content": "Who is the king?"},
        {"role": "assistant", "content": "The king is Richard."},
    ]
    console.print(f"  entrenamiento : [cyan]{build_chat_template(msgs)}[/cyan]")
    console.print(
        f"  inferencia    : [cyan]{build_chat_template(msgs[:1], add_generation_prompt=True)}"
        f"[/cyan]  [dim]<- abierta, el modelo continua aqui[/dim]\n"
    )
    console.print(
        "[dim]Los marcadores no son magicos: son texto que el modelo aprende a reconocer\n"
        "durante el SFT. El `<|end|>` es lo que le enseña CUANDO PARAR; sin el, generaria\n"
        "indefinidamente.[/dim]\n"
    )

    console.print("[bold]Y el enmascarado del prompt:[/bold]\n")
    ids = [10, 11, 12, 20, 21, 22]
    targets = mask_prompt_tokens(ids, 3)
    tabla = Table(header_style="bold")
    tabla.add_column("posicion", justify="right")
    tabla.add_column("input", justify="right")
    tabla.add_column("target", justify="right")
    tabla.add_column("")
    for i, (a, b) in enumerate(zip(ids, targets)):
        nota = ""
        if i < 2:
            nota = "[dim]prompt: ignorado[/dim]"
        elif i == 2:
            nota = "[bold green]la transicion: SI aprende[/bold green]"
        elif i == len(ids) - 1:
            nota = "[dim]no hay siguiente token[/dim]"
        else:
            nota = "respuesta"
        tabla.add_row(str(i), str(a), str(b) if b != -100 else "[dim]-100[/dim]", nota)
    console.print(tabla)
    console.print(
        "\n[dim]Fijate en la posicion 2: es el ULTIMO token del prompt, pero su objetivo ya\n"
        "es el primero de la respuesta. Esa transicion 'se acabo la pregunta, me toca' es lo\n"
        "mas importante que tiene que aprender, asi que NO se enmascara. Por eso hay dos\n"
        "posiciones ignoradas y no tres: es el off-by-one del ejercicio.[/dim]"
    )


def experimento_lora():
    console.rule("[bold]2. LoRA: entrenar el 1% del modelo[/bold]")

    cfg = RunConfig.from_yaml(configs_dir() / "tinystories_9m.yaml")
    tabla = Table(header_style="bold")
    tabla.add_column("rango r", justify="right")
    tabla.add_column("entrenables", justify="right")
    tabla.add_column("% del modelo", justify="right")
    tabla.add_column("memoria de AdamW", justify="right")

    rangos, porcentajes = [], []
    completo = 8_933_440
    tabla.add_row(
        "[dim]sin LoRA[/dim]", f"[dim]{completo:,}[/dim]", "[dim]100%[/dim]",
        f"[dim]{completo * 8 / 1e6:.0f} MB[/dim]"
    )
    for r in (4, 8, 16, 32):
        set_seed(0)
        modelo = apply_lora_to_model(GPT(cfg.model), r=r)
        c = count_trainable(modelo)
        rangos.append(r)
        porcentajes.append(c["porcentaje"])
        tabla.add_row(
            str(r), f"{c['entrenables']:,}", f"{c['porcentaje']:.2f}%",
            f"{c['entrenables'] * 8 / 1e6:.2f} MB"
        )
    console.print(tabla)
    console.print(
        "\n[dim]La memoria de AdamW son 8 bytes por parametro entrenable (dos momentos en\n"
        "fp32). Con LoRA r=8 baja de 71 MB a 0,5 MB. En un modelo de 70B, eso es la\n"
        "diferencia entre necesitar ocho GPUs o una.[/dim]\n"
    )

    console.print("[bold]Y la fusion de pesos:[/bold]")
    set_seed(0)
    base = torch.nn.Linear(320, 320)
    lora = LoRALinear(base, r=8)
    x = torch.randn(4, 320)
    console.print(
        f"  al inicializar, salida identica a la base: "
        f"[green]{torch.allclose(lora(x), base(x), atol=1e-6)}[/green]  "
        "[dim](porque lora_B empieza en ceros)[/dim]"
    )
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)
    fundida = merge_lora_weights(lora)
    err = float((lora(x) - fundida(x)).abs().max())
    console.print(
        f"  tras adaptar, la capa fundida da lo mismo: error {err:.2e}  "
        + ("[green]OK[/green]" if err < 1e-5 else "[red]MAL[/red]")
    )
    console.print(
        "\n[dim]Eso es lo que hace util a LoRA frente a otros metodos de fine-tuning\n"
        "eficiente: la adaptacion es EXACTAMENTE una suma de matrices, asi que se absorbe\n"
        "sin aproximar nada. El modelo resultante es indistinguible de uno normal.[/dim]"
    )
    return rangos, porcentajes


def experimento_sft(cfg_dev):
    console.rule("[bold]3. SFT de verdad[/bold]")

    cfg = RunConfig.from_yaml(configs_dir() / "tiny_char.yaml")
    dataset = preparar(cfg, quiet=True)
    cfg.model.vocab_size = dataset.vocab_size

    from llmfs.train import cargar_checkpoint

    modelo = GPT(cfg.model)
    ruta = cfg.run_dir / "best.pt"
    if not ruta.exists():
        console.print(
            "[yellow]No hay modelo entrenado. Entrena uno con "
            "`llmfs train --config tiny_char` y vuelve.[/yellow]"
        )
        return None
    cargar_checkpoint(ruta, modelo, map_location="cpu")
    modelo = modelo.to(cfg_dev.device)

    # El chat template usa caracteres que el tokenizador de caracteres no conoce, asi que
    # se usan marcadores simples con caracteres del corpus.
    def formatear(pregunta: str, respuesta: str) -> tuple[str, str]:
        return f"Q: {pregunta}\nA: ", f"{respuesta}\n"

    ejemplos_ids = []
    for pregunta, respuesta in EJEMPLOS:
        p, r = formatear(pregunta, respuesta)
        ids_p, ids_r = dataset.encode(p), dataset.encode(r)
        completo = ids_p + ids_r
        if len(completo) <= cfg.model.context_length:
            ejemplos_ids.append((completo, len(ids_p)))

    console.print(
        f"{len(ejemplos_ids)} ejemplos de instruccion-respuesta, formato `Q: ...\\nA: ...`\n"
        "[dim](se usa este formato en vez de los marcadores <|user|> porque el tokenizador\n"
        "de caracteres solo conoce los simbolos que aparecen en Shakespeare)[/dim]\n"
    )

    prompt_prueba = "Q: Who is the king?\nA:"

    def generar() -> str:
        ids = torch.tensor([dataset.encode(prompt_prueba)], device=cfg_dev.device)
        set_seed(7)
        salida = generate_with_cache(modelo, ids, 60, temperature=0.7, top_k=20)
        return dataset.decode(salida[0].tolist())

    modelo.eval()
    antes = generar()

    # SFT
    modelo.train()
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-4)
    historial = []
    set_seed(0)
    for paso in range(150):
        idx = torch.randint(0, len(ejemplos_ids), (8,))
        max_len = max(len(ejemplos_ids[i][0]) for i in idx)
        x = torch.full((8, max_len), 0, dtype=torch.long)
        y = torch.full((8, max_len), -100, dtype=torch.long)
        for fila, i in enumerate(idx):
            ids, plen = ejemplos_ids[i]
            x[fila, : len(ids)] = torch.tensor(ids)
            y[fila, : len(ids)] = torch.tensor(mask_prompt_tokens(ids, plen))
        x, y = x.to(cfg_dev.device), y.to(cfg_dev.device)

        _, perdida = modelo(x, y)
        opt.zero_grad(set_to_none=True)
        perdida.backward()
        opt.step()
        historial.append(float(perdida.detach()))

    modelo.eval()
    despues = generar()

    console.print(f"  perdida del SFT: {historial[0]:.4f} -> {historial[-1]:.4f}\n")
    console.print(Panel(antes.replace("\n", " ⏎ "), title="ANTES del SFT", border_style="red"))
    console.print(Panel(despues.replace("\n", " ⏎ "), title="DESPUES del SFT", border_style="green"))

    console.print(
        "\n[bold]Lo que hay que mirar no es si la respuesta es correcta.[/bold] Con 0,8M de\n"
        "parametros y 96 ejemplos, no lo va a ser.\n\n"
        "Lo que hay que mirar es el FORMATO: si tras el SFT responde algo corto y para, en\n"
        "vez de seguir escribiendo Shakespeare indefinidamente, el post-entrenamiento ha\n"
        "hecho su trabajo.\n\n"
        "[dim]Y esa es justamente la leccion del modulo: el post-entrenamiento NO anyade\n"
        "conocimiento. Saca a la superficie un comportamiento que ya estaba latente. Un\n"
        "modelo que no sabe algo tras el pretraining no lo aprende con mil ejemplos de\n"
        "conversacion.[/dim]"
    )
    return historial


def main() -> None:
    cfg_dev = get_device()
    experimento_formato()
    rangos, porcentajes = experimento_lora()
    historial = experimento_sft(cfg_dev)

    fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))

    izq.plot(rangos, porcentajes, marker="o", color="tab:green")
    izq.axhline(100, color="red", ls="--", lw=1, label="fine-tuning completo")
    izq.set_xlabel("rango r de LoRA")
    izq.set_ylabel("% de parametros entrenables")
    izq.set_yscale("log")
    izq.set_title("LoRA entrena una fraccion diminuta")
    izq.grid(alpha=0.3)
    izq.legend(fontsize=8)

    if historial:
        der.plot(historial, color="tab:blue")
        der.set_xlabel("paso de SFT")
        der.set_ylabel("perdida (solo sobre la respuesta)")
        der.set_title("El SFT aprende el formato")
        der.grid(alpha=0.3)

    fig.tight_layout()
    destino = figures_dir() / "16_finetuning.png"
    fig.savefig(destino, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figura guardada en {destino}[/green]")


if __name__ == "__main__":
    main()
