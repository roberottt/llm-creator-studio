"""Referencia del modulo 15: evaluacion."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import torch

#: Los prompts de la bateria cualitativa del paper TinyStories. Estan elegidos para probar
#: cosas distintas: continuacion narrativa, coherencia causal, uso de un objeto mencionado
#: antes, y cierre de historia.
PROMPTS_TINYSTORIES: tuple[tuple[str, str], ...] = (
    ("Once upon a time, there was a little girl named Lily. She", "continuacion basica"),
    ("Tom and Jane went to the park. They saw a big dog. The dog", "coherencia causal"),
    ("Sara had a red ball. She threw the ball and it", "seguimiento de objeto"),
    ("The cat was very hungry. It looked for food and finally", "resolucion"),
    ("One day, a boy found a shiny key. He used the key to", "uso de un objeto"),
    ("The sun was going down. The children said goodbye and", "cierre de historia"),
)


def perplexity_from_loss(loss: float) -> float:
    """Perplejidad a partir de la perdida media en nats: `exp(loss)`.

    Interpretacion: entre cuantas opciones equiprobables esta dudando el modelo,
    efectivamente. Perplejidad 10 significa que en promedio esta tan indeciso como si
    eligiera al azar entre 10 palabras.

    Con vocabulario de 4096 y un modelo sin entrenar, la perplejidad es 4096.
    """
    if not math.isfinite(loss):
        return float("inf")
    return math.exp(loss)


def bits_per_byte(total_loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    """Bits por byte: la metrica que SI se puede comparar entre tokenizadores distintos.

    EL PROBLEMA CON LA PERPLEJIDAD. Depende del tokenizador. Si tu vocabulario parte las
    palabras en trozos mas pequenyos, cada token individual es mas facil de predecir y tu
    perplejidad sale mejor sin que el modelo sea mejor. Comparar perplejidades entre
    modelos con tokenizadores distintos no significa absolutamente nada, y se hace
    constantemente.

    LA SOLUCION. Normalizar por BYTES de texto original en vez de por tokens. Los bytes no
    dependen de como trocees.

        bits_por_byte = (perdida_total_en_nats / ln(2)) / n_bytes

    El `/ ln(2)` convierte nats a bits.

    INTERPRETACION. Es literalmente cuantos bits necesitarias para transmitir el texto
    usando el modelo como compresor. Un modelo que da 1,0 bits/byte comprime el texto a la
    octava parte. Los mejores modelos de lenguaje rondan 0,6-0,8 bits/byte sobre texto en
    ingles; gzip anda por 2,5.

    Args:
        total_loss_nats: la SUMA de las perdidas, no la media.
        n_tokens: cuantos tokens se han evaluado (no se usa en el calculo, pero se pide
            para dejar claro que la perdida es total y no media).
        n_bytes: cuantos bytes tenia el texto original.
    """
    if n_bytes <= 0:
        raise ValueError("n_bytes tiene que ser positivo")
    return total_loss_nats / math.log(2) / n_bytes


@torch.no_grad()
def evaluate_perplexity(
    model: Any,
    data: Any,
    batch_size: int,
    context_length: int,
    iters: int = 100,
    device: Any = None,
    get_batch: Callable[..., Any] | None = None,
) -> dict[str, float]:
    """Perdida media y perplejidad sobre un conjunto de datos.

    No es un ejercicio; lo usa el informe del modulo 15.
    """
    from llmfs.bridge import resolve

    get_batch = get_batch or resolve("04_datos", "get_batch")
    model.eval()

    total, n = 0.0, 0
    for _ in range(iters):
        x, y = get_batch(data, batch_size, context_length, device=device)
        _, perdida = model(x, y)
        total += float(perdida.detach())
        n += 1

    media = total / max(1, n)
    return {"loss": media, "perplexity": perplexity_from_loss(media), "iters": n}


def run_prompt_battery(
    generate_fn: Callable[[str], str],
    prompts: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Genera una completion para cada prompt de la bateria.

    `generate_fn(prompt) -> texto` encapsula el modelo y el tokenizador, para que esta
    funcion no sepa nada de ninguno de los dos.

    Returns:
        Una lista de dicts con `prompt`, `que_prueba` y `completion`.
    """
    prompts = prompts or PROMPTS_TINYSTORIES
    return [
        {"prompt": prompt, "que_prueba": etiqueta, "completion": generate_fn(prompt)}
        for prompt, etiqueta in prompts
    ]


def write_eval_report(
    path: Any,
    metricas: dict[str, Any],
    bateria: list[dict[str, str]],
    config_resumen: str = "",
) -> Any:
    """Escribe el informe de evaluacion en markdown.

    No es un ejercicio: es el andamio para que la parte interesante (leer las completions)
    sea comoda.
    """
    from datetime import datetime
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lineas = [
        "# Informe de evaluación",
        "",
        f"Generado el {datetime.now():%Y-%m-%d %H:%M}.",
        "",
    ]
    if config_resumen:
        lineas += ["```", config_resumen, "```", ""]

    lineas += ["## Métricas", "", "| métrica | valor |", "|---|---|"]
    for clave, valor in metricas.items():
        formateado = f"{valor:.4f}" if isinstance(valor, float) else str(valor)
        lineas.append(f"| {clave} | {formateado} |")

    lineas += [
        "",
        "## Batería cualitativa",
        "",
        "Los prompts vienen del paper de TinyStories. La evaluación es tuya: léelos y",
        "juzga gramática, coherencia y si la historia tiene sentido.",
        "",
    ]
    for i, caso in enumerate(bateria, start=1):
        lineas += [
            f"### {i}. {caso['que_prueba']}",
            "",
            f"**Prompt:** `{caso['prompt']}`",
            "",
            "```",
            caso["completion"],
            "```",
            "",
        ]

    path.write_text("\n".join(lineas), encoding="utf-8")
    return path
