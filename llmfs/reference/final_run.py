"""Referencia del modulo 13: la tirada real."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch


def overfit_single_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    steps: int = 200,
    lr: float = 1e-3,
    optimizer_factory: Callable[..., Any] | None = None,
) -> list[float]:
    """Entrena UN SOLO batch hasta memorizarlo. El test que caza casi cualquier bug.

    LA IDEA. Un modelo con millones de parametros tiene capacidad de sobra para memorizar
    cuatro secuencias. Si le das el mismo batch una y otra vez, la perdida TIENE que bajar
    practicamente a cero.

    Si no baja, hay un bug, y lo sabes en 30 segundos en vez de en cuatro horas.

    QUE CAZA
        - gradientes que no llegan a alguna parte del modelo (un `detach()` de mas)
        - el `zero_grad()` olvidado
        - un learning rate absurdo
        - la mascara mal puesta (aunque este caso baja DEMASIADO deprisa, ojo)
        - una capa que no esta conectada al grafo
        - el optimizador construido sobre los parametros equivocados

    QUE NO CAZA
        Nada relacionado con generalizacion: los datos, el tamanyo del modelo, el
        regularizador. Un modelo que memoriza un batch puede seguir siendo inutil.

    ES LA PRIMERA COMPROBACION QUE HAY QUE HACER, siempre, antes de lanzar cualquier
    entrenamiento largo. Karpathy lo lleva anyos repitiendo y sigue siendo el consejo con
    mejor relacion entre coste y beneficio de todo el deep learning.

    Returns:
        El historial de perdidas, una por paso.
    """
    factory = optimizer_factory or (lambda params: torch.optim.AdamW(params, lr=lr))
    opt = factory(model.parameters())

    model.train()
    historial: list[float] = []
    for _ in range(steps):
        _, perdida = model(x, y)
        opt.zero_grad(set_to_none=True)
        perdida.backward()
        opt.step()
        historial.append(float(perdida.detach()))

    return historial


def format_eta(seconds: float) -> str:
    """Formatea una duracion en algo legible de un vistazo.

        45      -> "45s"
        125     -> "2m 5s"
        3725    -> "1h 2m"
        90000   -> "1d 1h"

    Parece cosmetico y no lo es: vas a mirar este numero muchas veces durante una tirada de
    horas, y "1h 2m" se lee al instante mientras que "3725 s" hay que dividirlo mentalmente.

    Los valores negativos o no finitos devuelven "?", que es lo honesto cuando todavia no
    hay datos suficientes para estimar.
    """
    if not math.isfinite(seconds) or seconds < 0:
        return "?"

    segundos = int(seconds)
    if segundos < 60:
        return f"{segundos}s"
    if segundos < 3600:
        return f"{segundos // 60}m {segundos % 60}s"
    if segundos < 86400:
        return f"{segundos // 3600}h {(segundos % 3600) // 60}m"
    return f"{segundos // 86400}d {(segundos % 86400) // 3600}h"


def estimate_remaining(
    step: int, max_steps: int, elapsed_seconds: float, warmup_steps: int = 10
) -> float:
    """Segundos que faltan, a partir del ritmo medido hasta ahora.

    Los primeros pasos son mas lentos (compilacion de kernels, caches frias), asi que
    incluirlos en la media da estimaciones pesimistas. Con `warmup_steps` se ignoran.

    Devuelve `inf` mientras no haya datos suficientes, y `format_eta` lo convierte en "?".
    Es mas honesto que dar un numero inventado.
    """
    if step <= warmup_steps or elapsed_seconds <= 0:
        return float("inf")

    por_paso = elapsed_seconds / step
    return por_paso * max(0, max_steps - step)
