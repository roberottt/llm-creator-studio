"""Modulo 13 - La tirada real.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa los dos ejercicios -> `llmfs check 13` -> y despues ENTRENA:

    llmfs train --config tiny_char

QUÉ VAS A CONSTRUIR
===================

    overfit_single_batch  (ej. 1)  la comprobacion de 30 segundos que caza casi todo
    format_eta            (ej. 2)  cuanto falta, en algo legible

Y con eso lanzas el entrenamiento de verdad.

EL EJERCICIO 1 ES EL QUE IMPORTA
================================

Un modelo con millones de parametros memoriza cuatro secuencias sin despeinarse. Si le das
el MISMO batch una y otra vez y la perdida no baja casi a cero, hay un bug.

Y lo sabes en 30 segundos en vez de en cuatro horas. Es el consejo con mejor relacion
coste/beneficio de todo el deep learning, y casi nadie lo aplica.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **overfit**: que el modelo memorice en vez de generalizar. Normalmente es malo; aqui se
  busca a proposito, como prueba de sanidad.
- **checkpoint**: una foto del entrenamiento (pesos, estado del optimizador, numero de
  paso) para poder reanudar.
- **ETA**: cuanto falta para terminar, estimado a partir del ritmo medido.
- **paso** (step): una actualizacion de los pesos. No confundir con epoca, que es una
  pasada completa por los datos.

    llmfs demo 13     hace el overfit y entrena un modelo completo
"""

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

    LA IDEA
        Un modelo con millones de parametros tiene capacidad de sobra para memorizar cuatro
        secuencias. Si le das el MISMO batch una y otra vez, la perdida TIENE que bajar
        practicamente a cero.

        Si no baja, hay un bug. Y lo sabes en 30 segundos en vez de en cuatro horas.

    QUE CAZA
        - gradientes que no llegan a alguna parte del modelo (un `detach()` de mas)
        - el `zero_grad()` olvidado
        - un learning rate absurdo
        - una capa desconectada del grafo
        - el optimizador construido sobre los parametros equivocados

    QUE NO CAZA
        Nada relacionado con generalizacion. Un modelo que memoriza un batch puede seguir
        siendo completamente inutil.

    UN AVISO
        Si la perdida baja a cero DEMASIADO deprisa (en 5 pasos, digamos), sospecha de una
        fuga de informacion: revisa que los targets vayan desplazados un token respecto a
        la entrada.

    EL BUCLE
        opt = optimizer_factory(model.parameters())   o AdamW si es None
        model.train()
        repetir `steps` veces:
            _, perdida = model(x, y)
            opt.zero_grad(set_to_none=True)
            perdida.backward()
            opt.step()
            historial.append(float(perdida.detach()))

        Es el bucle mas simple posible: sin scheduler, sin acumulacion, sin AMP. A
        proposito: cuantas menos piezas, menos sitios donde esconderse un bug.

    ESTA ES LA PRIMERA COMPROBACION QUE HAY QUE HACER, siempre, antes de lanzar cualquier
    entrenamiento largo. Es el consejo con mejor relacion coste/beneficio de todo el deep
    learning.

    Args:
        model: el modelo, con forward `(idx, targets) -> (logits, loss)`.
        x, y: el batch a memorizar.
        steps: cuantos pasos.
        lr: learning rate (solo si no pasas `optimizer_factory`).
        optimizer_factory: `fn(params) -> optimizador`, para poder usar el tuyo.

    Returns:
        El historial de perdidas, una por paso.
    """
    raise NotImplementedError("TODO: modulo 13, ejercicio 1 - overfit_single_batch")


def format_eta(seconds: float) -> str:
    """Formatea una duracion en algo legible de un vistazo.

        45      -> "45s"
        125     -> "2m 5s"
        3725    -> "1h 2m"
        90000   -> "1d 1h"

    Parece cosmetico y no lo es: vas a mirar este numero muchas veces durante una tirada de
    horas, y "1h 2m" se lee al instante mientras que "3725 s" hay que dividirlo
    mentalmente.

    LOS TRAMOS
        < 60      -> "{s}s"
        < 3600    -> "{m}m {s}s"          minutos y segundos
        < 86400   -> "{h}h {m}m"          horas y minutos, SIN segundos
        resto     -> "{d}d {h}h"

        Fijate en que a partir de una hora se dejan de mostrar los segundos: cuando faltan
        dos horas, los segundos son ruido.

    LOS CASOS RAROS
        Negativos o no finitos (`inf`, `nan`) devuelven "?". Es lo honesto cuando todavia
        no hay datos suficientes para estimar, y evita imprimir cosas como "-1s" o
        "infd 0h".

        Usa `math.isfinite(seconds)` para detectarlo.

    Args:
        seconds: la duracion.

    Returns:
        La cadena formateada.
    """
    raise NotImplementedError("TODO: modulo 13, ejercicio 2 - format_eta")
