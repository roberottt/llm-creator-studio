"""Modulo 13 - La tirada real.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa los dos ejercicios -> `llmfs check 13` -> y despues ENTRENA:

    llmfs train --config tiny_char

QUÉ VAS A CONSTRUIR
===================

    overfit_single_batch  (ej. 1)  la comprobacion de 30 segundos que caza casi todo
    format_eta            (ej. 2)  cuanto falta, en algo legible

Y con eso lanzas el entrenamiento de verdad. Ojo: el entrenamiento en si YA ESTA ESCRITO,
montado sobre las piezas de los modulos 04 a 12. Tu trabajo aqui es lanzarlo, saber leerlo y
decidir si va bien.

`TEORIA.md` sigue este mismo orden y cada docstring de aqui te dice que seccion le toca. Y hay
una seccion que no corresponde a ningun ejercicio y es la habilidad que de verdad te llevas:
"Como se lee la linea del log", que desmonta campo a campo la linea que vas a estar mirando
durante horas. Cada uno de sus seis campos viene de un modulo distinto del curso.

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

    Contexto en `TEORIA.md`: seccion "Ejercicio 1: la comprobacion de los 30 segundos", con la
    tabla medida de como baja la perdida (de 4,29 a 0,06 en 300 pasos y dos segundos de reloj) y
    la lista de que caza y que no.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    El bucle de entrenamiento mas simple que existe. Cuatro pasos.

        1. El optimizador, usando el que te pasen o AdamW si no:

               if optimizer_factory is None:
                   opt = torch.optim.AdamW(model.parameters(), lr=lr)
               else:
                   opt = optimizer_factory(model.parameters())

        2. Modo entrenamiento y la lista donde guardar el historial:

               model.train()
               historial = []

        3. El bucle, `steps` veces, siempre con EL MISMO batch:

               for _ in range(steps):
                   _, perdida = model(x, y)
                   opt.zero_grad(set_to_none=True)
                   perdida.backward()
                   opt.step()
                   historial.append(float(perdida.detach()))

        4. `return historial`

    Sin scheduler, sin acumulacion de gradientes, sin AMP. A proposito: cuantas menos piezas
    haya, menos sitios donde se pueda esconder un bug. Este bucle es el patron de referencia
    contra el que compararas el bucle de verdad cuando algo falle.

    QUÉ TIENE QUE SALIR
    -------------------
    La perdida arranca en `ln(vocab_size)` y baja hasta casi cero. Con el modelo juguete:

        paso   0:  4.17
        paso  50:  1.12
        paso 100:  0.21
        paso 200:  0.02

    Si `historial[-1]` no esta muy por debajo de `historial[0]`, PARA y busca el bug. No lances
    el entrenamiento largo.

    LA IDEA
    -------
    Un modelo con millones de parametros tiene capacidad de sobra para memorizar cuatro
    secuencias. Si le das el MISMO batch una y otra vez, la perdida TIENE que bajar
    practicamente a cero. No hay nada que generalizar: solo memorizar.

    Si no baja, hay un bug. Y lo sabes en 30 segundos en vez de en cuatro horas.

    QUÉ CAZA
    --------
        - gradientes que no llegan a alguna parte del modelo (un `detach()` de mas)
        - el `zero_grad()` olvidado (los gradientes se ACUMULAN por defecto en PyTorch)
        - un learning rate absurdo, por arriba o por abajo
        - una capa desconectada del grafo
        - el optimizador construido sobre los parametros equivocados

    QUÉ NO CAZA
    -----------
    Nada relacionado con la generalizacion. Un modelo que memoriza un batch perfectamente puede
    seguir siendo completamente inutil con datos nuevos. Esto comprueba que la MAQUINARIA
    funciona, no que el modelo sea bueno.

    SI BAJA DEMASIADO DEPRISA, TAMBIÉN SOSPECHA
    -------------------------------------------
    Si la perdida se planta en cero en 5 pasos, mira si hay una fuga de informacion: comprueba
    que `y` va desplazado UN token respecto a `x`. Si pasas `model(x, x)` el modelo solo tiene
    que copiar la entrada y la perdida se desploma. El sintoma es identico al de una mascara
    causal rota.

    EL `set_to_none=True`
    ---------------------
    `opt.zero_grad(set_to_none=True)` pone los `.grad` a `None` en vez de a ceros. Ahorra una
    pasada de escritura sobre todos los gradientes y algo de memoria. Es el defecto desde
    PyTorch 2.0, se pone explicito porque se lee mejor.

    ÉSTA ES LA PRIMERA COMPROBACIÓN QUE HAY QUE HACER, siempre, antes de lanzar cualquier
    entrenamiento largo. Es el consejo con mejor relacion coste/beneficio de todo el deep
    learning, y casi nadie lo aplica.

    Args:
        model: el modelo, con forward `(idx, targets) -> (logits, loss)`.
        x, y: el batch a memorizar. `y` desplazado un token respecto a `x`.
        steps: cuantos pasos dar.
        lr: learning rate (solo se usa si no pasas `optimizer_factory`).
        optimizer_factory: `fn(params) -> optimizador`, para poder probar el tuyo del modulo 11.

    Returns:
        El historial de perdidas, una por paso.
    """
    raise NotImplementedError("TODO: modulo 13, ejercicio 1 - overfit_single_batch")


def format_eta(seconds: float) -> str:
    """Formatea una duracion en algo legible de un vistazo.

    Contexto en `TEORIA.md`: seccion "Ejercicio 2: cuanto falta", con las dos razones por las que
    esto no es cosmetico. La segunda es una regla que sirve mucho mas alla de este ejercicio: la
    precision util de una estimacion siempre es proporcional a su magnitud.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Una cadena de `if`, de menor a mayor.

        1. Los casos raros, PRIMERO:

               if not math.isfinite(seconds) or seconds < 0:
                   return "?"

        2. A entero, que los segundos con decimales no aportan nada:

               seconds = int(seconds)

        3. Los cuatro tramos:

               if seconds < 60:
                   return f"{seconds}s"
               if seconds < 3600:
                   return f"{seconds // 60}m {seconds % 60}s"
               if seconds < 86400:
                   return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
               return f"{seconds // 86400}d {(seconds % 86400) // 3600}h"

    QUÉ TIENE QUE SALIR
    -------------------
            45   ->  "45s"
           125   ->  "2m 5s"
          3725   ->  "1h 2m"
         90000   ->  "1d 1h"
            -1   ->  "?"
        float("inf")  ->  "?"

    Comprueba el 3725 a mano: 3725 // 3600 = 1, y (3725 % 3600) // 60 = 125 // 60 = 2. Sale
    "1h 2m". Y fijate en que los 5 segundos sobrantes se pierden, que es justo lo que se quiere.

    POR QUÉ SE DEJAN DE MOSTRAR LOS SEGUNDOS A PARTIR DE UNA HORA
    -------------------------------------------------------------
    Cuando faltan dos horas, los segundos son ruido: cambian todo el rato, no aportan
    informacion y hacen que el numero baile en pantalla. La precision util de una estimacion
    siempre es proporcional a su magnitud.

    POR QUÉ "?" Y NO UN 0
    ---------------------
    Devolver "?" es lo honesto cuando todavia no hay datos suficientes para estimar (en los
    primeros pasos la velocidad media no significa nada). Y evita imprimir cosas como "-1s" o
    "infd 0h", que ademas de feas hacen dudar de si el entrenamiento va bien.

    `math.isfinite(x)` es False para `inf`, `-inf` y `nan`. Los tres salen de dividir por cero
    o de un cero entre cero al calcular el ritmo.

    ESTO PARECE COSMÉTICO Y NO LO ES
    --------------------------------
    Vas a mirar este numero muchas veces durante una tirada de horas. "1h 2m" se lee al
    instante; "3725s" hay que dividirlo mentalmente cada vez.

    Args:
        seconds: la duracion en segundos.

    Returns:
        La cadena formateada, o "?" si la entrada no es utilizable.
    """
    raise NotImplementedError("TODO: modulo 13, ejercicio 2 - format_eta")
