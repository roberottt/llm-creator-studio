"""Modulo 11 - El bucle de entrenamiento.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa -> `llmfs check 11` -> `llmfs hint 11 -e N`
-> `SOLUCION.md` tiene el codigo completo.

El ejercicio 1 es el mas largo del curso. Los otros tres son cortos.

QUÉ VAS A CONSTRUIR
===================

Las cuatro piezas que hacen que un entrenamiento funcione a escala:

    AdamWScratch        (ej. 1)  el optimizador, desde cero
    lr_at_step          (ej. 2)  como cambia el learning rate durante la tirada
    clip_grad_norm      (ej. 3)  que un batch raro no destruya horas de trabajo
    build_param_groups  (ej. 4)  que parametros decaen y cuales no

Cuando esten las cuatro en verde, el modelo final entrenara con TU optimizador.

`TEORIA.md` los sigue en este mismo orden y cada docstring de aqui te dice que seccion le toca.
Empieza por la seccion "El bucle, y que le falta": es el bucle del modulo 02 con las cuatro
piezas colocadas en el sitio exacto donde entra cada una.

Hay una pequenya dependencia circular entre el 1 y el 4 —el `step` que escribes en el 1 recorre
los grupos que construye el 4— pero ninguno necesita al otro para funcionar ni para pasar sus
tests. Si el ejercicio 1 se te atraganta, haz el 2, 3 y 4 primero y vuelve.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **optimizador**: el algoritmo que decide como aplicar los gradientes a los pesos.
- **learning rate** (lr): cuanto se mueven los pesos en cada paso. El hiperparametro que
  mas entrenamientos arruina.
- **momento**: una media movil de los gradientes recientes, para suavizar el ruido.
- **weight decay**: empujar los pesos hacia cero para que no crezcan sin control.
- **warmup**: subir el learning rate despacio en los primeros pasos.
- **AMP / GradScaler**: entrenar en 16 bits multiplicando la perdida por un numero grande
  para que los gradientes no se vayan a cero.
- **grupos de parametros**: subconjuntos con hiperparametros distintos. PyTorch los acepta
  como lista de dicts.

    llmfs demo 11     compara tu AdamW con el de PyTorch y mide el recorte
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn as nn


class AdamWScratch(torch.optim.Optimizer):
    """AdamW desde cero. Hereda de `torch.optim.Optimizer` y solo hay que escribir `step`.

    Contexto en `TEORIA.md`: seccion "Ejercicio 1: el optimizador", con el problema que resuelve
    (un solo lr no vale para un embedding frecuente y otro raro), las dos ideas de Adam medidas
    contra SGD, y la subseccion "Como se escribe un optimizador en PyTorch" para param_groups,
    self.state y el @torch.no_grad.

    Y lee tambien "Como saber si esta bien" antes de correr la demo: te ahorra buscar un bug que
    no existe cuando veas que a 200 pasos tu optimizador y el de PyTorch se separan.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    El `__init__` YA ESTÁ HECHO. Tu unico trabajo es el metodo `step()`, y tiene una estructura
    fija: dos bucles anidados (por grupo, por parametro) y dentro seis operaciones.

        1. El esqueleto de los dos bucles:

               @torch.no_grad()
               def step(self, closure=None):
                   loss = None
                   if closure is not None:
                       with torch.enable_grad():
                           loss = closure()

                   for group in self.param_groups:
                       beta1, beta2 = group["betas"]
                       lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]

                       for p in group["params"]:
                           if p.grad is None:
                               continue
                           grad = p.grad

        2. El estado de ese parametro, creandolo la primera vez:

                           state = self.state[p]
                           if len(state) == 0:
                               state["step"] = 0
                               state["exp_avg"] = torch.zeros_like(p)
                               state["exp_avg_sq"] = torch.zeros_like(p)

                           state["step"] += 1
                           t = state["step"]
                           m, v = state["exp_avg"], state["exp_avg_sq"]

        3. EL WEIGHT DECAY, DESACOPLADO (esto va sobre el parametro, NO sobre el gradiente):

                           if wd != 0.0:
                               p.mul_(1.0 - lr * wd)

        4. Las dos medias moviles:

                           m.mul_(beta1).add_(grad, alpha=1 - beta1)
                           v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        5. La correccion de sesgo:

                           bias1 = 1 - beta1 ** t
                           bias2 = 1 - beta2 ** t
                           step_size = lr / bias1
                           denom = (v.sqrt() / math.sqrt(bias2)).add_(eps)

        6. La actualizacion:

                           p.addcdiv_(m, denom, value=-step_size)

        7. Y fuera de todos los bucles: `return loss`

    QUÉ HACE ADAM, EN DOS IDEAS
    ---------------------------
    **Momento.** En vez de moverse segun el gradiente de ESTE paso, usa una media movil de los
    recientes (eso es `m`). Cada batch es una muestra distinta y sus gradientes son ruidosos;
    promediar cancela el ruido.

    **Escalado por dimension.** Lleva tambien una media movil del gradiente AL CUADRADO (`v`) y
    divide por su raiz. Un parametro con gradientes siempre grandes se mueve poco; uno que casi
    nunca recibe senyal se mueve mucho cuando la recibe. Cada parametro acaba con su propio
    learning rate efectivo, y por eso un unico `lr` global funciona para todo el modelo.

    LAS FÓRMULAS, con t el numero de paso EMPEZANDO EN 1
    ----------------------------------------------------
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g²

        m_hat = m / (1 - beta1^t)          <- la correccion de sesgo
        v_hat = v / (1 - beta2^t)

        p -= lr * m_hat / (sqrt(v_hat) + eps)
        p -= lr * weight_decay * p         <- por separado, ver mas abajo

    En el paso 5 de arriba la correccion esta reordenada (`lr/bias1` y `sqrt(v)/sqrt(bias2)`)
    para ahorrar una division por tensor. Es la misma formula.

    CÓMO SE ESCRIBE UN OPTIMIZADOR EN PYTORCH
    -----------------------------------------
    **`self.param_groups`** son los grupos del ejercicio 4: cada uno con sus propios parametros
    y su propio `weight_decay`. Por eso los hiperparametros se leen DENTRO del bucle de grupos y
    no una vez al principio.

    **`self.state[p]`** es un `defaultdict` por parametro donde guardas m, v y el contador. La
    primera vez que tocas un parametro esta vacio (`len(state) == 0`) y hay que inicializarlo.
    PyTorch lo serializa solo en `optimizer.state_dict()`, que es lo que permite reanudar un
    entrenamiento a mitad.

    **El `@torch.no_grad()`** es obligatorio. Estas modificando parametros que tienen
    `requires_grad=True`; sin el estarias construyendo grafo de autograd sobre las propias
    actualizaciones, y ademas de estar mal se comeria la memoria.

    LOS TRES ERRORES QUE HAY QUE EVITAR
    -----------------------------------
    **Empezar t en 0.** Con t=0, `1 - beta^0 = 0` y divides por cero. Incrementa
    `state["step"]` ANTES de usarlo.

    **Olvidar la correccion de sesgo.** `m` y `v` empiezan en cero, asi que los primeros pasos
    subestiman las magnitudes. Con beta2=0.95, tras un paso `v` vale solo el 5% de g², y dividir
    por su raiz daria un paso 4,5 veces mayor de lo debido. Sin correccion, el entrenamiento
    puede diverger en los primeros pasos y no lo achacaras nunca a esto.

    **Sumar el weight decay al gradiente.** Es LA diferencia entre Adam+L2 y AdamW:

        Adam + L2:  grad = grad + wd * p     <- MAL, no es lo que queremos
        AdamW:      p.mul_(1 - lr * wd)      <- directamente sobre el parametro

    Si lo sumas al gradiente, el decaimiento pasa por la division por `sqrt(v)` y su efecto real
    acaba dependiendo de la magnitud de los gradientes de ese parametro. Desacoplado es uniforme
    y predecible. Hay un test que distingue las dos versiones.

    LAS OPERACIONES IN-PLACE (recomendado, no obligatorio)
    ------------------------------------------------------
    Con 8,9 millones de parametros, reservar tensores nuevos en cada paso se nota. Las versiones
    in-place acaban en guion bajo:

        m.mul_(beta1).add_(grad, alpha=1 - beta1)           # m = beta1*m + (1-beta1)*g
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v = beta2*v + (1-beta2)*g²
        p.addcdiv_(m, denom, value=-step_size)               # p -= step_size * m/denom

    Si te resultan cripticas, escribelo con operaciones normales primero (`m = beta1*m + ...`) y
    optimiza despues. El test compara resultados, no estilo. Pero ojo: con la version no in-place
    tienes que volver a guardar el resultado en `state["exp_avg"]`, porque estarias creando
    tensores nuevos en vez de modificar los de siempre.

    CÓMO SABER SI ESTÁ BIEN
    -----------------------
    El test entrena el mismo problema 50 pasos con tu optimizador y con `torch.optim.AdamW`, y
    compara los pesos finales. Tienen que coincidir con `torch.allclose`.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr no puede ser negativo: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"las betas deben estar en [0, 1): {betas}")
        if eps < 0.0:
            raise ValueError(f"eps no puede ser negativo: {eps}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Un paso de optimizacion. Las instrucciones completas estan en el docstring de la clase.

        El `closure` es una convencion de PyTorch que casi nadie usa, pero se respeta por
        compatibilidad. Si no es None, se llama dentro de `torch.enable_grad()` y se devuelve su
        resultado:

            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            ... el resto ...
            return loss
        """
        raise NotImplementedError("TODO: modulo 11, ejercicio 1 - AdamWScratch.step")


def lr_at_step(
    step: int,
    max_steps: int,
    lr: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
    schedule: str = "cosine",
) -> float:
    """El learning rate que toca en un paso dado: warmup lineal + decaimiento coseno.

    Contexto en `TEORIA.md`: seccion "Ejercicio 2: el planificador del learning rate", con la
    tabla de que lr toca en cada paso de la tirada final y la comprobacion de la formula del
    coseno en sus dos extremos, que es como se sabe que esta bien sin ejecutar nada.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Tres tramos, en este orden (el orden importa: el warmup se comprueba primero).

        1. El suelo, que se usa en todos los tramos:

               min_lr = lr * min_lr_ratio

        2. WARMUP. Si `step < warmup_steps`:

               return lr * (step + 1) / warmup_steps

        3. Si `schedule == "constant"`, ya has terminado: `return lr`.

        4. El progreso, acotado a [0, 1]:

               progreso = (step - warmup_steps) / max(1, max_steps - warmup_steps)
               progreso = min(1.0, max(0.0, progreso))

        5. Y segun el schedule:

               if schedule == "cosine":
                   coef = 0.5 * (1.0 + math.cos(math.pi * progreso))
               elif schedule == "linear":
                   coef = 1.0 - progreso
               else:
                   raise ValueError(f"schedule desconocido: {schedule}")

               return min_lr + (lr - min_lr) * coef

    COMPRUEBA LOS EXTREMOS A MANO
    -----------------------------
    Es la forma de saber que la formula esta bien sin ejecutar nada:

        progreso = 0  ->  cos(0) = 1   ->  coef = 1  ->  devuelve lr          OK
        progreso = 1  ->  cos(pi) = -1 ->  coef = 0  ->  devuelve min_lr      OK

    Si te sale al reves, te has dejado el `0.5 * (1 + ...)` y estas usando el coseno crudo, que
    va de 1 a -1 en vez de 1 a 0.

    EL `+1` DEL WARMUP
    ------------------
    `lr * (step + 1) / warmup_steps` en vez de `lr * step / warmup_steps`. Sin el, el paso 0
    tendria lr exactamente cero: un paso que no aprende nada y esta desperdiciado. Es un detalle
    menor pero los tests lo comprueban.

    POR QUÉ EXISTE EL WARMUP
    ------------------------
    En los primeros pasos los momentos de Adam estan casi vacios y sus estimaciones son
    ruidosisimas (es el mismo problema que arregla la correccion de sesgo, pero la correccion no
    lo resuelve del todo). Y ademas los pesos recien inicializados producen gradientes grandes.
    Arrancar a lr completo suele producir un pico de perdida del que a veces el modelo no se
    recupera nunca.

    POR QUÉ UN COSENO Y NO UNA RECTA
    --------------------------------
    Baja despacio al principio (todavia interesa moverse rapido y explorar), deprisa en medio, y
    otra vez despacio al final (afinando en una zona buena). La diferencia frente a lineal es
    pequenya pero consistente en todos los papers que lo han medido.

    POR QUÉ NO SE DECAE HASTA CERO
    ------------------------------
    `min_lr_ratio=0.1` deja un suelo del 10%. Por debajo de cierto punto el modelo deja de
    aprender del todo y cada paso extra es computo tirado. Si vas a parar, mejor parar.

    DOS PROTECCIONES QUE NO SON DECORATIVAS
    ---------------------------------------
    El `max(1, ...)` del denominador evita dividir por cero si `max_steps <= warmup_steps`.
    El acotado a [0, 1] hace que llamar con `step > max_steps` devuelva `min_lr` en vez de
    empezar a SUBIR otra vez (el coseno es periodico: con progreso > 1 volveria a crecer).

    Args:
        step: el paso actual, empezando en 0.
        max_steps: el total de pasos de la tirada.
        lr: el learning rate maximo, el del pico.
        warmup_steps: cuantos pasos dura la subida.
        min_lr_ratio: la fraccion de `lr` que es el suelo.
        schedule: "cosine" (por defecto), "linear" o "constant".

    Returns:
        El learning rate de ese paso.

    Raises:
        ValueError: si `schedule` no es ninguno de los tres.
    """
    raise NotImplementedError("TODO: modulo 11, ejercicio 2 - lr_at_step")


def clip_grad_norm(parameters: Iterable[nn.Parameter], max_norm: float) -> float:
    """Recorta los gradientes para que su norma GLOBAL no pase de `max_norm`.

    Contexto en `TEORIA.md`: seccion "Ejercicio 3: el recorte de gradientes", con el coseno de
    0,99999994 que demuestra que la direccion no cambia, y el efecto medido de un batch
    envenenado: sin recortar la perdida SUBE 3x, con recorte ni se entera.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
        1. Reune los gradientes que existen:

               grads = [p.grad for p in parameters if p.grad is not None]
               if not grads:
                   return 0.0

        2. La norma global, como si todos fueran UN SOLO vector gigante:

               total = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
               total_norm = float(total)

        3. El recorte, solo si hace falta:

               if max_norm > 0 and total_norm > max_norm:
                   factor = max_norm / (total_norm + 1e-6)
                   for g in grads:
                       g.mul_(factor)

        4. `return total_norm`   <- la norma ANTES de recortar

    Ojo con el paso 1: `parameters` puede ser un generador (`model.parameters()` lo es), y un
    generador se agota al recorrerlo. Si lo recorres dos veces, la segunda esta vacio. Por eso
    se materializa la lista de gradientes UNA vez y se trabaja siempre sobre ella.

    QUÉ PROBLEMA RESUELVE
    ---------------------
    Ocasionalmente un batch produce gradientes enormes: una secuencia rara, un token muy poco
    frecuente, una linea corrupta del dataset. Sin proteccion, ese UNICO batch puede dar un
    salto que destruya horas de entrenamiento, y lo veras como un pico vertical en la curva de
    perdida del que el modelo tarda mucho en recuperarse (o no se recupera).

    POR QUÉ LA NORMA GLOBAL Y NO UNA POR TENSOR
    -------------------------------------------
    Recortar cada tensor por separado cambiaria la DIRECCION del gradiente conjunto, que es
    justo lo que no quieres tocar. El gradiente te dice hacia donde ir; tu solo estas limitando
    CUANTO avanzas en esa direccion. Multiplicando todos los tensores por el mismo escalar, la
    direccion se conserva exactamente.

    Hay un test que lo comprueba: el vector normalizado antes y despues del recorte tiene que
    ser identico.

    POR QUÉ SE DEVUELVE LA NORMA **ANTES** DE RECORTAR
    --------------------------------------------------
    Es lo que hace `torch.nn.utils.clip_grad_norm_`, y es lo util. Si la registras en el log y
    ves que sube de forma sostenida, el entrenamiento se esta desestabilizando y te enteras
    ANTES de que reviente. Si devolvieses la norma posterior verias `max_norm` clavado y no te
    enterarias de nada.

    EL `1e-6` Y EL `.detach()`
    --------------------------
    El `1e-6` del denominador evita dividir por cero si la norma es minuscula. Nunca pasa en la
    practica, pero cuesta cinco caracteres.

    El `.detach()` al calcular la norma: los gradientes no requieren gradiente, asi que ahora
    mismo da igual. Es la costumbre correcta y evita sorpresas si alguna vez usas grafos de
    orden superior.

    Args:
        parameters: los parametros del modelo (`model.parameters()`).
        max_norm: el umbral. Si es <= 0, no recorta nada (pero sigue devolviendo la norma).

    Returns:
        La norma global ANTES de recortar. `0.0` si no hay ningun gradiente.
    """
    raise NotImplementedError("TODO: modulo 11, ejercicio 3 - clip_grad_norm")


def build_param_groups(model: nn.Module, weight_decay: float = 0.1) -> list[dict[str, Any]]:
    """Separa los parametros en dos grupos: con weight decay y sin el.

    Contexto en `TEORIA.md`: seccion "Ejercicio 4: que parametros decaen", con el reparto real
    del modelo (43 tensores y 8.929.280 parametros con decay, 13 y 4.160 sin el) y por que esos
    13 son exactamente las capas de normalizacion que contaste en el modulo 10.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Cinco lineas.

        1. Las dos listas:

               decay, no_decay = [], []

        2. El reparto, saltando lo congelado:

               for param in model.parameters():
                   if not param.requires_grad:
                       continue
                   (decay if param.dim() >= 2 else no_decay).append(param)

        3. Los dos grupos, EN ESTE ORDEN (hay tests que dependen del orden):

               return [
                   {"params": decay, "weight_decay": weight_decay},
                   {"params": no_decay, "weight_decay": 0.0},
               ]

    LA REGLA, y es sorprendentemente simple
    ---------------------------------------
        Parametros de 2 dimensiones o mas   ->  CON weight decay
        Parametros de 1 dimension           ->  SIN weight decay

    O sea: las matrices decaen, y los sesgos y las escalas de normalizacion no. `param.dim()` da
    el numero de dimensiones: una matriz de pesos tiene 2, un sesgo tiene 1.

    POR QUÉ ESA REGLA
    -----------------
    El weight decay empuja los pesos hacia cero. En una matriz de proyeccion tiene sentido:
    penalizar magnitudes grandes reduce el sobreajuste.

    En la escala de un RMSNorm no tiene NINGUNO. Ese parametro arranca en 1 y su trabajo es
    reescalar la salida de la capa; empujarlo hacia cero es empujar la salida hacia cero, que es
    exactamente lo contrario de lo que hace falta.

    Lo mismo con los sesgos: son desplazamientos, no magnitudes que convenga limitar.

    Aplicar decay a todo es un error frecuente, NO da ningun error visible, y degrada el
    resultado. Solo se detecta comparando dos entrenamientos completos, que es caro. Por eso
    merece la pena tenerlo bien de entrada.

    EL FORMATO QUE ESPERA PYTORCH
    -----------------------------
    Una lista de diccionarios, cada uno con al menos la clave `"params"`. Cualquier clave
    adicional (`lr`, `betas`, `weight_decay`...) sobreescribe el valor por defecto del
    optimizador SOLO para ese grupo. Es el mecanismo estandar de PyTorch, y es lo que lee tu
    `AdamWScratch.step` cuando hace `for group in self.param_groups`.

    DOS DETALLES
    ------------
    **Saltar `requires_grad=False`.** Esos parametros no se van a actualizar; meterlos en el
    optimizador solo gasta memoria de estado (dos tensores por parametro). En el modulo 16, con
    LoRA, esto pasa de detalle a esencial: casi todo el modelo esta congelado.

    **Los pesos atados.** `model.parameters()` ya deduplica por identidad, asi que el embedding
    atado aparece UNA sola vez y va al grupo con decay (tiene 2 dimensiones). No hay que hacer
    nada especial.

    Args:
        model: el modelo.
        weight_decay: el valor para el grupo que si decae.

    Returns:
        La lista de dos grupos, el primero con decay y el segundo sin el, EN ESE ORDEN.
    """
    raise NotImplementedError("TODO: modulo 11, ejercicio 4 - build_param_groups")
