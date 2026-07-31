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

    QUE HACE ADAM, EN DOS IDEAS

    1. MOMENTO. En vez de moverse segun el gradiente de este paso, usa una media movil de
       los recientes. Cada batch es una muestra distinta y sus gradientes son ruidosos;
       promediar cancela el ruido.

    2. ESCALADO POR DIMENSION. Lleva tambien una media movil del gradiente AL CUADRADO y
       divide por su raiz. Un parametro con gradientes siempre grandes se mueve poco; uno
       que casi nunca recibe senyal se mueve mucho cuando la recibe. Cada parametro acaba
       con su propio learning rate efectivo, y por eso un unico lr global funciona.

    LAS FORMULAS, con t el numero de paso EMPEZANDO EN 1

        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g^2

        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)

        p -= lr * m_hat / (sqrt(v_hat) + eps)
        p -= lr * weight_decay * p          <- por separado, ver mas abajo

    COMO SE ESCRIBE UN OPTIMIZADOR EN PYTORCH

        El `__init__` ya esta hecho: guarda los defaults y llama a `super().__init__`.
        Tu escribes `step()`, con esta estructura:

            @torch.no_grad()
            def step(self, closure=None):
                for group in self.param_groups:
                    beta1, beta2 = group["betas"]
                    lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]

                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        grad = p.grad

                        state = self.state[p]          # dict por parametro, persiste
                        if len(state) == 0:            # primera vez: inicializar
                            state["step"] = 0
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)

                        state["step"] += 1
                        ...

        `self.param_groups` son los grupos del ejercicio 4: cada uno con sus propios
        parametros y su propio weight_decay.

        `self.state[p]` es un diccionario por parametro donde guardas m, v y el contador de
        pasos. PyTorch lo serializa solo cuando haces `optimizer.state_dict()`, que es lo
        que permite reanudar un entrenamiento.

        El `@torch.no_grad()` es obligatorio: estas modificando parametros que tienen
        `requires_grad=True`, y sin el estarias construyendo grafo de autograd sobre las
        propias actualizaciones.

    LOS TRES ERRORES QUE HAY QUE EVITAR

    1. EMPEZAR t EN 0. Con t=0, `1 - beta^0 = 0` y divides por cero. Incrementa
       `state["step"]` ANTES de usarlo.

    2. OLVIDAR LA CORRECCION DE SESGO. m y v empiezan en cero, asi que los primeros pasos
       subestiman las magnitudes. Con beta2=0.95, tras un paso v vale solo el 5% de g^2 y
       dividir por su raiz daria un paso 4,5 veces mayor de lo debido. Sin la correccion,
       el entrenamiento puede diverger en los primeros pasos.

    3. SUMAR EL WEIGHT DECAY AL GRADIENTE. Es la diferencia entre Adam+L2 y AdamW:

           Adam + L2:  grad = grad + wd * p     <- MAL, es lo que NO queremos
           AdamW:      p.mul_(1 - lr * wd)      <- directamente sobre el parametro

       Si lo sumas al gradiente, el decaimiento pasa por la division por sqrt(v) y su
       efecto real depende de la magnitud de los gradientes de ese parametro. Desacoplado
       es uniforme y predecible. Hay un test que distingue las dos versiones.

    OPERACIONES IN-PLACE (recomendado, no obligatorio)
        Con 8,9 millones de parametros, reservar tensores nuevos en cada paso se nota.
        Las versiones in-place de torch acaban en guion bajo:

            m.mul_(beta1).add_(grad, alpha=1 - beta1)          # m = beta1*m + (1-beta1)*g
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) # v = beta2*v + (1-beta2)*g^2
            p.addcdiv_(m, denom, value=-step_size)              # p -= step_size * m/denom

        Si te resultan crípticas, escríbelo con operaciones normales primero y optimiza
        después. El test compara resultados, no estilo.
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
        """Un paso de optimizacion. Ver el docstring de la clase.

        Si `closure` no es None, hay que llamarla dentro de `torch.enable_grad()` y
        devolver su resultado. Es una convencion de PyTorch que casi nadie usa, pero se
        respeta por compatibilidad:

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

    TRES TRAMOS

        1. step < warmup_steps      -> subida lineal de casi 0 hasta lr
        2. step >= max_steps        -> se queda fijo en lr * min_lr_ratio
        3. en medio                 -> coseno de lr hasta lr * min_lr_ratio

    EL WARMUP (tramo 1)

        return lr * (step + 1) / warmup_steps

        El `+1` es para que el paso 0 no tenga lr exactamente cero: un paso con lr=0 no
        aprende nada y es un paso desperdiciado.

        Por que existe el warmup: en los primeros pasos los momentos de Adam estan casi
        vacios y sus estimaciones son ruidosas, y ademas los pesos recien inicializados dan
        gradientes grandes. Arrancar a lr completo suele producir un pico de perdida del
        que a veces el modelo no se recupera.

    EL COSENO (tramo 3)

        progreso = (step - warmup_steps) / (max_steps - warmup_steps)     # de 0 a 1
        coef     = 0.5 * (1 + cos(pi * progreso))                         # de 1 a 0
        return min_lr + (lr - min_lr) * coef

        Comprueba los extremos: con progreso=0, cos(0)=1 y coef=1, asi que devuelve `lr`.
        Con progreso=1, cos(pi)=-1 y coef=0, asi que devuelve `min_lr`. Correcto.

        Por que un coseno y no una recta: baja despacio al principio (todavia interesa
        moverse rapido), deprisa en medio, y despacio al final (afinando). La diferencia
        frente a lineal es pequenya pero consistente.

    EL SUELO
        `min_lr = lr * min_lr_ratio`, con min_lr_ratio=0.1 por defecto. No se decae hasta
        cero porque por debajo de cierto punto el modelo deja de aprender del todo y se
        desperdicia computo.

    DETALLES
        - Acota `progreso` a [0, 1] con `min(1.0, max(0.0, progreso))`: protege contra
          divisiones raras si max_steps <= warmup_steps.
        - Usa `max(1, ...)` en el denominador para no dividir por cero.
        - `schedule` puede ser "cosine" (por defecto), "linear" o "constant". Los dos
          ultimos son de una linea cada uno y hay tests para ambos.

    Returns:
        El learning rate de ese paso.
    """
    raise NotImplementedError("TODO: modulo 11, ejercicio 2 - lr_at_step")


def clip_grad_norm(parameters: Iterable[nn.Parameter], max_norm: float) -> float:
    """Recorta los gradientes para que su norma GLOBAL no pase de `max_norm`.

    QUE ES ESTO
        Ocasionalmente un batch produce gradientes enormes: una secuencia rara, un token
        muy poco frecuente. Sin proteccion, ese unico batch puede dar un salto que destruya
        horas de entrenamiento.

    EL CALCULO

        1. Reune todos los `p.grad` que no sean None.
        2. Calcula la norma como si todos los gradientes fueran UN SOLO vector gigante:

               norma = sqrt( suma de (g**2).sum() sobre todos los tensores )

        3. Si `norma > max_norm`, multiplica TODOS los gradientes por
           `max_norm / (norma + 1e-6)`.

        El `1e-6` evita dividir por cero si la norma es minuscula.

    POR QUE LA NORMA GLOBAL Y NO UNA POR TENSOR
        Recortar cada tensor por separado cambiaria la DIRECCION del gradiente conjunto,
        que es justo lo que no quieres. El gradiente apunta a donde hay que ir; tu solo
        estas limitando cuanto avanzas en esa direccion. Con la norma global, la direccion
        se conserva exactamente.

        Hay un test que lo comprueba: la direccion normalizada antes y despues del recorte
        tiene que ser la misma.

    DEVUELVE LA NORMA **ANTES** DE RECORTAR
        Es lo que hace `torch.nn.utils.clip_grad_norm_` y es lo util: si la registras y
        ves que sube de forma sostenida, el entrenamiento se esta desestabilizando y te
        enteras antes de que reviente.

    OJO CON `.detach()`
        Al calcular la norma, usa `g.detach()`. Los gradientes no requieren gradiente, pero
        es la costumbre correcta y evita sorpresas si alguna vez usas grafos de orden
        superior.

    Args:
        parameters: los parametros del modelo (`model.parameters()`).
        max_norm: el umbral. Si es <= 0, no recorta nada (pero sigue devolviendo la norma).

    Returns:
        La norma global ANTES de recortar. `0.0` si no hay ningun gradiente.
    """
    raise NotImplementedError("TODO: modulo 11, ejercicio 3 - clip_grad_norm")


def build_param_groups(model: nn.Module, weight_decay: float = 0.1) -> list[dict[str, Any]]:
    """Separa los parametros en dos grupos: con weight decay y sin el.

    LA REGLA, y es sorprendentemente simple

        Parametros de 2 dimensiones o mas   -> CON weight decay
        Parametros de 1 dimension           -> SIN weight decay

    O sea: las matrices decaen, y los sesgos y las escalas de normalizacion no.

    POR QUE
        El weight decay empuja los pesos hacia cero. En una matriz de proyeccion tiene
        sentido: penalizar magnitudes grandes reduce el sobreajuste.

        En la escala de un RMSNorm no tiene ninguno. Ese parametro arranca en 1 y su
        trabajo es reescalar la salida de la capa; empujarlo hacia cero es empujar la
        salida hacia cero, que es exactamente lo contrario de lo que hace falta.

        Lo mismo con los sesgos: son desplazamientos, no magnitudes que convenga limitar.

        Aplicar decay a todo es un error frecuente, NO da ningun error visible, y degrada
        el resultado. Solo se detecta comparando dos entrenamientos completos, que es caro.

    EL FORMATO QUE ESPERA PYTORCH
        Una lista de diccionarios, cada uno con al menos la clave "params":

            [
                {"params": [...matrices...], "weight_decay": weight_decay},
                {"params": [...vectores...], "weight_decay": 0.0},
            ]

        Cualquier clave adicional (lr, betas...) sobreescribe el valor por defecto del
        optimizador SOLO para ese grupo. Es el mecanismo estandar de PyTorch para tener
        hiperparametros distintos por grupo.

    DETALLES
        - `param.dim()` da el numero de dimensiones. Una matriz de pesos tiene 2, un sesgo
          tiene 1.
        - Salta los parametros con `requires_grad=False`: no se van a actualizar y meterlos
          en el optimizador solo gasta memoria.
        - Recorre `model.parameters()`, que ya deduplica los pesos atados por identidad.
          El embedding atado aparece una sola vez y va al grupo con decay (tiene 2
          dimensiones).

    Returns:
        La lista de dos grupos, el primero con decay y el segundo sin el, EN ESE ORDEN
        (hay tests que dependen del orden).
    """
    raise NotImplementedError("TODO: modulo 11, ejercicio 4 - build_param_groups")
