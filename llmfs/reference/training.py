"""Referencia del modulo 11: el bucle de entrenamiento.

Aqui estan las cuatro piezas que convierten "tengo un modelo" en "tengo un modelo
entrenado": el optimizador, el planificador del learning rate, el recorte de gradientes y
la separacion de parametros para el weight decay.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.nn as nn


class AdamWScratch(torch.optim.Optimizer):
    """AdamW implementado desde cero, para que deje de ser una caja negra.

    Adam combina dos ideas:

    1. **Momento** (primer momento, `m`): en vez de moverse segun el gradiente de este paso,
       se mueve segun una media movil de los gradientes recientes. Suaviza el ruido del
       muestreo de batches.

    2. **Escalado por dimension** (segundo momento, `v`): cada parametro se divide por la
       raiz de la media movil de sus gradientes AL CUADRADO. Los parametros con gradientes
       consistentemente grandes se mueven poco, y los que casi nunca reciben senyal se
       mueven mucho cuando la reciben. Eso hace que un unico learning rate valga para toda
       la red.

    Las formulas, con `t` el numero de paso (empezando en 1):

        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g^2

        m_hat = m / (1 - beta1^t)          <- correccion de sesgo
        v_hat = v / (1 - beta2^t)

        p -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)

    LA CORRECCION DE SESGO. `m` y `v` empiezan en cero, asi que en los primeros pasos
    subestiman la magnitud real: con beta2=0.95, tras un paso `v` vale solo el 5% de g^2.
    Dividir por `1 - beta^t` lo compensa exactamente. Sin ella, los primeros pasos dan
    saltos enormes y el entrenamiento puede diverger antes de empezar.

    LA W DE ADAMW. El weight decay se aplica DIRECTAMENTE al parametro, no sumandolo al
    gradiente. La diferencia importa: si lo sumaras al gradiente, pasaria por el escalado
    de `v` y su efecto real dependeria de la magnitud de los gradientes de ese parametro.
    Desacoplado (Loshchilov y Hutter, 2019) el decaimiento es uniforme y predecible.
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

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                # Medias moviles, in-place para no reservar tensores nuevos cada paso.
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t

                denom = (v / bias_correction2).sqrt_().add_(eps)
                step_size = lr / bias_correction1

                # Weight decay DESACOPLADO: directamente sobre el parametro.
                if wd != 0.0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m, denom, value=-step_size)

        return loss


def lr_at_step(
    step: int,
    max_steps: int,
    lr: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
    schedule: str = "cosine",
) -> float:
    """El learning rate que toca en un paso dado: warmup lineal + decaimiento coseno.

    Tres tramos:

        1. `step < warmup_steps`   -> subida lineal de 0 a `lr`
        2. `step >= max_steps`     -> se queda en `lr * min_lr_ratio`
        3. en medio                -> coseno de `lr` a `lr * min_lr_ratio`

    EL WARMUP. En los primeros pasos, los momentos de Adam estan casi vacios y sus
    estimaciones son ruidosas. Ademas los pesos estan recien inicializados y los gradientes
    son grandes. Arrancar a lr completo suele producir un pico de perdida del que a veces
    el modelo no se recupera. Subir despacio evita ese destrozo inicial.

    EL COSENO. Al principio interesa moverse rapido; al final, afinar. El coseno baja
    despacio al principio, deprisa en medio y despacio al final. Frente a un decaimiento
    lineal la diferencia es pequenya pero consistente, y es lo que usa todo el mundo.

    EL SUELO DEL 10%. No se decae hasta cero: por debajo de cierto punto el modelo deja de
    aprender del todo y se desperdicia computo. El 10% es la convencion.

    Formula del tramo 3, con `progreso` de 0 a 1:

        coef = 0.5 * (1 + cos(pi * progreso))       va de 1 a 0
        lr_t = lr * (min_lr_ratio + (1 - min_lr_ratio) * coef)
    """
    min_lr = lr * min_lr_ratio

    if warmup_steps > 0 and step < warmup_steps:
        # +1 para que el paso 0 no tenga lr exactamente cero (no aprenderia nada).
        return lr * (step + 1) / warmup_steps

    if step >= max_steps:
        return min_lr

    if schedule == "constant":
        return lr

    progreso = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progreso = min(1.0, max(0.0, progreso))

    if schedule == "linear":
        return lr - (lr - min_lr) * progreso

    coef = 0.5 * (1.0 + math.cos(math.pi * progreso))
    return min_lr + (lr - min_lr) * coef


def clip_grad_norm(parameters: Iterable[nn.Parameter], max_norm: float) -> float:
    """Recorta los gradientes para que su norma GLOBAL no pase de `max_norm`.

    Se calcula la norma de todos los gradientes juntos, como si fueran un unico vector
    gigante:

        norma = sqrt( suma de ||g_i||^2 sobre todos los parametros )

    Si supera `max_norm`, TODOS los gradientes se multiplican por `max_norm / norma`.

    POR QUE LA NORMA GLOBAL Y NO UNA POR TENSOR. Recortar cada tensor por separado
    cambiaria la DIRECCION del gradiente conjunto, que es justo lo que no quieres: el
    gradiente apunta a donde hay que ir, y solo estas limitando cuanto avanzas. Con la
    norma global, la direccion se conserva exactamente.

    PARA QUE SIRVE. Ocasionalmente un batch produce gradientes enormes (una secuencia rara,
    un token muy poco frecuente). Sin recortar, ese unico batch puede dar un salto que
    destruya horas de entrenamiento. Con grad_clip=1.0 el danyo maximo esta acotado.

    Returns:
        La norma ANTES de recortar. Merece la pena registrarla: si sube de forma sostenida,
        el entrenamiento se esta desestabilizando.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    total = torch.sqrt(sum(g.detach().pow(2).sum() for g in grads))
    total_f = float(total)

    if max_norm > 0 and total_f > max_norm:
        # 1e-6 para no dividir por cero si la norma es minuscula.
        factor = max_norm / (total_f + 1e-6)
        for g in grads:
            g.mul_(factor)

    return total_f


def build_param_groups(
    model: nn.Module, weight_decay: float = 0.1
) -> list[dict[str, Any]]:
    """Separa los parametros en dos grupos: con weight decay y sin el.

    LA REGLA: **decay solo en los pesos de las matrices** (todo lo de 2 dimensiones o mas).
    Sesgos, escalas de normalizacion y cualquier parametro de 1 dimension van SIN decay.

    POR QUE. El weight decay empuja los pesos hacia cero, y eso tiene sentido en una matriz
    de proyeccion: penalizar magnitudes grandes reduce el sobreajuste.

    En una escala de RMSNorm no tiene ningun sentido. Ese parametro arranca en 1 y su
    trabajo es reescalar; empujarlo hacia cero es empujar la salida de la capa hacia cero,
    que es exactamente lo contrario de lo que hace falta. Lo mismo con los sesgos: son
    desplazamientos, no magnitudes que convenga limitar.

    Aplicar decay a todo es un error frecuente, no da ningun error visible y degrada el
    resultado. Se detecta comparando dos entrenamientos, que es caro.

    NOTA SOBRE EL WEIGHT TYING. `model.parameters()` deduplica por identidad, asi que el
    embedding atado aparece una sola vez y va al grupo con decay (tiene 2 dimensiones).

    Returns:
        Una lista de dos dicts en el formato que esperan los optimizadores de PyTorch:
        `[{"params": [...], "weight_decay": wd}, {"params": [...], "weight_decay": 0.0}]`
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
