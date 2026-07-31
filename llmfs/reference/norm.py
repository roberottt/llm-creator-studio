"""Referencia del modulo 07: normalizacion y conexiones residuales."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm: centra en 0, escala a varianza 1, y luego aplica gamma y beta.

    $$y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\cdot \\gamma + \\beta$$

    La media y la varianza se calculan sobre la ULTIMA dimension (las features de cada
    token), por separado para cada token. Nada que ver con BatchNorm, que normaliza a lo
    largo del batch: LayerNorm trata cada token de forma independiente, y por eso funciona
    igual con batch de 1 que de 1000 y no necesita estadisticas acumuladas.

    OJO con la varianza: se usa la POBLACIONAL (dividir entre n), no la muestral (entre
    n-1). Es lo que hace `F.layer_norm`, y usar `torch.var` con su `unbiased=True` por
    defecto da un resultado ligeramente distinto.
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(var + eps)

    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias
    return normalized


class RMSNorm(nn.Module):
    """RMSNorm: como LayerNorm pero sin restar la media ni sumar sesgo.

    $$y = \\frac{x}{\\sqrt{\\frac{1}{d}\\sum_i x_i^2 + \\epsilon}} \\cdot \\gamma$$

    Zhang y Sennrich (2019) observaron que casi todo el beneficio de LayerNorm viene de
    reescalar, no de recentrar. Quitando la media se ahorra una pasada por los datos y un
    tensor intermedio: entre un 7% y un 64% mas rapido segun el caso, y sin perdida de
    calidad medible. Por eso lo usan Llama, Mistral y practicamente todo lo moderno.

    Detalle de implementacion importante: el calculo se hace en float32 aunque la entrada
    venga en float16. Elevar al cuadrado valores de una activacion grande puede desbordar
    el rango de fp16 (65504), y el resultado seria `inf`.

    Submodulos:
        weight: parametro de forma `(dim,)`, inicializado a unos.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


def prenorm_residual(
    x: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    norm: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Pre-norm: `x + fn(norm(x))`.

    Compara con post-norm, que es `norm(x + fn(x))`. La diferencia parece cosmetica y
    decide si una red profunda entrena o no.

    En pre-norm, el camino `x -> salida` incluye un `+x` sin nada por medio. Derivando,
    ese sumando aporta un 1 al gradiente que llega intacto a las capas de abajo, capa tras
    capa. Es una autopista para el gradiente.

    En post-norm, la normalizacion esta ENCIMA del camino residual, asi que el gradiente
    la atraviesa en cada capa y se va reescalando. Con 6 capas se nota poco; con 40 hace
    falta un warmup cuidadoso para que no explote o se muera.

    El precio de pre-norm es que la salida del bloque tiene una escala que crece con la
    profundidad (cada capa suma algo a la corriente residual). Por eso los modelos
    pre-norm llevan SIEMPRE una normalizacion final antes de la capa de salida.
    """
    return x + fn(norm(x))


def postnorm_residual(
    x: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    norm: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Post-norm: `norm(x + fn(x))`. Lo que hacia el paper original de 2017.

    No es un ejercicio; esta aqui para que la demo pueda comparar ambas empiricamente.
    """
    return norm(x + fn(x))
