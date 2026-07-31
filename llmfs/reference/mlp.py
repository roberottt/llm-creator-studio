"""Referencia del modulo 08: la red feed-forward y sus activaciones."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU con la aproximacion por tanh.

    $$\\text{GELU}(x) \\approx 0.5x\\left(1 + \\tanh\\left[\\sqrt{2/\\pi}(x + 0.044715x^3)\\right]\\right)$$

    La definicion exacta es `x * Phi(x)`, con Phi la funcion de distribucion acumulada de
    la normal estandar. Interpretacion: en vez de decidir si dejar pasar `x` con un corte
    duro (como ReLU), lo multiplica por la probabilidad de que una normal sea menor que
    `x`. Los valores muy negativos se anulan casi del todo, los muy positivos pasan casi
    enteros, y en medio hay una transicion suave.

    La aproximacion por tanh existe porque `erf` era lento en las GPU de 2016. Hoy la
    diferencia es despreciable, pero GPT-2 se entreno con la aproximacion y por
    compatibilidad se sigue usando. Equivale a `F.gelu(x, approximate="tanh")`.

    Frente a ReLU, la ventaja practica es que la derivada no es exactamente cero para
    entradas negativas, asi que una neurona que se va a la zona negativa puede recuperarse.
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def swiglu_hidden_dim(
    d_model: int, multiple_of: int = 64, ffn_dim_multiplier: float | None = None
) -> int:
    """Calcula `d_ff` para SwiGLU: dos tercios del 4x clasico, redondeado hacia arriba.

    De donde sale:

    - El FFN clasico usa `d_ff = 4 * d_model` y tiene DOS matrices: `d -> 4d` y `4d -> d`.
      Son `8 * d^2` parametros.
    - SwiGLU tiene TRES matrices (gate, up, down), asi que con el mismo `d_ff` tendria
      `12 * d^2`: un 50% mas.
    - Para gastar los mismos parametros se multiplica por 2/3: `d_ff = (2/3) * 4 * d`.
      Entonces `3 * d * (8/3)d = 8 * d^2`, igual que antes.

    Despues se redondea HACIA ARRIBA al siguiente multiplo de `multiple_of` (64 por
    defecto). No es cosmetica: las dimensiones alineadas a potencias de dos permiten a los
    tensor cores usar sus rutas rapidas. Una matriz de 853 columnas es notablemente mas
    lenta que una de 896.

    Con `d_model=320`:
        (2/3) * 4 * 320 = 853,33  ->  ceil al multiplo de 64  ->  896   <- el del config
    """
    hidden = int(2 * (4 * d_model) / 3)
    if ffn_dim_multiplier is not None:
        hidden = int(ffn_dim_multiplier * hidden)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class SwiGLU(nn.Module):
    """FFN con puerta: `down(Swish(gate(x)) * up(x))`.

    $$\\text{SwiGLU}(x) = \\left(\\text{Swish}(x W_{gate}) \\odot x W_{up}\\right) W_{down}$$

    con `Swish(z) = z * sigmoid(z)` (tambien llamada SiLU).

    La idea de las variantes GLU es que una de las dos ramas actua como PUERTA: multiplica
    a la otra elemento a elemento y decide cuanta senyal pasa por cada dimension. A
    diferencia de una activacion normal, ese filtrado depende de la entrada.

    Shazeer (2020) las probo todas y SwiGLU salio la mejor de forma consistente. Su propia
    conclusion sobre el porque, citada literalmente, es: "We offer no explanation as to why
    these architectures seem to work; we attribute their success, as all else, to divine
    benevolence." Es una de las decisiones de arquitectura mas usadas y peor entendidas del
    campo, y conviene saberlo.

    Submodulos:
        gate_proj: nn.Linear(d_model, d_ff, bias=bias)
        up_proj:   nn.Linear(d_model, d_ff, bias=bias)
        down_proj: nn.Linear(d_ff, d_model, bias=bias)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class MLP(nn.Module):
    """El FFN clasico de dos matrices con GELU, para comparar en la demo.

    No es un ejercicio: sirve de punto de referencia frente a SwiGLU.

    Submodulos:
        fc_in:  nn.Linear(d_model, d_ff, bias=bias)
        fc_out: nn.Linear(d_ff, d_model, bias=bias)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff, bias=bias)
        self.fc_out = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(gelu(self.fc_in(x))))
