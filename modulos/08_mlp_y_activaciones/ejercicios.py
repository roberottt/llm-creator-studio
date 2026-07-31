"""Modulo 08 - FFN, GELU y SwiGLU.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa -> `llmfs check 08` -> `llmfs hint 08 -e N`
-> `SOLUCION.md` tiene el codigo completo.

QUÉ VAS A CONSTRUIR
===================

La parte del Transformer donde estan DOS TERCIOS de los parametros:

    gelu               (ej. 1)  la no-linealidad clasica
    swiglu_hidden_dim  (ej. 2)  aritmetica: de aqui sale el 896 del config final
    SwiGLU             (ej. 3)  el FFN con puerta que usa el modelo

El ejercicio 2 es el mas corto del curso y produce un numero que ya has visto en el YAML.

POR QUÉ HACE FALTA ESTO
=======================

La atencion es una media ponderada, o sea una operacion LINEAL. Y dos operaciones lineales
seguidas son una sola:

    W2 · (W1 · x) = (W2 · W1) · x

Cien capas lineales apiladas equivalen a UNA. Lo unico que impide que el Transformer entero
se derrumbe es la no-linealidad de este modulo.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **FFN / MLP** (feed-forward network): la parte de cada bloque que NO es atencion. Procesa
  cada token por separado, sin mirar a los demas.
- **activacion**: la funcion no lineal que va entre capas. ReLU, GELU, Swish.
- **no-linealidad**: cualquier funcion que no sea `f(ax+b) = a·f(x)+b`. Es lo que hace que
  apilar capas sirva de algo.
- **puerta** (gate): en SwiGLU, una de las dos ramas multiplica a la otra y decide cuanta
  senyal pasa por cada dimension. A diferencia de una activacion normal, ese filtrado
  depende de la entrada.
- **d_ff**: la dimension interna del FFN. En nuestro modelo, 896.

    llmfs demo 08     ensenya el colapso lineal y compara las activaciones
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU con la aproximacion por tanh.

    QUE ES ESTO
        La no-linealidad. Sin algo asi entre capa y capa, apilar capas no sirve de nada:
        dos multiplicaciones de matriz seguidas son una sola multiplicacion de matriz.

        GELU multiplica x por la probabilidad de que una normal estandar salga menor que x.
        En vez de decidir con un corte duro si dejar pasar el valor (como hace ReLU), lo
        atenua de forma gradual:

            x = -3  ->  GELU = -0.003     casi anulado
            x = -1  ->  GELU = -0.159     parcialmente
            x =  0  ->  GELU =  0
            x =  1  ->  GELU =  0.841     casi entero
            x =  3  ->  GELU =  2.996     entero

        La ventaja sobre ReLU: la derivada nunca es exactamente cero, asi que una neurona
        que se va a la zona negativa puede recuperarse. Con ReLU se queda muerta.

    LA FORMULA (la aproximacion por tanh, que es la que se pide)

        GELU(x) ~= 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))

        Escrita asi, de izquierda a derecha, sin reagrupar. Los numeros son los del paper:
        sqrt(2/pi) ~= 0.7978 y 0.044715.

    POR QUE UNA APROXIMACION Y NO LA EXACTA
        La definicion exacta usa `erf`, que era lento en las GPU de 2016. Hoy la diferencia
        es despreciable, pero GPT-2 se entreno con la aproximacion y por compatibilidad se
        sigue usando. Tu resultado tiene que coincidir con
        `F.gelu(x, approximate="tanh")`, no con `F.gelu(x)` a secas.

    PISTAS DE IMPLEMENTACION
        - `math.sqrt(2.0 / math.pi)` para la constante.
        - `x.pow(3)` o `x**3` para el cubo.
        - `torch.tanh(...)` para la tangente hiperbolica.
        - Es una expresion de una linea; no hace falta ningun bucle ni ninguna rama.

    Args:
        x: cualquier forma.

    Returns:
        Del mismo tamanyo que `x`.
    """
    raise NotImplementedError("TODO: modulo 08, ejercicio 1 - gelu")


def swiglu_hidden_dim(
    d_model: int, multiple_of: int = 64, ffn_dim_multiplier: float | None = None
) -> int:
    """Calcula `d_ff` para SwiGLU. Este ejercicio produce el 896 del config final.

    QUE ES ESTO
        Aritmetica, pero con una decision de disenyo detras.

        El FFN clasico usa d_ff = 4 * d_model y tiene DOS matrices (d->4d y 4d->d), o sea
        8*d^2 parametros.

        SwiGLU tiene TRES matrices (gate, up, down). Con el mismo d_ff serian 12*d^2: un
        50% mas. Para gastar los mismos parametros se multiplica por 2/3:

            3 * d * (2/3 * 4d) = 3 * d * (8/3)d = 8*d^2      igual que antes

    LOS PASOS
        1. hidden = int(2 * (4 * d_model) / 3)
        2. si `ffn_dim_multiplier` no es None: hidden = int(ffn_dim_multiplier * hidden)
        3. redondear HACIA ARRIBA al siguiente multiplo de `multiple_of`

    EL REDONDEO HACIA ARRIBA, SIN USAR math.ceil
        La forma idiomatica con enteros:

            multiple_of * ((hidden + multiple_of - 1) // multiple_of)

        Sumar (multiple_of - 1) antes de la division entera fuerza el redondeo hacia
        arriba. Si hidden ya es multiplo exacto, no lo cambia.

    POR QUE SE REDONDEA
        No es cosmetica. Las dimensiones alineadas a potencias de dos dejan que los tensor
        cores usen sus rutas rapidas. Una matriz de 853 columnas es notablemente mas lenta
        que una de 896, teniendo mas parametros.

    COMPROBACION
        d_model=320  ->  int(2*1280/3) = 853  ->  ceil_64(853) = 896     <- el config final
        d_model=128  ->  int(2*512/3)  = 341  ->  ceil_64(341) = 384     <- tiny_char

    Args:
        d_model: la dimension del modelo.
        multiple_of: al multiplo de cuanto redondear.
        ffn_dim_multiplier: factor extra opcional (Llama lo usa para ajustar a mano).

    Returns:
        `d_ff` como entero.
    """
    raise NotImplementedError("TODO: modulo 08, ejercicio 2 - swiglu_hidden_dim")


class SwiGLU(nn.Module):
    """El FFN con puerta que usa el modelo final. Dos tercios de sus parametros estan aqui.

    LA FORMULA

        SwiGLU(x) = down( Swish(gate(x)) * up(x) )

        donde `Swish(z) = z * sigmoid(z)`, que en PyTorch es `F.silu(z)`.

        El `*` es multiplicacion ELEMENTO A ELEMENTO, no matricial. Las dos ramas tienen
        la misma forma (B, T, d_ff) y se multiplican punto a punto.

    LA IDEA
        Hay dos proyecciones en paralelo desde la misma entrada. Una de ellas, tras pasar
        por Swish, actua como PUERTA: multiplica a la otra y decide cuanta senyal deja
        pasar por cada dimension.

        La diferencia con una activacion normal es que ese filtrado DEPENDE DE LA ENTRADA.
        Una activacion aplica la misma funcion a todo; una puerta decide, para cada
        dimension y cada token, cuanto pasa.

    UN DETALLE QUE CONVIENE TENER CLARO
        El FFN procesa cada token POR SEPARADO. No mezcla informacion entre posiciones,
        eso es trabajo de la atencion. Aqui no hace falta ninguna mascara ni nada
        parecido: cada token entra y sale por su cuenta.

    SUBMODULOS (respeta los nombres, los tests copian pesos por nombre)
        gate_proj: nn.Linear(d_model, d_ff, bias=bias)
        up_proj:   nn.Linear(d_model, d_ff, bias=bias)
        down_proj: nn.Linear(d_ff, d_model, bias=bias)
        dropout:   nn.Dropout(dropout)

    __init__(self, d_model, d_ff, dropout=0.0, bias=False)

    forward(self, x):
        Args:
            x: `(B, T, d_model)`.
        Returns:
            `(B, T, d_model)`.

        Es una sola linea:
            self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))

        Cuidado con cual de las dos ramas lleva la activacion. `F.silu` va sobre
        `gate_proj`, no sobre `up_proj`. Si las cambias, el modulo funciona igual de bien
        numericamente (es simetrico salvo por los pesos) pero NO coincidira con la
        referencia al copiar pesos, y el test fallara.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        raise NotImplementedError("TODO: modulo 08, ejercicio 3 - SwiGLU.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: modulo 08, ejercicio 3 - SwiGLU.forward")
