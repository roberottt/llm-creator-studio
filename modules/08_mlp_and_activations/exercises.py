"""Modulo 08 - FFN, GELU y SwiGLU.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `THEORY.md` -> implementa -> `llmfs check 08` -> `llmfs hint 08 -e N`
-> `SOLUTION.md` tiene el codigo completo.

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

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Una linea, transcribiendo la formula tal cual:

        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
        ))

    Sin bucles ni ramas. Los dos errores posibles son teclear mal una constante
    (`sqrt(2/pi) ~= 0.7978`) o reagrupar la expresion de forma que cambie el orden de
    operaciones.

    VALORES PARA COMPROBAR
    ----------------------
        x = -3  ->  -0.0036     casi anulado
        x = -1  ->  -0.1588     parcialmente
        x =  0  ->   0.0000
        x =  1  ->   0.8412     casi entero
        x =  3  ->   2.9964     entero

    QUÉ HACE Y POR QUÉ HACE FALTA
    -----------------------------
    Es la NO-LINEALIDAD. Sin algo asi entre capa y capa, apilar capas no sirve de nada: dos
    multiplicaciones de matriz seguidas son una sola multiplicacion de matriz.

    GELU multiplica `x` por la probabilidad de que una normal estandar salga menor que `x`. En
    vez de decidir con un corte duro si dejar pasar el valor (como hace ReLU), lo atenua de
    forma gradual.

    POR QUÉ LA APROXIMACIÓN Y NO LA EXACTA
    --------------------------------------
    La definicion real usa `erf`, que era lento en las GPU de 2016. Hoy la diferencia es
    irrelevante, pero GPT-2 se entreno con la aproximacion y por compatibilidad se sigue
    usando.

    Tu resultado tiene que coincidir con `F.gelu(x, approximate="tanh")`, NO con `F.gelu(x)` a
    secas: son funciones distintas y el test compara contra la primera.

    LO IMPORTANTE NO ES LA FÓRMULA, ES LA DERIVADA
    ----------------------------------------------
        x       ReLU     dReLU/dx      GELU      dGELU/dx
        -3.0    0.0000   0.0000       -0.0036   -0.0119
        -1.0    0.0000   0.0000       -0.1588   -0.0833

    Con ReLU la derivada en toda la zona negativa es CERO EXACTO. Una neurona que acabe dando
    siempre valores negativos deja de recibir gradiente para siempre: esta muerta y no hay
    forma de resucitarla. GELU tiene derivada pequenya pero no nula, asi que puede volver.

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

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Tres lineas.

        1. Los dos tercios del 4x clasico:

               hidden = int(2 * (4 * d_model) / 3)

        2. El multiplicador opcional:

               if ffn_dim_multiplier is not None:
                   hidden = int(ffn_dim_multiplier * hidden)

        3. Redondea HACIA ARRIBA al siguiente multiplo de `multiple_of`:

               return multiple_of * ((hidden + multiple_of - 1) // multiple_of)

    EL REDONDEO DEL PASO 3, SIN `math.ceil`
    ---------------------------------------
    Sumar `multiple_of - 1` antes de la division entera fuerza el redondeo hacia arriba, y si
    el valor ya era multiplo exacto no lo cambia. Es el idioma estandar y evita meter floats
    donde no hacen falta.

    COMPRUÉBALO CON LOS DOS CASOS DEL CURSO
    ---------------------------------------
        d_model = 320:  int(2*1280/3) = 853  ->  64 * ((853+63)//64) = 64 * 14 = 896
        d_model = 128:  int(2*512/3)  = 341  ->  64 * ((341+63)//64) = 64 *  6 = 384

    El 896 es el `d_ff` del config del modelo final, y el 384 el del juguete.

    DE DÓNDE SALE EL 2/3
    --------------------
    Es lo unico conceptual del ejercicio.

        FFN clasico:  2 matrices x d x 4d           = 8d²
        SwiGLU:       3 matrices x d x (2/3 * 4d)   = 8d²      <- mismo presupuesto
        SwiGLU sin ajustar: 3 x d x 4d              = 12d²     <- un 50% mas

    SwiGLU tiene TRES matrices donde el FFN clasico tiene dos. Con el mismo `d_ff` costaria un
    50% mas, asi que se reduce el hidden a dos tercios para gastar lo mismo y poder comparar de
    forma justa.

    POR QUÉ SE REDONDEA
    -------------------
    No es cosmetica. Las dimensiones alineadas a potencias de dos dejan que los tensor cores
    usen sus rutas rapidas. Una matriz de 853 columnas es notablemente mas lenta que una de
    896, teniendo MENOS parametros.

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

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    **En `__init__`** (cuatro lineas ademas del `super()`). Los nombres importan:

        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.up_proj   = nn.Linear(d_model, d_ff, bias=bias)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout   = nn.Dropout(dropout)

    **En `forward`** (una linea):

        return self.dropout(self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        ))

    TRES DETALLES
    -------------
    **El `*` es multiplicacion ELEMENTO A ELEMENTO**, no matricial. Las dos ramas salen con la
    misma forma `(B, T, d_ff)` y se multiplican punto a punto. Si pusieras `@` las formas ni
    siquiera cuadrarian.

    **La activacion va en `gate_proj`, NO en `up_proj`.** Numericamente el modulo funcionaria
    igual de bien con la asignacion invertida —es simetrico salvo por que pesos aprenden que—
    pero NO coincidiria con la referencia al copiar pesos y el test fallaria con una diferencia
    dificil de interpretar. Hay un test dedicado a senyalarlo.

    **`F.silu` es Swish**, o sea `z * sigmoid(z)`. Puedes escribirlo a mano
    (`x * torch.sigmoid(x)`) y da lo mismo, pero `F.silu` tiene un kernel fusionado.

    QUÉ ESTÁ PASANDO
    ----------------
    Hay DOS proyecciones en paralelo desde la misma entrada. Una de ellas, tras pasar por
    Swish, actua como PUERTA: multiplica a la otra y decide cuanta senyal deja pasar por cada
    dimension.

    La diferencia con una activacion normal es que ese filtrado DEPENDE DE LA ENTRADA. Una
    activacion aplica la misma funcion a todo; una puerta decide, para cada dimension y cada
    token, cuanto pasa.

    UN APUNTE QUE CONVIENE TENER CLARO
    ----------------------------------
    El FFN procesa cada token POR SEPARADO. No mezcla informacion entre posiciones: eso es
    trabajo de la atencion. Aqui no hace falta ninguna mascara ni nada parecido.

    `bias=False` por defecto es la config del modelo final, y es lo que hace que el conteo de
    parametros de exactamente `3 * d_model * d_ff`.

    forward(x):
        Args:
            x: `(B, T, d_model)`.
        Returns:
            `(B, T, d_model)`.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        raise NotImplementedError("TODO: modulo 08, ejercicio 3 - SwiGLU.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: modulo 08, ejercicio 3 - SwiGLU.forward")
