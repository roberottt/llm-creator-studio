"""Modulo 07 - Normalizacion y conexiones residuales.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa -> `llmfs check 07` -> `llmfs hint 07 -e N`
-> `SOLUCION.md` tiene el codigo completo.

QUÉ VAS A CONSTRUIR
===================

Las dos piezas que hacen que una red profunda entrene en vez de devolver NaN:

    layer_norm         (ej. 1)  centrar en 0 y escalar a varianza 1
    RMSNorm            (ej. 2)  lo mismo pero sin la media (lo que usa Llama)
    prenorm_residual   (ej. 3)  UNA LINEA, y es la mas importante del modulo

El tercero apenas tiene codigo. Lo que importa es entender POR QUE los parentesis van donde
van.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **normalizar**: reescalar unos numeros para que tengan una media y una dispersion
  conocidas. Aqui, media 0 y varianza 1.
- **varianza**: cuanto se dispersan los valores respecto a su media.
- **conexion residual**: sumar la entrada de un bloque a su salida (`x + f(x)`). Es lo que
  permite entrenar redes profundas.
- **corriente residual** (residual stream): esa suma acumulada que atraviesa toda la red.
  Cada capa le anyade su contribucion.
- **gradiente que se desvanece**: cuando el gradiente se hace tan pequenyo al atravesar
  capas que las primeras dejan de recibir senyal y no aprenden.
- **pre-norm / post-norm**: si la normalizacion va dentro de la rama (`x + f(norm(x))`) o
  envolviendo la suma (`norm(x + f(x))`).

    llmfs demo 07     mide cuanto gradiente llega a la primera capa en cada configuracion
"""

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
    """Normaliza cada token a media 0 y varianza 1, y luego aplica escala y desplazamiento.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Cinco lineas.

        1. La media y la varianza, sobre la ULTIMA dimension:

               mean = x.mean(dim=-1, keepdim=True)
               var = x.var(dim=-1, keepdim=True, unbiased=False)

        2. Normaliza:

               normalized = (x - mean) / torch.sqrt(var + eps)

        3. Aplica los parametros SI existen:

               if weight is not None:
                   normalized = normalized * weight
               if bias is not None:
                   normalized = normalized + bias

        4. `return normalized`

    EJEMPLO PARA COMPROBAR
    ----------------------
        x = [2.0, 8.0, 4.0, 6.0]

        media    = (2+8+4+6)/4 = 5.0
        varianza = ((2-5)² + (8-5)² + (4-5)² + (6-5)²)/4 = 20/4 = 5.0
        sqrt(5)  = 2.236

        y = [(2-5)/2.236, (8-5)/2.236, (4-5)/2.236, (6-5)/2.236]
          = [-1.342, 1.342, -0.447, 0.447]

    LA TRAMPA: `unbiased=False`
    ---------------------------
    `torch.var` divide por (n-1) por defecto (la varianza muestral, con la correccion de
    Bessel). LayerNorm usa la POBLACIONAL, que divide por n.

    Sin el `unbiased=False`, tu resultado se parecera mucho a `F.layer_norm` pero no
    coincidira. Con d=320 la diferencia es del 0,3% y podrias no verla; con d=4 es del 33%. El
    test compara tu resultado contra las dos versiones y te dice a cual te pareces.

    EL `keepdim=True` TAMPOCO ES OPCIONAL
    -------------------------------------
    Sin el, `mean(dim=-1)` sobre `(4, 8, 32)` devuelve `(4, 8)` en vez de `(4, 8, 1)`, y la
    resta `x - mean` intenta emitir mal las dimensiones. A veces lanza error y a veces —cuando
    las formas casualmente encajan— produce basura en silencio.

    EL eps VA DENTRO DE LA RAÍZ
    ---------------------------
    `sqrt(var + eps)`, no `sqrt(var) + eps`. Es lo que hace `F.layer_norm`, y con varianza
    pequenya la diferencia importa. Sirve para no dividir por cero cuando todas las componentes
    son iguales.

    POR QUÉ SOBRE LA ÚLTIMA DIMENSIÓN
    ---------------------------------
    Son las features de cada token. Cada token se normaliza por su cuenta, independientemente
    de los demas y del batch.

    Eso es lo que distingue LayerNorm de BatchNorm, y la razon de que funcione igual con batch
    de 1 que de 1000 y de que no necesite estadisticas guardadas para inferencia.

    Args:
        x: `(..., d)`. Se normaliza la ultima dimension.
        weight: `(d,)` o None.
        bias: `(d,)` o None.
        eps: para no dividir por cero.

    Returns:
        Del mismo tamanyo que `x`.
    """
    raise NotImplementedError("TODO: modulo 07, ejercicio 1 - layer_norm")


class RMSNorm(nn.Module):
    """LayerNorm sin la media y sin el sesgo. Lo que usan Llama, Mistral y nuestro modelo.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    **En `__init__`** (dos lineas ademas del `super()`):

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    **Un ayudante** (o inlinealo, como prefieras):

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    **En `forward`** (una linea):

        return self._norm(x.float()).type_as(x) * self.weight

    EJEMPLO PARA COMPROBAR
    ----------------------
        x = [2.0, 8.0, 4.0, 6.0]

        RMS = sqrt((4+64+16+36)/4) = sqrt(30) = 5.477
        y   = [2/5.477, 8/5.477, 4/5.477, 6/5.477]
            = [0.365, 1.461, 0.730, 1.096]

    `torch.ones` Y NO `torch.randn`
    -------------------------------
    Al inicializar, la capa tiene que ser la normalizacion pura. Si `weight` empezara
    aleatorio, estarias escalando cada dimension por un factor arbitrario antes de haber
    aprendido nada, y la perdida del paso 0 no cuadraria con `ln(V)`.

    EL `.float()` NO ES PARANOIA
    ----------------------------
    Con activaciones en fp16, elevar al cuadrado desborda antes de lo que uno espera:
    `300**2 = 90.000` y fp16 se acaba en 65.504.

    El resultado seria `inf`, luego la media seria `inf`, y `rsqrt(inf)` seria 0: la capa
    devolveria CEROS. Hay un test que reproduce ese caso exacto.

    UN DETALLE QUE SORPRENDE
    ------------------------
    Aunque hagas `.type_as(x)` para volver a fp16, la salida acaba siendo **fp32**, porque
    despues multiplicas por `self.weight`, que es un parametro fp32, y PyTorch promociona.

    No es un bug: es lo que hace la implementacion de Llama y es lo deseable. Bajo autocast los
    pesos se mantienen en fp32 y las operaciones siguientes convierten lo que necesiten. Hay un
    test que lo documenta.

    `torch.rsqrt(z)` calcula `1/sqrt(z)` de una vez: es un kernel menos que dividir y algo mas
    estable.

    QUÉ CAMBIA RESPECTO A LAYERNORM
    -------------------------------
    No se resta la media y no hay beta. Solo se reescala por la raiz del cuadrado medio (root
    mean square, de ahi el nombre).

    Zhang y Sennrich (2019) observaron que casi todo el beneficio de LayerNorm viene de
    REESCALAR, no de RECENTRAR. Quitandolo se ahorra una pasada por los datos y un tensor
    intermedio, sin perdida de calidad medible.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        raise NotImplementedError("TODO: modulo 07, ejercicio 2 - RMSNorm.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: modulo 07, ejercicio 2 - RMSNorm.forward")


def prenorm_residual(
    x: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    norm: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Pre-norm. Es UNA LINEA y es el ejercicio mas importante del modulo.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
        return x + fn(norm(x))

    Eso es todo.

    LO QUE TIENES QUE ENTENDER
    --------------------------
    Hay dos formas de colocar la normalizacion, y solo cambian los parentesis de sitio:

        post-norm (el paper de 2017):   norm(x + fn(x))
        pre-norm  (todo lo moderno):    x + fn(norm(x))

    Lo que cambia es POR DONDE PASA EL GRADIENTE.

    En **pre-norm**, el camino de `x` a la salida incluye un `+x` sin nada por medio. Al
    derivar:

        d(salida)/dx = 1 + d(fn(norm(x)))/dx

    Ese **1** es una autopista: aunque el segundo termino sea diminuto, el gradiente llega
    intacto a las capas de abajo, capa tras capa.

    En **post-norm**, la normalizacion esta ENCIMA de la suma, asi que el gradiente la
    atraviesa en cada capa y se va reescalando. Con 6 capas se nota poco; con 40 hace falta un
    warmup cuidadoso para que el entrenamiento no explote.

    SI TE SALE MAL
    --------------
    Si escribes `norm(x + fn(x))` has hecho post-norm, y hay un test que lo detecta.

    CÓMO SE COMPRUEBA QUE ESTÁ BIEN
    -------------------------------
    Hay un test que anula por completo el gradiente de la rama (`fn` devuelve un tensor
    detached por cero) y verifica que el gradiente en la entrada sigue siendo EXACTAMENTE 1.
    Ese 1 es toda la razon de ser del residual.

    CONSECUENCIA PARA EL MÓDULO 10
    ------------------------------
    Como la corriente residual nunca se normaliza por el camino, llega a la salida con una
    escala que crece con la profundidad. Por eso los modelos pre-norm llevan SIEMPRE una
    normalizacion final antes de la capa de logits. Se llamara `norm_f`.

    Args:
        x: la entrada.
        fn: el bloque (atencion o FFN).
        norm: la capa de normalizacion.

    Returns:
        Del mismo tamanyo que `x`.
    """
    raise NotImplementedError("TODO: modulo 07, ejercicio 3 - prenorm_residual")
