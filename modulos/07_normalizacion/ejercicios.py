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

    LA FORMULA
        y = (x - media) / sqrt(varianza + eps) * weight + bias

    EL EJEMPLO DE TEORIA.md
        x = [2.0, 8.0, 4.0, 6.0]
        media = 5.0,  varianza = 5.0,  sqrt(5) = 2.236
        y = [-1.342, 1.342, -0.447, 0.447]

    SOBRE QUE EJE
        Sobre la ULTIMA dimension, que son las features de cada token. Cada token se
        normaliza por su cuenta, independientemente de los demas y del batch.

        Eso es lo que distingue LayerNorm de BatchNorm, y es la razon de que funcione igual
        con batch de 1 que de 1000 y de que no necesite estadisticas guardadas para
        inferencia.

        Usa `dim=-1` y `keepdim=True` en la media y la varianza. El `keepdim` mantiene la
        forma (..., 1) para que el broadcast reste correctamente; sin el, la resta se
        alinearia mal y en el mejor de los casos petaria.

    LA TRAMPA: varianza POBLACIONAL, no muestral
        `torch.var` divide por (n-1) por defecto (`unbiased=True`), que es la varianza
        muestral. LayerNorm usa la POBLACIONAL, que divide por n.

        Tienes que pasar `unbiased=False`. Si no, tu resultado se parecera mucho al de
        `F.layer_norm` pero no coincidira, y con d_model pequenyo la diferencia es
        perfectamente visible.

    LOS PARAMETROS OPCIONALES
        `weight` (gamma) y `bias` (beta) pueden ser None. Si lo son, no los apliques: la
        funcion devuelve la normalizacion pura. Sirve para poder compararla con
        `F.layer_norm(x, (d,))` sin argumentos afines.

    Args:
        x: `(..., d)`. Se normaliza la ultima dimension.
        weight: `(d,)` o None.
        bias: `(d,)` o None.
        eps: para no dividir por cero cuando todas las componentes son iguales.

    Returns:
        Del mismo tamanyo que `x`.
    """
    raise NotImplementedError("TODO: modulo 07, ejercicio 1 - layer_norm")


class RMSNorm(nn.Module):
    """LayerNorm sin la media y sin el sesgo. Lo que usan Llama, Mistral y nuestro modelo.

    LA FORMULA
        y = x / sqrt( media(x^2) + eps ) * weight

    QUE CAMBIA RESPECTO A LAYERNORM
        No se resta la media y no hay beta. Solo se reescala por la raiz del cuadrado
        medio (root mean square, de ahi el nombre).

        Zhang y Sennrich (2019) observaron que casi todo el beneficio de LayerNorm viene
        de REESCALAR, no de RECENTRAR. Quitando la media se ahorra una pasada por los
        datos y un tensor intermedio: entre un 7% y un 64% mas rapido, sin perdida de
        calidad medible.

    EL EJEMPLO DE TEORIA.md
        x = [2.0, 8.0, 4.0, 6.0]
        RMS = sqrt((4+64+16+36)/4) = sqrt(30) = 5.477
        y   = [0.365, 1.461, 0.730, 1.096]

    EL DETALLE DE PRECISION QUE IMPORTA
        Haz el calculo en float32 aunque la entrada venga en float16:

            def forward(self, x):
                return self._norm(x.float()).type_as(x) * self.weight

        Motivo: elevar al cuadrado una activacion grande puede desbordar el rango de fp16
        (que se acaba en 65504) y dar `inf`. Con x=300, x^2 son 90.000: ya no cabe.
        Normalizar en fp32 y volver al dtype original despues cuesta muy poco y evita
        NaN que aparecerian solo a veces y serian dificiles de reproducir.

        Aviso para que no te sorprenda: aunque hagas `.type_as(x)`, la salida acabara
        siendo fp32, porque despues multiplicas por `self.weight`, que es un parametro
        fp32, y PyTorch promociona. Es correcto y es lo que hace Llama. Hay un test que
        lo documenta.

    UNA PISTA DE IMPLEMENTACION
        `torch.rsqrt(z)` calcula 1/sqrt(z) de una vez, y es mas rapido y mas estable que
        dividir por `torch.sqrt(z)`.

    SUBMODULOS
        weight: nn.Parameter de forma `(dim,)`, inicializado a UNOS (no a ceros, no
            aleatorio). Al arrancar, la capa tiene que ser la normalizacion pura.

    __init__(self, dim, eps=1e-6)
    forward(self, x) -> mismo tamanyo que x
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
    """Pre-norm: `x + fn(norm(x))`.

    ESTO ES UNA LINEA DE CODIGO Y ES EL EJERCICIO MAS IMPORTANTE DEL MODULO.

    Compara las dos opciones:

        post-norm (paper de 2017):   norm(x + fn(x))
        pre-norm  (todo lo moderno): x + fn(norm(x))

    Parece que solo cambian los parentesis de sitio. Lo que cambia es por donde pasa el
    gradiente.

    EN PRE-NORM el camino de x a la salida incluye un `+x` sin nada por medio. Al derivar:

        d(salida)/dx = 1 + d(fn(norm(x)))/dx

    Ese 1 es una autopista: aunque el segundo termino sea diminuto, el gradiente llega
    intacto a las capas de abajo, capa tras capa.

    EN POST-NORM la normalizacion esta ENCIMA de la suma, asi que el gradiente la atraviesa
    en cada capa y se va reescalando. Con 6 capas se nota poco; con 40 hace falta un warmup
    cuidadoso para que el entrenamiento no explote.

    EL PRECIO DE PRE-NORM: como la corriente residual nunca se normaliza por el camino,
    llega a la salida con una escala que crece con la profundidad. Por eso los modelos
    pre-norm llevan SIEMPRE una normalizacion final antes de la capa de salida. En el
    modulo 10 la veras: se llama `norm_f`.

    Args:
        x: la entrada.
        fn: el bloque (atencion o FFN).
        norm: la capa de normalizacion.

    Returns:
        Del mismo tamanyo que `x`.
    """
    raise NotImplementedError("TODO: modulo 07, ejercicio 3 - prenorm_residual")
