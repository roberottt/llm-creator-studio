"""Modulo 17 - Extras y limites honestos.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa -> `llmfs check 17` -> `llmfs hint 17 -e N`
-> `SOLUCION.md` tiene el codigo completo.

Son los ultimos tres ejercicios del curso, y son cortos.

QUÉ VAS A CONSTRUIR
===================

    quantize_int8_symmetric  (ej. 1)  guardar los pesos en 1 byte en vez de 4
    dequantize_int8          (ej. 2)  recuperarlos (aproximadamente)
    quantization_error       (ej. 3)  medir cuanto se ha perdido

Con eso el modelo pasa de 35,7 MB a 9,0 MB.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **cuantizar**: guardar los pesos con menos bits. De float32 (4 bytes) a int8 (1 byte).
- **escala**: el numero por el que hay que multiplicar los enteros para recuperar los
  valores originales. Se guarda junto a ellos.
- **simetrica / asimetrica**: si el rango se centra en cero o si ademas lleva un
  desplazamiento.
- **por canal / por tensor**: una escala por fila de la matriz, o una sola para toda.
- **error relativo**: el error dividido por la magnitud del original. Es la metrica que
  conviene mirar, porque no depende de la escala de los datos.

    llmfs demo 17     cuantiza tu modelo, mide el danyo, y cierra el curso
"""

from __future__ import annotations

import torch


def quantize_int8_symmetric(
    weight: torch.Tensor, per_channel: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convierte una matriz de pesos a int8 con escala.

    LA IDEA
        Un float32 ocupa 4 bytes y un int8 uno solo: el modelo ocupa la cuarta parte. El
        truco es guardar, junto a los enteros, una ESCALA que permite recuperar los valores
        aproximados.

    CON NUMEROS
        W = [0.12, -0.45, 0.03, 0.28]

        El mayor en valor absoluto es 0.45. Se mapea ese rango a [-127, +127]:

            escala = 0.45 / 127 = 0.003543
            W_int8 = round(W / escala) = [34, -127, 8, 79]

        Y para recuperar: W' = W_int8 * escala = [0.1204, -0.4500, 0.0283, 0.2799].
        No es exacto: el error es del orden de media unidad de escala.

    SIMETRICA significa que el rango se centra en cero. La alternativa (asimetrica) usa
    tambien un desplazamiento y aprovecha mejor el rango cuando los datos no estan
    centrados, pero es mas cara de aplicar. Los pesos de una red suelen estar bastante
    centrados.

    POR QUE 127 Y NO 128
        int8 va de -128 a 127. Usando 127 el rango queda SIMETRICO y el cero se representa
        exactamente. Eso importa mas de lo que parece: en una matriz con muchos valores
        pequenyos, que el cero sea exacto evita un sesgo sistematico que se acumularia capa
        tras capa.

    POR CANAL FRENTE A POR TENSOR
        Con `per_channel=True` se calcula una escala por FILA (`dim=-1, keepdim=True`) en
        vez de una para toda la matriz. Cuesta un vector de escalas mas (despreciable) y
        reduce bastante el error, porque una sola fila con valores grandes no arrastra a
        las demas.

        Medido sobre una matriz real del modelo: 0.71% de error por canal frente a 1.07%
        por tensor.

    LOS PASOS
        1. `max_abs = weight.abs().amax(dim=-1, keepdim=True)` si es por canal,
           o `weight.abs().amax()` si no.
        2. `escala = (max_abs / 127.0).clamp_min(1e-12)`
           El `clamp_min` evita dividir por cero si una fila es todo ceros.
        3. `cuantizado = torch.round(weight / escala).clamp(-127, 127).to(torch.int8)`
           El `clamp` protege del redondeo en el borde: sin el, un valor justo en el maximo
           podria dar 128, que no cabe en int8 y haria wrap a -128.

    Returns:
        `(cuantizado, escala)` con `cuantizado` de dtype `int8`.
    """
    raise NotImplementedError("TODO: modulo 17, ejercicio 1 - quantize_int8_symmetric")


def dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Vuelve a float multiplicando por la escala.

    Una linea:  `quantized.to(torch.float32) * scale`

    El `.to(torch.float32)` es obligatorio ANTES de multiplicar: si multiplicaras el int8
    directamente, PyTorch haria la operacion en enteros y el resultado seria basura.

    El resultado NO es igual al original: se ha perdido informacion. Lo que se recupera es
    el valor aproximado, con un error que depende de cuanto se haya tenido que redondear.
    """
    raise NotImplementedError("TODO: modulo 17, ejercicio 2 - dequantize_int8")


def quantization_error(original: torch.Tensor, per_channel: bool = True) -> dict[str, float]:
    """Mide cuanto danya la cuantizacion.

    QUE HACE
        Cuantiza, descuantiza, y compara con el original. Es la forma de saber si merece la
        pena antes de aplicarlo al modelo entero.

    LAS METRICAS

        error_relativo   = ||original - recuperado|| / ||original||

            La que conviene mirar: es independiente de la escala de los datos, asi que
            puedes comparar capas distintas. Con pesos de una red entrenada, int8 por canal
            ronda el 0.5-1%.

        error_maximo     = max(|original - recuperado|)
        error_medio      = mean(|original - recuperado|)
        compresion       = bytes por elemento original / bytes por elemento cuantizado
        bytes_original   = original.numel() * original.element_size()
        bytes_cuantizado = los del int8 MAS los de las escalas

        Ese ultimo detalle importa: las escalas tambien ocupan. Con una por fila son
        despreciables, pero contarlas es lo honesto.

    PISTA
        `tensor.element_size()` da los bytes por elemento (4 para float32, 1 para int8).
        Con eso la compresion sale sola sin numeros magicos.

    Returns:
        Un dict con `error_relativo`, `error_maximo`, `error_medio`, `compresion`,
        `bytes_original` y `bytes_cuantizado`.
    """
    raise NotImplementedError("TODO: modulo 17, ejercicio 3 - quantization_error")
