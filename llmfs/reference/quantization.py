"""Referencia del modulo 17: cuantizacion int8."""

from __future__ import annotations

import torch


def quantize_int8_symmetric(
    weight: torch.Tensor, per_channel: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convierte una matriz de pesos a int8 con escala.

    LA IDEA. Un float32 ocupa 4 bytes y un int8 uno solo: el modelo ocupa la cuarta parte.
    El truco es guardar, junto a los enteros, una ESCALA que permite recuperar los valores
    aproximados.

    SIMETRICA significa que el rango se centra en cero: se mapea `[-max, +max]` a
    `[-127, +127]`. La alternativa (asimetrica) usa tambien un desplazamiento y aprovecha
    mejor el rango cuando los datos no estan centrados, pero es mas cara de aplicar. Los
    pesos de una red suelen estar bastante centrados, asi que la simetrica va bien.

        escala = max(|W|) / 127
        W_int8 = round(W / escala)  acotado a [-127, 127]

    POR QUE 127 Y NO 128. int8 llega de -128 a 127. Usando 127 el rango queda simetrico y
    el cero se representa exactamente, que importa mas de lo que parece: en una matriz con
    muchos valores pequenyos, que el cero sea exacto evita un sesgo sistematico.

    POR CANAL FRENTE A POR TENSOR. Con `per_channel=True` se calcula una escala por FILA en
    vez de una para toda la matriz. Cuesta un vector de escalas mas (despreciable) y reduce
    bastante el error, porque una sola fila con valores grandes no arrastra a las demas.
    Es lo que hacen todas las implementaciones serias.

    Returns:
        `(cuantizado, escalas)` con `cuantizado` de dtype `int8` y `escalas` de forma
        `(filas, 1)` si es por canal, o un escalar si no.
    """
    if per_channel and weight.dim() >= 2:
        max_abs = weight.abs().amax(dim=-1, keepdim=True)
    else:
        max_abs = weight.abs().amax()

    # clamp_min evita dividir por cero si una fila es todo ceros.
    escala = (max_abs / 127.0).clamp_min(1e-12)
    cuantizado = torch.round(weight / escala).clamp(-127, 127).to(torch.int8)
    return cuantizado, escala


def dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Vuelve a float multiplicando por la escala.

    El resultado NO es igual al original: se ha perdido informacion. Lo que se recupera es
    el valor aproximado, con un error que depende de cuanto se haya tenido que redondear.
    """
    return quantized.to(torch.float32) * scale


def quantization_error(original: torch.Tensor, per_channel: bool = True) -> dict[str, float]:
    """Mide cuanto danya la cuantizacion.

    Tres numeros:

    - `error_relativo`: la norma del error dividida por la norma del original. Es la
      metrica que conviene mirar: independiente de la escala de los datos.
    - `error_maximo`: el peor caso.
    - `compresion`: cuantas veces mas pequenyo es (4x pasando de fp32 a int8).

    Con pesos de una red entrenada, el error relativo de int8 por canal ronda el 0,5-1%.
    Que eso apenas afecte a la calidad del modelo es un hecho empirico, no un teorema.
    """
    q, escala = quantize_int8_symmetric(original, per_channel=per_channel)
    recuperado = dequantize_int8(q, escala)

    error = (original - recuperado).abs()
    norma_original = float(original.norm())

    return {
        "error_relativo": float(error.norm()) / max(norma_original, 1e-12),
        "error_maximo": float(error.max()),
        "error_medio": float(error.mean()),
        "compresion": original.element_size() / q.element_size(),
        "bytes_original": original.numel() * original.element_size(),
        "bytes_cuantizado": q.numel() * q.element_size() + escala.numel() * escala.element_size(),
    }
