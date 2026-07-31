"""Referencia del modulo 00: medir el hardware y estimar el techo de velocidad."""

from __future__ import annotations

import time

import torch

from llmfs.device import DeviceConfig, get_device


def matmul_flops(size: int) -> int:
    """FLOPs de multiplicar dos matrices cuadradas de lado `size`.

    Cada elemento de la salida es un producto escalar de longitud `size`: `size`
    multiplicaciones y `size - 1` sumas. Por convencion se cuenta como `2 * size` (la
    diferencia de una suma es irrelevante y asi el numero es comparable con el resto del
    mundo). Hay `size^2` elementos de salida.
    """
    return 2 * size**3


def measure_matmul_tflops(
    cfg: DeviceConfig | None = None,
    size: int = 2048,
    dtype: torch.dtype | None = None,
    warmup: int = 3,
    iters: int = 10,
) -> float:
    """Mide los TFLOPS efectivos de un matmul grande en este dispositivo.

    El calentamiento no es opcional: la primera llamada paga la seleccion de kernel de
    cuBLAS/Metal y la asignacion de memoria, y sale entre 10 y 100 veces mas lenta.
    Y hay que sincronizar antes de mirar el reloj, porque las llamadas a GPU son
    asincronas: sin `synchronize()` estarias midiendo el tiempo de encolar, no el de
    calcular.
    """
    cfg = cfg or get_device()
    if dtype is None:
        dtype = cfg.amp_dtype or torch.float32

    a = torch.randn(size, size, device=cfg.device, dtype=dtype)
    b = torch.randn(size, size, device=cfg.device, dtype=dtype)

    for _ in range(warmup):
        a @ b
    cfg.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    cfg.synchronize()
    elapsed = time.perf_counter() - start

    return (matmul_flops(size) * iters) / elapsed / 1e12


def transformer_flops_per_token(
    n_layers: int,
    d_model: int,
    d_ff: int,
    context_length: int,
    vocab_size: int,
    n_ffn_matrices: int = 3,
    include_backward: bool = True,
) -> int:
    """FLOPs por token de entrenamiento de un transformer decoder.

    Se cuenta con la aproximacion estandar (Kaplan 2020, apendice B):

    - Un matmul con `P` parametros cuesta `2P` FLOPs por token en el forward.
    - El backward cuesta aproximadamente el doble que el forward (una pasada para el
      gradiente respecto a la entrada y otra respecto a los pesos). Total: `3x` forward.

    De ahi sale el `6N` que se ve en todas partes: `2 * P * 3 = 6P`.

    El termino cuadratico de la atencion se cuenta aparte porque no viene de parametros:
    `Q K^T` cuesta `2 * T * d_model` por token y capa, y `softmax @ V` otro tanto, de
    donde `4 * n_layers * T * d_model` en el forward.

    Simplificacion consciente: no se divide por dos pese a que la mascara causal calcula
    solo la mitad de la matriz. Es la convencion que usan nanoGPT y los papers, asi que
    los MFU que calcules seran comparables con los suyos. Tambien se ignoran softmax,
    normalizaciones y activaciones, que son memory-bound y aportan poco al conteo.

    Args:
        n_ffn_matrices: 3 para SwiGLU (gate, up, down), 2 para un FFN clasico.

    Returns:
        FLOPs por token, entero.
    """
    params_atencion = 4 * d_model**2
    params_ffn = n_ffn_matrices * d_model * d_ff
    params_matmul = n_layers * (params_atencion + params_ffn)
    params_matmul += d_model * vocab_size  # la proyeccion final a logits

    forward = 2 * params_matmul + 4 * n_layers * context_length * d_model
    return int(3 * forward if include_backward else forward)


def estimate_tokens_per_second(tflops: float, flops_per_token: int, mfu: float = 0.4) -> float:
    """Tokens por segundo alcanzables, dado el pico medido y una MFU supuesta.

    `mfu` (Model FLOPs Utilization) es la fraccion del pico teorico que consigues de
    verdad. Valores realistas: 0,4-0,5 para modelos grandes bien optimizados; 0,1-0,2
    para un modelo de 9M, donde el lanzamiento de kernels y el ancho de banda de memoria
    pesan mas que el calculo.
    """
    if flops_per_token <= 0:
        raise ValueError("flops_per_token debe ser positivo")
    return tflops * 1e12 * mfu / flops_per_token
