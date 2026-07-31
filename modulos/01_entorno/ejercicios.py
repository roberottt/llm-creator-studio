"""Modulo 01 - Entorno y hardware.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa -> `llmfs check 01` -> si te atascas, `llmfs hint 01 -e N`
-> y si sigues atascado, `SOLUCION.md` tiene el codigo completo para copiar.

QUÉ VAS A CONSTRUIR
===================

Una calculadora de cuanto va a tardar tu entrenamiento. Tres funciones:

    measure_matmul_tflops        cuantas operaciones por segundo da tu GPU DE VERDAD
    transformer_flops_per_token  cuantas operaciones cuesta procesar un token
    estimate_tokens_per_second   dividiendo lo uno entre lo otro, la velocidad

Con eso puedes contestar "¿esto tarda dos horas o dos semanas?" antes de escribir el modelo.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **FLOP**: una operacion con numeros decimales (una suma o una multiplicacion). La unidad
  con la que se mide cuanto cuesta entrenar algo.
- **TFLOPS**: billones de FLOPs por segundo. Lo que da tu GPU.
- **MFU** (Model FLOPs Utilization): que fraccion del pico teorico aprovechas de verdad.
  Nadie llega a 1; con un modelo pequenyo, 0,1-0,2 ya es bueno.
- **forward / backward**: pasar los datos por la red, y calcular como ajustar los pesos.
  El backward cuesta aproximadamente el doble que el forward.

    llmfs demo 01     mide tu GPU y estima el tiempo de la tirada final
"""

from __future__ import annotations

import time

import torch

from llmfs.device import DeviceConfig, get_device


def measure_matmul_tflops(
    cfg: DeviceConfig | None = None,
    size: int = 2048,
    dtype: torch.dtype | None = None,
    warmup: int = 3,
    iters: int = 10,
) -> float:
    """Mide los TFLOPS efectivos de un matmul cuadrado en este dispositivo.

    Multiplica dos matrices `(size, size)` unas cuantas veces y divide el numero de FLOPs
    realizados entre el tiempo transcurrido.

    Formula:
        FLOPs de un matmul cuadrado = 2 * size^3
        TFLOPS = (FLOPs * iters) / segundos / 1e12

    Dos cosas que hay que hacer bien o el numero no significa nada:

    1. **Calentar antes de medir.** La primera llamada paga la seleccion de kernel de
       cuBLAS/Metal y la reserva de memoria. Sale entre 10 y 100 veces mas lenta.
       Ejecuta `warmup` iteraciones cuyo tiempo NO cuentas.
    2. **Sincronizar antes de mirar el reloj.** Las llamadas a GPU son asincronas: la CPU
       encola el trabajo y sigue. Sin sincronizar medirias el tiempo de encolar, que es
       de microsegundos, y te saldrian TFLOPS imposibles. Usa `cfg.synchronize()` justo
       antes de tomar cada marca de tiempo.

    Args:
        cfg: dispositivo. Si es `None`, usa `get_device()`.
        size: lado de las matrices.
        dtype: tipo de los tensores. Si es `None`, usa `cfg.amp_dtype` y si tampoco hay,
            `torch.float32`.
        warmup: iteraciones de calentamiento (no se cronometran).
        iters: iteraciones cronometradas.

    Returns:
        TFLOPS efectivos (float positivo).
    """
    raise NotImplementedError("TODO: modulo 01, ejercicio 1 - measure_matmul_tflops")


def transformer_flops_per_token(
    n_layers: int,
    d_model: int,
    d_ff: int,
    context_length: int,
    vocab_size: int,
    n_ffn_matrices: int = 3,
    include_backward: bool = True,
) -> int:
    """FLOPs por token de entrenar un transformer decoder.

    Formula (Kaplan 2020, apendice B):

        params_matmul = n_layers * (4 * d_model^2 + n_ffn_matrices * d_model * d_ff)
                        + d_model * vocab_size

        forward = 2 * params_matmul + 4 * n_layers * context_length * d_model

        total   = 3 * forward     (si include_backward, si no solo forward)

    De donde viene cada pieza:

    - `4 * d_model^2` son las cuatro proyecciones de la atencion (Wq, Wk, Wv, Wo), cada
      una `d_model x d_model` y sin sesgo.
    - `n_ffn_matrices * d_model * d_ff` es el FFN: 3 matrices con SwiGLU (gate, up, down),
      2 con un FFN clasico.
    - `d_model * vocab_size` es la proyeccion final a logits. Cuenta aunque los pesos esten
      atados a los embeddings: el matmul se hace igual.
    - `2 * params` porque cada parametro participa en una multiplicacion y una suma.
    - `4 * n_layers * T * d_model` es la atencion propiamente dicha: `Q K^T` cuesta
      `2 * T * d_model` por token y capa, y `softmax @ V` otro tanto. Este termino NO
      viene de parametros: crece con el contexto, no con el tamanyo del modelo.
    - `3 *` porque el backward cuesta el doble que el forward.

    NOTA: no dividas por dos aunque la mascara causal solo calcule media matriz. Es la
    convencion de nanoGPT y de los papers; respetala para que tu MFU sea comparable.

    Returns:
        FLOPs por token, como entero.
    """
    raise NotImplementedError("TODO: modulo 01, ejercicio 2 - transformer_flops_per_token")


def estimate_tokens_per_second(tflops: float, flops_per_token: int, mfu: float = 0.4) -> float:
    """Tokens por segundo alcanzables con un pico medido y una MFU supuesta.

    Formula:
        tokens/s = tflops * 1e12 * mfu / flops_per_token

    Args:
        tflops: pico medido, en TFLOPS (lo que devuelve el ejercicio 1).
        flops_per_token: lo que devuelve el ejercicio 2.
        mfu: fraccion del pico que esperas alcanzar. 0,4-0,5 para modelos grandes bien
            optimizados; 0,1-0,2 para un modelo de 9M.

    Raises:
        ValueError: si `flops_per_token` no es positivo.

    Returns:
        Tokens por segundo (float).
    """
    raise NotImplementedError("TODO: modulo 01, ejercicio 3 - estimate_tokens_per_second")
