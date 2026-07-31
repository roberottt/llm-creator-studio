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
    """Mide cuantas operaciones por segundo hace tu GPU DE VERDAD.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Siete pasos. Ninguno tiene truco por separado; el orden es lo que importa.

        1. Si `cfg` es None, cogelo con `get_device()`.
           Si `dtype` es None, usa `cfg.amp_dtype`, y si eso tambien es None,
           `torch.float32`.

        2. Crea las dos matrices que vas a multiplicar:

               a = torch.randn(size, size, device=cfg.device, dtype=dtype)
               b = torch.randn(size, size, device=cfg.device, dtype=dtype)

        3. CALIENTA: repite `warmup` veces la operacion `a @ b`. NO cronometres esto.
           (Da igual que no guardes el resultado: PyTorch no elimina codigo muerto.)

        4. cfg.synchronize()          <- espera a que la GPU termine de verdad

        5. Arranca el cronometro y repite `iters` veces la multiplicacion:

               t0 = time.perf_counter()
               for _ in range(iters):
                   a @ b

        6. Sincroniza OTRA VEZ, y solo entonces para el cronometro:

               cfg.synchronize()
               segundos = time.perf_counter() - t0

        7. Devuelve los TFLOPS:

               return (2 * size**3 * iters) / segundos / 1e12

    POR QUÉ ESA FORMULA
    -------------------
    Multiplicar dos matrices de lado `size` produce `size**2` numeros, y cada uno sale de un
    producto escalar de longitud `size`: `size` multiplicaciones y `size-1` sumas, que por
    convencion se cuentan como `2 * size`. Total: `2 * size**3` operaciones.

    Dividiendo entre los segundos salen FLOPS, y entre `1e12` salen TeraFLOPS.

    CUIDADO CON LOS PASOS 3, 4 Y 6
    ------------------------------
    Son los que hacen que el numero signifique algo, y los tres son faciles de saltarse.

    **Sin calentar (paso 3)**, la primera multiplicacion de un tamanyo dado es entre 10 y 100
    veces mas lenta: la GPU esta eligiendo que kernel usar y reservando memoria. Con
    `iters=10` y sin calentamiento, esa primera llamada domina la media.

    **Sin sincronizar (pasos 4 y 6)**, `a @ b` en GPU no espera a nada: encola el trabajo y
    devuelve el control. Estarias midiendo lo que tarda la CPU en encolar una orden (unos 20
    microsegundos) y te saldrian miles de TFLOPS. Hay un test que acota el resultado justo
    para cazar esto.

    Args:
        cfg: el dispositivo. Si es `None`, se coge con `get_device()`.
        size: el lado de las matrices.
        dtype: el tipo de los tensores. Si es `None`, mira `cfg.amp_dtype`, y si tampoco hay,
            usa `torch.float32`.
        warmup: cuantas iteraciones de calentamiento (no se cronometran).
        iters: cuantas iteraciones cronometradas.

    Returns:
        Los TFLOPS efectivos, como float positivo.
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
    """Calcula cuantas operaciones cuesta procesar UN token.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Tres lineas de aritmetica. Sin bucles ni condiciones raras.

        1. Cuenta los parametros que participan en multiplicaciones de matrices:

               params = n_layers * (4 * d_model**2 + n_ffn_matrices * d_model * d_ff)
               params += d_model * vocab_size

        2. El coste del forward:

               forward = 2 * params + 4 * n_layers * context_length * d_model

        3. Devuelve, multiplicando por 3 si hay backward:

               return int(3 * forward if include_backward else forward)

    DE DÓNDE SALE CADA TÉRMINO
    --------------------------
    - `4 * d_model**2` son las cuatro proyecciones de la atencion (Wq, Wk, Wv, Wo), cada una
      una matriz `d_model x d_model` sin sesgo.
    - `n_ffn_matrices * d_model * d_ff` es el FFN: 3 matrices con SwiGLU (gate, up, down), 2
      con un FFN clasico.
    - `d_model * vocab_size` es la proyeccion final a logits.
    - El `2 *` sale de que cada parametro participa en una multiplicacion y una suma.
    - `4 * n_layers * context_length * d_model` es la atencion en si (`Q @ K^T` y
      `softmax @ V`). Ese termino NO viene de parametros: crece con el CONTEXTO, no con el
      tamanyo del modelo.
    - El `3 *` es porque el backward cuesta el doble que el forward: hace dos
      multiplicaciones por cada una del forward, una para el gradiente respecto a la entrada
      y otra respecto a los pesos.

    DOS COSAS QUE SE OLVIDAN
    ------------------------
    1. La proyeccion final cuenta AUNQUE uses weight tying. Atar los pesos ahorra memoria, no
       calculo: el matmul se hace igual.
    2. NO dividas por dos aunque la mascara causal solo calcule medio triangulo. Es la
       convencion de nanoGPT y de los papers; si divides, tu MFU no sera comparable con la de
       nadie.

    COMPRUÉBALO
    -----------
    Con la config del modelo final (6 capas, d_model 320, d_ff 896, contexto 512, vocab 4096)
    tiene que dar exactamente **65.372.160**.

    Args:
        n_layers: numero de capas.
        d_model: dimension del modelo.
        d_ff: dimension interna del FFN.
        context_length: longitud de la ventana de contexto.
        vocab_size: tamanyo del vocabulario.
        n_ffn_matrices: 3 para SwiGLU, 2 para un FFN clasico.
        include_backward: si `True`, multiplica el total por 3.

    Returns:
        Los FLOPs por token, como entero.
    """
    raise NotImplementedError("TODO: modulo 01, ejercicio 2 - transformer_flops_per_token")


def estimate_tokens_per_second(tflops: float, flops_per_token: int, mfu: float = 0.4) -> float:
    """Estima cuantos tokens por segundo vas a procesar.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Dos lineas.

        1. Si `flops_per_token` no es positivo, lanza `ValueError`.

        2. Devuelve:

               return tflops * 1e12 * mfu / flops_per_token

    El `1e12` convierte TeraFLOPS en FLOPS.

    POR QUÉ EL `mfu`
    ----------------
    Nunca aprovechas el 100% de la potencia de una GPU. La MFU (Model FLOPs Utilization) es
    la fraccion que consigues de verdad, y hay que multiplicar por ella o la estimacion sale
    absurdamente optimista.

    Valores realistas:
        0,4 - 0,5   modelos de miles de millones, bien optimizados
        0,1 - 0,2   nuestro modelo de 9M

    POR QUÉ VALIDAR
    ---------------
    Una division por cero aqui produce `inf` en silencio y estimaciones sin sentido, que
    descubririas mucho mas tarde. Un `ValueError` con un mensaje claro cuesta una linea.

    Args:
        tflops: el pico medido, en TFLOPS (lo que devuelve el ejercicio 1).
        flops_per_token: lo que devuelve el ejercicio 2.
        mfu: la fraccion del pico que esperas alcanzar.

    Returns:
        Tokens por segundo, como float.

    Raises:
        ValueError: si `flops_per_token` no es positivo.
    """
    raise NotImplementedError("TODO: modulo 01, ejercicio 3 - estimate_tokens_per_second")
