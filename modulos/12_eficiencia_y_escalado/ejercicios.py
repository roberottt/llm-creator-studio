"""Modulo 12 - Eficiencia y leyes de escala.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa -> `llmfs check 12` -> `llmfs hint 12 -e N`
-> `SOLUCION.md` tiene el codigo completo.

Ninguna de las tres funciones tiene mas de cinco lineas de codigo. La dificultad esta en
entender que significan los numeros.

QUÉ VAS A CONSTRUIR
===================

    model_flops_per_token          (ej. 1)  cuanto cuesta un token, desglosado
    compute_mfu                    (ej. 2)  que fraccion de tu GPU aprovechas
    chinchilla_optimal_allocation  (ej. 3)  como repartir el presupuesto de computo

El ejercicio 3 reproduce un resultado que en 2022 demostro que la industria entera estaba
entrenando mal sus modelos. Y lo vas a verificar contra modelos historicos reales.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **MFU** (Model FLOPs Utilization): tokens/s x FLOPs por token, dividido por el pico de tu
  hardware. Nadie llega a 1.
- **presupuesto de computo**: cuantos FLOPs totales te puedes permitir gastar entrenando.
- **Chinchilla**: el paper de 2022 que midio como repartir ese presupuesto entre tamanyo de
  modelo y cantidad de datos. Respuesta: ~20 tokens por parametro.
- **parametros no-embedding**: el total menos la tabla de embeddings. Es lo que usan las
  leyes de escala, porque los embeddings escalan distinto.
- **sobreentrenado / infraentrenado**: por encima o por debajo de esos 20 tokens por
  parametro. Ninguna de las dos cosas es necesariamente mala: depende de tu objetivo.

    llmfs demo 12     mide tu MFU real y reproduce el resultado de Chinchilla
"""

from __future__ import annotations

from llmfs.config import ModelConfig


def model_flops_per_token(cfg: ModelConfig, include_backward: bool = True) -> dict[str, int]:
    """FLOPs por token, separados en matmuls y atencion.

    QUE ES ESTO
        El mismo calculo del modulo 01, pero devolviendo el DESGLOSE en vez de un unico
        numero. Saber cuanto pesa cada parte es lo que permite decidir si conviene tocar
        el contexto o el tamanyo del modelo.

    LOS TERMINOS

        params_matmul = n_layers * (4*d^2 + n_ffn*d*d_ff) + d*vocab_size

            donde n_ffn = 3 con SwiGLU y 2 con un FFN clasico.
            Ojo: la proyeccion final `d*vocab_size` cuenta AUNQUE haya weight tying. Atar
            los pesos ahorra memoria, no calculo: el matmul se hace igual.

        matmul    = 2 * params_matmul
        attention = 4 * n_layers * context_length * d_model

        Si include_backward, MULTIPLICA AMBOS por 3 (el backward cuesta el doble que el
        forward).

    POR QUE SEPARARLOS
        El termino de atencion crece con el CONTEXTO y el de matmul con el TAMANYO del
        modelo. Con nuestra config (T=512) la atencion es un 18% del total; con T=4096
        seria el 64%. Ese desglose te dice al instante si alargar el contexto te va a salir
        caro.

    Returns:
        Un dict con las claves `matmul`, `attention`, `total` y `params_matmul`.
        `total` = matmul + attention.
    """
    raise NotImplementedError("TODO: modulo 12, ejercicio 1 - model_flops_per_token")


def compute_mfu(tokens_per_second: float, flops_per_token: int, peak_tflops: float) -> float:
    """Model FLOPs Utilization: que fraccion del pico del hardware estas aprovechando.

    LA FORMULA

        MFU = tokens_per_second * flops_per_token / (peak_tflops * 1e12)

    El `1e12` convierte TeraFLOPS en FLOPS. Es una linea.

    COMO SE INTERPRETA

        0.4 - 0.5   modelos grandes bien optimizados en A100/H100
        0.2 - 0.3   modelos medianos
        0.1 - 0.2   nuestro modelo de 9M
        < 0.05      algo va mal: mira el dataloader o sube el batch size

    NADIE LLEGA A 1. El pico teorico solo se alcanza con matmuls enormes perfectamente
    alineados y nada mas de por medio.

    Con un modelo pequenyo la MFU baja es inevitable: las matrices de 320x320 no dan para
    saturar los tensor cores. No es culpa tuya.

    PARA QUE SIRVE DE VERDAD
        No por su valor absoluto, sino porque es COMPARABLE: no depende del modelo ni del
        hardware. Cambias el batch size, activas torch.compile, mueves el dataloader a otro
        hilo, y miras si el numero sube.

    Raises:
        ValueError: si `peak_tflops` no es positivo (division por cero silenciosa).
    """
    raise NotImplementedError("TODO: modulo 12, ejercicio 2 - compute_mfu")


def chinchilla_optimal_allocation(
    compute_budget: float, tokens_per_param: float = 20.0
) -> dict[str, float]:
    """Reparte un presupuesto de computo entre parametros y tokens.

    EL PROBLEMA
        Tienes un presupuesto fijo de FLOPs. Puedes gastarlo en un modelo grande con pocos
        datos o en uno pequenyo con muchos. ¿Cual da menos perdida?

    LA RESPUESTA (Hoffmann et al. 2022, "Chinchilla")
        Los dos deben crecer PROPORCIONALMENTE: unos 20 tokens por parametro.

        Fue un resultado importante porque contradecia la practica del momento. GPT-3 tenia
        175.000 millones de parametros y 300.000 millones de tokens: 1,7 tokens por
        parametro, doce veces por debajo del optimo.

    LA ARITMETICA
        Partiendo de C = 6ND (modulo 01) y de D = k*N con k = tokens_per_param:

            C = 6 * N * (k*N) = 6k * N^2

            N = sqrt( C / (6*k) )
            D = k * N

    COMPRUEBALO CON EL PROPIO CHINCHILLA
        Su presupuesto fue 5.76e23 FLOPs. Con k=20:

            N = sqrt(5.76e23 / 120) = 6.9e10 = 69.000 millones

        El modelo real tenia 70.000 millones. La formula lo clava, y verlo funcionar con
        un caso historico da bastante mas confianza que leerla.

    Args:
        compute_budget: FLOPs disponibles. Tiene que ser positivo.
        tokens_per_param: la constante de Chinchilla. 20 por defecto.

    Returns:
        `{"params", "tokens", "tokens_per_param", "compute"}`, todos floats.

    Raises:
        ValueError: si el presupuesto no es positivo.
    """
    raise NotImplementedError("TODO: modulo 12, ejercicio 3 - chinchilla_optimal_allocation")
