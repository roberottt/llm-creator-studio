"""Referencia del modulo 12: eficiencia y leyes de escala."""

from __future__ import annotations

from llmfs.config import ModelConfig


def model_flops_per_token(cfg: ModelConfig, include_backward: bool = True) -> dict[str, int]:
    """FLOPs por token, separados en matmuls y atencion.

    Es el mismo calculo del modulo 01, pero devolviendo el desglose en vez de un unico
    numero. Saber cuanto pesa cada parte es lo que permite decidir si conviene tocar el
    contexto o el tamanyo del modelo.

    Returns:
        Un dict con `matmul`, `attention`, `total` y `params_matmul`.
    """
    d, ff, v = cfg.d_model, cfg.d_ff, cfg.vocab_size
    n_ffn = 3 if cfg.activation == "swiglu" else 2

    params_matmul = cfg.n_layers * (4 * d * d + n_ffn * d * ff) + d * v

    matmul = 2 * params_matmul
    attention = 4 * cfg.n_layers * cfg.context_length * d

    factor = 3 if include_backward else 1
    return {
        "matmul": matmul * factor,
        "attention": attention * factor,
        "total": (matmul + attention) * factor,
        "params_matmul": params_matmul,
    }


def compute_mfu(
    tokens_per_second: float, flops_per_token: int, peak_tflops: float
) -> float:
    """Model FLOPs Utilization: que fraccion del pico del hardware estas aprovechando.

    $$\\text{MFU} = \\frac{\\text{tokens/s} \\times C_{\\text{token}}}{\\text{FLOPS pico}}$$

    Es LA metrica para saber si tu entrenamiento va bien optimizado, y es independiente del
    modelo y del hardware, asi que se puede comparar entre configuraciones.

    Referencias de lo que es razonable:
        0.4 - 0.5   modelos grandes bien optimizados en A100/H100
        0.2 - 0.3   modelos medianos
        0.1 - 0.2   nuestro modelo de 9M
        < 0.05      algo va mal: mira el dataloader o el tamanyo de batch

    Con un modelo pequenyo la MFU baja es esperable: las matrices de 320x320 no dan para
    saturar los tensor cores, y el tiempo se va en lanzar kernels y en mover memoria.
    """
    if peak_tflops <= 0:
        raise ValueError("peak_tflops tiene que ser positivo")
    return tokens_per_second * flops_per_token / (peak_tflops * 1e12)


def chinchilla_optimal_allocation(
    compute_budget: float, tokens_per_param: float = 20.0
) -> dict[str, float]:
    """Reparte un presupuesto de computo entre parametros y tokens, segun Chinchilla.

    EL PROBLEMA. Tienes un presupuesto fijo de FLOPs. Puedes gastarlo en un modelo grande
    con pocos datos, o en uno pequenyo con muchos. ¿Cual da menos perdida?

    LA RESPUESTA (Hoffmann et al. 2022). Los dos deben crecer PROPORCIONALMENTE. Su
    resultado empirico: unos **20 tokens por parametro**.

    Fue un resultado importante porque contradecia la practica del momento. GPT-3 tenia
    175.000 millones de parametros entrenados con 300.000 millones de tokens: 1,7 tokens
    por parametro, casi doce veces por debajo del optimo. Chinchilla (70.000 millones de
    parametros, 1,4 billones de tokens) lo batio en casi todos los benchmarks con menos de
    la mitad de parametros.

    LA ARITMETICA. Con `C = 6ND`:

        C = 6 * N * (20N) = 120 N^2      ->     N = sqrt(C / 120)
        D = 20 * N

    NUESTRO CASO. El modelo tiene 7,62M de parametros no-embedding y va a ver 500M de
    tokens: 65 tokens por parametro, mas de tres veces por encima de lo "optimo". Es
    deliberado, y hay dos razones:

    1. Chinchilla optimiza el computo de ENTRENAMIENTO. Si el modelo se va a usar mucho
       despues, conviene un modelo mas pequenyo y mas entrenado: la inferencia se paga cada
       vez. Llama-3 lleva esto al extremo con ~1.800 tokens por parametro.
    2. A esta escala, entrenar de mas es barato (horas) y da un modelo notablemente mejor.

    Returns:
        `{"params", "tokens", "tokens_per_param", "compute"}`.
    """
    if compute_budget <= 0:
        raise ValueError("el presupuesto de computo tiene que ser positivo")

    params = (compute_budget / (6 * tokens_per_param)) ** 0.5
    tokens = tokens_per_param * params

    return {
        "params": params,
        "tokens": tokens,
        "tokens_per_param": tokens_per_param,
        "compute": compute_budget,
    }
