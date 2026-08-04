"""Modulo 12 - Eficiencia y leyes de escala.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `THEORY.md` -> implementa -> `llmfs check 12` -> `llmfs hint 12 -e N`
-> `SOLUTION.md` tiene el codigo completo.

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

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Cuatro pasos, todo aritmetica con los campos del config.

        1. Cuantas matrices tiene el FFN:

               n_ffn = 3 if cfg.activation == "swiglu" else 2

        2. Los parametros que participan en multiplicaciones de matriz:

               params_matmul = cfg.n_layers * (
                   4 * cfg.d_model**2 + n_ffn * cfg.d_model * cfg.d_ff
               ) + cfg.d_model * cfg.vocab_size

        3. Los dos terminos:

               matmul = 2 * params_matmul
               attention = 4 * cfg.n_layers * cfg.context_length * cfg.d_model

        4. El backward, si toca, y el dict de salida:

               if include_backward:
                   matmul *= 3
                   attention *= 3

               return {
                   "matmul": matmul,
                   "attention": attention,
                   "total": matmul + attention,
                   "params_matmul": params_matmul,
               }

    DE DÓNDE SALE CADA CONSTANTE
    ----------------------------
    **El 2 de `matmul`.** Multiplicar una matriz por un vector hace, por cada peso, una
    multiplicacion y una suma: 2 operaciones por parametro. De ahi el `2 * params`.

    **El 3 del backward.** El paso hacia atras cuesta aproximadamente el DOBLE que el hacia
    delante (hay que calcular el gradiente respecto a la entrada y respecto a los pesos, dos
    matmuls donde el forward hacia uno). Forward + backward = 1 + 2 = 3.

    **El `d*vocab_size` cuenta AUNQUE haya weight tying.** Atar los pesos ahorra MEMORIA, no
    CALCULO: la multiplicacion final se hace igual, con la misma matriz.

    **El 4 de `attention`.** Son los dos matmuls que NO involucran parametros: `Q@K^T` y
    `attn@V`. Cada uno cuesta `2 * T * d` por token, y son dos: `4 * T * d` por capa.

    POR QUÉ SE DEVUELVEN SEPARADOS
    ------------------------------
    Porque los dos terminos crecen con cosas distintas:

        matmul     crece con el TAMANYO del modelo (d_model, n_layers, d_ff)
        attention  crece con el CONTEXTO (context_length)

    Con nuestra config (T=512) la atencion es un 18% del total. Con T=4096 seria el 64%. Ese
    desglose te dice al instante si alargar el contexto te va a salir caro, sin tener que
    probarlo.

    Args:
        cfg: la configuracion del modelo.
        include_backward: si incluir el coste del paso hacia atras (True para entrenamiento,
            False para inferencia pura).

    Returns:
        Un dict con `matmul`, `attention`, `total` y `params_matmul`.
        `total` = `matmul` + `attention`.
    """
    raise NotImplementedError("TODO: modulo 12, ejercicio 1 - model_flops_per_token")


def compute_mfu(tokens_per_second: float, flops_per_token: int, peak_tflops: float) -> float:
    """Model FLOPs Utilization: que fraccion del pico del hardware estas aprovechando.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Dos lineas.

        1. La validacion (si no, una division por cero silenciosa te da `inf` y no te enteras):

               if peak_tflops <= 0:
                   raise ValueError(f"peak_tflops tiene que ser positivo: {peak_tflops}")

        2. La formula:

               return tokens_per_second * flops_per_token / (peak_tflops * 1e12)

    El `1e12` convierte TeraFLOPS en FLOPS, para que arriba y abajo esten en las mismas
    unidades. Es el unico sitio donde se puede meter la pata.

    UN EJEMPLO CON NÚMEROS REALES
    -----------------------------
    Los de la tirada del `tiny_char` en el MacBook:

        tokens_per_second = 112.000
        flops_per_token   = 2.6e7        (del ejercicio 1)
        peak_tflops       = 14.2         (M5, fp32)

        MFU = 112000 * 2.6e7 / 1.42e13 = 0.205,  o sea el 20,5%

    CÓMO SE INTERPRETA
    ------------------
        0,4 - 0,5   modelos grandes bien optimizados en A100/H100
        0,2 - 0,3   modelos medianos
        0,1 - 0,2   nuestro modelo de 9M
        < 0,05      algo va mal: mira el dataloader o sube el batch size

    NADIE LLEGA A 1. El pico teorico solo se alcanza con matmuls enormes perfectamente
    alineados y absolutamente nada mas de por medio. Con un modelo pequenyo la MFU baja es
    inevitable: las matrices de 320x320 no dan para saturar los tensor cores, y una parte
    importante del tiempo se va en lanzar kernels en vez de en calcular. No es culpa tuya.

    PARA QUÉ SIRVE DE VERDAD
    ------------------------
    No por su valor absoluto, sino porque es COMPARABLE: no depende del modelo ni del hardware.
    "1200 tokens por segundo" no te dice nada; "18% de MFU" si. Cambias el batch size, activas
    `torch.compile`, mueves el dataloader a otro hilo, y miras si el numero sube. Es el
    termometro de las optimizaciones.

    Args:
        tokens_per_second: el rendimiento medido, no el teorico.
        flops_per_token: de `model_flops_per_token(...)["total"]`.
        peak_tflops: el pico del hardware, de `llmfs device`.

    Returns:
        Una fraccion entre 0 y 1.

    Raises:
        ValueError: si `peak_tflops` no es positivo.
    """
    raise NotImplementedError("TODO: modulo 12, ejercicio 2 - compute_mfu")


def chinchilla_optimal_allocation(
    compute_budget: float, tokens_per_param: float = 20.0
) -> dict[str, float]:
    """Reparte un presupuesto de computo entre parametros y tokens.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Tres lineas y un dict.

        1. La validacion:

               if compute_budget <= 0:
                   raise ValueError(f"el presupuesto tiene que ser positivo: {compute_budget}")

        2. Despeja N y saca D:

               params = (compute_budget / (6 * tokens_per_param)) ** 0.5
               tokens = tokens_per_param * params

           (`** 0.5` es la raiz cuadrada. Se escribe asi para no tener que importar `math`
           en este fichero, que no lo importa.)

        3. Devuelve las cuatro cosas:

               return {
                   "params": params,
                   "tokens": tokens,
                   "tokens_per_param": tokens_per_param,
                   "compute": compute_budget,
               }

    DE DÓNDE SALE ESA RAÍZ CUADRADA
    -------------------------------
    Del `C = 6ND` del modulo 01 (coste total = 6 x parametros x tokens) mas la restriccion de
    que quieres `D = k*N`, con k los tokens por parametro:

        C = 6 * N * (k*N) = 6k * N²

        N = sqrt( C / (6k) )
        D = k * N

    O sea: si duplicas el presupuesto, el modelo optimo NO se duplica, crece un 41%
    (sqrt(2)). Y los datos tambien un 41%. Ambos a la vez.

    EL PROBLEMA QUE RESUELVE
    ------------------------
    Tienes un presupuesto fijo de FLOPs —una GPU y dos semanas, digamos—. Puedes gastarlo en un
    modelo grande con pocos datos o en uno pequenyo con muchos datos. ¿Cual acaba con menos
    perdida?

    LA RESPUESTA (Hoffmann et al. 2022, "Chinchilla")
    -------------------------------------------------
    Los dos deben crecer PROPORCIONALMENTE: unos 20 tokens por parametro.

    Fue un resultado importante porque contradecia la practica del momento. GPT-3 tenia 175.000
    millones de parametros y se entreno con 300.000 millones de tokens: 1,7 tokens por
    parametro, DOCE VECES por debajo del optimo. Chinchilla entreno un modelo cuatro veces mas
    pequenyo con cuatro veces mas datos, con el mismo computo, y gano en casi todas las
    evaluaciones.

    COMPRUÉBALO CON EL PROPIO CHINCHILLA
    ------------------------------------
    Su presupuesto real fue 5.76e23 FLOPs. Con k=20:

        N = sqrt(5.76e23 / 120) = 6.9e10 = 69.000 millones

    El modelo real tenia 70.000 millones. La formula lo clava. Verla funcionar sobre un caso
    historico da bastante mas confianza que leerla.

    Y AHORA MIRA NUESTRO MODELO
    ---------------------------
    Nuestro modelo tiene 7,62M de parametros no-embedding, asi que el optimo Chinchilla
    serian unos 152M de tokens. Vamos a entrenarlo con 500M: 65 tokens por parametro, mas
    de tres veces por encima del "optimo". Es deliberado, no es un error.

    Chinchilla optimiza la perdida por FLOP DE ENTRENAMIENTO. Si el modelo se va a usar mucho
    despues, compensa entrenar de mas un modelo pequenyo: cada inferencia sale mas barata para
    siempre. Es exactamente lo que hace Llama, y por eso Llama-7B se entreno con 1 billon de
    tokens (143 por parametro) en vez de con 140.000 millones.

    Args:
        compute_budget: FLOPs disponibles. Tiene que ser positivo.
        tokens_per_param: la constante de Chinchilla. 20 por defecto.

    Returns:
        `{"params", "tokens", "tokens_per_param", "compute"}`, todos floats.

    Raises:
        ValueError: si el presupuesto no es positivo.
    """
    raise NotImplementedError("TODO: modulo 12, ejercicio 3 - chinchilla_optimal_allocation")
