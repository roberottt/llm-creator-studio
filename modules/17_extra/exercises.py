"""Modulo 17 - Extras y limites honestos.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `THEORY.md` -> implementa -> `llmfs check 17` -> `llmfs hint 17 -e N`
-> `SOLUTION.md` tiene el codigo completo.

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

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Tres lineas.

        1. El mayor valor absoluto, por fila o de toda la matriz:

               if per_channel:
                   max_abs = weight.abs().amax(dim=-1, keepdim=True)
               else:
                   max_abs = weight.abs().amax()

        2. La escala:

               escala = (max_abs / 127.0).clamp_min(1e-12)

        3. Los enteros:

               cuantizado = torch.round(weight / escala).clamp(-127, 127).to(torch.int8)
               return cuantizado, escala

    CON NÚMEROS, SEGUIDO A MANO
    ---------------------------
        W = [0.12, -0.45, 0.03, 0.28]

    El mayor en valor absoluto es 0.45. Se mapea ese rango a [-127, +127]:

        escala = 0.45 / 127 = 0.003543
        W_int8 = round(W / escala) = [34, -127, 8, 79]

    Y para recuperar: `W' = W_int8 * escala = [0.1204, -0.4500, 0.0283, 0.2799]`.

    No es exacto: el error es del orden de MEDIA unidad de escala, que es lo que se pierde al
    redondear. Fijate en que el valor maximo (-0.45) se recupera EXACTO, y los pequenyos son los
    que mas error relativo acumulan.

    LA IDEA
    -------
    Un float32 ocupa 4 bytes y un int8 uno solo: el modelo ocupa la cuarta parte, de 35,7 MB a
    9,0 MB. El truco es guardar, junto a los enteros, una ESCALA que permite recuperar los
    valores aproximados.

    POR QUÉ 127 Y NO 128
    --------------------
    int8 va de -128 a 127. Usando 127 el rango queda SIMETRICO y el cero se representa
    EXACTAMENTE (el entero 0 vuelve a ser el float 0).

    Eso importa mas de lo que parece: en una matriz con muchos valores pequenyos, que el cero
    sea exacto evita un sesgo sistematico que se acumularia capa tras capa. Un error de
    redondeo aleatorio se cancela; un sesgo constante no.

    QUÉ SIGNIFICA "SIMÉTRICA"
    -------------------------
    Que el rango se centra en cero. La alternativa (asimetrica) usa tambien un DESPLAZAMIENTO
    —un "zero point"— y aprovecha mejor el rango cuando los datos estan sesgados hacia un lado,
    pero es mas cara de aplicar porque hay que sumar y restar ese desplazamiento en cada
    operacion. Los pesos de una red suelen estar bastante centrados, asi que la simetrica sale
    rentable.

    POR CANAL FRENTE A POR TENSOR
    -----------------------------
    Con `per_channel=True` se calcula una escala por FILA en vez de una sola para toda la
    matriz. Cuesta un vector de escalas mas (despreciable: una por fila) y reduce bastante el
    error, porque una unica fila con valores grandes ya no arrastra a todas las demas a usar
    una escala gruesa.

    Medido sobre una matriz real de nuestro modelo: **0,71% de error por canal frente a 1,07%
    por tensor**.

    LAS DOS PROTECCIONES, QUE NO SON DECORATIVAS
    --------------------------------------------
    **`clamp_min(1e-12)`**: evita dividir por cero si una fila entera es de ceros. Pasa mas de
    lo que parece con pesos podados o con capas que no han aprendido nada.

    **`clamp(-127, 127)`**: protege del redondeo en el borde. Sin el, un valor justo en el
    maximo podria redondear a 128, que no cabe en int8 y hace WRAP a -128. O sea, el peso mas
    grande de la fila se convertiria en el mas negativo. Es un bug espectacular y silencioso.

    Args:
        weight: la matriz de pesos, en float.
        per_channel: una escala por fila (True) o una para toda la matriz (False).

    Returns:
        `(cuantizado, escala)`, con `cuantizado` de dtype `int8`.
    """
    raise NotImplementedError("TODO: modulo 17, ejercicio 1 - quantize_int8_symmetric")


def dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Vuelve a float multiplicando por la escala.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Una linea:

        return quantized.to(torch.float32) * scale

    EL `.to(torch.float32)` ES OBLIGATORIO, Y VA ANTES
    --------------------------------------------------
    Si multiplicaras el int8 directamente, PyTorch haria la operacion en enteros y el resultado
    seria basura: la escala vale 0,0035, redondeada a entero es 0, y toda la matriz saldria a
    cero.

    Convertir primero y multiplicar despues es el orden correcto. Es un error facil de cometer y
    el sintoma —un modelo que de pronto genera siempre el mismo token— no apunta hacia aqui.

    QUÉ PASA CON EL BROADCAST
    -------------------------
    Con `per_channel=True`, `scale` tiene forma `(filas, 1)` gracias al `keepdim=True` del
    ejercicio anterior, y se emite sola contra `(filas, columnas)`. Cada fila se multiplica por
    su escala. Por eso el `keepdim` no era opcional.

    Con `per_channel=False`, `scale` es un escalar y multiplica a todo por igual.

    EL RESULTADO NO ES IGUAL AL ORIGINAL
    ------------------------------------
    Se ha perdido informacion en el redondeo y no hay forma de recuperarla. Lo que sale es el
    valor APROXIMADO, con un error que depende de cuanto se haya tenido que redondear. El
    ejercicio 3 lo mide.

    Args:
        quantized: el tensor int8.
        scale: la escala que devolvio `quantize_int8_symmetric`.

    Returns:
        El tensor en float32, aproximadamente igual al original.
    """
    raise NotImplementedError("TODO: modulo 17, ejercicio 2 - dequantize_int8")


def quantization_error(original: torch.Tensor, per_channel: bool = True) -> dict[str, float]:
    """Mide cuanto danya la cuantizacion.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Ida y vuelta, y comparar.

        1. El ciclo completo:

               q, escala = quantize_int8_symmetric(original, per_channel=per_channel)
               recuperado = dequantize_int8(q, escala)

        2. El error, elemento a elemento:

               error = (original - recuperado).abs()
               norma_original = float(original.norm())

        3. El dict con las seis metricas:

               return {
                   "relative_error": float(error.norm()) / max(norma_original, 1e-12),
                   "max_error": float(error.max()),
                   "mean_error": float(error.mean()),
                   "compression": original.element_size() / q.element_size(),
                   "original_bytes": original.numel() * original.element_size(),
                   "quantized_bytes": (
                       q.numel() * q.element_size() + escala.numel() * escala.element_size()
                   ),
               }

    QUÉ TIENE QUE SALIR
    -------------------
    Sobre la matriz `q_proj` del primer bloque de nuestro modelo, con `per_channel=True`:

        error_relativo    0.00714     <- el 0,71%
        error_maximo      0.000368
        error_medio       0.000123
        compresion        4.0
        bytes_original    409.600
        bytes_cuantizado  103.680     <- 102.400 del int8 + 1.280 de las 320 escalas

    Y con `per_channel=False`, el error relativo sube a 0.01068. Esa comparacion es el resultado
    del modulo: por canal es claramente mejor y practicamente gratis.

    CUÁL DE LAS SEIS HAY QUE MIRAR
    ------------------------------
    **`error_relativo`**, sin duda. Es el unico independiente de la escala de los datos, asi que
    puedes comparar capas distintas, modelos distintos y configuraciones distintas con el mismo
    numero. Con pesos de una red entrenada, int8 por canal ronda el 0,5-1%.

    `error_maximo` y `error_medio` estan en las unidades de los pesos, asi que solo sirven
    dentro de una misma matriz.

    `error.norm()` es la norma L2 del vector de errores, la misma idea que en `clip_grad_norm`
    del modulo 11: tratar todos los elementos como un unico vector gigante.

    EL DETALLE HONESTO DE `bytes_cuantizado`
    ----------------------------------------
    Se suman TAMBIEN los bytes de las escalas, no solo los del int8. Con una escala por fila son
    despreciables (320 floats frente a 102.400 int8), pero contarlas es lo correcto: si alguien
    usara una escala por cada 8 elementos, el ahorro real seria bastante menor que el 4x
    nominal, y esta metrica lo ensenyaria.

    Fijate en que `compresion` NO usa esos bytes: es la ratio nominal por elemento
    (`element_size()` de 4 frente a 1). Las dos cosas son utiles y miden cosas distintas: una es
    el limite teorico y la otra el tamanyo real en disco.

    `tensor.element_size()` da los bytes por elemento (4 para float32, 1 para int8). Usarlo en
    vez de escribir un 4 y un 1 hace que el codigo siga siendo correcto si un dia cuantizas
    desde float16.

    PARA QUÉ SIRVE
    --------------
    Para saber si merece la pena ANTES de cuantizar el modelo entero y descubrir que genera
    basura. Cuantizas una matriz, miras el error relativo, y decides.

    Args:
        original: la matriz de pesos en float.
        per_channel: que modo de cuantizacion medir.

    Returns:
        Un dict con `error_relativo`, `error_maximo`, `error_medio`, `compresion`,
        `bytes_original` y `bytes_cuantizado`.
    """
    raise NotImplementedError("TODO: modulo 17, ejercicio 3 - quantization_error")
