"""Modulo 15 - Evaluacion.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa -> `llmfs check 15` -> `llmfs hint 15 -e N`
-> `SOLUCION.md` tiene el codigo completo.

Los tres ejercicios son cortos. El tercero ni siquiera calcula nada: organiza el trabajo
para que la parte que de verdad importa (leer lo que escribe el modelo) sea comoda.

QUÉ VAS A CONSTRUIR
===================

Tres formas de responder a "¿es bueno mi modelo?":

    perplexity_from_loss  (ej. 1)  la metrica de siempre, y sus limites
    bits_per_byte         (ej. 2)  la que SI se puede comparar entre modelos
    run_prompt_battery    (ej. 3)  la que ninguna metrica automatica sustituye

VOCABULARIO QUE VAS A NECESITAR
===============================

- **perplejidad**: `e` elevado a la perdida. "Entre cuantas opciones esta dudando el
  modelo". Depende del tokenizador, y por eso comparar perplejidades entre modelos
  distintos suele no significar nada.
- **bits por byte**: la perdida normalizada por bytes de texto original en vez de por
  tokens. Si es comparable, y se interpreta como cuanto comprimiria el modelo el texto.
- **nat / bit**: unidades de informacion. Un nat son 1,4427 bits.
- **evaluacion cualitativa**: leer lo que escribe el modelo y juzgarlo. Sigue siendo
  imprescindible.
- **contaminacion**: cuando el conjunto de test aparece en los datos de entrenamiento. Es
  uno de los problemas metodologicos mas serios del campo ahora mismo.

    llmfs demo 15     evalua tu modelo y genera eval_report.md
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

# Los seis prompts de la bateria del paper TinyStories. Cada uno prueba algo distinto.
from llmfs.reference import PROMPTS_TINYSTORIES


def perplexity_from_loss(loss: float) -> float:
    """Perplejidad a partir de la perdida media en nats.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Dos lineas.

        1. La guarda:

               if not math.isfinite(loss):
                   return float("inf")

        2. La formula:

               return math.exp(loss)

    QUÉ TIENE QUE SALIR
    -------------------
        perdida 8.317  ->  perplejidad 4096.0    sin entrenar: duda entre TODO el vocabulario
        perdida 1.60   ->  perplejidad    4.95   duda entre unas 5 opciones
        perdida 0.0    ->  perplejidad    1.0    perfecto, no duda

    El primero merece una comprobacion: `ln(4096) = 8.317`, asi que `exp(8.317) = 4096`. Un
    modelo recien inicializado reparte la probabilidad por igual entre los 4096 tokens, y la
    perplejidad te lo dice literalmente: "estoy dudando entre 4096 opciones".

    CÓMO SE INTERPRETA
    ------------------
    La perdida en nats no se lee bien: 1.60 no dice nada por si solo. La perplejidad si, porque
    tiene unidades comprensibles: ENTRE CUANTAS OPCIONES EQUIPROBABLES esta dudando el modelo.

    Una perplejidad de 5 significa "de media, el modelo esta tan indeciso como si tuviera que
    elegir al azar entre 5 palabras". Es una medida de sorpresa.

    POR QUÉ LA GUARDA
    -----------------
    Sin ella, `math.exp(float("inf"))` lanza `OverflowError` y te quedas sin saber que paso. Y
    un `inf` o un `nan` en la perdida SI ocurre: es exactamente lo que ves cuando el
    entrenamiento diverge. Devolver `inf` es la respuesta correcta —la perplejidad de un modelo
    roto ES infinita— y ademas se imprime sin romper nada.

    `math.isfinite(x)` es False para `inf`, `-inf` y `nan`.

    Es una linea de codigo. Lo que importa es saber interpretarla, y saber que NO se puede
    comparar entre modelos con tokenizadores distintos, que es el ejercicio siguiente.

    Args:
        loss: la perdida MEDIA por token, en nats (lo que devuelve `F.cross_entropy`).

    Returns:
        La perplejidad. `inf` si la perdida no es finita.
    """
    raise NotImplementedError("TODO: modulo 15, ejercicio 1 - perplexity_from_loss")


def bits_per_byte(total_loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    """Bits por byte: la metrica que SI se puede comparar entre tokenizadores distintos.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Dos lineas.

        1. La validacion:

               if n_bytes <= 0:
                   raise ValueError(f"n_bytes tiene que ser positivo: {n_bytes}")

        2. La formula:

               return (total_loss_nats / math.log(2)) / n_bytes

    EL PROBLEMA CON LA PERPLEJIDAD
    ------------------------------
    Depende del tokenizador. Si tu vocabulario parte las palabras en trozos mas pequenyos, cada
    token individual es mas facil de predecir y tu perplejidad sale MEJOR sin que el modelo sea
    mejor.

    Caso extremo para verlo claro: un modelo que prediga bit a bit tendria perplejidad cercana a
    2 (solo hay dos opciones, 0 y 1) y seria completamente inutil. Su perplejidad seria la mejor
    del mundo.

    Comparar perplejidades entre modelos con tokenizadores distintos no significa nada, y se
    hace constantemente en blogs y papers.

    LA SOLUCIÓN
    -----------
    Normalizar por BYTES del texto original en vez de por tokens. Los bytes no dependen de como
    trocees: el mismo texto tiene los mismos bytes con cualquier tokenizador.

    El `/ math.log(2)` convierte nats a bits: un nat son `1/ln(2) = 1.4427` bits. Es solo un
    cambio de unidades, como pasar de metros a pies.

    LA INTERPRETACIÓN, QUE ES BONITA
    --------------------------------
    Es literalmente cuantos bits necesitarias para transmitir el texto usando el modelo como
    compresor. Un modelo de 1,0 bits/byte comprime el texto a la OCTAVA parte (un byte son 8
    bits).

        gzip sobre texto en ingles     ~2,5 bits/byte
        un buen modelo pequenyo        ~1,2
        los mejores LLM                 0,6 - 0,8

    Esta equivalencia entre prediccion y compresion viene de Shannon (1948) y no es una
    analogia: es una identidad matematica. Un modelo de lenguaje ES un compresor, y un buen
    compresor ES un modelo de lenguaje. Predecir bien y comprimir bien son la misma cosa.

    OJO CON EL PRIMER ARGUMENTO
    ---------------------------
    Es la perdida TOTAL (la suma sobre todos los tokens), NO la media. Si le pasas la media, el
    resultado sale dividido por el numero de tokens y no significa nada, y ademas no da ningun
    error: sale un numero plausible pero mil veces mas pequenyo.

    Para conseguirla: `perdida_media * n_tokens`, o acumula `loss.item() * y.numel()` en el
    bucle de evaluacion.

    Y `n_tokens` NO se usa en el calculo. Esta en la firma justo para dejar claro cual es la
    unidad del primer argumento, y para que puedas verificar de un vistazo que la perdida es
    total y no media. Es documentacion ejecutable.

    Args:
        total_loss_nats: la perdida TOTAL en nats, sumada sobre todos los tokens.
        n_tokens: cuantos tokens. No se usa; esta para dejar clara la unidad.
        n_bytes: los bytes del texto original (`len(texto.encode("utf-8"))`).

    Returns:
        Bits por byte.

    Raises:
        ValueError: si `n_bytes` no es positivo.
    """
    raise NotImplementedError("TODO: modulo 15, ejercicio 2 - bits_per_byte")


def run_prompt_battery(
    generate_fn: Callable[[str], str],
    prompts: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Genera una completion para cada prompt de la bateria cualitativa.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Tres lineas.

        1. Los prompts por defecto:

               if prompts is None:
                   prompts = PROMPTS_TINYSTORIES

        2. El recorrido, con las tres claves EXACTAS (hay un test que las comprueba):

               return [
                   {
                       "prompt": prompt,
                       "que_prueba": etiqueta,
                       "completion": generate_fn(prompt),
                   }
                   for prompt, etiqueta in prompts
               ]

    `PROMPTS_TINYSTORIES` es una tupla de tuplas `(prompt, etiqueta)`, asi que se desempaqueta
    directamente en el `for`.

    QUÉ ES ESTO
    -----------
    La parte de la evaluacion que ninguna metrica automatica sustituye: hacerle al modelo
    siempre las MISMAS preguntas y LEER lo que responde.

    Los seis prompts de `PROMPTS_TINYSTORIES` vienen del paper de TinyStories y cada uno prueba
    algo distinto: continuacion narrativa, coherencia causal, seguimiento de un objeto
    mencionado antes, resolucion y cierre de la historia. No son prompts al azar.

    POR QUÉ SE PASA `generate_fn` EN VEZ DEL MODELO
    -----------------------------------------------
    `generate_fn(prompt) -> texto` encapsula el modelo Y el tokenizador. Asi esta funcion no
    sabe nada de ninguno de los dos, y puedes usarla igual con tu modelo, con uno cargado de un
    checkpoint, con GPT-4 por API, o con una funcion falsa para probar el test sin entrenar
    nada.

    Es el mismo patron que `get_batch` en el modulo 04 o `optimizer_factory` en el 13: pasar la
    capacidad como funcion en vez de acoplarse a un objeto concreto.

    DÓNDE ESTÁ EL VALOR DE ESTE EJERCICIO
    -------------------------------------
    No esta en el codigo, que son tres lineas. Esta en que tengas una bateria FIJA que puedas
    volver a pasar cada vez que cambies algo, y comparar las respuestas lado a lado.

    Sin bateria fija, la evaluacion cualitativa se convierte en escribir prompts hasta encontrar
    uno donde el modelo quede bien. Con ella, ves cuando una tirada nueva empeora en algo que
    antes salia bien, que es la informacion que de verdad necesitas.

    Args:
        generate_fn: `fn(prompt) -> texto generado`.
        prompts: secuencia de `(prompt, etiqueta)`, o None para usar la bateria de TinyStories.

    Returns:
        Una lista de dicts con las claves `prompt`, `que_prueba` y `completion`.
    """
    raise NotImplementedError("TODO: modulo 15, ejercicio 3 - run_prompt_battery")
