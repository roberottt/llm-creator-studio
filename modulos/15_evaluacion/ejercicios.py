"""Modulo 15 - Evaluacion.

Tres ejercicios cortos. El tercero no calcula nada: organiza el trabajo para que la parte
que de verdad importa (leer lo que escribe el modelo) sea comoda.

    llmfs check 15
    llmfs demo 15     evalua tu modelo entrenado y genera eval_report.md
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

# Los seis prompts de la bateria del paper TinyStories. Cada uno prueba algo distinto.
from llmfs.reference import PROMPTS_TINYSTORIES


def perplexity_from_loss(loss: float) -> float:
    """Perplejidad a partir de la perdida media en nats.

    QUE ES ESTO
        La perdida en nats no se lee bien: 1.60 no dice mucho. La perplejidad si:

            perplejidad = exp(perdida)

        Y se interpreta como ENTRE CUANTAS OPCIONES EQUIPROBABLES esta dudando el modelo.

            perdida 8.317  ->  perplejidad 4096   (sin entrenar: duda entre todo el vocab)
            perdida 1.60   ->  perplejidad 4.95   (duda entre unas 5 opciones)
            perdida 0      ->  perplejidad 1      (perfecto, no duda)

    CASO BORDE
        Si la perdida no es finita (inf o nan), devuelve `inf`. Sin esa guarda,
        `math.exp(inf)` lanza OverflowError y te quedas sin saber que paso.

    Es una linea de codigo. Lo que importa es saber interpretarla.
    """
    raise NotImplementedError("TODO: modulo 15, ejercicio 1 - perplexity_from_loss")


def bits_per_byte(total_loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    """Bits por byte: la metrica que SI se puede comparar entre tokenizadores distintos.

    EL PROBLEMA CON LA PERPLEJIDAD
        Depende del tokenizador. Si tu vocabulario parte las palabras en trozos mas
        pequenyos, cada token individual es mas facil de predecir y tu perplejidad sale
        mejor SIN QUE EL MODELO SEA MEJOR.

        Caso extremo: un modelo a nivel de bit tendria perplejidad cercana a 2 y seria
        inutil.

        Comparar perplejidades entre modelos con tokenizadores distintos no significa
        nada, y se hace constantemente.

    LA SOLUCION
        Normalizar por BYTES del texto original en vez de por tokens. Los bytes no
        dependen de como trocees.

            bits_por_byte = (perdida_total_en_nats / ln(2)) / n_bytes

        El `/ ln(2)` convierte nats a bits: un nat son 1/ln(2) = 1.4427 bits.

    LA INTERPRETACION, que es bonita
        Es literalmente cuantos bits necesitarias para transmitir el texto usando el modelo
        como compresor. Un modelo de 1.0 bits/byte comprime el texto a la octava parte.

            gzip sobre texto en ingles  ~2.5 bits/byte
            un buen modelo pequenyo     ~1.2
            los mejores LLM              0.6-0.8

        Esta equivalencia entre prediccion y compresion viene de Shannon (1948) y no es
        una analogia: es una identidad. Un modelo de lenguaje ES un compresor.

    OJO CON EL PRIMER ARGUMENTO
        Es la perdida TOTAL (la suma), no la media. Si le pasas la media, el resultado sale
        dividido por el numero de tokens y no significa nada. El parametro `n_tokens` no se
        usa en el calculo; esta en la firma justo para dejar claro que la perdida es total.

    Raises:
        ValueError: si `n_bytes` no es positivo.
    """
    raise NotImplementedError("TODO: modulo 15, ejercicio 2 - bits_per_byte")


def run_prompt_battery(
    generate_fn: Callable[[str], str],
    prompts: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Genera una completion para cada prompt de la bateria cualitativa.

    QUE ES ESTO
        La parte de la evaluacion que ninguna metrica automatica sustituye: hacerle al
        modelo siempre las mismas preguntas y LEER lo que responde.

        Los seis prompts de `PROMPTS_TINYSTORIES` vienen del paper y cada uno prueba algo
        distinto: continuacion narrativa, coherencia causal, seguimiento de un objeto
        mencionado antes, resolucion y cierre de historia.

    POR QUE SE PASA `generate_fn` EN VEZ DEL MODELO
        `generate_fn(prompt) -> texto` encapsula el modelo Y el tokenizador. Asi esta
        funcion no sabe nada de ninguno de los dos, y puedes usarla igual con tu modelo,
        con uno cargado de un checkpoint, o con una funcion falsa para probar.

        Es el mismo patron que `get_batch` en el modulo 04 o `optimizer_factory` en el 13:
        pasar la capacidad como funcion en vez de acoplarse a un objeto concreto.

    QUE HAY QUE HACER
        Recorrer los prompts y devolver una lista de dicts con estas tres claves EXACTAS:

            {"prompt": ..., "que_prueba": ..., "completion": ...}

        `PROMPTS_TINYSTORIES` es una tupla de tuplas `(prompt, etiqueta)`, asi que se
        desempaqueta directamente.

        Si `prompts` es None, usa `PROMPTS_TINYSTORIES`.

    Son tres lineas. El valor del ejercicio no esta en el codigo: esta en que tengas una
    bateria FIJA que puedas volver a pasar cada vez que cambies algo, y comparar.
    """
    raise NotImplementedError("TODO: modulo 15, ejercicio 3 - run_prompt_battery")
