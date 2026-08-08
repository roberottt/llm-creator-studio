"""Modulo 04 - Datos: de texto a batches en la GPU.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa -> `llmfs check 04` -> `llmfs hint 04 -e N`
-> `SOLUCION.md` tiene el codigo completo.

QUÉ VAS A CONSTRUIR
===================

El puente entre el texto tokenizado y la GPU. Tres funciones:

    pack_tokens_uint16   (ej. 1)  ids -> array de 2 bytes por token, validando
    train_val_split      (ej. 2)  separar un trozo para validacion
    get_batch            (ej. 3)  sacar un lote de ventanas al azar

Encajan asi, y cada una se ejecuta en un momento muy distinto:

    modulo 03           AQUI                              modulo 11
    ---------           ----                              ---------
    texto     -->  lista de ids  -->  fichero .bin  -->  (x, y)  -->  modelo
                        (ej. 1)          (ej. 2)          (ej. 3)
                   \_________________________/          \_________/
                     UNA vez, al preparar                en CADA paso,
                     el corpus. Puede tardar,            decenas de miles
                     pero no puede fallar en             de veces.
                     silencio: el error se
                     queda grabado en disco.

La tercera es donde esta la idea importante del modulo: como un texto sin etiquetar se
convierte en una tarea de aprendizaje con su respuesta correcta incluida.

DONDE SE EXPLICA CADA COSA
==========================

Cada ejercicio tiene su seccion en TEORIA.md, con el ejemplo numerico correspondiente. Si
un docstring te dice QUE teclear pero no acabas de ver POR QUE, ese es el sitio:

    pack_tokens_uint16  ->  "Ejercicio 1: empaquetar el corpus"
    train_val_split     ->  "Ejercicio 2: apartar la validacion"
    get_batch           ->  "Ejercicio 3: sacar un batch"

Y antes de teclear nada, dos secciones que dan el contexto:

    "Que parte del LLM es esta"          en que fase de construir un LLM estas, que es el
                                         aprendizaje autosupervisado y por que el texto trae
                                         la respuesta puesta. Es LA idea del modulo.
    "Las tres cosas con las que se       la lista de ids, el array uint16 (o el memmap, que
     trabaja"                            se usa igual) y el par (x, y) de tensores: las tres
                                         estructuras que salen en todas las firmas.

El ejercicio 3 tiene ademas un ejemplo trabajado con numeros de verdad —corpus [0..999],
semilla 0, las cuatro posiciones de inicio que salen y las dos matrices completas— en la
seccion "Que hace la funcion, con numeros de verdad". Si tu implementacion reproduce esas
matrices, has terminado.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **batch**: un grupo de muestras que se procesan a la vez. Ir de una en una desaprovecha
  la GPU.
- **ventana / contexto**: cuantos tokens seguidos ve el modelo de golpe. El nuestro, 512.
- **memmap**: un array que vive en disco pero se usa como si estuviera en memoria. El
  sistema operativo carga solo lo que tocas.
- **conjunto de validacion**: texto que el modelo NO ve al entrenar, para saber si esta
  aprendiendo o solo memorizando.
- **uint16**: entero sin signo de 2 bytes, de 0 a 65.535.

    llmfs demo 04     el pipeline entero, de texto a batch en la GPU
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

#: uint16 llega hasta 65.535. Con vocab_size=4096 sobra, y ocupa la mitad que uint32.
MAX_UINT16 = 2**16


def pack_tokens_uint16(ids: Sequence[int], vocab_size: int) -> np.ndarray:
    """Convierte una lista de ids en un array `uint16`, validando que quepan.

    Se ejecuta UNA vez, al preparar el corpus, y por eso es casi toda validacion: lo que
    escriba se queda grabado en el fichero con el que entrenaras despues. La seccion
    "Ejercicio 1: empaquetar el corpus" de `TEORIA.md` tiene las cinco conversiones
    ejecutadas de verdad, incluida la que convierte un id fuera de rango en otro que parece
    perfectamente valido (66536 -> 1000).

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Cuatro pasos, y tres de ellos son la validacion.

        1. Si `vocab_size > 2**16`, lanza `ValueError` (no cabe en uint16, haria falta uint32).

        2. Convierte a `int64` PRIMERO, que es donde todo cabe:

               array = np.asarray(ids, dtype=np.int64)

        3. Si el array NO esta vacio, comprueba el rango:

               if array.size and (array.min() < 0 or array.max() >= vocab_size):
                   raise ValueError(f"... minimo={array.min()}, maximo={array.max()}")

        4. Y solo entonces convierte:

               return array.astype(np.uint16)

    POR QUÉ VALIDAR ANTES DE CONVERTIR
    ----------------------------------
    Numpy NO avisa si un numero no cabe: hace *wrap around* en silencio.

        np.array([65536], dtype=np.int64).astype(np.uint16)   ->   array([0])

    Sin excepcion, sin warning. Tus datos quedan corruptos, el modelo entrena peor, y no hay
    absolutamente nada que apunte a la causa. Podrias pasarte dias buscandolo.

    Si convirtieras primero y validaras despues, el desbordamiento ya habria ocurrido y
    estarias comprobando datos ya corruptos. Por eso el paso 2 va antes que el 3.

    POR QUÉ EL `array.size and ...`
    -------------------------------
    Sobre un array vacio, `.min()` lanza una excepcion sobre reducciones de secuencias vacias:
    un error real, pero que no tiene nada que ver con lo que estas validando y despista. El
    cortocircuito lo evita.

    PON LOS VALORES EN EL MENSAJE DE ERROR
    --------------------------------------
    "ids fuera de rango" no ayuda. "maximo=9999" te dice al instante que tu tokenizador esta
    produciendo ids que no deberia, y con que magnitud. Hay un test que comprueba que el valor
    aparece en el mensaje.

    POR QUÉ uint16
    --------------
        int64  (el de python)  ->  500M tokens = 4 GB
        uint32                 ->  2 GB
        uint16                 ->  1 GB

    uint16 llega hasta 65.535 y nuestros ids van de 0 a 4095: sobra de largo.

    Args:
        ids: los ids a empaquetar.
        vocab_size: el tamanyo del vocabulario. Todos los ids deben estar en [0, vocab_size).

    Returns:
        Un `np.ndarray` de dtype `uint16`.

    Raises:
        ValueError: si `vocab_size` no cabe en uint16, o si algun id se sale del rango.
    """
    raise NotImplementedError("TODO: modulo 04, ejercicio 1 - pack_tokens_uint16")


def train_val_split(tokens: np.ndarray, val_fraction: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """Separa un trozo del final para validacion.

    No baraja, no reordena, no copia: corta y devuelve dos vistas. Toda la sustancia esta en
    POR QUE el corte es por el final, y la seccion "Ejercicio 2: apartar la validacion" de
    `TEORIA.md` lo mide: repartiendo ventanas al azar, el 100% de los tokens de validacion
    resultan estar tambien en entrenamiento.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Cuatro lineas.

        1. Valida que `val_fraction` este en (0, 1) y lanza `ValueError` si no.

        2. Calcula cuantos tokens van a validacion:

               n_val = max(1, int(len(tokens) * val_fraction))

        3. Si `n_val >= len(tokens)`, lanza `ValueError` (dejaria entrenamiento vacio).

        4. Devuelve las dos partes:

               return tokens[:-n_val], tokens[-n_val:]

    LA ÚNICA DECISIÓN DEL EJERCICIO: POR QUÉ EL CORTE ES CONTIGUO Y POR EL FINAL
    ---------------------------------------------------------------------------
    El reflejo habitual seria barajar y repartir. Aqui es un error.

    Las ventanas de entrenamiento SE SOLAPAN: la que empieza en la posicion 100 y la que
    empieza en la 101 comparten 511 de sus 512 tokens. Si repartieras al azar —a nivel de
    token o incluso de ventana— tu conjunto de validacion estaria lleno de fragmentos que el
    modelo ya vio.

    El sintoma seria precioso y enganyoso: perdida de validacion bajisima, casi identica a la
    de entrenamiento, y ninguna senyal de sobreajuste jamas. Estarias midiendo memorizacion y
    llamandolo generalizacion.

    Cortando un bloque contiguo del final, lo que reservas son historias COMPLETAS que el
    modelo no ha visto nunca.

    DOS DETALLES
    ------------
    **El `max(1, ...)`.** Con un corpus de 50 tokens y `val_fraction=0.005`, `int(50*0.005)`
    da 0 y te quedarias sin conjunto de validacion.

    **Devuelve VISTAS, no copias.** El slicing de numpy no copia, y eso es lo que quieres: con
    500M tokens, un `.copy()` gratuito serian 1 GB de RAM a la basura. Hay un test que lo
    comprueba con `np.shares_memory`.

    Args:
        tokens: array 1-D con el corpus entero.
        val_fraction: la fraccion para validacion. Debe estar en (0, 1).

    Returns:
        `(train, val)`.

    Raises:
        ValueError: si `val_fraction` no esta en (0, 1), o si dejaria el entrenamiento vacio.
    """
    raise NotImplementedError("TODO: modulo 04, ejercicio 2 - train_val_split")


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device | str | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Saca un lote de ventanas al azar. Es la funcion que mas veces se ejecuta del curso.

    Antes de teclear, mira el ejemplo trabajado de "Ejercicio 3: sacar un batch" en
    `TEORIA.md`: corpus [0, 1, ..., 999], semilla 0, las cuatro posiciones de inicio que
    salen (842, 631, 506, 267) y las dos matrices completas. Si tu implementacion reproduce
    esas matrices, esta bien.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Seis pasos.

        1. Si `rng` es None, crea uno: `rng = rng or np.random.default_rng()`

        2. Calcula hasta donde puedes empezar, y valida:

               max_start = len(data) - context_length - 1
               if max_start < 1:
                   raise ValueError(f"el corpus ({len(data)}) es mas corto que el contexto ...")

        3. Elige las posiciones de inicio:

               starts = rng.integers(0, max_start, size=batch_size)

        4. Apila las ventanas, la de entrada y la desplazada:

               x_np = np.stack([data[i : i+context_length]     for i in starts]).astype(np.int64)
               y_np = np.stack([data[i+1 : i+1+context_length] for i in starts]).astype(np.int64)

        5. Pasa a tensores:

               x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)

        6. Si hay `device`, muevelos:

               device = torch.device(device)
               if device.type == "cuda":
                   x = x.pin_memory().to(device, non_blocking=True)
                   y = y.pin_memory().to(device, non_blocking=True)
               else:
                   x, y = x.to(device), y.to(device)

    LA IDEA, CON NÚMEROS
    --------------------
        data = [5, 8, 2, 9, 1, 7, ...]     empezando en 0, con context_length = 4

            x = [5, 8, 2, 9]      <- lo que ve el modelo
            y = [8, 2, 9, 1]      <- lo que tiene que predecir

    Leelo columna a columna:

        viendo [5]          -> predecir 8
        viendo [5,8]        -> predecir 2
        viendo [5,8,2]      -> predecir 9
        viendo [5,8,2,9]    -> predecir 1

    UNA ventana de 4 tokens son CUATRO ejemplos de entrenamiento. Con contexto 512, son 512
    predicciones por muestra, y un batch de 48x512 son 24.576. Por eso los modelos de lenguaje
    aprovechan tanto los datos.

    (Esto funciona gracias a la mascara causal del modulo 06, que impide que la posicion 2 vea
    el token 3. Sin ella el modelo veria la respuesta.)

    EL `-1` DEL PASO 2 ES EL OFF-BY-ONE DEL EJERCICIO
    -------------------------------------------------
    `y` necesita un token MAS alla del final de `x`. Si `x` llega hasta `i + context_length - 1`,
    `y` llega hasta `i + context_length`. Sin el `-1`, la ultima ventana posible desborda.

    Y numpy no lanza error al hacer slicing fuera de rango: simplemente devuelve menos
    elementos. Lo que veras es un `np.stack` fallando por formas incompatibles, tres lineas mas
    abajo y sin ninguna pista de la causa real.

    EL `.astype(np.int64)` DEL PASO 4 HACE DOS COSAS
    ------------------------------------------------
    La obvia: `nn.Embedding` exige indices `int64`, y los datos estan en `uint16`.

    La menos obvia: la conversion COPIA. Sin esa copia, `torch.from_numpy` se quedaria
    apuntando a memoria mapeada de disco y cada acceso del modelo seria potencialmente una
    lectura de fichero.

    EL `pin_memory` DEL PASO 6, SOLO EN CUDA
    ----------------------------------------
    Memoria "fijada" es memoria que el sistema operativo se compromete a no mover de sitio, lo
    que permite a la GPU leerla por DMA sin intervencion de la CPU. Con `non_blocking=True` la
    llamada vuelve inmediatamente y la copia se solapa con lo que la GPU este calculando.

    En MPS no aplica (la memoria es unificada, no hay copia que solapar) y en CPU tampoco.

    Args:
        data: array 1-D de tokens (normalmente un `np.memmap` de uint16).
        batch_size: cuantas ventanas.
        context_length: cuantos tokens por ventana.
        device: a donde mover los tensores. `None` los deja en CPU.
        rng: generador de numpy. Pasale uno con semilla fija y obtendras siempre el mismo
            batch, que es lo que permite al test comparar contra la referencia.

    Returns:
        `(x, y)`, ambos tensores `int64` de forma `(batch_size, context_length)`.

    Raises:
        ValueError: si el corpus es mas corto que `context_length + 1`.
    """
    raise NotImplementedError("TODO: modulo 04, ejercicio 3 - get_batch")
