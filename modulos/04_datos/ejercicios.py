"""Modulo 04 - Datos: de texto a batches en la GPU.

Tres funciones cortas. La tercera, `get_batch`, es la que usara el bucle de entrenamiento
del modulo 11 miles de veces, asi que merece la pena que la entiendas bien.

    llmfs check 04
    llmfs demo 04     tokeniza shakespeare, lo guarda en disco y ensenya un batch de verdad
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

#: uint16 llega hasta 65.535. Con vocab_size=4096 sobra, y ocupa la mitad que uint32.
MAX_UINT16 = 2**16


def pack_tokens_uint16(ids: Sequence[int], vocab_size: int) -> np.ndarray:
    """Convierte una lista de ids en un array `uint16`, validando que quepan.

    QUE ES ESTO
        Vas a guardar 500 millones de tokens en disco. El tipo que elijas multiplica o
        divide el tamanyo del fichero:

            int64  (el de python)  ->  4 GB
            uint32                 ->  2 GB
            uint16                 ->  1 GB

        Como nuestros ids van de 0 a 4095, uint16 (que llega a 65.535) sobra de largo.

    POR QUE HAY QUE VALIDAR ANTES
        Esta es la parte importante del ejercicio. NumPy NO avisa si un numero no cabe:
        hace wrap around en silencio.

            np.array([65536], dtype=np.int64).astype(np.uint16)   ->  [0]

        Sin excepcion, sin warning. Tus datos quedan corruptos, el modelo entrena peor y
        no hay absolutamente nada que apunte a la causa. Podrias pasarte dias buscandolo.

        Diez lineas de validacion aqui te ahorran eso.

    QUE HAY QUE COMPROBAR
        1. Que `vocab_size` quepa en uint16 (o sea, <= 65.536). Si no, hay que usar uint32
           y el doble de disco.
        2. Que ningun id sea negativo ni >= `vocab_size`.

        Y lanzar `ValueError` con un mensaje que diga QUE valor se ha salido, no solo que
        algo esta mal.

    COMO
        Convierte primero a `np.int64` (donde todo cabe), comprueba el minimo y el maximo,
        y solo entonces convierte a `uint16` con `.astype(np.uint16)`.

        Cuidado con el array vacio: `.min()` sobre un array de tamanyo 0 lanza excepcion.
        Comprueba `array.size` antes.

    Args:
        ids: los ids a empaquetar.
        vocab_size: tamanyo del vocabulario. Todos los ids deben estar en [0, vocab_size).

    Returns:
        Un `np.ndarray` de dtype `uint16`.

    Raises:
        ValueError: si `vocab_size` no cabe en uint16, o si algun id se sale del rango.
    """
    raise NotImplementedError("TODO: modulo 04, ejercicio 1 - pack_tokens_uint16")


def train_val_split(tokens: np.ndarray, val_fraction: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """Separa un trozo del final para validacion.

    QUE ES ESTO
        Necesitas texto que el modelo NO vea al entrenar, para poder comprobar si esta
        aprendiendo de verdad o solo memorizando.

    POR QUE EL CORTE ES CONTIGUO Y POR EL FINAL, Y NO ALEATORIO
        Es la unica decision de este ejercicio, y es la importante.

        Las ventanas de entrenamiento se solapan: la que empieza en la posicion 100 y la
        que empieza en la 101 comparten 511 de sus 512 tokens. Si repartieras al azar,
        tu conjunto de validacion estaria lleno de fragmentos que el modelo ya vio.
        La perdida de validacion saldria estupenda y no significaria nada.

        Cortando un bloque contiguo del final, lo que reservas son historias enteras que
        el modelo no ha visto jamas.

    COMO
            n_val = max(1, int(len(tokens) * val_fraction))
            train = tokens[:-n_val]
            val   = tokens[-n_val:]

        El `max(1, ...)` evita que con un corpus pequenyo salga un conjunto de validacion
        de tamanyo 0.

        Devuelve VISTAS del array original (que es lo que hace el slicing de numpy), no
        copias. Con 500M tokens, copiar seria un gigabyte de mas para nada.

    Args:
        tokens: array 1-D con el corpus entero.
        val_fraction: fraccion para validacion. Debe estar en (0, 1).

    Returns:
        `(train, val)`.

    Raises:
        ValueError: si `val_fraction` no esta en (0, 1), o si dejaria el conjunto de
            entrenamiento vacio.
    """
    raise NotImplementedError("TODO: modulo 04, ejercicio 2 - train_val_split")


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device | str | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Saca un batch de ventanas al azar. Esta es la funcion que mas veces se ejecuta.

    QUE ES ESTO
        Eliges `batch_size` posiciones al azar del corpus. De cada una coges una ventana
        de `context_length` tokens como entrada, y la MISMA ventana desplazada un token
        como objetivo.

    EL EJEMPLO QUE LO EXPLICA TODO
        data = [5, 8, 2, 9, 1, 7, ...]   con context_length = 4, empezando en la posicion 0

            x = [5, 8, 2, 9]      <- lo que ve el modelo
            y = [8, 2, 9, 1]      <- lo que tiene que predecir

        Leelo columna a columna:

            viendo [5]          -> predecir 8
            viendo [5,8]        -> predecir 2
            viendo [5,8,2]      -> predecir 9
            viendo [5,8,2,9]    -> predecir 1

        UNA ventana de 4 tokens son CUATRO ejemplos de entrenamiento. Con contexto 512 son
        512 predicciones por muestra. Por eso los modelos de lenguaje aprovechan tanto los
        datos.

        (Esto funciona gracias a la mascara causal del modulo 06, que impide que la
        posicion 2 pueda ver el token de la posicion 3. Sin ella el modelo veria la
        respuesta.)

    COMO
        1. `max_start = len(data) - context_length - 1`
           El `-1` es porque `y` necesita un token MAS alla del final de `x`. Sin el, la
           ultima ventana se sale del array.
           Si `max_start` sale menor que 1, el corpus es mas corto que el contexto: lanza
           `ValueError` con un mensaje que diga los dos numeros.

        2. `starts = rng.integers(0, max_start, size=batch_size)`

        3. Apila las ventanas:
               x_np = np.stack([data[i : i+context_length]     for i in starts])
               y_np = np.stack([data[i+1 : i+1+context_length] for i in starts])

        4. Convierte a `np.int64` con `.astype(np.int64)`.
           Es obligatorio: los indices de un `nn.Embedding` tienen que ser int64. Y de
           paso esa conversion COPIA los datos, que es justo lo que quieres: sin la copia
           te quedarias apuntando al fichero mapeado en disco y cada acceso seria una
           lectura.

        5. `torch.from_numpy(...)` para pasarlo a tensores.

        6. Si hay `device`:
               - en CUDA:  `.pin_memory().to(device, non_blocking=True)`
                 Memoria fijada + copia asincrona = la transferencia del siguiente batch
                 se solapa con el calculo del actual. En MPS y CPU esto no aplica.
               - en el resto: `.to(device)` a secas.

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
