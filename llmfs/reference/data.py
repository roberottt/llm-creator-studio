"""Referencia del modulo 03: de una lista de ids a batches en la GPU."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

#: uint16 llega hasta 65.535. Con vocab_size=4096 sobra de largo, y ocupa la mitad que
#: uint32: 500M tokens son 1 GB en vez de 2 GB.
MAX_UINT16 = 2**16


def pack_tokens_uint16(ids: Sequence[int], vocab_size: int) -> np.ndarray:
    """Empaqueta ids en un array `uint16`, validando que quepan.

    La validacion no es paranoia: si un id se sale de rango, numpy hace *wrap around* en
    silencio (65536 se convierte en 0) y acabas entrenando con datos corruptos que no
    dan ningun error visible. El modelo simplemente aprende peor y no sabes por que.

    Raises:
        ValueError: si `vocab_size` no cabe en uint16 o si algun id se sale del vocabulario.
    """
    if vocab_size > MAX_UINT16:
        raise ValueError(
            f"vocab_size={vocab_size} no cabe en uint16. Usa uint32 (y el doble de disco)."
        )

    array = np.asarray(ids, dtype=np.int64)
    if array.size and (array.min() < 0 or array.max() >= vocab_size):
        raise ValueError(
            f"ids fuera del vocabulario [0, {vocab_size}): "
            f"minimo={array.min()}, maximo={array.max()}"
        )
    return array.astype(np.uint16)


def train_val_split(tokens: np.ndarray, val_fraction: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    """Parte el corpus en entrenamiento y validacion.

    El corte es **contiguo y por el final**, no aleatorio. Motivo: con una ventana
    deslizante, dos muestras que se solapan comparten casi todos sus tokens. Si barajaras
    a nivel de token, el conjunto de validacion contendria trozos de secuencias vistas en
    entrenamiento y la perplejidad saldria optimista. Cortar por el final garantiza que
    validacion son historias que el modelo no ha visto nunca.

    Returns:
        `(train, val)` como vistas del array original (no copia nada).
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction debe estar en (0, 1), no {val_fraction}")

    n_val = max(1, int(len(tokens) * val_fraction))
    if n_val >= len(tokens):
        raise ValueError("val_fraction deja el conjunto de entrenamiento vacio")
    return tokens[:-n_val], tokens[-n_val:]


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device | str | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Muestrea un batch de ventanas contiguas.

    Se eligen `batch_size` posiciones de inicio al azar y de cada una se toman
    `context_length + 1` tokens: los primeros `context_length` son la entrada y los
    ultimos `context_length` (desplazados uno) son el objetivo. Es decir, la tarea es
    "predice el siguiente token" en cada posicion a la vez.

        data  = [ 5, 8, 2, 9, 1, ...]
        x     = [ 5, 8, 2, 9]
        y     = [ 8, 2, 9, 1]

    Con `T` posiciones se obtienen `T` predicciones por muestra, no una. Por eso el
    entrenamiento de un modelo de lenguaje es tan eficiente en datos.

    Args:
        data: array 1-D de tokens (tipicamente un `np.memmap` uint16).
        device: si es CUDA se usa `pin_memory` + `non_blocking` para solapar la copia
            con el calculo.
        rng: generador de numpy. Fijalo para tener batches reproducibles.

    Returns:
        `(x, y)`, ambos `int64` de forma `(batch_size, context_length)`.
    """
    rng = rng or np.random.default_rng()
    max_start = len(data) - context_length - 1
    if max_start < 1:
        raise ValueError(
            f"El corpus ({len(data)} tokens) es mas corto que el contexto "
            f"({context_length} + 1)."
        )

    starts = rng.integers(0, max_start, size=batch_size)
    # astype(int64) tambien materializa el memmap: sin la copia, torch se quedaria
    # apuntando a memoria mapeada de disco y cada acceso seria una lectura.
    x_np = np.stack([data[i : i + context_length] for i in starts]).astype(np.int64)
    y_np = np.stack([data[i + 1 : i + 1 + context_length] for i in starts]).astype(np.int64)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)

    if device is not None:
        device = torch.device(device)
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

    return x, y
