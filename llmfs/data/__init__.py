"""Descarga, tokenizacion y carga de datos.

Se construye a lo largo de los modulos 03, 04 y 13.
"""

from llmfs.data.download import fetch_tinyshakespeare
from llmfs.data.prepare import Dataset, hacer_get_batch, preparar, preparar_shakespeare

__all__ = [
    "fetch_tinyshakespeare",
    "Dataset",
    "preparar",
    "preparar_shakespeare",
    "hacer_get_batch",
]
