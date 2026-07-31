"""llmfs - el paquete del curso "LLM desde cero".

Aqui vive el codigo "de produccion": el que realmente entrena el modelo final.
Se divide en tres capas que conviene no confundir:

1. `llmfs.reference`  -> implementaciones canonicas, completas y correctas de cada pieza.
                         Escritas para leerse y para servir de red de seguridad.
2. `llmfs.bridge`     -> el resolvedor. Para cada pieza intenta usar TU ejercicio de
                         `modulos/NN_*/ejercicios.py`; si no esta implementado o falla el
                         smoke test, cae automaticamente a `llmfs.reference` y te avisa.
3. el resto           -> modelo, entrenamiento, inferencia y evaluacion, construidos sobre
                         las piezas que devuelve el bridge. Es decir: cuando tu ejercicio
                         del modulo 06 esta bien, el modelo final entrena con TU RMSNorm.

IMPORTANTE: este modulo NO importa torch. La CLI tiene que arrancar rapido, y torch tarda
del orden de un segundo en cargar. Los submodulos que lo necesitan lo importan ellos.
"""

from __future__ import annotations

import os

# Debe fijarse ANTES de que se importe torch para que el backend MPS enrute a CPU los ops
# que todavia no tiene implementados en lugar de reventar con NotImplementedError.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"

__all__ = ["__version__"]
