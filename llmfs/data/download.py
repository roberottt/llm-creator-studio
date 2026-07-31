"""Descarga de corpus, con cache en disco y con plan B si no hay internet.

Todo lo que se descarga se guarda en `data/` (que esta en .gitignore) y no se vuelve a
pedir. Si no hay red, las funciones devuelven un texto de reserva pequenyo en lugar de
reventar: un curso donde el modulo 00 falla porque estas en un tren es un mal curso.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from llmfs.paths import data_dir

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

#: Texto de reserva para cuando no hay conexion. Es original, escrito para el curso, y
#: deliberadamente repetitivo: con pocos kilobytes un modelo por conteo necesita
#: regularidad para que se note la diferencia entre contextos de 1 y de 4 caracteres.
FALLBACK_TEXT = """
El gato duerme sobre el tejado. El gato baja del tejado cuando tiene hambre.
La nina mira al gato desde la ventana y el gato mira a la nina desde el tejado.
El perro ladra en el patio. El perro ladra cuando el gato baja del tejado.
La nina abre la ventana y llama al gato. El gato no viene porque el perro ladra.
El perro se cansa de ladrar y se duerme en el patio. Entonces el gato baja.
El gato entra por la ventana y la nina le da comida. El gato come y ronronea.
La noche llega y la nina cierra la ventana. El gato duerme dentro de la casa.
El perro duerme en el patio. La casa entera duerme hasta que llega la manana.
Por la manana el gato quiere salir. La nina abre la ventana y el gato sube al tejado.
El perro despierta y ladra al gato. El gato mira al perro desde el tejado y no baja.
""".strip()


def _download_text(url: str, destination: Path, timeout: int = 30) -> str | None:
    """Descarga un fichero de texto. Devuelve `None` si no se puede."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(raw, encoding="utf-8")
    return raw


def fetch_tinyshakespeare(quiet: bool = False) -> tuple[str, str]:
    """Devuelve el texto de tiny-shakespeare, descargandolo la primera vez.

    Son ~1,1 MB de obras de Shakespeare concatenadas. Es el "hola mundo" de los modelos
    de lenguaje a nivel caracter: suficientemente pequenyo para entrenar en segundos y
    suficientemente estructurado para que se note cuando el modelo aprende algo.

    Returns:
        `(texto, origen)` donde origen es `"cache"`, `"descarga"` o `"reserva"`.
        Comprobar el origen permite a las demos avisar de que estan usando el plan B.
    """
    destino = data_dir() / "tinyshakespeare.txt"

    if destino.exists():
        return destino.read_text(encoding="utf-8"), "cache"

    if not quiet:
        print(f"[llmfs] descargando tiny-shakespeare (~1 MB) a {destino}...")

    texto = _download_text(TINYSHAKESPEARE_URL, destino)
    if texto is not None:
        return texto, "descarga"

    if not quiet:
        print("[llmfs] sin conexion: se usa el texto de reserva, mucho mas pequenyo.")
    return FALLBACK_TEXT, "reserva"
