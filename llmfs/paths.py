"""Localizacion de la raiz del repo y de sus directorios.

Existe porque `modulos/`, `configs/` y `data/` se resuelven relativos a la raiz del repo,
no al directorio desde el que ejecutes el comando. Asi `python -m llmfs status` funciona
igual desde la raiz que desde `modulos/05_atencion/`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_MARKERS = ("pyproject.toml", ".git")


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Raiz del repositorio.

    Orden de busqueda: `$LLMFS_ROOT` -> ancestro de este fichero que contenga
    `pyproject.toml` -> ancestro del cwd. Con instalacion editable (`uv sync`) la primera
    busqueda ya acierta, porque el paquete vive dentro del repo.
    """
    env = os.environ.get("LLMFS_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        for candidate in (start, *start.parents):
            if any((candidate / marker).exists() for marker in _MARKERS):
                return candidate

    return Path.cwd().resolve()


def modules_dir() -> Path:
    return repo_root() / "modulos"


def configs_dir() -> Path:
    return repo_root() / "configs"


def data_dir() -> Path:
    return repo_root() / "data"


def runs_dir() -> Path:
    return repo_root() / "runs"


def figures_dir() -> Path:
    """Donde escriben sus PNG las demos. Esta en .gitignore."""
    path = runs_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def progress_file() -> Path:
    return repo_root() / ".llmfs_progress.json"
