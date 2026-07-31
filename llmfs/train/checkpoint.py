"""Guardar y reanudar entrenamientos.

Una tirada de cuatro horas se interrumpe: se va la luz, cierras el portatil sin querer, o
simplemente quieres apagar el ordenador. Sin checkpoints eso significa empezar de cero.

Lo importante de un checkpoint reanudable es que NO basta con guardar los pesos. Hay que
guardar tambien el estado del optimizador (los momentos de Adam), el del GradScaler y el
numero de paso. Si reanudas solo con los pesos, Adam arranca con sus momentos a cero y el
modelo pega un bandazo justo al reanudar: se ve como un pico en la curva de perdida.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from llmfs.config import RunConfig


def guardar_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    tokens_vistos: int,
    best_val_loss: float,
    cfg: RunConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    """Guarda todo lo necesario para reanudar exactamente donde estabas.

    Se escribe primero en un fichero temporal y se renombra al final. Asi, si el proceso
    muere a mitad de la escritura, el checkpoint anterior sigue intacto. Un checkpoint a
    medias es peor que no tener checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "tokens_vistos": tokens_vistos,
        "best_val_loss": best_val_loss,
        "config": asdict(cfg),
        "extra": extra or {},
    }

    temporal = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporal)
    temporal.replace(path)


def cargar_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Carga un checkpoint y, si le pasas los objetos, restaura su estado.

    `map_location="cpu"` por defecto: asi un checkpoint guardado en CUDA se puede cargar en
    una maquina sin GPU. El modelo ya se mueve al dispositivo despues.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el checkpoint {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])

    return payload


def ultimo_checkpoint(directorio: str | Path) -> Path | None:
    """El checkpoint mas reciente de un directorio, o `None` si no hay ninguno."""
    directorio = Path(directorio)
    if not directorio.exists():
        return None
    candidatos = sorted(directorio.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return candidatos[-1] if candidatos else None
