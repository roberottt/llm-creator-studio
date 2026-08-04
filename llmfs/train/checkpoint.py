"""Saving and resuming training runs.

A four-hour run gets interrupted: the power goes out, you close the laptop by accident, or
you just want to turn the machine off. Without checkpoints that means starting over.

The important thing about a resumable checkpoint is that saving the weights is NOT enough.
You also have to save the optimizer state (Adam's moments), the GradScaler's state and the
step number. If you resume with the weights alone, Adam starts with its moments at zero and
the model lurches right at the restart: it shows up as a spike in the loss curve.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from llmfs.config import RunConfig


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    tokens_seen: int,
    best_val_loss: float,
    cfg: RunConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save everything needed to resume exactly where you were.

    It writes to a temporary file first and renames at the end. That way, if the process
    dies mid-write, the previous checkpoint is still intact. A half-written checkpoint is
    worse than no checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "config": asdict(cfg),
        "extra": extra or {},
    }

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and, if you pass the objects in, restore their state.

    `map_location="cpu"` by default: that way a checkpoint saved on CUDA can be loaded on a
    machine with no GPU. The model gets moved to the device afterwards anyway.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint {path} does not exist")

    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])

    return payload


def latest_checkpoint(directory: str | Path) -> Path | None:
    """The most recent checkpoint in a directory, or `None` if there is none."""
    directory = Path(directory)
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None
