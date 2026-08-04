"""Locating the repo root and its directories.

This exists because `modules/`, `configs/` and `data/` are resolved relative to the repo
root, not to the directory you run the command from. That way `python -m llmfs status`
works the same from the root as from `modules/06_attention/`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_MARKERS = ("pyproject.toml", ".git")


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Repository root.

    Search order: `$LLMFS_ROOT` -> ancestor of this file containing `pyproject.toml` ->
    ancestor of the cwd. With an editable install (`uv sync`) the first search already
    hits, because the package lives inside the repo.
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
    return repo_root() / "modules"


def configs_dir() -> Path:
    return repo_root() / "configs"


def data_dir() -> Path:
    return repo_root() / "data"


def runs_dir() -> Path:
    return repo_root() / "runs"


def figures_dir() -> Path:
    """Where the demos write their PNGs. It is in .gitignore."""
    path = runs_dir() / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def progress_file() -> Path:
    return repo_root() / ".llmfs_progress.json"
