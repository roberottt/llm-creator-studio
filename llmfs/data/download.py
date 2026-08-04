"""Corpus downloads, with an on-disk cache and a plan B when there is no internet.

Everything that gets downloaded is stored in `data/` (which is in .gitignore) and never
requested again. If there is no network, the functions return a small fallback text instead
of blowing up: a course where module 00 fails because you are on a train is a bad course.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from llmfs.paths import data_dir

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

#: Fallback text for when there is no connection. It is original, written for the course,
#: and deliberately repetitive: with only a few kilobytes, a count-based model needs
#: regularity for the difference between 1-character and 4-character contexts to show.
FALLBACK_TEXT = """
The cat sleeps on the roof. The cat comes down from the roof when it is hungry.
The girl watches the cat from the window and the cat watches the girl from the roof.
The dog barks in the yard. The dog barks when the cat comes down from the roof.
The girl opens the window and calls the cat. The cat does not come because the dog barks.
The dog gets tired of barking and falls asleep in the yard. Then the cat comes down.
The cat comes in through the window and the girl gives it food. The cat eats and purrs.
Night comes and the girl closes the window. The cat sleeps inside the house.
The dog sleeps in the yard. The whole house sleeps until the morning comes.
In the morning the cat wants to go out. The girl opens the window and the cat climbs
to the roof.
The dog wakes up and barks at the cat. The cat watches the dog from the roof and does not
come down.
""".strip()


def _download_text(url: str, destination: Path, timeout: int = 30) -> str | None:
    """Download a text file. Returns `None` if it cannot."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(raw, encoding="utf-8")
    return raw


def fetch_tinyshakespeare(quiet: bool = False) -> tuple[str, str]:
    """Return the tiny-shakespeare text, downloading it the first time.

    It is ~1.1 MB of Shakespeare's works concatenated. It is the "hello world" of
    character-level language models: small enough to train on in seconds and structured
    enough that you can tell when the model has learned something.

    Returns:
        `(text, origin)` where origin is `"cache"`, `"download"` or `"fallback"`.
        Checking the origin lets the demos warn that they are running on plan B.
    """
    target = data_dir() / "tinyshakespeare.txt"

    if target.exists():
        return target.read_text(encoding="utf-8"), "cache"

    if not quiet:
        print(f"[llmfs] downloading tiny-shakespeare (~1 MB) to {target}...")

    text = _download_text(TINYSHAKESPEARE_URL, target)
    if text is not None:
        return text, "download"

    if not quiet:
        print("[llmfs] no connection: using the much smaller fallback text.")
    return FALLBACK_TEXT, "fallback"
