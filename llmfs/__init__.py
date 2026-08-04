"""llmfs - the package behind the "LLM from scratch" course.

This is where the "production" code lives: the code that actually trains the final model.
It splits into three layers that are worth keeping straight:

1. `llmfs.reference`  -> canonical, complete and correct implementations of every piece.
                         Written to be read, and to act as a safety net.
2. `llmfs.bridge`     -> the resolver. For each piece it tries to use YOUR exercise from
                         `modules/NN_*/exercises.py`; if it is not implemented or fails the
                         smoke test, it falls back to `llmfs.reference` and tells you.
3. everything else    -> model, training, inference and evaluation, built on top of the
                         pieces the bridge hands back. In other words: once your exercise
                         from module 06 is right, the final model trains with YOUR RMSNorm.

IMPORTANT: this module does NOT import torch. The CLI has to start fast, and torch takes
about a second to load. The submodules that need it import it themselves.
"""

from __future__ import annotations

import os

# This must be set BEFORE torch is imported so that the MPS backend routes the ops it does
# not implement yet to the CPU instead of blowing up with NotImplementedError.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"

__all__ = ["__version__"]
