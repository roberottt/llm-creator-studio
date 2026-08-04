"""The bridge between your exercises and the code that actually trains the model.

This is the mechanism that makes sure you never get stuck.

When `llmfs/model/attention.py` needs `MultiHeadAttention`, it does not import it from any
fixed place: it asks the bridge for it. The bridge then:

  1. Tries to load `modules/06_attention/exercises.py`.
  2. Looks for the symbol `MultiHeadAttention`.
  3. Runs a *smoke test* on it with tiny tensors (see `llmfs/probes.py`).
  4. If it passes -> YOUR implementation is used. The 9M model trains with your code.
  5. If it fails, does not exist, or `exercises.py` does not even compile -> it uses
     `llmfs.reference`, and tells you on stderr which piece it is substituting.

Practical consequences:

- You can do the modules in any order, or skip one, and keep training.
- The stderr warning is deliberate: the safety net must not be invisible. Believing you
  are training with your own attention when you are not is worse than having no net.
- The smoke test is minimal on purpose: it checks "this is usable", not "this is correct".
  Correctness is the job of the module tests, which compare against the reference with
  `torch.allclose`. An implementation can pass the probe and still fail the test.

Environment variables:
    LLMFS_FORCE_REFERENCE=1   ignore your exercises and always use the reference.
                              The package's own test suite uses this.
    LLMFS_BRIDGE_VERBOSE=1    also report when it DOES use your implementation.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import os
import sys
import textwrap
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from llmfs.curriculum import Module, get_module

Source = Literal["exercise", "reference"]

_impl_cache: dict[tuple[str, str], Any] = {}
_module_cache: dict[str, ModuleType | None] = {}
_announced: set[tuple[str, str, Source]] = set()


def _force_reference() -> bool:
    return os.environ.get("LLMFS_FORCE_REFERENCE", "").strip().lower() in {"1", "true", "yes"}


def _verbose() -> bool:
    return os.environ.get("LLMFS_BRIDGE_VERBOSE", "").strip().lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------- loading


def _module_ref(module_ref: str | int | Module | Path) -> Module:
    """Accepts a Module, an id, a number, or the `__file__` of a test."""
    if isinstance(module_ref, Module):
        return module_ref
    if isinstance(module_ref, Path) or (
        isinstance(module_ref, str) and ("/" in module_ref or module_ref.endswith(".py"))
    ):
        # Comes from a test: `exercises(__file__)`. The id is the directory name.
        return get_module(Path(module_ref).resolve().parent.name)
    return get_module(module_ref)


def exercises(module_ref: str | int | Module | Path) -> ModuleType | None:
    """Load `modules/NN_*/exercises.py` and return the python module.

    Returns `None` if the file does not exist or does not compile (and in that case writes
    the traceback to stderr, because a broken `exercises.py` is something you want to see).

    It is loaded under a unique per-module name (`llmfs_exercises_06_attention`) using
    `importlib`, instead of pushing directories onto `sys.path`. This avoids the classic
    pytest clash when two different directories each hold an `exercises.py`.

    Typical use from a test:

        from llmfs.bridge import exercises
        ex = exercises(__file__)
    """
    module = _module_ref(module_ref)
    if module.id in _module_cache:
        return _module_cache[module.id]

    path = module.exercises_file
    loaded: ModuleType | None = None
    if path.exists():
        unique_name = f"llmfs_exercises_{module.id}"
        try:
            spec = importlib.util.spec_from_file_location(unique_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create the spec for {path}")
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[unique_name] = loaded
            spec.loader.exec_module(loaded)
        except Exception:  # noqa: BLE001 - a broken exercises.py must not take down the repo
            sys.modules.pop(unique_name, None)
            print(
                f"\n[llmfs] {module.id}/exercises.py could not be imported. "
                f"The reference will be used for the whole module.\n"
                f"{'-' * 70}",
                file=sys.stderr,
            )
            traceback.print_exc()
            print("-" * 70, file=sys.stderr)
            loaded = None

    _module_cache[module.id] = loaded
    return loaded


def reference() -> ModuleType:
    """The `llmfs.reference` package, where everything already implemented lives."""
    return importlib.import_module("llmfs.reference")


# ---------------------------------------------------------------------------- resolution


@dataclass(frozen=True)
class Resolution:
    """Where a piece came from, and why."""

    module_id: str
    name: str
    source: Source
    #: Reason for falling back to the reference. Empty if the exercise was used.
    reason: str = ""


def _is_stub_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """`True` if the body is only a docstring + `raise NotImplementedError` (or `pass`)."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # drop the docstring
    if len(body) != 1:
        return False
    stmt = body[0]
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Raise):
        exc = stmt.exc
        if isinstance(exc, ast.Call):
            exc = exc.func
        return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"
    return False


def looks_unimplemented(impl: Any) -> bool:
    """Detect an exercise that is still in its template state.

    This is done by reading the AST rather than calling the function, for two reasons:

    - Calling it does not work: `def f(x, y): raise NotImplementedError` with fake
      arguments raises `TypeError` on the signature before ever reaching the body, so we
      would never see the `NotImplementedError`.
    - It is pure: it does not execute user code and has no side effects.

    For a class, it is enough that ONE of its methods is still a template: an `nn.Module`
    with an unimplemented `forward` is not usable, whatever its `__init__` looks like.
    """
    try:
        source = textwrap.dedent(inspect.getsource(impl))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return False  # with no source available, assume it is implemented

    if not tree.body:
        return False
    node = tree.body[0]

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _is_stub_body(node)

    if isinstance(node, ast.ClassDef):
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        return any(_is_stub_body(m) for m in methods)

    return False


def _probe(name: str, impl: Any) -> None:
    """Check that the implementation is usable. Raises if it is not.

    Two layers:
      1. Static analysis: if it is still the template, out it goes. This works for ALL
         exercises without having to write anything.
      2. A specific probe, if one is registered in `llmfs/probes.py`: it calls the
         implementation with tiny data and checks shapes. This catches the case of
         "written, but returns garbage".
    """
    from llmfs.probes import PROBES

    if impl is None:
        raise ValueError(f"{name} is None")

    if looks_unimplemented(impl):
        raise NotImplementedError(f"{name} is still the exercise template")

    check = PROBES.get(name)
    if check is not None:
        check(impl)


def _resolve_uncached(module: Module, name: str) -> tuple[Any, Resolution]:
    ref = reference()

    def use_reference(reason: str) -> tuple[Any, Resolution]:
        if not hasattr(ref, name):
            raise AttributeError(
                f"COURSE BUG: `{name}` does not exist in llmfs.reference. "
                f"The safety net has a hole in it; please open an issue."
            )
        return getattr(ref, name), Resolution(module.id, name, "reference", reason)

    if _force_reference():
        return use_reference("LLMFS_FORCE_REFERENCE=1")

    user_module = exercises(module)
    if user_module is None:
        return use_reference("exercises.py does not exist or does not compile")

    impl = getattr(user_module, name, None)
    if impl is None:
        return use_reference(f"`{name}` is not defined in exercises.py")

    try:
        _probe(name, impl)
    except NotImplementedError:
        return use_reference("not implemented yet (NotImplementedError)")
    except Exception as exc:  # noqa: BLE001
        return use_reference(f"{exc.__class__.__name__}: {exc}")

    return impl, Resolution(module.id, name, "exercise")


def _announce(res: Resolution) -> None:
    key = (res.module_id, res.name, res.source)
    if key in _announced:
        return
    _announced.add(key)
    if res.source == "reference":
        print(
            f"[llmfs] {res.module_id}: using the REFERENCE for `{res.name}` "
            f"({res.reason}).",
            file=sys.stderr,
        )
    elif _verbose():
        print(f"[llmfs] {res.module_id}: using YOUR `{res.name}`.", file=sys.stderr)


def resolve(module_ref: str | int | Module, name: str) -> Any:
    """Return the usable implementation of `name`: yours if it works, the reference if not.

    Args:
        module_ref: curriculum module that defines the exercise (`"06_attention"`, `6`...).
        name: name of the symbol, exactly as it appears in `curriculum.py`.

    Returns:
        The function or class, ready to use.
    """
    module = _module_ref(module_ref)
    key = (module.id, name)
    if key in _impl_cache:
        return _impl_cache[key]

    impl, res = _resolve_uncached(module, name)
    _impl_cache[key] = impl
    _announce(res)
    return impl


def resolution(module_ref: str | int | Module, name: str) -> Resolution:
    """Like `resolve`, but returns the diagnosis instead of the implementation.

    `llmfs status` uses this to show which pieces of the model are yours.
    """
    module = _module_ref(module_ref)
    _, res = _resolve_uncached(module, name)
    return res


def module_resolutions(module_ref: str | int | Module) -> list[Resolution]:
    """Diagnosis for every exercise in a module."""
    module = _module_ref(module_ref)
    return [resolution(module, ex.name) for ex in module.exercises]


def clear_cache() -> None:
    """Forget what has been resolved. Needed in tests that rewrite `exercises.py`."""
    _impl_cache.clear()
    _module_cache.clear()
    _announced.clear()
    for key in [k for k in sys.modules if k.startswith("llmfs_exercises_")]:
        del sys.modules[key]
