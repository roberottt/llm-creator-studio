"""Curriculum state: runs each module's tests and persists the result.

The state is not declared by hand anywhere: it is computed by running the tests. A module
is green when its tests pass, full stop. The `.llmfs_progress.json` file is only a cache so
that `llmfs status` can answer instantly without running the whole suite again.

The tests run in the same process (`pytest.main`) instead of one subprocess per module.
Reason: importing torch costs ~1-2 s and 17 subprocesses would be half a minute of waiting
on imports alone. The price is having to clear the bridge state between modules, which is
what `bridge.clear_cache()` does.
"""

from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from llmfs.curriculum import Module, all_modules, get_module
from llmfs.paths import progress_file

State = Literal["done", "in_progress", "todo", "no_tests"]

ICONS: dict[State, str] = {
    "done": "✅",
    "in_progress": "\U0001f527",
    "todo": "⬜",
    "no_tests": "⬜",
}

LABELS: dict[State, str] = {
    "done": "complete",
    "in_progress": "in progress",
    "todo": "not started",
    "no_tests": "no tests",
}

PROGRESS_VERSION = 2


@dataclass
class ModuleStatus:
    """Result of running a module's tests."""

    module_id: str
    state: State = "no_tests"
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    total: int = 0
    duration_s: float = 0.0
    updated_at: str = ""
    #: Exercises solved with YOUR code, according to the bridge.
    mine: list[str] = field(default_factory=list)
    #: Exercises that currently fall back to the reference.
    borrowed: list[str] = field(default_factory=list)
    #: First failure, summarized, so `status` can give a hint without opening the log.
    first_failure: str = ""

    @property
    def icon(self) -> str:
        return ICONS[self.state]

    @property
    def label(self) -> str:
        return LABELS[self.state]

    @property
    def ratio(self) -> str:
        if self.total == 0:
            return "-"
        return f"{self.passed}/{self.total}"


class _Collector:
    """pytest plugin that collects the outcome of each test."""

    def __init__(self) -> None:
        self.outcomes: dict[str, str] = {}
        self.first_failure: str = ""

    def pytest_runtest_logreport(self, report: Any) -> None:
        nodeid = report.nodeid
        if report.when == "call":
            self.outcomes[nodeid] = report.outcome
        elif report.when in {"setup", "teardown"} and report.outcome == "failed":
            # An error during setup counts as a test failure.
            self.outcomes[nodeid] = "failed"
        elif report.when == "setup" and report.outcome == "skipped":
            self.outcomes.setdefault(nodeid, "skipped")

        if report.outcome == "failed" and not self.first_failure:
            self.first_failure = _summarize_failure(report)


def _summarize_failure(report: Any) -> str:
    """Extract the useful line from a pytest failure (the `assert` or the exception)."""
    text = str(getattr(report, "longrepr", "") or "")
    interesting: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("E "):
            interesting.append(stripped[2:].strip())
    if interesting:
        return interesting[0][:200]
    node = report.nodeid.split("::")[-1]
    return f"{node} failed"


def _bridge_split(module: Module) -> tuple[list[str], list[str]]:
    """Which of the module's exercises use your code and which use the reference."""
    from llmfs import bridge

    mine: list[str] = []
    borrowed: list[str] = []
    for ex in module.exercises:
        try:
            res = bridge.resolution(module, ex.name)
        except Exception:  # noqa: BLE001 - the reference may not exist yet
            borrowed.append(ex.name)
            continue
        (mine if res.source == "exercise" else borrowed).append(ex.name)
    return mine, borrowed


def run_module_tests(module: Module | str | int, quiet: bool = True) -> ModuleStatus:
    """Run a module's tests and return its state.

    Args:
        module: the module, or a reference to it.
        quiet: if `True`, swallow pytest's output. `llmfs check` wants to see it.
    """
    import pytest

    from llmfs import bridge

    module = module if isinstance(module, Module) else get_module(module)
    status = ModuleStatus(module_id=module.id, updated_at=_now())

    if not module.test_file.exists():
        status.state = "no_tests"
        return status

    bridge.clear_cache()
    collector = _Collector()
    args = [
        str(module.test_file),
        "-q",
        "--tb=no" if quiet else "--tb=short",
        "-p",
        "no:cacheprovider",
        "--no-header",
    ]

    started = time.perf_counter()
    buffer_out, buffer_err = io.StringIO(), io.StringIO()
    try:
        if quiet:
            with redirect_stdout(buffer_out), redirect_stderr(buffer_err):
                pytest.main(args, plugins=[collector])
        else:
            pytest.main(args, plugins=[collector])
    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001 - a broken module must not take down `status`
        status.state = "in_progress"
        status.first_failure = f"pytest blew up: {exc.__class__.__name__}: {exc}"
        return status
    status.duration_s = time.perf_counter() - started

    outcomes = list(collector.outcomes.values())
    status.total = len(outcomes)
    status.passed = sum(1 for o in outcomes if o == "passed")
    status.failed = sum(1 for o in outcomes if o == "failed")
    status.skipped = sum(1 for o in outcomes if o == "skipped")
    status.first_failure = collector.first_failure

    if status.total == 0:
        status.state = "no_tests"
    elif status.failed == 0 and status.passed > 0:
        status.state = "done"
    elif status.passed > 0:
        status.state = "in_progress"
    else:
        status.state = "todo"

    bridge.clear_cache()
    status.mine, status.borrowed = _bridge_split(module)
    # If you have implemented something but the tests do not pass yet, that is "in
    # progress", not "not started". Useful: it separates "untouched" from "trying".
    if status.state == "todo" and status.mine:
        status.state = "in_progress"

    bridge.clear_cache()
    return status


def run_all(modules: list[Module] | None = None, on_module: Any = None) -> dict[str, ModuleStatus]:
    """Run the tests of every module (or of the given ones).

    Args:
        on_module: optional callback `fn(module)` before each module, to render progress
            while it runs.
    """
    modules = modules if modules is not None else list(all_modules())
    results: dict[str, ModuleStatus] = {}
    for module in modules:
        if on_module is not None:
            on_module(module)
        results[module.id] = run_module_tests(module, quiet=True)
    return results


# ---------------------------------------------------------------------------- persistence


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(results: dict[str, ModuleStatus]) -> None:
    """Merge the results with what is already stored and write `.llmfs_progress.json`."""
    stored = _read_raw()
    modules = stored.get("modules", {}) if stored.get("version") == PROGRESS_VERSION else {}
    for module_id, status in results.items():
        modules[module_id] = asdict(status)
    payload = {
        "version": PROGRESS_VERSION,
        "updated_at": _now(),
        "modules": modules,
    }
    progress_file().write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_raw() -> dict[str, Any]:
    path = progress_file()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load() -> dict[str, ModuleStatus]:
    """Read the progress cache. Returns `{}` if it does not exist or is another version."""
    raw = _read_raw()
    if raw.get("version") != PROGRESS_VERSION:
        return {}
    out: dict[str, ModuleStatus] = {}
    for module_id, data in (raw.get("modules") or {}).items():
        try:
            out[module_id] = ModuleStatus(**data)
        except TypeError:
            continue
    return out


def last_updated() -> str:
    return _read_raw().get("updated_at", "")


# ---------------------------------------------------------------------------- queries


def next_module(results: dict[str, ModuleStatus]) -> Module | None:
    """The first module of the curriculum that is not complete."""
    for module in all_modules():
        status = results.get(module.id)
        if status is None or status.state != "done":
            return module
    return None


def get_hint_level(module_id: str, exercise: str) -> int:
    """Last hint level shown for an exercise (0 if none)."""
    raw = _read_raw()
    return int((raw.get("hints") or {}).get(f"{module_id}:{exercise}", 0))


def set_hint_level(module_id: str, exercise: str, level: int) -> None:
    """Remember which hint level you reached, so the next one is more explicit."""
    raw = _read_raw()
    if raw.get("version") != PROGRESS_VERSION:
        raw = {"version": PROGRESS_VERSION, "modules": {}}
    hints = raw.setdefault("hints", {})
    hints[f"{module_id}:{exercise}"] = level
    raw["updated_at"] = _now()
    progress_file().write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")


def summary_counts(results: dict[str, ModuleStatus]) -> dict[State, int]:
    counts: dict[State, int] = {"done": 0, "in_progress": 0, "todo": 0, "no_tests": 0}
    for module in all_modules():
        status = results.get(module.id)
        counts[status.state if status else "no_tests"] += 1
    return counts
