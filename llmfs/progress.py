"""Estado del curriculo: ejecuta los tests de cada modulo y lo persiste.

El estado no se declara a mano en ningun sitio: se calcula ejecutando los tests. Un
modulo esta verde cuando sus tests pasan, y punto. El fichero `.llmfs_progress.json` es
solo una cache para que `llmfs status` pueda responder al instante sin volver a correr
toda la suite.

Los tests se ejecutan en el mismo proceso (`pytest.main`) en lugar de con un subproceso
por modulo. Motivo: importar torch cuesta ~1-2 s y 17 subprocesos serian medio minuto de
espera solo en imports. A cambio hay que limpiar el estado del bridge entre modulos, que
es lo que hace `bridge.clear_cache()`.
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
    "done": "completo",
    "in_progress": "en progreso",
    "todo": "sin empezar",
    "no_tests": "sin tests",
}

PROGRESS_VERSION = 2


@dataclass
class ModuleStatus:
    """Resultado de correr los tests de un modulo."""

    module_id: str
    state: State = "no_tests"
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    total: int = 0
    duration_s: float = 0.0
    updated_at: str = ""
    #: Ejercicios resueltos con TU codigo, segun el bridge.
    mine: list[str] = field(default_factory=list)
    #: Ejercicios que ahora mismo caen a la referencia.
    borrowed: list[str] = field(default_factory=list)
    #: Primer fallo, resumido, para que `status` pueda dar una pista sin abrir el log.
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
    """Plugin de pytest que recoge el resultado de cada test."""

    def __init__(self) -> None:
        self.outcomes: dict[str, str] = {}
        self.first_failure: str = ""

    def pytest_runtest_logreport(self, report: Any) -> None:
        nodeid = report.nodeid
        if report.when == "call":
            self.outcomes[nodeid] = report.outcome
        elif report.when in {"setup", "teardown"} and report.outcome == "failed":
            # Un error en setup cuenta como fallo del test.
            self.outcomes[nodeid] = "failed"
        elif report.when == "setup" and report.outcome == "skipped":
            self.outcomes.setdefault(nodeid, "skipped")

        if report.outcome == "failed" and not self.first_failure:
            self.first_failure = _summarize_failure(report)


def _summarize_failure(report: Any) -> str:
    """Extrae la linea util de un fallo de pytest (el `assert` o la excepcion)."""
    text = str(getattr(report, "longrepr", "") or "")
    interesting: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("E "):
            interesting.append(stripped[2:].strip())
    if interesting:
        return interesting[0][:200]
    node = report.nodeid.split("::")[-1]
    return f"{node} ha fallado"


def _bridge_split(module: Module) -> tuple[list[str], list[str]]:
    """Que ejercicios del modulo usan tu codigo y cuales la referencia."""
    from llmfs import bridge

    mine: list[str] = []
    borrowed: list[str] = []
    for ex in module.exercises:
        try:
            res = bridge.resolution(module, ex.name)
        except Exception:  # noqa: BLE001 - la referencia puede no existir todavia
            borrowed.append(ex.name)
            continue
        (mine if res.source == "ejercicio" else borrowed).append(ex.name)
    return mine, borrowed


def run_module_tests(module: Module | str | int, quiet: bool = True) -> ModuleStatus:
    """Ejecuta los tests de un modulo y devuelve su estado.

    Args:
        module: modulo o referencia a el.
        quiet: si `True`, se traga la salida de pytest. `llmfs check` la quiere ver.
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
    except Exception as exc:  # noqa: BLE001 - un modulo roto no debe tumbar `status`
        status.state = "in_progress"
        status.first_failure = f"pytest ha reventado: {exc.__class__.__name__}: {exc}"
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
    # Si has implementado algo pero los tests aun no pasan, eso es "en progreso",
    # no "sin empezar". Es informacion util: distingue "no lo he tocado" de "lo intento".
    if status.state == "todo" and status.mine:
        status.state = "in_progress"

    bridge.clear_cache()
    return status


def run_all(modules: list[Module] | None = None, on_module: Any = None) -> dict[str, ModuleStatus]:
    """Corre los tests de todos los modulos (o de los indicados).

    Args:
        on_module: callback opcional `fn(module)` antes de cada modulo, para pintar
            progreso mientras se ejecuta.
    """
    modules = modules if modules is not None else list(all_modules())
    results: dict[str, ModuleStatus] = {}
    for module in modules:
        if on_module is not None:
            on_module(module)
        results[module.id] = run_module_tests(module, quiet=True)
    return results


# ---------------------------------------------------------------------------- persistencia


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(results: dict[str, ModuleStatus]) -> None:
    """Fusiona los resultados con lo ya guardado y escribe `.llmfs_progress.json`."""
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
    """Lee la cache de progreso. Devuelve `{}` si no existe o es de otra version."""
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


# ---------------------------------------------------------------------------- consultas


def next_module(results: dict[str, ModuleStatus]) -> Module | None:
    """El primer modulo del curriculo que no esta completo."""
    for module in all_modules():
        status = results.get(module.id)
        if status is None or status.state != "done":
            return module
    return None


def get_hint_level(module_id: str, exercise: str) -> int:
    """Ultimo nivel de pista mostrado para un ejercicio (0 si ninguno)."""
    raw = _read_raw()
    return int((raw.get("hints") or {}).get(f"{module_id}:{exercise}", 0))


def set_hint_level(module_id: str, exercise: str, level: int) -> None:
    """Recuerda hasta que nivel de pista has llegado, para que la siguiente sea mas explicita."""
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
