"""El puente entre tus ejercicios y el codigo que de verdad entrena el modelo.

Este es el mecanismo que hace que nunca te quedes bloqueado.

Cuando `llmfs/model/attention.py` necesita `MultiHeadAttention`, no la importa de ningun
sitio fijo: se la pide al bridge. El bridge entonces:

  1. Intenta cargar `modulos/05_atencion/ejercicios.py`.
  2. Busca el simbolo `MultiHeadAttention`.
  3. Le pasa un *smoke test* con tensores minusculos (ver `llmfs/probes.py`).
  4. Si pasa -> se usa TU implementacion. El modelo de 9M entrena con tu codigo.
  5. Si falla, no existe, o `ejercicios.py` ni siquiera compila -> se usa
     `llmfs.reference`, y te avisa por stderr de que pieza esta sustituyendo.

Consecuencias practicas:

- Puedes hacer los modulos en cualquier orden, o saltarte uno, y seguir entrenando.
- El aviso por stderr es deliberado: la red de seguridad no debe ser invisible. Si crees
  que estas entrenando con tu atencion y en realidad no, eso es peor que no tenerla.
- El smoke test es minimo a proposito: comprueba "esto es usable", no "esto es correcto".
  De la correccion se encargan los tests del modulo, que comparan contra referencia con
  `torch.allclose`. Una implementacion puede pasar el probe y fallar el test.

Variables de entorno:
    LLMFS_FORCE_REFERENCE=1   ignora tus ejercicios y usa siempre la referencia.
                              Lo usa la suite de tests del propio paquete.
    LLMFS_BRIDGE_VERBOSE=1    informa tambien cuando SI usa tu implementacion.
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

Source = Literal["ejercicio", "referencia"]

_impl_cache: dict[tuple[str, str], Any] = {}
_module_cache: dict[str, ModuleType | None] = {}
_announced: set[tuple[str, str, Source]] = set()


def _force_reference() -> bool:
    return os.environ.get("LLMFS_FORCE_REFERENCE", "").strip().lower() in {"1", "true", "yes"}


def _verbose() -> bool:
    return os.environ.get("LLMFS_BRIDGE_VERBOSE", "").strip().lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------- carga


def _module_ref(module_ref: str | int | Module | Path) -> Module:
    """Acepta un Module, un id, un numero, o el `__file__` de un test."""
    if isinstance(module_ref, Module):
        return module_ref
    if isinstance(module_ref, Path) or (
        isinstance(module_ref, str) and ("/" in module_ref or module_ref.endswith(".py"))
    ):
        # Viene de un test: `exercises(__file__)`. El id es el nombre del directorio.
        return get_module(Path(module_ref).resolve().parent.name)
    return get_module(module_ref)


def exercises(module_ref: str | int | Module | Path) -> ModuleType | None:
    """Carga `modulos/NN_*/ejercicios.py` y devuelve el modulo python.

    Devuelve `None` si el fichero no existe o no compila (y en ese caso escribe el
    traceback en stderr, porque un `ejercicios.py` roto es algo que quieres ver).

    Se carga con un nombre unico por modulo (`llmfs_ejercicios_05_atencion`) usando
    `importlib`, en lugar de meter directorios en `sys.path`. Esto evita el clasico
    choque de pytest cuando dos directorios distintos tienen un `ejercicios.py` cada uno.

    Uso tipico desde un test:

        from llmfs.bridge import exercises
        ej = exercises(__file__)
    """
    module = _module_ref(module_ref)
    if module.id in _module_cache:
        return _module_cache[module.id]

    path = module.exercises_file
    loaded: ModuleType | None = None
    if path.exists():
        unique_name = f"llmfs_ejercicios_{module.id}"
        try:
            spec = importlib.util.spec_from_file_location(unique_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"No se ha podido crear el spec de {path}")
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[unique_name] = loaded
            spec.loader.exec_module(loaded)
        except Exception:  # noqa: BLE001 - un ejercicios.py roto no debe tumbar el repo
            sys.modules.pop(unique_name, None)
            print(
                f"\n[llmfs] {module.id}/ejercicios.py no se ha podido importar. "
                f"Se usara la referencia para todo el modulo.\n"
                f"{'-' * 70}",
                file=sys.stderr,
            )
            traceback.print_exc()
            print("-" * 70, file=sys.stderr)
            loaded = None

    _module_cache[module.id] = loaded
    return loaded


def reference() -> ModuleType:
    """El paquete `llmfs.reference`, donde vive todo ya implementado."""
    return importlib.import_module("llmfs.reference")


# ---------------------------------------------------------------------------- resolucion


@dataclass(frozen=True)
class Resolution:
    """De donde ha salido una pieza y por que."""

    module_id: str
    name: str
    source: Source
    #: Motivo de la caida a referencia. Vacio si se ha usado el ejercicio.
    reason: str = ""


def _is_stub_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """`True` si el cuerpo es solo docstring + `raise NotImplementedError` (o `pass`)."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # descarta el docstring
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
    """Detecta un ejercicio que todavia esta en su estado de plantilla.

    Se hace leyendo el AST en lugar de llamando a la funcion, por dos motivos:

    - Llamarla no sirve: `def f(x, y): raise NotImplementedError` con argumentos falsos
      lanza `TypeError` por la firma antes de llegar al cuerpo, asi que nunca veriamos
      el `NotImplementedError`.
    - Es puro: no ejecuta codigo del usuario y no tiene efectos secundarios.

    Para una clase, basta con que UNO de sus metodos siga siendo una plantilla: una
    `nn.Module` con `forward` sin implementar no es usable, tenga el `__init__` que tenga.
    """
    try:
        source = textwrap.dedent(inspect.getsource(impl))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return False  # sin fuente disponible, asumimos que esta implementado

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
    """Comprueba que la implementacion es usable. Lanza si no lo es.

    Dos capas:
      1. Analisis estatico: si sigue siendo la plantilla, fuera. Esto funciona para
         TODOS los ejercicios sin necesidad de escribir nada.
      2. Probe especifico, si esta registrado en `llmfs/probes.py`: lo llama con datos
         minusculos y comprueba formas. Caza el caso de "escrito pero devuelve basura".
    """
    from llmfs.probes import PROBES

    if impl is None:
        raise ValueError(f"{name} es None")

    if looks_unimplemented(impl):
        raise NotImplementedError(f"{name} sigue siendo la plantilla del ejercicio")

    check = PROBES.get(name)
    if check is not None:
        check(impl)


def _resolve_uncached(module: Module, name: str) -> tuple[Any, Resolution]:
    ref = reference()

    def use_reference(reason: str) -> tuple[Any, Resolution]:
        if not hasattr(ref, name):
            raise AttributeError(
                f"BUG del curso: `{name}` no existe en llmfs.reference. "
                f"La red de seguridad tiene un agujero; abre una incidencia."
            )
        return getattr(ref, name), Resolution(module.id, name, "referencia", reason)

    if _force_reference():
        return use_reference("LLMFS_FORCE_REFERENCE=1")

    user_module = exercises(module)
    if user_module is None:
        return use_reference("ejercicios.py no existe o no compila")

    impl = getattr(user_module, name, None)
    if impl is None:
        return use_reference(f"`{name}` no esta definido en ejercicios.py")

    try:
        _probe(name, impl)
    except NotImplementedError:
        return use_reference("todavia sin implementar (NotImplementedError)")
    except Exception as exc:  # noqa: BLE001
        return use_reference(f"{exc.__class__.__name__}: {exc}")

    return impl, Resolution(module.id, name, "ejercicio")


def _announce(res: Resolution) -> None:
    key = (res.module_id, res.name, res.source)
    if key in _announced:
        return
    _announced.add(key)
    if res.source == "referencia":
        print(
            f"[llmfs] {res.module_id}: usando la REFERENCIA para `{res.name}` "
            f"({res.reason}).",
            file=sys.stderr,
        )
    elif _verbose():
        print(f"[llmfs] {res.module_id}: usando TU `{res.name}`.", file=sys.stderr)


def resolve(module_ref: str | int | Module, name: str) -> Any:
    """Devuelve la implementacion usable de `name`: la tuya si vale, si no la de referencia.

    Args:
        module_ref: modulo del curriculo que define el ejercicio (`"05_atencion"`, `5`...).
        name: nombre del simbolo, tal y como aparece en `curriculum.py`.

    Returns:
        La funcion o clase lista para usar.
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
    """Como `resolve`, pero devuelve el diagnostico en lugar de la implementacion.

    Lo usa `llmfs status` para pintar que piezas del modelo son tuyas.
    """
    module = _module_ref(module_ref)
    _, res = _resolve_uncached(module, name)
    return res


def module_resolutions(module_ref: str | int | Module) -> list[Resolution]:
    """Diagnostico de todos los ejercicios de un modulo."""
    module = _module_ref(module_ref)
    return [resolution(module, ex.name) for ex in module.exercises]


def clear_cache() -> None:
    """Olvida lo resuelto. Necesario en tests que reescriben `ejercicios.py`."""
    _impl_cache.clear()
    _module_cache.clear()
    _announced.clear()
    for key in [k for k in sys.modules if k.startswith("llmfs_ejercicios_")]:
        del sys.modules[key]
