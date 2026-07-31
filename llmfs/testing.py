"""Utilidades compartidas por los tests de los modulos.

El patron habitual para validar una `nn.Module` que has escrito tu:

    ref = ReferenceRMSNorm(320)
    mine = MiRMSNorm(320)
    copy_parameters(ref, mine)          # mismos pesos exactos
    assert_close(mine(x), ref(x))       # entonces la salida debe coincidir

Copiar los pesos es imprescindible: dos modulos con la misma arquitectura pero inicializados
distinto dan salidas distintas y el test no diria nada. Lo que se compara es la FUNCION que
calcula el modulo, no sus valores iniciales.
"""

from __future__ import annotations

from typing import Any

import torch


def load_exercises(test_file: str) -> Any:
    """Carga el `ejercicios.py` del modulo al que pertenece este test.

    Uso, siempre igual, al principio de cada `test_NN.py`:

        from llmfs.testing import load_exercises
        ej = load_exercises(__file__)

    A diferencia del bridge, aqui NO hay red de seguridad: los tests validan TU codigo,
    asi que si un ejercicio no esta hecho el test tiene que ponerse rojo.

    Excepcion: con `LLMFS_TEST_REFERENCE=1` se devuelve `llmfs.reference` en lugar de tus
    ejercicios. Eso hace que la suite entera valide el propio curso: si un test falla en
    ese modo, el test esta mal escrito o la referencia esta mal, y en cualquier caso es un
    bug del repo y no tuyo. Es lo que corre `make test-reference`.
    """
    import os

    if os.environ.get("LLMFS_TEST_REFERENCE", "").strip().lower() in {"1", "true", "yes"}:
        import llmfs.reference

        return llmfs.reference

    from llmfs.bridge import exercises

    modulo = exercises(test_file)
    if modulo is None:
        raise AssertionError(
            "No se ha podido importar ejercicios.py de este modulo. "
            "Mira el traceback de mas arriba: casi siempre es un error de sintaxis."
        )
    return modulo


def copy_parameters(source: torch.nn.Module, target: torch.nn.Module) -> None:
    """Copia los pesos de `source` a `target`.

    Falla con un mensaje util si los nombres o las formas no coinciden, que es la forma
    habitual de descubrir que tu modulo tiene la arquitectura mal montada.
    """
    src_state = source.state_dict()
    dst_state = target.state_dict()

    solo_ref = sorted(set(src_state) - set(dst_state))
    solo_tuyo = sorted(set(dst_state) - set(src_state))
    if solo_ref or solo_tuyo:
        raise AssertionError(
            "Los parametros no coinciden con los de la referencia.\n"
            f"  te faltan  : {solo_ref}\n"
            f"  te sobran  : {solo_tuyo}\n"
            "Revisa los nombres de los submodulos en el docstring del ejercicio."
        )

    for name, tensor in src_state.items():
        if tensor.shape != dst_state[name].shape:
            raise AssertionError(
                f"`{name}` tiene forma {tuple(dst_state[name].shape)} y deberia tener "
                f"{tuple(tensor.shape)}."
            )
    target.load_state_dict(src_state)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-6,
    what: str = "el resultado",
) -> None:
    """`torch.allclose` con un mensaje de error que sirve para depurar.

    Informa de la forma, del error maximo y de donde esta, en vez del `assert False`
    pelado que no dice nada.
    """
    actual = actual if isinstance(actual, torch.Tensor) else torch.as_tensor(actual)
    expected = expected if isinstance(expected, torch.Tensor) else torch.as_tensor(expected)

    if actual.shape != expected.shape:
        raise AssertionError(
            f"{what} tiene forma {tuple(actual.shape)} y se esperaba {tuple(expected.shape)}."
        )

    a32, e32 = actual.detach().float(), expected.detach().float()
    if torch.allclose(a32, e32, rtol=rtol, atol=atol):
        return

    diff = (a32 - e32).abs()
    idx = int(diff.argmax())
    pos = torch.unravel_index(torch.tensor(idx), diff.shape)
    pos_txt = tuple(int(p) for p in pos)
    raise AssertionError(
        f"{what} no coincide con la referencia.\n"
        f"  forma           : {tuple(actual.shape)}\n"
        f"  error maximo    : {diff.max().item():.3e} en la posicion {pos_txt}\n"
        f"  tuyo / esperado : {a32.flatten()[idx].item():.6f} / {e32.flatten()[idx].item():.6f}\n"
        f"  error medio     : {diff.mean().item():.3e}"
    )


def assert_scalar_close(
    actual: Any, expected: Any, rtol: float = 1e-5, atol: float = 1e-6, what: str = "el valor"
) -> None:
    """Como `assert_close` pero para escalares de python.

    Hace `detach()` de lo que llegue: los tests comparan perdidas que suelen venir con
    grafo de autograd colgando, y `float()` sobre esos tensores lanza un UserWarning.
    """
    a = float(actual.detach()) if isinstance(actual, torch.Tensor) else float(actual)
    e = float(expected.detach()) if isinstance(expected, torch.Tensor) else float(expected)
    if abs(a - e) <= atol + rtol * abs(e):
        return
    raise AssertionError(f"{what}: {a!r}, se esperaba {e!r} (error {abs(a - e):.3e})")


def seeded_generator(seed: int = 0, device: torch.device | str = "cpu") -> torch.Generator:
    """Generador reproducible. Los tests comparan contra referencia, asi que hace falta."""
    gen = torch.Generator(device=str(device))
    gen.manual_seed(seed)
    return gen
