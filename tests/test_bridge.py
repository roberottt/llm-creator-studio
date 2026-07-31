"""La red de seguridad tiene que funcionar de verdad.

Si estos tests fallan, el curso pierde su promesa central: que nunca te quedas bloqueado
porque un ejercicio anterior este a medias.
"""

from __future__ import annotations

import pytest

import llmfs.curriculum as curriculum_mod
import llmfs.reference as reference_mod
from llmfs import bridge
from llmfs.curriculum import Exercise, Module

# Se reutiliza la identidad del modulo 05 (numero y slug reales) pero apuntando a un
# directorio temporal. Asi tambien se ejercita la resolucion por ruta, que necesita que
# el id exista en el curriculo.
NUMBER, SLUG = 6, "atencion"
MODULE_ID = f"{NUMBER:02d}_{SLUG}"


@pytest.fixture(autouse=True)
def limpia_cache():
    bridge.clear_cache()
    yield
    bridge.clear_cache()


@pytest.fixture
def modulo(tmp_path, monkeypatch):
    """Un modulo sintetico en tmp_path, con `suma` como unico ejercicio."""
    monkeypatch.setattr(curriculum_mod, "modules_dir", lambda: tmp_path)
    (tmp_path / MODULE_ID).mkdir()

    # La pieza de referencia: siempre correcta, siempre disponible.
    monkeypatch.setattr(reference_mod, "suma", lambda a, b: a + b, raising=False)

    return Module(
        number=NUMBER,
        slug=SLUG,
        title="Modulo de prueba",
        part="test",
        summary="solo para los tests del bridge",
        est_minutes=1,
        exercises=(Exercise("suma", "sumar dos numeros"),),
    )


def escribe_ejercicios(modulo: Module, codigo: str) -> None:
    modulo.exercises_file.write_text(codigo, encoding="utf-8")
    bridge.clear_cache()


# ---------------------------------------------------------------------------- casos


def test_usa_tu_implementacion_cuando_funciona(modulo):
    escribe_ejercicios(modulo, "def suma(a, b):\n    return a + b\n")

    assert bridge.resolve(modulo, "suma")(2, 3) == 5
    assert bridge.resolution(modulo, "suma").source == "ejercicio"


def test_cae_a_referencia_si_el_ejercicio_lanza_not_implemented(modulo):
    escribe_ejercicios(
        modulo,
        "def suma(a, b):\n    raise NotImplementedError('TODO: modulo 99')\n",
    )

    res = bridge.resolution(modulo, "suma")
    assert res.source == "referencia"
    assert "NotImplementedError" in res.reason
    # y aun asi el codigo de produccion sigue funcionando
    assert bridge.resolve(modulo, "suma")(2, 3) == 5


def test_cae_a_referencia_si_ejercicios_py_no_existe(modulo):
    res = bridge.resolution(modulo, "suma")
    assert res.source == "referencia"
    assert bridge.resolve(modulo, "suma")(1, 1) == 2


def test_cae_a_referencia_si_ejercicios_py_no_compila(modulo, capsys):
    escribe_ejercicios(modulo, "def suma(a, b)\n    return a + b\n")  # falta el ':'

    res = bridge.resolution(modulo, "suma")
    assert res.source == "referencia"
    assert bridge.resolve(modulo, "suma")(4, 5) == 9

    # Un fichero que no compila es algo que el usuario TIENE que ver.
    salida = capsys.readouterr()
    assert "no se ha podido importar" in salida.err


def test_cae_a_referencia_si_falta_el_simbolo(modulo):
    escribe_ejercicios(modulo, "def otra_cosa():\n    return 1\n")

    res = bridge.resolution(modulo, "suma")
    assert res.source == "referencia"
    assert "no esta definido" in res.reason


def test_el_probe_descarta_implementaciones_que_devuelven_basura(modulo, monkeypatch):
    """Un ejercicio puede estar 'escrito' y aun asi no ser usable."""
    from llmfs.probes import PROBES

    def comprueba(impl):
        if impl(2, 3) != 5:
            raise ValueError("2 + 3 deberia dar 5")

    monkeypatch.setitem(PROBES, "suma", comprueba)
    escribe_ejercicios(modulo, "def suma(a, b):\n    return None\n")

    res = bridge.resolution(modulo, "suma")
    assert res.source == "referencia"
    assert "ValueError" in res.reason


def test_force_reference_ignora_tu_codigo_aunque_sea_correcto(modulo, monkeypatch):
    escribe_ejercicios(modulo, "def suma(a, b):\n    return a + b\n")
    monkeypatch.setenv("LLMFS_FORCE_REFERENCE", "1")
    bridge.clear_cache()

    res = bridge.resolution(modulo, "suma")
    assert res.source == "referencia"
    assert "LLMFS_FORCE_REFERENCE" in res.reason


def test_avisa_por_stderr_una_sola_vez_al_usar_referencia(modulo, capsys):
    for _ in range(3):
        bridge.resolve(modulo, "suma")

    err = capsys.readouterr().err
    assert err.count("usando la REFERENCIA para `suma`") == 1, (
        "el aviso debe salir, pero no debe inundar la consola"
    )


def test_si_falta_la_pieza_de_referencia_el_error_es_explicito(modulo, monkeypatch):
    monkeypatch.delattr(reference_mod, "suma")
    with pytest.raises(AttributeError, match="BUG del curso"):
        bridge.resolve(modulo, "suma")


def test_exercises_acepta_la_ruta_de_un_test(modulo):
    """Los `test_NN.py` se identifican con `exercises(__file__)`."""
    escribe_ejercicios(modulo, "MARCA = 42\n")
    cargado = bridge.exercises(str(modulo.path / f"test_{NUMBER:02d}.py"))
    assert cargado is not None and cargado.MARCA == 42


def test_una_clase_con_forward_sin_implementar_no_es_usable(modulo):
    """El caso real: una `nn.Module` con el `__init__` hecho y el `forward` a medias."""
    escribe_ejercicios(
        modulo,
        "class suma:\n"
        "    def __init__(self, n):\n"
        "        self.n = n\n"
        "    def forward(self, x):\n"
        "        raise NotImplementedError('TODO')\n",
    )
    assert bridge.resolution(modulo, "suma").source == "referencia"


def test_un_cuerpo_con_solo_pass_tambien_cuenta_como_plantilla(modulo):
    escribe_ejercicios(modulo, "def suma(a, b):\n    pass\n")
    assert bridge.resolution(modulo, "suma").source == "referencia"


def test_un_docstring_largo_no_confunde_al_detector(modulo):
    escribe_ejercicios(
        modulo,
        'def suma(a, b):\n'
        '    """Suma dos numeros.\n\n'
        '    Args:\n        a: primero\n        b: segundo\n\n'
        '    Returns:\n        a + b\n    """\n'
        '    return a + b\n',
    )
    assert bridge.resolution(modulo, "suma").source == "ejercicio"


def test_module_resolutions_cubre_todos_los_ejercicios(modulo):
    resoluciones = bridge.module_resolutions(modulo)
    assert [r.name for r in resoluciones] == ["suma"]
