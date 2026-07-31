"""El curriculo tiene que ser internamente consistente.

Estos tests no validan tu aprendizaje: validan que el repo no esta roto. Si alguno falla
es un bug del curso, no un ejercicio pendiente.
"""

from __future__ import annotations

import pytest

from llmfs.curriculum import CURRICULUM, all_modules, get_module, parts, total_exercises


def test_hay_18_modulos_numerados_del_0_al_17():
    numbers = [m.number for m in CURRICULUM]
    assert numbers == list(range(18)), f"numeracion con huecos o desordenada: {numbers}"


def test_los_ids_son_unicos():
    ids = [m.id for m in CURRICULUM]
    assert len(ids) == len(set(ids))


def test_los_nombres_de_ejercicio_son_unicos_en_todo_el_curso():
    """`llmfs.reference` reexporta todos los simbolos por nombre plano.

    Si dos modulos definieran un ejercicio con el mismo nombre, el bridge resolveria
    ambos a la misma pieza de referencia sin avisar. Mejor que reviente aqui.
    """
    seen: dict[str, str] = {}
    for module in CURRICULUM:
        for ex in module.exercises:
            assert ex.name not in seen, (
                f"`{ex.name}` esta en {module.id} y tambien en {seen[ex.name]}"
            )
            seen[ex.name] = module.id


def test_cada_modulo_tiene_titulo_resumen_y_estimacion():
    for module in CURRICULUM:
        assert module.title.strip()
        assert module.summary.strip()
        assert module.est_minutes > 0, f"{module.id} sin estimacion de tiempo"


def test_todos_los_modulos_tienen_ejercicios():
    for module in CURRICULUM:
        assert module.exercises, f"{module.id} no define ningun ejercicio"


def test_el_total_de_ejercicios_cuadra():
    assert total_exercises() == sum(len(m.exercises) for m in all_modules())


@pytest.mark.parametrize("ref", [6, "6", "06", "06_atencion", "aten"])
def test_get_module_acepta_varias_formas(ref):
    assert get_module(ref).id == "06_atencion"


def test_get_module_falla_con_mensaje_util():
    with pytest.raises(KeyError) as exc:
        get_module("no_existe")
    assert "06_atencion" in str(exc.value), "el error deberia listar los modulos validos"


def test_las_partes_estan_en_orden():
    assert list(parts()) == [
        "0 - Antes de empezar",
        "I - Fundamentos",
        "II - Arquitectura",
        "III - Entrenamiento",
        "IV - Uso y evaluacion",
    ]


def test_los_indices_de_ejercicio_son_1_indexados():
    module = get_module("06_atencion")
    assert module.exercise(1) is module.exercises[0]
    assert module.exercise("causal_mask").name == "causal_mask"
    with pytest.raises(KeyError):
        module.exercise(0)
