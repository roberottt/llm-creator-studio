"""Persistencia del progreso y calculo de estado."""

from __future__ import annotations

import pytest

import llmfs.progress as progress_mod
from llmfs.curriculum import all_modules, get_module
from llmfs.progress import ModuleStatus, next_module, summary_counts


@pytest.fixture(autouse=True)
def fichero_temporal(tmp_path, monkeypatch):
    """Nunca tocar el .llmfs_progress.json real durante los tests."""
    destino = tmp_path / ".llmfs_progress.json"
    monkeypatch.setattr(progress_mod, "progress_file", lambda: destino)
    return destino


def test_estado_por_defecto_es_sin_tests():
    status = ModuleStatus(module_id="06_atencion")
    assert status.state == "no_tests"
    assert status.icon == "⬜"
    assert status.ratio == "-"


def test_roundtrip_de_guardado_y_carga():
    original = {
        "06_atencion": ModuleStatus(
            module_id="06_atencion", state="done", passed=3, total=3, mine=["causal_mask"]
        )
    }
    progress_mod.save(original)
    cargado = progress_mod.load()
    assert cargado["06_atencion"] == original["06_atencion"]


def test_guardar_fusiona_en_vez_de_pisar():
    progress_mod.save({"01_entorno": ModuleStatus(module_id="01_entorno", state="done")})
    progress_mod.save({"02_autograd": ModuleStatus(module_id="02_autograd", state="done")})
    cargado = progress_mod.load()
    assert set(cargado) == {"01_entorno", "02_autograd"}


def test_un_json_corrupto_no_tumba_la_cli(fichero_temporal):
    fichero_temporal.write_text("{esto no es json", encoding="utf-8")
    assert progress_mod.load() == {}


def test_sin_fichero_no_hay_progreso():
    assert progress_mod.load() == {}


def test_next_module_es_el_primero_no_completo():
    resultados = {
        m.id: ModuleStatus(module_id=m.id, state="done") for m in list(all_modules())[:3]
    }
    assert next_module(resultados).id == "03_tokenizacion"


def test_next_module_devuelve_none_si_todo_esta_hecho():
    resultados = {m.id: ModuleStatus(module_id=m.id, state="done") for m in all_modules()}
    assert next_module(resultados) is None


def test_summary_counts_suma_todos_los_modulos():
    counts = summary_counts({})
    assert sum(counts.values()) == len(all_modules())


def test_niveles_de_pista_se_recuerdan():
    assert progress_mod.get_hint_level("06_atencion", "causal_mask") == 0
    progress_mod.set_hint_level("06_atencion", "causal_mask", 2)
    assert progress_mod.get_hint_level("06_atencion", "causal_mask") == 2


def test_las_pistas_no_borran_el_progreso_de_los_modulos():
    progress_mod.save({"01_entorno": ModuleStatus(module_id="01_entorno", state="done")})
    progress_mod.set_hint_level("01_entorno", "measure_matmul_tflops", 1)
    assert progress_mod.load()["01_entorno"].state == "done"


def test_un_modulo_sin_fichero_de_test_no_revienta():
    status = progress_mod.run_module_tests(get_module("17_extra"))
    assert status.state in {"no_tests", "todo", "in_progress", "done"}
