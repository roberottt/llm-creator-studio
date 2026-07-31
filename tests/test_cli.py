"""La CLI tiene que responder algo util en todos los casos, tambien en los de error."""

from __future__ import annotations

import pytest

import llmfs.cli as cli_mod
import llmfs.progress as progress_mod
from llmfs.cli import build_parser, main


@pytest.fixture(autouse=True)
def fichero_temporal(tmp_path, monkeypatch):
    monkeypatch.setattr(progress_mod, "progress_file", lambda: tmp_path / ".llmfs_progress.json")


@pytest.fixture(autouse=True)
def sin_recursion(monkeypatch):
    """`status` y `next` corren la suite entera; dentro de la suite eso no puede pasar."""
    monkeypatch.setattr(cli_mod, "_refresh", lambda modules: {})


@pytest.mark.parametrize(
    "argv",
    [
        ["status"],
        ["status", "--cached"],
        ["next"],
        ["check", "06"],
        ["demo", "06"],
        ["hint", "06", "-e", "1"],
        ["hint", "06", "-e", "causal_mask", "--nivel", "2"],
        ["device"],
        ["train"],
    ],
)
def test_el_parser_acepta_todos_los_comandos(argv):
    args = build_parser().parse_args(argv)
    assert callable(args.func)


def test_sin_comando_es_error_de_uso():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_device_imprime_el_hardware(capsys):
    assert main(["device"]) == 0
    assert "dispositivo" in capsys.readouterr().out


def test_status_pinta_los_18_modulos(capsys):
    assert main(["status"]) == 0
    salida = capsys.readouterr().out
    assert "00_que_es_un_llm" in salida and "17_extra" in salida
    assert "0/18 modulos completos" in salida


def test_next_apunta_al_primer_modulo_pendiente(capsys):
    assert main(["next"]) == 0
    assert "00_que_es_un_llm" in capsys.readouterr().out


def test_un_modulo_todavia_no_escrito_avisa_sin_reventar(monkeypatch, capsys):
    monkeypatch.setattr(cli_mod, "_module_missing", lambda module: True)
    assert main(["check", "06"]) == 2
    assert "no existe" in capsys.readouterr().out


def test_demo_de_un_modulo_no_escrito_tampoco_revienta(monkeypatch, capsys):
    monkeypatch.setattr(cli_mod, "_module_missing", lambda module: True)
    assert main(["demo", "06"]) == 2


def test_hint_de_un_ejercicio_inexistente_lista_los_validos(capsys):
    assert main(["hint", "06", "-e", "99"]) == 2
    assert "causal_mask" in capsys.readouterr().out


def test_modulo_desconocido_devuelve_codigo_2():
    with pytest.raises(SystemExit) as exc:
        main(["check", "no_existe"])
    assert exc.value.code == 2


def test_los_comandos_de_fases_futuras_explican_donde_se_construyen(capsys):
    """`sample` todavia no existe; tiene que decir en que modulo se construye."""
    assert main(["sample"]) == 2
    assert "14_inferencia" in capsys.readouterr().out


def test_train_ya_no_es_un_stub():
    """Se implementa en el modulo 11 y se usa en el 13."""
    args = build_parser().parse_args(["train", "--config", "tiny_char"])
    assert args.func.__name__ == "cmd_train"


def test_train_avisa_si_el_config_no_existe(capsys):
    assert main(["train", "--config", "no_existe"]) == 2
    assert "tiny_char" in capsys.readouterr().out
