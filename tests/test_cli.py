"""The CLI has to answer something useful in every case, including the error cases."""

from __future__ import annotations

import pytest

import llmfs.cli as cli_mod
import llmfs.progress as progress_mod
from llmfs.cli import build_parser, main


@pytest.fixture(autouse=True)
def temporary_file(tmp_path, monkeypatch):
    monkeypatch.setattr(progress_mod, "progress_file", lambda: tmp_path / ".llmfs_progress.json")


@pytest.fixture(autouse=True)
def no_recursion(monkeypatch):
    """`status` and `next` run the whole suite; inside the suite that cannot happen."""
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
        ["hint", "06", "-e", "causal_mask", "--level", "2"],
        ["device"],
        ["train"],
        ["sample"],
    ],
)
def test_the_parser_accepts_every_command(argv):
    args = build_parser().parse_args(argv)
    assert callable(args.func)


def test_no_command_is_a_usage_error():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_device_prints_the_hardware(capsys):
    assert main(["device"]) == 0
    assert "device" in capsys.readouterr().out


def test_status_renders_the_18_modules(capsys):
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "00_what_is_an_llm" in out and "17_extra" in out
    assert "0/18 modules complete" in out


def test_next_points_at_the_first_pending_module(capsys):
    assert main(["next"]) == 0
    assert "00_what_is_an_llm" in capsys.readouterr().out


def test_a_module_not_written_yet_warns_without_crashing(monkeypatch, capsys):
    monkeypatch.setattr(cli_mod, "_module_missing", lambda module: True)
    assert main(["check", "06"]) == 2
    assert "does not exist" in capsys.readouterr().out


def test_demo_of_an_unwritten_module_does_not_crash_either(monkeypatch, capsys):
    monkeypatch.setattr(cli_mod, "_module_missing", lambda module: True)
    assert main(["demo", "06"]) == 2


def test_hint_for_a_nonexistent_exercise_lists_the_valid_ones(capsys):
    assert main(["hint", "06", "-e", "99"]) == 2
    assert "causal_mask" in capsys.readouterr().out


def test_unknown_module_returns_exit_code_2():
    with pytest.raises(SystemExit) as exc:
        main(["check", "does_not_exist"])
    assert exc.value.code == 2


def test_commands_from_future_phases_explain_where_they_get_built(capsys):
    """`data` does not exist yet; it has to say which module builds it."""
    assert main(["data"]) == 2
    assert "04_data" in capsys.readouterr().out


def test_train_is_no_longer_a_stub():
    """It is implemented in module 11 and used in module 13."""
    args = build_parser().parse_args(["train", "--config", "tiny_char"])
    assert args.func.__name__ == "cmd_train"


def test_train_warns_if_the_config_does_not_exist(capsys):
    assert main(["train", "--config", "does_not_exist"]) == 2
    assert "tiny_char" in capsys.readouterr().out


def test_sample_is_no_longer_a_stub():
    """It is implemented in module 14: generate text from a saved checkpoint."""
    args = build_parser().parse_args(["sample", "--config", "tiny_char"])
    assert args.func.__name__ == "cmd_sample"


def test_sample_warns_if_the_config_does_not_exist(capsys):
    assert main(["sample", "--config", "does_not_exist"]) == 2
    assert "does_not_exist" in capsys.readouterr().out


def test_sample_warns_if_there_is_no_checkpoint(tmp_path, capsys):
    assert main(["sample", "--checkpoint", str(tmp_path / "nope.pt")]) == 2
    assert "checkpoint" in capsys.readouterr().out.lower()
