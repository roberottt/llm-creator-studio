"""Progress persistence and state computation."""

from __future__ import annotations

import pytest

import llmfs.progress as progress_mod
from llmfs.curriculum import all_modules, get_module
from llmfs.progress import ModuleStatus, next_module, summary_counts


@pytest.fixture(autouse=True)
def temporary_file(tmp_path, monkeypatch):
    """Never touch the real .llmfs_progress.json during the tests."""
    target = tmp_path / ".llmfs_progress.json"
    monkeypatch.setattr(progress_mod, "progress_file", lambda: target)
    return target


def test_the_default_state_is_no_tests():
    status = ModuleStatus(module_id="06_attention")
    assert status.state == "no_tests"
    assert status.icon == "⬜"
    assert status.ratio == "-"


def test_save_and_load_roundtrip():
    original = {
        "06_attention": ModuleStatus(
            module_id="06_attention", state="done", passed=3, total=3, mine=["causal_mask"]
        )
    }
    progress_mod.save(original)
    loaded = progress_mod.load()
    assert loaded["06_attention"] == original["06_attention"]


def test_saving_merges_instead_of_overwriting():
    progress_mod.save({"01_environment": ModuleStatus(module_id="01_environment", state="done")})
    progress_mod.save({"02_autograd": ModuleStatus(module_id="02_autograd", state="done")})
    loaded = progress_mod.load()
    assert set(loaded) == {"01_environment", "02_autograd"}


def test_a_corrupted_json_does_not_take_down_the_cli(temporary_file):
    temporary_file.write_text("{this is not json", encoding="utf-8")
    assert progress_mod.load() == {}


def test_with_no_file_there_is_no_progress():
    assert progress_mod.load() == {}


def test_next_module_is_the_first_incomplete_one():
    results = {
        m.id: ModuleStatus(module_id=m.id, state="done") for m in list(all_modules())[:3]
    }
    assert next_module(results).id == "03_tokenization"


def test_next_module_returns_none_when_everything_is_done():
    results = {m.id: ModuleStatus(module_id=m.id, state="done") for m in all_modules()}
    assert next_module(results) is None


def test_summary_counts_adds_up_every_module():
    counts = summary_counts({})
    assert sum(counts.values()) == len(all_modules())


def test_hint_levels_are_remembered():
    assert progress_mod.get_hint_level("06_attention", "causal_mask") == 0
    progress_mod.set_hint_level("06_attention", "causal_mask", 2)
    assert progress_mod.get_hint_level("06_attention", "causal_mask") == 2


def test_hints_do_not_wipe_the_modules_progress():
    progress_mod.save({"01_environment": ModuleStatus(module_id="01_environment", state="done")})
    progress_mod.set_hint_level("01_environment", "measure_matmul_tflops", 1)
    assert progress_mod.load()["01_environment"].state == "done"


def test_a_module_with_no_test_file_does_not_blow_up():
    status = progress_mod.run_module_tests(get_module("17_extra"))
    assert status.state in {"no_tests", "todo", "in_progress", "done"}
