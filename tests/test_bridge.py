"""The safety net has to actually work.

If these tests fail, the course loses its central promise: that you never get stuck because
an earlier exercise is half done.
"""

from __future__ import annotations

import pytest

import llmfs.curriculum as curriculum_mod
import llmfs.reference as reference_mod
from llmfs import bridge
from llmfs.curriculum import Exercise, Module

# Module 06's identity (its real number and slug) is reused, but pointing at a temporary
# directory. That also exercises resolution by path, which needs the id to exist in the
# curriculum.
NUMBER, SLUG = 6, "attention"
MODULE_ID = f"{NUMBER:02d}_{SLUG}"


@pytest.fixture(autouse=True)
def clear_cache():
    bridge.clear_cache()
    yield
    bridge.clear_cache()


@pytest.fixture
def module(tmp_path, monkeypatch):
    """A synthetic module in tmp_path, with `add` as its only exercise."""
    monkeypatch.setattr(curriculum_mod, "modules_dir", lambda: tmp_path)
    (tmp_path / MODULE_ID).mkdir()

    # The reference piece: always correct, always available.
    monkeypatch.setattr(reference_mod, "add", lambda a, b: a + b, raising=False)

    return Module(
        number=NUMBER,
        slug=SLUG,
        title="Test module",
        part="test",
        summary="only for the bridge tests",
        est_minutes=1,
        exercises=(Exercise("add", "add two numbers"),),
    )


def write_exercises(module: Module, code: str) -> None:
    module.exercises_file.write_text(code, encoding="utf-8")
    bridge.clear_cache()


# ---------------------------------------------------------------------------- cases


def test_uses_your_implementation_when_it_works(module):
    write_exercises(module, "def add(a, b):\n    return a + b\n")

    assert bridge.resolve(module, "add")(2, 3) == 5
    assert bridge.resolution(module, "add").source == "exercise"


def test_falls_back_to_reference_if_the_exercise_raises_not_implemented(module):
    write_exercises(
        module,
        "def add(a, b):\n    raise NotImplementedError('TODO: module 99')\n",
    )

    res = bridge.resolution(module, "add")
    assert res.source == "reference"
    assert "NotImplementedError" in res.reason
    # and the production code still works anyway
    assert bridge.resolve(module, "add")(2, 3) == 5


def test_falls_back_to_reference_if_exercises_py_does_not_exist(module):
    res = bridge.resolution(module, "add")
    assert res.source == "reference"
    assert bridge.resolve(module, "add")(1, 1) == 2


def test_falls_back_to_reference_if_exercises_py_does_not_compile(module, capsys):
    write_exercises(module, "def add(a, b)\n    return a + b\n")  # missing ':'

    res = bridge.resolution(module, "add")
    assert res.source == "reference"
    assert bridge.resolve(module, "add")(4, 5) == 9

    # A file that does not compile is something the user MUST see.
    out = capsys.readouterr()
    assert "could not be imported" in out.err


def test_falls_back_to_reference_if_the_symbol_is_missing(module):
    write_exercises(module, "def something_else():\n    return 1\n")

    res = bridge.resolution(module, "add")
    assert res.source == "reference"
    assert "is not defined" in res.reason


def test_the_probe_rejects_implementations_that_return_garbage(module, monkeypatch):
    """An exercise can be 'written' and still not be usable."""
    from llmfs.probes import PROBES

    def check(impl):
        if impl(2, 3) != 5:
            raise ValueError("2 + 3 should give 5")

    monkeypatch.setitem(PROBES, "add", check)
    write_exercises(module, "def add(a, b):\n    return None\n")

    res = bridge.resolution(module, "add")
    assert res.source == "reference"
    assert "ValueError" in res.reason


def test_force_reference_ignores_your_code_even_when_it_is_correct(module, monkeypatch):
    write_exercises(module, "def add(a, b):\n    return a + b\n")
    monkeypatch.setenv("LLMFS_FORCE_REFERENCE", "1")
    bridge.clear_cache()

    res = bridge.resolution(module, "add")
    assert res.source == "reference"
    assert "LLMFS_FORCE_REFERENCE" in res.reason


def test_warns_on_stderr_exactly_once_when_using_the_reference(module, capsys):
    for _ in range(3):
        bridge.resolve(module, "add")

    err = capsys.readouterr().err
    assert err.count("using the REFERENCE for `add`") == 1, (
        "the warning must show up, but it must not flood the console"
    )


def test_if_the_reference_piece_is_missing_the_error_is_explicit(module, monkeypatch):
    monkeypatch.delattr(reference_mod, "add")
    with pytest.raises(AttributeError, match="COURSE BUG"):
        bridge.resolve(module, "add")


def test_exercises_accepts_a_test_file_path(module):
    """The `test_NN.py` files identify themselves with `exercises(__file__)`."""
    write_exercises(module, "MARKER = 42\n")
    loaded = bridge.exercises(str(module.path / f"test_{NUMBER:02d}.py"))
    assert loaded is not None and loaded.MARKER == 42


def test_a_class_with_an_unimplemented_forward_is_not_usable(module):
    """The real case: an `nn.Module` with `__init__` done and `forward` half finished."""
    write_exercises(
        module,
        "class add:\n"
        "    def __init__(self, n):\n"
        "        self.n = n\n"
        "    def forward(self, x):\n"
        "        raise NotImplementedError('TODO')\n",
    )
    assert bridge.resolution(module, "add").source == "reference"


def test_a_body_with_only_pass_also_counts_as_a_template(module):
    write_exercises(module, "def add(a, b):\n    pass\n")
    assert bridge.resolution(module, "add").source == "reference"


def test_a_long_docstring_does_not_confuse_the_detector(module):
    write_exercises(
        module,
        'def add(a, b):\n'
        '    """Add two numbers.\n\n'
        '    Args:\n        a: the first\n        b: the second\n\n'
        '    Returns:\n        a + b\n    """\n'
        '    return a + b\n',
    )
    assert bridge.resolution(module, "add").source == "exercise"


def test_module_resolutions_covers_every_exercise(module):
    resolutions = bridge.module_resolutions(module)
    assert [r.name for r in resolutions] == ["add"]
