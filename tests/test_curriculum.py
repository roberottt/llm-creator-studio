"""The curriculum has to be internally consistent.

These tests do not validate your learning: they validate that the repo is not broken. If
one of them fails it is a bug in the course, not a pending exercise.
"""

from __future__ import annotations

import pytest

from llmfs.curriculum import CURRICULUM, all_modules, get_module, parts, total_exercises


def test_there_are_18_modules_numbered_0_to_17():
    numbers = [m.number for m in CURRICULUM]
    assert numbers == list(range(18)), f"numbering has gaps or is out of order: {numbers}"


def test_the_ids_are_unique():
    ids = [m.id for m in CURRICULUM]
    assert len(ids) == len(set(ids))


def test_exercise_names_are_unique_across_the_whole_course():
    """`llmfs.reference` re-exports every symbol under a flat name.

    If two modules defined an exercise with the same name, the bridge would resolve both to
    the same reference piece without warning. Better that it blows up here.
    """
    seen: dict[str, str] = {}
    for module in CURRICULUM:
        for ex in module.exercises:
            assert ex.name not in seen, (
                f"`{ex.name}` is in {module.id} and also in {seen[ex.name]}"
            )
            seen[ex.name] = module.id


def test_every_module_has_title_summary_and_estimate():
    for module in CURRICULUM:
        assert module.title.strip()
        assert module.summary.strip()
        assert module.est_minutes > 0, f"{module.id} has no time estimate"


def test_every_module_has_exercises():
    for module in CURRICULUM:
        assert module.exercises, f"{module.id} defines no exercises"


def test_the_exercise_total_adds_up():
    assert total_exercises() == sum(len(m.exercises) for m in all_modules())


@pytest.mark.parametrize("ref", [6, "6", "06", "06_attention", "atten"])
def test_get_module_accepts_several_forms(ref):
    assert get_module(ref).id == "06_attention"


def test_get_module_fails_with_a_useful_message():
    with pytest.raises(KeyError) as exc:
        get_module("does_not_exist")
    assert "06_attention" in str(exc.value), "the error should list the valid modules"


def test_the_parts_are_in_order():
    assert list(parts()) == [
        "0 - Before you start",
        "I - Foundations",
        "II - Architecture",
        "III - Training",
        "IV - Use and evaluation",
    ]


def test_exercise_indices_are_1_indexed():
    module = get_module("06_attention")
    assert module.exercise(1) is module.exercises[0]
    assert module.exercise("causal_mask").name == "causal_mask"
    with pytest.raises(KeyError):
        module.exercise(0)
