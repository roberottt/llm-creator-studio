"""Integrity guard: the safety net cannot have holes.

The course's promise is that you can skip any module and carry on. That only holds if EVERY
exercise of an already-written module has its equivalent piece in `llmfs.reference`. This
test checks that automatically, so the hole gets detected when the module is added and not
three phases later.
"""

from __future__ import annotations

import llmfs.reference as reference_mod
from llmfs.curriculum import all_modules


def written_modules():
    return [m for m in all_modules() if m.path.exists()]


def test_every_written_exercise_has_its_reference_piece():
    missing: list[str] = []
    for module in written_modules():
        for ex in module.exercises:
            if not hasattr(reference_mod, ex.name):
                missing.append(f"{module.id}.{ex.name}")
    assert not missing, (
        "These exercises have no reference implementation, so the bridge would blow up "
        f"when falling back to it: {missing}"
    )


def test_the_written_modules_have_their_five_files():
    incomplete: list[str] = []
    for module in written_modules():
        for path in (
            module.theory_file,
            module.exercises_file,
            module.demo_file,
            module.test_file,
            module.solution_file,
        ):
            if not path.exists():
                incomplete.append(f"{module.id}/{path.name}")
    assert not incomplete, f"missing files: {incomplete}"


def test_every_declared_exercise_exists_in_exercises_py():
    """If `curriculum.py` declares an exercise, `exercises.py` has to define it."""
    from llmfs import bridge

    missing: list[str] = []
    for module in written_modules():
        loaded = bridge.exercises(module)
        if loaded is None:
            missing.append(f"{module.id}/exercises.py not importable")
            continue
        for ex in module.exercises:
            if not hasattr(loaded, ex.name):
                missing.append(f"{module.id}.{ex.name}")
    assert not missing, f"declared in curriculum.py but absent: {missing}"


def test_every_written_exercise_has_its_three_hints():
    """Nobody should be able to get stuck with no net. Three levels: conceptual, technical,
    structural."""
    from llmfs.hints import get_hints

    missing: list[str] = []
    for module in written_modules():
        for ex in module.exercises:
            hints = get_hints(module.id, ex.name)
            if len(hints) != 3:
                missing.append(f"{module.id}.{ex.name} has {len(hints)} hints")
    assert not missing, f"incomplete hints in llmfs/hints.py: {missing}"


def test_the_theory_follows_the_pedagogical_structure():
    """Every THEORY.md closes by acknowledging what is not settled, and links to the
    glossary."""
    problems: list[str] = []
    for module in written_modules():
        if not module.theory_file.exists():
            continue
        text = module.theory_file.read_text(encoding="utf-8")
        if "## Where the debate is" not in text:
            problems.append(f"{module.id}: no 'Where the debate is' section")
        if "GLOSSARY.md" not in text:
            problems.append(f"{module.id}: does not link to the glossary")
    assert not problems, problems


def test_the_theory_has_enough_body():
    """Minimum 900 words. No ceiling.

    There is deliberately no upper limit: every concept has to be explained for as long as
    it takes, with its "why it matters" section and with concrete examples. What is required
    is a minimum, because a 300-word theory explains nothing from scratch.
    """
    short: list[str] = []
    for module in written_modules():
        if not module.theory_file.exists():
            continue
        words = len(module.theory_file.read_text(encoding="utf-8").split())
        if words < 900:
            short.append(f"{module.id}: {words} words")
    assert not short, f"THEORY.md too short (minimum 900 words): {short}"


def test_every_theory_explains_why_the_module_matters():
    """Before any concept you have to say what the module is for.

    Someone who does not know about LLMs cannot judge whether four hours on attention are
    worth it unless you tell them first that it is THE piece separating a mediocre model
    from ChatGPT.
    """
    missing: list[str] = []
    for module in written_modules():
        if not module.theory_file.exists():
            continue
        text = module.theory_file.read_text(encoding="utf-8")
        if "## Why this module matters" not in text:
            missing.append(module.id)
    assert not missing, f"no 'Why this module matters' section: {missing}"


def test_every_solution_includes_the_complete_code():
    """A stuck student has to be able to copy the solution, not just read about it.

    Every SOLUTION.md ends with a section of complete, copy-pasteable code for all of the
    module's exercises.
    """
    missing: list[str] = []
    for module in written_modules():
        if not module.solution_file.exists():
            continue
        text = module.solution_file.read_text(encoding="utf-8")
        if "## The complete code" not in text:
            missing.append(module.id)
    assert not missing, f"no 'The complete code' section: {missing}"


def test_no_test_file_defines_the_same_test_twice():
    """A repeated name makes python keep the last one, and the first one does NOT run.

    pytest does not warn about this: the test silently disappears and the "all green"
    counter lies. It has happened twice in this repo already, so it gets caught here.
    """
    import ast
    import collections

    from llmfs.paths import repo_root

    duplicates: list[str] = []
    files = [*(repo_root() / "tests").glob("test_*.py")]
    files += [*(repo_root() / "modules").glob("*/test_*.py")]

    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = [
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
        ]
        for name, times in collections.Counter(names).items():
            if times > 1:
                duplicates.append(f"{path.name}::{name} defined {times} times")

    assert not duplicates, duplicates


def test_the_code_of_every_solution_can_be_copy_pasted():
    """The promise of `## The complete code`: that it works as-is.

    The code block of each SOLUTION.md is extracted and checked to see that:
      1. it is valid python,
      2. it defines every exercise in the module,
      3. it uses no name that is not available in the student's `exercises.py` (neither
         imported there, nor imported in the block itself, nor defined in it).

    Point 3 is the one that matters: a stuck student copies that block, and if it uses a
    type alias or a helper function that only exists in `llmfs/reference/`, it will not
    compile for them.

    This test does not execute the code (that would be slow). That is what the verification
    script, run by hand when the solutions change, is for.
    """
    import ast
    import builtins
    import re

    from llmfs.curriculum import all_modules

    problems: list[str] = []
    for module in all_modules():
        if not module.solution_file.exists():
            continue
        text = module.solution_file.read_text(encoding="utf-8")
        idx = text.find("## The complete code")
        if idx == -1:
            continue

        block = re.search(r"```python\n(.*?)```", text[idx:], re.DOTALL)
        if block is None:
            problems.append(f"{module.id}: the section has no ```python block")
            continue

        try:
            tree = ast.parse(block.group(1))
        except SyntaxError as exc:
            problems.append(f"{module.id}: the code does not compile ({exc})")
            continue

        defined = {
            n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        for ex in module.exercises:
            if ex.name not in defined:
                problems.append(f"{module.id}: `{ex.name}` missing from the complete code")

        # Which names the student has available in their exercises.py
        available = {"self"} | set(dir(builtins))
        for node in ast.walk(ast.parse(module.exercises_file.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                available |= {a.asname or a.name.split(".")[0] for a in node.names}
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                available |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                available.add(node.name)

        # And which ones the block itself brings in
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                available |= {a.asname or a.name.split(".")[0] for a in node.names}

        locals_: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.For, ast.comprehension,
                                 ast.withitem)):
                locals_ |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                locals_ |= {a.arg for a in args.args + args.kwonlyargs + args.posonlyargs}
            elif isinstance(node, ast.ExceptHandler) and node.name:
                locals_.add(node.name)

        used = {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        missing = used - defined - available - locals_
        if missing:
            problems.append(
                f"{module.id}: the code uses names the student does not have: {sorted(missing)}"
            )

    assert not problems, "\n  " + "\n  ".join(problems)
