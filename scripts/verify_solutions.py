"""Pastes each SOLUTION.md's code over its exercises.py and runs the tests.

This is the check that the promise of `## The complete code` actually holds: that a stuck
student can copy that block and see the tests go green.

    uv run python scripts/verify_solutions.py

It takes a couple of minutes because it runs the whole suite 18 times. That is why it is not
in `make test`: there is a static version there (that the code compiles and uses no
nonexistent names) which runs in milliseconds.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys

from llmfs.curriculum import all_modules


def main() -> int:
    failures: list[str] = []

    for module in all_modules():
        text = module.solution_file.read_text(encoding="utf-8")
        idx = text.find("## The complete code")
        block = re.search(r"```python\n(.*?)```", text[idx:], re.DOTALL)
        if block is None:
            failures.append(f"{module.id}: no code block")
            continue
        code = block.group(1)

        original = module.exercises_file.read_text(encoding="utf-8")
        backup = module.exercises_file.with_suffix(".py.bak")
        shutil.copy(module.exercises_file, backup)

        try:
            tree = ast.parse(original)
            names = {
                n.name
                for n in ast.parse(code).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }
            lines = original.split("\n")
            to_delete = {
                i
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
                for i in range(n.lineno - 1, n.end_lineno)
            }
            module.exercises_file.write_text(
                "\n".join(l for i, l in enumerate(lines) if i not in to_delete) + "\n\n" + code,
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(module.test_file),
                 "-q", "-p", "no:warnings", "--no-header"],
                capture_output=True, text=True, timeout=900,
            )
            if result.returncode == 0:
                print(f"  OK    {module.id}")
            else:
                errors = [l for l in result.stdout.split("\n") if "FAILED" in l][:3]
                failures.append(f"{module.id}\n       " + "\n       ".join(errors))
                print(f"  FAIL  {module.id}")
        finally:
            shutil.move(backup, module.exercises_file)

    print()
    if failures:
        print("=== solutions that CANNOT be copy-pasted ===")
        for f in failures:
            print(f"  {f}")
        return 1
    print(f"=== all {len(list(all_modules()))} solutions paste in and pass their tests ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
