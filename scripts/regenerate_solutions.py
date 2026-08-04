"""Generates the 'The complete code' section of each SOLUTION.md from the reference."""
import ast, inspect, pathlib, textwrap
import llmfs.reference as R
from llmfs.curriculum import all_modules

HEADER = """

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
"""

FOOTER = """```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
"""

# Helper functions the solution code needs that are NOT exercises.
# Without them, copy-pasting would not compile.
HELPERS = {
    "01_environment": ["matmul_flops"],
    "03_tokenization": ["_encode_chunk"],
    "09_position": ["rotate_half"],
}

# Type aliases that only exist in llmfs/reference: replaced by the literal type.
SUBSTITUTIONS = {
    "CountTable": "dict[str, dict[str, int]]",
}

# Extra imports that have to be added above the block so it compiles as-is.
IMPORTS = {
    "06_attention": "from typing import Any",
    "10_the_full_gpt": "from llmfs.reference import make_ffn, make_norm, sinusoidal_embeddings",
}


def strip_docstrings(source: str) -> str:
    """Removes the function/class docstring so the code reads at a glance."""
    source = textwrap.dedent(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return source

    lines = source.split("\n")
    to_delete = set()

    def mark_docstring(n):
        body = getattr(n, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            for i in range(body[0].lineno - 1, body[0].end_lineno):
                to_delete.add(i)

    mark_docstring(node)
    if isinstance(node, ast.ClassDef):
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                mark_docstring(child)

    kept = [l for i, l in enumerate(lines) if i not in to_delete]
    # Collapse the duplicate blank lines the removal leaves behind
    result, previous_blank = [], False
    for l in kept:
        blank = not l.strip()
        if blank and previous_blank:
            continue
        result.append(l)
        previous_blank = blank
    return "\n".join(result).rstrip() + "\n"


done, failures = 0, []
for m in all_modules():
    if not m.solution_file.exists():
        continue
    parts = []
    for helper in HELPERS.get(m.id, []):
        helper_impl = getattr(R, helper, None)
        if helper_impl is None:
            # It may be private: look for it in the module where the first exercise lives
            import importlib
            for module_name in ("tokenizer", "position", "hardware"):
                mod = importlib.import_module(f"llmfs.reference.{module_name}")
                if hasattr(mod, helper):
                    helper_impl = getattr(mod, helper)
                    break
        if helper_impl is not None:
            parts.append(strip_docstrings(inspect.getsource(helper_impl)))

    for ex in m.exercises:
        impl = getattr(R, ex.name, None)
        if impl is None:
            failures.append(f"{m.id}.{ex.name}: not in reference")
            continue
        try:
            source = inspect.getsource(impl)
        except (OSError, TypeError) as exc:
            failures.append(f"{m.id}.{ex.name}: {exc}")
            continue
        parts.append(strip_docstrings(source))

    if not parts:
        failures.append(f"{m.id}: no code")
        continue

    code = "\n\n".join(parts).rstrip()
    for old, new in SUBSTITUTIONS.items():
        code = code.replace(f": {old}", f": {new}")
    if m.id in IMPORTS:
        code = IMPORTS[m.id] + "\n\n" + code

    text = m.solution_file.read_text(encoding="utf-8")
    # Remove any previous section (idempotent)
    if "## The complete code" in text:
        text = text[: text.index("\n---\n\n## The complete code")]
    m.solution_file.write_text(
        text.rstrip() + HEADER + code + "\n" + FOOTER,
        encoding="utf-8",
    )
    done += 1

print(f"{done} solutions with complete code")
for f in failures:
    print(f"  WARNING {f}")
