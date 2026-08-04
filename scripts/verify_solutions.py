"""Pega el codigo de cada SOLUTION.md sobre su exercises.py y corre los tests.

Es la comprobacion de que la promesa del `## El código completo` se cumple de verdad:
que un alumno atascado puede copiar ese bloque y ver los tests en verde.

    uv run python scripts/verify_solutions.py

Tarda un par de minutos porque ejecuta la suite entera 18 veces. Por eso no esta en
`make test`: alli hay una version estatica (que el codigo compile y no use nombres
inexistentes) que corre en milisegundos.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys

from llmfs.curriculum import all_modules


def main() -> int:
    fallos: list[str] = []

    for module in all_modules():
        texto = module.solution_file.read_text(encoding="utf-8")
        idx = texto.find("## El código completo")
        bloque = re.search(r"```python\n(.*?)```", texto[idx:], re.DOTALL)
        if bloque is None:
            fallos.append(f"{module.id}: sin bloque de codigo")
            continue
        codigo = bloque.group(1)

        original = module.exercises_file.read_text(encoding="utf-8")
        backup = module.exercises_file.with_suffix(".py.bak")
        shutil.copy(module.exercises_file, backup)

        try:
            arbol = ast.parse(original)
            nombres = {
                n.name
                for n in ast.parse(codigo).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            }
            lineas = original.split("\n")
            borrar = {
                i
                for n in arbol.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in nombres
                for i in range(n.lineno - 1, n.end_lineno)
            }
            module.exercises_file.write_text(
                "\n".join(l for i, l in enumerate(lineas) if i not in borrar) + "\n\n" + codigo,
                encoding="utf-8",
            )

            resultado = subprocess.run(
                [sys.executable, "-m", "pytest", str(module.test_file),
                 "-q", "-p", "no:warnings", "--no-header"],
                capture_output=True, text=True, timeout=900,
            )
            if resultado.returncode == 0:
                print(f"  OK   {module.id}")
            else:
                errores = [l for l in resultado.stdout.split("\n") if "FAILED" in l][:3]
                fallos.append(f"{module.id}\n       " + "\n       ".join(errores))
                print(f"  MAL  {module.id}")
        finally:
            shutil.move(backup, module.exercises_file)

    print()
    if fallos:
        print("=== soluciones que NO se pueden copiar y pegar ===")
        for f in fallos:
            print(f"  {f}")
        return 1
    print(f"=== las {len(list(all_modules()))} soluciones se pegan y pasan sus tests ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
