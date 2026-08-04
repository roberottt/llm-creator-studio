"""Genera la seccion 'El codigo completo' de cada SOLUTION.md a partir de la referencia."""
import ast, inspect, pathlib, textwrap
import llmfs.reference as R
from llmfs.curriculum import all_modules

CABECERA = """

---

## El código completo

Si te has atascado, aquí está la implementación entera. **Cópiala, pégala y ejecuta los
tests**: verlos pasar con código que entiendes es mejor que quedarte bloqueado.

Y después vuelve al ejercicio y escríbela tú. Leer una solución que ya has peleado funciona
muy bien; leerla en frío, no funciona nada.

```python
"""

PIE = """```

Los imports que hacen falta ya están en el `exercises.py` del módulo, salvo los que
aparezcan arriba del bloque.
"""

# Funciones auxiliares que el codigo de la solucion necesita y que NO son ejercicios.
# Sin ellas, copiar y pegar no compilaria.
AUXILIARES = {
    "01_environment": ["matmul_flops"],
    "03_tokenization": ["_encode_chunk"],
    "09_position": ["rotate_half"],
}

# Alias de tipo que solo existen en llmfs/reference: se sustituyen por el tipo literal.
SUSTITUCIONES = {
    "CountTable": "dict[str, dict[str, int]]",
}

# Imports extra que hay que anyadir arriba del bloque para que compile tal cual.
IMPORTS = {
    "06_attention": "from typing import Any",
    "10_the_full_gpt": "from llmfs.reference import make_ffn, make_norm, sinusoidal_embeddings",
}


def limpia(fuente: str) -> str:
    """Quita el docstring de la funcion/clase para que el codigo se lea de un vistazo."""
    fuente = textwrap.dedent(fuente)
    try:
        arbol = ast.parse(fuente)
    except SyntaxError:
        return fuente
    nodo = arbol.body[0]
    if not isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return fuente

    lineas = fuente.split("\n")
    a_borrar = set()

    def marca_docstring(n):
        cuerpo = getattr(n, "body", [])
        if (
            cuerpo
            and isinstance(cuerpo[0], ast.Expr)
            and isinstance(cuerpo[0].value, ast.Constant)
            and isinstance(cuerpo[0].value.value, str)
        ):
            for i in range(cuerpo[0].lineno - 1, cuerpo[0].end_lineno):
                a_borrar.add(i)

    marca_docstring(nodo)
    if isinstance(nodo, ast.ClassDef):
        for hijo in nodo.body:
            if isinstance(hijo, (ast.FunctionDef, ast.AsyncFunctionDef)):
                marca_docstring(hijo)

    salida = [l for i, l in enumerate(lineas) if i not in a_borrar]
    # Colapsar lineas en blanco duplicadas que deja el borrado
    resultado, previa_vacia = [], False
    for l in salida:
        vacia = not l.strip()
        if vacia and previa_vacia:
            continue
        resultado.append(l)
        previa_vacia = vacia
    return "\n".join(resultado).rstrip() + "\n"


hechos, fallos = 0, []
for m in all_modules():
    if not m.solution_file.exists():
        continue
    partes = []
    for aux in AUXILIARES.get(m.id, []):
        impl_aux = getattr(R, aux, None)
        if impl_aux is None:
            # Puede ser privada: buscarla en el modulo donde vive el primer ejercicio
            import importlib
            for nombre_mod in ("tokenizer", "position", "hardware"):
                mod = importlib.import_module(f"llmfs.reference.{nombre_mod}")
                if hasattr(mod, aux):
                    impl_aux = getattr(mod, aux)
                    break
        if impl_aux is not None:
            partes.append(limpia(inspect.getsource(impl_aux)))

    for ex in m.exercises:
        impl = getattr(R, ex.name, None)
        if impl is None:
            fallos.append(f"{m.id}.{ex.name}: no esta en reference")
            continue
        try:
            fuente = inspect.getsource(impl)
        except (OSError, TypeError) as exc:
            fallos.append(f"{m.id}.{ex.name}: {exc}")
            continue
        partes.append(limpia(fuente))

    if not partes:
        fallos.append(f"{m.id}: sin codigo")
        continue

    codigo = "\n\n".join(partes).rstrip()
    for viejo, nuevo in SUSTITUCIONES.items():
        codigo = codigo.replace(f": {viejo}", f": {nuevo}")
    if m.id in IMPORTS:
        codigo = IMPORTS[m.id] + "\n\n" + codigo

    texto = m.solution_file.read_text(encoding="utf-8")
    # Quitar una seccion previa si la hubiera (idempotente)
    if "## El código completo" in texto:
        texto = texto[: texto.index("\n---\n\n## El código completo")]
    m.solution_file.write_text(
        texto.rstrip() + CABECERA + codigo + "\n" + PIE,
        encoding="utf-8",
    )
    hechos += 1

print(f"{hechos} soluciones con codigo completo")
for f in fallos:
    print(f"  AVISO {f}")
