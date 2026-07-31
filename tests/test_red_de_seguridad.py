"""Guardia de integridad: la red de seguridad no puede tener agujeros.

La promesa del curso es que puedes saltarte cualquier modulo y seguir adelante. Eso solo
se sostiene si CADA ejercicio de un modulo ya escrito tiene su pieza equivalente en
`llmfs.reference`. Este test lo comprueba de forma automatica, asi que el agujero se
detecta al anyadir el modulo, no tres fases despues.
"""

from __future__ import annotations

import llmfs.reference as reference_mod
from llmfs.curriculum import all_modules


def modulos_escritos():
    return [m for m in all_modules() if m.path.exists()]


def test_cada_ejercicio_escrito_tiene_su_pieza_de_referencia():
    faltan: list[str] = []
    for module in modulos_escritos():
        for ex in module.exercises:
            if not hasattr(reference_mod, ex.name):
                faltan.append(f"{module.id}.{ex.name}")
    assert not faltan, (
        "Estos ejercicios no tienen implementacion de referencia, asi que el bridge "
        f"reventaria al caer a ella: {faltan}"
    )


def test_los_modulos_escritos_tienen_sus_cinco_ficheros():
    incompletos: list[str] = []
    for module in modulos_escritos():
        for fichero in (
            module.theory_file,
            module.exercises_file,
            module.demo_file,
            module.test_file,
            module.solution_file,
        ):
            if not fichero.exists():
                incompletos.append(f"{module.id}/{fichero.name}")
    assert not incompletos, f"faltan ficheros: {incompletos}"


def test_todos_los_ejercicios_declarados_existen_en_ejercicios_py():
    """Si `curriculum.py` declara un ejercicio, `ejercicios.py` tiene que definirlo."""
    from llmfs import bridge

    faltan: list[str] = []
    for module in modulos_escritos():
        cargado = bridge.exercises(module)
        if cargado is None:
            faltan.append(f"{module.id}/ejercicios.py no importable")
            continue
        for ex in module.exercises:
            if not hasattr(cargado, ex.name):
                faltan.append(f"{module.id}.{ex.name}")
    assert not faltan, f"declarados en curriculum.py pero ausentes: {faltan}"


def test_cada_ejercicio_escrito_tiene_sus_tres_pistas():
    """Nadie debe poder atascarse sin red. Tres niveles: conceptual, tecnica, estructural."""
    from llmfs.hints import get_hints

    faltan: list[str] = []
    for module in modulos_escritos():
        for ex in module.exercises:
            pistas = get_hints(module.id, ex.name)
            if len(pistas) != 3:
                faltan.append(f"{module.id}.{ex.name} tiene {len(pistas)} pistas")
    assert not faltan, f"pistas incompletas en llmfs/hints.py: {faltan}"


def test_la_teoria_sigue_la_estructura_pedagogica():
    """Cada TEORIA.md cierra reconociendo lo que no esta cerrado, y enlaza al glosario."""
    problemas: list[str] = []
    for module in modulos_escritos():
        if not module.theory_file.exists():
            continue
        texto = module.theory_file.read_text(encoding="utf-8")
        if "## Dónde está el debate" not in texto:
            problemas.append(f"{module.id}: sin seccion 'Dónde está el debate'")
        if "GLOSARIO.md" not in texto:
            problemas.append(f"{module.id}: no enlaza al glosario")
    assert not problemas, problemas


def test_la_teoria_tiene_cuerpo_suficiente():
    """Minimo 900 palabras. Sin techo.

    No hay limite superior a proposito: cada concepto tiene que explicarse lo que haga
    falta, con su seccion de "por que importa" y con ejemplos concretos. Lo que si se
    exige es un minimo, porque una teoria de 300 palabras no explica nada desde cero.
    """
    cortas: list[str] = []
    for module in modulos_escritos():
        if not module.theory_file.exists():
            continue
        palabras = len(module.theory_file.read_text(encoding="utf-8").split())
        if palabras < 900:
            cortas.append(f"{module.id}: {palabras} palabras")
    assert not cortas, f"TEORIA.md demasiado corta (minimo 900 palabras): {cortas}"


def test_cada_teoria_explica_por_que_importa_el_modulo():
    """Antes de cualquier concepto hay que decir para que sirve el modulo.

    Alguien que no sabe de LLMs no puede juzgar si merece la pena leer cuatro horas sobre
    atencion si no le dices primero que es LA pieza que separa un modelo mediocre de
    ChatGPT.
    """
    faltan: list[str] = []
    for module in modulos_escritos():
        if not module.theory_file.exists():
            continue
        texto = module.theory_file.read_text(encoding="utf-8")
        if "## Por qué importa este módulo" not in texto:
            faltan.append(module.id)
    assert not faltan, f"sin seccion 'Por qué importa este módulo': {faltan}"


def test_cada_solucion_incluye_el_codigo_completo():
    """El alumno atascado tiene que poder copiar la solucion, no solo leer sobre ella.

    Cada SOLUCION.md termina con una seccion de codigo completo y copiable de todos los
    ejercicios del modulo.
    """
    faltan: list[str] = []
    for module in modulos_escritos():
        if not module.solution_file.exists():
            continue
        texto = module.solution_file.read_text(encoding="utf-8")
        if "## El código completo" not in texto:
            faltan.append(module.id)
    assert not faltan, f"sin seccion 'El código completo': {faltan}"


def test_ningun_fichero_de_test_define_dos_veces_el_mismo_test():
    """Un nombre repetido hace que python se quede con el ultimo y el primero NO se ejecute.

    pytest no avisa de esto: el test desaparece en silencio y el contador de "todo en
    verde" miente. Ha pasado ya dos veces en este repo, asi que queda cazado por aqui.
    """
    import ast
    import collections

    from llmfs.paths import repo_root

    duplicados: list[str] = []
    ficheros = [*(repo_root() / "tests").glob("test_*.py")]
    ficheros += [*(repo_root() / "modulos").glob("*/test_*.py")]

    for fichero in ficheros:
        arbol = ast.parse(fichero.read_text(encoding="utf-8"))
        nombres = [
            nodo.name
            for nodo in arbol.body
            if isinstance(nodo, ast.FunctionDef) and nodo.name.startswith("test_")
        ]
        for nombre, veces in collections.Counter(nombres).items():
            if veces > 1:
                duplicados.append(f"{fichero.name}::{nombre} definido {veces} veces")

    assert not duplicados, duplicados


def test_el_codigo_de_cada_solucion_se_puede_copiar_y_pegar():
    """La promesa del `## El código completo`: que funcione tal cual.

    Se extrae el bloque de codigo de cada SOLUCION.md y se comprueba que:
      1. es python valido,
      2. define todos los ejercicios del modulo,
      3. no usa ningun nombre que no este disponible en el `ejercicios.py` del alumno
         (ni importado alli, ni importado en el propio bloque, ni definido en el).

    El punto 3 es el que importa: un alumno atascado copia ese bloque, y si usa un alias
    de tipo o una funcion auxiliar que solo existe en `llmfs/reference/`, no le compila.

    Este test no ejecuta el codigo (seria lento). Para eso esta el script de verificacion
    que se corre a mano al cambiar las soluciones.
    """
    import ast
    import builtins
    import re

    from llmfs.curriculum import all_modules

    problemas: list[str] = []
    for module in all_modules():
        if not module.solution_file.exists():
            continue
        texto = module.solution_file.read_text(encoding="utf-8")
        idx = texto.find("## El código completo")
        if idx == -1:
            continue

        bloque = re.search(r"```python\n(.*?)```", texto[idx:], re.DOTALL)
        if bloque is None:
            problemas.append(f"{module.id}: la seccion no tiene bloque ```python")
            continue

        try:
            arbol = ast.parse(bloque.group(1))
        except SyntaxError as exc:
            problemas.append(f"{module.id}: el codigo no compila ({exc})")
            continue

        definidos = {
            n.name for n in ast.walk(arbol) if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        for ex in module.exercises:
            if ex.name not in definidos:
                problemas.append(f"{module.id}: falta `{ex.name}` en el codigo completo")

        # Que nombres tiene disponibles el alumno en su ejercicios.py
        disponibles = {"self"} | set(dir(builtins))
        for nodo in ast.walk(ast.parse(module.exercises_file.read_text(encoding="utf-8"))):
            if isinstance(nodo, (ast.Import, ast.ImportFrom)):
                disponibles |= {a.asname or a.name.split(".")[0] for a in nodo.names}
            elif isinstance(nodo, (ast.Assign, ast.AnnAssign)):
                disponibles |= {n.id for n in ast.walk(nodo) if isinstance(n, ast.Name)}
            elif isinstance(nodo, (ast.FunctionDef, ast.ClassDef)):
                disponibles.add(nodo.name)

        # Y cuales aporta el propio bloque
        for nodo in ast.walk(arbol):
            if isinstance(nodo, (ast.Import, ast.ImportFrom)):
                disponibles |= {a.asname or a.name.split(".")[0] for a in nodo.names}

        locales: set[str] = set()
        for nodo in ast.walk(arbol):
            if isinstance(nodo, (ast.Assign, ast.AnnAssign, ast.For, ast.comprehension,
                                 ast.withitem)):
                locales |= {n.id for n in ast.walk(nodo) if isinstance(n, ast.Name)}
            elif isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = nodo.args
                locales |= {a.arg for a in args.args + args.kwonlyargs + args.posonlyargs}
            elif isinstance(nodo, ast.ExceptHandler) and nodo.name:
                locales.add(nodo.name)

        usados = {
            n.id for n in ast.walk(arbol) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        faltan = usados - definidos - disponibles - locales
        if faltan:
            problemas.append(
                f"{module.id}: el codigo usa nombres que el alumno no tiene: {sorted(faltan)}"
            )

    assert not problemas, "\n  " + "\n  ".join(problemas)
