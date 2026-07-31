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


def test_la_teoria_esta_dentro_del_presupuesto_de_palabras():
    """Entre 900 y 1800 palabras.

    El presupuesto es amplio porque la teoria explica desde una base baja: cada concepto
    entra tres veces (intuicion, ejemplo con numeros, formula). Pero sigue habiendo un
    techo: si un modulo no cabe en 1800 palabras, es que son dos modulos.
    """
    fuera: list[str] = []
    for module in modulos_escritos():
        if not module.theory_file.exists():
            continue
        palabras = len(module.theory_file.read_text(encoding="utf-8").split())
        if not 900 <= palabras <= 1800:
            fuera.append(f"{module.id}: {palabras} palabras")
    assert not fuera, f"TEORIA.md fuera del rango 900-1800 palabras: {fuera}"


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
