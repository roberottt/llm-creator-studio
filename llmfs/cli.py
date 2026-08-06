"""La CLI del curso: `python -m llmfs ...` (o simplemente `llmfs ...`).

    llmfs status              tabla de progreso de los 18 modulos
    llmfs next                en que modulo estas y que ejercicio toca
    llmfs check 06            corre los tests del modulo 06 con pistas si falla
    llmfs demo 06             ejecuta el experimento del modulo 06
    llmfs hint 06 -e 2        pista progresiva del ejercicio 2 (repite para mas)
    llmfs device              que hardware ha detectado y con que precision va a entrenar

Torch se importa dentro de cada comando, no arriba: `llmfs next` responde en decimas y
no hay motivo para pagar el segundo largo que cuesta cargar torch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llmfs import progress as prog
from llmfs.curriculum import Module, all_modules, get_module, parts, total_exercises, total_minutes

console = Console()


# ---------------------------------------------------------------------------- helpers


def _fmt_minutes(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes}m"
    hours, rest = divmod(minutes, 60)
    return f"{hours}h" if rest == 0 else f"{hours}h{rest:02d}"


def _resolve_module(ref: str) -> Module:
    try:
        return get_module(ref)
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        raise SystemExit(2) from exc


def _module_missing(module: Module) -> bool:
    """`True` si el modulo todavia no ha sido escrito en el repo."""
    return not module.path.exists()


def _warn_missing(module: Module) -> None:
    console.print(
        Panel(
            f"El modulo [bold]{module.id}[/bold] todavia no existe en el repo.\n"
            f"Se escribe en la fase correspondiente del plan de construccion.\n\n"
            f"Modulos disponibles ahora mismo: "
            + (", ".join(m.id for m in all_modules() if m.path.exists()) or "ninguno"),
            title="Aun no disponible",
            border_style="yellow",
        )
    )


# ---------------------------------------------------------------------------- status


def cmd_status(args: argparse.Namespace) -> int:
    modules = list(all_modules())

    if args.cached:
        results = prog.load()
        if not results:
            console.print("[yellow]No hay progreso cacheado. Corriendo los tests...[/yellow]")
            results = _refresh(modules)
    else:
        results = _refresh(modules)

    table = Table(
        title="LLM desde cero - progreso del curriculo",
        title_style="bold",
        header_style="bold",
        expand=False,
    )
    table.add_column("", width=2, justify="center")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("modulo")
    table.add_column("tests", justify="right")
    table.add_column("mio", justify="right")
    table.add_column("est.", justify="right", style="dim")

    for part, part_modules in parts().items():
        table.add_section()
        table.add_row("", f"[bold magenta]{part}[/bold magenta]", "", "", "", "")
        for module in part_modules:
            status = results.get(module.id) or prog.ModuleStatus(module_id=module.id)
            n_ex = len(module.exercises)
            mine = f"{len(status.mine)}/{n_ex}" if n_ex else "-"
            title = module.title
            if _module_missing(module):
                title = f"[dim]{title}[/dim]"
            table.add_row(
                status.icon,
                module.id,
                title,
                status.ratio,
                mine,
                _fmt_minutes(module.est_minutes),
            )

    console.print(table)

    counts = prog.summary_counts(results)
    done = counts["done"]
    total = len(modules)
    pct = 100.0 * done / total if total else 0.0
    mine_total = sum(len(s.mine) for s in results.values())

    console.print(
        f"\n  [bold]{done}/{total}[/bold] modulos completos ({pct:.0f}%)  |  "
        f"[bold]{mine_total}/{total_exercises()}[/bold] ejercicios implementados  |  "
        f"trabajo estimado restante: {_fmt_minutes(_remaining_minutes(results))} "
        f"de {_fmt_minutes(total_minutes())}"
    )

    nxt = prog.next_module(results)
    if nxt is not None:
        console.print(f"  siguiente: [cyan]{nxt.id}[/cyan] - {nxt.summary}")
        console.print(f"  [dim]llmfs next[/dim] para el detalle\n")
    else:
        console.print("  [green]Curriculo completo. Enhorabuena.[/green]\n")
    return 0


def _remaining_minutes(results: dict[str, prog.ModuleStatus]) -> int:
    return sum(
        m.est_minutes
        for m in all_modules()
        if (results.get(m.id) or prog.ModuleStatus(module_id=m.id)).state != "done"
    )


def _refresh(modules: list[Module]) -> dict[str, prog.ModuleStatus]:
    results: dict[str, prog.ModuleStatus] = {}
    with console.status("[dim]corriendo tests...[/dim]") as spinner:
        for module in modules:
            spinner.update(f"[dim]corriendo tests de {module.id}...[/dim]")
            results[module.id] = prog.run_module_tests(module, quiet=True)
    prog.save(results)
    return results


# ---------------------------------------------------------------------------- next


def cmd_next(args: argparse.Namespace) -> int:
    results = prog.load()
    if not results:
        results = _refresh(list(all_modules()))

    module = prog.next_module(results)
    if module is None:
        console.print("[green]Has terminado todos los modulos. Ve a entrenar el modelo final.[/green]")
        return 0

    status = results.get(module.id) or prog.ModuleStatus(module_id=module.id)

    body = [
        f"[bold cyan]{module.id}[/bold cyan] - {module.title}",
        f"[dim]{module.part} | unas {_fmt_minutes(module.est_minutes)} de trabajo[/dim]",
        "",
        f"[bold]Que vas a aprender:[/bold] {module.summary}",
    ]

    if _module_missing(module):
        body += ["", "[yellow]Este modulo todavia no esta escrito en el repo.[/yellow]"]
    else:
        # Lo pendiente se deriva de `mine` y no de `borrowed`: sin datos cacheados ambas
        # listas estan vacias, y en ese caso lo correcto es "no has hecho nada todavia",
        # no "lo tienes todo hecho".
        pending = [ex for ex in module.exercises if ex.name not in status.mine]
        hechos = len(status.mine)

        if hechos == 0:
            body += [
                "",
                "Todavia no has empezado este modulo. El camino es siempre el mismo:",
                "",
                "  1. lee la teoria (10-15 min, no te la saltes)",
                "  2. abre ejercicios.py y lee el enunciado del primero",
                "  3. implementalo y ejecuta los tests",
                "  4. rojo -> pista; verde -> siguiente ejercicio",
            ]
        else:
            body += ["", f"[bold]Llevas {hechos} de {len(module.exercises)} ejercicios.[/bold] "
                     f"Tests: {status.ratio}"]

        if pending:
            first = pending[0]
            idx = module.exercises.index(first) + 1
            body += [
                "",
                f"[bold]Ahora toca el ejercicio {idx}:[/bold] [yellow]{first.name}[/yellow]",
                f"  {first.title}",
            ]
            if len(pending) > 1:
                body.append(f"  [dim]quedan {len(pending) - 1} mas despues de ese[/dim]")
        elif status.state != "done":
            body += [
                "",
                "Has implementado todos los ejercicios pero algun test sigue en rojo.",
                "Eso es normal y es justo el momento util del ejercicio: el test te esta",
                "diciendo exactamente que no cuadra.",
            ]
            if status.first_failure:
                body.append(f"  [red]{status.first_failure}[/red]")

        body += [
            "",
            f"[dim]1. leer  :[/dim] {module.theory_file.relative_to(module.path.parent.parent)}",
            f"[dim]2. editar:[/dim] {module.exercises_file.relative_to(module.path.parent.parent)}",
            f"[dim]3. probar:[/dim] llmfs check {module.number:02d}",
            f"[dim]   atasco :[/dim] llmfs hint {module.number:02d} -e 1  "
            f"[dim](repite el comando para pistas mas explicitas)[/dim]",
            f"[dim]   ver    :[/dim] llmfs demo {module.number:02d}  "
            f"[dim](el concepto, en graficas y numeros)[/dim]",
        ]

    console.print(Panel("\n".join(body), title="Por donde sigues", border_style="cyan"))
    return 0


# ---------------------------------------------------------------------------- check


def cmd_check(args: argparse.Namespace) -> int:
    module = _resolve_module(args.modulo)
    if _module_missing(module):
        _warn_missing(module)
        return 2
    if not module.test_file.exists():
        console.print(f"[yellow]{module.id} no tiene test_{module.number:02d}.py todavia.[/yellow]")
        return 2

    import pytest

    console.rule(f"[bold cyan]{module.id}[/bold cyan] - {module.title}")
    pytest_args = [str(module.test_file), "-v", "--tb=short", "--no-header", "-p", "no:cacheprovider"]
    if args.k:
        pytest_args += ["-k", args.k]
    code = pytest.main(pytest_args)

    # Recalcula y persiste el estado del modulo despues de la ejecucion.
    status = prog.run_module_tests(module, quiet=True)
    prog.save({module.id: status})

    console.rule(style="dim")
    if status.state == "done":
        console.print(
            f"[green]Modulo {module.id} completo: {status.ratio} tests en verde.[/green]\n"
            f"[dim]Si quieres contrastar tu solucion con la explicada: "
            f"{module.solution_file.name}[/dim]"
        )
        nxt = prog.next_module(prog.load())
        if nxt is not None and nxt.id != module.id:
            console.print(f"Siguiente: [cyan]{nxt.id}[/cyan] - {nxt.title}")
    else:
        console.print(
            f"[yellow]Vas por {status.ratio} tests.[/yellow] "
            "Un test en rojo no es un fallo tuyo: es informacion. Lee el mensaje, que "
            "esta escrito para decirte que esperaba y que ha recibido."
        )
        if status.borrowed:
            console.print(
                "\nEjercicios que todavia no estan (el resto del repo tira de la "
                "referencia mientras tanto): "
                + ", ".join(f"[yellow]{n}[/yellow]" for n in status.borrowed)
            )
            first = status.borrowed[0]
            idx = next(
                (i + 1 for i, ex in enumerate(module.exercises) if ex.name == first), 1
            )
            console.print(
                f"[dim]Si te has quedado sin ideas: llmfs hint {module.number:02d} -e {idx}"
                f"  (tres niveles, cada vez mas explicito)[/dim]"
            )
    return 0 if code == 0 else 1


# ---------------------------------------------------------------------------- demo


def cmd_demo(args: argparse.Namespace) -> int:
    import runpy

    module = _resolve_module(args.modulo)
    if _module_missing(module):
        _warn_missing(module)
        return 2
    if not module.demo_file.exists():
        console.print(f"[yellow]{module.id} no tiene demo.py.[/yellow]")
        return 2

    console.rule(f"[bold cyan]demo {module.id}[/bold cyan] - {module.title}")
    old_argv = sys.argv[:]
    sys.argv = [str(module.demo_file), *args.extra]
    try:
        runpy.run_path(str(module.demo_file), run_name="__main__")
    finally:
        sys.argv = old_argv
    return 0


# ---------------------------------------------------------------------------- hint


def cmd_hint(args: argparse.Namespace) -> int:
    from llmfs.hints import get_hints

    module = _resolve_module(args.modulo)
    if not module.exercises:
        console.print(f"[yellow]{module.id} no tiene ejercicios.[/yellow]")
        return 2

    ref: str | int = args.ejercicio
    if isinstance(ref, str) and ref.isdigit():
        ref = int(ref)
    try:
        exercise = module.exercise(ref)
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        console.print("Ejercicios de este modulo:")
        for i, ex in enumerate(module.exercises, start=1):
            console.print(f"  {i}. [cyan]{ex.name}[/cyan] - {ex.title}")
        return 2

    hints = get_hints(module.id, exercise.name)
    if not hints:
        console.print(
            f"[yellow]Todavia no hay pistas escritas para {exercise.name}.[/yellow]\n"
            f"Mira {module.theory_file.name} y el docstring del ejercicio."
        )
        return 2

    if args.nivel is not None:
        level = max(1, min(args.nivel, len(hints)))
    else:
        seen = prog.get_hint_level(module.id, exercise.name)
        level = min(seen + 1, len(hints))
    prog.set_hint_level(module.id, exercise.name, level)

    text = Text()
    text.append(f"{exercise.name}\n", style="bold cyan")
    text.append(f"{exercise.title}\n\n", style="dim")
    text.append(hints[level - 1])

    console.print(
        Panel(
            text,
            title=f"pista {level}/{len(hints)}",
            border_style="yellow" if level < len(hints) else "red",
        )
    )
    if level < len(hints):
        console.print(
            f"[dim]Repite el comando para la pista {level + 1}, "
            f"o `llmfs hint {module.number:02d} -e {args.ejercicio} --nivel {level + 1}`.[/dim]"
        )
    else:
        console.print(
            f"[dim]Ultima pista. Si sigues atascado: {module.solution_file.name} "
            f"explica la solucion.[/dim]"
        )
    return 0


# ---------------------------------------------------------------------------- device


def cmd_device(args: argparse.Namespace) -> int:
    from llmfs.device import get_device

    cfg = get_device(prefer=args.dispositivo)
    console.print(Panel(cfg.summary(), title="hardware detectado", border_style="green"))
    return 0


# ---------------------------------------------------------------------------- stubs


_LATER = {
    "data": ("04_datos", "la preparacion del dataset"),
}


def cmd_train(args: argparse.Namespace) -> int:
    """Entrena un modelo. Se construye en el modulo 11 y se usa en el 13."""
    import torch

    from llmfs.config import RunConfig
    from llmfs.data import hacer_get_batch, preparar
    from llmfs.device import get_device
    from llmfs.paths import configs_dir
    from llmfs.train import Entrenador

    ruta = Path(args.config)
    if not ruta.exists():
        ruta = configs_dir() / args.config
    if not ruta.exists() and not str(ruta).endswith(".yaml"):
        ruta = configs_dir() / f"{args.config}.yaml"
    if not ruta.exists():
        console.print(f"[red]No encuentro el config {args.config}[/red]")
        console.print(
            "Disponibles: " + ", ".join(p.stem for p in configs_dir().glob("*.yaml"))
        )
        return 2

    cfg = RunConfig.from_yaml(ruta)
    if args.max_steps is not None:
        cfg.train.max_tokens = args.max_steps * cfg.tokens_per_step
    if args.device:
        cfg.train.device = args.device

    dispositivo = get_device(prefer=cfg.train.device, amp=cfg.train.amp)
    console.print(Panel(cfg.summary(), title="configuracion", border_style="cyan"))
    console.print(dispositivo.summary() + "\n")

    dataset = preparar(cfg)
    if dataset.vocab_size != cfg.model.vocab_size:
        console.print(
            f"[yellow]El dataset tiene {dataset.vocab_size} tokens y el config dice "
            f"{cfg.model.vocab_size}. Uso el del dataset.[/yellow]"
        )
        cfg.model.vocab_size = dataset.vocab_size

    GPT = _resolve_gpt()
    modelo = GPT(cfg.model)
    count_parameters = __import__(
        "llmfs.bridge", fromlist=["resolve"]
    ).resolve("10_el_gpt_completo", "count_parameters")
    conteo = count_parameters(modelo)
    console.print(
        f"modelo: [bold]{conteo['total']:,}[/bold] parametros "
        f"({conteo['non_embedding']:,} no-embedding)\n"
    )

    get_batch = hacer_get_batch(dataset, cfg, dispositivo.device)

    def muestra(step: int) -> str:
        modelo.eval()
        inicio = torch.tensor(
            [dataset.encode(args.prompt or "\n")], dtype=torch.long, device=dispositivo.device
        )
        with torch.no_grad():
            salida = modelo.generate(inicio, max_new_tokens=200, temperature=0.8, top_k=40)
        modelo.train()
        texto = dataset.decode(salida[0].tolist())
        console.print(Panel(texto, title=f"muestra del paso {step:,}", border_style="dim"))
        return texto

    entrenador = Entrenador(
        cfg, modelo, get_batch, device=dispositivo, on_sample=muestra
    )

    reanudar = None
    if args.resume:
        candidato = cfg.run_dir / "last.pt"
        if candidato.exists():
            reanudar = candidato
        else:
            console.print("[yellow]No hay checkpoint que reanudar; empiezo de cero.[/yellow]")

    estado = entrenador.entrenar(reanudar=reanudar, console=console)

    console.print(
        f"\n[green]Terminado.[/green] {estado.step:,} pasos, "
        f"{estado.tokens_vistos:,} tokens, mejor perdida de validacion "
        f"[bold]{estado.best_val_loss:.4f}[/bold]"
    )
    console.print(f"checkpoints y curvas en [cyan]{cfg.run_dir}[/cyan]")
    return 0


def _resolve_gpt():
    from llmfs.bridge import resolve

    return resolve("10_el_gpt_completo", "GPT")


def cmd_sample(args: argparse.Namespace) -> int:
    """Genera texto a partir de un checkpoint entrenado. Se construye en el modulo 14."""
    import torch

    from llmfs.bridge import resolve
    from llmfs.config import RunConfig
    from llmfs.data import preparar
    from llmfs.device import get_device
    from llmfs.paths import configs_dir
    from llmfs.train import cargar_checkpoint

    if args.checkpoint:
        ruta = Path(args.checkpoint)
    else:
        ruta_config = Path(args.config)
        if not ruta_config.exists():
            ruta_config = configs_dir() / args.config
        if not ruta_config.exists() and not str(ruta_config).endswith(".yaml"):
            ruta_config = configs_dir() / f"{args.config}.yaml"
        if not ruta_config.exists():
            console.print(f"[red]No encuentro el config {args.config}[/red]")
            return 2
        ruta = RunConfig.from_yaml(ruta_config).run_dir / "best.pt"

    if not ruta.exists():
        console.print(
            f"[red]No hay checkpoint en {ruta}.[/red] Entrena uno primero con "
            f"[cyan]llmfs train --config {args.config}[/cyan]."
        )
        return 2

    payload = cargar_checkpoint(ruta, map_location="cpu")
    cfg = RunConfig.from_dict(payload["config"])

    dispositivo = get_device(prefer=args.device, amp=cfg.train.amp)
    dataset = preparar(cfg, quiet=True)

    GPT = _resolve_gpt()
    modelo = GPT(cfg.model)
    modelo.load_state_dict(payload["model"])
    modelo.to(dispositivo.device)
    modelo.eval()

    generate_with_cache = resolve("14_inferencia", "generate_with_cache")
    prompt = args.prompt or "\n"
    idx = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=dispositivo.device)

    salida = generate_with_cache(
        modelo,
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    texto = dataset.decode(salida[0].tolist())
    console.print(Panel(texto, title=f"muestra de {ruta}", border_style="green"))
    return 0


def _make_stub(name: str) -> Any:
    def stub(args: argparse.Namespace) -> int:
        module_id, what = _LATER[name]
        console.print(
            Panel(
                f"`llmfs {name}` implementa {what}, que se construye en el modulo "
                f"[cyan]{module_id}[/cyan].\n"
                f"Todavia no esta escrito en el repo.",
                title="Aun no disponible",
                border_style="yellow",
            )
        )
        return 2

    return stub


# ---------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmfs",
        description="Curso 'LLM desde cero': progreso, tests, demos y pistas.",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p_status = sub.add_parser("status", help="tabla de progreso de todos los modulos")
    p_status.add_argument(
        "--cached",
        action="store_true",
        help="usa el ultimo resultado guardado en vez de volver a correr los tests",
    )
    p_status.set_defaults(func=cmd_status)

    p_next = sub.add_parser("next", help="en que modulo estas y que ejercicio toca")
    p_next.set_defaults(func=cmd_next)

    p_check = sub.add_parser("check", help="corre los tests de un modulo")
    p_check.add_argument("modulo", help="numero (6, 06) o id (06_atencion)")
    p_check.add_argument("-k", help="filtro de pytest, p.ej. -k causal", default=None)
    p_check.set_defaults(func=cmd_check)

    p_demo = sub.add_parser("demo", help="ejecuta el experimento de un modulo")
    p_demo.add_argument("modulo", help="numero (6, 06) o id (06_atencion)")
    # REMAINDER y no "*": asi `llmfs demo 03 --rapido` pasa el flag a demo.py en lugar de
    # que argparse lo intente interpretar como una opcion suya y falle.
    p_demo.add_argument(
        "extra", nargs=argparse.REMAINDER, help="argumentos que se pasan tal cual a demo.py"
    )
    p_demo.set_defaults(func=cmd_demo)

    p_hint = sub.add_parser("hint", help="pista progresiva de un ejercicio")
    p_hint.add_argument("modulo", help="numero (6, 06) o id (06_atencion)")
    p_hint.add_argument(
        "-e", "--ejercicio", required=True, help="indice (1, 2, 3...) o nombre del ejercicio"
    )
    p_hint.add_argument(
        "--nivel", type=int, default=None, help="fuerza un nivel de pista (1-3)"
    )
    p_hint.set_defaults(func=cmd_hint)

    p_dev = sub.add_parser("device", help="hardware detectado y politica de precision")
    p_dev.add_argument("--dispositivo", default=None, help="fuerza cuda, mps o cpu")
    p_dev.set_defaults(func=cmd_device)

    p_train = sub.add_parser("train", help="entrena un modelo")
    p_train.add_argument(
        "--config", default="tiny_char", help="nombre o ruta del YAML (por defecto tiny_char)"
    )
    p_train.add_argument("--resume", action="store_true", help="reanuda desde el ultimo checkpoint")
    p_train.add_argument("--max-steps", type=int, default=None, help="acorta la tirada")
    p_train.add_argument("--device", default=None, help="fuerza cuda, mps o cpu")
    p_train.add_argument("--prompt", default=None, help="prompt para las muestras periodicas")
    p_train.set_defaults(func=cmd_train)

    p_sample = sub.add_parser("sample", help="genera texto a partir de un checkpoint entrenado")
    p_sample.add_argument(
        "--checkpoint", default=None,
        help="ruta a un fichero .pt (por defecto: <out_dir>/<config>/best.pt)",
    )
    p_sample.add_argument(
        "--config", default="tiny_char",
        help="se usa para encontrar el checkpoint por defecto si no se pasa --checkpoint",
    )
    p_sample.add_argument("--prompt", default=None, help="texto a continuar (por defecto: vacio)")
    p_sample.add_argument("--max-new-tokens", type=int, default=200)
    p_sample.add_argument("--temperature", type=float, default=0.8)
    p_sample.add_argument("--top-k", type=int, default=40)
    p_sample.add_argument("--top-p", type=float, default=None)
    p_sample.add_argument("--repetition-penalty", type=float, default=1.0)
    p_sample.add_argument("--device", default=None, help="fuerza cuda, mps o cpu")
    p_sample.set_defaults(func=cmd_sample)

    for name, (module_id, _) in _LATER.items():
        p = sub.add_parser(name, help=f"(modulo {module_id})")
        p.add_argument("rest", nargs=argparse.REMAINDER)
        p.set_defaults(func=_make_stub(name))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        console.print("\n[dim]interrumpido[/dim]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
