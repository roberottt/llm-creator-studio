"""The course CLI: `python -m llmfs ...` (or just `llmfs ...`).

    llmfs status              progress table for the 18 modules
    llmfs next                which module you are on and which exercise is next
    llmfs check 06            runs the tests of module 06, with hints if they fail
    llmfs demo 06             runs the experiment of module 06
    llmfs hint 06 -e 2        progressive hint for exercise 2 (repeat for more)
    llmfs device              what hardware was detected and what precision it will use

Torch is imported inside each command, not at the top: `llmfs next` answers in tenths of a
second and there is no reason to pay the long second it costs to load torch.
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
    """`True` if the module has not been written in the repo yet."""
    return not module.path.exists()


def _warn_missing(module: Module) -> None:
    console.print(
        Panel(
            f"Module [bold]{module.id}[/bold] does not exist in the repo yet.\n"
            f"It gets written in the corresponding phase of the build plan.\n\n"
            f"Modules available right now: "
            + (", ".join(m.id for m in all_modules() if m.path.exists()) or "none"),
            title="Not available yet",
            border_style="yellow",
        )
    )


# ---------------------------------------------------------------------------- status


def cmd_status(args: argparse.Namespace) -> int:
    modules = list(all_modules())

    if args.cached:
        results = prog.load()
        if not results:
            console.print("[yellow]No cached progress. Running the tests...[/yellow]")
            results = _refresh(modules)
    else:
        results = _refresh(modules)

    table = Table(
        title="LLM from scratch - curriculum progress",
        title_style="bold",
        header_style="bold",
        expand=False,
    )
    table.add_column("", width=2, justify="center")
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("module")
    table.add_column("tests", justify="right")
    table.add_column("mine", justify="right")
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
        f"\n  [bold]{done}/{total}[/bold] modules complete ({pct:.0f}%)  |  "
        f"[bold]{mine_total}/{total_exercises()}[/bold] exercises implemented  |  "
        f"estimated work left: {_fmt_minutes(_remaining_minutes(results))} "
        f"of {_fmt_minutes(total_minutes())}"
    )

    nxt = prog.next_module(results)
    if nxt is not None:
        console.print(f"  next: [cyan]{nxt.id}[/cyan] - {nxt.summary}")
        console.print(f"  [dim]llmfs next[/dim] for the detail\n")
    else:
        console.print("  [green]Curriculum complete. Congratulations.[/green]\n")
    return 0


def _remaining_minutes(results: dict[str, prog.ModuleStatus]) -> int:
    return sum(
        m.est_minutes
        for m in all_modules()
        if (results.get(m.id) or prog.ModuleStatus(module_id=m.id)).state != "done"
    )


def _refresh(modules: list[Module]) -> dict[str, prog.ModuleStatus]:
    results: dict[str, prog.ModuleStatus] = {}
    with console.status("[dim]running tests...[/dim]") as spinner:
        for module in modules:
            spinner.update(f"[dim]running the tests of {module.id}...[/dim]")
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
        console.print("[green]You have finished every module. Go train the final model.[/green]")
        return 0

    status = results.get(module.id) or prog.ModuleStatus(module_id=module.id)

    body = [
        f"[bold cyan]{module.id}[/bold cyan] - {module.title}",
        f"[dim]{module.part} | about {_fmt_minutes(module.est_minutes)} of work[/dim]",
        "",
        f"[bold]What you will learn:[/bold] {module.summary}",
    ]

    if _module_missing(module):
        body += ["", "[yellow]This module has not been written in the repo yet.[/yellow]"]
    else:
        # What is pending is derived from `mine`, not from `borrowed`: with no cached data
        # both lists are empty, and in that case the right answer is "you have not done
        # anything yet", not "you have it all done".
        pending = [ex for ex in module.exercises if ex.name not in status.mine]
        done = len(status.mine)

        if done == 0:
            body += [
                "",
                "You have not started this module yet. The path is always the same:",
                "",
                "  1. read the theory (10-15 min, do not skip it)",
                "  2. open exercises.py and read the first problem statement",
                "  3. implement it and run the tests",
                "  4. red -> hint; green -> next exercise",
            ]
        else:
            body += [
                "",
                f"[bold]You are {done} of {len(module.exercises)} exercises in.[/bold] "
                f"Tests: {status.ratio}",
            ]

        if pending:
            first = pending[0]
            idx = module.exercises.index(first) + 1
            body += [
                "",
                f"[bold]Next up is exercise {idx}:[/bold] [yellow]{first.name}[/yellow]",
                f"  {first.title}",
            ]
            if len(pending) > 1:
                body.append(f"  [dim]{len(pending) - 1} more after that one[/dim]")
        elif status.state != "done":
            body += [
                "",
                "You have implemented every exercise but some test is still red.",
                "That is normal, and it is exactly the useful part of the exercise: the",
                "test is telling you precisely what does not add up.",
            ]
            if status.first_failure:
                body.append(f"  [red]{status.first_failure}[/red]")

        body += [
            "",
            f"[dim]1. read :[/dim] {module.theory_file.relative_to(module.path.parent.parent)}",
            f"[dim]2. edit :[/dim] {module.exercises_file.relative_to(module.path.parent.parent)}",
            f"[dim]3. test :[/dim] llmfs check {module.number:02d}",
            f"[dim]   stuck:[/dim] llmfs hint {module.number:02d} -e 1  "
            f"[dim](repeat the command for more explicit hints)[/dim]",
            f"[dim]   see  :[/dim] llmfs demo {module.number:02d}  "
            f"[dim](the concept, in plots and numbers)[/dim]",
        ]

    console.print(Panel("\n".join(body), title="Where you pick up", border_style="cyan"))
    return 0


# ---------------------------------------------------------------------------- check


def cmd_check(args: argparse.Namespace) -> int:
    module = _resolve_module(args.module)
    if _module_missing(module):
        _warn_missing(module)
        return 2
    if not module.test_file.exists():
        console.print(f"[yellow]{module.id} has no test_{module.number:02d}.py yet.[/yellow]")
        return 2

    import pytest

    console.rule(f"[bold cyan]{module.id}[/bold cyan] - {module.title}")
    pytest_args = [str(module.test_file), "-v", "--tb=short", "--no-header", "-p", "no:cacheprovider"]
    if args.k:
        pytest_args += ["-k", args.k]
    code = pytest.main(pytest_args)

    # Recompute and persist the module's state after the run.
    status = prog.run_module_tests(module, quiet=True)
    prog.save({module.id: status})

    console.rule(style="dim")
    if status.state == "done":
        console.print(
            f"[green]Module {module.id} complete: {status.ratio} tests green.[/green]\n"
            f"[dim]If you want to compare your solution with the explained one: "
            f"{module.solution_file.name}[/dim]"
        )
        nxt = prog.next_module(prog.load())
        if nxt is not None and nxt.id != module.id:
            console.print(f"Next: [cyan]{nxt.id}[/cyan] - {nxt.title}")
    else:
        console.print(
            f"[yellow]You are at {status.ratio} tests.[/yellow] "
            "A red test is not your failure: it is information. Read the message, which "
            "is written to tell you what it expected and what it got."
        )
        if status.borrowed:
            console.print(
                "\nExercises that are not there yet (the rest of the repo uses the "
                "reference in the meantime): "
                + ", ".join(f"[yellow]{n}[/yellow]" for n in status.borrowed)
            )
            first = status.borrowed[0]
            idx = next(
                (i + 1 for i, ex in enumerate(module.exercises) if ex.name == first), 1
            )
            console.print(
                f"[dim]If you have run out of ideas: llmfs hint {module.number:02d} -e {idx}"
                f"  (three levels, each more explicit)[/dim]"
            )
    return 0 if code == 0 else 1


# ---------------------------------------------------------------------------- demo


def cmd_demo(args: argparse.Namespace) -> int:
    import runpy

    module = _resolve_module(args.module)
    if _module_missing(module):
        _warn_missing(module)
        return 2
    if not module.demo_file.exists():
        console.print(f"[yellow]{module.id} has no demo.py.[/yellow]")
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

    module = _resolve_module(args.module)
    if not module.exercises:
        console.print(f"[yellow]{module.id} has no exercises.[/yellow]")
        return 2

    ref: str | int = args.exercise
    if isinstance(ref, str) and ref.isdigit():
        ref = int(ref)
    try:
        exercise = module.exercise(ref)
    except KeyError as exc:
        console.print(f"[red]{exc.args[0]}[/red]")
        console.print("Exercises in this module:")
        for i, ex in enumerate(module.exercises, start=1):
            console.print(f"  {i}. [cyan]{ex.name}[/cyan] - {ex.title}")
        return 2

    hints = get_hints(module.id, exercise.name)
    if not hints:
        console.print(
            f"[yellow]There are no hints written for {exercise.name} yet.[/yellow]\n"
            f"Look at {module.theory_file.name} and the exercise docstring."
        )
        return 2

    if args.level is not None:
        level = max(1, min(args.level, len(hints)))
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
            title=f"hint {level}/{len(hints)}",
            border_style="yellow" if level < len(hints) else "red",
        )
    )
    if level < len(hints):
        console.print(
            f"[dim]Repeat the command for hint {level + 1}, "
            f"or `llmfs hint {module.number:02d} -e {args.exercise} --level {level + 1}`.[/dim]"
        )
    else:
        console.print(
            f"[dim]Last hint. If you are still stuck: {module.solution_file.name} "
            f"explains the solution.[/dim]"
        )
    return 0


# ---------------------------------------------------------------------------- device


def cmd_device(args: argparse.Namespace) -> int:
    from llmfs.device import get_device

    cfg = get_device(prefer=args.device)
    console.print(Panel(cfg.summary(), title="detected hardware", border_style="green"))
    return 0


# ---------------------------------------------------------------------------- stubs


_LATER = {
    "sample": ("14_inference", "text generation"),
    "data": ("04_data", "dataset preparation"),
}


def cmd_train(args: argparse.Namespace) -> int:
    """Train a model. Built in module 11 and used in module 13."""
    import torch

    from llmfs.config import RunConfig
    from llmfs.data import make_get_batch, prepare
    from llmfs.device import get_device
    from llmfs.paths import configs_dir
    from llmfs.train import Trainer

    path = Path(args.config)
    if not path.exists():
        path = configs_dir() / args.config
    if not path.exists() and not str(path).endswith(".yaml"):
        path = configs_dir() / f"{args.config}.yaml"
    if not path.exists():
        console.print(f"[red]Cannot find the config {args.config}[/red]")
        console.print(
            "Available: " + ", ".join(p.stem for p in configs_dir().glob("*.yaml"))
        )
        return 2

    cfg = RunConfig.from_yaml(path)
    if args.max_steps is not None:
        cfg.train.max_tokens = args.max_steps * cfg.tokens_per_step
    if args.device:
        cfg.train.device = args.device

    device = get_device(prefer=cfg.train.device, amp=cfg.train.amp)
    console.print(Panel(cfg.summary(), title="configuration", border_style="cyan"))
    console.print(device.summary() + "\n")

    dataset = prepare(cfg)
    if dataset.vocab_size != cfg.model.vocab_size:
        console.print(
            f"[yellow]The dataset has {dataset.vocab_size} tokens and the config says "
            f"{cfg.model.vocab_size}. Using the dataset's.[/yellow]"
        )
        cfg.model.vocab_size = dataset.vocab_size

    GPT = _resolve_gpt()
    model = GPT(cfg.model)
    count_parameters = __import__(
        "llmfs.bridge", fromlist=["resolve"]
    ).resolve("10_the_full_gpt", "count_parameters")
    counts = count_parameters(model)
    console.print(
        f"model: [bold]{counts['total']:,}[/bold] parameters "
        f"({counts['non_embedding']:,} non-embedding)\n"
    )

    get_batch = make_get_batch(dataset, cfg, device.device)

    def sample(step: int) -> str:
        model.eval()
        start = torch.tensor(
            [dataset.encode(args.prompt or "\n")], dtype=torch.long, device=device.device
        )
        with torch.no_grad():
            out = model.generate(start, max_new_tokens=200, temperature=0.8, top_k=40)
        model.train()
        text = dataset.decode(out[0].tolist())
        console.print(Panel(text, title=f"sample at step {step:,}", border_style="dim"))
        return text

    trainer = Trainer(cfg, model, get_batch, device=device, on_sample=sample)

    resume_from = None
    if args.resume:
        candidate = cfg.run_dir / "last.pt"
        if candidate.exists():
            resume_from = candidate
        else:
            console.print("[yellow]No checkpoint to resume from; starting over.[/yellow]")

    state = trainer.train(resume=resume_from, console=console)

    console.print(
        f"\n[green]Done.[/green] {state.step:,} steps, "
        f"{state.tokens_seen:,} tokens, best validation loss "
        f"[bold]{state.best_val_loss:.4f}[/bold]"
    )
    console.print(f"checkpoints and curves in [cyan]{cfg.run_dir}[/cyan]")
    return 0


def _resolve_gpt():
    from llmfs.bridge import resolve

    return resolve("10_the_full_gpt", "GPT")


def _make_stub(name: str) -> Any:
    def stub(args: argparse.Namespace) -> int:
        module_id, what = _LATER[name]
        console.print(
            Panel(
                f"`llmfs {name}` implements {what}, which is built in module "
                f"[cyan]{module_id}[/cyan].\n"
                f"It has not been written in the repo yet.",
                title="Not available yet",
                border_style="yellow",
            )
        )
        return 2

    return stub


# ---------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmfs",
        description="'LLM from scratch' course: progress, tests, demos and hints.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="progress table for every module")
    p_status.add_argument(
        "--cached",
        action="store_true",
        help="use the last saved result instead of running the tests again",
    )
    p_status.set_defaults(func=cmd_status)

    p_next = sub.add_parser("next", help="which module you are on and which exercise is next")
    p_next.set_defaults(func=cmd_next)

    p_check = sub.add_parser("check", help="run a module's tests")
    p_check.add_argument("module", help="number (6, 06) or id (06_attention)")
    p_check.add_argument("-k", help="pytest filter, e.g. -k causal", default=None)
    p_check.set_defaults(func=cmd_check)

    p_demo = sub.add_parser("demo", help="run a module's experiment")
    p_demo.add_argument("module", help="number (6, 06) or id (06_attention)")
    # REMAINDER instead of "*": this way `llmfs demo 03 --fast` passes the flag on to
    # demo.py rather than argparse trying to read it as one of its own options and failing.
    p_demo.add_argument(
        "extra", nargs=argparse.REMAINDER, help="arguments passed through to demo.py as-is"
    )
    p_demo.set_defaults(func=cmd_demo)

    p_hint = sub.add_parser("hint", help="progressive hint for an exercise")
    p_hint.add_argument("module", help="number (6, 06) or id (06_attention)")
    p_hint.add_argument(
        "-e", "--exercise", required=True, help="index (1, 2, 3...) or exercise name"
    )
    p_hint.add_argument(
        "--level", type=int, default=None, help="force a hint level (1-3)"
    )
    p_hint.set_defaults(func=cmd_hint)

    p_dev = sub.add_parser("device", help="detected hardware and precision policy")
    p_dev.add_argument("--device", default=None, help="force cuda, mps or cpu")
    p_dev.set_defaults(func=cmd_device)

    p_train = sub.add_parser("train", help="train a model")
    p_train.add_argument(
        "--config", default="tiny_char", help="name or path of the YAML (tiny_char by default)"
    )
    p_train.add_argument("--resume", action="store_true", help="resume from the last checkpoint")
    p_train.add_argument("--max-steps", type=int, default=None, help="shorten the run")
    p_train.add_argument("--device", default=None, help="force cuda, mps or cpu")
    p_train.add_argument("--prompt", default=None, help="prompt for the periodic samples")
    p_train.set_defaults(func=cmd_train)

    for name, (module_id, _) in _LATER.items():
        p = sub.add_parser(name, help=f"(module {module_id})")
        p.add_argument("rest", nargs=argparse.REMAINDER)
        p.set_defaults(func=_make_stub(name))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        console.print("\n[dim]interrupted[/dim]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
