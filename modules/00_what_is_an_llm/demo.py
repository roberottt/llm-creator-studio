"""Demo for module 00: your first language model, generating real text.

    llmfs demo 00

It trains counting models with contexts of 1, 2, 3, 4 and 6 characters over Shakespeare and
shows you what each one writes. The jump in quality between the one with 1 and the one with
4 is the whole argument of the course: looking further back helps enormously... until the
table does not fit on disk. The final plot shows exactly that.
"""

from __future__ import annotations

import random

import matplotlib

matplotlib.use("Agg")  # no window: this has to run over SSH and in CI

import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.data import fetch_tinyshakespeare
from llmfs.paths import figures_dir
from llmfs.reference import build_count_table

console = Console()

next_token_probs = resolve("00_what_is_an_llm", "next_token_probs")
sample_next_token = resolve("00_what_is_an_llm", "sample_next_token")
generate_naive = resolve("00_what_is_an_llm", "generate_naive")

CONTEXTS = [1, 2, 3, 4, 6]


def table_size(vocab: int, context: int) -> float:
    """How many entries the table WOULD have if every combination were present."""
    return float(vocab) ** context


def main() -> None:
    text, origin = fetch_tinyshakespeare()
    if origin == "fallback":
        console.print("[yellow]No connection: using the fallback text (much shorter).[/yellow]")

    vocab = sorted(set(text))
    console.print(
        Panel(
            f"corpus    : {len(text):,} characters\n"
            f"vocabulary: {len(vocab)} distinct characters\n"
            f"sample    : {text[:80]!r}",
            title="the training text",
            border_style="cyan",
        )
    )

    # ------------------------------------------------------------------ the distribution
    table_1 = build_count_table(text, context_size=1)
    console.rule("[bold]1. What is inside the model[/bold]")
    console.print("After the letter 'q', this is what the model believes can come next:\n")

    probs_q = next_token_probs(table_1["q"])
    probs_table = Table(header_style="bold")
    probs_table.add_column("character")
    probs_table.add_column("probability", justify="right")
    probs_table.add_column("", width=30)
    for char, p in sorted(probs_q.items(), key=lambda kv: -kv[1])[:6]:
        probs_table.add_row(repr(char), f"{p:.4f}", "#" * int(p * 30))
    console.print(probs_table)
    console.print(
        f"[dim]They sum to {sum(probs_q.values()):.6f}. That is a probability "
        f"distribution, and it is THE output of any language model.[/dim]"
    )

    # ------------------------------------------------------------------ generation
    console.rule("[bold]2. What each model writes[/bold]")
    realism: list[float] = []
    real_words = set(text.lower().split())

    for n in CONTEXTS:
        table = build_count_table(text, context_size=n)
        start = text[:n]
        out = generate_naive(table, start, length=240, rng=random.Random(1234))

        # Fraction of generated words that really exist in the corpus. It is the honest
        # measure: counting trigrams would give 100% whenever the context is >= 2, because
        # by construction every generated trigram already existed.
        generated = out.lower().split()
        pct = 100 * sum(w in real_words for w in generated) / max(1, len(generated))
        realism.append(pct)

        console.print(
            Panel(
                out.replace("\n", " "),
                title=f"context of {n} character{'s' if n > 1 else ''}  "
                f"| {len(table):,} contexts seen | {pct:.0f}% real words",
                border_style="green" if pct > 80 else "yellow" if pct > 50 else "red",
            )
        )

    # ------------------------------------------------------------------ the wall
    console.rule("[bold]3. Why this does not scale[/bold]")
    wall_table = Table(header_style="bold")
    wall_table.add_column("context")
    wall_table.add_column("contexts seen", justify="right")
    wall_table.add_column("possible combinations", justify="right")
    wall_table.add_column("% covered", justify="right")

    for n in CONTEXTS:
        seen = len(build_count_table(text, context_size=n))
        possible = table_size(len(vocab), n)
        wall_table.add_row(
            f"{n} ch.", f"{seen:,}", f"{possible:.3g}", f"{100 * seen / possible:.2g}%"
        )
    console.print(wall_table)

    console.print(
        "\n[bold]Look at the last column.[/bold] With a context of 6 characters, the corpus\n"
        "covers a ridiculous fraction of the possible combinations. And this is at the\n"
        "CHARACTER level. Our final model will use 4096 tokens and a context of 512:\n"
        f"[red]4096^512[/red] combinations, a number with more than 1800 digits.\n\n"
        "Counting cannot solve this. That is why neural networks are needed: instead of\n"
        "memorizing every context, they learn that 'cat' and 'dog' are similar and share\n"
        "what they learn between them. That is module 05 onwards."
    )

    # ------------------------------------------------------------------ plot
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    left.plot(CONTEXTS, realism, marker="o", color="tab:green")
    left.set_xlabel("characters of context")
    left.set_ylabel("% of words that exist in the original")
    left.set_title("More context -> more realistic text")
    left.set_ylim(0, 105)
    left.grid(alpha=0.3)

    seen = [len(build_count_table(text, context_size=n)) for n in CONTEXTS]
    possible = [table_size(len(vocab), n) for n in CONTEXTS]
    right.plot(CONTEXTS, possible, marker="o", label="possible combinations")
    right.plot(CONTEXTS, seen, marker="s", label="contexts seen in the corpus")
    right.set_yscale("log")
    right.set_xlabel("characters of context")
    right.set_ylabel("number of contexts (log scale)")
    right.set_title("...but the table explodes")
    right.grid(alpha=0.3)
    right.legend()

    fig.tight_layout()
    target = figures_dir() / "00_counting_vs_generalization.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")


if __name__ == "__main__":
    main()
