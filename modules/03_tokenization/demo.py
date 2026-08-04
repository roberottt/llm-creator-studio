"""Demo for module 03: what a BPE really learns, and how much it compresses.

    llmfs demo 03
    llmfs demo 03 --fast      smaller sample, takes half the time

Four things:
  1. The first merges it learns, in order. You can see it go from letters to syllables and
     from syllables to whole words.
  2. The same sentence tokenized with vocabularies of different sizes.
  3. The compression curve, and next to it the parameter cost of the embedding table. That
     is the trade-off that decides the final model's vocab_size.
  4. A comparison with tiktoken (GPT-4's tokenizer), if it is installed.

Training BPE in pure python is not fast: count on one or two minutes.
"""

from __future__ import annotations

import sys
import time

import matplotlib

matplotlib.use("Agg")  # no window: this has to run over SSH and in CI

import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.data import fetch_tinyshakespeare
from llmfs.paths import figures_dir

console = Console()

train_bpe = resolve("03_tokenization", "train_bpe")
bpe_encode = resolve("03_tokenization", "bpe_encode")
bpe_decode = resolve("03_tokenization", "bpe_decode")

SENTENCE = "The king shall speak to his people tomorrow morning."
D_MODEL = 320  # the final model's, to compute the embedding table's cost


def main() -> None:
    fast = "--fast" in sys.argv
    text, origin = fetch_tinyshakespeare()
    sample = text[: 60_000 if fast else 200_000]
    vocabularies = [300, 512, 1024] if fast else [300, 512, 1024, 2048]

    console.print(
        f"Training on {len(sample):,} characters "
        f"({'reduced sample' if fast else 'normal sample'}).\n"
    )

    # ------------------------------------------------------------- 1. what it learns
    console.rule("[bold]1. The first merges, in order[/bold]")
    started = time.perf_counter()
    merges_512, vocab_512 = train_bpe(sample, 512)
    console.print(f"[dim]512 tokens trained in {time.perf_counter() - started:.1f} s[/dim]\n")

    table = Table(header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("id", justify="right")
    table.add_column("new token")
    table.add_column("formed from")

    items = list(merges_512.items())
    for n, ((a, b), new) in enumerate(items[:12] + items[-6:]):
        if n == 12:
            table.add_row("...", "...", "...", "...")
        pa = vocab_512[a].decode("utf-8", errors="replace")
        pb = vocab_512[b].decode("utf-8", errors="replace")
        table.add_row(
            str(items.index(((a, b), new)) + 1),
            str(new),
            repr(vocab_512[new].decode("utf-8", errors="replace")),
            f"{pa!r} + {pb!r}",
        )
    console.print(table)
    console.print(
        "[dim]The first merges are pairs of very frequent letters. The last ones are\n"
        "already whole words or complete endings. Nobody told the algorithm that words\n"
        "exist: it deduced them from the frequencies.[/dim]"
    )

    # ------------------------------------------------------------- 2. the same sentence
    console.rule("[bold]2. The same sentence, with different vocabularies[/bold]")
    console.print(f"Sentence: [cyan]{SENTENCE}[/cyan]\n")

    trained: dict[int, tuple[dict, dict]] = {}
    ratios: list[float] = []
    for vs in vocabularies:
        merges, vocab = train_bpe(sample, vs)
        trained[vs] = (merges, vocab)

        ids = bpe_encode(SENTENCE, merges)
        pieces = [bpe_decode([i], vocab) for i in ids]
        console.print(f"[bold]vocab {vs}[/bold] -> {len(ids)} tokens")
        console.print("  " + " | ".join(p.replace("\n", "\\n") for p in pieces) + "\n")

        sample_ids = bpe_encode(sample[:20_000], merges)
        ratios.append(len(sample[:20_000].encode("utf-8")) / len(sample_ids))

    # ------------------------------------------------------------- 3. the trade-off
    console.rule("[bold]3. The trade-off that decides vocab_size[/bold]")
    table2 = Table(header_style="bold")
    table2.add_column("vocabulary", justify="right")
    table2.add_column("bytes/token", justify="right")
    table2.add_column("tokens for 1 MB", justify="right")
    table2.add_column(f"embedding params (d={D_MODEL})", justify="right")
    table2.add_column("% of an 8.9M model", justify="right")

    for vs, ratio in zip(vocabularies, ratios):
        params = vs * D_MODEL
        table2.add_row(
            f"{vs:,}",
            f"{ratio:.2f}",
            f"{1_000_000 / ratio / 1e3:.0f}k",
            f"{params:,}",
            f"{100 * params / 8_933_440:.0f}%",
        )
    for vs in (4096, 32000, 50257):
        params = vs * D_MODEL
        table2.add_row(
            f"{vs:,}",
            "[dim]?[/dim]",
            "[dim]?[/dim]",
            f"{params:,}",
            f"{100 * params / 8_933_440:.0f}%",
            style="dim" if vs != 4096 else "bold",
        )
    console.print(table2)
    console.print(
        "\nA larger vocabulary compresses better (fewer tokens per text, shorter\n"
        "sequences, fewer training steps) but it eats the parameter budget on a lookup\n"
        "table. With 50,257 tokens like GPT-2, the embedding table alone would be\n"
        "[red]almost twice[/red] our whole model.\n"
        "[bold]That is why the final model uses 4,096.[/bold]"
    )

    # ------------------------------------------------------------- 4. tiktoken
    console.rule("[bold]4. Comparison with tiktoken (GPT-4)[/bold]")
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        gpt4_ids = enc.encode(sample[:20_000])
        gpt4_ratio = len(sample[:20_000].encode("utf-8")) / len(gpt4_ids)
        gpt4_pieces = [enc.decode([i]) for i in enc.encode(SENTENCE)]

        console.print(f"cl100k_base has [bold]{enc.n_vocab:,}[/bold] tokens.")
        console.print(f"  compression : {gpt4_ratio:.2f} bytes/token")
        console.print(f"  the sentence: {len(gpt4_pieces)} tokens")
        console.print("  " + " | ".join(gpt4_pieces))
        better = gpt4_ratio / ratios[-1]
        console.print(
            f"\nIt compresses [bold]{better:.1f}x[/bold] better than our vocab of "
            f"{vocabularies[-1]}, with {enc.n_vocab / vocabularies[-1]:.0f}x more tokens.\n"
            "[dim]Diminishing returns: multiplying the vocabulary by 50 does not multiply "
            "the compression by 50.[/dim]"
        )
    except ImportError:
        console.print(
            "[yellow]tiktoken is not installed. Install it with `uv sync --extra compare` "
            "to see this comparison.[/yellow]"
        )

    # ------------------------------------------------------------- plot
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    left.plot(vocabularies, ratios, marker="o", color="tab:blue")
    left.set_xscale("log", base=2)
    left.set_xlabel("vocabulary size")
    left.set_ylabel("bytes per token")
    left.set_title("More vocabulary compresses better...")
    left.grid(alpha=0.3)

    all_sizes = [*vocabularies, 4096, 32000, 50257]
    right.plot(all_sizes, [v * D_MODEL / 1e6 for v in all_sizes], marker="o", color="tab:red")
    right.axhline(8.93, color="gray", ls="--", lw=1, label="whole model (8.9M)")
    right.axvline(4096, color="tab:green", ls=":", lw=2, label="our choice")
    right.set_xscale("log", base=2)
    right.set_xlabel("vocabulary size")
    right.set_ylabel("millions of parameters in embeddings alone")
    right.set_title("...but it eats the model")
    right.grid(alpha=0.3)
    right.legend()

    fig.tight_layout()
    target = figures_dir() / "03_tokenization.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")


if __name__ == "__main__":
    main()
