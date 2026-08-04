"""Demo for module 04: the complete data pipeline, with real files.

    llmfs demo 04

It does the whole journey: text -> tokens -> binary file -> memmap -> batch on the GPU.
And along the way it shows:
  1. How much the corpus takes in each format, and why uint16.
  2. A real batch, with the x/y correspondence token by token.
  3. How many batches per second your disk delivers, compared with how long a training step
     takes. If the data is slower than the computation, the GPU sits idle.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")  # no window: this has to run over SSH and in CI

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.data import fetch_tinyshakespeare
from llmfs.device import get_device
from llmfs.paths import data_dir, figures_dir

console = Console()

pack_tokens_uint16 = resolve("04_data", "pack_tokens_uint16")
train_val_split = resolve("04_data", "train_val_split")
get_batch = resolve("04_data", "get_batch")

CONTEXT = 64
BATCH = 8


def main() -> None:
    cfg = get_device()
    text, _ = fetch_tinyshakespeare()

    # ------------------------------------------------------------- 1. tokenize
    console.rule("[bold]1. From text to tokens[/bold]")
    vocab_chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab_chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = [stoi[c] for c in text]

    console.print(
        f"Character tokenizer (the simplest possible, to keep the demo fast).\n"
        f"  text      : {len(text):,} characters\n"
        f"  vocabulary: {len(vocab_chars)} tokens\n"
        f"  tokens    : {len(ids):,}\n"
    )

    tokens = pack_tokens_uint16(ids, len(vocab_chars))

    table = Table(title="What the corpus takes in each format", header_style="bold")
    table.add_column("format")
    table.add_column("bytes/token", justify="right")
    table.add_column("this corpus", justify="right")
    table.add_column("500M tokens", justify="right")
    for name, width in [("int64 (python)", 8), ("uint32", 4), ("uint16", 2)]:
        table.add_row(
            name,
            str(width),
            f"{len(ids) * width / 1e6:.1f} MB",
            f"{500e6 * width / 1e9:.1f} GB",
            style="bold green" if width == 2 else "",
        )
    console.print(table)

    # ------------------------------------------------------------- 2. to disk
    console.rule("[bold]2. To disk, and back with memmap[/bold]")
    train, val = train_val_split(tokens, 0.05)
    path = data_dir() / "demo_shakespeare_char.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    train.tofile(path)

    mb = path.stat().st_size / 1e6

    # Each is measured several times and the best is taken, so the order does not decide the
    # result: the first read warms the operating system's page cache and the second
    # measurement comes out artificially fast.
    def time_it(fn) -> float:
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    t_memmap = time_it(lambda: np.memmap(path, dtype=np.uint16, mode="r"))
    t_ram = time_it(lambda: np.fromfile(path, dtype=np.uint16))
    mapped = np.memmap(path, dtype=np.uint16, mode="r")

    console.print(
        f"  training  : {len(train):,} tokens  ({mb:.1f} MB on disk)\n"
        f"  validation: {len(val):,} tokens (the last 5%, contiguous)\n\n"
        f"  open with memmap    : {t_memmap * 1000:6.2f} ms\n"
        f"  load fully into RAM : {t_ram * 1000:6.2f} ms\n"
    )
    console.print(
        f"[bold yellow]Do not read anything into these two numbers.[/bold yellow] With "
        f"{mb:.0f} MB, the file fits entirely in\nthe operating system's cache and what you "
        "are measuring is noise and the fixed cost\nof opening a descriptor. Which one comes "
        "out faster depends on the day.\n"
    )
    console.print(
        "And I am not extrapolating to 1 GB from them either, because that would be the\n"
        "same mistake: the speed you just measured is the cache's, not the disk's.\n\n"
        "The argument for memmap is not read speed, it is this:\n\n"
        "  - [bold]startup[/bold]: loading the whole file costs what it costs, and you pay "
        "it every\n    time you launch the script. memmap reads nothing until you touch the "
        "data.\n"
        "  - [bold]management[/bold]: since you access random positions over and over, the "
        "operating\n    system's cache ends up holding what you use most. Free, and better "
        "than you\n    would do it.\n"
        "  - [bold]scale[/bold]: the same code works just as well if tomorrow the corpus is "
        "50 GB.\n\n"
        "[dim]With 16 GB of RAM, the real 1 GB corpus would also fit fully loaded. If you\n"
        "prefer np.fromfile for simplicity, that is a perfectly defensible option at this\n"
        "scale. There is no magic here.[/dim]"
    )

    # ------------------------------------------------------------- 3. a batch
    console.rule("[bold]3. A real batch[/bold]")
    x, y = get_batch(mapped, BATCH, CONTEXT, device=cfg.device, rng=np.random.default_rng(0))
    console.print(f"x: {tuple(x.shape)} {x.dtype} on {x.device}")
    console.print(f"y: {tuple(y.shape)} {y.dtype} on {y.device}\n")

    row = x[0, :12].tolist()
    target = y[0, :12].tolist()
    console.print("The first 12 tokens of the first sample:\n")
    console.print("  x (input) : " + " ".join(f"{repr(itos[i])[1:-1]:>4}" for i in row))
    console.print("  y (target): " + " ".join(f"{repr(itos[i])[1:-1]:>4}" for i in target))
    console.print(
        f"\n  as text   : x = {''.join(itos[i] for i in row)!r}"
        f"\n              y = {''.join(itos[i] for i in target)!r}\n"
    )

    console.print("And this is what the model learns from this SINGLE sample:\n")
    for k in range(1, 6):
        context = "".join(itos[i] for i in row[:k])
        console.print(f"  seeing {context!r:<12} -> it must predict {itos[target[k - 1]]!r}")
    console.print(
        f"  [dim]... and so on up to {CONTEXT} predictions, from a single window.[/dim]\n"
        f"[bold]A {BATCH}x{CONTEXT} batch is {BATCH * CONTEXT:,} predictions.[/bold] "
        "That is why language\nmodels get so much out of each pass over the data."
    )

    # ------------------------------------------------------------- 4. speed
    console.rule("[bold]4. Pipeline speed[/bold]")
    sizes = [(8, 64), (16, 128), (32, 256), (48, 512)]
    tokens_per_second: list[float] = []
    labels: list[str] = []

    rng = np.random.default_rng(0)
    for b, t in sizes:
        if len(mapped) < t + 2:
            continue
        for _ in range(3):  # warm up
            get_batch(mapped, b, t, device=cfg.device, rng=rng)
        cfg.synchronize()
        started = time.perf_counter()
        n = 20
        for _ in range(n):
            get_batch(mapped, b, t, device=cfg.device, rng=rng)
        cfg.synchronize()
        elapsed = time.perf_counter() - started
        tps = n * b * t / elapsed
        tokens_per_second.append(tps)
        labels.append(f"{b}x{t}")
        console.print(
            f"  batch {b:>2} x {t:>3} = {b * t:>6,} tokens  ->  "
            f"{elapsed / n * 1000:6.2f} ms/batch   {tps / 1e6:6.2f} M tokens/s"
        )

    console.print(
        "\n[dim]Compare the ms/batch of the last row (48x512, the final model's) with how\n"
        "long a training step takes. You will measure that in module 12: if the data takes\n"
        "longer than the computation, the GPU spends its time waiting and the loading has\n"
        "to be parallelized.[/dim]"
    )

    # ------------------------------------------------------------- plot
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    widths = [8, 4, 2]
    names = ["int64", "uint32", "uint16"]
    colours = ["tab:red", "tab:orange", "tab:green"]
    left.bar(names, [500e6 * w / 1e9 for w in widths], color=colours)
    for i, w in enumerate(widths):
        left.text(i, 500e6 * w / 1e9, f"{500e6 * w / 1e9:.1f} GB", ha="center", va="bottom")
    left.set_ylabel("GB on disk")
    left.set_title("500M tokens, by the chosen type")
    left.grid(alpha=0.3, axis="y")

    right.bar(labels, [t / 1e6 for t in tokens_per_second], color="tab:blue")
    right.set_ylabel("millions of tokens/s")
    right.set_xlabel("batch x context")
    right.set_title(f"get_batch throughput ({cfg.kind})")
    right.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    target_path = figures_dir() / "04_data.png"
    fig.savefig(target_path, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target_path}[/green]")


if __name__ == "__main__":
    main()
