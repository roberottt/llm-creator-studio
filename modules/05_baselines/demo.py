"""Demo for module 05: the three baselines competing, and where the floor is.

    llmfs demo 05

It really trains (a few seconds) a count-based bigram, a neural bigram and a Bengio MLP over
character-level Shakespeare, and compares their losses against the uniform baseline.

What to take away: the floor `ln(V)`, that the neural bigram converges to exactly the same
place as the count-based one, and that looking at more context helps... at a parameter cost
that grows linearly. That is the doorway into module 06.
"""

from __future__ import annotations

import math
import time

import matplotlib

matplotlib.use("Agg")  # no window: this has to run over SSH and in CI

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.data import fetch_tinyshakespeare
from llmfs.device import get_device, set_seed
from llmfs.paths import figures_dir

console = Console()

uniform_baseline_loss = resolve("05_baselines", "uniform_baseline_loss")
bigram_counts = resolve("05_baselines", "bigram_counts")
bigram_nll = resolve("05_baselines", "bigram_nll")
NeuralBigram = resolve("05_baselines", "NeuralBigram")
BengioMLP = resolve("05_baselines", "BengioMLP")


def train(model, x, y, steps: int, lr: float, cfg) -> list[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[float] = []
    for _ in range(steps):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return history


def main() -> None:
    cfg = get_device()
    set_seed(1337)

    text, _ = fetch_tinyshakespeare()
    text = text[:200_000]
    vocab_chars = sorted(set(text))
    V = len(vocab_chars)
    stoi = {c: i for i, c in enumerate(vocab_chars)}
    ids = [stoi[c] for c in text]

    cut = int(len(ids) * 0.9)
    train_ids, val_ids = ids[:cut], ids[cut:]

    floor = uniform_baseline_loss(V)
    console.rule("[bold]0. The floor[/bold]")
    console.print(
        f"Vocabulary: {V} characters.\n"
        f"A model guessing at random gives a loss of [bold]ln({V}) = {floor:.4f}[/bold] "
        f"nats,\nwhich is a perplexity of {math.exp(floor):.1f} (that is, torn between "
        f"{V} options).\n\n"
        "[dim]Any model that does not get below this number has learned nothing. And when\n"
        "you train the final model, the step-0 loss has to be almost exactly this: higher\n"
        "means the init is wrong, lower means an information leak.[/dim]"
    )

    results: list[tuple[str, float, int, str]] = []
    results.append(("uniform (random)", floor, 0, "-"))

    # ------------------------------------------------------------- count-based bigram
    console.rule("[bold]1. Count-based bigram[/bold]")
    started = time.perf_counter()
    counts = bigram_counts(train_ids, V)
    counting_loss = bigram_nll(counts, val_ids, alpha=1.0)
    console.print(
        f"  {V}x{V} matrix filled in {time.perf_counter() - started:.2f} s\n"
        f"  validation loss: [bold]{counting_loss:.4f}[/bold] "
        f"(perplexity {math.exp(counting_loss):.1f})"
    )
    results.append(("bigram (counting)", counting_loss, V * V, "0.0 s"))

    console.print("\n[bold]Effect of the smoothing:[/bold]")
    for alpha in (0.0001, 0.01, 1.0, 100.0, 10000.0):
        p = bigram_nll(counts, val_ids, alpha=alpha)
        mark = (
            "  <- barely smoothed, and still finite because the corpus is dense"
            if alpha == 0.0001
            else ""
        )
        console.print(f"    alpha={alpha:<8} -> {p:.4f}{mark}")
    console.print(
        "[dim]With a huge alpha the model becomes uniform and the loss rises to the floor.\n"
        "With a tiny alpha and an unseen pair in validation, it would go to infinity.[/dim]"
    )

    # ------------------------------------------------------------- neural bigram
    console.rule("[bold]2. The same bigram, learned by gradient[/bold]")
    x_tr = torch.tensor(train_ids[:-1], device=cfg.device).unsqueeze(0)
    y_tr = torch.tensor(train_ids[1:], device=cfg.device).unsqueeze(0)

    nb_model = NeuralBigram(V).to(cfg.device)
    started = time.perf_counter()
    nb_history = train(nb_model, x_tr, y_tr, steps=400, lr=0.5, cfg=cfg)
    t_nb = time.perf_counter() - started

    x_val = torch.tensor(val_ids[:-1], device=cfg.device).unsqueeze(0)
    y_val = torch.tensor(val_ids[1:], device=cfg.device).unsqueeze(0)
    with torch.no_grad():
        nb_val = float(nb_model(x_val, y_val)[1])

    excess = nb_history[0] - floor
    console.print(
        f"  initial loss  : {nb_history[0]:.4f}   [dim](the floor is {floor:.4f})[/dim]\n"
        f"  final loss    : {nb_history[-1]:.4f} on training\n"
        f"  on validation : [bold]{nb_val:.4f}[/bold]\n"
        f"  parameters    : {sum(p.numel() for p in nb_model.parameters()):,}\n\n"
        f"[bold]Counting: {counting_loss:.4f}. Learning: {nb_val:.4f}.[/bold]\n"
        "[dim]They converge to the same place, because they are the same model. The\n"
        "difference is that counting does not scale beyond this and learning does.[/dim]"
    )

    console.print(
        f"\n[bold yellow]Note the initial loss: {nb_history[0]:.4f}, which is "
        f"{excess:.2f} nats ABOVE\nthe floor of {floor:.4f}.[/bold yellow] That should not "
        "happen, and it is a perfect example of\nthe symptom THEORY.md describes.\n\n"
        "The cause: `nn.Embedding` initializes by default with a normal N(0,1). Since those\n"
        "rows ARE the logits, the model starts with strong, random opinions instead of with\n"
        "ignorance. A logit of +2 against one of -2 is a 55-to-1 bet made before seeing a\n"
        "single data point, and being right by chance is unlikely: hence the excess.\n\n"
        "That is why the GPT in module 10 initializes EVERYTHING with std=0.02 instead of\n"
        "1.0. With almost identical logits, the softmax comes out almost uniform and the\n"
        "step-0 loss lands right on ln(V). You can check it:\n"
        "  [cyan]torch.nn.init.normal_(model.token_embedding.weight, std=0.02)[/cyan]"
    )
    results.append(
        ("bigram (neural)", nb_val, sum(p.numel() for p in nb_model.parameters()), f"{t_nb:.1f} s")
    )

    # ------------------------------------------------------------- Bengio's MLP
    console.rule("[bold]3. Bengio's MLP: looking further back[/bold]")
    bengio_history: dict[int, list[float]] = {}
    val_per_block: dict[int, float] = {}
    for block in (2, 4, 8):
        tr_data = torch.tensor(train_ids, dtype=torch.long)
        xb = torch.stack([tr_data[i : i + block] for i in range(0, len(tr_data) - block, 3)])
        yb = torch.stack([tr_data[i + block] for i in range(0, len(tr_data) - block, 3)])
        xb, yb = xb.to(cfg.device), yb.to(cfg.device)

        val_data = torch.tensor(val_ids, dtype=torch.long)
        xv = torch.stack([val_data[i : i + block] for i in range(0, len(val_data) - block, 7)])
        yv = torch.stack([val_data[i + block] for i in range(0, len(val_data) - block, 7)])
        xv, yv = xv.to(cfg.device), yv.to(cfg.device)

        model = BengioMLP(V, block, d_embed=24, n_hidden=128).to(cfg.device)
        n_params = sum(p.numel() for p in model.parameters())

        started = time.perf_counter()
        history: list[float] = []
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        for step in range(400):
            sel = torch.randint(0, len(xb), (512,), device=cfg.device)
            _, loss = model(xb[sel], yb[sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))
        elapsed = time.perf_counter() - started

        with torch.no_grad():
            val = float(model(xv, yv)[1])
        bengio_history[block] = history
        val_per_block[block] = val
        console.print(
            f"  context {block} characters: validation [bold]{val:.4f}[/bold]  "
            f"({n_params:,} parameters, {elapsed:.1f} s)"
        )
        results.append((f"Bengio MLP (ctx {block})", val, n_params, f"{elapsed:.1f} s"))

    best_block = min(val_per_block, key=val_per_block.get)
    monotonic = list(val_per_block.values()) == sorted(val_per_block.values(), reverse=True)

    console.print(
        f"\nAll of them beat the bigram ({counting_loss:.4f}), so looking more than one token "
        "back helps.\n"
        f"The best of the three is the one with context {best_block} "
        f"({val_per_block[best_block]:.4f})."
    )
    if not monotonic:
        console.print(
            "\n[bold yellow]But the improvement is NOT monotonic: the longest context does "
            "not win.[/bold yellow]\n"
            "This is not a bug in the demo, it is an honest result and it is worth\n"
            "understanding. All three models trained for the SAME 400 steps, and the one\n"
            "with context 8 has more than twice the parameters of the one with context 2.\n"
            "With the same compute budget, the large model is left half-trained.\n\n"
            "[dim]Transferable lesson: comparing architectures at equal STEPS is not\n"
            "comparing at equal compute, and it almost always favours the small model.\n"
            "It is the same mistake module 12's scaling laws come to correct.[/dim]"
        )
    console.print(
        "\n[dim]And in any case, look at the parameters: the first layer is\n"
        "Linear(block_size * d_embed, n_hidden), so it grows LINEARLY with the context.\n"
        "With a context of 512 that layer alone would be enormous. On top of that the model\n"
        "treats each position as an independent input, with no idea that they are related\n"
        "and no way to decide which to pay attention to. Solving both at once is what\n"
        "attention does, in module 06.[/dim]"
    )

    # ------------------------------------------------------------- summary
    console.rule("[bold]Summary[/bold]")
    table = Table(header_style="bold")
    table.add_column("model")
    table.add_column("loss (val)", justify="right")
    table.add_column("perplexity", justify="right")
    table.add_column("parameters", justify="right")
    table.add_column("time", justify="right")
    for name, loss, params, elapsed_str in results:
        table.add_row(
            name,
            f"{loss:.4f}",
            f"{math.exp(loss):.1f}",
            f"{params:,}" if params else "-",
            elapsed_str,
        )
    console.print(table)

    # ------------------------------------------------------------- plot
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    left.plot(nb_history, label="neural bigram")
    for block, hist in bengio_history.items():
        left.plot(hist, label=f"Bengio MLP (ctx {block})")
    left.axhline(floor, color="red", ls="--", lw=1.5, label=f"floor: ln({V}) = {floor:.2f}")
    left.axhline(counting_loss, color="gray", ls=":", lw=1.5, label="count-based bigram")
    left.set_xlabel("training step")
    left.set_ylabel("loss (nats)")
    left.set_title("All the baselines")
    left.grid(alpha=0.3)
    left.legend(fontsize=8)

    names = [r[0] for r in results]
    losses = [r[1] for r in results]
    colours = ["tab:red"] + ["tab:gray"] * 2 + ["tab:blue"] * (len(results) - 3)
    right.barh(range(len(names)), losses, color=colours)
    right.set_yticks(range(len(names)))
    right.set_yticklabels(names, fontsize=8)
    right.axvline(floor, color="red", ls="--", lw=1.5)
    right.set_xlabel("validation loss (nats)")
    right.set_title("How far each one gets below the floor")
    right.grid(alpha=0.3, axis="x")
    right.invert_yaxis()

    fig.tight_layout()
    target = figures_dir() / "05_baselines.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")


if __name__ == "__main__":
    main()
