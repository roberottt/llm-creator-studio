"""Module 14 demo: the sampling strategies, and the cache that makes everything faster.

    llmfs demo 14

It uses the model trained in module 13 if it exists (`checkpoints/tiny_char/best.pt`); if
not, an untrained one (which still works for measuring speed, even if the text is noise).

Three experiments:
  1. The filters, with small numbers you can follow by hand.
  2. The same prompt with different sampling strategies. You can see greedy's loop.
  3. The KV cache: identical output and a measured speedup.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.config import RunConfig
from llmfs.device import get_device, set_seed
from llmfs.paths import configs_dir, figures_dir
from llmfs.reference import GPT

console = Console()

apply_repetition_penalty = resolve("14_inference", "apply_repetition_penalty")
top_k_filter = resolve("14_inference", "top_k_filter")
top_p_filter = resolve("14_inference", "top_p_filter")
KVCache = resolve("14_inference", "KVCache")
generate_with_cache = resolve("14_inference", "generate_with_cache")


def load_model(cfg_dev):
    """The trained model from module 13, or a new one if it does not exist."""
    from llmfs.data import prepare
    from llmfs.train import load_checkpoint

    cfg = RunConfig.from_yaml(configs_dir() / "tiny_char.yaml")
    dataset = prepare(cfg, quiet=True)
    cfg.model.vocab_size = dataset.vocab_size

    model = GPT(cfg.model)
    path = cfg.run_dir / "best.pt"
    trained = False
    if path.exists():
        try:
            load_checkpoint(path, model, map_location="cpu")
            trained = True
        except Exception:
            pass

    return model.to(cfg_dev.device).eval(), dataset, cfg, trained


def filters_experiment():
    console.rule("[bold]1. The filters, with numbers you can follow by hand[/bold]")

    logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, -1.0, -3.0]])
    probs = F.softmax(logits, dim=-1)[0]

    table = Table(header_style="bold")
    table.add_column("token", justify="right")
    table.add_column("logit", justify="right")
    table.add_column("prob", justify="right")
    table.add_column("cumulative", justify="right")
    table.add_column("top-k=2")
    table.add_column("top-p=0.9")

    tk = top_k_filter(logits.clone(), 2)[0]
    tp = top_p_filter(logits.clone(), 0.9)[0]
    cum = 0.0
    for i in range(6):
        cum += float(probs[i])
        table.add_row(
            str(i),
            f"{float(logits[0, i]):+.1f}",
            f"{float(probs[i]):.3f}",
            f"{cum:.3f}",
            "[green]yes[/green]" if torch.isfinite(tk[i]) else "[dim]no[/dim]",
            "[green]yes[/green]" if torch.isfinite(tp[i]) else "[dim]no[/dim]",
        )
    console.print(table)
    console.print(
        f"[dim]top-p=0.9 leaves {int(torch.isfinite(tp).sum())} candidates: the smallest set "
        "whose mass EXCEEDS 0.9.\nNote that the token crossing the threshold goes in: without "
        "it, the mass would stay below.[/dim]\n"
    )

    console.print("[bold]The temperature, on the same logits:[/bold]")
    table2 = Table(header_style="bold")
    table2.add_column("T", justify="right")
    for i in range(4):
        table2.add_column(f"tok {i}", justify="right")
    table2.add_column("effect")
    for T, effect in [
        (0.5, "sharp: almost always the first one"),
        (1.0, "the distribution as it is"),
        (2.0, "flat: more variety"),
    ]:
        p = F.softmax(logits / T, dim=-1)[0]
        table2.add_row(f"{T}", *[f"{float(p[i]):.3f}" for i in range(4)], effect)
    console.print(table2)

    console.print("\n[bold]The repetition penalty:[/bold]")
    pen = apply_repetition_penalty(logits.clone(), torch.tensor([[0, 5]]), 2.0)[0]
    console.print(
        f"  token 0 (logit [green]+3.0[/green]) -> {float(pen[0]):+.2f}   "
        "[dim]positive: it is DIVIDED[/dim]\n"
        f"  token 5 (logit [red]-3.0[/red]) -> {float(pen[5]):+.2f}   "
        "[dim]negative: it is MULTIPLIED[/dim]\n\n"
        "[dim]If you always divided, the -3.0 would become -1.5 and the token would become\n"
        "MORE likely: exactly the opposite of penalizing it. And since negative logits are\n"
        "the majority, you would be rewarding almost everything that already came out.[/dim]"
    )


def sampling_experiment(model, dataset, cfg_dev, trained):
    console.rule("[bold]2. The same sentence with different strategies[/bold]")
    if not trained:
        console.print(
            "[yellow]There is no trained model at checkpoints/tiny_char/best.pt.[/yellow]\n"
            "[dim]Train one with `llmfs train --config tiny_char` and run this again:\n"
            "the text will be readable and the comparison will make sense.[/dim]\n"
        )

    prompt = "The king"
    ids = torch.tensor([dataset.encode(prompt)], device=cfg_dev.device)

    strategies = [
        ("greedy (T=0)", dict(temperature=0.0)),
        ("T=0.5", dict(temperature=0.5)),
        ("T=0.8 + top-k 40", dict(temperature=0.8, top_k=40)),
        ("T=0.8 + top-p 0.9", dict(temperature=0.8, top_p=0.9)),
        ("T=1.5 (too much)", dict(temperature=1.5)),
        ("greedy + penalty 1.3", dict(temperature=0.0, repetition_penalty=1.3)),
    ]

    for name, kwargs in strategies:
        set_seed(1234)
        out = generate_with_cache(model, ids.clone(), 150, **kwargs)
        text = dataset.decode(out[0].tolist()).replace("\n", " ")
        # How much repetition there is: fraction of distinct 4-grams
        grams = [text[i : i + 4] for i in range(len(text) - 4)]
        variety = len(set(grams)) / max(1, len(grams))
        color = "red" if variety < 0.5 else "yellow" if variety < 0.8 else "green"
        console.print(
            Panel(
                text[:180],
                title=f"{name}  |  4-gram variety: [{color}]{variety:.0%}[/{color}]",
                border_style=color,
            )
        )

    console.print(
        "[dim]Look at greedy's variety compared to the rest. Greedy always picks the most\n"
        "likely token, it is deterministic, and it gets stuck in loops: human text does NOT\n"
        "maximize probability, and that is the central observation of Holtzman et al. (2020).\n\n"
        "The repetition penalty rescues greedy without it ceasing to be deterministic.[/dim]"
    )


def cache_experiment(model, cfg_dev, cfg):
    console.rule("[bold]3. The KV cache[/bold]")

    prompt = torch.randint(0, cfg.model.vocab_size, (1, 8), device=cfg_dev.device)

    # To measure the speed you need a model with a long context: with the toy one
    # (context 128) runs of 100+ tokens hit the limit and the comparison flattens out right
    # where it was starting to get interesting.
    from llmfs.config import ModelConfig

    set_seed(0)
    long_cfg = ModelConfig(
        vocab_size=cfg.model.vocab_size,
        n_layers=4,
        d_model=128,
        n_heads=4,
        d_ff=384,
        context_length=1024,
    )
    long_model = GPT(long_cfg).to(cfg_dev.device).eval()
    long_prompt = torch.randint(0, cfg.model.vocab_size, (1, 8), device=cfg_dev.device)

    console.print("[bold]First the important thing: does it give the same output?[/bold]\n")
    without = model.generate(prompt.clone(), 40, temperature=0.0)
    with_ = generate_with_cache(model, prompt.clone(), 40, temperature=0.0)
    equal = torch.equal(without, with_)
    console.print(
        f"  without cache: {without[0, -10:].tolist()}\n"
        f"  with cache: {with_[0, -10:].tolist()}\n"
        + (
            "  [bold green]IDENTICAL.[/bold green] The cache is a pure optimization: it does "
            "not change the result.\n"
            if equal
            else "  [bold red]THEY DIVERGE.[/bold red] Look at RoPE's pos_offset.\n"
        )
    )

    console.print("[bold]And now the speed:[/bold]\n")
    table = Table(header_style="bold")
    table.add_column("tokens", justify="right")
    table.add_column("without cache", justify="right")
    table.add_column("with cache", justify="right")
    table.add_column("speedup", justify="right")
    table.add_column("cache memory", justify="right")

    console.print(
        f"[dim](measuring on a model with context {long_cfg.context_length}: with the toy one\n"
        f"of context {cfg.model.context_length} the long runs would hit the limit and the\n"
        f"comparison would flatten out right where it gets interesting)[/dim]\n"
    )

    lengths, speedups = [], []
    for n in (50, 100, 200, 400, 800):
        t0 = time.perf_counter()
        long_model.generate(long_prompt.clone(), n, temperature=0.0)
        cfg_dev.synchronize()
        t_without = time.perf_counter() - t0

        t0 = time.perf_counter()
        generate_with_cache(long_model, long_prompt.clone(), n, temperature=0.0)
        cfg_dev.synchronize()
        t_with = time.perf_counter() - t0

        cache = KVCache(long_cfg.n_layers)
        long_model(long_prompt, use_cache=True, cache=cache)
        for _ in range(n):
            long_model(long_prompt[:, -1:], use_cache=True, cache=cache)

        lengths.append(n)
        speedups.append(t_without / max(t_with, 1e-9))
        table.add_row(
            str(n),
            f"{t_without * 1000:.0f} ms",
            f"{t_with * 1000:.0f} ms",
            f"[bold]{t_without / max(t_with, 1e-9):.2f}x[/bold]",
            f"{cache.memory_bytes() / 1024:.0f} KB",
        )
    console.print(table)

    bytes_full = 2 * 6 * 512 * 320 * 2  # the final model: 6 layers, ctx 512, d 320, fp16
    console.print(
        f"\n[dim]The speedup grows with the length: without the cache, generating N tokens\n"
        f"costs O(N^2) and with it O(N). With longer sequences the difference takes off.\n\n"
        f"The cache memory is 2 * n_layers * T * d_model * bytes. For the FINAL 9M model with\n"
        f"context 512 in fp16 that is {bytes_full / 1e6:.1f} MB: nothing. For a 70B model with\n"
        f"a 100,000-token context, tens of gigabytes, more than the weights themselves. Hence\n"
        f"techniques like grouped-query attention.[/dim]"
    )
    return lengths, speedups


def main() -> None:
    cfg_dev = get_device()
    model, dataset, cfg, trained = load_model(cfg_dev)

    if trained:
        console.print(
            "[green]Using the trained model from checkpoints/tiny_char/best.pt[/green]\n"
        )

    filters_experiment()
    sampling_experiment(model, dataset, cfg_dev, trained)
    lengths, speedups = cache_experiment(model, cfg_dev, cfg)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, -1.0, -3.0]])
    for T in (0.5, 1.0, 2.0):
        left.plot(F.softmax(logits / T, dim=-1)[0].numpy(), marker="o", label=f"T = {T}")
    left.set_xlabel("token")
    left.set_ylabel("probability")
    left.set_title("The effect of the temperature")
    left.grid(alpha=0.3)
    left.legend()

    right.plot(lengths, speedups, marker="o", color="tab:green")
    right.axhline(1.0, color="gray", ls="--", lw=1, label="no gain")
    right.set_xlabel("tokens generated")
    right.set_ylabel("KV cache speedup")
    right.set_title("The gain grows with the length")
    right.grid(alpha=0.3)
    right.legend(fontsize=8)

    fig.tight_layout()
    target = figures_dir() / "14_inference.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")


if __name__ == "__main__":
    main()
