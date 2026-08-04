"""Demo for module 06: look at where attention looks.

    llmfs demo 06

Three experiments:
  1. The scaling by sqrt(d_k): what happens to the softmax when you remove it. With numbers.
  2. The causal mask, drawn out.
  3. A one-layer attention model actually trained (about 20 s) on Shakespeare, and the
     heatmap of what each character looks at. The heads specialize on their own and it shows.
"""

from __future__ import annotations

import math
import time

import matplotlib

matplotlib.use("Agg")  # no window: this has to run over SSH and in CI

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.data import fetch_tinyshakespeare
from llmfs.device import get_device, set_seed
from llmfs.paths import figures_dir

console = Console()

causal_mask = resolve("06_attention", "causal_mask")
single_head_attention = resolve("06_attention", "single_head_attention")
MultiHeadAttention = resolve("06_attention", "MultiHeadAttention")

SENTENCE = "the king shall speak to his people"


def entropy(weights: torch.Tensor) -> float:
    """Mean entropy of the attention distributions, in nats.

    High = the token spreads its attention over many. Low = it fixates on one.
    """
    return float(-(weights * torch.log(weights + 1e-12)).sum(-1).mean())


# --------------------------------------------------------------- 1. the scaling


def scaling_experiment() -> tuple[list[int], list[float], list[float]]:
    console.rule("[bold]1. Why we divide by sqrt(d_k)[/bold]")
    console.print(
        "The dot product of two d_k-dimensional vectors has variance d_k. The larger d_k\n"
        "is, the more the scores spread out, and softmax is exponential: with widely\n"
        "separated scores it returns almost [0, 0, ..., 1, ..., 0].\n\n"
        "It is measured with the ENTROPY of the attention distribution. Maximum = spread\n"
        "over everyone; close to zero = fixated on one and learning nothing from the rest.\n"
    )

    dims = [8, 32, 128, 512, 2048]
    scaled, unscaled = [], []
    seq = 16
    maximum = math.log(seq)

    table = Table(header_style="bold")
    table.add_column("d_k", justify="right")
    table.add_column("entropy WITH scaling", justify="right")
    table.add_column("entropy WITHOUT scaling", justify="right")
    table.add_column("max weight unscaled", justify="right")

    for d_k in dims:
        torch.manual_seed(0)
        q, k, v = torch.randn(1, seq, d_k), torch.randn(1, seq, d_k), torch.randn(1, seq, d_k)

        _, scaled_weights = single_head_attention(q, k, v)
        unscaled_weights = torch.softmax(q @ k.transpose(-2, -1), dim=-1)

        scaled.append(entropy(scaled_weights))
        unscaled.append(entropy(unscaled_weights))
        table.add_row(
            str(d_k),
            f"{scaled[-1]:.3f}",
            f"{unscaled[-1]:.3f}",
            f"{float(unscaled_weights.max()):.4f}",
        )

    console.print(table)
    console.print(
        f"[dim]The maximum possible entropy with {seq} positions is ln({seq}) = "
        f"{maximum:.3f}.[/dim]\n"
    )
    console.print(
        f"With scaling the entropy stays high whatever happens to d_k.\n"
        f"Without scaling, at d_k={dims[-1]} it falls to {unscaled[-1]:.3f} and the maximum "
        f"weight reaches {1.0:.2f}:\nthe token fixates on ONE and ignores everything else.\n\n"
        "[bold]And the real problem is not this, it is the gradient.[/bold] The softmax's\n"
        "derivative is p(1-p). With p pinned to 0 or 1 it is practically zero, so the layer\n"
        "stops learning. Without the sqrt(d_k), a large Transformer does not train."
    )
    return dims, scaled, unscaled


# --------------------------------------------------------------- 2. the mask


def mask_experiment() -> None:
    console.rule("[bold]2. The causal mask[/bold]")
    m = causal_mask(8)
    console.print("Mask for 8 tokens ([green]#[/green] = can look, . = forbidden):\n")
    for i in range(8):
        row = " ".join("[green]#[/green]" if bool(m[i, j]) else "[dim].[/dim]" for j in range(8))
        console.print(f"  token {i}: {row}")

    torch.manual_seed(0)
    q, k, v = torch.randn(1, 8, 16), torch.randn(1, 8, 16), torch.randn(1, 8, 16)
    _, with_mask = single_head_attention(q, k, v, m)
    _, without_mask = single_head_attention(q, k, v)

    console.print(
        f"\n  total weight on the future WITH the mask:    {float(with_mask.triu(1).sum()):.6f}\n"
        f"  total weight on the future WITHOUT the mask: {float(without_mask.triu(1).sum()):.6f}\n"
    )
    console.print(
        "[dim]-inf goes in BEFORE the softmax, the weights are not zeroed afterwards. If\n"
        "you zeroed them afterwards, the rows would no longer sum to 1. Check: each row\n"
        f"with the mask sums to {float(with_mask[0].sum(-1).mean()):.6f}.[/dim]"
    )


# --------------------------------------------------------------- 3. real attention


class AttentionModel(nn.Module):
    """The minimum for attention to have something to learn: embedding + MHA + output."""

    def __init__(self, vocab: int, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, d_model)
        self.pos_embedding = nn.Embedding(256, d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx, targets=None, return_weights=False):
        seq = idx.shape[1]
        x = self.token_embedding(idx) + self.pos_embedding(torch.arange(seq, device=idx.device))
        if return_weights:
            out, weights = self.attn(self.norm(x), return_weights=True)
            return self.head(x + out), weights
        x = x + self.attn(self.norm(x))
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        return logits, loss


def heatmap_experiment(cfg) -> tuple[torch.Tensor, list[str]]:
    console.rule("[bold]3. Attention from a model actually trained[/bold]")

    text, _ = fetch_tinyshakespeare()
    text = text[:150_000].lower()
    vocab_chars = sorted(set(text + SENTENCE))
    stoi = {c: i for i, c in enumerate(vocab_chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    set_seed(0)
    model = AttentionModel(len(vocab_chars), d_model=64, n_heads=4).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    ctx, batch, steps = 48, 64, 400
    console.print(f"Training a 1-layer attention model, {steps} steps...")
    started = time.perf_counter()
    for step in range(steps):
        i = torch.randint(0, len(data) - ctx - 1, (batch,))
        x = torch.stack([data[j : j + ctx] for j in i]).to(cfg.device)
        y = torch.stack([data[j + 1 : j + 1 + ctx] for j in i]).to(cfg.device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    console.print(
        f"  final loss: {float(loss.detach()):.4f}  "
        f"(the floor is ln({len(vocab_chars)}) = {math.log(len(vocab_chars)):.4f})  "
        f"[dim]{time.perf_counter() - started:.1f} s[/dim]\n"
    )

    model.eval()
    ids = torch.tensor([[stoi[c] for c in SENTENCE]], device=cfg.device)
    with torch.no_grad():
        _, weights = model(ids, return_weights=True)

    weights = weights[0].cpu()  # (n_heads, T, T)
    characters = list(SENTENCE)

    console.print(f'Sentence: [cyan]"{SENTENCE}"[/cyan]\n')
    for head in range(weights.shape[0]):
        w = weights[head]
        # How far back each token looks, as a weighted average
        positions = torch.arange(len(characters)).float()
        mean_distance = float(
            sum(
                (positions[i] - positions[: i + 1]) @ w[i, : i + 1]
                for i in range(1, len(characters))
            )
            / (len(characters) - 1)
        )
        # The token the last character looks at most (not counting itself)
        last = w[-1, :-1]
        favourite = int(last.argmax())
        console.print(
            f"  head {head}: mean distance {mean_distance:5.2f} positions   |   "
            f"the last character mostly looks at {characters[favourite]!r} (pos {favourite})"
        )

    console.print(
        "\n[dim]Note the mean distances: they differ between heads. Each one has specialized\n"
        "in a different range of context, and nobody told it to. In large models this goes\n"
        "much further: there are heads that pair opening and closing quotes, and induction\n"
        "heads, which detect '...A B ... A' and predict B.[/dim]"
    )
    return weights, characters


# --------------------------------------------------------------- plot


def main() -> None:
    cfg = get_device()
    dims, scaled, unscaled = scaling_experiment()
    mask_experiment()
    weights, characters = heatmap_experiment(cfg)

    n_heads = weights.shape[0]
    fig = plt.figure(figsize=(13, 8))

    ax = fig.add_subplot(2, 3, 1)
    ax.plot(dims, scaled, marker="o", label="with /sqrt(d_k)")
    ax.plot(dims, unscaled, marker="s", label="unscaled")
    ax.axhline(math.log(16), color="gray", ls="--", lw=1, label="maximum entropy")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("d_k")
    ax.set_ylabel("attention entropy (nats)")
    ax.set_title("Unscaled, the softmax saturates")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 2)
    # The same colour map as the attention heatmaps, so it reads the same: light is where
    # the weight is and dark is where there is none. With "Greens" the light one was the
    # forbidden region and it read backwards.
    ax.imshow(causal_mask(16).numpy().astype(float), cmap="viridis", interpolation="nearest")
    ax.set_title("Causal mask (light = allowed)")
    ax.set_xlabel("token being looked at")
    ax.set_ylabel("token doing the looking")

    for head in range(min(n_heads, 4)):
        ax = fig.add_subplot(2, 3, 3 + head)
        ax.imshow(weights[head].numpy(), cmap="viridis", interpolation="nearest")
        ax.set_title(f"head {head}", fontsize=10)
        ax.set_xticks(range(len(characters)))
        ax.set_yticks(range(len(characters)))
        ax.set_xticklabels(characters, fontsize=5)
        ax.set_yticklabels(characters, fontsize=5)

    fig.suptitle("Module 06: self-attention", fontsize=13)
    fig.tight_layout()
    target = figures_dir() / "06_attention.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")
    console.print(
        "[dim]Open the figure: the four heatmaps at the bottom are the four heads. The\n"
        "upper triangle is always black (the mask), and each head has a visibly different\n"
        "pattern.[/dim]"
    )


if __name__ == "__main__":
    main()
