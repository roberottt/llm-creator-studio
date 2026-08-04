"""Demo for module 09: RoPE's frequencies and extrapolation to long contexts.

    llmfs demo 09

Three experiments:
  1. The frequencies, drawn out. You can see the ladder from fast to slow.
  2. The relative invariance, checked with numbers: <R(m)q, R(n)k> depends only on n-m.
  3. Training three identical models except for the positional encoding (learned, sinusoidal
     and RoPE) with a context of 32, and evaluating them with contexts of up to 128.
     It takes about 30 s.
"""

from __future__ import annotations

import math
import time

import matplotlib

matplotlib.use("Agg")  # no window: this has to run over SSH and in CI

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.data import fetch_tinyshakespeare
from llmfs.device import get_device, set_seed
from llmfs.paths import figures_dir
from llmfs.reference import MultiHeadAttention, RMSNorm

console = Console()

sinusoidal_embeddings = resolve("09_position", "sinusoidal_embeddings")
rope_frequencies = resolve("09_position", "rope_frequencies")
apply_rope = resolve("09_position", "apply_rope")

TRAIN_CTX = 32
EVAL_CONTEXTS = [8, 16, 32, 48, 64, 96, 128]


def frequencies_experiment() -> tuple[torch.Tensor, torch.Tensor]:
    console.rule("[bold]1. The frequency ladder[/bold]")
    cos, sin = rope_frequencies(64, 256)

    table = Table(header_style="bold")
    table.add_column("pair of dimensions", justify="right")
    table.add_column("frequency (rad/position)", justify="right")
    table.add_column("completes a full turn every", justify="right")

    for i in (0, 4, 8, 16, 24, 31):
        # The angle at position 1 is exactly the frequency
        freq = float(torch.acos(cos[1, i].clamp(-1, 1)))
        period = 2 * math.pi / freq if freq > 1e-12 else float("inf")
        table.add_row(
            f"{i}", f"{freq:.6f}", f"{period:,.0f} positions" if period < 1e9 else "never"
        )
    console.print(table)
    console.print(
        "[dim]The first pairs turn fast and distinguish neighbouring positions; the last\n"
        "ones turn so slowly that within the trained context they barely complete a fraction\n"
        "of a turn, and they capture long distances.\n\n"
        "That last row is the key to RoPE's limitations: if the model has only seen 512\n"
        "positions, the large angles of the slow pairs are unseen territory.[/dim]"
    )
    return cos, sin


def invariance_experiment(cos: torch.Tensor, sin: torch.Tensor) -> None:
    console.rule("[bold]2. The relative invariance[/bold]")
    console.print(
        "The property that justifies RoPE:  <R(m)q, R(n)k>  depends ONLY on n-m.\n\n"
        "The SAME pair of vectors q and k is taken, placed at different positions, and the\n"
        "attention score between them is measured.\n"
    )

    set_seed(0)
    head_dim = 64
    q_base, k_base = torch.randn(head_dim), torch.randn(head_dim)

    def score(pos_q: int, pos_k: int) -> float:
        q = torch.zeros(1, 1, 256, head_dim)
        k = torch.zeros(1, 1, 256, head_dim)
        q[0, 0, pos_q] = q_base
        k[0, 0, pos_k] = k_base
        return float(apply_rope(q, cos, sin)[0, 0, pos_q] @ apply_rope(k, cos, sin)[0, 0, pos_k])

    table = Table(header_style="bold")
    table.add_column("positions (q, k)")
    table.add_column("distance", justify="right")
    table.add_column("score", justify="right")

    for pq, pk in [(0, 3), (2, 5), (10, 13), (100, 103), (200, 203)]:
        table.add_row(f"({pq}, {pk})", str(pk - pq), f"{score(pq, pk):.10f}")
    table.add_section()
    for pq, pk in [(0, 7), (50, 57)]:
        table.add_row(f"({pq}, {pk})", str(pk - pq), f"{score(pq, pk):.10f}")
    console.print(table)

    console.print(
        "[bold]The first five rows give the same number to the last decimal.[/bold]\n"
        "All of them are at distance 3, even though one is at the start of the sequence and\n"
        "another at position 200.\n\n"
        "That means the model does not learn 'token number 3': it learns 'the token three\n"
        "positions back'. And that is why it can apply what it learned anywhere in the\n"
        "sequence.\n\n"
        "[dim]The last two rows, at distance 7, give a different value but one that is also\n"
        "equal between them.[/dim]"
    )


# ------------------------------------------------------------- 3. extrapolation


class PositionalModel(nn.Module):
    """A one-layer transformer. The only thing changing between variants is the position."""

    def __init__(self, vocab: int, d_model: int, n_heads: int, kind: str, max_ctx: int) -> None:
        super().__init__()
        self.kind = kind
        self.token_embedding = nn.Embedding(vocab, d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

        if kind == "learned":
            self.pos_embedding = nn.Embedding(max_ctx, d_model)
        elif kind == "sinusoidal":
            self.register_buffer("table", sinusoidal_embeddings(4096, d_model), persistent=False)
        elif kind == "rope":
            cos, sin = rope_frequencies(d_model // n_heads, 4096)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        seq = idx.shape[1]
        x = self.token_embedding(idx)

        cos = sin = None
        if self.kind == "learned":
            if seq > self.pos_embedding.num_embeddings:
                raise IndexError("position outside the learned table")
            x = x + self.pos_embedding(torch.arange(seq, device=idx.device))
        elif self.kind == "sinusoidal":
            x = x + self.table[:seq].to(x.device)
        elif self.kind == "rope":
            cos, sin = self.rope_cos, self.rope_sin

        x = x + self.attn(self.norm(x), cos=cos, sin=sin)
        logits = self.head(x)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def extrapolation_experiment(cfg) -> dict[str, list[float | None]]:
    console.rule("[bold]3. Extrapolation beyond the trained context[/bold]")

    text, _ = fetch_tinyshakespeare()
    text = text[:120_000].lower()
    vocab_chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(vocab_chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    cut = int(len(data) * 0.9)
    train, val = data[:cut], data[cut:]

    console.print(
        f"Three identical models except for the positional encoding.\n"
        f"Trained with a context of {TRAIN_CTX}, evaluated with contexts from "
        f"{EVAL_CONTEXTS[0]} to {EVAL_CONTEXTS[-1]}.\n"
    )

    results: dict[str, list[float | None]] = {}
    for kind in ("learned", "sinusoidal", "rope"):
        set_seed(0)
        model = PositionalModel(
            len(vocab_chars), d_model=64, n_heads=4, kind=kind, max_ctx=TRAIN_CTX
        ).to(cfg.device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

        started = time.perf_counter()
        for _ in range(400):
            i = torch.randint(0, len(train) - TRAIN_CTX - 1, (64,))
            x = torch.stack([train[j : j + TRAIN_CTX] for j in i]).to(cfg.device)
            y = torch.stack([train[j + 1 : j + 1 + TRAIN_CTX] for j in i]).to(cfg.device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        elapsed = time.perf_counter() - started

        model.eval()
        losses: list[float | None] = []
        for ctx in EVAL_CONTEXTS:
            try:
                with torch.no_grad():
                    total, n = 0.0, 0
                    for _ in range(20):
                        i = torch.randint(0, len(val) - ctx - 1, (16,))
                        x = torch.stack([val[j : j + ctx] for j in i]).to(cfg.device)
                        y = torch.stack([val[j + 1 : j + 1 + ctx] for j in i]).to(cfg.device)
                        total += float(model(x, y)[1])
                        n += 1
                    losses.append(total / n)
            except (IndexError, RuntimeError):
                losses.append(None)  # it cannot process that length
        results[kind] = losses
        console.print(f"  [dim]{kind} trained in {elapsed:.1f} s[/dim]")

    table = Table(header_style="bold")
    table.add_column("context", justify="right")
    for kind in results:
        table.add_column(kind, justify="right")
    for i, ctx in enumerate(EVAL_CONTEXTS):
        mark = " (trained)" if ctx == TRAIN_CTX else ""
        row = [f"{ctx}{mark}"]
        for kind in results:
            v = results[kind][i]
            row.append("[red]cannot[/red]" if v is None else f"{v:.4f}")
        table.add_row(*row, style="bold" if ctx == TRAIN_CTX else "")
    console.print(table)

    trained_idx = EVAL_CONTEXTS.index(TRAIN_CTX)
    console.print(
        "\n[bold]What to look at:[/bold]\n\n"
        "1. The [bold]learned[/bold] encoding literally [red]cannot[/red] process sequences "
        "longer than\n   the trained ones: there is no row in the table to look up. It is a "
        "hard ceiling.\n\n"
        "2. [bold]sinusoidal[/bold] and [bold]rope[/bold] do produce an answer for any "
        "length.\n"
    )

    for kind in ("sinusoidal", "rope"):
        base = results[kind][trained_idx]
        long = results[kind][-1]
        if base is not None and long is not None:
            console.print(
                f"   {kind}: from {base:.4f} at context {TRAIN_CTX} to {long:.4f} at "
                f"{EVAL_CONTEXTS[-1]}  ([bold]{100 * (long / base - 1):+.0f}%[/bold])"
            )

    console.print(
        "\n[bold yellow]And here is the honest part:[/bold yellow] 'being able to process' is "
        "not the same as 'working well'.\nBoth degrade. RoPE has the relative property, but "
        "the slow frequencies barely\ncomplete a fraction of a turn within the trained range, "
        "so the large angles are\nunseen territory.\n\n"
        "[dim]That is why there is a whole family of techniques for extending the context "
        "AFTER\ntraining (position interpolation, NTK-aware scaling, YaRN): because direct\n"
        "extrapolation is not enough. When you read that 'RoPE extrapolates', this is what\n"
        "lies behind it.[/dim]"
    )
    return results


def main() -> None:
    cfg = get_device()
    cos, sin = frequencies_experiment()
    invariance_experiment(cos, sin)
    results = extrapolation_experiment(cfg)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for i, pair in enumerate((0, 4, 8, 16, 31)):
        axes[0].plot(cos[:128, pair].numpy(), label=f"pair {pair}", lw=1.2)
    axes[0].set_xlabel("position")
    axes[0].set_ylabel("cos(angle)")
    axes[0].set_title("RoPE's frequencies")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    im = axes[1].imshow(cos[:64, :32].numpy(), aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    axes[1].set_xlabel("dimension (first half)")
    axes[1].set_ylabel("position")
    axes[1].set_title("The cosine table")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    for kind, losses in results.items():
        xs = [c for c, v in zip(EVAL_CONTEXTS, losses) if v is not None]
        ys = [v for v in losses if v is not None]
        axes[2].plot(xs, ys, marker="o", label=kind)
    axes[2].axvline(TRAIN_CTX, color="gray", ls="--", lw=1.5, label="training context")
    axes[2].set_xlabel("context at evaluation")
    axes[2].set_ylabel("validation loss")
    axes[2].set_title("Extrapolation beyond what was trained")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    target = figures_dir() / "09_position.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")


if __name__ == "__main__":
    main()
