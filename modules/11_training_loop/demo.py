"""Demo for module 11: the loop's four pieces, compared and measured.

    llmfs demo 11

Four experiments:
  1. Your AdamW against PyTorch's: the same weights to the last decimal.
  2. What happens without the bias correction, without momentum and without scaling. With
     curves.
  3. The scheduler, drawn out, and why the warmup is needed.
  4. Gradient clipping in the face of a toxic batch.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")  # no window: this has to run over SSH and in CI

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.config import ModelConfig
from llmfs.device import set_seed
from llmfs.paths import figures_dir
from llmfs.reference import GPT

console = Console()

AdamWScratch = resolve("11_training_loop", "AdamWScratch")
lr_at_step = resolve("11_training_loop", "lr_at_step")
clip_grad_norm = resolve("11_training_loop", "clip_grad_norm")
build_param_groups = resolve("11_training_loop", "build_param_groups")


def task():
    """Any old nonlinear regression, just to have something to optimize."""
    set_seed(0)
    x = torch.randn(256, 16)
    y = (x[:, :1] * x[:, 1:2]).sin() + 0.5 * x[:, 2:3].abs()
    return x, y


def train(make_opt, steps=200, lr=1e-2):
    set_seed(0)
    model = nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 1))
    opt = make_opt(model.parameters(), lr)
    x, y = task()
    history = []
    for _ in range(steps):
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))
    return history, [p.detach().clone() for p in model.parameters()]


# ------------------------------------------------------------- 1. against torch


def vs_torch_experiment():
    console.rule("[bold]1. Your AdamW against PyTorch's[/bold]")

    h_mine, p_mine = train(
        lambda p, lr: AdamWScratch(p, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    )
    h_torch, p_torch = train(
        lambda p, lr: torch.optim.AdamW(p, lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    )

    err = max(float((a - b).abs().max()) for a, b in zip(p_mine, p_torch))
    console.print(
        f"  final loss:  yours {h_mine[-1]:.8f}   torch {h_torch[-1]:.8f}\n"
        f"  maximum weight error after 200 steps: [bold]{err:.2e}[/bold]\n"
    )
    console.print(
        "[green]Identical apart from fp32 rounding.[/green] You are doing exactly the same\n"
        "operations in the same order."
        if err < 1e-4
        else "[red]They diverge. Check the bias correction or the weight decay.[/red]"
    )
    return h_mine, h_torch


# ------------------------------------------------------------- 2. the pieces


class SimpleSGD(torch.optim.Optimizer):
    """Bare gradient descent, for comparison."""

    def __init__(self, params, lr=1e-2):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-g["lr"])


class AdamWithoutCorrection(torch.optim.Optimizer):
    """Adam without the bias correction, to see the damage in the first steps."""

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.95), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                # WITHOUT dividing by (1 - beta^t)
                p.addcdiv_(st["m"], st["v"].sqrt().add(g["eps"]), value=-g["lr"])


def pieces_experiment():
    console.rule("[bold]2. What each piece contributes[/bold]")

    variants = {
        "SGD (no momentum or scaling)": lambda p, lr: SimpleSGD(p, lr=lr),
        "Adam without bias correction": lambda p, lr: AdamWithoutCorrection(p, lr=lr),
        "full AdamW (yours)": lambda p, lr: AdamWScratch(p, lr=lr, betas=(0.9, 0.95)),
    }

    results = {}
    table = Table(header_style="bold")
    table.add_column("variant")
    table.add_column("loss step 1", justify="right")
    table.add_column("loss step 10", justify="right")
    table.add_column("final loss", justify="right")

    for name, make in variants.items():
        h, _ = train(make, steps=200)
        results[name] = h
        table.add_row(name, f"{h[0]:.4f}", f"{h[9]:.4f}", f"{h[-1]:.6f}")
    console.print(table)

    sgd = results["SGD (no momentum or scaling)"]
    full = results["full AdamW (yours)"]
    console.print(
        f"\nAdamW reaches {full[-1]:.6f} and SGD stops at {sgd[-1]:.6f} with the same lr\n"
        "and the same steps. That difference is the per-dimension scaling: each parameter\n"
        "ends up with its own effective learning rate.\n\n"
        "[dim]Look at the step-10 column of the version without the bias correction. That is\n"
        "where it shows: the first steps take larger jumps than they should, because m and v\n"
        "start at zero and underestimate the real magnitudes.[/dim]"
    )
    return results


# ------------------------------------------------------------- 3. the scheduler


def scheduler_experiment():
    console.rule("[bold]3. The learning-rate scheduler[/bold]")

    max_steps, lr, warmup = 10_172, 1e-3, 500
    steps = list(range(0, max_steps + 500, 20))
    curve = [lr_at_step(s, max_steps, lr, warmup, 0.1) for s in steps]

    table = Table(header_style="bold")
    table.add_column("step", justify="right")
    table.add_column("lr", justify="right")
    table.add_column("segment")
    for s, segment in [
        (0, "warmup (starts near zero)"),
        (250, "warmup (halfway)"),
        (500, "end of warmup: the maximum"),
        (2500, "cosine"),
        (5086, "cosine (halfway through training)"),
        (10_172, "end: the 10% floor"),
        (12_000, "past the end: it stays at the floor"),
    ]:
        table.add_row(f"{s:,}", f"{lr_at_step(s, max_steps, lr, warmup, 0.1):.3e}", segment)
    console.print(table)

    console.print(
        "\n[bold]Why the warmup.[/bold] In the first steps Adam's moments are nearly empty\n"
        "and its estimates are noisy, and on top of that the freshly initialized weights\n"
        "give large gradients. Starting at full lr usually produces a loss spike the model\n"
        "sometimes never recovers from.\n\n"
        "[bold]Why a cosine.[/bold] It drops slowly at the start (you still want to move\n"
        "fast), fast in the middle, and slowly at the end (fine-tuning).\n\n"
        "[bold]Why not to zero.[/bold] Below a certain point the model stops learning\n"
        "entirely and compute is wasted. 10% is the convention."
    )
    return steps, curve


# ------------------------------------------------------------- 4. the clipping


def clip_experiment():
    console.rule("[bold]4. Gradient clipping[/bold]")

    set_seed(0)
    model = nn.Linear(64, 64)
    # A "toxic batch": it produces enormous gradients
    ((model(torch.randn(8, 64)) ** 2).sum() * 500).backward()

    before = torch.cat([p.grad.flatten().clone() for p in model.parameters()])
    norm = clip_grad_norm(model.parameters(), 1.0)
    after = torch.cat([p.grad.flatten() for p in model.parameters()])

    cosine = float(torch.dot(before, after) / (before.norm() * after.norm()))
    console.print(
        f"  norm BEFORE clipping : [red]{norm:,.1f}[/red]\n"
        f"  norm AFTER           : [green]{float(after.norm()):.4f}[/green]\n"
        f"  cosine between the two directions: [bold]{cosine:.8f}[/bold]\n"
    )
    console.print(
        "That cosine of 1.0 is what matters: [bold]the direction has not changed[/bold], only\n"
        "the magnitude. The gradient still points exactly where it pointed.\n\n"
        "[dim]If you clipped each tensor separately, each would be scaled by a different\n"
        "factor and the combined direction WOULD change. That is why the global norm is\n"
        "used.[/dim]"
    )

    # And the effect on a training run
    console.print("\n[bold]On a training run with a poisoned batch at step 50:[/bold]\n")
    results = {}
    for label, threshold in (("without clipping", 0.0), ("with grad_clip=1.0", 1.0)):
        set_seed(0)
        m = nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 1))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        x, y = task()
        hist = []
        for step in range(150):
            xb, yb = (x, y) if step != 50 else (x * 50, y * 50)  # the toxic batch
            loss = ((m(xb) - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            if threshold > 0:
                clip_grad_norm(m.parameters(), threshold)
            opt.step()
            hist.append(float(((m(x) - y) ** 2).mean().detach()))
        results[label] = hist
        console.print(
            f"  {label:<20} loss at step 49: {hist[49]:.4f}  ->  "
            f"step 55: {hist[55]:.4f}  ->  final: {hist[-1]:.4f}"
        )

    without, with_clip = results["without clipping"], results["with grad_clip=1.0"]
    damage_without = without[55] / max(without[49], 1e-9)
    damage_with = with_clip[55] / max(with_clip[49], 1e-9)
    verdict = (
        f"  without clipping  : the loss RISES {damage_without:.1f}x after the toxic batch\n"
        + (
            f"  with grad_clip=1.0: it does not even notice ({damage_with:.1f}x, still "
            "dropping)\n"
            if damage_with < 1.0
            else f"  with grad_clip=1.0: it only rises {damage_with:.1f}x\n"
        )
    )
    console.print(
        f"\n{verdict}"
        "[dim]With grad_clip the maximum damage from any one batch is bounded. In a run of\n"
        "thousands of steps, a single odd batch can cost you the whole training.[/dim]"
    )
    return results


# ------------------------------------------------------------- 5. param groups


def groups_experiment():
    console.rule("[bold]5. Which parameters decay[/bold]")
    model = GPT(ModelConfig())
    groups = build_param_groups(model, 0.1)

    table = Table(header_style="bold")
    table.add_column("group")
    table.add_column("weight_decay", justify="right")
    table.add_column("tensors", justify="right")
    table.add_column("parameters", justify="right")
    table.add_column("what is inside")
    table.add_row(
        "with decay",
        f"{groups[0]['weight_decay']}",
        str(len(groups[0]["params"])),
        f"{sum(p.numel() for p in groups[0]['params']):,}",
        "matrices: embeddings and projections",
    )
    table.add_row(
        "without decay",
        f"{groups[1]['weight_decay']}",
        str(len(groups[1]["params"])),
        f"{sum(p.numel() for p in groups[1]['params']):,}",
        "RMSNorm scales",
    )
    console.print(table)
    console.print(
        "\nAn RMSNorm's scale starts at 1 and its job is to rescale the layer's output.\n"
        "Pushing it towards zero is pushing the output towards zero: exactly the opposite\n"
        "of what is needed.\n\n"
        "[dim]Applying decay to everything is a common mistake, it produces no visible\n"
        "error, and it degrades the result. It can only be detected by comparing two\n"
        "complete training runs.[/dim]"
    )


def main() -> None:
    h_mine, h_torch = vs_torch_experiment()
    variants = pieces_experiment()
    steps, curve = scheduler_experiment()
    clip = clip_experiment()
    groups_experiment()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    for name, hist in variants.items():
        axes[0].plot(hist, label=name, lw=1.2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss (log scale)")
    axes[0].set_title("What each piece of Adam contributes")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].plot(steps, curve, color="tab:orange")
    axes[1].axvline(500, color="gray", ls="--", lw=1, label="end of warmup")
    axes[1].axhline(1e-4, color="red", ls=":", lw=1, label="floor (10%)")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("Warmup + cosine")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    for name, hist in clip.items():
        axes[2].plot(hist, label=name, lw=1.2)
    axes[2].axvline(50, color="red", ls="--", lw=1, label="toxic batch")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("loss (log scale)")
    axes[2].set_title("An odd batch, with and without clipping")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    target = figures_dir() / "11_training_loop.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")


if __name__ == "__main__":
    main()
