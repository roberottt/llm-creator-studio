"""Demo for module 02: look inside the compute graph.

    llmfs demo 02

Three experiments:
  1. A small graph, node by node, with the gradient each one receives. Next to it, what
     torch.autograd says for the same expression.
  2. The `+=` experiment: what exactly happens if you accumulate gradients wrongly.
  3. Training an MLP with your engine and plotting the loss curve.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no window: this has to run over SSH and in CI

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.paths import figures_dir

console = Console()

Value = resolve("02_autograd", "Value")
topological_order = resolve("02_autograd", "topological_order")
train_scalar_mlp = resolve("02_autograd", "train_scalar_mlp")


def experiment_1_the_graph() -> None:
    console.rule("[bold]1. A compute graph, from the inside[/bold]")
    console.print("Expression:  [cyan]L = (a * b + a.tanh()) * 2[/cyan]   with a=2.0, b=-3.0\n")

    a, b = Value(2.0), Value(-3.0)
    a.label, b.label = "a", "b"
    two = Value(2.0)
    two.label = "2 (constant)"
    product = a * b
    activation = a.tanh()
    total = product + activation
    L = total * two
    L.backward()

    # The same thing with torch, for a second opinion.
    ta = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    ((ta * tb + ta.tanh()) * 2.0).backward()

    order = topological_order(L)
    table = Table(header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("op")
    table.add_column("value", justify="right")
    table.add_column("gradient", justify="right")
    table.add_column("what it means")

    meanings = {
        "": "leaf: an input number, it comes from no operation",
        "*": "product",
        "+": "sum",
        "tanh": "nonlinear activation",
    }
    for i, node in enumerate(order):
        table.add_row(
            str(i),
            node._op or (node.label or "leaf"),
            f"{node.data:.4f}",
            f"{node.grad:+.4f}",
            meanings.get(node._op, ""),
        )
    console.print(table)

    console.print(
        f"\nTopological order: {len(order)} nodes. The backward pass walks them IN REVERSE,\n"
        f"from the last one (the output, gradient 1) to the first.\n"
    )
    console.print("[bold]Check against torch.autograd:[/bold]")
    console.print(f"  dL/da  ->  yours {a.grad:+.10f}   torch {ta.grad.item():+.10f}")
    console.print(f"  dL/db  ->  yours {b.grad:+.10f}   torch {tb.grad.item():+.10f}")
    matches = abs(a.grad - ta.grad.item()) < 1e-9 and abs(b.grad - tb.grad.item()) < 1e-9
    console.print(
        "[green]  Identical. Your engine computes the same thing as PyTorch.[/green]"
        if matches
        else "[red]  They do not match. Check the local derivatives.[/red]"
    )


def experiment_2_accumulation() -> None:
    console.rule("[bold]2. Why the gradient ACCUMULATES[/bold]")
    console.print(
        "The minimal case that separates `+=` from `=`:  [cyan]y = x + x[/cyan]  with x = 3.\n"
        "Since y = 2x, the correct derivative is 2.\n"
    )

    x = Value(3.0)
    (x + x).backward()
    console.print(f"  with accumulation (`+=`) :  dy/dx = {x.grad:.1f}   [green](correct)[/green]")
    console.print(
        "  with assignment (`=`)    :  dy/dx = 1.0   "
        "[red](wrong: the second branch overwrites the first)[/red]"
    )

    console.print("\nAnd with a node used 10 times:")
    z = Value(1.5)
    total = z
    for _ in range(9):
        total = total + z
    total.backward()
    console.print(f"  x appears 10 times  ->  dL/dx = {z.grad:.1f}   [green](correct: 10)[/green]")

    console.print(
        "\n[dim]This is exactly why optimizer.zero_grad() exists. If you do not clear the\n"
        "gradients between steps, step N uses the sum of the gradients from steps 1 to N.\n"
        "It gives no error at all: the model simply does not learn.[/dim]"
    )


def experiment_3_training() -> tuple[list[float], list[float]]:
    console.rule("[bold]3. Training an MLP with your engine[/bold]")

    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    console.print("Four points, two classes (+1 / -1). MLP of 3 -> 8 -> 8 -> 1.\n")

    fast = train_scalar_mlp(xs, ys, hidden=(8, 8), steps=100, lr=0.05, seed=0)
    slow = train_scalar_mlp(xs, ys, hidden=(8, 8), steps=100, lr=0.005, seed=0)

    console.print(f"  lr=0.05  :  loss {fast[0]:.4f}  ->  {fast[-1]:.6f}")
    console.print(f"  lr=0.005 :  loss {slow[0]:.4f}  ->  {slow[-1]:.6f}")
    console.print(
        "\n[dim]Same model, same initialization, only the learning rate changes. Ten times\n"
        "smaller and it does not have time to converge. This hyperparameter ruins more\n"
        "training runs than any other, and you will see it again in module 11.[/dim]"
    )
    return fast, slow


def main() -> None:
    experiment_1_the_graph()
    experiment_2_accumulation()
    fast, slow = experiment_3_training()

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    left.plot(fast, label="lr = 0.05")
    left.plot(slow, label="lr = 0.005")
    left.set_xlabel("step")
    left.set_ylabel("loss (MSE)")
    left.set_title("Training with a hand-written autodiff engine")
    left.grid(alpha=0.3)
    left.legend()

    right.plot(fast, label="lr = 0.05")
    right.plot(slow, label="lr = 0.005")
    right.set_yscale("log")
    right.set_xlabel("step")
    right.set_ylabel("loss (log scale)")
    right.set_title("The same curve in log: the convergence shows")
    right.grid(alpha=0.3)
    right.legend()

    fig.tight_layout()
    target = figures_dir() / "02_autograd.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")


if __name__ == "__main__":
    main()
