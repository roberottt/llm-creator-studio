"""Module 17 demo: quantizes your model and measures the damage. And closes the course.

    llmfs demo 17
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmfs.bridge import resolve
from llmfs.config import ModelConfig
from llmfs.device import get_device, set_seed
from llmfs.paths import figures_dir
from llmfs.reference import GPT

console = Console()

quantize_int8_symmetric = resolve("17_extra", "quantize_int8_symmetric")
dequantize_int8 = resolve("17_extra", "dequantize_int8")
quantization_error = resolve("17_extra", "quantization_error")


def example_experiment():
    console.rule("[bold]1. Quantizing, with numbers you can follow[/bold]")
    w = torch.tensor([[0.12, -0.45, 0.03, 0.28]])
    q, scale = quantize_int8_symmetric(w)
    rec = dequantize_int8(q, scale)

    table = Table(header_style="bold")
    table.add_column("original", justify="right")
    table.add_column("int8", justify="right")
    table.add_column("recovered", justify="right")
    table.add_column("error", justify="right")
    for i in range(4):
        table.add_row(
            f"{float(w[0, i]):+.4f}",
            f"{int(q[0, i]):>4}",
            f"{float(rec[0, i]):+.4f}",
            f"{abs(float(w[0, i]) - float(rec[0, i])):.4f}",
        )
    console.print(table)
    console.print(
        f"  scale = max(|W|) / 127 = {float(w.abs().max()):.2f} / 127 = "
        f"[bold]{float(scale):.6f}[/bold]\n\n"
        "[dim]The -0.45 is recovered exactly because it is the maximum and maps right onto\n"
        "-127. The rest lose up to half a unit of scale. And zero, if there were one, would\n"
        "be represented exactly: that is why 127 is used and not 128, so the range stays\n"
        "symmetric.[/dim]"
    )


def model_experiment():
    console.rule("[bold]2. On the real model[/bold]")
    set_seed(0)
    model = GPT(ModelConfig())

    table = Table(header_style="bold")
    table.add_column("matrix")
    table.add_column("shape", justify="right")
    table.add_column("error per channel", justify="right")
    table.add_column("error per tensor", justify="right")

    examples = [
        ("token_embedding", model.token_embedding.weight),
        ("q_proj (layer 0)", model.blocks[0].attn.q_proj.weight),
        ("gate_proj (layer 0)", model.blocks[0].ffn.gate_proj.weight),
        ("down_proj (layer 5)", model.blocks[5].ffn.down_proj.weight),
    ]
    for name, w in examples:
        e_c = quantization_error(w.data, per_channel=True)["relative_error"]
        e_t = quantization_error(w.data, per_channel=False)["relative_error"]
        table.add_row(name, str(tuple(w.shape)), f"{e_c * 100:.3f}%", f"{e_t * 100:.3f}%")
    console.print(table)

    total_fp32 = sum(p.numel() * 4 for p in model.parameters())
    matrices = [p for p in model.parameters() if p.dim() >= 2]
    total_int8 = sum(quantization_error(p.data)["quantized_bytes"] for p in matrices) + sum(
        p.numel() * 4 for p in model.parameters() if p.dim() < 2
    )

    console.print(
        f"\n  model in fp32 : [bold]{total_fp32 / 1e6:.1f} MB[/bold]\n"
        f"  model in int8 : [bold]{total_int8 / 1e6:.1f} MB[/bold]  "
        f"([green]{total_fp32 / total_int8:.1f}x smaller[/green])\n\n"
        "[dim]Per channel is always better than per tensor: a single row with large values\n"
        "does not drag the others along. It costs one extra vector of scales, which is\n"
        "negligible.\n\n"
        "That a 0.7% error in the weights barely affects the model's quality is an EMPIRICAL\n"
        "FACT, not a theorem. Nobody predicted that networks would be so robust to\n"
        "quantization: it was discovered by trying.\n\n"
        "And a nuance that is usually left out: quantizing the weights does NOT speed\n"
        "anything up on its own if afterwards you convert to float to multiply. Real\n"
        "acceleration requires kernels that operate natively in int8.[/dim]"
    )
    return examples


def closing_experiment():
    console.rule("[bold]3. What separates you from a frontier model[/bold]")

    table = Table(header_style="bold")
    table.add_column("")
    table.add_column("yours", justify="right")
    table.add_column("GPT-4 (estimated)", justify="right")
    table.add_column("factor", justify="right")

    rows = [
        ("parameters", 8.93e6, 1.8e12),
        ("training tokens", 5e8, 1.3e13),
        ("training FLOPs", 2.3e16, 2e25),
        ("approximate cost", 0.5, 1e8),
    ]
    units = ["", "", " FLOPs", " EUR"]
    for (name, mine, theirs), unit in zip(rows, units):
        table.add_row(name, f"{mine:.3g}{unit}", f"{theirs:.3g}{unit}", f"{theirs / mine:.0e}x")
    console.print(table)

    console.print(
        "\n[bold]But size is only one of five things:[/bold]\n\n"
        "  1. [bold]DATA[/bold]: 15 trillion tokens filtered with classifiers, deduplicated,\n"
        "     and with a lot of code and mathematics because they improve reasoning on tasks\n"
        "     that are NEITHER code nor mathematics. Nobody publishes the exact recipe.\n\n"
        "  2. [bold]COMPUTE[/bold]: nine orders of magnitude, and not only in GPUs: the data\n"
        "     centre, the network, and the engineers keeping that up for months.\n\n"
        "  3. [bold]ARCHITECTURE[/bold]: Mixture of Experts, which activates only a fraction\n"
        "     of the parameters per token. A trillion parameters with the cost of a hundred\n"
        "     billion.\n\n"
        "  4. [bold]POST-TRAINING[/bold]: RLHF/DPO after the SFT, plus months of red-teaming\n"
        "     and adjustment. It is the phase that turns a text predictor into something you\n"
        "     would want to use, and it employs more people than pretraining.\n\n"
        "  5. [bold]INFRASTRUCTURE[/bold]: parallelism along several dimensions, fault\n"
        "     tolerance (with thousands of GPUs one fails every few hours), and being able to\n"
        "     resume without losing days.\n"
    )

    console.print(
        Panel(
            "[bold]And this is what you have achieved:[/bold]\n\n"
            "You have written ALL the pieces: attention, RoPE, SwiGLU, AdamW, the KV cache,\n"
            "the tokenizer. All validated numerically against PyTorch. A frontier model uses\n"
            "exactly these pieces, bigger and with more engineering around them.\n\n"
            "You know how to read an architecture paper. When the next model comes out and\n"
            "they say it uses grouped-query attention or RMSNorm, you know what they are and\n"
            "why.\n\n"
            "You know how to debug a training run: the step-0 loss against ln(V), the overfit\n"
            "on a batch, the gradient norm, the MFU. That is what separates someone who knows\n"
            "how to train models from someone who copies scripts.\n\n"
            "And you know WHAT IS NOT KNOWN: that SwiGLU works without an explanation, that\n"
            "Adam dominates without anyone quite knowing why, that scaling laws have wider\n"
            "intervals than reported, and that the benchmarks are contaminated. That part\n"
            "does not usually appear in the tutorials, and it is the one that helps most for\n"
            "reading with judgement.",
            title="[bold green]End of the course[/bold green]",
            border_style="green",
        )
    )


def main() -> None:
    get_device()
    example_experiment()
    examples = model_experiment()
    closing_experiment()

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    names = [n.split(" ")[0] for n, _ in examples]
    e_channel = [quantization_error(w.data, True)["relative_error"] * 100 for _, w in examples]
    e_tensor = [quantization_error(w.data, False)["relative_error"] * 100 for _, w in examples]
    x = range(len(names))
    left.bar([i - 0.2 for i in x], e_channel, 0.4, label="per channel")
    left.bar([i + 0.2 for i in x], e_tensor, 0.4, label="per tensor")
    left.set_xticks(list(x))
    left.set_xticklabels(names, fontsize=7, rotation=15)
    left.set_ylabel("relative error (%)")
    left.set_title("The damage of int8, per matrix")
    left.grid(alpha=0.3, axis="y")
    left.legend(fontsize=8)

    scales = ["yours\n8.9M", "GPT-2\n1.5B", "Llama-3\n70B", "GPT-4\n~1.8T"]
    params = [8.93e6, 1.5e9, 7e10, 1.8e12]
    right.bar(scales, params, color=["tab:green", "tab:blue", "tab:orange", "tab:red"])
    right.set_yscale("log")
    right.set_ylabel("parameters (log scale)")
    right.set_title("Where your model is")
    right.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    target = figures_dir() / "17_extra.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")


if __name__ == "__main__":
    main()
