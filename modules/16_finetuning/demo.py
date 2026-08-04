"""Module 16 demo: real SFT on the model from module 13.

    llmfs demo 16

It trains the model on a small instruction-answer set and compares the behaviour before and
after. It takes about 30 seconds.
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
from llmfs.config import RunConfig
from llmfs.data import prepare
from llmfs.device import get_device, set_seed
from llmfs.paths import configs_dir, figures_dir
from llmfs.reference import GPT, apply_lora_to_model, count_trainable

console = Console()

build_chat_template = resolve("16_finetuning", "build_chat_template")
mask_prompt_tokens = resolve("16_finetuning", "mask_prompt_tokens")
LoRALinear = resolve("16_finetuning", "LoRALinear")
merge_lora_weights = resolve("16_finetuning", "merge_lora_weights")
generate_with_cache = resolve("14_inference", "generate_with_cache")

# A tiny SFT set, in the style of the corpus (Shakespeare) so the model has some chance.
# With 9M parameters and a dataset like this, what gets learned is the FORMAT, not the
# content.
EXAMPLES = [
    ("Who is the king?", "The king is Richard."),
    ("What did the lady say?", "She said farewell."),
    ("Where is the lord?", "The lord is in the castle."),
    ("Who loves Juliet?", "Romeo loves Juliet."),
    ("What is the news?", "The news is grave."),
    ("Who speaks now?", "The duke speaks now."),
    ("Where is the queen?", "The queen is in her chamber."),
    ("What say you?", "I say we must go."),
] * 12


def format_experiment():
    console.rule("[bold]1. The chat template[/bold]")
    msgs = [
        {"role": "user", "content": "Who is the king?"},
        {"role": "assistant", "content": "The king is Richard."},
    ]
    console.print(f"  training  : [cyan]{build_chat_template(msgs)}[/cyan]")
    console.print(
        f"  inference : [cyan]{build_chat_template(msgs[:1], add_generation_prompt=True)}"
        f"[/cyan]  [dim]<- open, the model continues here[/dim]\n"
    )
    console.print(
        "[dim]The markers are not magical: they are text the model learns to recognize\n"
        "during SFT. The `<|end|>` is what teaches it WHEN TO STOP; without it, it would\n"
        "generate indefinitely.[/dim]\n"
    )

    console.print("[bold]And the prompt masking:[/bold]\n")
    ids = [10, 11, 12, 20, 21, 22]
    targets = mask_prompt_tokens(ids, 3)
    table = Table(header_style="bold")
    table.add_column("position", justify="right")
    table.add_column("input", justify="right")
    table.add_column("target", justify="right")
    table.add_column("")
    for i, (a, b) in enumerate(zip(ids, targets)):
        if i < 2:
            note = "[dim]prompt: ignored[/dim]"
        elif i == 2:
            note = "[bold green]the transition: it DOES learn[/bold green]"
        elif i == len(ids) - 1:
            note = "[dim]there is no next token[/dim]"
        else:
            note = "answer"
        table.add_row(str(i), str(a), str(b) if b != -100 else "[dim]-100[/dim]", note)
    console.print(table)
    console.print(
        "\n[dim]Look at position 2: it is the LAST token of the prompt, but its target is\n"
        "already the first one of the answer. That 'the question is over, my turn'\n"
        "transition is the most important thing it has to learn, so it is NOT masked. That\n"
        "is why there are two ignored positions and not three: it is the exercise's\n"
        "off-by-one.[/dim]"
    )


def lora_experiment():
    console.rule("[bold]2. LoRA: training 1% of the model[/bold]")

    cfg = RunConfig.from_yaml(configs_dir() / "tinystories_9m.yaml")
    table = Table(header_style="bold")
    table.add_column("rank r", justify="right")
    table.add_column("trainable", justify="right")
    table.add_column("% of the model", justify="right")
    table.add_column("AdamW memory", justify="right")

    ranks, percentages = [], []
    full = 8_933_440
    table.add_row(
        "[dim]no LoRA[/dim]",
        f"[dim]{full:,}[/dim]",
        "[dim]100%[/dim]",
        f"[dim]{full * 8 / 1e6:.0f} MB[/dim]",
    )
    for r in (4, 8, 16, 32):
        set_seed(0)
        model = apply_lora_to_model(GPT(cfg.model), r=r)
        c = count_trainable(model)
        ranks.append(r)
        percentages.append(c["percent"])
        table.add_row(
            str(r),
            f"{c['trainable']:,}",
            f"{c['percent']:.2f}%",
            f"{c['trainable'] * 8 / 1e6:.2f} MB",
        )
    console.print(table)
    console.print(
        "\n[dim]AdamW's memory is 8 bytes per trainable parameter (two moments in fp32).\n"
        "With LoRA r=8 it drops from 71 MB to 0.5 MB. On a 70B model, that is the difference\n"
        "between needing eight GPUs or one.[/dim]\n"
    )

    console.print("[bold]And the weight merging:[/bold]")
    set_seed(0)
    base = torch.nn.Linear(320, 320)
    lora = LoRALinear(base, r=8)
    x = torch.randn(4, 320)
    console.print(
        f"  on initialization, output identical to the base: "
        f"[green]{torch.allclose(lora(x), base(x), atol=1e-6)}[/green]  "
        "[dim](because lora_B starts at zeros)[/dim]"
    )
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)
    merged = merge_lora_weights(lora)
    err = float((lora(x) - merged(x)).abs().max())
    console.print(
        f"  after adapting, the merged layer gives the same: error {err:.2e}  "
        + ("[green]OK[/green]" if err < 1e-5 else "[red]WRONG[/red]")
    )
    console.print(
        "\n[dim]That is what makes LoRA useful compared to other efficient fine-tuning\n"
        "methods: the adaptation is EXACTLY a sum of matrices, so it is absorbed without\n"
        "approximating anything. The resulting model is indistinguishable from a normal\n"
        "one.[/dim]"
    )
    return ranks, percentages


def sft_experiment(cfg_dev):
    console.rule("[bold]3. Real SFT[/bold]")

    cfg = RunConfig.from_yaml(configs_dir() / "tiny_char.yaml")
    dataset = prepare(cfg, quiet=True)
    cfg.model.vocab_size = dataset.vocab_size

    from llmfs.train import load_checkpoint

    model = GPT(cfg.model)
    path = cfg.run_dir / "best.pt"
    if not path.exists():
        console.print(
            "[yellow]There is no trained model. Train one with "
            "`llmfs train --config tiny_char` and come back.[/yellow]"
        )
        return None
    load_checkpoint(path, model, map_location="cpu")
    model = model.to(cfg_dev.device)

    # The chat template uses characters the character tokenizer does not know, so simple
    # markers made of characters from the corpus are used instead.
    def format_pair(question: str, answer: str) -> tuple[str, str]:
        return f"Q: {question}\nA: ", f"{answer}\n"

    example_ids = []
    for question, answer in EXAMPLES:
        p, a = format_pair(question, answer)
        ids_p, ids_a = dataset.encode(p), dataset.encode(a)
        whole = ids_p + ids_a
        if len(whole) <= cfg.model.context_length:
            example_ids.append((whole, len(ids_p)))

    console.print(
        f"{len(example_ids)} instruction-answer examples, format `Q: ...\\nA: ...`\n"
        "[dim](this format is used instead of the <|user|> markers because the character\n"
        "tokenizer only knows the symbols that appear in Shakespeare)[/dim]\n"
    )

    test_prompt = "Q: Who is the king?\nA:"

    def generate() -> str:
        ids = torch.tensor([dataset.encode(test_prompt)], device=cfg_dev.device)
        set_seed(7)
        out = generate_with_cache(model, ids, 60, temperature=0.7, top_k=20)
        return dataset.decode(out[0].tolist())

    model.eval()
    before = generate()

    # SFT
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    history = []
    set_seed(0)
    for _ in range(150):
        idx = torch.randint(0, len(example_ids), (8,))
        max_len = max(len(example_ids[i][0]) for i in idx)
        x = torch.full((8, max_len), 0, dtype=torch.long)
        y = torch.full((8, max_len), -100, dtype=torch.long)
        for row, i in enumerate(idx):
            ids, plen = example_ids[i]
            x[row, : len(ids)] = torch.tensor(ids)
            y[row, : len(ids)] = torch.tensor(mask_prompt_tokens(ids, plen))
        x, y = x.to(cfg_dev.device), y.to(cfg_dev.device)

        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))

    model.eval()
    after = generate()

    console.print(f"  SFT loss: {history[0]:.4f} -> {history[-1]:.4f}\n")
    console.print(Panel(before.replace("\n", " ⏎ "), title="BEFORE the SFT", border_style="red"))
    console.print(Panel(after.replace("\n", " ⏎ "), title="AFTER the SFT", border_style="green"))

    console.print(
        "\n[bold]What you have to look at is not whether the answer is correct.[/bold] With\n"
        "0.8M parameters and 96 examples, it is not going to be.\n\n"
        "What you have to look at is the FORMAT: if after the SFT it answers something short\n"
        "and stops, instead of carrying on writing Shakespeare indefinitely, the\n"
        "post-training has done its job.\n\n"
        "[dim]And that is precisely the module's lesson: post-training does NOT add\n"
        "knowledge. It brings to the surface a behaviour that was already latent. A model\n"
        "that does not know something after pretraining does not learn it from a thousand\n"
        "conversation examples.[/dim]"
    )
    return history


def main() -> None:
    cfg_dev = get_device()
    format_experiment()
    ranks, percentages = lora_experiment()
    history = sft_experiment(cfg_dev)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    left.plot(ranks, percentages, marker="o", color="tab:green")
    left.axhline(100, color="red", ls="--", lw=1, label="full fine-tuning")
    left.set_xlabel("LoRA rank r")
    left.set_ylabel("% of trainable parameters")
    left.set_yscale("log")
    left.set_title("LoRA trains a tiny fraction")
    left.grid(alpha=0.3)
    left.legend(fontsize=8)

    if history:
        right.plot(history, color="tab:blue")
        right.set_xlabel("SFT step")
        right.set_ylabel("loss (only over the answer)")
        right.set_title("SFT learns the format")
        right.grid(alpha=0.3)

    fig.tight_layout()
    target = figures_dir() / "16_finetuning.png"
    fig.savefig(target, dpi=120)
    plt.close(fig)
    console.print(f"\n[green]figure saved to {target}[/green]")


if __name__ == "__main__":
    main()
