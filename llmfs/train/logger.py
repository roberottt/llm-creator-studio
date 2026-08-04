"""Training logs: CSV, text samples and plots.

During a run that lasts hours you need two things: numbers you can plot afterwards, and
text samples so you can SEE the model learning to write. The second is what makes the log
worth looking at.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class TrainingLogger:
    """Writes the metrics CSV and the text-sample log.

    The CSV is opened in append mode and `flush()`ed on every row. It costs a little more,
    but if the process dies you do not lose the history, and you can plot the curve while
    it trains from another terminal.
    """

    COLUMNS = [
        "step",
        "tokens",
        "train_loss",
        "val_loss",
        "lr",
        "grad_norm",
        "tokens_per_second",
        "mfu",
        "seconds",
    ]

    def __init__(self, run_dir: str | Path, resuming: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.run_dir / "metrics.csv"
        self.samples_path = self.run_dir / "samples.md"
        self.started = time.perf_counter()

        fresh = not self.csv_path.exists() or not resuming
        self._csv = self.csv_path.open("a" if resuming else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv, fieldnames=self.COLUMNS)
        if fresh:
            self._writer.writeheader()
            self._csv.flush()

        if not resuming:
            self.samples_path.write_text(
                f"# Training samples\n\n"
                f"Started on {datetime.now():%Y-%m-%d %H:%M}.\n\n"
                f"Read this top to bottom when it finishes: it is the model learning to\n"
                f"write, step by step.\n",
                encoding="utf-8",
            )

    def log(self, **fields: Any) -> None:
        row = {c: fields.get(c, "") for c in self.COLUMNS}
        row["seconds"] = round(time.perf_counter() - self.started, 2)
        self._writer.writerow(row)
        self._csv.flush()

    def log_sample(self, step: int, prompt: str, text: str, val_loss: float | None = None) -> None:
        """Append a generated text sample to the samples file."""
        with self.samples_path.open("a", encoding="utf-8") as f:
            header = f"\n## Step {step:,}"
            if val_loss is not None:
                header += f" — validation loss {val_loss:.4f}"
            f.write(f"{header}\n\n**Prompt:** `{prompt}`\n\n```\n{text}\n```\n")

    def close(self) -> None:
        if not self._csv.closed:
            self._csv.close()

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ plots

    def plot(self, target: str | Path | None = None) -> Path | None:
        """Draw the loss and learning-rate curves from the CSV."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = self.read_csv()
        if not rows:
            return None

        steps_tr = [r["step"] for r in rows if r.get("train_loss") is not None]
        losses_tr = [r["train_loss"] for r in rows if r.get("train_loss") is not None]
        steps_val = [r["step"] for r in rows if r.get("val_loss") is not None]
        losses_val = [r["val_loss"] for r in rows if r.get("val_loss") is not None]

        fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

        left.plot(steps_tr, losses_tr, lw=1, alpha=0.7, label="train")
        if losses_val:
            left.plot(steps_val, losses_val, marker="o", ms=3, label="validation")
        left.set_xlabel("step")
        left.set_ylabel("loss (nats)")
        left.set_title("Loss curve")
        left.grid(alpha=0.3)
        left.legend()

        lrs = [(r["step"], r["lr"]) for r in rows if r.get("lr") is not None]
        if lrs:
            right.plot([s for s, _ in lrs], [v for _, v in lrs], color="tab:orange")
        right.set_xlabel("step")
        right.set_ylabel("learning rate")
        right.set_title("Scheduler (warmup + cosine)")
        right.grid(alpha=0.3)

        fig.tight_layout()
        target = Path(target) if target else self.run_dir / "curves.png"
        fig.savefig(target, dpi=120)
        plt.close(fig)
        return target

    def read_csv(self) -> list[dict[str, Any]]:
        """Read the CSV, converting to float whatever can be converted."""
        if not self.csv_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.csv_path.open(encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                row: dict[str, Any] = {}
                for k, v in raw.items():
                    if v == "" or v is None:
                        row[k] = None
                    else:
                        try:
                            row[k] = float(v)
                        except ValueError:
                            row[k] = v
                rows.append(row)
        return rows
