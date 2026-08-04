"""The complete training loop.

Everything from modules 11-13 put together and working: gradient accumulation, mixed
precision, clipping, scheduler, periodic evaluation, text samples and resumable
checkpoints.

The pieces are requested from the bridge, so when your exercises are right, this trains
with YOUR AdamW and YOUR scheduler.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from llmfs.bridge import resolve
from llmfs.config import RunConfig
from llmfs.device import DeviceConfig, get_device, maybe_compile, set_seed
from llmfs.train.checkpoint import load_checkpoint, save_checkpoint
from llmfs.train.logger import TrainingLogger


@dataclass
class TrainState:
    """Everything that changes during training and has to go into the checkpoint."""

    step: int = 0
    tokens_seen: int = 0
    best_val_loss: float = float("inf")
    history: list[dict[str, Any]] = field(default_factory=list)


class Trainer:
    """Orchestrates the training of a model.

    Usage:
        trainer = Trainer(cfg, model, get_batch_fn)
        trainer.train()

    `get_batch_fn(split, batch_size)` returns `(x, y)` already on the device. It is passed
    in as a function so the trainer knows nothing about how the data is stored.
    """

    def __init__(
        self,
        cfg: RunConfig,
        model: torch.nn.Module,
        get_batch: Callable[[str, int], tuple[torch.Tensor, torch.Tensor]],
        device: DeviceConfig | None = None,
        run_dir: str | Path | None = None,
        on_sample: Callable[[int], str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or get_device(prefer=cfg.train.device, amp=cfg.train.amp)
        self.model = model.to(self.device.device)
        self.get_batch = get_batch
        self.on_sample = on_sample
        self.run_dir = Path(run_dir) if run_dir else cfg.run_dir
        self.state = TrainState()

        # The student's pieces, or the reference if they have not been written yet.
        self._lr_at_step = resolve("11_training_loop", "lr_at_step")
        self._clip = resolve("11_training_loop", "clip_grad_norm")
        self._param_groups = resolve("11_training_loop", "build_param_groups")
        self._flops = resolve("12_efficiency_and_scaling", "model_flops_per_token")
        self._mfu = resolve("12_efficiency_and_scaling", "compute_mfu")

        self.optimizer = self._build_optimizer()
        self.scaler = self.device.grad_scaler()
        self.model = maybe_compile(self.model, cfg.train.compile, self.device)

    # ------------------------------------------------------------------ setup

    def _build_optimizer(self) -> torch.optim.Optimizer:
        groups = self._param_groups(self.model, self.cfg.train.weight_decay)
        if self.cfg.train.optimizer == "adamw_scratch":
            AdamWScratch = resolve("11_training_loop", "AdamWScratch")
            return AdamWScratch(
                groups, lr=self.cfg.train.lr, betas=tuple(self.cfg.train.betas)
            )
        # `fused=True` is noticeably faster on CUDA; it does not exist on MPS/CPU.
        extra: dict[str, Any] = {}
        if self.device.kind == "cuda":
            extra["fused"] = True
        return torch.optim.AdamW(
            groups, lr=self.cfg.train.lr, betas=tuple(self.cfg.train.betas), **extra
        )

    # ------------------------------------------------------------------ evaluation

    @torch.no_grad()
    def evaluate(self, iters: int | None = None) -> dict[str, float]:
        """Mean loss on train and on validation.

        `model.eval()` and `torch.no_grad()` are both essential: without the first, dropout
        would still be active and the loss would come out worse than it really is; without
        the second, the autograd graph would be built for nothing and burn memory.
        """
        iters = iters or self.cfg.train.eval_iters
        self.model.eval()
        out: dict[str, float] = {}

        for split in ("train", "val"):
            total = 0.0
            n = 0
            for _ in range(iters):
                try:
                    x, y = self.get_batch(split, self.cfg.train.batch_size)
                except (KeyError, ValueError):
                    break
                with self.device.autocast():
                    _, loss = self.model(x, y)
                total += float(loss.detach())
                n += 1
            if n:
                out[split] = total / n

        self.model.train()
        return out

    # ------------------------------------------------------------------ one step

    def step(self) -> dict[str, float]:
        """One complete optimization step, with gradient accumulation.

        The order matters, and it is this one:

            1. lr from the scheduler -> into the optimizer's groups
            2. `grad_accum` micro-batches: forward + backward, accumulating gradients
            3. unscale the gradients (if there is a scaler) BEFORE clipping
            4. clip by global norm
            5. optimizer step
            6. clear the gradients

        Step 3 is easy to forget and silent: with AMP the gradients are multiplied by the
        scaler's factor (around 65,000), so their norm is too. Clipping without unscaling
        would clip to a threshold 65,000 times smaller than you think, and training would
        crawl with nothing to indicate why.
        """
        t = self.cfg.train
        lr = self._lr_at_step(
            self.state.step,
            self.cfg.max_steps,
            t.lr,
            warmup_steps=t.warmup_steps,
            min_lr_ratio=t.min_lr_ratio,
            schedule=t.schedule,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        total_loss = 0.0
        for _ in range(t.grad_accum):
            x, y = self.get_batch("train", t.batch_size)
            with self.device.autocast():
                _, loss = self.model(x, y)
                # Divide by grad_accum: the sum over the micro-batches has to equal the
                # mean of the full batch, not its sum.
                loss = loss / t.grad_accum
            self.scaler.scale(loss).backward()
            total_loss += float(loss.detach()) * t.grad_accum

        if t.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = self._clip(self.model.parameters(), t.grad_clip)
        else:
            grad_norm = 0.0

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.state.step += 1
        self.state.tokens_seen += self.cfg.tokens_per_step

        return {
            "loss": total_loss / t.grad_accum,
            "lr": lr,
            "grad_norm": grad_norm,
        }

    # ------------------------------------------------------------------ the loop

    def train(
        self,
        max_steps: int | None = None,
        resume: str | Path | None = None,
        console: Any = None,
    ) -> TrainState:
        """The full loop. Returns the final state."""
        from rich.console import Console

        console = console or Console()
        t = self.cfg.train
        max_steps = max_steps or self.cfg.max_steps

        resuming = False
        if resume is not None and Path(resume).exists():
            payload = load_checkpoint(
                resume, self.model, self.optimizer, self.scaler, map_location=self.device.device
            )
            self.state.step = payload["step"]
            self.state.tokens_seen = payload["tokens_seen"]
            self.state.best_val_loss = payload["best_val_loss"]
            resuming = True
            console.print(
                f"[green]Resuming from step {self.state.step:,} "
                f"({self.state.tokens_seen:,} tokens seen).[/green]"
            )

        set_seed(t.seed + self.state.step)
        self.model.train()

        flops_per_token = self._flops(self.cfg.model)["total"]
        logger = TrainingLogger(self.run_dir, resuming=resuming)

        # The step-0 loss against ln(V): the bug detector from module 05.
        if self.state.step == 0:
            initial = self.evaluate(iters=5)
            floor = math.log(self.cfg.model.vocab_size)
            drift = initial.get("val", float("nan")) - floor
            color = "green" if abs(drift) < 0.2 else "red"
            console.print(
                f"initial loss: [{color}]{initial.get('val', float('nan')):.4f}[/{color}]  "
                f"(ln({self.cfg.model.vocab_size}) = {floor:.4f}, drift {drift:+.4f})"
            )
            if abs(drift) >= 0.2:
                console.print(
                    "[yellow]Out of range. Higher: the init is too aggressive. "
                    "Lower: information leak, look at the causal mask.[/yellow]"
                )

        started = time.perf_counter()
        last_log = started
        tokens_since_log = 0

        try:
            while self.state.step < max_steps:
                metrics = self.step()
                tokens_since_log += self.cfg.tokens_per_step

                if self.state.step % t.log_interval == 0:
                    self.device.synchronize()
                    now = time.perf_counter()
                    elapsed = now - last_log
                    tps = tokens_since_log / elapsed if elapsed > 0 else 0.0
                    mfu = self._mfu(tps, flops_per_token, self._peak_tflops())

                    remaining = (max_steps - self.state.step) * (
                        (now - started) / max(1, self.state.step)
                    )
                    format_eta = resolve("13_final_training", "format_eta")

                    console.print(
                        f"step {self.state.step:>6,}/{max_steps:,}  "
                        f"loss {metrics['loss']:.4f}  "
                        f"lr {metrics['lr']:.2e}  "
                        f"|g| {metrics['grad_norm']:.2f}  "
                        f"{tps / 1e3:.1f}k tok/s  "
                        f"MFU {mfu:.1%}  "
                        f"{format_eta(remaining)} left"
                    )
                    logger.log(
                        step=self.state.step,
                        tokens=self.state.tokens_seen,
                        train_loss=metrics["loss"],
                        lr=metrics["lr"],
                        grad_norm=metrics["grad_norm"],
                        tokens_per_second=tps,
                        mfu=mfu,
                    )
                    last_log, tokens_since_log = now, 0

                if self.state.step % t.eval_interval == 0:
                    losses = self.evaluate()
                    val = losses.get("val", float("nan"))
                    is_best = val < self.state.best_val_loss
                    console.print(
                        f"  [bold]eval[/bold] train {losses.get('train', float('nan')):.4f}  "
                        f"val {val:.4f}" + ("  [green]<- best[/green]" if is_best else "")
                    )
                    logger.log(
                        step=self.state.step,
                        tokens=self.state.tokens_seen,
                        train_loss=losses.get("train"),
                        val_loss=val,
                        lr=metrics["lr"],
                    )
                    if is_best:
                        self.state.best_val_loss = val
                        save_checkpoint(
                            self.run_dir / "best.pt",
                            self.model,
                            self.optimizer,
                            self.scaler,
                            self.state.step,
                            self.state.tokens_seen,
                            self.state.best_val_loss,
                            self.cfg,
                        )
                    save_checkpoint(
                        self.run_dir / "last.pt",
                        self.model,
                        self.optimizer,
                        self.scaler,
                        self.state.step,
                        self.state.tokens_seen,
                        self.state.best_val_loss,
                        self.cfg,
                    )

                if self.on_sample is not None and self.state.step % t.sample_interval == 0:
                    text = self.on_sample(self.state.step)
                    logger.log_sample(self.state.step, "", text, self.state.best_val_loss)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Saving checkpoint...[/yellow]")
            save_checkpoint(
                self.run_dir / "last.pt",
                self.model,
                self.optimizer,
                self.scaler,
                self.state.step,
                self.state.tokens_seen,
                self.state.best_val_loss,
                self.cfg,
            )
            console.print(
                f"[green]Saved to {self.run_dir / 'last.pt'}. "
                f"Resume with --resume.[/green]"
            )

        finally:
            plot = logger.plot()
            logger.close()
            if plot is not None:
                console.print(f"[green]curves saved to {plot}[/green]")

        return self.state

    def _peak_tflops(self) -> float:
        """Theoretical hardware peak, for the MFU. Conservative estimates."""
        if self.device.kind == "cuda":
            name = self.device.name.lower()
            if "2060" in name:
                return 51.6  # fp16 with tensor cores
            if "3090" in name:
                return 71.0
            if "4090" in name:
                return 165.0
            if "a100" in name:
                return 312.0
            return 50.0
        if self.device.kind == "mps":
            return 14.0  # approximate for M-series
        return 1.0
