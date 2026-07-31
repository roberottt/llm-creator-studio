"""El bucle de entrenamiento completo.

Todo lo de los modulos 11-13 junto y funcionando: acumulacion de gradientes, precision
mixta, recorte, planificador, evaluacion periodica, muestras de texto y checkpoints
reanudables.

Las piezas se piden al bridge, asi que cuando tus ejercicios estan bien, esto entrena con
TU AdamW y TU planificador.
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
from llmfs.train.checkpoint import cargar_checkpoint, guardar_checkpoint
from llmfs.train.logger import EntrenamientoLogger


@dataclass
class TrainState:
    """Todo lo que cambia durante el entrenamiento y hay que guardar en el checkpoint."""

    step: int = 0
    tokens_vistos: int = 0
    best_val_loss: float = float("inf")
    historial: list[dict[str, Any]] = field(default_factory=list)


class Entrenador:
    """Orquesta el entrenamiento de un modelo.

    Uso:
        entrenador = Entrenador(cfg, modelo, get_batch_fn)
        entrenador.entrenar()

    `get_batch_fn(split, batch_size)` devuelve `(x, y)` ya en el dispositivo. Se pasa como
    funcion para que el entrenador no sepa nada de como estan guardados los datos.
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

        # Las piezas del alumno, o la referencia si todavia no las ha hecho.
        self._lr_at_step = resolve("11_bucle_de_entrenamiento", "lr_at_step")
        self._clip = resolve("11_bucle_de_entrenamiento", "clip_grad_norm")
        self._param_groups = resolve("11_bucle_de_entrenamiento", "build_param_groups")
        self._flops = resolve("12_eficiencia_y_escalado", "model_flops_per_token")
        self._mfu = resolve("12_eficiencia_y_escalado", "compute_mfu")

        self.optimizer = self._construir_optimizador()
        self.scaler = self.device.grad_scaler()
        self.model = maybe_compile(self.model, cfg.train.compile, self.device)

    # ------------------------------------------------------------------ montaje

    def _construir_optimizador(self) -> torch.optim.Optimizer:
        grupos = self._param_groups(self.model, self.cfg.train.weight_decay)
        if self.cfg.train.optimizer == "adamw_scratch":
            AdamWScratch = resolve("11_bucle_de_entrenamiento", "AdamWScratch")
            return AdamWScratch(
                grupos, lr=self.cfg.train.lr, betas=tuple(self.cfg.train.betas)
            )
        # `fused=True` acelera bastante en CUDA; en MPS/CPU no existe.
        extra: dict[str, Any] = {}
        if self.device.kind == "cuda":
            extra["fused"] = True
        return torch.optim.AdamW(
            grupos, lr=self.cfg.train.lr, betas=tuple(self.cfg.train.betas), **extra
        )

    # ------------------------------------------------------------------ evaluacion

    @torch.no_grad()
    def evaluar(self, iters: int | None = None) -> dict[str, float]:
        """Perdida media en entrenamiento y en validacion.

        `model.eval()` y `torch.no_grad()` son imprescindibles: sin el primero el dropout
        seguiria activo y la perdida saldria peor de lo real; sin el segundo se
        construiria el grafo de autograd para nada y se gastaria memoria.
        """
        iters = iters or self.cfg.train.eval_iters
        self.model.eval()
        salida: dict[str, float] = {}

        for split in ("train", "val"):
            total = 0.0
            n = 0
            for _ in range(iters):
                try:
                    x, y = self.get_batch(split, self.cfg.train.batch_size)
                except (KeyError, ValueError):
                    break
                with self.device.autocast():
                    _, perdida = self.model(x, y)
                total += float(perdida.detach())
                n += 1
            if n:
                salida[split] = total / n

        self.model.train()
        return salida

    # ------------------------------------------------------------------ un paso

    def paso(self) -> dict[str, float]:
        """Un paso completo de optimizacion, con acumulacion de gradientes.

        El orden importa y es este:

            1. lr del planificador -> a los grupos del optimizador
            2. `grad_accum` micro-batches: forward + backward, acumulando gradientes
            3. desescalar los gradientes (si hay scaler) ANTES de recortar
            4. recortar por norma global
            5. paso del optimizador
            6. limpiar los gradientes

        El paso 3 es facil de olvidar y silencioso: con AMP, los gradientes estan
        multiplicados por el factor del scaler (unos 65.000), asi que su norma tambien.
        Recortar sin desescalar recortaria a un umbral 65.000 veces mas pequenyo del que
        crees, y el entrenamiento se arrastraria sin que nada lo indique.
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
        for grupo in self.optimizer.param_groups:
            grupo["lr"] = lr

        perdida_total = 0.0
        for _ in range(t.grad_accum):
            x, y = self.get_batch("train", t.batch_size)
            with self.device.autocast():
                _, perdida = self.model(x, y)
                # Dividir entre grad_accum: la suma de los micro-batches tiene que dar la
                # media del batch completo, no su suma.
                perdida = perdida / t.grad_accum
            self.scaler.scale(perdida).backward()
            perdida_total += float(perdida.detach()) * t.grad_accum

        if t.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = self._clip(self.model.parameters(), t.grad_clip)
        else:
            grad_norm = 0.0

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.state.step += 1
        self.state.tokens_vistos += self.cfg.tokens_per_step

        return {
            "loss": perdida_total / t.grad_accum,
            "lr": lr,
            "grad_norm": grad_norm,
        }

    # ------------------------------------------------------------------ el bucle

    def entrenar(
        self,
        max_steps: int | None = None,
        reanudar: str | Path | None = None,
        console: Any = None,
    ) -> TrainState:
        """El bucle completo. Devuelve el estado final."""
        from rich.console import Console

        console = console or Console()
        t = self.cfg.train
        max_steps = max_steps or self.cfg.max_steps

        reanudando = False
        if reanudar is not None and Path(reanudar).exists():
            payload = cargar_checkpoint(
                reanudar, self.model, self.optimizer, self.scaler, map_location=self.device.device
            )
            self.state.step = payload["step"]
            self.state.tokens_vistos = payload["tokens_vistos"]
            self.state.best_val_loss = payload["best_val_loss"]
            reanudando = True
            console.print(
                f"[green]Reanudando desde el paso {self.state.step:,} "
                f"({self.state.tokens_vistos:,} tokens vistos).[/green]"
            )

        set_seed(t.seed + self.state.step)
        self.model.train()

        flops_por_token = self._flops(self.cfg.model)["total"]
        logger = EntrenamientoLogger(self.run_dir, reanudando=reanudando)

        # La perdida del paso 0 frente a ln(V): el detector de bugs del modulo 05.
        if self.state.step == 0:
            inicial = self.evaluar(iters=5)
            suelo = math.log(self.cfg.model.vocab_size)
            desvio = inicial.get("val", float("nan")) - suelo
            color = "green" if abs(desvio) < 0.2 else "red"
            console.print(
                f"perdida inicial: [{color}]{inicial.get('val', float('nan')):.4f}[/{color}]  "
                f"(ln({self.cfg.model.vocab_size}) = {suelo:.4f}, desvio {desvio:+.4f})"
            )
            if abs(desvio) >= 0.2:
                console.print(
                    "[yellow]Fuera de rango. Mas alta: init demasiado agresiva. "
                    "Mas baja: fuga de informacion, mira la mascara causal.[/yellow]"
                )

        inicio = time.perf_counter()
        ultimo_log = inicio
        tokens_desde_log = 0

        try:
            while self.state.step < max_steps:
                metricas = self.paso()
                tokens_desde_log += self.cfg.tokens_per_step

                if self.state.step % t.log_interval == 0:
                    self.device.synchronize()
                    ahora = time.perf_counter()
                    transcurrido = ahora - ultimo_log
                    tps = tokens_desde_log / transcurrido if transcurrido > 0 else 0.0
                    mfu = self._mfu(tps, flops_por_token, self._pico_tflops())

                    restante = (max_steps - self.state.step) * (
                        (ahora - inicio) / max(1, self.state.step)
                    )
                    format_eta = resolve("13_entrenamiento_final", "format_eta")

                    console.print(
                        f"paso {self.state.step:>6,}/{max_steps:,}  "
                        f"perdida {metricas['loss']:.4f}  "
                        f"lr {metricas['lr']:.2e}  "
                        f"|g| {metricas['grad_norm']:.2f}  "
                        f"{tps / 1e3:.1f}k tok/s  "
                        f"MFU {mfu:.1%}  "
                        f"faltan {format_eta(restante)}"
                    )
                    logger.log(
                        step=self.state.step,
                        tokens=self.state.tokens_vistos,
                        train_loss=metricas["loss"],
                        lr=metricas["lr"],
                        grad_norm=metricas["grad_norm"],
                        tokens_por_segundo=tps,
                        mfu=mfu,
                    )
                    ultimo_log, tokens_desde_log = ahora, 0

                if self.state.step % t.eval_interval == 0:
                    perdidas = self.evaluar()
                    val = perdidas.get("val", float("nan"))
                    mejor = val < self.state.best_val_loss
                    console.print(
                        f"  [bold]evaluacion[/bold] train {perdidas.get('train', float('nan')):.4f}  "
                        f"val {val:.4f}" + ("  [green]<- mejor[/green]" if mejor else "")
                    )
                    logger.log(
                        step=self.state.step,
                        tokens=self.state.tokens_vistos,
                        train_loss=perdidas.get("train"),
                        val_loss=val,
                        lr=metricas["lr"],
                    )
                    if mejor:
                        self.state.best_val_loss = val
                        guardar_checkpoint(
                            self.run_dir / "best.pt",
                            self.model,
                            self.optimizer,
                            self.scaler,
                            self.state.step,
                            self.state.tokens_vistos,
                            self.state.best_val_loss,
                            self.cfg,
                        )
                    guardar_checkpoint(
                        self.run_dir / "last.pt",
                        self.model,
                        self.optimizer,
                        self.scaler,
                        self.state.step,
                        self.state.tokens_vistos,
                        self.state.best_val_loss,
                        self.cfg,
                    )

                if self.on_sample is not None and self.state.step % t.sample_interval == 0:
                    texto = self.on_sample(self.state.step)
                    logger.log_muestra(self.state.step, "", texto, self.state.best_val_loss)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrumpido. Guardando checkpoint...[/yellow]")
            guardar_checkpoint(
                self.run_dir / "last.pt",
                self.model,
                self.optimizer,
                self.scaler,
                self.state.step,
                self.state.tokens_vistos,
                self.state.best_val_loss,
                self.cfg,
            )
            console.print(
                f"[green]Guardado en {self.run_dir / 'last.pt'}. "
                f"Reanuda con --resume.[/green]"
            )

        finally:
            grafica = logger.grafica()
            logger.cerrar()
            if grafica is not None:
                console.print(f"[green]curvas guardadas en {grafica}[/green]")

        return self.state

    def _pico_tflops(self) -> float:
        """Pico teorico del hardware, para la MFU. Estimaciones conservadoras."""
        if self.device.kind == "cuda":
            nombre = self.device.name.lower()
            if "2060" in nombre:
                return 51.6  # fp16 con tensor cores
            if "3090" in nombre:
                return 71.0
            if "4090" in nombre:
                return 165.0
            if "a100" in nombre:
                return 312.0
            return 50.0
        if self.device.kind == "mps":
            return 14.0  # aproximado para M-series
        return 1.0
