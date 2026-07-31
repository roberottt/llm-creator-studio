"""Registro del entrenamiento: CSV, muestras de texto y graficas.

Durante una tirada de horas necesitas dos cosas: numeros que puedas graficar despues, y
muestras de texto para VER como aprende el modelo a escribir. Lo segundo es lo que hace
que valga la pena mirar el log.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class EntrenamientoLogger:
    """Escribe el CSV de metricas y el log de muestras de texto.

    El CSV se abre en modo append y se hace `flush()` en cada fila. Cuesta un poco mas,
    pero si el proceso muere no pierdes el historial, y puedes graficar la curva mientras
    entrena desde otra terminal.
    """

    COLUMNAS = [
        "step",
        "tokens",
        "train_loss",
        "val_loss",
        "lr",
        "grad_norm",
        "tokens_por_segundo",
        "mfu",
        "segundos",
    ]

    def __init__(self, run_dir: str | Path, reanudando: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.run_dir / "metrics.csv"
        self.samples_path = self.run_dir / "samples.md"
        self.inicio = time.perf_counter()

        nuevo = not self.csv_path.exists() or not reanudando
        self._csv = self.csv_path.open("a" if reanudando else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv, fieldnames=self.COLUMNAS)
        if nuevo:
            self._writer.writeheader()
            self._csv.flush()

        if not reanudando:
            self.samples_path.write_text(
                f"# Muestras del entrenamiento\n\n"
                f"Empezado el {datetime.now():%Y-%m-%d %H:%M}.\n\n"
                f"Lee esto de arriba abajo cuando termine: es el modelo aprendiendo a\n"
                f"escribir, paso a paso.\n",
                encoding="utf-8",
            )

    def log(self, **campos: Any) -> None:
        fila = {c: campos.get(c, "") for c in self.COLUMNAS}
        fila["segundos"] = round(time.perf_counter() - self.inicio, 2)
        self._writer.writerow(fila)
        self._csv.flush()

    def log_muestra(self, step: int, prompt: str, texto: str, val_loss: float | None = None) -> None:
        """Anyade una muestra de texto generado al fichero de muestras."""
        with self.samples_path.open("a", encoding="utf-8") as f:
            cabecera = f"\n## Paso {step:,}"
            if val_loss is not None:
                cabecera += f" — pérdida de validación {val_loss:.4f}"
            f.write(f"{cabecera}\n\n**Prompt:** `{prompt}`\n\n```\n{texto}\n```\n")

    def cerrar(self) -> None:
        if not self._csv.closed:
            self._csv.close()

    def __enter__(self) -> "EntrenamientoLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.cerrar()

    # ------------------------------------------------------------------ graficas

    def grafica(self, destino: str | Path | None = None) -> Path | None:
        """Dibuja las curvas de perdida y de learning rate a partir del CSV."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        filas = self.leer_csv()
        if not filas:
            return None

        pasos_tr = [f["step"] for f in filas if f.get("train_loss") is not None]
        perdidas_tr = [f["train_loss"] for f in filas if f.get("train_loss") is not None]
        pasos_val = [f["step"] for f in filas if f.get("val_loss") is not None]
        perdidas_val = [f["val_loss"] for f in filas if f.get("val_loss") is not None]

        fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))

        izq.plot(pasos_tr, perdidas_tr, lw=1, alpha=0.7, label="entrenamiento")
        if perdidas_val:
            izq.plot(pasos_val, perdidas_val, marker="o", ms=3, label="validacion")
        izq.set_xlabel("paso")
        izq.set_ylabel("perdida (nats)")
        izq.set_title("Curva de perdida")
        izq.grid(alpha=0.3)
        izq.legend()

        lrs = [(f["step"], f["lr"]) for f in filas if f.get("lr") is not None]
        if lrs:
            der.plot([s for s, _ in lrs], [v for _, v in lrs], color="tab:orange")
        der.set_xlabel("paso")
        der.set_ylabel("learning rate")
        der.set_title("Planificador (warmup + coseno)")
        der.grid(alpha=0.3)

        fig.tight_layout()
        destino = Path(destino) if destino else self.run_dir / "curvas.png"
        fig.savefig(destino, dpi=120)
        plt.close(fig)
        return destino

    def leer_csv(self) -> list[dict[str, Any]]:
        """Lee el CSV convirtiendo a float lo que se pueda."""
        if not self.csv_path.exists():
            return []
        filas: list[dict[str, Any]] = []
        with self.csv_path.open(encoding="utf-8") as f:
            for cruda in csv.DictReader(f):
                fila: dict[str, Any] = {}
                for k, v in cruda.items():
                    if v == "" or v is None:
                        fila[k] = None
                    else:
                        try:
                            fila[k] = float(v)
                        except ValueError:
                            fila[k] = v
                filas.append(fila)
        return filas
