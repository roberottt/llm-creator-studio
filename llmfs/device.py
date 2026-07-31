"""Deteccion de dispositivo y politica de precision mixta.

Todo el codigo del repo pide su dispositivo aqui y en ningun otro sitio. Asi el mismo
script corre en la RTX 2060 (CUDA, Turing), en el MacBook (MPS) y en CPU sin tocar nada.

Por que este modulo existe y no es un simple `torch.device("cuda" if ...)`:

- **Turing (sm_75) no tiene bfloat16 en hardware.** El autocast debe usar float16, y float16
  tiene un rango exponencial pequenyo: los gradientes se van a cero (underflow) si no se
  escalan. De ahi que fp16 exija `GradScaler` y bf16 no.
- **`torch.cuda.is_bf16_supported()` MIENTE en Turing.** Desde torch 2.4 devuelve `True`
  contando emulacion por software, que es correcta pero brutalmente lenta. Por eso aqui
  miramos la compute capability a pelo: bf16 real es sm_80 (Ampere) en adelante.
- **FlashAttention-2 tampoco existe en sm_75.** `F.scaled_dot_product_attention` lo sabe y
  cae al backend `memory_efficient`, que si funciona en Turing y ya evita materializar la
  matriz TxT de atencion. No hace falta hacer nada especial, pero conviene saberlo: en la
  2060 la atencion no va a ir a la velocidad de los benchmarks que leas por ahi.
- **`torch.compile` en Turing es una loteria.** Inductor genera kernels Triton que en sm_75
  a veces fallan a compilar y a veces salen mas lentos que el eager. Queda como flag
  opcional, desactivado por defecto.
- **MPS necesita `PYTORCH_ENABLE_MPS_FALLBACK=1`** (lo fija `llmfs/__init__.py` antes de
  importar torch). Con eso, los ops que Apple todavia no ha implementado en Metal se
  ejecutan en CPU en lugar de reventar. El precio es una copia GPU->CPU->GPU silenciosa:
  si un entrenamiento va inexplicablemente lento en Mac, esta es la primera sospechosa.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, ContextManager, Literal

import torch

DeviceKind = Literal["cuda", "mps", "cpu"]

# Primera compute capability con bfloat16 y tensor cores de tercera generacion (Ampere).
_BF16_MIN_CAPABILITY = (8, 0)
# FlashAttention (el backend `flash` de SDPA) tambien pide Ampere o superior.
_FLASH_MIN_CAPABILITY = (8, 0)


@dataclass(frozen=True)
class DeviceConfig:
    """Todo lo que el resto del repo necesita saber sobre donde esta corriendo."""

    device: torch.device
    kind: DeviceKind
    name: str
    #: dtype del autocast. `None` significa "entrena en fp32, sin autocast".
    amp_dtype: torch.dtype | None
    #: fp16 necesita escalado de gradiente; bf16 y fp32 no.
    use_grad_scaler: bool
    supports_bf16: bool
    supports_flash_sdpa: bool
    compile_recommended: bool
    compute_capability: tuple[int, int] | None = None
    total_memory_gb: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------ propiedades

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    @property
    def amp_enabled(self) -> bool:
        return self.amp_dtype is not None

    @property
    def autocast_dtype_name(self) -> str:
        return "fp32" if self.amp_dtype is None else str(self.amp_dtype).replace("torch.", "")

    # ------------------------------------------------------------------ contextos

    def autocast(self, enabled: bool = True) -> ContextManager[Any]:
        """Contexto de precision mixta adecuado a este dispositivo.

        Uso:
            with dev.autocast():
                logits, loss = model(x, y)

        En fp32 devuelve un `nullcontext`, asi que el mismo codigo vale para las tres
        plataformas sin ramificar.
        """
        if not enabled or self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.kind, dtype=self.amp_dtype)

    def grad_scaler(self) -> torch.amp.GradScaler:
        """`GradScaler` ya configurado (deshabilitado si no hace falta).

        Un scaler con `enabled=False` es un no-op transparente: `scale(loss)` devuelve
        `loss` y `step(opt)` llama a `opt.step()`. Por eso el bucle de entrenamiento no
        necesita ningun `if` alrededor de la precision mixta.
        """
        try:
            return torch.amp.GradScaler(self.kind, enabled=self.use_grad_scaler)
        except (RuntimeError, ValueError):
            # Algunas versiones de torch no aceptan GradScaler en MPS/CPU.
            return torch.amp.GradScaler("cpu", enabled=False)

    # ------------------------------------------------------------------ utilidades

    def memory_summary(self) -> str:
        """Memoria en uso, en GB. Cadena vacia si el backend no la expone."""
        if self.kind == "cuda":
            used = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            return f"{used:.2f} GB en uso / {peak:.2f} GB pico"
        if self.kind == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            return f"{torch.mps.current_allocated_memory() / 1e9:.2f} GB en uso"
        return ""

    def synchronize(self) -> None:
        """Espera a que la GPU termine. Imprescindible antes de medir tiempos."""
        if self.kind == "cuda":
            torch.cuda.synchronize()
        elif self.kind == "mps":
            torch.mps.synchronize()

    def empty_cache(self) -> None:
        if self.kind == "cuda":
            torch.cuda.empty_cache()
        elif self.kind == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def summary(self) -> str:
        lines = [
            f"dispositivo   : {self.kind} ({self.name})",
            f"precision     : {self.autocast_dtype_name}"
            + (" + GradScaler" if self.use_grad_scaler else ""),
        ]
        if self.compute_capability is not None:
            lines.append(
                f"capability    : sm_{self.compute_capability[0]}{self.compute_capability[1]}"
            )
        if self.total_memory_gb is not None:
            lines.append(f"memoria       : {self.total_memory_gb:.1f} GB")
        lines.append(f"flash sdpa    : {'si' if self.supports_flash_sdpa else 'no (mem_efficient)'}")
        lines.append(f"torch.compile : {'recomendado' if self.compile_recommended else 'no recomendado'}")
        for note in self.notes:
            lines.append(f"  - {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------- deteccion


def _cuda_config(index: int, amp: bool | None) -> DeviceConfig:
    props = torch.cuda.get_device_properties(index)
    capability = (props.major, props.minor)

    # OJO: no usamos torch.cuda.is_bf16_supported(); ver la nota de cabecera del modulo.
    supports_bf16 = capability >= _BF16_MIN_CAPABILITY
    supports_flash = capability >= _FLASH_MIN_CAPABILITY

    use_amp = True if amp is None else amp
    if use_amp:
        amp_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    else:
        amp_dtype = None

    notes: list[str] = []
    if not supports_bf16:
        notes.append(
            "GPU pre-Ampere: sin bfloat16 en hardware, se usa float16 + GradScaler. "
            "Si ves 'inf/nan' en la perdida, el scaler esta haciendo su trabajo; "
            "solo preocupate si no se recupera en unos pasos."
        )
    if not supports_flash:
        notes.append(
            "Sin FlashAttention-2 (pide sm_80+). SDPA usara el backend memory_efficient, "
            "que ya evita materializar la matriz T x T."
        )
    if capability >= (8, 0):
        # TF32 solo existe en Ampere+. En Turing la llamada no hace nada util.
        torch.set_float32_matmul_precision("high")

    return DeviceConfig(
        device=torch.device("cuda", index),
        kind="cuda",
        name=props.name,
        amp_dtype=amp_dtype,
        use_grad_scaler=(amp_dtype == torch.float16),
        supports_bf16=supports_bf16,
        supports_flash_sdpa=supports_flash,
        compile_recommended=capability >= (8, 0),
        compute_capability=capability,
        total_memory_gb=props.total_memory / 1e9,
        notes=tuple(notes),
    )


def _mps_config(amp: bool | None) -> DeviceConfig:
    # En MPS el default es fp32: el autocast fp16 de Metal es correcto pero da menos margen
    # numerico y la ganancia en M-series es modesta porque la memoria es unificada (no hay
    # trasiego PCIe que amortizar). Se activa explicitamente con amp=True o LLMFS_AMP=1.
    use_amp = False if amp is None else amp
    amp_dtype = torch.float16 if use_amp else None

    total_gb = None
    if hasattr(torch.mps, "recommended_max_memory"):
        total_gb = torch.mps.recommended_max_memory() / 1e9

    notes = [
        "PYTORCH_ENABLE_MPS_FALLBACK=1 activo: los ops sin kernel Metal caen a CPU. "
        "Es la causa mas comun de lentitud inexplicable en Mac.",
    ]
    if use_amp:
        notes.append("float16 en MPS es opcional y esta menos rodado que en CUDA. Si algo sale nan, quitalo.")

    return DeviceConfig(
        device=torch.device("mps"),
        kind="mps",
        name="Apple Silicon (Metal)",
        amp_dtype=amp_dtype,
        use_grad_scaler=False,  # el GradScaler de MPS no esta rodado; fp16 aqui va sin escalar
        supports_bf16=False,
        supports_flash_sdpa=False,
        compile_recommended=False,
        total_memory_gb=total_gb,
        notes=tuple(notes),
    )


def _cpu_config() -> DeviceConfig:
    return DeviceConfig(
        device=torch.device("cpu"),
        kind="cpu",
        name="CPU",
        amp_dtype=None,
        use_grad_scaler=False,
        supports_bf16=False,
        supports_flash_sdpa=False,
        compile_recommended=False,
        notes=("Sin GPU: usa los configs juguete (tiny_char). El modelo de 9M en CPU no es viable.",),
    )


def get_device(prefer: str | None = None, amp: bool | None = None) -> DeviceConfig:
    """Devuelve la configuracion de dispositivo para esta maquina.

    Args:
        prefer: fuerza un backend ("cuda", "mps", "cpu"). Si es `None` se lee la variable
            de entorno `LLMFS_DEVICE` y, si tampoco esta, se autodetecta con el orden
            cuda > mps > cpu.
        amp: fuerza precision mixta on/off. `None` = politica por defecto del backend
            (CUDA: activada; MPS: desactivada; CPU: desactivada). Tambien se puede fijar
            con `LLMFS_AMP=0|1`.

    Returns:
        DeviceConfig listo para usar: `cfg.device`, `cfg.autocast()`, `cfg.grad_scaler()`.
    """
    prefer = prefer or os.environ.get("LLMFS_DEVICE") or None
    if amp is None and "LLMFS_AMP" in os.environ:
        amp = os.environ["LLMFS_AMP"].strip().lower() in {"1", "true", "yes", "si"}

    if prefer is not None:
        prefer = prefer.strip().lower()
        if prefer.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("Has pedido CUDA pero torch.cuda.is_available() es False.")
            index = int(prefer.split(":")[1]) if ":" in prefer else 0
            return _cuda_config(index, amp)
        if prefer == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("Has pedido MPS pero torch.backends.mps.is_available() es False.")
            return _mps_config(amp)
        if prefer == "cpu":
            return _cpu_config()
        raise ValueError(f"Dispositivo desconocido: {prefer!r}. Usa cuda, mps o cpu.")

    if torch.cuda.is_available():
        return _cuda_config(0, amp)
    if torch.backends.mps.is_available():
        return _mps_config(amp)
    return _cpu_config()


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Fija las semillas de python, numpy y torch (todos los backends).

    `deterministic=True` desactiva los algoritmos no deterministas de cuDNN. Cuesta
    velocidad; solo tiene sentido cuando estas cazando un bug de reproducibilidad.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def maybe_compile(model: Any, enabled: bool, cfg: DeviceConfig) -> Any:
    """Aplica `torch.compile` solo si se pide explicitamente y el backend lo aguanta.

    Nunca se activa por defecto: en Turing (sm_75) Inductor falla a compilar con cierta
    frecuencia y cuando compila no siempre gana. Si `enabled=True` en un backend no
    recomendado, se intenta igual pero se avisa y se cae al modelo sin compilar si peta.
    """
    if not enabled:
        return model
    if not cfg.compile_recommended:
        print(
            f"[llmfs] AVISO: torch.compile no esta recomendado en {cfg.kind} "
            f"({cfg.name}). Se intenta igualmente."
        )
    try:
        return torch.compile(model)
    except Exception as exc:  # noqa: BLE001 - queremos degradar, no romper el entrenamiento
        print(f"[llmfs] torch.compile ha fallado ({exc.__class__.__name__}: {exc}). Sigo sin compilar.")
        return model
