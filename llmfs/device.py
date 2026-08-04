"""Device detection and mixed-precision policy.

All the code in the repo asks for its device here and nowhere else. That way the same
script runs on the RTX 2060 (CUDA, Turing), on the MacBook (MPS) and on CPU untouched.

Why this module exists instead of a plain `torch.device("cuda" if ...)`:

- **Turing (sm_75) has no bfloat16 in hardware.** Autocast has to use float16, and float16
  has a small exponent range: gradients go to zero (underflow) unless they are scaled.
  Hence fp16 requires a `GradScaler` and bf16 does not.
- **`torch.cuda.is_bf16_supported()` LIES on Turing.** Since torch 2.4 it returns `True`,
  counting software emulation, which is correct but brutally slow. That is why we look at
  the compute capability directly here: real bf16 is sm_80 (Ampere) and up.
- **FlashAttention-2 does not exist on sm_75 either.** `F.scaled_dot_product_attention`
  knows this and falls back to the `memory_efficient` backend, which does work on Turing
  and already avoids materializing the TxT attention matrix. Nothing special needs doing,
  but it is worth knowing: on the 2060 attention will not run at the speed of the
  benchmarks you read elsewhere.
- **`torch.compile` on Turing is a lottery.** Inductor generates Triton kernels that on
  sm_75 sometimes fail to compile and sometimes come out slower than eager. It stays as an
  optional flag, off by default.
- **MPS needs `PYTORCH_ENABLE_MPS_FALLBACK=1`** (set by `llmfs/__init__.py` before torch is
  imported). With that, ops Apple has not implemented in Metal yet run on the CPU instead
  of blowing up. The price is a silent GPU->CPU->GPU copy: if a training run is
  inexplicably slow on a Mac, that is the first suspect.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, ContextManager, Literal

import torch

DeviceKind = Literal["cuda", "mps", "cpu"]

# First compute capability with bfloat16 and third-generation tensor cores (Ampere).
_BF16_MIN_CAPABILITY = (8, 0)
# FlashAttention (SDPA's `flash` backend) also requires Ampere or newer.
_FLASH_MIN_CAPABILITY = (8, 0)


@dataclass(frozen=True)
class DeviceConfig:
    """Everything the rest of the repo needs to know about where it is running."""

    device: torch.device
    kind: DeviceKind
    name: str
    #: autocast dtype. `None` means "train in fp32, no autocast".
    amp_dtype: torch.dtype | None
    #: fp16 needs gradient scaling; bf16 and fp32 do not.
    use_grad_scaler: bool
    supports_bf16: bool
    supports_flash_sdpa: bool
    compile_recommended: bool
    compute_capability: tuple[int, int] | None = None
    total_memory_gb: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------ properties

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    @property
    def amp_enabled(self) -> bool:
        return self.amp_dtype is not None

    @property
    def autocast_dtype_name(self) -> str:
        return "fp32" if self.amp_dtype is None else str(self.amp_dtype).replace("torch.", "")

    # ------------------------------------------------------------------ contexts

    def autocast(self, enabled: bool = True) -> ContextManager[Any]:
        """The mixed-precision context appropriate for this device.

        Usage:
            with dev.autocast():
                logits, loss = model(x, y)

        In fp32 it returns a `nullcontext`, so the same code works on all three platforms
        without branching.
        """
        if not enabled or self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.kind, dtype=self.amp_dtype)

    def grad_scaler(self) -> torch.amp.GradScaler:
        """A pre-configured `GradScaler` (disabled when it is not needed).

        A scaler with `enabled=False` is a transparent no-op: `scale(loss)` returns `loss`
        and `step(opt)` calls `opt.step()`. That is why the training loop needs no `if`
        around mixed precision.
        """
        try:
            return torch.amp.GradScaler(self.kind, enabled=self.use_grad_scaler)
        except (RuntimeError, ValueError):
            # Some torch versions do not accept a GradScaler on MPS/CPU.
            return torch.amp.GradScaler("cpu", enabled=False)

    # ------------------------------------------------------------------ helpers

    def memory_summary(self) -> str:
        """Memory in use, in GB. Empty string if the backend does not expose it."""
        if self.kind == "cuda":
            used = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            return f"{used:.2f} GB in use / {peak:.2f} GB peak"
        if self.kind == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            return f"{torch.mps.current_allocated_memory() / 1e9:.2f} GB in use"
        return ""

    def synchronize(self) -> None:
        """Wait for the GPU to finish. Essential before timing anything."""
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
            f"device        : {self.kind} ({self.name})",
            f"precision     : {self.autocast_dtype_name}"
            + (" + GradScaler" if self.use_grad_scaler else ""),
        ]
        if self.compute_capability is not None:
            lines.append(
                f"capability    : sm_{self.compute_capability[0]}{self.compute_capability[1]}"
            )
        if self.total_memory_gb is not None:
            lines.append(f"memory        : {self.total_memory_gb:.1f} GB")
        lines.append(f"flash sdpa    : {'yes' if self.supports_flash_sdpa else 'no (mem_efficient)'}")
        lines.append(
            f"torch.compile : {'recommended' if self.compile_recommended else 'not recommended'}"
        )
        for note in self.notes:
            lines.append(f"  - {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------- detection


def _cuda_config(index: int, amp: bool | None) -> DeviceConfig:
    props = torch.cuda.get_device_properties(index)
    capability = (props.major, props.minor)

    # NOTE: we do not use torch.cuda.is_bf16_supported(); see the module header.
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
            "Pre-Ampere GPU: no bfloat16 in hardware, so float16 + GradScaler is used. "
            "If you see 'inf/nan' in the loss, the scaler is doing its job; "
            "only worry if it does not recover within a few steps."
        )
    if not supports_flash:
        notes.append(
            "No FlashAttention-2 (needs sm_80+). SDPA will use the memory_efficient "
            "backend, which already avoids materializing the T x T matrix."
        )
    if capability >= (8, 0):
        # TF32 only exists on Ampere+. On Turing the call does nothing useful.
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
    # On MPS the default is fp32: Metal's fp16 autocast is correct but leaves less numerical
    # headroom, and the gain on M-series is modest because memory is unified (there is no
    # PCIe traffic to amortize). Turn it on explicitly with amp=True or LLMFS_AMP=1.
    use_amp = False if amp is None else amp
    amp_dtype = torch.float16 if use_amp else None

    total_gb = None
    if hasattr(torch.mps, "recommended_max_memory"):
        total_gb = torch.mps.recommended_max_memory() / 1e9

    notes = [
        "PYTORCH_ENABLE_MPS_FALLBACK=1 is on: ops without a Metal kernel fall back to CPU. "
        "It is the most common cause of unexplained slowness on a Mac.",
    ]
    if use_amp:
        notes.append(
            "float16 on MPS is optional and less battle-tested than on CUDA. "
            "If something goes nan, turn it off."
        )

    return DeviceConfig(
        device=torch.device("mps"),
        kind="mps",
        name="Apple Silicon (Metal)",
        amp_dtype=amp_dtype,
        use_grad_scaler=False,  # the MPS GradScaler is not battle-tested; fp16 runs unscaled here
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
        notes=(
            "No GPU: use the toy configs (tiny_char). The 9M model on CPU is not viable.",
        ),
    )


def get_device(prefer: str | None = None, amp: bool | None = None) -> DeviceConfig:
    """Return the device configuration for this machine.

    Args:
        prefer: force a backend ("cuda", "mps", "cpu"). If `None`, the `LLMFS_DEVICE`
            environment variable is read and, failing that, it is autodetected in the order
            cuda > mps > cpu.
        amp: force mixed precision on/off. `None` = the backend's default policy
            (CUDA: on; MPS: off; CPU: off). It can also be set with `LLMFS_AMP=0|1`.

    Returns:
        A ready-to-use DeviceConfig: `cfg.device`, `cfg.autocast()`, `cfg.grad_scaler()`.
    """
    prefer = prefer or os.environ.get("LLMFS_DEVICE") or None
    if amp is None and "LLMFS_AMP" in os.environ:
        amp = os.environ["LLMFS_AMP"].strip().lower() in {"1", "true", "yes"}

    if prefer is not None:
        prefer = prefer.strip().lower()
        if prefer.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("You asked for CUDA but torch.cuda.is_available() is False.")
            index = int(prefer.split(":")[1]) if ":" in prefer else 0
            return _cuda_config(index, amp)
        if prefer == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "You asked for MPS but torch.backends.mps.is_available() is False."
                )
            return _mps_config(amp)
        if prefer == "cpu":
            return _cpu_config()
        raise ValueError(f"Unknown device: {prefer!r}. Use cuda, mps or cpu.")

    if torch.cuda.is_available():
        return _cuda_config(0, amp)
    if torch.backends.mps.is_available():
        return _mps_config(amp)
    return _cpu_config()


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set the seeds for python, numpy and torch (all backends).

    `deterministic=True` disables cuDNN's non-deterministic algorithms. It costs speed; it
    only makes sense when you are hunting a reproducibility bug.
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
    """Apply `torch.compile` only if explicitly asked for and the backend can take it.

    It is never on by default: on Turing (sm_75) Inductor fails to compile fairly often, and
    when it does compile it does not always win. If `enabled=True` on a backend that is not
    recommended, it is attempted anyway but warned about, falling back to the uncompiled
    model if it blows up.
    """
    if not enabled:
        return model
    if not cfg.compile_recommended:
        print(
            f"[llmfs] WARNING: torch.compile is not recommended on {cfg.kind} "
            f"({cfg.name}). Trying anyway."
        )
    try:
        return torch.compile(model)
    except Exception as exc:  # noqa: BLE001 - we want to degrade, not break the training run
        print(
            f"[llmfs] torch.compile failed ({exc.__class__.__name__}: {exc}). "
            "Continuing without compiling."
        )
        return model
