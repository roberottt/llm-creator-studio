"""`device.py` is the only place in the repo that knows what hardware we are on.

These tests run on cuda, mps and cpu unchanged: they check invariants of the precision
policy, not specific values from one machine.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import pytest
import torch

from llmfs.device import DeviceConfig, get_device, set_seed


def test_the_mps_fallback_is_enabled_when_llmfs_is_imported():
    """It has to be set BEFORE torch is imported, or it is useless."""
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_autodetection_returns_something_coherent():
    cfg = get_device()
    assert cfg.kind in {"cuda", "mps", "cpu"}
    assert cfg.device.type == cfg.kind
    assert isinstance(cfg.summary(), str) and cfg.summary()


def test_cpu_is_always_available_and_in_fp32():
    cfg = get_device("cpu")
    assert cfg.kind == "cpu"
    assert cfg.amp_dtype is None
    assert cfg.use_grad_scaler is False
    assert cfg.autocast_dtype_name == "fp32"


def test_in_fp32_autocast_is_a_nullcontext():
    """That way the training loop does not need to branch by device."""
    cfg = get_device("cpu")
    assert isinstance(cfg.autocast(), nullcontext)


def test_only_fp16_needs_a_grad_scaler():
    """bf16 has the same exponent range as fp32: no scaling needed."""
    cfg = get_device()
    if cfg.amp_dtype == torch.float16 and cfg.kind == "cuda":
        assert cfg.use_grad_scaler is True
    if cfg.amp_dtype in (None, torch.bfloat16):
        assert cfg.use_grad_scaler is False


def test_bf16_only_on_ampere_or_newer():
    """Turing (sm_75) does NOT have bf16 in hardware, whatever is_bf16_supported() says."""
    cfg = get_device()
    if cfg.compute_capability is not None:
        assert cfg.supports_bf16 == (cfg.compute_capability >= (8, 0))
        assert cfg.supports_flash_sdpa == (cfg.compute_capability >= (8, 0))


def test_torch_compile_is_never_recommended_on_turing_or_mps():
    cfg = get_device()
    if cfg.kind == "mps" or (cfg.compute_capability is not None and cfg.compute_capability < (8, 0)):
        assert cfg.compile_recommended is False


def test_the_grad_scaler_is_transparent_when_disabled():
    cfg = get_device("cpu")
    scaler = cfg.grad_scaler()
    loss = torch.tensor(2.0, requires_grad=True)
    assert torch.equal(scaler.scale(loss), loss)


def test_one_complete_training_step_on_the_real_device():
    """The exact pattern from module 11: autocast + scaler + step."""
    cfg = get_device()
    model = torch.nn.Linear(16, 16).to(cfg.device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = cfg.grad_scaler()

    x = torch.randn(4, 16, device=cfg.device)
    with cfg.autocast():
        loss = model(x).square().mean()

    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)

    assert torch.isfinite(loss.detach()).item()


def test_amp_can_be_forced_off():
    cfg = get_device(amp=False)
    assert cfg.amp_dtype is None
    assert isinstance(cfg.autocast(), nullcontext)


def test_an_unknown_device_fails_with_a_useful_message():
    with pytest.raises(ValueError, match="cuda, mps or cpu"):
        get_device("tpu")


def test_the_llmfs_device_environment_variable(monkeypatch):
    monkeypatch.setenv("LLMFS_DEVICE", "cpu")
    assert get_device().kind == "cpu"


def test_set_seed_makes_sampling_reproducible():
    set_seed(1234)
    a = torch.randn(8)
    set_seed(1234)
    assert torch.equal(a, torch.randn(8))


def test_device_config_is_immutable():
    cfg = get_device("cpu")
    with pytest.raises(Exception):
        cfg.kind = "cuda"  # type: ignore[misc]
    assert isinstance(cfg, DeviceConfig)


def test_synchronize_and_empty_cache_do_not_blow_up_on_any_backend():
    cfg = get_device()
    cfg.synchronize()
    cfg.empty_cache()
    assert isinstance(cfg.memory_summary(), str)
