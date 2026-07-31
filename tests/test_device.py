"""`device.py` es el unico sitio del repo que sabe en que hardware estamos.

Estos tests corren en cuda, mps y cpu sin cambios: comprueban invariantes de la politica
de precision, no valores concretos de una maquina.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import pytest
import torch

from llmfs.device import DeviceConfig, get_device, set_seed


def test_el_fallback_de_mps_se_activa_al_importar_llmfs():
    """Tiene que fijarse ANTES de que torch se importe, o no sirve de nada."""
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_autodeteccion_devuelve_algo_coherente():
    cfg = get_device()
    assert cfg.kind in {"cuda", "mps", "cpu"}
    assert cfg.device.type == cfg.kind
    assert isinstance(cfg.summary(), str) and cfg.summary()


def test_cpu_siempre_disponible_y_en_fp32():
    cfg = get_device("cpu")
    assert cfg.kind == "cpu"
    assert cfg.amp_dtype is None
    assert cfg.use_grad_scaler is False
    assert cfg.autocast_dtype_name == "fp32"


def test_en_fp32_el_autocast_es_un_nullcontext():
    """Asi el bucle de entrenamiento no necesita ramificar por dispositivo."""
    cfg = get_device("cpu")
    assert isinstance(cfg.autocast(), nullcontext)


def test_solo_fp16_necesita_grad_scaler():
    """bf16 tiene el mismo rango exponencial que fp32: no hace falta escalar."""
    cfg = get_device()
    if cfg.amp_dtype == torch.float16 and cfg.kind == "cuda":
        assert cfg.use_grad_scaler is True
    if cfg.amp_dtype in (None, torch.bfloat16):
        assert cfg.use_grad_scaler is False


def test_bf16_solo_en_ampere_o_superior():
    """Turing (sm_75) NO tiene bf16 en hardware, diga lo que diga is_bf16_supported()."""
    cfg = get_device()
    if cfg.compute_capability is not None:
        assert cfg.supports_bf16 == (cfg.compute_capability >= (8, 0))
        assert cfg.supports_flash_sdpa == (cfg.compute_capability >= (8, 0))


def test_torch_compile_nunca_se_recomienda_en_turing_ni_en_mps():
    cfg = get_device()
    if cfg.kind == "mps" or (cfg.compute_capability is not None and cfg.compute_capability < (8, 0)):
        assert cfg.compile_recommended is False


def test_el_grad_scaler_es_transparente_cuando_esta_desactivado():
    cfg = get_device("cpu")
    scaler = cfg.grad_scaler()
    perdida = torch.tensor(2.0, requires_grad=True)
    assert torch.equal(scaler.scale(perdida), perdida)


def test_un_paso_completo_de_entrenamiento_en_el_dispositivo_real():
    """El patron exacto del modulo 10: autocast + scaler + step."""
    cfg = get_device()
    modelo = torch.nn.Linear(16, 16).to(cfg.device)
    opt = torch.optim.SGD(modelo.parameters(), lr=0.01)
    scaler = cfg.grad_scaler()

    x = torch.randn(4, 16, device=cfg.device)
    with cfg.autocast():
        perdida = modelo(x).square().mean()

    scaler.scale(perdida).backward()
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)

    assert torch.isfinite(perdida.detach()).item()


def test_amp_puede_forzarse_a_off():
    cfg = get_device(amp=False)
    assert cfg.amp_dtype is None
    assert isinstance(cfg.autocast(), nullcontext)


def test_dispositivo_desconocido_falla_con_mensaje_util():
    with pytest.raises(ValueError, match="cuda, mps o cpu"):
        get_device("tpu")


def test_variable_de_entorno_llmfs_device(monkeypatch):
    monkeypatch.setenv("LLMFS_DEVICE", "cpu")
    assert get_device().kind == "cpu"


def test_set_seed_hace_reproducible_el_muestreo():
    set_seed(1234)
    a = torch.randn(8)
    set_seed(1234)
    assert torch.equal(a, torch.randn(8))


def test_device_config_es_inmutable():
    cfg = get_device("cpu")
    with pytest.raises(Exception):
        cfg.kind = "cuda"  # type: ignore[misc]
    assert isinstance(cfg, DeviceConfig)


def test_synchronize_y_empty_cache_no_petan_en_ningun_backend():
    cfg = get_device()
    cfg.synchronize()
    cfg.empty_cache()
    assert isinstance(cfg.memory_summary(), str)
