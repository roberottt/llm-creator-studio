"""Tests del modulo 17. Ejecutalos con `llmfs check 17`. Son los ultimos del curso."""

from __future__ import annotations

import pytest
import torch

import llmfs.reference as ref
from llmfs.config import ModelConfig
from llmfs.device import set_seed
from llmfs.testing import assert_close, assert_scalar_close, load_exercises

ej = load_exercises(__file__)


# --------------------------------------------------- ejercicio 1: quantize_int8_symmetric


def test_el_dtype_de_salida_es_int8():
    q, _ = ej.quantize_int8_symmetric(torch.randn(4, 8))
    assert q.dtype == torch.int8


def test_los_valores_caben_en_int8():
    q, _ = ej.quantize_int8_symmetric(torch.randn(16, 16) * 100)
    assert int(q.max()) <= 127 and int(q.min()) >= -127


def test_el_ejemplo_del_enunciado():
    w = torch.tensor([[0.12, -0.45, 0.03, 0.28]])
    q, escala = ej.quantize_int8_symmetric(w)
    assert_scalar_close(float(escala), 0.45 / 127, rtol=1e-4, what="la escala")
    assert q[0].tolist() == [34, -127, 8, 79]


def test_el_maximo_absoluto_se_mapea_a_127():
    w = torch.tensor([[0.1, -0.5, 0.3]])
    q, _ = ej.quantize_int8_symmetric(w)
    assert int(q.abs().max()) == 127


def test_el_cero_se_representa_exactamente():
    """Por eso se usa 127 y no 128: el rango queda simetrico."""
    w = torch.tensor([[0.0, 0.5, -0.5]])
    q, escala = ej.quantize_int8_symmetric(w)
    assert int(q[0, 0]) == 0
    assert float(ej.dequantize_int8(q, escala)[0, 0]) == 0.0


def test_por_canal_da_una_escala_por_fila():
    q, escala = ej.quantize_int8_symmetric(torch.randn(5, 8), per_channel=True)
    assert escala.shape == (5, 1)


def test_por_tensor_da_una_escala_unica():
    _, escala = ej.quantize_int8_symmetric(torch.randn(5, 8), per_channel=False)
    assert escala.numel() == 1


def test_por_canal_es_mas_preciso_que_por_tensor():
    """Una fila con valores grandes no debe arrastrar a las demas."""
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    w[0] *= 100  # una fila muy grande

    e_canal = ej.quantization_error(w, per_channel=True)["relative_error"]
    e_tensor = ej.quantization_error(w, per_channel=False)["relative_error"]
    assert e_canal < e_tensor, (
        f"por canal {e_canal:.4f} deberia ser mejor que por tensor {e_tensor:.4f}"
    )


def test_una_fila_de_ceros_no_revienta():
    """El clamp_min de la escala evita dividir por cero."""
    w = torch.zeros(3, 4)
    q, escala = ej.quantize_int8_symmetric(w)
    assert torch.isfinite(escala).all() and int(q.abs().max()) == 0


def test_la_cuantizacion_coincide_con_la_referencia():
    torch.manual_seed(0)
    w = torch.randn(8, 16)
    q_mio, s_mio = ej.quantize_int8_symmetric(w)
    q_ref, s_ref = ref.quantize_int8_symmetric(w)
    assert torch.equal(q_mio, q_ref)
    assert_close(s_mio, s_ref, what="las escalas")


# --------------------------------------------------------- ejercicio 2: dequantize_int8


def test_el_roundtrip_se_acerca_al_original():
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    q, escala = ej.quantize_int8_symmetric(w)
    recuperado = ej.dequantize_int8(q, escala)
    assert_close(recuperado, w, atol=0.02, what="el roundtrip")


def test_devuelve_float():
    q, escala = ej.quantize_int8_symmetric(torch.randn(4, 8))
    assert ej.dequantize_int8(q, escala).dtype == torch.float32


def test_conserva_la_forma():
    w = torch.randn(6, 10)
    q, escala = ej.quantize_int8_symmetric(w)
    assert ej.dequantize_int8(q, escala).shape == w.shape


def test_el_roundtrip_no_es_exacto():
    """Se pierde informacion: eso es lo que se paga."""
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    q, escala = ej.quantize_int8_symmetric(w)
    assert not torch.allclose(ej.dequantize_int8(q, escala), w, atol=1e-6)


def test_la_descuantizacion_coincide_con_la_referencia():
    torch.manual_seed(0)
    q, escala = ref.quantize_int8_symmetric(torch.randn(4, 16))
    assert_close(ej.dequantize_int8(q, escala), ref.dequantize_int8(q, escala))


# ------------------------------------------------------ ejercicio 3: quantization_error


def test_tiene_todas_las_claves():
    e = ej.quantization_error(torch.randn(4, 8))
    for clave in ("relative_error", "max_error", "mean_error", "compression",
                  "original_bytes", "quantized_bytes"):
        assert clave in e, f"falta la clave {clave!r}"


def test_la_compresion_de_fp32_a_int8_es_4x():
    assert ej.quantization_error(torch.randn(4, 8))["compression"] == 4.0


def test_el_error_relativo_de_pesos_reales_es_pequenyo():
    """Con pesos de una red entrenada, int8 por canal ronda el 0.5-1%."""
    set_seed(0)
    w = ref.GPT(ModelConfig()).blocks[0].attn.q_proj.weight.data
    error = ej.quantization_error(w)["relative_error"]
    assert 0.001 < error < 0.03, f"error relativo {error:.4f} fuera del rango esperado"


def test_los_bytes_cuantizados_incluyen_las_escalas():
    """Las escalas tambien ocupan: contarlas es lo honesto."""
    w = torch.randn(10, 20)
    e = ej.quantization_error(w, per_channel=True)
    assert e["quantized_bytes"] > w.numel(), (
        "los bytes cuantizados tienen que incluir el vector de escalas"
    )
    assert e["original_bytes"] == w.numel() * 4


def test_el_error_relativo_es_independiente_de_la_escala():
    """Multiplicar los pesos por 1000 no cambia el error RELATIVO."""
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    a = ej.quantization_error(w)["relative_error"]
    b = ej.quantization_error(w * 1000)["relative_error"]
    assert_scalar_close(a, b, rtol=1e-3, what="el error relativo")


def test_el_error_del_modelo_completo_es_asumible():
    """Sobre todas las matrices del modelo final."""
    set_seed(0)
    modelo = ref.GPT(ModelConfig())
    errores = [
        ej.quantization_error(p.data)["relative_error"]
        for p in modelo.parameters()
        if p.dim() >= 2
    ]
    assert max(errores) < 0.05, f"la peor matriz tiene un error del {max(errores):.1%}"


def test_el_error_coincide_con_la_referencia():
    torch.manual_seed(0)
    w = torch.randn(8, 32)
    mio, suyo = ej.quantization_error(w), ref.quantization_error(w)
    for clave in suyo:
        assert_scalar_close(mio[clave], suyo[clave], what=clave)
