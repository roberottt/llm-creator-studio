"""Tests del modulo 01. Ejecutalos con `llmfs check 01`."""

from __future__ import annotations

import math

import pytest
import torch

import llmfs.reference as ref
from llmfs.device import get_device
from llmfs.testing import assert_scalar_close, load_exercises

ej = load_exercises(__file__)


# --------------------------------------------------------------- ejercicio 2: FLOPs


@pytest.mark.parametrize(
    "params",
    [
        # el modelo final de este curso
        dict(n_layers=6, d_model=320, d_ff=896, context_length=512, vocab_size=4096),
        # el juguete de shakespeare
        dict(n_layers=4, d_model=128, d_ff=384, context_length=128, vocab_size=65),
        # algo grande, para que el termino de atencion pese distinto
        dict(n_layers=32, d_model=4096, d_ff=11008, context_length=4096, vocab_size=32000),
    ],
)
def test_flops_por_token_coincide_con_la_referencia(params):
    assert ej.transformer_flops_per_token(**params) == ref.transformer_flops_per_token(**params)


def test_el_conteo_del_modelo_de_9m_es_el_esperado():
    """Numero fijo: si cambia, el THEORY.md y el modulo 11 mienten."""
    flops = ej.transformer_flops_per_token(
        n_layers=6, d_model=320, d_ff=896, context_length=512, vocab_size=4096
    )
    assert flops == 65_372_160


def test_el_backward_cuesta_el_doble_que_el_forward():
    comun = dict(n_layers=6, d_model=320, d_ff=896, context_length=512, vocab_size=4096)
    solo_fwd = ej.transformer_flops_per_token(**comun, include_backward=False)
    con_bwd = ej.transformer_flops_per_token(**comun, include_backward=True)
    assert con_bwd == 3 * solo_fwd


def test_el_termino_de_atencion_crece_con_el_contexto():
    """La parte 6N no depende de T; la de atencion si, linealmente."""
    base = dict(n_layers=6, d_model=320, d_ff=896, vocab_size=4096)
    corto = ej.transformer_flops_per_token(**base, context_length=512)
    largo = ej.transformer_flops_per_token(**base, context_length=1024)
    termino_atencion = 3 * 4 * 6 * 512 * 320
    assert largo - corto == termino_atencion


def test_un_ffn_clasico_de_dos_matrices_cuesta_menos_que_swiglu():
    base = dict(n_layers=6, d_model=320, d_ff=896, context_length=512, vocab_size=4096)
    assert ej.transformer_flops_per_token(**base, n_ffn_matrices=2) < ej.transformer_flops_per_token(
        **base, n_ffn_matrices=3
    )


def test_devuelve_un_entero():
    flops = ej.transformer_flops_per_token(
        n_layers=6, d_model=320, d_ff=896, context_length=512, vocab_size=4096
    )
    assert isinstance(flops, int)


# ------------------------------------------------------- ejercicio 3: tokens/segundo


def test_tokens_por_segundo_coincide_con_la_referencia():
    assert_scalar_close(
        ej.estimate_tokens_per_second(51.6, 65_372_160, mfu=0.15),
        ref.estimate_tokens_per_second(51.6, 65_372_160, mfu=0.15),
        what="tokens/s",
    )


def test_tokens_por_segundo_escala_linealmente_con_la_mfu():
    a = ej.estimate_tokens_per_second(50.0, 1_000_000, mfu=0.1)
    b = ej.estimate_tokens_per_second(50.0, 1_000_000, mfu=0.2)
    assert_scalar_close(b, 2 * a, what="el doble de MFU")


def test_flops_por_token_no_positivo_es_un_error():
    with pytest.raises(ValueError):
        ej.estimate_tokens_per_second(50.0, 0)


# ------------------------------------------------------- ejercicio 1: medicion real


def test_la_medicion_devuelve_un_numero_plausible():
    """No se puede comparar contra una referencia: es una medida. Se acota."""
    tflops = ej.measure_matmul_tflops(size=512, warmup=2, iters=5)
    assert isinstance(tflops, float)
    assert math.isfinite(tflops)
    assert 0.001 < tflops < 10_000, (
        f"{tflops:.3f} TFLOPS no es plausible. Si sale disparado (miles), te has "
        "olvidado de sincronizar y estas midiendo el tiempo de encolar."
    )


def test_medir_en_cpu_da_menos_que_el_dispositivo_por_defecto_o_lo_mismo():
    """En CPU no hay asincronia, asi que este numero SI es de fiar sin sincronizar."""
    cpu = ej.measure_matmul_tflops(cfg=get_device("cpu"), size=512, warmup=2, iters=5)
    assert 0.001 < cpu < 100


def test_matrices_mas_grandes_no_bajan_el_rendimiento_a_plomo():
    """Un matmul mayor deberia aprovechar mejor el hardware, no peor.

    Si esto falla con una diferencia enorme, casi siempre es que falta el calentamiento:
    el coste fijo de la primera llamada se reparte distinto segun el tamanyo.
    """
    pequeno = ej.measure_matmul_tflops(size=256, warmup=3, iters=10)
    grande = ej.measure_matmul_tflops(size=1024, warmup=3, iters=10)
    assert grande > pequeno / 10


def test_acepta_un_dtype_explicito():
    valor = ej.measure_matmul_tflops(cfg=get_device("cpu"), size=256, dtype=torch.float32, iters=3)
    assert math.isfinite(valor) and valor > 0
