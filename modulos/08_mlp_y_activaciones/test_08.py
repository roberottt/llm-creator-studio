"""Tests del modulo 08. Ejecutalos con `llmfs check 08`.

El oraculo externo es `F.gelu(x, approximate="tanh")`.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import llmfs.reference as ref
from llmfs.testing import assert_close, assert_scalar_close, copy_parameters, load_exercises

ej = load_exercises(__file__)


# --------------------------------------------------------------------- ejercicio 1: gelu


def test_coincide_con_f_gelu_aproximado():
    x = torch.linspace(-6, 6, 500)
    assert_close(ej.gelu(x), F.gelu(x, approximate="tanh"), what="GELU")


def test_los_valores_del_enunciado():
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
    esperado = torch.tensor([-0.0036, -0.1588, 0.0, 0.8412, 2.9964])
    assert_close(ej.gelu(x), esperado, atol=1e-3, what="los valores de TEORIA.md")


def test_gelu_en_cero_es_cero():
    assert_scalar_close(ej.gelu(torch.zeros(1))[0], 0.0, atol=1e-9, what="GELU(0)")


def test_es_casi_la_identidad_para_valores_grandes():
    x = torch.tensor([10.0, 20.0])
    assert_close(ej.gelu(x), x, rtol=1e-4, what="GELU de valores muy positivos")


def test_anula_casi_del_todo_los_valores_muy_negativos():
    assert bool((ej.gelu(torch.tensor([-10.0, -20.0])).abs() < 1e-6).all())


def test_la_derivada_no_es_cero_en_la_zona_negativa():
    """La ventaja sobre ReLU: una neurona en negativo puede recuperarse."""
    x = torch.tensor([-1.0], requires_grad=True)
    ej.gelu(x).backward()
    assert abs(float(x.grad)) > 0.01, (
        "la derivada en x=-1 es practicamente cero: eso es ReLU, no GELU"
    )


def test_no_es_relu():
    x = torch.tensor([-2.0, -1.0, -0.5])
    assert not torch.allclose(ej.gelu(x), torch.zeros(3), atol=1e-3), (
        "anulas por completo los negativos: has implementado ReLU"
    )


def test_conserva_la_forma():
    for forma in [(10,), (4, 8), (2, 3, 16)]:
        assert ej.gelu(torch.randn(*forma)).shape == forma


def test_gelu_coincide_con_la_referencia():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)
    assert_close(ej.gelu(x), ref.gelu(x), what="GELU")


# -------------------------------------------------------- ejercicio 2: swiglu_hidden_dim


def test_produce_el_896_del_config_final():
    assert ej.swiglu_hidden_dim(320) == 896


def test_produce_el_384_del_config_juguete():
    assert ej.swiglu_hidden_dim(128) == 384


@pytest.mark.parametrize("d_model", [64, 128, 256, 320, 512, 768, 1024, 4096])
def test_d_ff_coincide_con_la_referencia(d_model):
    assert ej.swiglu_hidden_dim(d_model) == ref.swiglu_hidden_dim(d_model)


def test_el_resultado_es_multiplo_de_64():
    for d in (64, 100, 320, 777, 1024):
        assert ej.swiglu_hidden_dim(d) % 64 == 0


def test_redondea_hacia_arriba_y_no_hacia_abajo():
    """int(2*4*320/3) = 853, y el multiplo de 64 mas cercano por arriba es 896, no 832."""
    assert ej.swiglu_hidden_dim(320) == 896
    assert ej.swiglu_hidden_dim(320) != 832


def test_respeta_otro_multiple_of():
    valor = ej.swiglu_hidden_dim(320, multiple_of=256)
    assert valor % 256 == 0 and valor >= 853


def test_un_valor_ya_multiplo_no_se_infla():
    """96 -> int(2*384/3) = 256, que ya es multiplo de 64: debe quedarse en 256."""
    assert ej.swiglu_hidden_dim(96) == 256


def test_el_multiplicador_extra_se_aplica():
    """Llama lo usa para ajustar a mano el tamanyo del FFN."""
    normal = ej.swiglu_hidden_dim(4096)
    con_factor = ej.swiglu_hidden_dim(4096, ffn_dim_multiplier=1.3)
    assert con_factor > normal


def test_swiglu_gasta_los_mismos_parametros_que_un_ffn_clasico():
    """La razon de ser del 2/3, comprobada con numeros."""
    d = 320
    clasico = 2 * d * (4 * d)
    swiglu = 3 * d * ej.swiglu_hidden_dim(d)
    assert abs(swiglu - clasico) / clasico < 0.06, (
        f"SwiGLU gasta {swiglu:,} y el clasico {clasico:,}: deberian parecerse mucho"
    )


def test_devuelve_un_entero():
    assert isinstance(ej.swiglu_hidden_dim(320), int)


# ------------------------------------------------------------------ ejercicio 3: SwiGLU


def test_swiglu_tiene_la_arquitectura_esperada():
    copy_parameters(ref.SwiGLU(64, 128), ej.SwiGLU(64, 128))


def test_swiglu_devuelve_la_forma_correcta():
    ffn = ej.SwiGLU(64, 128)
    assert ffn(torch.randn(2, 7, 64)).shape == (2, 7, 64)


def test_swiglu_coincide_con_la_referencia():
    torch.manual_seed(0)
    mio, suyo = ej.SwiGLU(64, 128), ref.SwiGLU(64, 128)
    copy_parameters(suyo, mio)
    mio.eval()
    suyo.eval()
    x = torch.randn(2, 5, 64)
    assert_close(mio(x), suyo(x), what="la salida de SwiGLU")


def test_swiglu_coincide_con_la_formula():
    torch.manual_seed(0)
    ffn = ej.SwiGLU(64, 128)
    ffn.eval()
    x = torch.randn(2, 5, 64)
    manual = ffn.down_proj(F.silu(ffn.gate_proj(x)) * ffn.up_proj(x))
    assert_close(ffn(x), manual, what="SwiGLU frente a la formula")


def test_swiglu_aplica_la_activacion_a_la_rama_gate():
    """Si la aplicas a up_proj en vez de a gate_proj, no coincide con la referencia."""
    torch.manual_seed(0)
    ffn = ej.SwiGLU(64, 128)
    ffn.eval()
    x = torch.randn(2, 5, 64)
    al_reves = ffn.down_proj(ffn.gate_proj(x) * F.silu(ffn.up_proj(x)))
    assert not torch.allclose(ffn(x), al_reves, atol=1e-5), (
        "has puesto la activacion en up_proj; va en gate_proj"
    )


def test_swiglu_tiene_tres_matrices():
    ffn = ej.SwiGLU(320, 896, bias=False)
    assert sum(p.numel() for p in ffn.parameters()) == 3 * 320 * 896


def test_swiglu_reproduce_los_parametros_por_capa_del_modelo_final():
    """860.160, el numero del desglose del config."""
    assert sum(p.numel() for p in ej.SwiGLU(320, 896, bias=False).parameters()) == 860_160


def test_swiglu_procesa_cada_token_por_separado():
    """El FFN no mezcla posiciones: eso es trabajo de la atencion."""
    torch.manual_seed(0)
    ffn = ej.SwiGLU(32, 64)
    ffn.eval()
    x = torch.randn(1, 4, 32)

    entero = ffn(x)
    solo_uno = ffn(x[:, 2:3, :])
    assert_close(solo_uno[0, 0], entero[0, 2], atol=1e-5, what="el token 2 aislado")


def test_swiglu_es_no_lineal():
    """Si fuera lineal, apilar bloques no serviria de nada."""
    torch.manual_seed(0)
    ffn = ej.SwiGLU(32, 64)
    ffn.eval()
    x = torch.randn(1, 3, 32)
    assert not torch.allclose(ffn(2 * x), 2 * ffn(x), atol=1e-3), (
        "f(2x) == 2*f(x): tu modulo es lineal, falta la activacion"
    )


def test_swiglu_sin_sesgos_por_defecto():
    ffn = ej.SwiGLU(32, 64)
    assert ffn.gate_proj.bias is None, "bias=False por defecto, como el config del modelo"
