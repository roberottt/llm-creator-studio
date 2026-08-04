"""Tests del modulo 12. Ejecutalos con `llmfs check 12`."""

from __future__ import annotations

import math

import pytest

import llmfs.reference as ref
from llmfs.config import ModelConfig
from llmfs.testing import assert_scalar_close, load_exercises

ej = load_exercises(__file__)


# ------------------------------------------------- ejercicio 1: model_flops_per_token


def test_el_total_es_la_suma_de_las_partes():
    f = ej.model_flops_per_token(ModelConfig())
    assert f["total"] == f["matmul"] + f["attention"]


def test_reproduce_el_numero_del_modulo_01():
    """65,4 MFLOPs por token. El mismo calculo, ahora desglosado."""
    assert ej.model_flops_per_token(ModelConfig())["total"] == 65_372_160


def test_el_desglose_del_modelo_final():
    f = ej.model_flops_per_token(ModelConfig())
    assert f["attention"] / f["total"] == pytest.approx(0.18, abs=0.02), (
        "con contexto 512 la atencion deberia ser ~18% del total"
    )


def test_la_atencion_crece_con_el_contexto_y_el_matmul_no():
    corto = ej.model_flops_per_token(ModelConfig(context_length=512))
    largo = ej.model_flops_per_token(ModelConfig(context_length=1024))
    assert largo["matmul"] == corto["matmul"], "el termino de matmul no depende del contexto"
    assert largo["attention"] == 2 * corto["attention"], "la atencion crece linealmente con T"


def test_con_contexto_muy_largo_la_atencion_domina():
    f = ej.model_flops_per_token(ModelConfig(context_length=4096))
    assert f["attention"] > f["matmul"], (
        "con contexto 4096 la atencion deberia pesar mas que los matmuls"
    )


def test_el_backward_triplica_el_coste():
    solo_fwd = ej.model_flops_per_token(ModelConfig(), include_backward=False)
    con_bwd = ej.model_flops_per_token(ModelConfig(), include_backward=True)
    assert con_bwd["total"] == 3 * solo_fwd["total"]


def test_swiglu_cuesta_mas_que_un_ffn_clasico():
    swiglu = ej.model_flops_per_token(ModelConfig(activation="swiglu"))
    clasico = ej.model_flops_per_token(ModelConfig(activation="gelu"))
    assert swiglu["matmul"] > clasico["matmul"]


def test_la_proyeccion_final_cuenta_aunque_haya_tying():
    """Atar los pesos ahorra memoria, no calculo: el matmul se hace igual."""
    con = ej.model_flops_per_token(ModelConfig(tie_embeddings=True))
    sin = ej.model_flops_per_token(ModelConfig(tie_embeddings=False))
    assert con["total"] == sin["total"]


@pytest.mark.parametrize(
    "cfg",
    [
        ModelConfig(),
        ModelConfig(n_layers=1),
        ModelConfig(context_length=128),
        ModelConfig(activation="gelu"),
        ModelConfig(vocab_size=128, d_model=64, n_heads=8, d_ff=192),
    ],
)
def test_los_flops_coinciden_con_la_referencia(cfg):
    assert ej.model_flops_per_token(cfg) == ref.model_flops_per_token(cfg)


# ---------------------------------------------------------- ejercicio 2: compute_mfu


def test_la_formula_es_la_esperada():
    assert_scalar_close(
        ej.compute_mfu(1000, 1_000_000, 10.0), 1e9 / 1e13, rtol=1e-9, what="la MFU"
    )


def test_mfu_uno_cuando_se_aprovecha_todo():
    """Si tokens/s * flops/token == el pico exacto, la MFU es 1."""
    assert_scalar_close(ej.compute_mfu(1e6, 1e7, 10.0), 1.0, rtol=1e-9, what="la MFU maxima")


def test_escala_linealmente_con_los_tokens_por_segundo():
    a = ej.compute_mfu(1000, 65_372_160, 51.6)
    b = ej.compute_mfu(2000, 65_372_160, 51.6)
    assert_scalar_close(b, 2 * a, what="el doble de tokens/s")


def test_un_pico_no_positivo_es_un_error():
    for malo in (0.0, -10.0):
        with pytest.raises(ValueError):
            ej.compute_mfu(1000, 1_000_000, malo)


def test_el_caso_del_modelo_final_en_la_2060():
    """3.000 tok/s en una RTX 2060 con nuestro modelo."""
    mfu = ej.compute_mfu(3000, 65_372_160, 51.6)
    assert 0.001 < mfu < 0.02, f"MFU {mfu:.4f} fuera del rango esperable"


def test_la_mfu_coincide_con_la_referencia():
    assert_scalar_close(
        ej.compute_mfu(3000, 65_372_160, 51.6),
        ref.compute_mfu(3000, 65_372_160, 51.6),
        what="la MFU",
    )


# ------------------------------------------ ejercicio 3: chinchilla_optimal_allocation


def test_reproduce_el_tamanyo_de_chinchilla():
    """EL test del modulo. Con el presupuesto real de Chinchilla (5.76e23 FLOPs), la
    formula tiene que dar ~70.000 millones de parametros, que es lo que tenia el modelo."""
    a = ej.chinchilla_optimal_allocation(5.76e23)
    assert 6.5e10 < a["params"] < 7.5e10, (
        f"la formula da {a['params']:.2e} parametros y Chinchilla tenia 7.0e10"
    )


def test_reproduce_los_tokens_de_chinchilla():
    """1,4 billones de tokens."""
    a = ej.chinchilla_optimal_allocation(5.76e23)
    assert 1.2e12 < a["tokens"] < 1.6e12


def test_la_relacion_es_de_veinte_tokens_por_parametro():
    a = ej.chinchilla_optimal_allocation(1e21)
    assert_scalar_close(a["tokens"] / a["params"], 20.0, rtol=1e-6, what="tokens por parametro")


def test_el_presupuesto_cuadra_con_6nd():
    """C = 6ND tiene que devolver el presupuesto de partida."""
    C = 1e22
    a = ej.chinchilla_optimal_allocation(C)
    assert_scalar_close(6 * a["params"] * a["tokens"], C, rtol=1e-6, what="C = 6ND")


def test_cuadruplicar_el_computo_dobla_ambos():
    """N y D crecen con la raiz del presupuesto."""
    a = ej.chinchilla_optimal_allocation(1e21)
    b = ej.chinchilla_optimal_allocation(4e21)
    assert_scalar_close(b["params"] / a["params"], 2.0, rtol=1e-6, what="el factor en N")
    assert_scalar_close(b["tokens"] / a["tokens"], 2.0, rtol=1e-6, what="el factor en D")


def test_gpt3_estaba_infraentrenado():
    """Con su presupuesto, GPT-3 deberia haber tenido muchos menos parametros."""
    C_gpt3 = 6 * 175e9 * 300e9
    a = ej.chinchilla_optimal_allocation(C_gpt3)
    assert a["params"] < 175e9 / 2, (
        f"con el presupuesto de GPT-3 lo optimo eran {a['params']:.2e} parametros, "
        f"no 1.75e11"
    )


def test_otro_valor_de_tokens_por_parametro():
    """Llama-3 usa ~1800, muy por encima de Chinchilla."""
    a = ej.chinchilla_optimal_allocation(1e22, tokens_per_param=1800.0)
    assert_scalar_close(a["tokens"] / a["params"], 1800.0, rtol=1e-6)
    assert a["params"] < ej.chinchilla_optimal_allocation(1e22)["params"]


def test_un_presupuesto_no_positivo_es_un_error():
    for malo in (0.0, -1e20):
        with pytest.raises(ValueError):
            ej.chinchilla_optimal_allocation(malo)


def test_tiene_todas_las_claves():
    a = ej.chinchilla_optimal_allocation(1e21)
    for clave in ("params", "tokens", "tokens_per_param", "compute"):
        assert clave in a


def test_chinchilla_coincide_con_la_referencia():
    mio = ej.chinchilla_optimal_allocation(5.76e23)
    suyo = ref.chinchilla_optimal_allocation(5.76e23)
    for clave in suyo:
        assert_scalar_close(mio[clave], suyo[clave], what=clave)
