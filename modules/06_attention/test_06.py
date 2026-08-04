"""Tests del modulo 06. Ejecutalos con `llmfs check 06`.

La verdad la ponen dos oraculos: `llmfs.reference` y `torch.nn.MultiheadAttention`.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import llmfs.reference as ref
from llmfs.testing import assert_close, copy_parameters, load_exercises

ej = load_exercises(__file__)


# ------------------------------------------------------------- ejercicio 1: causal_mask


def test_la_mascara_es_triangular_inferior():
    m = ej.causal_mask(4)
    esperado = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(m, esperado)


def test_la_mascara_es_booleana():
    assert ej.causal_mask(5).dtype == torch.bool


def test_la_diagonal_esta_incluida():
    """Un token si puede mirarse a si mismo."""
    m = ej.causal_mask(6)
    assert bool(m.diagonal().all())


def test_ninguna_posicion_mira_al_futuro():
    m = ej.causal_mask(8)
    assert not bool(m.triu(diagonal=1).any()), "hay posiciones que ven hacia adelante"


def test_la_mascara_coincide_con_la_referencia():
    for n in (1, 2, 7, 64):
        assert torch.equal(ej.causal_mask(n), ref.causal_mask(n))


def test_la_mascara_respeta_el_dispositivo():
    from llmfs.device import get_device

    cfg = get_device()
    assert ej.causal_mask(4, device=cfg.device).device.type == cfg.kind


# --------------------------------------------------- ejercicio 2: single_head_attention


def datos(batch=2, t=5, d=8, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, t, d), torch.randn(batch, t, d), torch.randn(batch, t, d)


def test_las_formas_de_salida_son_correctas():
    q, k, v = datos()
    salida, pesos = ej.single_head_attention(q, k, v)
    assert salida.shape == (2, 5, 8)
    assert pesos.shape == (2, 5, 5)


def test_cada_fila_de_pesos_suma_uno():
    q, k, v = datos()
    _, pesos = ej.single_head_attention(q, k, v, ref.causal_mask(5))
    assert torch.allclose(pesos.sum(dim=-1), torch.ones(2, 5), atol=1e-6), (
        "cada fila debe sumar 1. Si no, probablemente has usado dim=-2 en el softmax"
    )


def test_la_mascara_pone_a_cero_el_futuro():
    q, k, v = datos()
    _, pesos = ej.single_head_attention(q, k, v, ref.causal_mask(5))
    assert torch.allclose(pesos[0].triu(1), torch.zeros(5, 5), atol=1e-9), (
        "hay peso en posiciones futuras: la mascara no se esta aplicando"
    )


def test_coincide_con_la_referencia():
    q, k, v = datos()
    mask = ref.causal_mask(5)
    salida_mia, pesos_mios = ej.single_head_attention(q, k, v, mask)
    salida_ref, pesos_ref = ref.single_head_attention(q, k, v, mask)
    assert_close(salida_mia, salida_ref, what="la salida")
    assert_close(pesos_mios, pesos_ref, what="los pesos")


def test_sin_mascara_tambien_funciona():
    q, k, v = datos()
    salida, pesos = ej.single_head_attention(q, k, v)
    assert torch.allclose(pesos.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
    assert torch.isfinite(salida).all()


def test_el_escalado_por_raiz_de_dk_esta_aplicado():
    """Se comprueba comparando contra el calculo explicito con y sin escalar."""
    torch.manual_seed(1)
    q, k, v = torch.randn(1, 4, 64), torch.randn(1, 4, 64), torch.randn(1, 4, 64)
    _, pesos = ej.single_head_attention(q, k, v)

    con_escala = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(64), dim=-1)
    sin_escala = torch.softmax(q @ k.transpose(-2, -1), dim=-1)

    err_con = (pesos - con_escala).abs().max()
    err_sin = (pesos - sin_escala).abs().max()
    assert err_con < err_sin, (
        f"tus pesos se parecen mas a la version SIN escalar (err {err_sin:.2e}) que a la "
        f"escalada (err {err_con:.2e}). Falta dividir por sqrt(d_k)."
    )
    assert err_con < 1e-6


def test_el_escalado_evita_que_el_softmax_se_sature():
    """El punto del ejercicio: con d_k grande y sin escalar, la atencion colapsa."""
    torch.manual_seed(2)
    q, k, v = torch.randn(1, 16, 256), torch.randn(1, 16, 256), torch.randn(1, 16, 256)
    _, pesos = ej.single_head_attention(q, k, v)

    # Entropia media de las distribuciones. Saturado = entropia cercana a 0.
    entropia = float(-(pesos * torch.log(pesos + 1e-12)).sum(-1).mean())
    entropia_maxima = math.log(16)
    assert entropia > 0.3 * entropia_maxima, (
        f"la entropia media es {entropia:.3f} de un maximo de {entropia_maxima:.3f}: "
        "el softmax esta saturado, lo que pasa cuando no se divide por sqrt(d_k)"
    )


def test_funciona_con_q_y_k_de_longitudes_distintas():
    """No es el caso del curso, pero la formula es general."""
    torch.manual_seed(3)
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 7, 8)
    v = torch.randn(2, 7, 16)
    salida, pesos = ej.single_head_attention(q, k, v)
    assert salida.shape == (2, 3, 16) and pesos.shape == (2, 3, 7)


def test_con_un_solo_token_la_atencion_es_la_identidad():
    """Con T=1 solo puede mirarse a si mismo, asi que la salida es v."""
    q, k, v = torch.randn(1, 1, 8), torch.randn(1, 1, 8), torch.randn(1, 1, 8)
    salida, pesos = ej.single_head_attention(q, k, v, ref.causal_mask(1))
    assert_close(salida, v, what="la salida con un solo token")
    assert_close(pesos, torch.ones(1, 1, 1), what="el peso unico")


# --------------------------------------------------- ejercicio 3: MultiHeadAttention


def test_mha_tiene_la_arquitectura_esperada():
    mio = ej.MultiHeadAttention(32, 4)
    copy_parameters(ref.MultiHeadAttention(32, 4), mio)


def test_mha_devuelve_la_forma_correcta():
    mha = ej.MultiHeadAttention(32, 4)
    salida = mha(torch.randn(2, 7, 32))
    assert salida.shape == (2, 7, 32)


def test_mha_valida_que_d_model_sea_divisible():
    with pytest.raises(ValueError):
        ej.MultiHeadAttention(d_model=32, n_heads=5)


def test_mha_coincide_con_la_referencia():
    torch.manual_seed(0)
    mio, suyo = ej.MultiHeadAttention(32, 4), ref.MultiHeadAttention(32, 4)
    copy_parameters(suyo, mio)
    mio.eval()
    suyo.eval()
    x = torch.randn(2, 7, 32)
    assert_close(mio(x), suyo(x), what="la salida de multi-head")


def test_mha_coincide_con_nn_multiheadattention():
    """El oraculo externo: la implementacion de PyTorch.

    `nn.MultiheadAttention` guarda las tres proyecciones concatenadas en un unico
    `in_proj_weight` de (3*d, d), asi que hay que trasladar los pesos a mano.
    Y su `attn_mask` usa el convenio CONTRARIO: True significa PROHIBIDO.
    """
    torch.manual_seed(0)
    d_model, n_heads, seq = 32, 4, 6

    mio = ej.MultiHeadAttention(d_model, n_heads, bias=True)
    suyo = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=True)

    with torch.no_grad():
        suyo.in_proj_weight.copy_(
            torch.cat([mio.q_proj.weight, mio.k_proj.weight, mio.v_proj.weight])
        )
        suyo.in_proj_bias.copy_(
            torch.cat([mio.q_proj.bias, mio.k_proj.bias, mio.v_proj.bias])
        )
        suyo.out_proj.weight.copy_(mio.out_proj.weight)
        suyo.out_proj.bias.copy_(mio.out_proj.bias)

    mio.eval()
    suyo.eval()
    x = torch.randn(2, seq, d_model)
    mask = ref.causal_mask(seq)

    salida_mia = mio(x, mask=mask)
    salida_suya, _ = suyo(x, x, x, attn_mask=~mask, need_weights=False)
    assert_close(salida_mia, salida_suya, atol=1e-5, what="la salida frente a PyTorch")


def test_mha_devuelve_pesos_si_se_los_pides():
    mha = ej.MultiHeadAttention(32, 4)
    salida, pesos = mha(torch.randn(2, 7, 32), return_weights=True)
    assert salida.shape == (2, 7, 32)
    assert pesos.shape == (2, 4, 7, 7), "los pesos son (B, n_heads, T, T), uno por cabeza"


def test_mha_aplica_la_mascara_causal_por_defecto():
    mha = ej.MultiHeadAttention(32, 4)
    mha.eval()
    _, pesos = mha(torch.randn(1, 6, 32), return_weights=True)
    assert torch.allclose(pesos[0, 0].triu(1), torch.zeros(6, 6), atol=1e-9)


def test_mha_no_mezcla_informacion_entre_cabezas():
    """Cada cabeza tiene que operar en su propio subespacio de head_dim dimensiones."""
    torch.manual_seed(0)
    mha = ej.MultiHeadAttention(32, 4)
    mha.eval()
    _, pesos = mha(torch.randn(1, 5, 32), return_weights=True)
    # Si las cabezas fueran identicas, seria senyal de que la particion esta mal.
    assert not torch.allclose(pesos[0, 0], pesos[0, 1], atol=1e-4), (
        "todas las cabezas dan el mismo patron: revisa el view/transpose de la particion"
    )


def test_mha_tiene_los_parametros_esperados():
    """4 * d_model^2 sin sesgos, que es lo que asume el conteo del modelo final."""
    mha = ej.MultiHeadAttention(320, 8, bias=False)
    assert sum(p.numel() for p in mha.parameters()) == 4 * 320 * 320


def test_mha_acepta_rope_y_cambia_el_resultado():
    """Con cos/sin, q y k se rotan y la salida tiene que ser distinta."""
    torch.manual_seed(0)
    mha = ej.MultiHeadAttention(32, 4)
    mha.eval()
    x = torch.randn(2, 6, 32)
    cos, sin = ref.rope_frequencies(8, 16)

    sin_rope = mha(x)
    con_rope = mha(x, cos=cos, sin=sin)
    assert not torch.allclose(sin_rope, con_rope, atol=1e-5), "RoPE no se esta aplicando"

    esperado = ref.MultiHeadAttention(32, 4)
    copy_parameters(mha, esperado)
    esperado.eval()
    assert_close(con_rope, esperado(x, cos=cos, sin=sin), what="la salida con RoPE")


def test_mha_con_sdpa_da_el_mismo_resultado():
    """El kernel fusionado del modulo 12 no puede cambiar la salida, solo la velocidad."""
    torch.manual_seed(0)
    mha = ej.MultiHeadAttention(32, 4)
    mha.eval()
    x = torch.randn(2, 7, 32)

    explicito = mha(x)
    mha.use_sdpa = True
    fusionado = mha(x)
    assert_close(fusionado, explicito, atol=1e-5, what="la salida con SDPA")
