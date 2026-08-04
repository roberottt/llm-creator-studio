"""Tests del modulo 07. Ejecutalos con `llmfs check 07`.

El oraculo externo es `F.layer_norm` de PyTorch.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import llmfs.reference as ref
from llmfs.testing import assert_close, copy_parameters, load_exercises

ej = load_exercises(__file__)


# -------------------------------------------------------------- ejercicio 1: layer_norm


def test_el_ejemplo_de_la_teoria():
    x = torch.tensor([[2.0, 8.0, 4.0, 6.0]])
    esperado = torch.tensor([[-1.3416, 1.3416, -0.4472, 0.4472]])
    assert_close(ej.layer_norm(x), esperado, atol=1e-3, what="el ejemplo del THEORY.md")


def test_coincide_con_f_layer_norm_sin_parametros():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)
    assert_close(ej.layer_norm(x), F.layer_norm(x, (32,)), what="layer_norm sin afin")


def test_coincide_con_f_layer_norm_con_peso_y_sesgo():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)
    w, b = torch.randn(32), torch.randn(32)
    assert_close(ej.layer_norm(x, w, b), F.layer_norm(x, (32,), w, b), what="layer_norm afin")


def test_usa_la_varianza_poblacional_y_no_la_muestral():
    """`torch.var` divide por (n-1) por defecto. LayerNorm divide por n."""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    correcto = F.layer_norm(x, (4,))
    con_unbiased = (x - x.mean()) / torch.sqrt(x.var(unbiased=True) + 1e-5)

    mio = ej.layer_norm(x)
    err_bien = (mio - correcto).abs().max()
    err_mal = (mio - con_unbiased).abs().max()
    assert err_bien < err_mal, (
        "tu resultado se parece mas a la varianza muestral (n-1) que a la poblacional (n). "
        "Pasa unbiased=False a torch.var."
    )
    assert err_bien < 1e-5


def test_la_salida_tiene_media_cero_y_varianza_uno():
    torch.manual_seed(0)
    y = ej.layer_norm(torch.randn(4, 8, 64) * 10 + 5)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(4, 8), atol=1e-5)
    assert torch.allclose(y.var(dim=-1, unbiased=False), torch.ones(4, 8), atol=1e-3)


def test_normaliza_cada_token_por_separado():
    """Dos tokens con escalas muy distintas deben salir igual de normalizados."""
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [100.0, 200.0, 300.0, 400.0]]])
    y = ej.layer_norm(x)
    assert_close(y[0, 0], y[0, 1], atol=1e-4, what="los dos tokens normalizados")


def test_no_depende_del_tamanyo_del_batch():
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16)
    solo = ej.layer_norm(x)
    en_lote = ej.layer_norm(x.repeat(8, 1, 1))
    assert_close(en_lote[0], solo[0], what="el resultado con batch 1 y con batch 8")


def test_el_epsilon_evita_dividir_por_cero():
    """Un vector constante tiene varianza 0."""
    y = ej.layer_norm(torch.full((1, 8), 3.0))
    assert torch.isfinite(y).all(), "hay inf o nan: falta el eps en el denominador"


def test_coincide_con_la_referencia():
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    w, b = torch.randn(16), torch.randn(16)
    assert_close(ej.layer_norm(x, w, b), ref.layer_norm(x, w, b), what="layer_norm")


# ----------------------------------------------------------------- ejercicio 2: RMSNorm


def test_rmsnorm_reproduce_el_ejemplo_de_la_teoria():
    norm = ej.RMSNorm(4)
    x = torch.tensor([[2.0, 8.0, 4.0, 6.0]])
    esperado = torch.tensor([[0.3651, 1.4606, 0.7303, 1.0954]])
    assert_close(norm(x), esperado, atol=1e-3, what="el ejemplo del THEORY.md")


def test_rmsnorm_tiene_la_arquitectura_esperada():
    copy_parameters(ref.RMSNorm(32), ej.RMSNorm(32))


def test_rmsnorm_arranca_con_pesos_a_uno():
    """Al inicializar tiene que ser la normalizacion pura, sin escalar nada."""
    assert torch.allclose(ej.RMSNorm(16).weight, torch.ones(16))


def test_rmsnorm_no_resta_la_media():
    """La diferencia con LayerNorm: si restara la media, la salida sumaria 0."""
    x = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    y = ej.RMSNorm(4)(x)
    assert not torch.allclose(y.mean(), torch.zeros(1), atol=1e-3), (
        "la salida tiene media 0: estas restando la media, y RMSNorm no lo hace"
    )


def test_rmsnorm_coincide_con_la_formula():
    torch.manual_seed(0)
    x = torch.randn(4, 8, 32)
    norm = ej.RMSNorm(32, eps=1e-6)
    manual = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    assert_close(norm(x), manual, atol=1e-4, what="RMSNorm frente a la formula")


def test_rmsnorm_coincide_con_la_referencia():
    torch.manual_seed(0)
    mio, suyo = ej.RMSNorm(32), ref.RMSNorm(32)
    torch.nn.init.normal_(suyo.weight)
    copy_parameters(suyo, mio)
    x = torch.randn(2, 5, 32)
    assert_close(mio(x), suyo(x), atol=1e-5, what="RMSNorm")


def test_rmsnorm_tiene_solo_dim_parametros():
    """La mitad que LayerNorm, que tiene escala y sesgo. Cuenta para el total del modelo."""
    assert sum(p.numel() for p in ej.RMSNorm(320).parameters()) == 320


def test_rmsnorm_calcula_en_fp32_y_no_desborda():
    """Con activaciones grandes en fp16, x^2 se sale del rango si no promocionas."""
    x = torch.full((1, 8), 300.0, dtype=torch.float16)
    y = ej.RMSNorm(8)(x)
    assert torch.isfinite(y).all(), (
        "hay inf: 300^2 = 90.000 no cabe en fp16. Haz el calculo en float() y vuelve "
        "al dtype original con .type_as(x)"
    )


def test_rmsnorm_devuelve_fp32_aunque_la_entrada_sea_fp16():
    """Comportamiento deliberado, y conviene entenderlo antes de que sorprenda.

    El `.type_as(x)` devuelve la normalizacion a fp16, pero despues se multiplica por
    `self.weight`, que es un parametro fp32. La promocion de tipos de PyTorch hace que el
    resultado salga en fp32.

    No es un bug: es lo que hace la implementacion de Llama y es lo que quieres. Bajo
    autocast, los pesos del modelo se mantienen en fp32 y las operaciones siguientes
    convierten lo que necesiten. Dejar la salida de una normalizacion en la precision alta
    es gratis y da margen numerico.
    """
    x = torch.randn(2, 8, dtype=torch.float16)
    salida = ej.RMSNorm(8)(x)
    assert salida.dtype == torch.float32
    assert torch.isfinite(salida).all()


def test_rmsnorm_el_peso_escala_la_salida():
    norm = ej.RMSNorm(4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    base = norm(x).clone()
    with torch.no_grad():
        norm.weight.mul_(2.0)
    assert_close(norm(x), base * 2, atol=1e-4, what="la salida con el peso doblado")


# -------------------------------------------------------- ejercicio 3: prenorm_residual


def test_prenorm_es_x_mas_fn_de_norm_de_x():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    norm = ref.RMSNorm(8)
    fn = torch.nn.Linear(8, 8)
    assert_close(
        ej.prenorm_residual(x, fn, norm), x + fn(norm(x)), what="la formula de pre-norm"
    )


def test_prenorm_no_es_postnorm():
    """El error tipico: norm(x + fn(x)) en vez de x + fn(norm(x))."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    norm = ref.RMSNorm(8)
    fn = torch.nn.Linear(8, 8)
    assert not torch.allclose(ej.prenorm_residual(x, fn, norm), norm(x + fn(x)), atol=1e-4), (
        "tu resultado coincide con post-norm: has puesto la normalizacion fuera de la rama"
    )


def test_prenorm_coincide_con_la_referencia():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8)
    norm, fn = ref.RMSNorm(8), torch.nn.Linear(8, 8)
    assert_close(ej.prenorm_residual(x, fn, norm), ref.prenorm_residual(x, fn, norm))


def test_si_el_bloque_no_hace_nada_la_entrada_pasa_intacta():
    """La propiedad clave del residual: el camino x -> x esta libre."""
    x = torch.randn(2, 4, 8)
    assert_close(
        ej.prenorm_residual(x, lambda z: torch.zeros_like(z), ref.RMSNorm(8)),
        x,
        what="la salida con un bloque nulo",
    )


def test_el_gradiente_llega_intacto_a_traves_del_residual():
    """El 1 de d(x + f(x))/dx = 1 + df/dx. Es toda la razon de ser del residual."""
    x = torch.randn(2, 4, 8, requires_grad=True)
    # Un bloque que anula por completo el gradiente de su rama.
    salida = ej.prenorm_residual(x, lambda z: z.detach() * 0, ref.RMSNorm(8))
    salida.sum().backward()
    assert_close(x.grad, torch.ones_like(x), what="el gradiente por el atajo residual")


def test_apilar_muchos_bloques_no_desvanece_el_gradiente():
    """Con 40 capas y sin residual el gradiente se muere; con residual, no."""
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, requires_grad=True)
    norm = ref.RMSNorm(16)
    capa = torch.nn.Linear(16, 16)
    with torch.no_grad():
        capa.weight.mul_(0.1)  # una capa que encoge mucho su salida

    h = x
    for _ in range(40):
        h = ej.prenorm_residual(h, capa, norm)
    h.sum().backward()

    norma = float(x.grad.norm())
    assert norma > 0.1, (
        f"la norma del gradiente en la entrada es {norma:.2e}. Con residuales no deberia "
        "desvanecerse ni con 40 capas."
    )
    assert math.isfinite(norma)
