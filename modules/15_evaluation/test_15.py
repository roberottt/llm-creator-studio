"""Tests del modulo 15. Ejecutalos con `llmfs check 15`."""

from __future__ import annotations

import math

import pytest

import llmfs.reference as ref
from llmfs.testing import assert_scalar_close, load_exercises

ej = load_exercises(__file__)


# ------------------------------------------------------ ejercicio 1: perplexity_from_loss


def test_es_la_exponencial_de_la_perdida():
    assert_scalar_close(ej.perplexity_from_loss(1.6), math.exp(1.6), what="la perplejidad")


def test_el_modelo_sin_entrenar_tiene_perplejidad_igual_al_vocabulario():
    """Con perdida ln(V), la perplejidad es exactamente V. Es la comprobacion util."""
    for vocab in (65, 4096, 50257):
        assert_scalar_close(
            ej.perplexity_from_loss(math.log(vocab)), vocab, rtol=1e-9,
            what=f"la perplejidad con perdida ln({vocab})"
        )


def test_perdida_cero_da_perplejidad_uno():
    assert_scalar_close(ej.perplexity_from_loss(0.0), 1.0, what="el modelo perfecto")


def test_una_perdida_no_finita_da_infinito_sin_reventar():
    """`math.exp(inf)` lanza OverflowError: hay que cazarlo."""
    for malo in (float("inf"), float("nan")):
        assert ej.perplexity_from_loss(malo) == float("inf")


def test_es_monotona_creciente():
    valores = [ej.perplexity_from_loss(x) for x in (0.5, 1.0, 2.0, 4.0)]
    assert all(a < b for a, b in zip(valores, valores[1:]))


def test_la_perplejidad_coincide_con_la_referencia():
    for p in (0.0, 1.6, 8.317):
        assert_scalar_close(ej.perplexity_from_loss(p), ref.perplexity_from_loss(p))


# ------------------------------------------------------------ ejercicio 2: bits_per_byte


def test_la_formula_es_la_esperada():
    assert_scalar_close(
        ej.bits_per_byte(1000.0, 200, 500), 1000.0 / math.log(2) / 500, what="bits por byte"
    )


def test_convierte_nats_a_bits():
    """Un nat son 1/ln(2) = 1.4427 bits."""
    assert_scalar_close(ej.bits_per_byte(math.log(2), 1, 1), 1.0, rtol=1e-9,
                        what="ln(2) nats sobre 1 byte")


def test_mas_bytes_para_la_misma_perdida_da_menos_bits_por_byte():
    """Es lo que hace la metrica comparable entre tokenizadores."""
    assert ej.bits_per_byte(1000.0, 200, 1000) < ej.bits_per_byte(1000.0, 200, 500)


def test_no_depende_del_numero_de_tokens():
    """El punto del ejercicio: la metrica normaliza por BYTES, no por tokens."""
    a = ej.bits_per_byte(1000.0, 200, 500)
    b = ej.bits_per_byte(1000.0, 800, 500)
    assert_scalar_close(a, b, what="el resultado con distinto numero de tokens")


def test_un_valor_realista_esta_en_el_rango_esperado():
    """Perdida media 1.2 nats/token, 3 bytes por token: ~0.58 bits/byte."""
    n_tokens, perdida_media, bytes_por_token = 1000, 1.2, 3
    bpb = ej.bits_per_byte(perdida_media * n_tokens, n_tokens, n_tokens * bytes_por_token)
    assert 0.3 < bpb < 1.5, f"{bpb:.3f} fuera del rango de un modelo razonable"


def test_bytes_no_positivos_es_un_error():
    for malo in (0, -100):
        with pytest.raises(ValueError):
            ej.bits_per_byte(1000.0, 200, malo)


def test_bits_por_byte_coincide_con_la_referencia():
    assert_scalar_close(
        ej.bits_per_byte(1234.5, 300, 900), ref.bits_per_byte(1234.5, 300, 900)
    )


# --------------------------------------------------------- ejercicio 3: run_prompt_battery


def generador_falso(prompt: str) -> str:
    return prompt + " [continuacion generada]"


def test_devuelve_un_resultado_por_prompt():
    resultados = ej.run_prompt_battery(generador_falso)
    assert len(resultados) == len(ref.PROMPTS_TINYSTORIES)


def test_cada_resultado_tiene_las_tres_claves():
    for caso in ej.run_prompt_battery(generador_falso):
        assert set(caso) == {"prompt", "tests", "completion"}


def test_la_completion_sale_del_generador():
    resultados = ej.run_prompt_battery(generador_falso)
    for caso in resultados:
        assert caso["completion"] == generador_falso(caso["prompt"])


def test_conserva_el_orden_y_las_etiquetas():
    resultados = ej.run_prompt_battery(generador_falso)
    for caso, (prompt, etiqueta) in zip(resultados, ref.PROMPTS_TINYSTORIES):
        assert caso["prompt"] == prompt
        assert caso["tests"] == etiqueta


def test_admite_una_bateria_propia():
    mios = (("Hola", "saludo"), ("Adios", "despedida"))
    resultados = ej.run_prompt_battery(generador_falso, mios)
    assert len(resultados) == 2
    assert resultados[0]["tests"] == "saludo"


def test_la_bateria_prueba_cosas_distintas():
    """Seis prompts con seis etiquetas distintas: no son variaciones de lo mismo."""
    etiquetas = [e for _, e in ref.PROMPTS_TINYSTORIES]
    assert len(set(etiquetas)) == len(etiquetas)


def test_la_bateria_coincide_con_la_referencia():
    mios = ej.run_prompt_battery(generador_falso)
    suyos = ref.run_prompt_battery(generador_falso)
    assert mios == suyos
