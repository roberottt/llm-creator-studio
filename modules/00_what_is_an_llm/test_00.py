"""Tests del modulo 00. Ejecutalos con `llmfs check 00`."""

from __future__ import annotations

import random

import pytest

import llmfs.reference as ref
from llmfs.testing import assert_scalar_close, load_exercises

ej = load_exercises(__file__)

TEXTO = "banana bandana cabana " * 30


# ------------------------------------------------- ejercicio 1: next_token_probs


def test_las_probabilidades_suman_uno():
    probs = ej.next_token_probs({"a": 40, "b": 25, "c": 20, "d": 15})
    assert_scalar_close(sum(probs.values()), 1.0, what="la suma de las probabilidades")


def test_el_ejemplo_del_enunciado():
    probs = ej.next_token_probs({"b": 3, "c": 1})
    assert_scalar_close(probs["b"], 0.75, what="P(b)")
    assert_scalar_close(probs["c"], 0.25, what="P(c)")


def test_conserva_las_claves_y_su_orden():
    """El orden importa: el muestreo recorre el diccionario y depende de el."""
    counts = {"z": 1, "a": 2, "m": 3}
    assert list(ej.next_token_probs(counts).keys()) == ["z", "a", "m"]


def test_un_unico_caracter_tiene_probabilidad_uno():
    assert_scalar_close(ej.next_token_probs({"x": 7})["x"], 1.0, what="P(x) cuando es el unico")


def test_coincide_con_la_referencia():
    counts = {"n": 40, "r": 25, " ": 20, "s": 15}
    mio, suyo = ej.next_token_probs(counts), ref.next_token_probs(counts)
    assert mio == suyo


def test_una_tabla_vacia_es_un_error_explicito():
    with pytest.raises(ValueError):
        ej.next_token_probs({})


def test_conteos_todos_a_cero_tambien_son_un_error():
    with pytest.raises(ValueError):
        ej.next_token_probs({"a": 0, "b": 0})


# ------------------------------------------------ ejercicio 2: sample_next_token


def test_siempre_devuelve_una_clave_de_la_distribucion():
    probs = {"a": 0.5, "b": 0.3, "c": 0.2}
    rng = random.Random(0)
    for _ in range(200):
        assert ej.sample_next_token(probs, rng) in probs


def test_con_probabilidad_uno_sale_siempre_el_mismo():
    rng = random.Random(0)
    for _ in range(50):
        assert ej.sample_next_token({"unico": 1.0}, rng) == "unico"


def test_las_frecuencias_se_parecen_a_las_probabilidades():
    """Muestrear 20.000 veces de {a:0.7, b:0.3} debe dar ~70% de 'a'."""
    probs = {"a": 0.7, "b": 0.3}
    rng = random.Random(1234)
    muestras = [ej.sample_next_token(probs, rng) for _ in range(20_000)]
    frecuencia_a = muestras.count("a") / len(muestras)
    assert abs(frecuencia_a - 0.7) < 0.02, (
        f"'a' ha salido el {frecuencia_a:.1%} de las veces y deberia rondar el 70%. "
        "Revisa el acumulador de la ruleta."
    )


def test_respeta_el_orden_del_diccionario():
    """Un r pequenyo tiene que caer en el PRIMER trozo de la ruleta, sea cual sea."""

    class RngFalso:
        def random(self) -> float:
            return 0.01

    assert ej.sample_next_token({"primero": 0.5, "segundo": 0.5}, RngFalso()) == "primero"


def test_un_r_casi_uno_cae_en_el_ultimo():
    class RngFalso:
        def random(self) -> float:
            return 0.999999999

    assert ej.sample_next_token({"a": 0.5, "b": 0.5}, RngFalso()) == "b"


def test_nunca_devuelve_none_aunque_las_probabilidades_no_sumen_exacto():
    """Error de redondeo tipico: 0.1 * 10 no da 1.0 en coma flotante."""

    class RngFalso:
        def random(self) -> float:
            return 0.9999999999

    probs = {chr(97 + i): 0.1 for i in range(10)}
    assert ej.sample_next_token(probs, RngFalso()) is not None


def test_la_misma_semilla_da_la_misma_secuencia_que_la_referencia():
    probs = {"n": 0.4, "r": 0.25, " ": 0.2, "s": 0.15}
    mio = [ej.sample_next_token(probs, random.Random(7)) for _ in range(1)]
    suyo = [ref.sample_next_token(probs, random.Random(7)) for _ in range(1)]
    assert mio == suyo

    rng_mio, rng_suyo = random.Random(99), random.Random(99)
    assert [ej.sample_next_token(probs, rng_mio) for _ in range(50)] == [
        ref.sample_next_token(probs, rng_suyo) for _ in range(50)
    ]


# --------------------------------------------------- ejercicio 3: generate_naive


def test_devuelve_la_longitud_pedida():
    tabla = ref.build_count_table(TEXTO)
    salida = ej.generate_naive(tabla, "b", length=50, rng=random.Random(0))
    assert len(salida) == 50


def test_empieza_por_el_contexto_dado():
    tabla = ref.build_count_table(TEXTO, context_size=2)
    salida = ej.generate_naive(tabla, "ba", length=30, rng=random.Random(0))
    assert salida.startswith("ba")


def test_la_longitud_incluye_el_start():
    """start='ba' y length=5 devuelve 5 caracteres, no 7."""
    tabla = ref.build_count_table(TEXTO, context_size=2)
    assert len(ej.generate_naive(tabla, "ba", length=5, rng=random.Random(0))) == 5


def test_la_longitud_de_start_tiene_que_cuadrar_con_la_de_la_tabla():
    """Trampa clasica: tabla de contexto 1 y start de 2 caracteres.

    La tabla tiene claves de un caracter ('a', 'b'...), asi que buscar 'ba' no encuentra
    nada y la generacion para en el primer paso. No es un bug: es el comportamiento
    documentado ante un contexto desconocido. Pero conviene haberlo visto una vez.
    """
    tabla_de_1 = ref.build_count_table(TEXTO, context_size=1)
    assert ej.generate_naive(tabla_de_1, "ba", length=50, rng=random.Random(0)) == "ba"


def test_solo_genera_caracteres_que_existen_en_el_texto():
    tabla = ref.build_count_table(TEXTO)
    salida = ej.generate_naive(tabla, "b", length=300, rng=random.Random(3))
    assert set(salida) <= set(TEXTO)


def test_para_si_el_contexto_es_desconocido():
    """Con una tabla que solo conoce 'a'->'b', tras la 'b' no hay a donde ir."""
    tabla = {"a": {"b": 1}}
    salida = ej.generate_naive(tabla, "a", length=100, rng=random.Random(0))
    assert salida == "ab", f"deberia parar al llegar a un contexto desconocido, dio {salida!r}"


def test_el_texto_generado_es_identico_al_de_la_referencia():
    """Mismo texto, misma semilla, mismo resultado exacto."""
    tabla = ref.build_count_table(TEXTO, context_size=2)
    mio = ej.generate_naive(tabla, "ba", length=200, rng=random.Random(42))
    suyo = ref.generate_naive(tabla, "ba", length=200, rng=random.Random(42))
    assert mio == suyo


def test_funciona_con_contextos_de_mas_de_un_caracter():
    tabla = ref.build_count_table(TEXTO, context_size=3)
    salida = ej.generate_naive(tabla, "ban", length=60, rng=random.Random(5))
    assert len(salida) == 60 and salida.startswith("ban")


def test_con_contexto_mas_largo_el_texto_se_parece_mas_al_original():
    """El punto pedagogico del modulo: mas contexto, mejor imitacion.

    Se mide como fraccion de PALABRAS generadas que existen en el corpus. Contar
    trigramas no valdria: con contexto >= 2 saldria 100% por construccion, porque cada
    trigrama generado es contexto + siguiente, y eso ya estaba en la tabla.
    """
    palabras_reales = set(TEXTO.split())

    def realismo(context_size: int) -> float:
        tabla = ref.build_count_table(TEXTO, context_size=context_size)
        inicio = TEXTO[:context_size]
        salida = ej.generate_naive(tabla, inicio, length=600, rng=random.Random(11))
        generadas = salida.split()
        return sum(w in palabras_reales for w in generadas) / max(1, len(generadas))

    con_poco, con_mucho = realismo(1), realismo(4)
    assert con_mucho > con_poco, (
        f"con contexto de 4 el {con_mucho:.0%} de las palabras son reales y con contexto "
        f"de 1 el {con_poco:.0%}. Mirar mas atras tiene que ayudar."
    )
