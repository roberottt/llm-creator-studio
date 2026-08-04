"""Tests del modulo 03. Ejecutalos con `llmfs check 03`."""

from __future__ import annotations

import pytest

import llmfs.reference as ref
from llmfs.reference import GPT4_SPLIT_PATTERN
from llmfs.testing import load_exercises

ej = load_exercises(__file__)

# Texto con suficiente variedad para que quepan al menos 90 merges sin agotar los pares.
TEXTO = (
    "el gato negro come pescado fresco por la manana. "
    "la nina pequena mira al perro blanco desde la ventana grande. "
    "el perro corre rapido hacia el jardin verde y ladra mucho. "
    "manana volveremos a la playa con los ninos y sus juguetes. "
    "un pajaro azul canta sobre la rama mas alta del arbol viejo. "
) * 12
UNICODE = "ñandú 🦙 café über 日本語 naïve"


# ----------------------------------------------------------------- ejercicio 1: get_stats


def test_cuenta_pares_del_ejemplo_del_enunciado():
    assert ej.get_stats([97, 97, 97, 98]) == {(97, 97): 2, (97, 98): 1}


def test_los_pares_se_solapan_al_contar():
    """En [1,1,1,1] el par (1,1) sale 3 veces, no 2."""
    assert ej.get_stats([1, 1, 1, 1]) == {(1, 1): 3}


def test_una_secuencia_de_un_elemento_no_tiene_pares():
    assert ej.get_stats([5]) == {}
    assert ej.get_stats([]) == {}


def test_acumula_sobre_el_diccionario_que_le_pasas():
    """Es lo que permite a train_bpe sumar varios chunks sin concatenarlos."""
    acumulador: dict[tuple[int, int], int] = {}
    ej.get_stats([1, 2], acumulador)
    ej.get_stats([1, 2, 3], acumulador)
    assert acumulador == {(1, 2): 2, (2, 3): 1}


def test_devuelve_el_mismo_diccionario_que_recibe():
    acumulador: dict[tuple[int, int], int] = {}
    assert ej.get_stats([1, 2], acumulador) is acumulador


def test_coincide_con_la_referencia():
    ids = list(TEXTO.encode("utf-8"))
    assert ej.get_stats(ids) == ref.get_stats(ids)


# -------------------------------------------------------------------- ejercicio 2: merge


def test_fusiona_el_ejemplo_del_enunciado():
    assert ej.merge([97, 97, 97, 98, 97, 97], (97, 97), 256) == [256, 97, 98, 256]


def test_al_fusionar_no_se_solapa():
    """[1,1,1] fusionando (1,1) da [256,1], no [256,256]."""
    assert ej.merge([1, 1, 1], (1, 1), 256) == [256, 1]


def test_no_toca_nada_si_el_par_no_aparece():
    assert ej.merge([1, 2, 3], (7, 8), 256) == [1, 2, 3]


def test_fusiona_un_par_al_final_de_la_lista():
    assert ej.merge([1, 2, 3, 4], (3, 4), 256) == [1, 2, 256]


def test_devuelve_una_lista_nueva_sin_modificar_la_original():
    original = [1, 1, 2]
    resultado = ej.merge(original, (1, 1), 256)
    assert original == [1, 1, 2], "has modificado la lista de entrada"
    assert resultado is not original


def test_coincide_con_la_referencia_sobre_texto_real():
    ids = list(TEXTO.encode("utf-8"))
    par = max(ref.get_stats(ids), key=lambda p: (ref.get_stats(ids)[p], p))
    assert ej.merge(ids, par, 256) == ref.merge(ids, par, 256)


# ---------------------------------------------------------------- ejercicio 3: train_bpe


def test_el_ejemplo_de_la_teoria_paso_a_paso():
    """'aaabdaaabac' con 2 merges: (a,a)->256 y (256,a)->257."""
    merges, vocab = ej.train_bpe("aaabdaaabac", 258)
    assert merges == {(97, 97): 256, (256, 97): 257}
    assert vocab[256] == b"aa"
    assert vocab[257] == b"aaa"


def test_el_vocabulario_arranca_con_los_256_bytes():
    _, vocab = ej.train_bpe(TEXTO, 300)
    for i in range(256):
        assert vocab[i] == bytes([i]), f"falta el byte {i} en el vocabulario"


def test_produce_el_numero_de_merges_pedido():
    merges, vocab = ej.train_bpe(TEXTO, 300)
    assert len(merges) == 300 - 256
    assert len(vocab) == 300


def test_los_ids_nuevos_empiezan_en_256_y_son_consecutivos():
    merges, _ = ej.train_bpe(TEXTO, 290)
    assert list(merges.values()) == list(range(256, 290))


def test_un_vocabulario_menor_que_256_es_un_error():
    with pytest.raises(ValueError):
        ej.train_bpe(TEXTO, 100)


def test_para_si_se_queda_sin_pares_que_fusionar():
    """Con un texto minusculo no hay 1000 merges posibles: hay que parar, no reventar."""
    merges, vocab = ej.train_bpe("abc", 1000)
    assert len(merges) < 744
    assert len(vocab) == 256 + len(merges)


def test_los_merges_coinciden_exactamente_con_la_referencia():
    assert ej.train_bpe(TEXTO, 320)[0] == ref.train_bpe(TEXTO, 320)[0]


def test_tambien_coincide_usando_el_pre_tokenizador():
    mios = ej.train_bpe(TEXTO, 320, pattern=GPT4_SPLIT_PATTERN)[0]
    suyos = ref.train_bpe(TEXTO, 320, pattern=GPT4_SPLIT_PATTERN)[0]
    assert mios == suyos


def test_el_pre_tokenizador_impide_merges_a_traves_de_los_espacios():
    """Sin patron, BPE aprende 'o e' uniendo el final de una palabra con la siguiente."""
    _, vocab_con = ej.train_bpe(TEXTO, 340, pattern=GPT4_SPLIT_PATTERN)
    tokens_con = [vocab_con[i] for i in range(256, 340)]
    # Un token con un espacio EN MEDIO significa que el merge cruzo una frontera.
    for token in tokens_con:
        texto_token = token.decode("utf-8", errors="replace")
        assert " " not in texto_token.strip(), (
            f"el token {texto_token!r} contiene un espacio interior: el pre-tokenizador "
            "no se esta aplicando"
        )


def test_el_vocabulario_es_coherente_con_los_merges():
    """vocab[nuevo] tiene que ser la concatenacion de los bytes de sus dos padres."""
    merges, vocab = ej.train_bpe(TEXTO, 300)
    for (a, b), nuevo in merges.items():
        assert vocab[nuevo] == vocab[a] + vocab[b]


# --------------------------------------------------------- ejercicios 4 y 5: encode/decode


def test_codifica_el_ejemplo_de_la_teoria():
    merges, _ = ref.train_bpe("aaabdaaabac", 258)
    assert ej.bpe_encode("aaabdaaabac", merges) == [257, 98, 100, 257, 98, 97, 99]


def test_codificar_y_decodificar_devuelve_el_texto_original():
    merges, vocab = ref.train_bpe(TEXTO, 400)
    assert ej.bpe_decode(ej.bpe_encode(TEXTO, merges), vocab) == TEXTO


def test_el_roundtrip_funciona_con_texto_que_no_se_uso_al_entrenar():
    merges, vocab = ref.train_bpe(TEXTO, 400)
    nuevo = "un texto completamente distinto, con numeros 12345 y signos !?"
    assert ej.bpe_decode(ej.bpe_encode(nuevo, merges), vocab) == nuevo


def test_el_roundtrip_sobrevive_a_unicode():
    """Emoji, acentos y japones. Aqui es donde se ve por que se trabaja con bytes."""
    merges, vocab = ref.train_bpe(TEXTO, 400)
    assert ej.bpe_decode(ej.bpe_encode(UNICODE, merges), vocab) == UNICODE


def test_el_roundtrip_funciona_con_el_pre_tokenizador():
    merges, vocab = ref.train_bpe(TEXTO, 400, pattern=GPT4_SPLIT_PATTERN)
    ids = ej.bpe_encode(UNICODE, merges, pattern=GPT4_SPLIT_PATTERN)
    assert ej.bpe_decode(ids, vocab) == UNICODE


def test_codificar_comprime_de_verdad():
    merges, _ = ref.train_bpe(TEXTO, 400)
    ids = ej.bpe_encode(TEXTO, merges)
    assert len(ids) < len(TEXTO.encode("utf-8")) / 2, (
        "con 144 merges sobre un texto muy repetitivo deberia comprimir al menos a la mitad"
    )


def test_la_codificacion_coincide_con_la_referencia():
    merges, _ = ref.train_bpe(TEXTO, 400)
    assert ej.bpe_encode(TEXTO, merges) == ref.bpe_encode(TEXTO, merges)


def test_los_merges_se_aplican_en_el_orden_en_que_se_aprendieron():
    """El caso que distingue 'aplicar en orden de aprendizaje' de 'aplicar el que pille'.

    merges = {(a,b): 256, (a,a): 257}  sobre el texto "aab".

    Ambos pares estan presentes. Como (a,b) se aprendio ANTES (id 256 < 257), es el que
    hay que aplicar primero:

        "aab" -> [a, 256]

    Si aplicaras (a,a) por ser el primero que encuentras, o por frecuencia, saldria
    [257, b]. Las dos tokenizaciones tienen dos tokens y las dos son "validas", pero solo
    una es la que vio el modelo al entrenar. La otra no la entiende.
    """
    merges = {(97, 98): 256, (97, 97): 257}
    assert ej.bpe_encode("aab", merges) == [97, 256]


def test_un_merge_consume_todas_sus_apariciones_antes_de_pasar_al_siguiente():
    """Consecuencia sutil del algoritmo, y merece la pena verla una vez.

    Con merges (a,a)->256 y (256,a)->257, la cadena "aaaa" NO da [257, a].

    Da [256, 256]: el primer merge se aplica a TODA la secuencia de golpe y se lleva las
    cuatro 'a' de dos en dos, asi que el par (256, a) nunca llega a existir. Con tres 'a'
    si saldria [257], porque quedaria [256, a] y ahi el segundo merge si aplica.
    """
    merges = {(97, 97): 256, (256, 97): 257}
    assert ej.bpe_encode("aaaa", merges) == [256, 256]
    assert ej.bpe_encode("aaa", merges) == [257]


def test_decodificar_no_rompe_ante_bytes_invalidos():
    """El bytes-fallback: un modelo a medio entrenar genera esto constantemente."""
    vocab = {i: bytes([i]) for i in range(256)}
    resultado = ej.bpe_decode([0xC3], vocab)  # medio caracter multibyte
    assert isinstance(resultado, str), "no debe lanzar excepcion, debe usar errors='replace'"


def test_decodificar_junta_los_bytes_antes_de_decodificar():
    """Dos tokens que juntos forman una 'n' con tilde, pero por separado no son validos."""
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256], vocab[257] = b"\xc3", b"\xb1"
    assert ej.bpe_decode([256, 257], vocab) == "ñ"


def test_una_lista_vacia_decodifica_a_cadena_vacia():
    assert ej.bpe_decode([], {i: bytes([i]) for i in range(256)}) == ""
