"""Tests del modulo 14. Ejecutalos con `llmfs check 14`.

El test que mas importa es `test_la_cache_da_exactamente_la_misma_salida`.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import llmfs.reference as ref
from llmfs.config import ModelConfig
from llmfs.device import set_seed
from llmfs.testing import assert_close, load_exercises

ej = load_exercises(__file__)


def modelito(vocab: int = 64) -> torch.nn.Module:
    set_seed(0)
    return ref.GPT(
        ModelConfig(
            vocab_size=vocab, n_layers=2, d_model=32, n_heads=4, d_ff=96, context_length=64
        )
    ).eval()


# ------------------------------------------- ejercicio 1: apply_repetition_penalty


def test_los_logits_positivos_se_dividen():
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    salida = ej.apply_repetition_penalty(logits, torch.tensor([[0]]), penalty=2.0)
    assert abs(float(salida[0, 0]) - 1.5) < 1e-5, "un logit positivo tiene que DIVIDIRSE"


def test_los_logits_negativos_se_multiplican():
    """El detalle que casi todo el mundo implementa mal."""
    logits = torch.tensor([[-3.0, 2.0, 1.0]])
    salida = ej.apply_repetition_penalty(logits, torch.tensor([[0]]), penalty=2.0)
    assert abs(float(salida[0, 0]) - (-6.0)) < 1e-5, (
        "un logit negativo tiene que MULTIPLICARSE. Si lo divides, el token se vuelve MAS "
        "probable: justo lo contrario de penalizarlo."
    )


def test_penalizar_siempre_baja_la_probabilidad():
    """La propiedad que de verdad importa, sea cual sea el signo."""
    torch.manual_seed(0)
    logits = torch.randn(1, 20)
    generados = torch.tensor([[3, 7, 15]])
    antes = F.softmax(logits, dim=-1)
    despues = F.softmax(ej.apply_repetition_penalty(logits, generados, 1.5), dim=-1)
    for tok in (3, 7, 15):
        assert despues[0, tok] < antes[0, tok], f"el token {tok} no ha bajado"


def test_no_toca_los_tokens_que_no_han_salido():
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    salida = ej.apply_repetition_penalty(logits, torch.tensor([[0]]), penalty=2.0)
    assert abs(float(salida[0, 1]) - 2.0) < 1e-6
    assert abs(float(salida[0, 2]) - 1.0) < 1e-6


def test_penalty_uno_no_hace_nada():
    logits = torch.randn(2, 10)
    salida = ej.apply_repetition_penalty(logits, torch.tensor([[0, 1], [2, 3]]), penalty=1.0)
    assert_close(salida, logits, what="los logits con penalty=1.0")


def test_no_modifica_la_entrada():
    logits = torch.tensor([[3.0, 2.0]])
    copia = logits.clone()
    ej.apply_repetition_penalty(logits, torch.tensor([[0]]), penalty=2.0)
    assert_close(logits, copia, what="los logits originales")


def test_funciona_con_batch():
    logits = torch.randn(3, 10)
    salida = ej.apply_repetition_penalty(logits, torch.tensor([[0], [1], [2]]), 1.5)
    assert salida.shape == (3, 10)


def test_la_penalizacion_coincide_con_la_referencia():
    torch.manual_seed(0)
    logits, gen = torch.randn(2, 20), torch.tensor([[1, 5, 9], [2, 6, 10]])
    assert_close(
        ej.apply_repetition_penalty(logits, gen, 1.2),
        ref.apply_repetition_penalty(logits, gen, 1.2),
    )


# ----------------------------------------------------------- ejercicio 2: top_k_filter


def test_top_k_deja_solo_los_k_mayores():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, -1.0]])
    salida = ej.top_k_filter(logits, 2)
    assert float(salida[0, 0]) == 3.0 and float(salida[0, 1]) == 2.0
    assert all(math.isinf(float(salida[0, i])) for i in (2, 3, 4))


def test_top_k_conserva_el_umbral():
    """El k-esimo logit tiene que sobrevivir: usa < y no <=."""
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    assert float(ej.top_k_filter(logits, 2)[0, 1]) == 2.0


def test_top_k_con_valores_extremos_no_filtra():
    logits = torch.randn(1, 10)
    assert_close(ej.top_k_filter(logits, 0), logits, what="k=0")
    assert_close(ej.top_k_filter(logits, 100), logits, what="k mayor que el vocabulario")


def test_top_k_funciona_con_batch():
    logits = torch.randn(4, 50)
    salida = ej.top_k_filter(logits, 10)
    for fila in range(4):
        assert int(torch.isfinite(salida[fila]).sum()) == 10


def test_top_k_coincide_con_la_referencia():
    torch.manual_seed(0)
    logits = torch.randn(3, 100)
    assert_close(ej.top_k_filter(logits, 20), ref.top_k_filter(logits, 20))


# ----------------------------------------------------------- ejercicio 3: top_p_filter


def test_top_p_del_ejemplo_del_enunciado():
    """probs [0.60, 0.25, 0.10, 0.03, 0.02] con p=0.9 deja 3 candidatos.

    El token que CRUZA el umbral entra: la definicion de Holtzman es "el conjunto mas
    pequenyo cuya probabilidad acumulada EXCEDE p", y [0.60, 0.25] suma 0.85, que no
    excede 0.9. Hace falta el tercero para llegar a 0.95.
    """
    probs = torch.tensor([[0.60, 0.25, 0.10, 0.03, 0.02]])
    salida = ej.top_p_filter(torch.log(probs), 0.9)
    assert int(torch.isfinite(salida).sum()) == 3
    # Y los tres que quedan suman mas de 0.9
    supervivientes = probs[0][torch.isfinite(salida[0])]
    assert float(supervivientes.sum()) > 0.9


def test_top_p_se_queda_con_el_conjunto_MAS_PEQUENYO_que_excede_p():
    """No vale quedarse con mas de la cuenta: quitando el ultimo, ya no llegaria a p."""
    probs = torch.tensor([[0.60, 0.25, 0.10, 0.03, 0.02]])
    salida = ej.top_p_filter(torch.log(probs), 0.9)
    n = int(torch.isfinite(salida).sum())
    sin_el_ultimo = float(probs[0][: n - 1].sum())
    assert sin_el_ultimo <= 0.9, (
        f"con {n - 1} candidatos ya se llegaria a {sin_el_ultimo:.2f} > 0.9: sobra uno"
    )


def test_el_numero_de_candidatos_se_adapta():
    """Lo que distingue top-p de top-k."""
    seguro = torch.log(torch.tensor([[0.9, 0.05, 0.03, 0.02]]))
    indeciso = torch.log(torch.tensor([[0.25, 0.25, 0.25, 0.25]]))
    n_seguro = int(torch.isfinite(ej.top_p_filter(seguro, 0.9)).sum())
    n_indeciso = int(torch.isfinite(ej.top_p_filter(indeciso, 0.9)).sum())
    assert n_seguro < n_indeciso, (
        f"con el modelo seguro deja {n_seguro} candidatos y con el indeciso {n_indeciso}: "
        "el numero tiene que ADAPTARSE"
    )


def test_el_mas_probable_siempre_sobrevive():
    """Con p=0.5 y un token de probabilidad 0.9, sin esta guarda no quedaria ninguno."""
    logits = torch.log(torch.tensor([[0.9, 0.05, 0.05]]))
    salida = ej.top_p_filter(logits, 0.5)
    assert torch.isfinite(salida[0, 0]), "el token mas probable no puede filtrarse nunca"
    assert int(torch.isfinite(salida).sum()) >= 1


def test_top_p_uno_no_filtra_nada():
    logits = torch.randn(1, 10)
    assert_close(ej.top_p_filter(logits, 1.0), logits)


def test_top_p_muy_bajo_deja_solo_uno():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2]]))
    assert int(torch.isfinite(ej.top_p_filter(logits, 0.01)).sum()) == 1


def test_top_p_funciona_con_batch():
    torch.manual_seed(0)
    salida = ej.top_p_filter(torch.randn(4, 50), 0.9)
    assert salida.shape == (4, 50)
    assert all(int(torch.isfinite(salida[f]).sum()) >= 1 for f in range(4))


def test_top_p_coincide_con_la_referencia():
    torch.manual_seed(0)
    logits = torch.randn(3, 100)
    assert_close(ej.top_p_filter(logits, 0.9), ref.top_p_filter(logits, 0.9))


# ---------------------------------------------------------------- ejercicio 4: KVCache


def test_la_cache_arranca_vacia():
    cache = ej.KVCache(4)
    assert cache.seq_len == 0


def test_la_cache_acumula_en_la_dimension_de_tiempo():
    cache = ej.KVCache(2)
    k = torch.randn(1, 4, 3, 8)
    v = torch.randn(1, 4, 3, 8)
    K, V = cache.update(0, k, v)
    assert K.shape == (1, 4, 3, 8)

    k2, v2 = torch.randn(1, 4, 1, 8), torch.randn(1, 4, 1, 8)
    K, V = cache.update(0, k2, v2)
    assert K.shape == (1, 4, 4, 8), "tiene que concatenar en dim=-2 (el tiempo)"
    assert_close(K[:, :, :3], k, what="lo que ya estaba guardado")
    assert_close(K[:, :, 3:], k2, what="lo nuevo")


def test_las_capas_son_independientes():
    cache = ej.KVCache(3)
    cache.update(0, torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))
    cache.update(1, torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4))
    assert cache.keys[0].shape[-2] == 5
    assert cache.keys[1].shape[-2] == 2
    assert cache.keys[2] is None


def test_seq_len_refleja_lo_guardado():
    cache = ej.KVCache(2)
    for i in range(1, 4):
        cache.update(0, torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))
        assert cache.seq_len == i


def test_reset_la_vacia():
    cache = ej.KVCache(2)
    cache.update(0, torch.randn(1, 2, 5, 4), torch.randn(1, 2, 5, 4))
    cache.reset()
    assert cache.seq_len == 0 and cache.keys[0] is None


def test_memory_bytes_cuenta_lo_guardado():
    cache = ej.KVCache(1)
    cache.update(0, torch.randn(1, 2, 10, 8), torch.randn(1, 2, 10, 8))
    esperado = 2 * 1 * 2 * 10 * 8 * 4  # K y V, float32
    assert cache.memory_bytes() == esperado


# ------------------------------------------------------- ejercicio 5: generate_with_cache


def test_la_cache_da_exactamente_la_misma_salida():
    """EL test del modulo. No parecida: identica, token a token.

    Con temperature=0 la generacion es greedy y por tanto determinista, asi que las dos
    versiones tienen que coincidir exactamente.
    """
    modelo = modelito()
    prompt = torch.randint(0, 64, (1, 8))

    sin_cache = modelo.generate(prompt.clone(), 30, temperature=0.0)
    con_cache = ej.generate_with_cache(modelo, prompt.clone(), 30, temperature=0.0)

    assert torch.equal(sin_cache, con_cache), (
        f"divergen.\n  sin cache: {sin_cache[0, -10:].tolist()}\n"
        f"  con cache: {con_cache[0, -10:].tolist()}\n"
        "Lo primero que hay que mirar es el pos_offset de RoPE: al generar el token 50 se "
        "le pasa un tensor de longitud 1, y hay que rotarlo con el angulo de la posicion "
        "50, no de la 0."
    )


def test_la_forma_de_salida_es_la_esperada():
    modelo = modelito()
    salida = ej.generate_with_cache(modelo, torch.randint(0, 64, (1, 5)), 20)
    assert salida.shape == (1, 25)


def test_el_prompt_se_conserva_intacto():
    modelo = modelito()
    prompt = torch.randint(0, 64, (1, 6))
    salida = ej.generate_with_cache(modelo, prompt.clone(), 10)
    assert torch.equal(salida[:, :6], prompt)


def test_todos_los_tokens_estan_en_el_vocabulario():
    modelo = modelito()
    salida = ej.generate_with_cache(modelo, torch.randint(0, 64, (1, 4)), 40, top_k=10)
    assert int(salida.max()) < 64 and int(salida.min()) >= 0


def test_temperatura_cero_es_determinista():
    modelo = modelito()
    prompt = torch.randint(0, 64, (1, 5))
    a = ej.generate_with_cache(modelo, prompt.clone(), 15, temperature=0.0)
    b = ej.generate_with_cache(modelo, prompt.clone(), 15, temperature=0.0)
    assert torch.equal(a, b)


def test_los_filtros_se_aplican():
    """Con top_k=1 la generacion es determinista aunque la temperatura no sea cero."""
    modelo = modelito()
    prompt = torch.randint(0, 64, (1, 5))
    a = ej.generate_with_cache(modelo, prompt.clone(), 15, temperature=1.0, top_k=1)
    b = ej.generate_with_cache(modelo, prompt.clone(), 15, temperature=1.0, top_k=1)
    assert torch.equal(a, b), "con top_k=1 solo hay un candidato: tiene que ser determinista"


def test_para_al_encontrar_el_token_de_fin():
    modelo = modelito()
    prompt = torch.randint(0, 64, (1, 4))
    # Se fuerza el eos al token que saldria de todas formas con greedy
    normal = ej.generate_with_cache(modelo, prompt.clone(), 20, temperature=0.0)
    eos = int(normal[0, 4])
    con_eos = ej.generate_with_cache(modelo, prompt.clone(), 20, temperature=0.0, eos_token=eos)
    assert con_eos.shape[1] < normal.shape[1], "deberia haber parado antes"


def test_funciona_con_top_p():
    modelo = modelito()
    salida = ej.generate_with_cache(
        modelo, torch.randint(0, 64, (1, 4)), 20, temperature=0.8, top_p=0.9
    )
    assert salida.shape == (1, 24)


def test_para_al_llegar_al_limite_de_contexto():
    """Recortar con cache exigiria remapear las posiciones de RoPE: se para y punto."""
    modelo = modelito()  # context_length=64
    prompt = torch.randint(0, 64, (1, 50))
    salida = ej.generate_with_cache(modelo, prompt, 100, temperature=0.0)
    assert salida.shape[1] <= 64, (
        f"ha generado {salida.shape[1]} tokens con un contexto de 64"
    )


def test_un_prompt_que_ya_llena_el_contexto_es_un_error():
    modelo = modelito()
    with pytest.raises(ValueError):
        ej.generate_with_cache(modelo, torch.randint(0, 64, (1, 70)), 10)


def test_la_generacion_coincide_con_la_referencia():
    modelo = modelito()
    prompt = torch.randint(0, 64, (1, 6))
    mio = ej.generate_with_cache(modelo, prompt.clone(), 20, temperature=0.0)
    suyo = ref.generate_with_cache(modelo, prompt.clone(), 20, temperature=0.0)
    assert torch.equal(mio, suyo)
