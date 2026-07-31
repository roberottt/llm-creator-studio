"""Tests del modulo 16. Ejecutalos con `llmfs check 16`."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import llmfs.reference as ref
from llmfs.config import ModelConfig
from llmfs.device import set_seed
from llmfs.testing import assert_close, load_exercises

ej = load_exercises(__file__)


# ------------------------------------------------------ ejercicio 1: build_chat_template


def test_el_ejemplo_del_enunciado():
    msgs = [{"role": "user", "content": "Hola"}, {"role": "assistant", "content": "Que tal"}]
    assert ej.build_chat_template(msgs) == "<|user|>Hola<|end|><|assistant|>Que tal<|end|>"


def test_el_generation_prompt_deja_la_cadena_abierta():
    msgs = [{"role": "user", "content": "Hola"}]
    salida = ej.build_chat_template(msgs, add_generation_prompt=True)
    assert salida == "<|user|>Hola<|end|><|assistant|>"
    assert not salida.endswith("<|end|>"), "tiene que quedar ABIERTA para que el modelo siga"


def test_admite_mensaje_de_sistema():
    msgs = [{"role": "system", "content": "Se breve"}, {"role": "user", "content": "Hola"}]
    assert ej.build_chat_template(msgs).startswith("<|system|>Se breve<|end|>")


def test_varios_turnos_se_encadenan():
    msgs = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
    ]
    salida = ej.build_chat_template(msgs)
    assert salida.count("<|user|>") == 2 and salida.count("<|assistant|>") == 1


def test_cada_mensaje_se_cierra():
    """El <|end|> es lo que le enseña al modelo cuando parar."""
    msgs = [{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}]
    assert ej.build_chat_template(msgs).count("<|end|>") == 2


def test_un_rol_desconocido_es_un_error():
    with pytest.raises(ValueError):
        ej.build_chat_template([{"role": "robot", "content": "beep"}])


def test_una_conversacion_vacia_da_cadena_vacia():
    assert ej.build_chat_template([]) == ""


def test_el_template_coincide_con_la_referencia():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]
    assert ej.build_chat_template(msgs) == ref.build_chat_template(msgs)
    assert ej.build_chat_template(msgs, True) == ref.build_chat_template(msgs, True)


# ------------------------------------------------------- ejercicio 2: mask_prompt_tokens


def test_el_ejemplo_del_enunciado_con_su_off_by_one():
    """DOS posiciones ignoradas al principio con prompt_len=3, no tres."""
    assert ej.mask_prompt_tokens([10, 11, 12, 20, 21, 22], 3) == [-100, -100, 20, 21, 22, -100]


def test_la_ultima_posicion_del_prompt_si_aprende():
    """Es la transicion pregunta->respuesta: lo mas importante que tiene que aprender."""
    targets = ej.mask_prompt_tokens([10, 11, 12, 20, 21], 3)
    assert targets[2] == 20, (
        "en la ultima posicion del prompt el objetivo ya es el primer token de la "
        "respuesta, y ESE si interesa"
    )


def test_la_ultima_posicion_siempre_se_ignora():
    """No hay input_ids[i+1] que predecir."""
    assert ej.mask_prompt_tokens([1, 2, 3, 4], 2)[-1] == -100


def test_los_targets_van_desplazados_un_token():
    ids = [5, 6, 7, 8, 9]
    targets = ej.mask_prompt_tokens(ids, 1)
    for i in range(len(ids) - 1):
        assert targets[i] == ids[i + 1]


def test_la_longitud_se_conserva():
    for n in (2, 5, 10):
        assert len(ej.mask_prompt_tokens(list(range(n)), 1)) == n


def test_con_prompt_de_uno_casi_todo_aprende():
    targets = ej.mask_prompt_tokens([1, 2, 3, 4], 1)
    assert targets.count(-100) == 1  # solo la ultima


def test_admite_otro_ignore_index():
    targets = ej.mask_prompt_tokens([1, 2, 3, 4], 2, ignore_index=-1)
    assert targets[0] == -1 and -100 not in targets


def test_prompt_len_invalido_es_un_error():
    for malo in (0, -1, 10):
        with pytest.raises(ValueError):
            ej.mask_prompt_tokens([1, 2, 3], malo)


def test_cross_entropy_ignora_las_posiciones_marcadas():
    """La comprobacion de que sirve para lo que sirve."""
    logits = torch.randn(5, 10)
    targets = torch.tensor(ej.mask_prompt_tokens([1, 2, 3, 4, 5], 3))
    perdida = F.cross_entropy(logits, targets, ignore_index=-100)
    assert torch.isfinite(perdida), "-100 tiene que ser ignorado, no indexado"


def test_la_mascara_coincide_con_la_referencia():
    ids = [10, 11, 12, 20, 21, 22, 23]
    assert ej.mask_prompt_tokens(ids, 3) == ref.mask_prompt_tokens(ids, 3)


# ------------------------------------------------------------- ejercicio 3: LoRALinear


def test_al_inicializar_la_salida_es_identica_a_la_base():
    """LA propiedad de LoRA: arranca sin perturbar nada, gracias a que B empieza en ceros."""
    set_seed(0)
    base = nn.Linear(64, 64)
    lora = ej.LoRALinear(base, r=8)
    x = torch.randn(4, 64)
    assert_close(lora(x), base(x), atol=1e-6, what="la salida al inicializar")


def test_lora_b_arranca_en_ceros():
    lora = ej.LoRALinear(nn.Linear(32, 32), r=4)
    assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))


def test_lora_a_no_arranca_en_ceros():
    """Si las dos fueran cero, el gradiente de ambas seria cero y nunca aprenderian."""
    lora = ej.LoRALinear(nn.Linear(32, 32), r=4)
    assert not torch.allclose(lora.lora_A, torch.zeros_like(lora.lora_A))


def test_la_capa_base_esta_congelada():
    lora = ej.LoRALinear(nn.Linear(32, 32), r=4)
    assert not lora.base.weight.requires_grad, "la base tiene que congelarse: es el punto de LoRA"
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad


def test_las_formas_de_los_adaptadores():
    lora = ej.LoRALinear(nn.Linear(64, 128), r=8)
    assert lora.lora_A.shape == (8, 64)
    assert lora.lora_B.shape == (128, 8)


def test_solo_se_entrena_una_fraccion_de_los_parametros():
    lora = ej.LoRALinear(nn.Linear(320, 320), r=8)
    entrenables = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora.parameters())
    assert entrenables == 2 * 8 * 320
    assert entrenables / total < 0.06, f"deberia entrenar ~5%, entrena {entrenables/total:.1%}"


def test_la_salida_cambia_cuando_b_deja_de_ser_cero():
    set_seed(0)
    base = nn.Linear(32, 32)
    lora = ej.LoRALinear(base, r=4)
    x = torch.randn(2, 32)
    antes = lora(x).clone()
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)
    assert not torch.allclose(lora(x), antes, atol=1e-4)


def test_la_forma_de_salida_es_la_de_la_base():
    lora = ej.LoRALinear(nn.Linear(64, 128), r=8)
    assert lora(torch.randn(4, 64)).shape == (4, 128)


def test_un_rango_no_positivo_es_un_error():
    for malo in (0, -1):
        with pytest.raises(ValueError):
            ej.LoRALinear(nn.Linear(32, 32), r=malo)


def test_lora_coincide_con_la_referencia():
    set_seed(0)
    base = nn.Linear(64, 64)
    mio, suyo = ej.LoRALinear(base, r=8, alpha=16.0), ref.LoRALinear(base, r=8, alpha=16.0)
    with torch.no_grad():
        suyo.lora_A.copy_(mio.lora_A)
        suyo.lora_B.copy_(mio.lora_B)
        mio.lora_B.normal_(0, 0.1)
        suyo.lora_B.copy_(mio.lora_B)
    mio.eval()
    suyo.eval()
    x = torch.randn(4, 64)
    assert_close(mio(x), suyo(x), what="la salida de LoRA")


# -------------------------------------------------------- ejercicio 4: merge_lora_weights


def test_la_capa_fundida_da_la_misma_salida():
    """LA propiedad que hace util a LoRA: la adaptacion se absorbe sin aproximar nada."""
    set_seed(0)
    lora = ej.LoRALinear(nn.Linear(64, 64), r=8)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)
    lora.eval()

    fundida = ej.merge_lora_weights(lora)
    x = torch.randn(4, 64)
    assert_close(fundida(x), lora(x), atol=1e-5, what="la salida de la capa fundida")


def test_devuelve_un_linear_normal():
    fundida = ej.merge_lora_weights(ej.LoRALinear(nn.Linear(32, 64), r=4))
    assert isinstance(fundida, nn.Linear)
    assert not isinstance(fundida, ej.LoRALinear)
    assert fundida.in_features == 32 and fundida.out_features == 64


def test_sin_adaptar_los_pesos_no_cambian():
    """Con B en ceros, W_nueva == W."""
    set_seed(0)
    base = nn.Linear(32, 32)
    fundida = ej.merge_lora_weights(ej.LoRALinear(base, r=4))
    assert_close(fundida.weight, base.weight, atol=1e-6, what="los pesos sin adaptar")


def test_conserva_el_sesgo():
    base = nn.Linear(32, 32, bias=True)
    fundida = ej.merge_lora_weights(ej.LoRALinear(base, r=4))
    assert fundida.bias is not None
    assert_close(fundida.bias, base.bias, what="el sesgo")


def test_funciona_sin_sesgo():
    fundida = ej.merge_lora_weights(ej.LoRALinear(nn.Linear(32, 32, bias=False), r=4))
    assert fundida.bias is None


def test_los_pesos_fundidos_son_la_suma_esperada():
    set_seed(0)
    base = nn.Linear(32, 32)
    lora = ej.LoRALinear(base, r=4, alpha=8.0)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)
    esperado = base.weight + (lora.lora_B @ lora.lora_A) * (8.0 / 4)
    assert_close(ej.merge_lora_weights(lora).weight, esperado, atol=1e-5)


def test_la_fusion_coincide_con_la_referencia():
    set_seed(0)
    lora = ej.LoRALinear(nn.Linear(64, 64), r=8)
    with torch.no_grad():
        lora.lora_B.normal_(0, 0.1)
    mio = ej.merge_lora_weights(lora)
    suyo = ref.merge_lora_weights(lora)
    assert_close(mio.weight, suyo.weight, what="los pesos fundidos")
