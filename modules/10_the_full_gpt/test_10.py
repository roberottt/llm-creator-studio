"""Tests del modulo 10. Ejecutalos con `llmfs check 10`.

El test que mas importa es `test_el_modelo_final_tiene_8_933_440_parametros`.
"""

from __future__ import annotations

import math

import pytest
import torch

import llmfs.reference as ref
from llmfs.config import ModelConfig, RunConfig
from llmfs.paths import configs_dir
from llmfs.testing import assert_close, assert_scalar_close, copy_parameters, load_exercises

ej = load_exercises(__file__)


def cfg_pequeno(**kwargs) -> ModelConfig:
    """Un modelo diminuto para que los tests corran rapido."""
    base = dict(vocab_size=64, n_layers=2, d_model=32, n_heads=4, d_ff=96, context_length=16)
    base.update(kwargs)
    return ModelConfig(**base)


# ------------------------------------------------- ejercicio 1: expected_param_count


def test_el_modelo_final_tiene_8_933_440_parametros():
    """EL numero del curso. Si esto falla, el config o la formula mienten."""
    assert ej.expected_param_count(ModelConfig()) == 8_933_440


def test_el_desglose_del_teoria_cuadra_termino_a_termino():
    m = ModelConfig()
    embeddings = m.vocab_size * m.d_model
    atencion = 4 * m.d_model**2
    swiglu = 3 * m.d_model * m.d_ff
    normas = 2 * m.d_model
    por_capa = atencion + swiglu + normas

    assert embeddings == 1_310_720
    assert atencion == 409_600
    assert swiglu == 860_160
    assert por_capa == 1_270_400
    assert embeddings + m.n_layers * por_capa + m.d_model == ej.expected_param_count(m)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"n_layers": 1},
        {"tie_embeddings": False},
        {"norm": "layernorm"},
        {"activation": "gelu"},
        {"pos": "learned"},
        {"bias": True},
        {"vocab_size": 128, "d_model": 64, "n_heads": 8, "d_ff": 192},
    ],
)
def test_coincide_con_la_referencia_en_muchas_configuraciones(kwargs):
    cfg = cfg_pequeno(**kwargs)
    assert ej.expected_param_count(cfg) == ref.expected_param_count(cfg)


def test_sin_tying_hay_1_310_720_parametros_mas():
    con = ej.expected_param_count(ModelConfig(tie_embeddings=True))
    sin = ej.expected_param_count(ModelConfig(tie_embeddings=False))
    assert sin - con == 4096 * 320 == 1_310_720


def test_rope_no_anyade_ni_un_parametro():
    """Sus tablas son buffers calculados con una formula, no parametros."""
    con_rope = ej.expected_param_count(cfg_pequeno(pos="rope"))
    sin_posicion = ej.expected_param_count(cfg_pequeno(pos="none"))
    assert con_rope == sin_posicion


def test_las_posiciones_aprendidas_si_anyaden_parametros():
    aprendida = ej.expected_param_count(cfg_pequeno(pos="learned"))
    rope = ej.expected_param_count(cfg_pequeno(pos="rope"))
    assert aprendida - rope == 16 * 32  # context_length * d_model


def test_devuelve_un_entero():
    assert isinstance(ej.expected_param_count(ModelConfig()), int)


# ----------------------------------------------------- ejercicio 2: count_parameters


def test_el_conteo_real_coincide_con_la_formula():
    """Los dos ejercicios tienen que decir lo mismo. Es la comprobacion cruzada."""
    for kwargs in ({}, {"tie_embeddings": False}, {"norm": "layernorm"}, {"pos": "learned"}):
        cfg = cfg_pequeno(**kwargs)
        modelo = ref.GPT(cfg)
        assert ej.count_parameters(modelo)["total"] == ej.expected_param_count(cfg), (
            f"con {kwargs} la formula y el conteo no coinciden"
        )


def test_el_total_coincide_con_sum_de_parameters():
    modelo = ref.GPT(cfg_pequeno())
    assert ej.count_parameters(modelo)["total"] == sum(p.numel() for p in modelo.parameters())


def test_no_cuenta_dos_veces_los_pesos_atados():
    """Con weight tying, el mismo tensor aparece bajo dos nombres distintos.

    Dato util y poco conocido: `named_parameters()` DEDUPLICA por defecto
    (`remove_duplicate=True`), asi que sumar sobre el ya sale bien. Con
    `remove_duplicate=False` se ve el tensor repetido y el conteo ingenuo se dispara.
    """
    modelo = ref.GPT(cfg_pequeno(tie_embeddings=True))
    conteo = ej.count_parameters(modelo)

    con_duplicados = sum(
        p.numel() for _, p in modelo.named_parameters(remove_duplicate=False)
    )
    assert con_duplicados > conteo["total"], "el modelo de prueba deberia tener pesos atados"
    assert con_duplicados - conteo["total"] == 64 * 32, "el embedding contado dos veces"
    assert conteo["total"] == sum(p.numel() for p in modelo.parameters())


def test_tiene_todas_las_claves():
    conteo = ej.count_parameters(ref.GPT(cfg_pequeno()))
    for clave in ("embeddings", "attention", "ffn", "norms", "lm_head", "other",
                  "total", "non_embedding"):
        assert clave in conteo, f"falta la clave {clave!r}"


def test_non_embedding_es_total_menos_embeddings():
    conteo = ej.count_parameters(ref.GPT(cfg_pequeno()))
    assert conteo["non_embedding"] == conteo["total"] - conteo["embeddings"]


def test_el_desglose_del_modelo_final():
    """Los numeros que usara el modulo 12 para las leyes de escala."""
    conteo = ej.count_parameters(ref.GPT(ModelConfig()))
    assert conteo["embeddings"] == 1_310_720
    assert conteo["attention"] == 6 * 409_600
    assert conteo["ffn"] == 6 * 860_160
    assert conteo["total"] == 8_933_440
    assert conteo["non_embedding"] == 7_622_720


def test_con_tying_la_lm_head_no_aporta_nada():
    conteo = ej.count_parameters(ref.GPT(cfg_pequeno(tie_embeddings=True)))
    assert conteo["lm_head"] == 0


def test_coincide_con_la_referencia():
    modelo = ref.GPT(cfg_pequeno())
    assert ej.count_parameters(modelo) == ref.count_parameters(modelo)


# ---------------------------------------------------- ejercicio 3: TransformerBlock


def test_el_bloque_tiene_la_arquitectura_esperada():
    cfg = cfg_pequeno()
    copy_parameters(ref.TransformerBlock(cfg), ej.TransformerBlock(cfg))


def test_el_bloque_conserva_la_forma():
    bloque = ej.TransformerBlock(cfg_pequeno())
    assert bloque(torch.randn(2, 8, 32)).shape == (2, 8, 32)


def test_el_bloque_coincide_con_la_referencia():
    torch.manual_seed(0)
    cfg = cfg_pequeno()
    mio, suyo = ej.TransformerBlock(cfg), ref.TransformerBlock(cfg)
    copy_parameters(suyo, mio)
    mio.eval()
    suyo.eval()
    x = torch.randn(2, 8, 32)
    cos, sin = ref.rope_frequencies(cfg.head_dim, 16)
    assert_close(mio(x, cos=cos, sin=sin), suyo(x, cos=cos, sin=sin), what="la salida del bloque")


def test_el_bloque_usa_residuales():
    """Con los pesos de salida a cero, la salida tiene que ser exactamente la entrada."""
    torch.manual_seed(0)
    bloque = ej.TransformerBlock(cfg_pequeno())
    bloque.eval()
    with torch.no_grad():
        for nombre, p in bloque.named_parameters():
            if nombre.endswith(("out_proj.weight", "down_proj.weight", "fc_out.weight")):
                p.zero_()
    x = torch.randn(2, 8, 32)
    assert_close(bloque(x), x, atol=1e-5, what="la salida con las ramas anuladas")


def test_el_bloque_tiene_los_parametros_del_desglose():
    """1.270.400 por capa con el config final."""
    bloque = ej.TransformerBlock(ModelConfig())
    assert sum(p.numel() for p in bloque.parameters()) == 1_270_400


# ------------------------------------------------------------- ejercicio 4: GPT


def test_el_gpt_completo_tiene_8_933_440_parametros():
    """El test que cierra la Parte II."""
    modelo = ej.GPT(ModelConfig())
    assert sum(p.numel() for p in modelo.parameters()) == 8_933_440


def test_el_gpt_devuelve_las_formas_correctas():
    modelo = ej.GPT(cfg_pequeno())
    idx = torch.randint(0, 64, (2, 8))
    logits, perdida = modelo(idx, idx)
    assert logits.shape == (2, 8, 64)
    assert perdida is not None and perdida.ndim == 0


def test_sin_targets_no_hay_perdida():
    modelo = ej.GPT(cfg_pequeno())
    _, perdida = modelo(torch.randint(0, 64, (2, 8)))
    assert perdida is None


def test_los_pesos_atados_son_el_mismo_tensor():
    modelo = ej.GPT(cfg_pequeno(tie_embeddings=True))
    assert modelo.lm_head.weight is modelo.token_embedding.weight, (
        "tienen que ser el MISMO objeto, no copias con los mismos valores"
    )


def test_sin_tying_son_tensores_distintos():
    modelo = ej.GPT(cfg_pequeno(tie_embeddings=False))
    assert modelo.lm_head.weight is not modelo.token_embedding.weight


def test_la_perdida_inicial_vale_ln_del_vocabulario():
    """El detector de bugs del modulo 05, aplicado al modelo de verdad.

    Si sale mucho mas alta, la inicializacion es demasiado agresiva.
    Si sale mas baja, hay fuga de informacion (mascara causal mal puesta).

    OJO con como se construye el batch: los targets tienen que ir DESPLAZADOS un token.
    Si pasas `modelo(idx, idx)`, en la posicion t el modelo ve el token idx[t] y le pides
    que prediga idx[t]: puede leerlo de su propia entrada. Con weight tying la fuga es
    todavia mas directa, porque los logits son `x @ W_emb.T` y el producto de un embedding
    consigo mismo es grande. La perdida sale por debajo de ln(V) y parece un bug del
    modelo cuando es un bug del test.
    """
    torch.manual_seed(0)
    cfg = cfg_pequeno(vocab_size=256)
    modelo = ej.GPT(cfg)
    modelo.eval()
    secuencia = torch.randint(0, 256, (8, 17))
    x, y = secuencia[:, :-1], secuencia[:, 1:]
    with torch.no_grad():
        _, perdida = modelo(x, y)

    esperado = math.log(256)
    assert abs(float(perdida) - esperado) < 0.15, (
        f"la perdida inicial es {float(perdida):.4f} y deberia rondar ln(256)={esperado:.4f}. "
        "Mas alta: init demasiado agresiva. Mas baja: fuga de informacion."
    )


def test_el_modelo_es_causal():
    """Cambiar un token NO puede afectar a las predicciones anteriores."""
    torch.manual_seed(0)
    modelo = ej.GPT(cfg_pequeno())
    modelo.eval()
    idx = torch.randint(0, 64, (1, 10))

    with torch.no_grad():
        original = modelo(idx)[0]
        modificado = idx.clone()
        modificado[0, 7] = (modificado[0, 7] + 1) % 64
        alterado = modelo(modificado)[0]

    assert_close(alterado[:, :7], original[:, :7], atol=1e-5,
                 what="los logits anteriores al token cambiado")
    assert not torch.allclose(alterado[:, 7:], original[:, 7:], atol=1e-5), (
        "cambiar el token 7 deberia afectar a las predicciones de la 7 en adelante"
    )


def test_la_inicializacion_escalada_por_profundidad_esta_aplicada():
    """Las proyecciones que escriben en el residual arrancan mas pequenyas."""
    torch.manual_seed(0)
    cfg = cfg_pequeno(n_layers=8, d_model=128, n_heads=8, d_ff=256)
    modelo = ej.GPT(cfg)

    residuales, normales = [], []
    for nombre, p in modelo.named_parameters():
        if nombre.endswith(("out_proj.weight", "down_proj.weight")):
            residuales.append(float(p.std()))
        elif nombre.endswith(("q_proj.weight", "gate_proj.weight")):
            normales.append(float(p.std()))

    std_residual = sum(residuales) / len(residuales)
    std_normal = sum(normales) / len(normales)
    esperado = 0.02 / math.sqrt(2 * cfg.n_layers)

    assert abs(std_residual - esperado) < 0.002, (
        f"las proyecciones residuales tienen std {std_residual:.5f} y deberia ser "
        f"0.02/sqrt(2*{cfg.n_layers}) = {esperado:.5f}"
    )
    assert std_normal > std_residual * 2, "el resto deberia arrancar con std=0.02"


def test_rope_se_guarda_como_buffer_no_persistente():
    modelo = ej.GPT(cfg_pequeno(pos="rope"))
    nombres_buffer = [n for n, _ in modelo.named_buffers()]
    assert any("rope" in n for n in nombres_buffer), "las tablas de RoPE deben ser buffers"
    assert not any("rope" in n for n in modelo.state_dict()), (
        "con persistent=False no deben aparecer en el state_dict: se recalculan al construir"
    )


def test_una_secuencia_mas_larga_que_el_contexto_es_un_error():
    modelo = ej.GPT(cfg_pequeno(context_length=16))
    with pytest.raises(ValueError):
        modelo(torch.randint(0, 64, (1, 32)))


def test_el_gpt_coincide_con_la_referencia():
    torch.manual_seed(0)
    cfg = cfg_pequeno()
    mio, suyo = ej.GPT(cfg), ref.GPT(cfg)
    copy_parameters(suyo, mio)
    mio.eval()
    suyo.eval()
    idx = torch.randint(0, 64, (2, 8))
    logits_mio, perdida_mia = mio(idx, idx)
    logits_suyo, perdida_suya = suyo(idx, idx)
    assert_close(logits_mio, logits_suyo, atol=1e-5, what="los logits")
    assert_scalar_close(perdida_mia, perdida_suya, what="la perdida")


def test_el_modelo_puede_aprender():
    """Overfit a un batch diminuto. Si esto no baja, algo esta roto."""
    torch.manual_seed(0)
    cfg = cfg_pequeno()
    modelo = ej.GPT(cfg)
    opt = torch.optim.AdamW(modelo.parameters(), lr=3e-3)
    idx = torch.randint(0, 64, (4, 8))

    inicial = None
    for _ in range(120):
        _, perdida = modelo(idx, idx)
        inicial = inicial if inicial is not None else float(perdida)
        opt.zero_grad(set_to_none=True)
        perdida.backward()
        opt.step()

    assert float(perdida) < inicial * 0.2, (
        f"la perdida ha pasado de {inicial:.4f} a {float(perdida):.4f} memorizando un solo "
        "batch: deberia bajar mucho mas"
    )


def test_el_config_yaml_del_repo_produce_el_modelo_esperado():
    """De punta a punta: el fichero de configs da los 8.933.440."""
    cfg = RunConfig.from_yaml(configs_dir() / "tinystories_9m.yaml")
    assert ej.expected_param_count(cfg.model) == 8_933_440
