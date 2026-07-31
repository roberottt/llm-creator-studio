"""La configuracion tiene que validar pronto y cuadrar con el modelo de 9M."""

from __future__ import annotations

import pytest
import yaml

from llmfs.config import ModelConfig, RunConfig, TrainConfig


def test_los_valores_por_defecto_son_el_modelo_final():
    m = ModelConfig()
    assert (m.vocab_size, m.n_layers, m.d_model, m.n_heads) == (4096, 6, 320, 8)
    assert m.head_dim == 40
    assert m.d_ff == 896
    assert m.context_length == 512
    assert m.tie_embeddings is True


def test_d_model_debe_dividirse_entre_las_cabezas():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=320, n_heads=7)


def test_rope_exige_head_dim_par():
    with pytest.raises(ValueError, match="par"):
        ModelConfig(d_model=15, n_heads=5, pos="rope")


def test_dropout_fuera_de_rango():
    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(dropout=1.5)


def test_betas_debe_tener_dos_elementos():
    with pytest.raises(ValueError, match="betas"):
        TrainConfig(betas=(0.9, 0.95, 0.99))  # type: ignore[arg-type]


def test_yaml_plano_y_por_secciones_son_equivalentes():
    plano = {"n_layers": 4, "d_model": 128, "n_heads": 4, "lr": 3.0e-4, "batch_size": 16}
    anidado = {
        "model": {"n_layers": 4, "d_model": 128, "n_heads": 4},
        "train": {"lr": 3.0e-4, "batch_size": 16},
    }
    assert RunConfig.from_dict(plano) == RunConfig.from_dict(anidado)


def test_una_clave_desconocida_revienta_al_arrancar():
    """Un typo en el config no puede descubrirse tres horas despues de empezar."""
    with pytest.raises(ValueError) as exc:
        RunConfig.from_dict({"n_layerz": 6})
    assert "n_layerz" in str(exc.value)
    assert "n_layers" in str(exc.value), "el error deberia listar las claves validas"


def test_yaml_admite_guiones_bajos_en_los_enteros():
    """`max_tokens: 500_000_000` tiene que parsearse como int, no como string."""
    cargado = yaml.safe_load("max_tokens: 500_000_000\n")
    assert cargado["max_tokens"] == 500_000_000


def test_la_aritmetica_de_pasos_y_tokens():
    cfg = RunConfig()
    assert cfg.tokens_per_step == 48 * 2 * 512 == 49_152
    assert cfg.max_steps == 500_000_000 // 49_152 == 10_172


def test_roundtrip_por_yaml(tmp_path):
    cfg = RunConfig(name="prueba")
    destino = tmp_path / "cfg.yaml"
    cfg.to_yaml(destino)
    assert RunConfig.from_yaml(destino) == cfg


def test_el_nombre_sale_del_fichero_si_no_se_indica(tmp_path):
    destino = tmp_path / "tinystories_9m.yaml"
    destino.write_text("n_layers: 6\n", encoding="utf-8")
    assert RunConfig.from_yaml(destino).name == "tinystories_9m"


def test_tokenizador_de_caracteres_con_vocabulario_absurdo():
    with pytest.raises(ValueError, match="caracteres"):
        RunConfig.from_dict({"tokenizer": "char", "vocab_size": 4096})


def test_los_configs_reales_del_repo_cargan():
    from llmfs.paths import configs_dir

    encontrados = sorted(p.name for p in configs_dir().glob("*.yaml"))
    assert encontrados == ["tiny_char.yaml", "tinystories_9m.yaml"]
    for path in configs_dir().glob("*.yaml"):
        RunConfig.from_yaml(path)


def test_el_config_de_9m_cuadra_con_el_desglose_documentado():
    """El conteo que promete el modulo 09. Si esto cambia, el TEORIA.md miente."""
    from llmfs.paths import configs_dir

    cfg = RunConfig.from_yaml(configs_dir() / "tinystories_9m.yaml")
    m = cfg.model

    embeddings = m.vocab_size * m.d_model
    atencion = 4 * m.d_model**2
    swiglu = 3 * m.d_model * m.d_ff
    normas = 2 * m.d_model
    por_capa = atencion + swiglu + normas
    total = embeddings + m.n_layers * por_capa + m.d_model  # + rmsnorm final

    assert embeddings == 1_310_720
    assert atencion == 409_600
    assert swiglu == 860_160
    assert por_capa == 1_270_400
    assert total == 8_933_440
    assert m.tie_embeddings, "sin tying habria 1,3M parametros mas en la lm_head"


def test_d_ff_es_el_multiplo_de_64_que_toca_en_ambos_configs():
    """SwiGLU usa 2/3 del hidden para compensar que tiene 3 matrices en vez de 2."""
    from llmfs.paths import configs_dir

    for nombre in ("tiny_char.yaml", "tinystories_9m.yaml"):
        m = RunConfig.from_yaml(configs_dir() / nombre).model
        objetivo = 2 / 3 * 4 * m.d_model
        esperado = 64 * -(-int(objetivo) // 64)  # ceil al multiplo de 64
        assert m.d_ff == esperado, f"{nombre}: d_ff={m.d_ff}, se esperaba {esperado}"


def test_el_juguete_entrena_en_segundos_no_en_horas():
    from llmfs.paths import configs_dir

    cfg = RunConfig.from_yaml(configs_dir() / "tiny_char.yaml")
    assert cfg.max_steps < 5_000, "el config juguete tiene que ser realmente juguete"
    assert cfg.model.dropout > 0, "1MB de texto sin dropout se memoriza"


def test_el_resumen_menciona_lo_importante():
    texto = RunConfig(name="x").summary()
    for esperado in ("6L x 320d x 8h", "head_dim=40", "rmsnorm", "rope", "swiglu"):
        assert esperado in texto
