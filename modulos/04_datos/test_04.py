"""Tests del modulo 04. Ejecutalos con `llmfs check 04`."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import llmfs.reference as ref
from llmfs.testing import load_exercises

ej = load_exercises(__file__)


def corpus(n: int = 10_000, vocab: int = 4096, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, vocab, size=n).astype(np.uint16)


# ------------------------------------------------------- ejercicio 1: pack_tokens_uint16


def test_devuelve_un_array_uint16():
    salida = ej.pack_tokens_uint16([0, 1, 2, 4095], 4096)
    assert isinstance(salida, np.ndarray)
    assert salida.dtype == np.uint16


def test_conserva_los_valores():
    ids = [0, 1, 2, 4095, 300]
    assert ej.pack_tokens_uint16(ids, 4096).tolist() == ids


def test_ocupa_dos_bytes_por_token():
    """La razon de ser del ejercicio: la mitad de disco que uint32."""
    salida = ej.pack_tokens_uint16(list(range(1000)), 4096)
    assert salida.nbytes == 2000


def test_un_id_fuera_del_vocabulario_es_un_error():
    """El wrap-around silencioso de numpy es lo que hay que evitar."""
    with pytest.raises(ValueError):
        ej.pack_tokens_uint16([0, 1, 4096], 4096)


def test_un_id_negativo_es_un_error():
    with pytest.raises(ValueError):
        ej.pack_tokens_uint16([0, -1, 2], 4096)


def test_un_vocabulario_que_no_cabe_en_uint16_es_un_error():
    with pytest.raises(ValueError):
        ej.pack_tokens_uint16([0, 1], 100_000)


def test_el_mensaje_de_error_dice_que_valor_se_ha_salido():
    with pytest.raises(ValueError) as exc:
        ej.pack_tokens_uint16([0, 1, 9999], 4096)
    assert "9999" in str(exc.value), "el error deberia decir cual es el id problematico"


def test_una_lista_vacia_no_revienta():
    """`.min()` sobre un array vacio lanza excepcion: hay que comprobar el tamanyo."""
    salida = ej.pack_tokens_uint16([], 4096)
    assert salida.dtype == np.uint16 and len(salida) == 0


def test_el_empaquetado_coincide_con_la_referencia():
    ids = list(range(0, 4096, 7))
    assert np.array_equal(ej.pack_tokens_uint16(ids, 4096), ref.pack_tokens_uint16(ids, 4096))


# ---------------------------------------------------------- ejercicio 2: train_val_split


def test_las_dos_partes_suman_el_total():
    datos = corpus(10_000)
    train, val = ej.train_val_split(datos, 0.01)
    assert len(train) + len(val) == len(datos)


def test_la_fraccion_es_la_pedida():
    datos = corpus(10_000)
    _, val = ej.train_val_split(datos, 0.05)
    assert len(val) == 500


def test_validacion_es_el_final_del_corpus():
    """Contiguo y por el final, no aleatorio. Ver TEORIA.md."""
    datos = corpus(1000)
    train, val = ej.train_val_split(datos, 0.1)
    assert np.array_equal(val, datos[-100:])
    assert np.array_equal(train, datos[:-100])


def test_no_hay_solapamiento_entre_las_dos_partes():
    datos = np.arange(1000, dtype=np.uint16)
    train, val = ej.train_val_split(datos, 0.1)
    assert set(train.tolist()).isdisjoint(set(val.tolist()))


def test_devuelve_vistas_y_no_copias():
    """Con 500M tokens, copiar es un gigabyte tirado."""
    datos = corpus(1000)
    train, _ = ej.train_val_split(datos, 0.1)
    assert train.base is datos or np.shares_memory(train, datos)


def test_validacion_nunca_queda_vacia():
    """Con un corpus pequenyo, int(len * fraction) podria dar 0."""
    _, val = ej.train_val_split(corpus(50), 0.005)
    assert len(val) >= 1


def test_una_fraccion_invalida_es_un_error():
    datos = corpus(1000)
    for fraccion in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            ej.train_val_split(datos, fraccion)


def test_la_particion_coincide_con_la_referencia():
    datos = corpus(5000)
    mio_tr, mio_val = ej.train_val_split(datos, 0.02)
    ref_tr, ref_val = ref.train_val_split(datos, 0.02)
    assert np.array_equal(mio_tr, ref_tr) and np.array_equal(mio_val, ref_val)


# ----------------------------------------------------------------- ejercicio 3: get_batch


def test_las_formas_son_las_correctas():
    x, y = ej.get_batch(corpus(), batch_size=4, context_length=16, rng=np.random.default_rng(0))
    assert x.shape == (4, 16) and y.shape == (4, 16)


def test_el_dtype_es_int64():
    """Los indices de nn.Embedding tienen que ser int64."""
    x, y = ej.get_batch(corpus(), 4, 16, rng=np.random.default_rng(0))
    assert x.dtype == torch.int64 and y.dtype == torch.int64


def test_y_es_x_desplazado_un_token():
    """La propiedad que define la tarea: predecir el siguiente."""
    x, y = ej.get_batch(corpus(), 8, 32, rng=np.random.default_rng(1))
    assert torch.equal(x[:, 1:], y[:, :-1]), (
        "y[i] tiene que ser el token que sigue a x[i]. Si esto falla, el modelo estaria "
        "aprendiendo a predecir cualquier otra cosa."
    )


def test_los_tokens_salen_del_corpus_y_estan_en_orden():
    """Comprueba contra el corpus real que la ventana es contigua."""
    datos = np.arange(5000, dtype=np.uint16)
    x, y = ej.get_batch(datos, 4, 10, rng=np.random.default_rng(3))
    for fila in range(4):
        inicio = int(x[fila, 0])
        assert x[fila].tolist() == list(range(inicio, inicio + 10))
        assert y[fila].tolist() == list(range(inicio + 1, inicio + 11))


def test_nunca_se_sale_del_final_del_corpus():
    """El -1 del max_start. Sin el, la ultima ventana de `y` desborda."""
    datos = np.arange(200, dtype=np.uint16)
    for semilla in range(30):
        x, y = ej.get_batch(datos, 16, 50, rng=np.random.default_rng(semilla))
        assert int(y.max()) <= 199


def test_un_corpus_mas_corto_que_el_contexto_es_un_error():
    with pytest.raises(ValueError):
        ej.get_batch(corpus(10), batch_size=2, context_length=64)


def test_la_misma_semilla_da_el_mismo_batch():
    datos = corpus()
    a = ej.get_batch(datos, 4, 16, rng=np.random.default_rng(42))
    b = ej.get_batch(datos, 4, 16, rng=np.random.default_rng(42))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_semillas_distintas_dan_batches_distintos():
    datos = corpus()
    a = ej.get_batch(datos, 8, 32, rng=np.random.default_rng(1))
    b = ej.get_batch(datos, 8, 32, rng=np.random.default_rng(2))
    assert not torch.equal(a[0], b[0])


def test_coincide_exactamente_con_la_referencia():
    datos = corpus()
    mio = ej.get_batch(datos, 8, 32, rng=np.random.default_rng(7))
    suyo = ref.get_batch(datos, 8, 32, rng=np.random.default_rng(7))
    assert torch.equal(mio[0], suyo[0]) and torch.equal(mio[1], suyo[1])


def test_funciona_sobre_un_memmap_en_disco(tmp_path):
    """El caso real: el corpus no esta en RAM, esta en un fichero."""
    ruta = tmp_path / "tokens.bin"
    corpus(20_000).tofile(ruta)
    datos = np.memmap(ruta, dtype=np.uint16, mode="r")

    x, y = ej.get_batch(datos, 4, 64, rng=np.random.default_rng(0))
    assert x.shape == (4, 64)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_mueve_los_tensores_al_dispositivo():
    from llmfs.device import get_device

    cfg = get_device()
    x, y = ej.get_batch(corpus(), 4, 16, device=cfg.device, rng=np.random.default_rng(0))
    assert x.device.type == cfg.kind and y.device.type == cfg.kind
