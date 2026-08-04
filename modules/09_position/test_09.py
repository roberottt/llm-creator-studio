"""Tests del modulo 09. Ejecutalos con `llmfs check 09`."""

from __future__ import annotations

import math

import pytest
import torch

import llmfs.reference as ref
from llmfs.testing import assert_close, assert_scalar_close, load_exercises

ej = load_exercises(__file__)


# ---------------------------------------------- ejercicio 1: sinusoidal_embeddings


def test_sinusoidales_tienen_la_forma_correcta():
    assert ej.sinusoidal_embeddings(64, 32).shape == (64, 32)


def test_sinusoidales_coinciden_con_la_referencia():
    assert_close(
        ej.sinusoidal_embeddings(64, 32), ref.sinusoidal_embeddings(64, 32), what="la tabla"
    )


def test_las_dimensiones_pares_son_seno_y_las_impares_coseno():
    tabla = ej.sinusoidal_embeddings(16, 8)
    # En la posicion 0 todos los angulos valen 0: sin(0)=0 y cos(0)=1
    assert_close(tabla[0, 0::2], torch.zeros(4), atol=1e-6, what="las dimensiones pares en pos 0")
    assert_close(tabla[0, 1::2], torch.ones(4), atol=1e-6, what="las impares en pos 0")


def test_los_valores_estan_acotados_entre_menos_uno_y_uno():
    tabla = ej.sinusoidal_embeddings(500, 64)
    assert float(tabla.max()) <= 1.0 + 1e-6 and float(tabla.min()) >= -1.0 - 1e-6


def test_cada_posicion_tiene_una_firma_distinta():
    tabla = ej.sinusoidal_embeddings(128, 32)
    distancias = torch.cdist(tabla, tabla)
    distancias.fill_diagonal_(float("inf"))
    assert float(distancias.min()) > 1e-3, "hay dos posiciones con el mismo vector"


def test_las_primeras_dimensiones_oscilan_mas_rapido_que_las_ultimas():
    """La escalera de frecuencias, que es la idea del contador binario."""
    tabla = ej.sinusoidal_embeddings(200, 64)
    variacion_rapida = float((tabla[1:, 0] - tabla[:-1, 0]).abs().mean())
    variacion_lenta = float((tabla[1:, -2] - tabla[:-1, -2]).abs().mean())
    assert variacion_rapida > variacion_lenta * 10, (
        f"la primera dimension varia {variacion_rapida:.4f} por posicion y la ultima "
        f"{variacion_lenta:.6f}: deberian diferenciarse en ordenes de magnitud"
    )


def test_esta_definida_para_cualquier_posicion():
    """A diferencia de una tabla aprendida, no hay techo."""
    assert torch.isfinite(ej.sinusoidal_embeddings(100_000, 16)).all()


# ------------------------------------------------------ ejercicio 2: rope_frequencies


def test_rope_devuelve_dos_tablas_con_head_dim_columnas():
    cos, sin = ej.rope_frequencies(16, 64)
    assert cos.shape == (64, 16), "las tablas tienen head_dim columnas, no head_dim/2"
    assert sin.shape == (64, 16)


def test_rope_coincide_con_la_referencia():
    cos_mio, sin_mio = ej.rope_frequencies(40, 512)
    cos_ref, sin_ref = ref.rope_frequencies(40, 512)
    assert_close(cos_mio, cos_ref, what="la tabla de cosenos")
    assert_close(sin_mio, sin_ref, what="la tabla de senos")


def test_rope_duplica_las_frecuencias_por_mitades():
    """El convenio de mitades: la columna i y la i+head_dim/2 llevan el MISMO angulo."""
    cos, sin = ej.rope_frequencies(16, 32)
    assert_close(cos[:, :8], cos[:, 8:], what="las dos mitades de cos")
    assert_close(sin[:, :8], sin[:, 8:], what="las dos mitades de sin")


def test_en_la_posicion_cero_no_hay_rotacion():
    cos, sin = ej.rope_frequencies(16, 32)
    assert_close(cos[0], torch.ones(16), atol=1e-6, what="cos en la posicion 0")
    assert_close(sin[0], torch.zeros(16), atol=1e-6, what="sin en la posicion 0")


def test_cos_al_cuadrado_mas_sin_al_cuadrado_es_uno():
    """Identidad trigonometrica basica: si falla, las tablas no son de un angulo real."""
    cos, sin = ej.rope_frequencies(16, 64)
    assert_close(cos**2 + sin**2, torch.ones(64, 16), atol=1e-5, what="cos^2 + sin^2")


def test_un_head_dim_impar_es_un_error():
    with pytest.raises(ValueError):
        ej.rope_frequencies(15, 32)


def test_la_primera_frecuencia_gira_mas_rapido_que_la_ultima():
    cos, _ = ej.rope_frequencies(64, 256)
    rapida = float((cos[1:, 0] - cos[:-1, 0]).abs().mean())
    lenta = float((cos[1:, 31] - cos[:-1, 31]).abs().mean())
    assert rapida > lenta * 100


def test_respeta_el_dispositivo():
    from llmfs.device import get_device

    cfg = get_device()
    cos, sin = ej.rope_frequencies(16, 32, device=cfg.device)
    assert cos.device.type == cfg.kind and sin.device.type == cfg.kind


# ------------------------------------------------------------ ejercicio 3: apply_rope


def test_apply_rope_conserva_la_forma():
    cos, sin = ref.rope_frequencies(16, 64)
    x = torch.randn(2, 4, 8, 16)
    assert ej.apply_rope(x, cos, sin).shape == (2, 4, 8, 16)


def test_apply_rope_coincide_con_la_referencia():
    torch.manual_seed(0)
    cos, sin = ref.rope_frequencies(16, 64)
    x = torch.randn(2, 4, 8, 16)
    assert_close(ej.apply_rope(x, cos, sin), ref.apply_rope(x, cos, sin), what="RoPE")


def test_rotar_no_cambia_la_longitud_del_vector():
    """La ventaja sobre sumar un embedding posicional: solo cambia la direccion."""
    torch.manual_seed(0)
    cos, sin = ref.rope_frequencies(16, 64)
    x = torch.randn(2, 4, 8, 16)
    assert_close(
        ej.apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5, what="la norma"
    )


def test_en_la_posicion_cero_el_vector_no_cambia():
    cos, sin = ref.rope_frequencies(16, 64)
    x = torch.randn(1, 1, 1, 16)
    assert_close(ej.apply_rope(x, cos, sin), x, atol=1e-6, what="el vector en la posicion 0")


def test_posiciones_distintas_rotan_distinto():
    cos, sin = ref.rope_frequencies(16, 64)
    x = torch.randn(1, 1, 1, 16).repeat(1, 1, 5, 1)  # el mismo vector en 5 posiciones
    rotado = ej.apply_rope(x, cos, sin)
    for pos in range(1, 5):
        assert not torch.allclose(rotado[0, 0, 0], rotado[0, 0, pos], atol=1e-4)


def test_la_invariancia_relativa_es_exacta():
    """LA propiedad que justifica RoPE.

    El producto escalar de dos vectores rotados depende solo de la DIFERENCIA de
    posiciones: <R(m)q, R(n)k> = <q, R(n-m)k>.

    Aqui se comprueba que la puntuacion entre las posiciones (2,5) es la misma que entre
    las (4,7): en ambos casos la distancia es 3.
    """
    torch.manual_seed(0)
    cos, sin = ref.rope_frequencies(16, 64)
    q_base, k_base = torch.randn(16), torch.randn(16)

    def puntuacion(pos_q: int, pos_k: int) -> float:
        q = torch.zeros(1, 1, 32, 16)
        k = torch.zeros(1, 1, 32, 16)
        q[0, 0, pos_q] = q_base
        k[0, 0, pos_k] = k_base
        qr = ej.apply_rope(q, cos, sin)
        kr = ej.apply_rope(k, cos, sin)
        return float(qr[0, 0, pos_q] @ kr[0, 0, pos_k])

    assert_scalar_close(
        puntuacion(2, 5), puntuacion(4, 7), rtol=1e-5, what="la puntuacion a distancia 3"
    )
    assert_scalar_close(
        puntuacion(0, 10), puntuacion(20, 30), rtol=1e-5, what="la puntuacion a distancia 10"
    )
    # Y a distancias distintas tiene que dar cosas distintas
    assert abs(puntuacion(2, 5) - puntuacion(2, 9)) > 1e-4


def test_funciona_con_una_secuencia_mas_corta_que_las_tablas():
    """cos/sin vienen con max_seq_len filas: hay que recortarlas a la longitud real."""
    cos, sin = ref.rope_frequencies(16, 512)
    x = torch.randn(2, 4, 7, 16)
    assert ej.apply_rope(x, cos, sin).shape == (2, 4, 7, 16)


def test_funciona_en_fp16():
    """Con AMP, x llega en fp16 y las tablas estan en fp32: hay que convertir."""
    cos, sin = ref.rope_frequencies(16, 64)
    x = torch.randn(1, 2, 4, 16, dtype=torch.float16)
    salida = ej.apply_rope(x, cos, sin)
    assert salida.dtype == torch.float16
    assert torch.isfinite(salida).all()


def test_es_lineal_en_x():
    """Rotar es una transformacion lineal: R(2x) = 2·R(x)."""
    cos, sin = ref.rope_frequencies(16, 64)
    x = torch.randn(1, 1, 4, 16)
    assert_close(ej.apply_rope(2 * x, cos, sin), 2 * ej.apply_rope(x, cos, sin), atol=1e-5)
