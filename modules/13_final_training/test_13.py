"""Tests del modulo 13. Ejecutalos con `llmfs check 13`."""

from __future__ import annotations

import math

import pytest
import torch

import llmfs.reference as ref
from llmfs.config import ModelConfig
from llmfs.device import set_seed
from llmfs.testing import load_exercises

ej = load_exercises(__file__)


def modelito() -> torch.nn.Module:
    set_seed(0)
    return ref.GPT(
        ModelConfig(vocab_size=64, n_layers=2, d_model=32, n_heads=4, d_ff=96, context_length=16)
    )


def batch(vocab: int = 64, b: int = 4, t: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    set_seed(1)
    seq = torch.randint(0, vocab, (b, t + 1))
    return seq[:, :-1], seq[:, 1:]


# ------------------------------------------------- ejercicio 1: overfit_single_batch


def test_la_perdida_baja_casi_a_cero():
    """LA comprobacion: un modelo sano memoriza cuatro secuencias sin despeinarse."""
    x, y = batch()
    historial = ej.overfit_single_batch(modelito(), x, y, steps=300, lr=3e-3)
    assert historial[-1] < 0.1, (
        f"la perdida se ha quedado en {historial[-1]:.4f} memorizando un solo batch. "
        "Si no baja, hay un bug en el modelo o en el bucle."
    )


def test_arranca_cerca_de_ln_del_vocabulario():
    x, y = batch()
    historial = ej.overfit_single_batch(modelito(), x, y, steps=10, lr=1e-3)
    assert abs(historial[0] - math.log(64)) < 0.5, (
        f"arranca en {historial[0]:.4f} y deberia rondar ln(64)={math.log(64):.4f}"
    )


def test_devuelve_una_perdida_por_paso():
    x, y = batch()
    assert len(ej.overfit_single_batch(modelito(), x, y, steps=37, lr=1e-3)) == 37


def test_la_perdida_es_monotona_a_grandes_rasgos():
    x, y = batch()
    h = ej.overfit_single_batch(modelito(), x, y, steps=200, lr=3e-3)
    primer_tercio = sum(h[:60]) / 60
    ultimo_tercio = sum(h[-60:]) / 60
    assert ultimo_tercio < primer_tercio / 5


def test_acepta_un_optimizador_propio():
    x, y = batch()
    h = ej.overfit_single_batch(
        modelito(),
        x,
        y,
        steps=100,
        optimizer_factory=lambda p: ref.AdamWScratch(p, lr=3e-3),
    )
    assert h[-1] < h[0]


def test_los_valores_son_finitos():
    x, y = batch()
    h = ej.overfit_single_batch(modelito(), x, y, steps=100, lr=3e-3)
    assert all(math.isfinite(v) for v in h), "hay inf o nan en el historial"


def test_deja_el_modelo_en_modo_entrenamiento():
    x, y = batch()
    modelo = modelito()
    modelo.eval()
    ej.overfit_single_batch(modelo, x, y, steps=5, lr=1e-3)
    assert modelo.training, "hay que llamar a model.train() antes del bucle"


def test_el_overfit_coincide_con_la_referencia():
    x, y = batch()
    set_seed(0)
    mio = ej.overfit_single_batch(modelito(), x, y, steps=50, lr=3e-3)
    set_seed(0)
    suyo = ref.overfit_single_batch(modelito(), x, y, steps=50, lr=3e-3)
    for i, (a, b) in enumerate(zip(mio, suyo)):
        assert abs(a - b) < 1e-4, f"divergen en el paso {i}: {a:.6f} vs {b:.6f}"


# ------------------------------------------------------------ ejercicio 2: format_eta


@pytest.mark.parametrize(
    "segundos,esperado",
    [
        (0, "0s"),
        (1, "1s"),
        (45, "45s"),
        (59, "59s"),
        (60, "1m 0s"),
        (125, "2m 5s"),
        (3599, "59m 59s"),
        (3600, "1h 0m"),
        (3725, "1h 2m"),
        (86399, "23h 59m"),
        (86400, "1d 0h"),
        (90000, "1d 1h"),
    ],
)
def test_los_formatos_de_cada_tramo(segundos, esperado):
    assert ej.format_eta(segundos) == esperado


def test_a_partir_de_una_hora_no_se_muestran_los_segundos():
    """Cuando faltan dos horas, los segundos son ruido."""
    assert ej.format_eta(7265).count("s") == 0 or "m" in ej.format_eta(7265)
    assert ej.format_eta(7265) == "2h 1m"


def test_los_valores_raros_devuelven_interrogacion():
    for malo in (-1, -1000, float("inf"), float("nan"), float("-inf")):
        assert ej.format_eta(malo) == "?", f"{malo} deberia dar '?'"


def test_admite_floats():
    assert ej.format_eta(45.7) == "45s"


def test_el_formato_coincide_con_la_referencia():
    for s in (0, 30, 59, 60, 125, 3599, 3600, 3725, 86400, 123456, -5, float("inf")):
        assert ej.format_eta(s) == ref.format_eta(s), f"discrepan en {s}"
