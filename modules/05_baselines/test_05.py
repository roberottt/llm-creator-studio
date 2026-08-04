"""Tests del modulo 05. Ejecutalos con `llmfs check 05`."""

from __future__ import annotations

import math

import pytest
import torch

import llmfs.reference as ref
from llmfs.testing import assert_close, assert_scalar_close, copy_parameters, load_exercises

ej = load_exercises(__file__)

SECUENCIA = [0, 1, 2, 1, 0, 1, 2, 2, 1, 0] * 40
VOCAB = 3


# --------------------------------------------------- ejercicio 1: uniform_baseline_loss


@pytest.mark.parametrize("vocab", [2, 65, 4096, 50257])
def test_es_el_logaritmo_del_vocabulario(vocab):
    assert_scalar_close(ej.uniform_baseline_loss(vocab), math.log(vocab), what=f"ln({vocab})")


def test_el_numero_del_modelo_final():
    """8.317 es lo que tiene que valer la perdida del paso 0 en el modulo 11."""
    assert_scalar_close(ej.uniform_baseline_loss(4096), 8.3178, rtol=1e-4, what="ln(4096)")


def test_un_vocabulario_de_uno_da_perdida_cero():
    """Si solo hay una opcion posible, acertar es gratis."""
    assert_scalar_close(ej.uniform_baseline_loss(1), 0.0, atol=1e-12, what="ln(1)")


def test_un_vocabulario_no_positivo_es_un_error():
    for malo in (0, -5):
        with pytest.raises(ValueError):
            ej.uniform_baseline_loss(malo)


def test_coincide_con_la_cross_entropy_de_torch():
    """Comprobacion cruzada: logits uniformes deben dar exactamente ln(V)."""
    vocab = 512
    logits = torch.zeros(1, vocab)  # todos iguales -> softmax uniforme
    objetivo = torch.tensor([7])
    assert_scalar_close(
        ej.uniform_baseline_loss(vocab),
        float(torch.nn.functional.cross_entropy(logits, objetivo)),
        rtol=1e-5,
        what="ln(V) vs cross_entropy con logits planos",
    )


# ---------------------------------------------------------- ejercicio 2: bigram_counts


def test_cuenta_el_ejemplo_del_enunciado():
    counts = ej.bigram_counts([0, 1, 0, 1, 2], 3)
    assert counts[0][1].item() == 2
    assert counts[1][0].item() == 1
    assert counts[1][2].item() == 1
    assert counts.sum().item() == 4


def test_la_forma_y_el_tipo_son_los_correctos():
    counts = ej.bigram_counts(SECUENCIA, VOCAB)
    assert counts.shape == (VOCAB, VOCAB)
    assert counts.dtype == torch.int64


def test_el_total_es_el_numero_de_pares():
    counts = ej.bigram_counts(SECUENCIA, VOCAB)
    assert counts.sum().item() == len(SECUENCIA) - 1


def test_las_repeticiones_se_acumulan_no_se_pisan():
    """Sin accumulate=True todos los conteos saldrian 1."""
    counts = ej.bigram_counts([0, 0, 0, 0, 0], 2)
    assert counts[0][0].item() == 4


def test_una_secuencia_corta_da_la_matriz_de_ceros():
    assert ej.bigram_counts([5], 10).sum().item() == 0
    assert ej.bigram_counts([], 10).sum().item() == 0


def test_los_conteos_coinciden_con_la_referencia():
    assert torch.equal(ej.bigram_counts(SECUENCIA, VOCAB), ref.bigram_counts(SECUENCIA, VOCAB))


# ------------------------------------------------------------- ejercicio 3: bigram_nll


def test_la_perdida_coincide_con_la_referencia():
    counts = ref.bigram_counts(SECUENCIA, VOCAB)
    assert_scalar_close(
        ej.bigram_nll(counts, SECUENCIA), ref.bigram_nll(counts, SECUENCIA), what="la perdida"
    )


def test_el_bigrama_bate_al_baseline_uniforme():
    """Si esto falla, el modelo de bigramas no esta aprendiendo nada."""
    counts = ref.bigram_counts(SECUENCIA, VOCAB)
    perdida = ej.bigram_nll(counts, SECUENCIA)
    assert perdida < math.log(VOCAB), (
        f"el bigrama da {perdida:.4f} y adivinar al azar da {math.log(VOCAB):.4f}"
    )


def test_una_secuencia_perfectamente_predecible_da_perdida_casi_cero():
    ciclo = [0, 1] * 500
    counts = ref.bigram_counts(ciclo, 2)
    assert ej.bigram_nll(counts, ciclo, alpha=1e-6) < 0.01


def test_el_suavizado_evita_el_infinito():
    """El punto del ejercicio: un par no visto no puede mandar la perdida a infinito."""
    counts = ref.bigram_counts([0, 1, 0, 1], 3)  # el token 2 nunca aparece
    perdida = ej.bigram_nll(counts, [2, 0, 1, 2], alpha=1.0)
    assert math.isfinite(perdida), "con alpha > 0 la perdida no puede ser infinita"


def test_mas_suavizado_acerca_la_perdida_al_baseline_uniforme():
    """alpha enorme aplasta los conteos y el modelo se vuelve uniforme."""
    counts = ref.bigram_counts(SECUENCIA, VOCAB)
    poco = ej.bigram_nll(counts, SECUENCIA, alpha=0.01)
    mucho = ej.bigram_nll(counts, SECUENCIA, alpha=1e6)
    assert poco < mucho
    assert_scalar_close(mucho, math.log(VOCAB), rtol=1e-3, what="perdida con alpha enorme")


def test_las_probabilidades_de_cada_fila_suman_uno():
    """Comprobacion indirecta del denominador: hay que sumar alpha*V, no solo alpha."""
    counts = ref.bigram_counts(SECUENCIA, VOCAB)
    suavizado = counts.double() + 2.5
    probs = suavizado / suavizado.sum(dim=1, keepdim=True)
    assert torch.allclose(probs.sum(dim=1), torch.ones(VOCAB, dtype=torch.float64))


def test_una_secuencia_de_menos_de_dos_tokens_es_un_error():
    counts = ref.bigram_counts(SECUENCIA, VOCAB)
    with pytest.raises(ValueError):
        ej.bigram_nll(counts, [1])


# ----------------------------------------------------------- ejercicio 4: NeuralBigram


def test_neural_bigram_tiene_la_arquitectura_esperada():
    modelo = ej.NeuralBigram(VOCAB)
    esperado = ref.NeuralBigram(VOCAB)
    copy_parameters(esperado, modelo)  # falla con mensaje util si los nombres no cuadran


def test_neural_bigram_devuelve_las_formas_correctas():
    modelo = ej.NeuralBigram(VOCAB)
    idx = torch.randint(0, VOCAB, (4, 8))
    logits, perdida = modelo(idx, idx)
    assert logits.shape == (4, 8, VOCAB)
    assert perdida is not None and perdida.ndim == 0


def test_neural_bigram_sin_targets_no_devuelve_perdida():
    modelo = ej.NeuralBigram(VOCAB)
    logits, perdida = modelo(torch.randint(0, VOCAB, (2, 5)))
    assert perdida is None


def test_neural_bigram_coincide_con_la_referencia():
    torch.manual_seed(0)
    mio, suyo = ej.NeuralBigram(VOCAB), ref.NeuralBigram(VOCAB)
    copy_parameters(suyo, mio)

    idx = torch.randint(0, VOCAB, (4, 8))
    tgt = torch.randint(0, VOCAB, (4, 8))
    logits_mio, perdida_mio = mio(idx, tgt)
    logits_suyo, perdida_suyo = suyo(idx, tgt)

    assert_close(logits_mio, logits_suyo, what="los logits")
    assert_scalar_close(perdida_mio, perdida_suyo, what="la perdida")


def test_neural_bigram_tiene_v_al_cuadrado_parametros():
    modelo = ej.NeuralBigram(50)
    assert sum(p.numel() for p in modelo.parameters()) == 50 * 50


def test_neural_bigram_entrenado_se_acerca_al_bigrama_por_conteo():
    """Contar y aprender por gradiente dan lo mismo con un modelo asi de simple."""
    torch.manual_seed(0)
    datos = torch.tensor(SECUENCIA, dtype=torch.long)
    modelo = ej.NeuralBigram(VOCAB)
    opt = torch.optim.AdamW(modelo.parameters(), lr=0.5)

    x, y = datos[:-1].unsqueeze(0), datos[1:].unsqueeze(0)
    for _ in range(300):
        _, perdida = modelo(x, y)
        opt.zero_grad()
        perdida.backward()
        opt.step()

    por_conteo = ref.bigram_nll(ref.bigram_counts(SECUENCIA, VOCAB), SECUENCIA, alpha=1e-4)
    assert abs(float(perdida) - por_conteo) < 0.1, (
        f"entrenado da {float(perdida):.4f} y contando da {por_conteo:.4f}; "
        "deberian converger al mismo sitio"
    )


# ------------------------------------------------------------- ejercicio 5: BengioMLP


def test_bengio_tiene_la_arquitectura_esperada():
    modelo = ej.BengioMLP(VOCAB, block_size=4)
    copy_parameters(ref.BengioMLP(VOCAB, block_size=4), modelo)


def test_bengio_devuelve_las_formas_correctas():
    modelo = ej.BengioMLP(VOCAB, block_size=4)
    idx = torch.randint(0, VOCAB, (6, 4))
    logits, perdida = modelo(idx, torch.randint(0, VOCAB, (6,)))
    assert logits.shape == (6, VOCAB), "un solo vector de logits por muestra, no uno por token"
    assert perdida is not None and perdida.ndim == 0


def test_bengio_coincide_con_la_referencia():
    torch.manual_seed(0)
    mio = ej.BengioMLP(VOCAB, block_size=4, d_embed=8, n_hidden=16)
    suyo = ref.BengioMLP(VOCAB, block_size=4, d_embed=8, n_hidden=16)
    copy_parameters(suyo, mio)

    idx = torch.randint(0, VOCAB, (6, 4))
    tgt = torch.randint(0, VOCAB, (6,))
    logits_mio, perdida_mio = mio(idx, tgt)
    logits_suyo, perdida_suyo = suyo(idx, tgt)

    assert_close(logits_mio, logits_suyo, what="los logits")
    assert_scalar_close(perdida_mio, perdida_suyo, what="la perdida")


def test_bengio_concatena_en_vez_de_promediar():
    """Si promediaras los embeddings, el orden de los tokens daria igual. No da igual."""
    torch.manual_seed(0)
    modelo = ej.BengioMLP(VOCAB, block_size=3, d_embed=8, n_hidden=16)
    a = modelo(torch.tensor([[0, 1, 2]]))[0]
    b = modelo(torch.tensor([[2, 1, 0]]))[0]
    assert not torch.allclose(a, b), (
        "el modelo da lo mismo con el contexto al reves: estas promediando, no concatenando"
    )


def test_bengio_crece_en_parametros_con_el_contexto():
    """Su limitacion, y la razon de que exista la atencion."""
    pequeno = sum(p.numel() for p in ej.BengioMLP(VOCAB, 2, d_embed=8, n_hidden=16).parameters())
    grande = sum(p.numel() for p in ej.BengioMLP(VOCAB, 8, d_embed=8, n_hidden=16).parameters())
    assert grande > pequeno


def test_bengio_aprende_de_verdad():
    torch.manual_seed(0)
    datos = torch.tensor(SECUENCIA, dtype=torch.long)
    bloque = 3
    x = torch.stack([datos[i : i + bloque] for i in range(len(datos) - bloque)])
    y = datos[bloque:]

    modelo = ej.BengioMLP(VOCAB, bloque, d_embed=8, n_hidden=32)
    opt = torch.optim.AdamW(modelo.parameters(), lr=0.05)
    inicial = None
    for _ in range(200):
        _, perdida = modelo(x, y)
        inicial = inicial if inicial is not None else float(perdida)
        opt.zero_grad()
        perdida.backward()
        opt.step()

    assert float(perdida) < inicial
    assert float(perdida) < math.log(VOCAB), "deberia batir al baseline uniforme"
