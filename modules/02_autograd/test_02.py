"""Tests del modulo 02. La verdad la pone `torch.autograd`."""

from __future__ import annotations

import math

import pytest
import torch

import llmfs.reference as ref
from llmfs.testing import assert_scalar_close, load_exercises

ej = load_exercises(__file__)
Value = ej.Value


def torch_scalar(x: float) -> torch.Tensor:
    return torch.tensor(float(x), dtype=torch.float64, requires_grad=True)


# --------------------------------------------------------------- forward


def test_las_operaciones_basicas_dan_el_valor_correcto():
    a, b = Value(3.0), Value(-2.0)
    assert_scalar_close((a + b).data, 1.0, what="a + b")
    assert_scalar_close((a * b).data, -6.0, what="a * b")
    assert_scalar_close((a - b).data, 5.0, what="a - b")
    assert_scalar_close((a / b).data, -1.5, what="a / b")
    assert_scalar_close((a**2).data, 9.0, what="a ** 2")
    assert_scalar_close((-a).data, -3.0, what="-a")


def test_las_funciones_no_lineales_dan_el_valor_correcto():
    a = Value(0.7)
    assert_scalar_close(a.tanh().data, math.tanh(0.7), what="tanh")
    assert_scalar_close(a.exp().data, math.exp(0.7), what="exp")
    assert_scalar_close(a.log().data, math.log(0.7), what="log")
    assert_scalar_close(a.relu().data, 0.7, what="relu(positivo)")
    assert_scalar_close(Value(-0.7).relu().data, 0.0, what="relu(negativo)")


def test_los_operadores_por_la_derecha_funcionan():
    """`2 * a` llama a `a.__rmul__(2)`, no a `__mul__`."""
    a = Value(4.0)
    assert_scalar_close((2 + a).data, 6.0, what="2 + a")
    assert_scalar_close((2 * a).data, 8.0, what="2 * a")
    assert_scalar_close((2 - a).data, -2.0, what="2 - a")
    assert_scalar_close((2 / a).data, 0.5, what="2 / a")


def test_operar_con_un_numero_suelto_lo_envuelve_en_value():
    assert isinstance(Value(1.0) + 3, Value)
    assert isinstance(Value(1.0) * 3, Value)


# --------------------------------------------------------------- gradientes vs torch


def test_gradiente_de_una_expresion_mixta_igual_que_torch():
    a, b = Value(2.0), Value(-3.0)
    c = a * b + a.tanh() - b**2
    c.backward()

    ta, tb = torch_scalar(2.0), torch_scalar(-3.0)
    (ta * tb + ta.tanh() - tb**2).backward()

    assert_scalar_close(a.grad, ta.grad.item(), rtol=1e-9, what="dc/da")
    assert_scalar_close(b.grad, tb.grad.item(), rtol=1e-9, what="dc/db")


def test_gradiente_con_exp_log_y_division_igual_que_torch():
    a, b = Value(1.3), Value(0.4)
    out = (a.exp() / (b + 1.0)).log() * a
    out.backward()

    ta, tb = torch_scalar(1.3), torch_scalar(0.4)
    ((ta.exp() / (tb + 1.0)).log() * ta).backward()

    assert_scalar_close(a.grad, ta.grad.item(), rtol=1e-9, what="da")
    assert_scalar_close(b.grad, tb.grad.item(), rtol=1e-9, what="db")


def test_relu_corta_el_gradiente_en_la_rama_negativa():
    positivo, negativo = Value(2.0), Value(-2.0)
    (positivo.relu() + negativo.relu()).backward()
    assert_scalar_close(positivo.grad, 1.0, what="grad por la rama positiva")
    assert_scalar_close(negativo.grad, 0.0, what="grad por la rama negativa")


def test_el_gradiente_se_acumula_cuando_un_nodo_se_reutiliza():
    """El caso minimo que distingue `+=` de `=`. Con `=` saldria 1.0."""
    x = Value(3.0)
    (x + x).backward()
    assert_scalar_close(x.grad, 2.0, what="d(x+x)/dx")


def test_acumulacion_en_un_grafo_con_forma_de_rombo():
    """x se usa por dos caminos distintos que vuelven a juntarse."""
    x = Value(2.0)
    izquierda = x * 3.0
    derecha = x.tanh()
    (izquierda + derecha).backward()

    tx = torch_scalar(2.0)
    (tx * 3.0 + tx.tanh()).backward()
    assert_scalar_close(x.grad, tx.grad.item(), rtol=1e-9, what="grad por dos caminos")


def test_un_nodo_usado_muchas_veces_suma_todas_las_contribuciones():
    x = Value(1.5)
    total = x
    for _ in range(9):
        total = total + x  # x aparece 10 veces
    total.backward()
    assert_scalar_close(x.grad, 10.0, what="grad de x usado 10 veces")


def test_la_raiz_arranca_con_gradiente_1():
    x = Value(5.0)
    x.backward()
    assert_scalar_close(x.grad, 1.0, what="dx/dx")


# --------------------------------------------------------------- orden topologico


def test_cada_nodo_va_despues_de_sus_hijos():
    a, b = Value(1.0), Value(2.0)
    c = a * b
    d = c + a
    e = d.tanh()

    orden = ej.topological_order(e)
    posicion = {id(n): i for i, n in enumerate(orden)}

    for nodo in orden:
        for hijo in nodo._prev:
            assert posicion[id(hijo)] < posicion[id(nodo)], (
                "un hijo aparece despues que su padre: el orden topologico esta al reves"
            )


def test_la_raiz_es_el_ultimo_elemento():
    a = Value(1.0)
    raiz = (a * 2.0).tanh()
    assert ej.topological_order(raiz)[-1] is raiz


def test_cada_nodo_aparece_una_sola_vez():
    x = Value(2.0)
    raiz = x * x + x
    orden = ej.topological_order(raiz)
    assert len({id(n) for n in orden}) == len(orden), "hay nodos repetidos"


def test_incluye_todos_los_nodos_del_grafo():
    a, b = Value(1.0), Value(2.0)
    raiz = (a + b) * (a - b)
    # a, b, (a+b), (-b), (a-b) [= a + (-b)], (b*-1)... el grafo del azucar sintactico
    # anyade nodos, asi que solo comprobamos que estan las hojas y la raiz.
    ids = {id(n) for n in ej.topological_order(raiz)}
    assert id(a) in ids and id(b) in ids and id(raiz) in ids


def test_no_revienta_con_un_grafo_profundo():
    """Si lo has hecho recursivo, esto lanza RecursionError."""
    x = Value(0.5)
    nodo = x
    for _ in range(3000):
        nodo = nodo + 0.001
    orden = ej.topological_order(nodo)
    assert len(orden) > 3000


def test_backward_funciona_sobre_un_grafo_profundo():
    x = Value(0.0)
    nodo = x
    for _ in range(2000):
        nodo = nodo + x
    nodo.backward()
    assert_scalar_close(x.grad, 2001.0, what="x sumado 2001 veces")


# --------------------------------------------------------------- entrenamiento


def test_el_entrenamiento_baja_la_perdida():
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    history = ej.train_scalar_mlp(xs, ys, hidden=(8, 8), steps=80, lr=0.05, seed=0)

    assert len(history) == 80
    assert history[-1] < history[0] / 10, (
        f"la perdida ha pasado de {history[0]:.4f} a {history[-1]:.4f}. "
        "Si apenas baja, revisa que llamas a zero_grad() ANTES del backward."
    )
    assert history[-1] < 0.05


def test_el_historial_coincide_con_la_referencia():
    """Mismo seed, mismo lr, mismos pasos -> misma trayectoria exacta."""
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]

    mio = ej.train_scalar_mlp(xs, ys, hidden=(4,), steps=25, lr=0.05, seed=7)
    suyo = ref.train_scalar_mlp(xs, ys, hidden=(4,), steps=25, lr=0.05, seed=7, value_cls=ref.Value)

    for paso, (a, b) in enumerate(zip(mio, suyo)):
        assert_scalar_close(a, b, rtol=1e-6, atol=1e-9, what=f"perdida en el paso {paso}")


def test_sin_poner_los_gradientes_a_cero_el_entrenamiento_no_converge():
    """Comprobacion indirecta: la trayectoria correcta es la de la referencia."""
    xs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
    ys = [1.0, 1.0, -1.0, -1.0]
    history = ej.train_scalar_mlp(xs, ys, hidden=(6,), steps=60, lr=0.1, seed=3)
    assert all(math.isfinite(p) for p in history), (
        "hay inf/nan en la perdida: sintoma tipico de gradientes acumulados sin limpiar"
    )
