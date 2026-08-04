"""Tests del modulo 11. Ejecutalos con `llmfs check 11`.

El oraculo externo es `torch.optim.AdamW` y `torch.nn.utils.clip_grad_norm_`.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import llmfs.reference as ref
from llmfs.config import ModelConfig
from llmfs.testing import assert_close, assert_scalar_close, load_exercises

ej = load_exercises(__file__)


def modelito(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 1))


def entrena(clase_opt, pasos: int = 50, **kwargs) -> tuple[list[float], list[torch.Tensor]]:
    """Entrena un modelito con el optimizador dado y devuelve perdidas y pesos finales."""
    modelo = modelito()
    opt = clase_opt(modelo.parameters(), **kwargs)
    torch.manual_seed(1)
    x, y = torch.randn(32, 8), torch.randn(32, 1)

    historial = []
    for _ in range(pasos):
        perdida = ((modelo(x) - y) ** 2).mean()
        opt.zero_grad()
        perdida.backward()
        opt.step()
        historial.append(float(perdida.detach()))
    return historial, [p.detach().clone() for p in modelo.parameters()]


# ---------------------------------------------------------- ejercicio 1: AdamWScratch


def test_adamw_coincide_con_el_de_torch():
    """El oraculo externo. Mismos pesos tras 50 pasos, hasta la precision de fp32."""
    kwargs = dict(lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    _, pesos_mios = entrena(ej.AdamWScratch, **kwargs)
    _, pesos_torch = entrena(torch.optim.AdamW, **kwargs)

    for i, (mio, suyo) in enumerate(zip(pesos_mios, pesos_torch)):
        assert_close(mio, suyo, atol=1e-5, what=f"el parametro {i} tras 50 pasos")


def test_adamw_coincide_sin_weight_decay():
    kwargs = dict(lr=1e-2, betas=(0.9, 0.999), weight_decay=0.0)
    _, mios = entrena(ej.AdamWScratch, **kwargs)
    _, suyos = entrena(torch.optim.AdamW, **kwargs)
    for mio, suyo in zip(mios, suyos):
        assert_close(mio, suyo, atol=1e-5, what="los pesos sin weight decay")


def test_adamw_baja_la_perdida():
    historial, _ = entrena(ej.AdamWScratch, pasos=100, lr=1e-2)
    assert historial[-1] < historial[0] / 2, (
        f"la perdida ha pasado de {historial[0]:.4f} a {historial[-1]:.4f}"
    )


def test_la_correccion_de_sesgo_esta_aplicada():
    """En el primer paso, el tamanyo del salto tiene que ser ~lr, no lr/sqrt(1-beta2).

    Sin correccion de sesgo, v vale (1-beta2)*g^2 tras un paso, y dividir por su raiz da
    un paso 1/sqrt(1-beta2) veces mayor: con beta2=0.95, unas 4,5 veces.
    """
    torch.manual_seed(0)
    p = nn.Parameter(torch.zeros(100))
    opt = ej.AdamWScratch([p], lr=0.1, betas=(0.9, 0.95), weight_decay=0.0)
    p.grad = torch.ones(100)
    opt.step()

    salto = float(p.detach().abs().mean())
    assert abs(salto - 0.1) < 0.01, (
        f"el primer paso mueve el parametro {salto:.4f} y deberia moverlo ~0.1 (el lr). "
        f"Si sale ~{0.1 / math.sqrt(0.05):.2f}, falta la correccion de sesgo."
    )


def test_el_weight_decay_esta_desacoplado():
    """AdamW (decay sobre el parametro) frente a Adam+L2 (decay sumado al gradiente).

    Con gradiente CERO, AdamW sigue encogiendo el parametro; Adam+L2 no lo tocaria porque
    el gradiente que le llegaria a Adam seria wd*p y pasaria por la division por sqrt(v).
    """
    p = nn.Parameter(torch.full((10,), 2.0))
    opt = ej.AdamWScratch([p], lr=0.1, weight_decay=0.5)
    p.grad = torch.zeros(10)
    opt.step()

    valor = float(p.detach()[0])
    esperado = 2.0 * (1 - 0.1 * 0.5)
    assert abs(valor - esperado) < 1e-4, (
        f"con gradiente cero y wd=0.5, el parametro deberia pasar de 2.0 a {esperado} "
        f"(decay desacoplado), y vale {valor:.4f}"
    )


def test_los_grupos_tienen_su_propio_weight_decay():
    a = nn.Parameter(torch.full((5,), 1.0))
    b = nn.Parameter(torch.full((5,), 1.0))
    opt = ej.AdamWScratch(
        [{"params": [a], "weight_decay": 0.5}, {"params": [b], "weight_decay": 0.0}],
        lr=0.1,
    )
    a.grad, b.grad = torch.zeros(5), torch.zeros(5)
    opt.step()
    assert float(a[0]) < 1.0, "el grupo con decay deberia haber encogido"
    assert_scalar_close(float(b[0]), 1.0, what="el grupo sin decay")


def test_adamw_ignora_los_parametros_sin_gradiente():
    a = nn.Parameter(torch.ones(5))
    b = nn.Parameter(torch.ones(5))
    opt = ej.AdamWScratch([a, b], lr=0.1)
    a.grad = torch.ones(5)  # b se queda sin gradiente
    opt.step()
    assert_scalar_close(float(b[0]), 1.0, what="el parametro sin gradiente")


def test_el_estado_se_puede_guardar_y_restaurar():
    """Imprescindible para reanudar un entrenamiento sin que el modelo pegue un bandazo."""
    modelo = modelito()
    opt = ej.AdamWScratch(modelo.parameters(), lr=1e-2)
    x, y = torch.randn(8, 8), torch.randn(8, 1)
    for _ in range(5):
        ((modelo(x) - y) ** 2).mean().backward()
        opt.step()
        opt.zero_grad()

    estado = opt.state_dict()
    opt2 = ej.AdamWScratch(modelo.parameters(), lr=1e-2)
    opt2.load_state_dict(estado)

    algun_estado = next(iter(opt2.state.values()))
    assert "exp_avg" in algun_estado or len(algun_estado) > 0


def test_valida_los_hiperparametros():
    p = [nn.Parameter(torch.ones(2))]
    with pytest.raises(ValueError):
        ej.AdamWScratch(p, lr=-1.0)
    with pytest.raises(ValueError):
        ej.AdamWScratch(p, betas=(1.5, 0.9))


def test_adamw_coincide_con_la_referencia():
    kwargs = dict(lr=1e-2, betas=(0.9, 0.95), weight_decay=0.1)
    _, mios = entrena(ej.AdamWScratch, **kwargs)
    _, suyos = entrena(ref.AdamWScratch, **kwargs)
    for mio, suyo in zip(mios, suyos):
        assert_close(mio, suyo, what="los pesos")


# -------------------------------------------------------------- ejercicio 2: lr_at_step


def test_el_warmup_sube_linealmente():
    lrs = [ej.lr_at_step(s, 1000, 1e-3, warmup_steps=100) for s in range(100)]
    assert lrs[0] < lrs[50] < lrs[99]
    assert_scalar_close(lrs[99], 1e-3, rtol=1e-9, what="el lr al final del warmup")
    # Lineal: la diferencia entre pasos consecutivos es constante
    difs = [b - a for a, b in zip(lrs, lrs[1:])]
    assert max(difs) - min(difs) < 1e-12


def test_el_paso_cero_no_tiene_lr_cero():
    """Un paso con lr=0 no aprende nada: es un paso desperdiciado."""
    assert ej.lr_at_step(0, 1000, 1e-3, warmup_steps=100) > 0


def test_justo_despues_del_warmup_esta_en_el_maximo():
    assert_scalar_close(
        ej.lr_at_step(500, 10000, 1e-3, warmup_steps=500), 1e-3, rtol=1e-6, what="el lr en el pico"
    )


def test_el_coseno_decae_hasta_el_suelo():
    lr, ratio = 1e-3, 0.1
    final = ej.lr_at_step(10000, 10000, lr, warmup_steps=500, min_lr_ratio=ratio)
    assert_scalar_close(final, lr * ratio, rtol=1e-6, what="el lr al final")


def test_nunca_baja_del_suelo():
    for s in (10_000, 15_000, 1_000_000):
        assert ej.lr_at_step(s, 10_000, 1e-3, 500, 0.1) >= 1e-4 - 1e-12


def test_el_coseno_es_monotono_decreciente():
    lrs = [ej.lr_at_step(s, 10_000, 1e-3, 500, 0.1) for s in range(500, 10_000, 100)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:])), "el coseno no puede subir"


def test_a_mitad_de_camino_esta_a_mitad_de_altura():
    """cos(pi/2) = 0, asi que coef = 0.5 y el lr es la media entre lr y min_lr."""
    lr, ratio = 1e-3, 0.1
    medio = ej.lr_at_step(5000, 10000, lr, warmup_steps=0, min_lr_ratio=ratio)
    esperado = lr * ratio + (lr - lr * ratio) * 0.5
    assert_scalar_close(medio, esperado, rtol=1e-6, what="el lr a mitad de entrenamiento")


def test_sin_warmup_arranca_en_el_maximo():
    assert_scalar_close(
        ej.lr_at_step(0, 1000, 1e-3, warmup_steps=0), 1e-3, rtol=1e-6, what="el lr sin warmup"
    )


def test_el_schedule_constante():
    for s in (0, 500, 5000):
        assert_scalar_close(
            ej.lr_at_step(s, 10000, 1e-3, warmup_steps=0, schedule="constant"), 1e-3
        )


def test_el_schedule_lineal():
    lr = 1e-3
    medio = ej.lr_at_step(5000, 10000, lr, warmup_steps=0, min_lr_ratio=0.0, schedule="linear")
    assert_scalar_close(medio, lr * 0.5, rtol=1e-6, what="el lr lineal a mitad")


@pytest.mark.parametrize("step", [0, 1, 250, 499, 500, 501, 2500, 9999, 10172, 20000])
def test_coincide_con_la_referencia_en_todos_los_tramos(step):
    assert_scalar_close(
        ej.lr_at_step(step, 10172, 1e-3, 500, 0.1),
        ref.lr_at_step(step, 10172, 1e-3, 500, 0.1),
        what=f"el lr en el paso {step}",
    )


# --------------------------------------------------------- ejercicio 3: clip_grad_norm


def gradientes(escala: float = 1.0, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    modelo = nn.Linear(16, 16)
    ((modelo(torch.randn(4, 16)) ** 2).sum() * escala).backward()
    return modelo


def test_coincide_con_el_clip_de_torch():
    """El oraculo externo."""
    mio = gradientes(100.0)
    suyo = gradientes(100.0)

    norma_mia = ej.clip_grad_norm(mio.parameters(), 1.0)
    norma_suya = float(torch.nn.utils.clip_grad_norm_(list(suyo.parameters()), 1.0))

    assert_scalar_close(norma_mia, norma_suya, rtol=1e-5, what="la norma devuelta")
    for a, b in zip(mio.parameters(), suyo.parameters()):
        assert_close(a.grad, b.grad, atol=1e-6, what="los gradientes recortados")


def test_devuelve_la_norma_antes_de_recortar():
    modelo = gradientes(100.0)
    esperada = float(
        torch.sqrt(sum(p.grad.pow(2).sum() for p in modelo.parameters()))
    )
    devuelta = ej.clip_grad_norm(modelo.parameters(), 1.0)
    assert_scalar_close(devuelta, esperada, rtol=1e-5, what="la norma ANTES de recortar")
    assert devuelta > 1.0, "el caso de prueba deberia superar el umbral"


def test_la_norma_final_es_el_umbral():
    modelo = gradientes(100.0)
    ej.clip_grad_norm(modelo.parameters(), 1.0)
    final = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in modelo.parameters())))
    assert_scalar_close(final, 1.0, rtol=1e-3, what="la norma despues de recortar")


def test_no_toca_nada_si_la_norma_esta_por_debajo():
    modelo = gradientes(0.001)
    antes = [p.grad.clone() for p in modelo.parameters()]
    ej.clip_grad_norm(modelo.parameters(), 1000.0)
    for a, p in zip(antes, modelo.parameters()):
        assert_close(p.grad, a, what="los gradientes sin recortar")


def test_conserva_la_direccion_del_gradiente():
    """Lo que distingue el recorte global del recorte por tensor."""
    modelo = gradientes(100.0)
    antes = torch.cat([p.grad.flatten().clone() for p in modelo.parameters()])
    ej.clip_grad_norm(modelo.parameters(), 1.0)
    despues = torch.cat([p.grad.flatten() for p in modelo.parameters()])

    coseno = float(
        torch.dot(antes, despues) / (antes.norm() * despues.norm())
    )
    assert coseno > 0.9999, (
        f"la direccion ha cambiado (coseno {coseno:.6f}): estas recortando por tensor "
        "en vez de con la norma global"
    )


def test_sin_gradientes_devuelve_cero():
    modelo = nn.Linear(4, 4)  # sin backward, todos los grad son None
    assert ej.clip_grad_norm(modelo.parameters(), 1.0) == 0.0


def test_el_clip_ignora_los_parametros_sin_gradiente():
    a = nn.Parameter(torch.ones(4))
    b = nn.Parameter(torch.ones(4))
    a.grad = torch.full((4,), 3.0)
    norma = ej.clip_grad_norm([a, b], 100.0)
    assert_scalar_close(norma, 6.0, rtol=1e-5, what="la norma de un solo tensor")


def test_el_clip_coincide_con_la_referencia():
    mio, suyo = gradientes(50.0), gradientes(50.0)
    assert_scalar_close(
        ej.clip_grad_norm(mio.parameters(), 1.0),
        ref.clip_grad_norm(suyo.parameters(), 1.0),
        what="la norma",
    )


# ------------------------------------------------------ ejercicio 4: build_param_groups


def test_hay_exactamente_dos_grupos():
    grupos = ej.build_param_groups(ref.GPT(ModelConfig()), 0.1)
    assert len(grupos) == 2


def test_el_primer_grupo_lleva_decay_y_el_segundo_no():
    grupos = ej.build_param_groups(ref.GPT(ModelConfig()), 0.1)
    assert grupos[0]["weight_decay"] == 0.1
    assert grupos[1]["weight_decay"] == 0.0


def test_las_matrices_van_con_decay_y_los_vectores_sin_el():
    grupos = ej.build_param_groups(ref.GPT(ModelConfig()), 0.1)
    assert all(p.dim() >= 2 for p in grupos[0]["params"]), (
        "en el grupo con decay solo deberia haber parametros de 2+ dimensiones"
    )
    assert all(p.dim() < 2 for p in grupos[1]["params"]), (
        "en el grupo sin decay solo deberia haber parametros de 1 dimension"
    )


def test_no_se_pierde_ni_se_duplica_ningun_parametro():
    modelo = ref.GPT(ModelConfig())
    grupos = ej.build_param_groups(modelo, 0.1)
    total = sum(p.numel() for g in grupos for p in g["params"])
    assert total == 8_933_440
    ids = [id(p) for g in grupos for p in g["params"]]
    assert len(ids) == len(set(ids)), "hay parametros repetidos entre los grupos"


def test_las_normas_del_modelo_final_estan_sin_decay():
    """4.160 parametros: 13 RMSNorm x 320. Todos tienen que quedar fuera del decay."""
    grupos = ej.build_param_groups(ref.GPT(ModelConfig()), 0.1)
    assert sum(p.numel() for p in grupos[1]["params"]) == 4_160


def test_salta_los_parametros_congelados():
    modelo = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    modelo[0].weight.requires_grad_(False)
    grupos = ej.build_param_groups(modelo, 0.1)
    # Por identidad: `x in lista_de_tensores` usa `==`, que en tensores devuelve
    # comparacion elemento a elemento y revienta con "Boolean value ... is ambiguous".
    ids = {id(p) for g in grupos for p in g["params"]}
    assert id(modelo[0].weight) not in ids
    assert id(modelo[1].weight) in ids


def test_el_formato_lo_acepta_el_optimizador_de_torch():
    modelo = ref.GPT(ModelConfig(n_layers=1, vocab_size=64, d_model=32, n_heads=4, d_ff=64))
    opt = torch.optim.AdamW(ej.build_param_groups(modelo, 0.1), lr=1e-3)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 0.1


def test_los_grupos_coinciden_con_la_referencia():
    modelo = ref.GPT(ModelConfig())
    mios = ej.build_param_groups(modelo, 0.1)
    suyos = ref.build_param_groups(modelo, 0.1)
    for mio, suyo in zip(mios, suyos):
        assert [id(p) for p in mio["params"]] == [id(p) for p in suyo["params"]]
        assert mio["weight_decay"] == suyo["weight_decay"]
