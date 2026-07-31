# 12 — Solución comentada

## Ejercicio 1 — `model_flops_per_token`

```python
d, ff, v = cfg.d_model, cfg.d_ff, cfg.vocab_size
n_ffn = 3 if cfg.activation == "swiglu" else 2

params_matmul = cfg.n_layers * (4 * d * d + n_ffn * d * ff) + d * v

matmul = 2 * params_matmul
attention = 4 * cfg.n_layers * cfg.context_length * d

factor = 3 if include_backward else 1
return {
    "matmul": matmul * factor,
    "attention": attention * factor,
    "total": (matmul + attention) * factor,
    "params_matmul": params_matmul,
}
```

Es el cálculo del módulo 01, pero devolviendo el desglose. Y el desglose es lo que hace útil
la función:

| contexto | matmul | atención | % atención |
|---|---|---|---|
| 128 | 160,8M | 2,9M | 2% |
| **512** | **160,8M** | **11,8M** | **7%** |
| 2048 | 160,8M | 47,2M | 23% |
| 8192 | 160,8M | 188,7M | **54%** |

El término de matmul **no se mueve**: solo depende del tamaño del modelo. El de atención
crece linealmente con el contexto, y a partir de 2048 empieza a dominar. Con esa tabla
delante, la decisión de "¿alargo el contexto?" deja de ser a ciegas.

**La proyección final cuenta aunque haya weight tying.** Atar los pesos ahorra memoria, no
cómputo: el matmul $(B\cdot T, d) \times (d, V)$ se ejecuta igual. Hay un test que lo
comprueba.

## Ejercicio 2 — `compute_mfu`

```python
if peak_tflops <= 0:
    raise ValueError("peak_tflops tiene que ser positivo")
return tokens_per_second * flops_per_token / (peak_tflops * 1e12)
```

Una línea. Lo interesante es cómo se interpreta.

En el demo, midiendo de verdad sobre este hardware, la MFU sube con el tamaño del batch y
luego se estanca. Ese punto de estancamiento es donde dejas de estar limitado por el
lanzamiento de kernels y pasas a estarlo por el cálculo.

**Nadie llega a 1.** Y con un modelo de 9M, 0,1–0,2 ya es bueno: las matrices de 320×320 no
dan para saturar los tensor cores. Es el mismo fenómeno que mediste en el módulo 01.

**El valor de la MFU no es su número absoluto, es que es comparable.** No depende del modelo
ni del hardware, así que puedes cambiar el batch size, activar `torch.compile` o mover el
dataloader a otro hilo, y ver si sube.

## Ejercicio 3 — `chinchilla_optimal_allocation`

```python
if compute_budget <= 0:
    raise ValueError("el presupuesto de computo tiene que ser positivo")

params = (compute_budget / (6 * tokens_per_param)) ** 0.5
tokens = tokens_per_param * params

return {"params": params, "tokens": tokens,
        "tokens_per_param": tokens_per_param, "compute": compute_budget}
```

La derivación, partiendo de $C = 6ND$ y $D = kN$:

$$C = 6N(kN) = 6kN^2 \quad \Longrightarrow \quad N = \sqrt{\frac{C}{6k}}$$

### La comprobación que da confianza

Esto no es un ejercicio de aritmética abstracta. Métele el presupuesto real de Chinchilla,
$5{,}88 \times 10^{23}$ FLOPs:

```
N = √(5,88·10²³ / 120) = 7,0·10¹⁰ = 70,0 mil millones
```

**El modelo real tenía 70.000 millones de parámetros.** La fórmula lo clava.

### La tabla que hizo famoso al paper

| modelo | params | tokens | tok/param | óptimo |
|---|---|---|---|---|
| GPT-3 | 175 B | 300 B | **1,7** | 51 B |
| Gopher | 280 B | 300 B | **1,1** | 65 B |
| Chinchilla | 70 B | 1,4 T | 20 | 70 B ✓ |
| Llama-3 8B | 8 B | 15 T | **1875** | 78 B |
| **el nuestro** | 7,6 M | 500 M | **66** | 14 M |

GPT-3 estaba **doce veces infra-entrenado**. Con su presupuesto de cómputo, lo óptimo habría
sido un modelo de 51 mil millones de parámetros —la tercera parte— entrenado con más del
triple de datos.

Y Llama-3 va **noventa veces por encima** de Chinchilla, lo cual no es un error: es que su
función objetivo es otra. Chinchilla optimiza el cómputo de **entrenamiento**; si el modelo
se va a ejecutar millones de veces después, conviene uno más pequeño y más entrenado, porque
la inferencia se paga cada vez.

Nuestro modelo está a 66 tokens por parámetro, tres veces por encima. Deliberado, por la
misma razón y porque a esta escala entrenar de más cuesta horas.

## Sobre lo que la fórmula no dice

Merece la pena tener presente la sección de debate del `TEORIA.md`: los coeficientes de
Chinchilla se ajustaron a un rango concreto de escalas, un reanálisis de 2024 encontró que
los intervalos de confianza eran mucho más amplios de lo reportado, y las leyes de escala
predicen **pérdida**, no capacidades.

Y sobre todo: ninguna ley de escala captura la **calidad de los datos**. El paper de
TinyStories muestra que un dataset pequeño y muy limpio permite a modelos diminutos generar
texto coherente, algo que no se consigue con la misma cantidad de texto de internet. Ningún
$N$ ni $D$ recoge eso.

---

## El código completo

Si te has atascado, aquí está la implementación entera. **Cópiala, pégala y ejecuta los
tests**: verlos pasar con código que entiendes es mejor que quedarte bloqueado.

Y después vuelve al ejercicio y escríbela tú. Leer una solución que ya has peleado funciona
muy bien; leerla en frío, no funciona nada.

```python
def model_flops_per_token(cfg: ModelConfig, include_backward: bool = True) -> dict[str, int]:
    d, ff, v = cfg.d_model, cfg.d_ff, cfg.vocab_size
    n_ffn = 3 if cfg.activation == "swiglu" else 2

    params_matmul = cfg.n_layers * (4 * d * d + n_ffn * d * ff) + d * v

    matmul = 2 * params_matmul
    attention = 4 * cfg.n_layers * cfg.context_length * d

    factor = 3 if include_backward else 1
    return {
        "matmul": matmul * factor,
        "attention": attention * factor,
        "total": (matmul + attention) * factor,
        "params_matmul": params_matmul,
    }


def compute_mfu(
    tokens_per_second: float, flops_per_token: int, peak_tflops: float
) -> float:
    if peak_tflops <= 0:
        raise ValueError("peak_tflops tiene que ser positivo")
    return tokens_per_second * flops_per_token / (peak_tflops * 1e12)


def chinchilla_optimal_allocation(
    compute_budget: float, tokens_per_param: float = 20.0
) -> dict[str, float]:
    if compute_budget <= 0:
        raise ValueError("el presupuesto de computo tiene que ser positivo")

    params = (compute_budget / (6 * tokens_per_param)) ** 0.5
    tokens = tokens_per_param * params

    return {
        "params": params,
        "tokens": tokens,
        "tokens_per_param": tokens_per_param,
        "compute": compute_budget,
    }
```

Los imports que hacen falta ya están en el `ejercicios.py` del módulo, salvo los que
aparezcan arriba del bloque.
