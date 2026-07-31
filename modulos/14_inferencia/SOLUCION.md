# 14 — Solución comentada

## Ejercicio 1 — `apply_repetition_penalty`

```python
if penalty == 1.0:
    return logits

salida = logits.clone()
for fila in range(logits.shape[0]):
    vistos = torch.unique(generated[fila])
    valores = salida[fila, vistos]
    salida[fila, vistos] = torch.where(valores > 0, valores / penalty, valores * penalty)
return salida
```

**El `torch.where` es todo el ejercicio.** Positivos se dividen, negativos se multiplican.

Compruébalo con los números del demo, con `penalty=2.0`:

```
logit +3.0  →  3.0 / 2.0 = +1.50    menos probable  ✓
logit −3.0  →  −3.0 × 2.0 = −6.00   menos probable  ✓
```

Si dividieras siempre, el −3,0 pasaría a −1,5 y el token se volvería **más** probable. Y
como los logits negativos son la mayoría del vocabulario, estarías premiando casi todo lo
que ya salió: exactamente lo contrario de lo que pretendes.

El `torch.unique` evita penalizar dos veces un token que salió dos veces. Hay
implementaciones que sí acumulan; nosotros no, para que el efecto sea predecible.

## Ejercicio 2 — `top_k_filter`

```python
if k <= 0 or k >= logits.shape[-1]:
    return logits

umbral = torch.topk(logits, k, dim=-1).values[..., -1:]
return logits.masked_fill(logits < umbral, float("-inf"))
```

**El `[..., -1:]` con dos puntos** conserva la dimensión para que el broadcast funcione. Con
`[..., -1]` la perderías y `masked_fill` compararía mal.

**`<` y no `<=`**: el propio umbral —el k-ésimo logit— tiene que sobrevivir.

## Ejercicio 3 — `top_p_filter`

```python
if p >= 1.0:
    return logits

ordenados, indices = torch.sort(logits, descending=True, dim=-1)
probs = F.softmax(ordenados, dim=-1)
acumulada = torch.cumsum(probs, dim=-1)

quitar = acumulada - probs > p
quitar[..., 0] = False

a_quitar = quitar.scatter(-1, indices, quitar)
return logits.masked_fill(a_quitar, float("-inf"))
```

### El off-by-one que yo mismo tuve mal

Escribiendo este módulo puse en la teoría que `[0.60, 0.25, 0.10, 0.03, 0.02]` con `p=0.9`
dejaba **2** candidatos. El test dijo 3, y el test tenía razón.

La definición de Holtzman es *"el conjunto más pequeño cuya probabilidad acumulada **excede**
p"*. Y `0.60 + 0.25 = 0.85` **no** excede 0,9. Hace falta el tercero, que lleva la masa a
0,95.

Por eso la comparación es `acumulada - probs > p`: se mira la acumulada **antes** de incluir
cada token, de forma que el que cruza el umbral todavía entra. Si compararas `acumulada > p`
a secas, cortarías uno de más.

Es el mismo criterio que usa la implementación de HuggingFace, que lo resuelve desplazando
la máscara una posición a la derecha.

### El `quitar[..., 0] = False`

Con `p=0.5` y un token de probabilidad 0,9, sin esa línea no quedaría ningún candidato y
`torch.multinomial` reventaría. El token más probable **siempre** sobrevive.

### El `scatter`, que es lo que más cuesta ver

Has ordenado los logits, así que las marcas de "quitar" están en orden de probabilidad, no
en orden de token. `scatter(-1, indices, quitar)` las devuelve a su sitio: para cada posición
`j` del tensor ordenado, escribe su marca en la posición `indices[j]` del resultado.

## Ejercicio 4 — `KVCache`

```python
def update(self, layer, k, v):
    if self.keys[layer] is None:
        self.keys[layer] = k
        self.values[layer] = v
    else:
        self.keys[layer] = torch.cat([self.keys[layer], k], dim=-2)
        self.values[layer] = torch.cat([self.values[layer], v], dim=-2)
    return self.keys[layer], self.values[layer]
```

**`dim=-2`** es la dimensión de tiempo con la forma `(B, n_heads, T, head_dim)`. Índice
negativo: con `dim=2` funcionaría aquí y se rompería si algún día cambia el número de
dimensiones.

Lo demás es contabilidad. La clase es deliberadamente sencilla porque la dificultad no está
aquí, está en el ejercicio 5.

## Ejercicio 5 — `generate_with_cache`

### El detalle que rompe todo

**RoPE tiene que rotar el token nuevo con el ángulo de su posición real.**

Al generar el token 50 le pasas un tensor de longitud 1. Si aplicas RoPE tal cual, lo rota
como si fuera la posición 0. El resultado: la generación con cache produce texto **distinto
y peor** que sin ella, y nada falla — simplemente el modelo escribe mal.

Por eso la atención recibe `pos_offset` y recorta las tablas:

```python
cos_t = cos[pos_offset : pos_offset + seq_len]
```

El test `test_la_cache_da_exactamente_la_misma_salida` es lo que lo caza, y su mensaje de
error apunta directamente aquí.

### El orden de los filtros

```
penalización → temperatura → top-k → top-p
```

La temperatura va antes de los filtros porque cambia las probabilidades acumuladas que mira
top-p. (No cambia el *ranking*: dividir por una constante positiva no reordena nada.)

### El `.float()` de los logits

Bajo AMP los logits llegan en fp16, y `torch.multinomial` sobre fp16 puede dar resultados
raros con probabilidades muy pequeñas. Convertir a fp32 antes de muestrear es barato.

### Un bug que encontré escribiendo el demo

La primera versión reventaba al generar más allá del contexto: `model.generate` recorta con
`idx[:, -context_length:]`, pero con cache eso no vale.

Recortar con cache exigiría descartar las entradas antiguas **y remapear las posiciones de
RoPE** de todo lo que queda, porque los tokens supervivientes pasarían a ocupar posiciones
distintas de aquellas con las que se rotaron. Eso es *sliding window attention* y da para un
módulo entero.

La solución que adopté es parar limpiamente al llegar al límite, y lanzar un `ValueError`
claro si el prompt ya lo llena. Parar es lo honesto: la alternativa silenciosa sería generar
texto incorrecto sin avisar.

## Lo que deberías ver en la demo

**El bucle de greedy**, medido como fracción de 4-gramas distintos:

| estrategia | variedad | texto |
|---|---|---|
| greedy (T=0) | **29%** | `The king of the sea of the sea That shall see the sea of the sea` |
| T=0,8 + top-k 40 | 96% | `The king; To bring what heart you but dead-look'd me to-morrow` |
| T=1,5 | 100% | `Tak't I am fan undooses our very looks, Were stewest, grounde;` |
| greedy + penalty 1,3 | 93% | `The king, As to my lady's brother with the prince.` |

Greedy se atasca en `the sea of the sea` de forma perfectamente visible. **El texto humano no
maximiza la probabilidad**, y esa es la observación central de Holtzman et al.

Fíjate también en la última fila: la penalización rescata a greedy sin quitarle el
determinismo. Y en T=1,5, donde el 100% de variedad es señal de que ya desvaría.

**Y la cache:**

```
sin cache: [43, 1, 57, 43, 39, 0, 32, 46, 39, 58]
con cache: [43, 1, 57, 43, 39, 0, 32, 46, 39, 58]   IDÉNTICOS
```

| tokens | sin cache | con cache | speedup |
|---|---|---|---|
| 50 | 159 ms | 133 ms | 1,20x |
| 200 | 1115 ms | 833 ms | 1,34x |
| 800 | 4330 ms | 1962 ms | **2,21x** |

**El speedup crece con la longitud**, que es exactamente lo que predice el análisis: sin
cache es $O(N^2)$ y con cache $O(N)$. Con las secuencias cortas de este ejemplo la ganancia
es modesta; con contextos de miles de tokens, la diferencia es de otro orden.
