# 09 — Solución comentada

## Ejercicio 1 — `sinusoidal_embeddings`

```python
position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)      # (T, 1)
div_term = torch.exp(
    torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(base) / d_model)
)                                                                        # (d/2,)

embeddings = torch.zeros(seq_len, d_model)
embeddings[:, 0::2] = torch.sin(position * div_term)
embeddings[:, 1::2] = torch.cos(position * div_term)
return embeddings
```

**El truco del `exp(-log(base) · 2i/d)`.** Es matemáticamente idéntico a `base ** (-2i/d)`,
pero mucho más estable. Elevar 10000 a una potencia negativa grande pierde precisión en
coma flotante; hacerlo pasando por logaritmos, no. Es el idioma estándar y merece la pena
tenerlo en el repertorio: siempre que veas una potencia con exponente grande, `exp(log(...))`
suele ser mejor.

**`position * div_term` emite solo.** `(T, 1)` por `(d/2,)` da `(T, d/2)`: todos los ángulos
de todas las posiciones de una vez, sin bucles.

**`0::2` y `1::2`** significan "desde 0 de dos en dos" y "desde 1 de dos en dos". Es la forma
de intercalar senos y cosenos sin escribir un `for`.

## Ejercicio 2 — `rope_frequencies`

```python
if head_dim % 2 != 0:
    raise ValueError(f"head_dim ({head_dim}) tiene que ser par")

inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
positions = torch.arange(max_seq_len, dtype=torch.float32)

angles = torch.outer(positions, inv_freq)          # (T, head_dim/2)
angles = torch.cat([angles, angles], dim=-1)       # (T, head_dim)

cos, sin = angles.cos(), angles.sin()
if device is not None:
    cos, sin = cos.to(device), sin.to(device)
return cos, sin
```

**El paso que confunde es la duplicación.** Las tablas salen con `head_dim` columnas, no
`head_dim/2`, y no es un error: es lo que hace que el ejercicio 3 sea una línea.

La razón está en el convenio de emparejamiento. Con el convenio de **mitades** que usamos
(el de Llama y HuggingFace), la dimensión `i` se empareja con la `i + head_dim/2`. Ambas
necesitan el **mismo ángulo**, así que cada frecuencia aparece dos veces, una en cada mitad
de la tabla. Con el convenio del paper original —emparejar consecutivas— habría que
intercalar en vez de concatenar, y `apply_rope` necesitaría reordenar dimensiones.

Los dos convenios son equivalentes salvo una permutación de las dimensiones, que la red
aprende sin enterarse. Uno es más limpio de implementar y por eso ganó.

**`torch.outer(a, b)[i,j] = a[i] * b[j]`.** Justo lo que hace falta: todas las combinaciones
posición × frecuencia.

## Ejercicio 3 — `apply_rope`

```python
def rotate_half(x):
    mitad = x.shape[-1] // 2
    x1, x2 = x[..., :mitad], x[..., mitad:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    seq_len = x.shape[-2]
    cos = cos[:seq_len].to(dtype=x.dtype, device=x.device)
    sin = sin[:seq_len].to(dtype=x.dtype, device=x.device)
    return x * cos + rotate_half(x) * sin
```

**De dónde sale esa línea.** Rotar un par $(x_1, x_2)$ un ángulo $t$:

```
x1' = x1·cos(t) − x2·sin(t)
x2' = x2·cos(t) + x1·sin(t)
```

Y ahora comprueba que `x * cos + rotate_half(x) * sin` produce exactamente eso, sabiendo
que `rotate_half([a, b]) = [-b, a]`:

```
componente 1:  x1·cos + (−x2)·sin  =  x1·cos − x2·sin    ✓
componente 2:  x2·cos + ( x1)·sin  =  x2·cos + x1·sin    ✓
```

Es la matriz de rotación de toda la vida, escrita con operaciones vectorizadas.

**El recorte `cos[:seq_len]`** no es opcional. Las tablas se precalculan hasta
`max_seq_len` (512 en el modelo final) y tu secuencia casi nunca tiene esa longitud
exacta. Sin recortar, el broadcast falla o —peor— acierta por casualidad con las formas
equivocadas.

**El `.to(dtype=x.dtype)`** importa bajo AMP: las tablas se crean en fp32 y `x` llega en
fp16. Mezclar tipos hace que PyTorch promocione, y acabas con la mitad del cálculo en la
precisión que no querías.

**No hace falta ningún `unsqueeze`.** `x` es `(B, n_heads, T, head_dim)` y `cos` es
`(T, head_dim)`; el broadcast alinea desde la derecha y se encarga solo de las dos primeras
dimensiones.

## Lo que deberías ver en la demo

**La invariancia relativa**, que es la propiedad que justifica todo:

| posiciones (q,k) | distancia | puntuación |
|---|---|---|
| (0, 3) | 3 | 0,1264068037 |
| (2, 5) | 3 | 0,1264068037 |
| (100, 103) | 3 | 0,1264068037 |
| (200, 203) | 3 | 0,1264068037 |

**Idénticas hasta el último decimal.** El modelo no aprende "el token número 3", aprende
"el token de tres posiciones atrás", y por eso puede aplicar lo aprendido en cualquier
parte de la secuencia.

**La extrapolación**, que es el experimento que de verdad importa. Tres modelos idénticos
salvo por la codificación posicional, entrenados con contexto 32:

| contexto | aprendida | sinusoidal | RoPE |
|---|---|---|---|
| 16 | 2,1139 | 2,1088 | 2,0665 |
| **32 (entrenado)** | **2,1296** | **2,0823** | **2,0376** |
| 48 | **no puede** | 2,3490 | 2,1049 |
| 128 | **no puede** | 2,7601 | 2,6324 |

Tres lecturas.

**La aprendida tiene un techo duro.** No es que funcione mal más allá de 32: es que *no
puede*. No hay fila en la tabla que consultar. El código lanza una excepción, y eso es lo
honesto.

**RoPE gana en todos los contextos**, incluido dentro del rango entrenado. La codificación
relativa ayuda incluso sin extrapolar.

**Y las dos se degradan.** De 2,04 a 2,63 es un 29% peor. Aquí conviene tener cuidado con lo
que se lee por ahí: se repite mucho que "RoPE extrapola", y lo que la demo enseña es que
extrapola *mejor que las alternativas*, no que extrapole *bien*.

La razón: las frecuencias lentas apenas completan una fracción de vuelta dentro del rango
entrenado —mira la tabla de períodos del primer experimento— así que los ángulos grandes son
literalmente territorio no visto. Existe toda una familia de técnicas para extender el
contexto después de entrenar (interpolación de posiciones, NTK-aware scaling, YaRN)
precisamente porque la extrapolación directa no basta.

---

## El código completo

Si te has atascado, aquí está la implementación entera. **Cópiala, pégala y ejecuta los
tests**: verlos pasar con código que entiendes es mejor que quedarte bloqueado.

Y después vuelve al ejercicio y escríbela tú. Leer una solución que ya has peleado funciona
muy bien; leerla en frío, no funciona nada.

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    mitad = x.shape[-1] // 2
    x1, x2 = x[..., :mitad], x[..., mitad:]
    return torch.cat([-x2, x1], dim=-1)


def sinusoidal_embeddings(seq_len: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    # exp(-ln(base) * 2i/d) es numericamente mas estable que base ** (-2i/d)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(base) / d_model)
    )  # (d/2,)

    embeddings = torch.zeros(seq_len, d_model)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings


def rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) tiene que ser par: RoPE rota pares.")

    # theta^(-2i/d) para i = 0, 1, ..., d/2-1
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)

    angles = torch.outer(positions, inv_freq)  # (T, head_dim/2)
    angles = torch.cat([angles, angles], dim=-1)  # (T, head_dim), duplicado por mitades

    cos, sin = angles.cos(), angles.sin()
    if device is not None:
        cos, sin = cos.to(device), sin.to(device)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[-2]
    cos = cos[:seq_len].to(dtype=x.dtype, device=x.device)
    sin = sin[:seq_len].to(dtype=x.dtype, device=x.device)
    return x * cos + rotate_half(x) * sin
```

Los imports que hacen falta ya están en el `exercises.py` del módulo, salvo los que
aparezcan arriba del bloque.
