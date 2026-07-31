# 05 — Solución comentada

## Ejercicio 1 — `uniform_baseline_loss`

```python
if vocab_size < 1:
    raise ValueError("vocab_size debe ser positivo")
return math.log(vocab_size)
```

Una línea. Lo que importa no es el código, es **para qué lo vas a usar**.

Este número es tu detector de bugs más barato. Cuando arranques el entrenamiento del
módulo 11, la pérdida del paso 0 tiene que valer casi exactamente `ln(vocab_size)`:

| lo que ves en el paso 0 | qué significa |
|---|---|
| ≈ ln(V) | correcto, el modelo empieza sin opiniones |
| notablemente más alta | la inicialización es demasiado agresiva |
| más baja | fuga de información: casi siempre la máscara causal |

El caso "más baja" merece un momento de atención, porque parece una buena noticia y es el
bug más caro del curso. Si el modelo puede ver el token que tiene que predecir, la pérdida
baja a casi cero enseguida, todo parece ir de maravilla, y el modelo entrenado no sirve
para nada porque en generación ese futuro no existe.

**El demo te enseña el caso "más alta" en vivo.** El `NeuralBigram` arranca en ~4,64
cuando el suelo es 4,13, porque `nn.Embedding` inicializa por defecto con una normal de
desviación 1. Como esas filas *son* los logits, el modelo empieza con apuestas fuertes y
aleatorias. Ese medio nat de exceso es exactamente el precio de opinar sin información. Por
eso el GPT del módulo 10 usa `std=0.02` en todas partes.

## Ejercicio 2 — `bigram_counts`

```python
counts = torch.zeros(vocab_size, vocab_size, dtype=torch.int64)
tokens = torch.as_tensor(ids, dtype=torch.int64)
if tokens.numel() < 2:
    return counts
counts.index_put_(
    (tokens[:-1], tokens[1:]),
    torch.ones(tokens.numel() - 1, dtype=torch.int64),
    accumulate=True,
)
return counts
```

**`accumulate=True` no es opcional.** Sin él, `index_put_` *asigna* en vez de sumar: cada
par repetido pisa al anterior y todos los conteos acaban valiendo 1. El test
`test_las_repeticiones_se_acumulan_no_se_pisan` existe exactamente para eso: con
`[0,0,0,0,0]` el resultado correcto es 4, no 1.

**Por qué vectorizado.** Un bucle `for` funciona y es más legible, pero con 500M tokens son
500 millones de iteraciones de Python. `tokens[:-1]` da todos los "desde" y `tokens[1:]`
todos los "hasta"; PyTorch los procesa de golpe.

## Ejercicio 3 — `bigram_nll`

```python
tokens = torch.as_tensor(ids, dtype=torch.int64)
if tokens.numel() < 2:
    raise ValueError("hacen falta al menos 2 tokens")

smoothed = counts.double() + alpha
probs = smoothed / smoothed.sum(dim=1, keepdim=True)
selected = probs[tokens[:-1], tokens[1:]]
return float(-torch.log(selected).mean())
```

**El suavizado es el corazón del ejercicio.** Sin él, un solo par no visto en validación
tiene probabilidad 0, su logaritmo es `-inf`, y como la pérdida es una **media**, ese `-inf`
contamina el resultado entero. La perplejidad de todo tu conjunto de validación se va a
infinito por un par que no viste.

Sumar `alpha` a todo antes de normalizar es admitir que "no lo he visto" no es lo mismo que
"es imposible".

**El detalle del denominador.** Al sumar `alpha` a las `V` entradas de una fila, el total de
esa fila crece en `alpha * V`, no en `alpha`. Si normalizaras dividiendo por
`suma_original + alpha`, las probabilidades no sumarían 1. Hacer `smoothed.sum(dim=1)`
*después* de sumar alpha resuelve esto solo, sin que tengas que escribir el término.

**El `keepdim=True`.** Sin él, `sum(dim=1)` devuelve forma `(V,)` en vez de `(V, 1)`, y el
broadcast dividiría por *columnas* en lugar de por filas. El resultado sería numéricamente
plausible y completamente incorrecto. Es un bug clásico y silencioso; el test
`test_las_probabilidades_de_cada_fila_suman_uno` lo caza.

**`.double()` y no `.float()`.** Con corpus grandes se suman millones de conteos; float32
tiene 24 bits de mantisa y empieza a perder precisión antes de lo que uno espera.

## Ejercicio 4 — `NeuralBigram`

```python
def __init__(self, vocab_size):
    super().__init__()
    self.vocab_size = vocab_size
    self.token_embedding = nn.Embedding(vocab_size, vocab_size)

def forward(self, idx, targets=None):
    logits = self.token_embedding(idx)
    if targets is None:
        return logits, None
    loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
    return logits, loss
```

**Una `nn.Embedding(V, V)` como modelo de lenguaje** parece un truco y no lo es: la fila
`i` de la tabla son literalmente los logits del token que sigue a `i`. Entrenar esto con
cross-entropy converge a los conteos normalizados del ejercicio 2. El demo lo comprueba:
2,4916 contando y 2,4838 aprendiendo.

**Por qué `nn.Embedding` y no `nn.Linear`.** Son la misma operación: un Embedding es un
Linear cuya entrada es un vector one-hot. La diferencia es que el Embedding *lee* la fila
que necesita en lugar de multiplicar por una matriz llena de ceros. Con V=4096, leer 4096
números frente a hacer 16 millones de multiplicaciones.

**El `reshape(-1, V)`.** `F.cross_entropy` espera `(N, clases)` y `(N,)`, pero tú tienes
`(B, T, V)` y `(B, T)`. Aplanar batch y tiempo en una sola dimensión es el patrón que vas a
repetir en todos los modelos del curso, incluido el GPT final.

## Ejercicio 5 — `BengioMLP`

```python
def forward(self, idx, targets=None):
    batch = idx.shape[0]
    emb = self.embedding(idx)          # (B, block_size, d_embed)
    flat = emb.reshape(batch, -1)      # (B, block_size * d_embed)
    h = torch.tanh(self.hidden(flat))
    logits = self.output(h)            # (B, V)
    if targets is None:
        return logits, None
    return logits, F.cross_entropy(logits, targets)
```

**Concatenar, no promediar.** El `reshape(batch, -1)` pega los embeddings de los tokens uno
detrás de otro. Si en vez de eso hicieras `emb.mean(dim=1)`, el modelo perdería el orden: le
daría lo mismo `[el, gato, come]` que `[come, gato, el]`. El test
`test_bengio_concatena_en_vez_de_promediar` lo comprueba pasando el contexto al revés.

Cuidado con el `-1`: va en la **segunda** dimensión. Un `reshape(-1, batch)` compila y
produce basura.

**Su limitación, que es el motivo de que exista el módulo 06.** La capa `hidden` es
`Linear(block_size * d_embed, n_hidden)`, así que sus parámetros crecen **linealmente con
la longitud del contexto**. Con contexto 512 y `d_embed=320`, esa capa sola tendría
163.840 entradas.

Y hay un problema más profundo que el tamaño: el modelo trata cada posición como una
entrada independiente. No tiene forma de decir "de estos 512 tokens, los que me importan
ahora son el 3 y el 47". La atención resuelve las dos cosas a la vez.

## Lo que deberías ver en la demo

```
uniforme (azar)      4.1271   perplejidad 62.0
bigrama (conteo)     2.4916   perplejidad 12.1
bigrama (neuronal)   2.4838   perplejidad 12.0
Bengio MLP (ctx 4)   2.0939   perplejidad  8.1
```

Dos cosas que la demo señala y que no conviene pasar por alto.

La primera, que el bigrama contado y el aprendido dan **el mismo número**. Son el mismo
modelo por dos caminos.

La segunda, que el MLP de contexto 8 sale *peor* que el de contexto 4. No es un error: los
tres han entrenado los mismos 400 pasos y el de contexto 8 tiene el doble de parámetros, así
que se queda a medio entrenar. Comparar arquitecturas a igualdad de **pasos** no es
compararlas a igualdad de **cómputo**, y sistemáticamente favorece al modelo pequeño. Es
justo el error que las leyes de escala del módulo 12 vienen a corregir.
