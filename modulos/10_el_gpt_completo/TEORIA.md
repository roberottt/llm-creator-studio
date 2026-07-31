# 10 — El GPT completo

## Por qué importa este módulo

**Porque aquí se junta todo y sale un número exacto.**

Tienes la atención, la normalización, el FFN y RoPE. Este módulo los ensambla en el modelo
que vas a entrenar, y termina con una comprobación que o cuadra o no cuadra:
**8.933.440 parámetros**. Ni uno más.

Ese número no es un adorno. Que la fórmula que derivas a mano coincida con el conteo real
del modelo significa que has entendido dónde está cada peso y por qué. Si no cuadra, algo
de tu arquitectura no es lo que crees.

Y hay tres decisiones de diseño aquí que se saltan casi todos los tutoriales y que son las
que hacen que el modelo entrene bien: el weight tying, la inicialización escalada por
profundidad, y la normalización final.

### Qué sabrás al terminar

- Cómo se monta un Transformer completo, de ids de token a logits
- Cómo ahorrar el 15% de los parámetros reutilizando una matriz que ya tienes
- Por qué la inicialización de las capas profundas **no puede ser la misma** que la del resto
- Verificar que tu modelo es causal de verdad, con una comprobación que da cero exacto

### Cuánto cuesta

3 horas. Cierra la Parte II: al terminarlo tienes el modelo montado y auditado.

---

## El bloque

Un bloque del Transformer es esto, y nada más:

```
x = x + atención(norm1(x))
x = x + ffn(norm2(x))
```

Dos sub-bloques, cada uno con su normalización pre-norm y su residual. La atención **mueve
información entre tokens**; el FFN **procesa cada token por separado**. Alternan.

Los dos residuales son independientes a propósito: cada sub-bloque puede aportar poco o
mucho a la corriente residual sin condicionar al otro.

## El modelo entero

```
ids de token
    ↓ tabla de embeddings
vectores
    ↓ bloque × 6
vectores
    ↓ normalización final
vectores
    ↓ proyección a logits
puntuaciones sobre los 4096 tokens
```

Con RoPE no hay embedding posicional que sumar al principio: la posición se inyecta dentro
de la atención, rotando Q y K. Por eso la primera capa es solo la tabla de tokens.

Ahora las tres decisiones que hacen que el modelo sea el que es.

## Decisión 1: weight tying

La tabla de embeddings convierte un id en un vector: es una matriz de $4096 \times 320$. La
capa de salida convierte un vector en puntuaciones sobre el vocabulario: es una matriz de
$320 \times 4096$.

**Son la misma matriz, transpuesta.** ¿Por qué no reutilizarla?

```python
self.lm_head.weight = self.token_embedding.weight
```

Eso no copia: hace que las dos capas apunten **al mismo tensor**. El ahorro:

```
sin tying:  4096 × 320 × 2 = 2.621.440 parámetros
con tying:  4096 × 320     = 1.310.720 parámetros
ahorro:                      1.310.720   (el 15% del modelo)
```

Con un modelo de 9M eso es enorme. Y además suele **mejorar la calidad**, no solo ahorrar:
cada peso recibe gradiente por dos caminos distintos —una vez como embedding de entrada, otra
como proyección de salida— así que se entrena con el doble de señal.

La justificación conceptual es que un token debería estar "cerca" en el espacio de
embeddings de aquellos con los que se puede confundir, y esa noción de cercanía sirve tanto
para leer como para escribir.

Detalle práctico: `model.parameters()` deduplica por identidad, así que el total sale bien
solo. Pero si desglosas por componente tienes que llevar cuenta de los `id()` ya vistos o
contarás la matriz dos veces.

## Decisión 2: inicialización escalada por profundidad

Este es el detalle que más gente se salta y que explica por qué a veces los modelos
profundos no entrenan bien.

Piensa en la corriente residual. Cada bloque le **suma** su contribución:

```
x₀ → x₁ = x₀ + algo₁ → x₂ = x₁ + algo₂ → ... → x₆
```

Si las contribuciones son independientes y todas tienen varianza $\sigma^2$, la varianza de
la suma crece **linealmente con el número de sumandos**. Con 6 capas y 2 sub-bloques cada
una, son 12 contribuciones: la salida tiene 12 veces más varianza que la entrada.

La solución de GPT-2, y la que usamos: inicializar con desviación más pequeña **solo las
proyecciones que escriben en la corriente residual**, que son `out_proj` de la atención y
`down_proj` del FFN:

$$\sigma = \frac{0{,}02}{\sqrt{2 \cdot n_{\text{layers}}}}$$

El 2 es porque cada bloque escribe dos veces. Con 6 capas: $0{,}02/\sqrt{12} = 0{,}0058$.

Todo lo demás se inicializa con $\sigma = 0{,}02$ a secas.

**Y el 0,02 tampoco es arbitrario.** Es lo que hace que la pérdida del paso 0 valga
$\ln(V)$: con logits casi idénticos, el softmax sale casi uniforme. Si inicializaras con la
normal estándar de PyTorch ($\sigma = 1$), el modelo arrancaría con opiniones fuertes y
aleatorias y la pérdida saldría por encima de $\ln(V)$ — exactamente lo que viste en la demo
del módulo 05.

## Decisión 3: la normalización final

En pre-norm, la corriente residual **nunca se normaliza por el camino**. Llega a la salida
con una escala que crece con la profundidad. Por eso hay una normalización justo antes de la
proyección a logits. No es opcional: sin ella, los logits salen con una escala arbitraria y
el entrenamiento es mucho más frágil.

## El conteo exacto

Y ahora el número. Deriva la fórmula tú antes de mirar:

| componente | fórmula | valor |
|---|---|---|
| embeddings de token | $V \cdot d$ | 4096 × 320 = **1.310.720** |
| atención por capa | $4d^2$ | 4 × 320² = 409.600 |
| SwiGLU por capa | $3 \cdot d \cdot d_{ff}$ | 3 × 320 × 896 = 860.160 |
| RMSNorm por capa | $2d$ | 2 × 320 = 640 |
| **por capa** | | **1.270.400** |
| × 6 capas | | **7.622.400** |
| RMSNorm final | $d$ | 320 |
| lm_head | atada | **0** |
| **TOTAL** | | **8.933.440** |

Tres cosas que conviene notar en esa tabla:

- La **atención no tiene sesgos**: son cuatro matrices $d \times d$ limpias. Los LLM
  modernos los han ido eliminando; aportan poco y complican el weight decay del módulo 11.
- **RMSNorm solo tiene escala** ($d$ parámetros), no escala y sesgo ($2d$) como LayerNorm.
- **RoPE no aporta ni un parámetro.** Las tablas de cos y sin se calculan con una fórmula y
  se guardan como *buffers*, no como parámetros. Por eso se registran con
  `persistent=False`: se recalculan al construir el modelo y no hace falta guardarlas en el
  checkpoint.

**Parámetros no-embedding: 7.622.720.** Ese es el número que usa el módulo 12 para las
leyes de escala, porque los embeddings escalan distinto al resto y Chinchilla los trata
aparte.

## Dónde está el debate

Merece la pena separar lo que está determinado de lo que es convención.

**Bien fundamentado:** el escalado por $\sqrt{d_k}$ de la atención (hay un argumento de
varianza claro), la necesidad de residuales, la normalización final en pre-norm.

**Convención con apoyo empírico:** pre-norm sobre post-norm, RMSNorm sobre LayerNorm,
SwiGLU sobre GELU, el weight tying. Funcionan mejor en los benchmarks; no hay teoría.

**Prácticamente arbitrario:** el 0,02 de la inicialización (viene de GPT-2 y nadie lo ha
vuelto a justificar), el factor 4x del FFN, el $\theta = 10000$ de RoPE, la proporción entre
profundidad y anchura. Se han probado alternativas y las diferencias son pequeñas.

Y una honesta sobre este modelo concreto: **6 capas de 320 dimensiones no es una elección
óptima derivada de nada**. Es un punto razonable para que quepa en una RTX 2060 y entrene en
horas. Con los mismos 9M de parámetros podrías hacer 12 capas de 224, o 3 de 512, y
funcionarían de forma parecida. La relación entre profundidad y anchura está poco explorada
a esta escala.

---

**Para ampliar:** Radford et al. 2019,
[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
(de donde salen la init escalada y el 0,02) · Press & Wolf 2017,
[Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
(weight tying) · [nanoGPT](https://github.com/karpathy/nanoGPT). Términos sueltos, en
[GLOSARIO.md](../../GLOSARIO.md).
