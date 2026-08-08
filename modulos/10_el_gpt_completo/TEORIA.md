# 10 — El GPT completo: juntarlo todo y auditarlo

## Por qué importa este módulo

**Porque aquí se junta todo y sale un número exacto.**

Tienes la atención, la normalización, el FFN y RoPE, cada uno probado por su cuenta. Este
módulo los ensambla en el modelo que vas a entrenar, y termina con una comprobación que o
cuadra o no cuadra: **8.933.440 parámetros**. Ni uno más.

Ese número no es un adorno. Que la fórmula que derivas a mano coincida con el conteo real del
modelo significa que has entendido dónde está cada peso y por qué. Si no cuadra, algo de tu
arquitectura no es lo que crees, y saberlo ahora es mucho más barato que descubrirlo a mitad de
un entrenamiento de horas.

### Qué sabrás al terminar

- Cómo se monta un Transformer completo, de ids de token a logits, con las formas de cada paso
- Las tres decisiones de diseño que se saltan casi todos los tutoriales —weight tying,
  inicialización escalada y normalización final— y que son las que hacen que el modelo entrene
- Qué es un *buffer* de PyTorch y por qué RoPE vive en uno
- Las tres comprobaciones con las que auditas el modelo antes de gastar un solo euro de GPU
- Dónde se va de verdad la memoria, que no es donde uno piensa

### Qué vas a escribir

Cuatro ejercicios:

| Ejercicio | Qué hace |
|---|---|
| 1. `expected_param_count` | La fórmula de cuántos parámetros tendrá |
| 2. `count_parameters` | Contarlos de verdad, desglosados |
| 3. `TransformerBlock` | Un bloque: atención + FFN con sus residuales |
| 4. `GPT` | El modelo entero |

Van en ese orden y esta teoría los sigue. Los dos primeros son de contar, y puedes hacerlos sin
haber montado nada: sus tests usan el modelo de referencia. De hecho ya tienes todos los números
—el módulo 06 te dio los 409.600 de la atención, el 07 los 4.160 de las normas, el 08 los 860.160
del FFN y el 09 te dijo que RoPE no cuesta ni un parámetro—, así que contar es el remate de los
cuatro módulos anteriores más que un ejercicio nuevo.

Una cosa sí conviene hacerla de otra manera: **el ejercicio 1, con papel, antes de teclear**. Si
vas directo al código acabarás probando números hasta que cuadre.

El ejercicio 3 son **dos líneas**. El 4 es el más largo del curso, y no por difícil: es que hay
muchas piezas y hay que colocarlas en orden.

### Cuánto cuesta

3 horas. Cierra la Parte II: al terminarlo tienes el modelo montado y auditado, listo para
entrenar en el módulo 11.

---

## El modelo entero, de un vistazo

Los cuatro módulos anteriores han ido produciendo piezas sueltas. Todas encajan aquí:

```
   módulo 06    MultiHeadAttention, causal_mask
   módulo 07    RMSNorm, la idea de pre-norm + residual
   módulo 08    SwiGLU
   módulo 09    rope_frequencies, apply_rope
```

Y una tranquilidad antes de empezar: **no hace falta que los cuatro estén terminados.** Arriba de
`ejercicios.py` las piezas se traen con `resolve(...)`, que es el bridge del curso: si tu
`SwiGLU` todavía lanza `NotImplementedError`, usa el de la referencia y te avisa por consola.
Puedes montar el GPT hoy y volver a completar el 08 mañana.

Así queda ensamblado, con la forma de los tensores en cada punto (`B` secuencias de `T` tokens,
`d_model=320`, `vocab=4096`):

```
   idx                       (B, T)          enteros: los ids de token del módulo 04
     │
     │  token_embedding      nn.Embedding(4096, 320)
     ▼
   x                         (B, T, 320)     cada id, convertido en su vector
     │
     │  drop                 dropout (0 en este curso)
     ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  bloque 1        x = x + attn(attn_norm(x), cos, sin, mask) │  ← ejercicio 3
   │                  x = x + ffn(ffn_norm(x))                   │
   │  ...             × 6                                        │
   └─────────────────────────────────────────────────────────────┘
     │
   x                         (B, T, 320)     misma forma que entró, seis veces enriquecida
     │
     │  norm_f               la normalización final
     ▼
   x                         (B, T, 320)
     │
     │  lm_head              nn.Linear(320, 4096, bias=False), atada a los embeddings
     ▼
   logits                    (B, T, 4096)    una puntuación por token del vocabulario,
                                             en CADA posición
```

Cuatro cosas de ese esquema que merecen un momento antes de seguir, porque las dos primeras
deciden el conteo de parámetros de la sección siguiente:

**La primera y la última capa son la misma matriz.** La tabla de embeddings convierte un id en un
vector: es de $4096 \times 320$. La `lm_head` convierte un vector en puntuaciones sobre el
vocabulario: es de $320 \times 4096$. **Son la misma matriz, transpuesta**, así que se reutiliza
una sola: es el *weight tying*, y ahorra 1.310.720 parámetros, el 15% del modelo. El cómo se
escribe está en el ejercicio 4; para contar basta con saber que la `lm_head` no cuesta nada.

**No hay embedding posicional al principio.** Con RoPE la posición se inyecta dentro de la
atención, rotando Q y K (módulo 09). Por eso la primera capa es sólo la tabla de tokens, y por
eso `cos` y `sin` viajan como argumentos hasta el fondo: el `forward` del GPT se los pasa a cada
bloque, y cada bloque a su atención. Y sus tablas **no son parámetros**: nadie las entrena, así
que tampoco cuentan.

**La forma no cambia en todo el cuerpo del modelo.** Entra `(B, T, 320)` en el bloque 1 y sale
`(B, T, 320)` del bloque 6. Ésa es la propiedad que permite apilar seis, o sesenta: cada bloque
es un *refinamiento* de la representación, no una transformación a otro espacio.

**Los logits son `(B, T, 4096)`: una predicción por posición**, no una por secuencia. `T`
predicciones en una sola pasada, que es la idea del módulo 04. Con `B=48` y `T=512` son cien
millones de números en un tensor, y eso tendrá consecuencias que veremos al final.

---

## Ejercicios 1 y 2: el conteo exacto

Ahora que sabes qué hay dentro, cuenta. Y hazlo **dos veces por caminos independientes**, que es
de lo que van estos dos ejercicios:

- **`expected_param_count(cfg)`** lo calcula con una fórmula, a partir del config, **sin
  construir el modelo**. Sirve para diseñar: cambias `d_model` en el YAML y sabes al instante si
  te cabe en la GPU.
- **`count_parameters(model)`** lo cuenta recorriendo el modelo ya construido, desglosado por
  componente.

Si los dos no dan el mismo número, **o tu fórmula o tu modelo mienten**, y hay que averiguar
cuál. Ésa es toda la gracia: es una auditoría cruzada, no un ejercicio de aritmética.

### La tabla, para que la derives tú primero

Con papel. Después compara:

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

Tres cosas que conviene notar:

- La **atención no tiene sesgos**: cuatro matrices $d \times d$ limpias, porque el config usa
  `bias=False`. Los LLM modernos los han ido eliminando; aportan poco y complican el weight decay
  del módulo 11.
- **RMSNorm sólo tiene escala** ($d$ parámetros), no escala y sesgo ($2d$) como LayerNorm. De ahí
  el $2d$ por bloque: son dos normas de $d$ cada una.
- **RoPE no aporta ni un parámetro.** Si tu cuenta lo incluye, está mal.

Dos errores que reconocerás por el número que da:

```
   te sale 10.244.160       ->  has olvidado el weight tying (la diferencia es exacta)
   te sale de más por poco  ->  estás contando sesgos que no existen
```

### El desglose del ejercicio 2

| componente | parámetros | % |
|---|---|---|
| embeddings | 1.310.720 | 14,7% |
| atención | 2.457.600 | 27,5% |
| FFN | 5.160.960 | **57,8%** |
| normas | 4.160 | 0,05% |
| lm_head | 0 | 0% (atada) |
| **TOTAL** | **8.933.440** | 100% |

**Parámetros no-embedding: 7.622.720.** Ése es el número que usa el módulo 12 para las leyes de
escala, porque los embeddings escalan distinto al resto: crecen con el vocabulario, no con la
profundidad. Por eso el ejercicio lo devuelve aparte.

Dos avisos de implementación:

**El `set` de `id()`.** Aquí es donde muerde el weight tying. `named_parameters()` deduplica por
identidad por defecto, así que el **total** sale bien solo; pero si sumas cada tensor donde te lo
encuentras, contarás la matriz compartida dos veces y te sobrarán exactamente 1.310.720. Y ojo
con la trampa de Python: `if param in vistos` **no vale**, porque `in` usa `==`, que en tensores
es elemento a elemento y revienta con *"Boolean value of Tensor is ambiguous"*. Compara por
`id()`.

**El orden de los `if/elif`.** Clasificas por subcadenas del nombre, y
`blocks.0.attn_norm.weight` contiene tanto `attn` como `norm`: quieres que cuente como norma.
Ordena de más específico a más general. Y antes de escribir nada:

```python
print([n for n, _ in model.named_parameters()])
```

En treinta segundos ves cómo está montado el modelo entero y qué subcadenas buscar. No adivines.

---

## Ejercicio 3: el bloque (`TransformerBlock`)

El `forward` entero son dos líneas:

```python
x = x + self.attn(self.attn_norm(x), cos=cos, sin=sin, mask=mask)
x = x + self.ffn(self.ffn_norm(x))
return x
```

Ése es el bloque del Transformer, la unidad que se repite seis veces y de la que se compone todo
lo que has leído sobre estos modelos.

Cada línea es un **pre-norm + residual** del módulo 07: normaliza, calcula algo, y **suma** el
resultado a lo que ya había. La `x` a la izquierda del `+` es la corriente residual, la autopista
que recorre el modelo de arriba abajo sin que nada la interrumpa. Y el reparto de trabajo es el
que ya conoces: la atención **mueve** información entre tokens, el FFN la **procesa** token a
token. Alternan, y esa alternancia es todo el Transformer.

Los dos residuales son independientes a propósito: cada sub-bloque puede aportar poco o mucho a
la corriente sin condicionar al otro.

En `__init__` creas cuatro cosas: dos normas y dos sub-bloques. Las normas salen de
`make_norm(cfg)` y el FFN de `make_ffn(cfg)`, dos ayudantes ya hechos que miran el config para
decidir si toca RMSNorm o LayerNorm, SwiGLU o el MLP clásico. No los reimplementes.

### Los cuatro sitios donde se falla

Los cuatro son silenciosos: el modelo se construye, entrena y da números plausibles.

**Normalizar la corriente en vez de la rama.** Es `x + attn(norm(x))`, no `norm(x + attn(x))`.
Lo segundo es post-norm y rompe la propiedad del módulo 07 que hace que el gradiente llegue
limpio a la capa 1.

**Reutilizar la misma norma para los dos sub-bloques.** `self.ffn_norm = self.attn_norm` compila
perfectamente y está mal: son dos objetos distintos con pesos propios, y por eso el conteo dará
`2 × d_model` por bloque y no `d_model`.

**Olvidar pasarle `cos`, `sin` o `mask` a la atención.** Sin `cos`/`sin` el modelo pierde toda la
información posicional y aun así entrena, mal. Sin `mask`, cada token ve el futuro y la pérdida
baja de forma sospechosamente buena.

**Pasarle `cos`/`sin`/`mask` al FFN.** No los acepta, y es que no los necesita: el FFN no mira a
otros tokens, así que no hay nada que enmascarar ni ninguna posición que inyectar.

---

## Ejercicio 4: el modelo entero (`GPT`)

El ejercicio más largo del curso. Vamos por partes, en el mismo orden en el que se escribe el
código, y cada decisión de diseño explicada donde toca escribirla.

### `__init__`, parte 1: los submódulos

La tabla de embeddings, el dropout, los seis bloques en un `nn.ModuleList`, la norma final y la
capa de salida. Sin misterio. Los nombres importan por partida doble: el test copia pesos por
nombre y el ejercicio 2 clasifica los parámetros por nombre.

### `__init__`, parte 2: weight tying

```python
if cfg.tie_embeddings:
    self.lm_head.weight = self.token_embedding.weight
```

Así se escribe el weight tying del que salía el ahorro de 1.310.720 parámetros. Una línea, y es
la primera de las tres decisiones del módulo.

Lo importante es que esa línea **no copia nada**: hace que los dos módulos apunten al mismo
tensor en memoria. Se comprueba con `is`, y hay un test que lo hace
(`test_los_pesos_atados_son_el_mismo_tensor`).

Y además de ahorrar, **suele mejorar la calidad**: cada peso recibe gradiente por dos caminos
distintos —una vez como embedding de entrada, otra como proyección de salida— así que se entrena
con el doble de señal. La justificación conceptual es que un token debería estar "cerca" en el
espacio de embeddings de aquellos con los que se puede confundir, y esa cercanía sirve tanto para
leer como para escribir.

### `__init__`, parte 3: las tablas de RoPE, y qué es un buffer

```python
cos, sin = rope_frequencies(cfg.head_dim, cfg.context_length, cfg.rope_theta)
self.register_buffer("rope_cos", cos, persistent=False)
self.register_buffer("rope_sin", sin, persistent=False)
```

Un `nn.Module` guarda dos clases de tensores. Los **parámetros** (`nn.Parameter`, los del módulo
07) son los que se entrenan: salen en `model.parameters()`, el optimizador los actualiza y
reciben gradiente. Los **buffers** son tensores que acompañan al modelo pero **no se entrenan**:
se mueven con `model.to(device)`, salen en el `state_dict`, y el optimizador ni los ve.

Las tablas de RoPE son el caso de libro. Se calculan con una fórmula cerrada, valen siempre lo
mismo y nadie las ajusta. Si las guardaras como parámetro, el optimizador intentaría entrenarlas
y romperías la propiedad relativa del módulo 09.

El `persistent=False` añade una cosa más: **tampoco se guardan en el checkpoint**. Como se
recalculan solas al construir el modelo, meterlas en el fichero de pesos sería desperdiciar
espacio. Hay un test que lo comprueba
(`test_rope_se_guarda_como_buffer_no_persistente`).

### `__init__`, parte 4: la inicialización escalada

Segunda decisión del módulo, y el detalle que más gente se salta.

```python
self.apply(self._init_weights)                    # 1. todo con std=0.02

scale = 0.02 / math.sqrt(2 * cfg.n_layers)        # 2. y después, PISANDO:
for name, param in self.named_parameters():
    if name.endswith(("out_proj.weight", "down_proj.weight", "fc_out.weight")):
        nn.init.normal_(param, mean=0.0, std=scale)
```

Dos pasadas: primero se inicializa todo igual, después se pisa el subconjunto que necesita otra
cosa. Es más simple que intentar acertar de una sola pasada.

**Por qué ese subconjunto.** Piensa en la corriente residual: cada sub-bloque le **suma** su
contribución. Si las contribuciones fueran independientes y todas de varianza $\sigma^2$, la
varianza de la suma crecería linealmente con el número de sumandos. Con 6 capas y 2 sub-bloques
cada una son 12 contribuciones, así que la salida tendría 12 veces más varianza que la entrada.

La solución de GPT-2, y la nuestra: reducir la desviación **sólo en las proyecciones que escriben
en la corriente residual**, que son `out_proj` de la atención (módulo 06) y `down_proj` del FFN
(módulo 08):

$$\sigma = \frac{0{,}02}{\sqrt{2 \cdot n_{\text{layers}}}}$$

El 2 es porque cada bloque escribe dos veces. Con 6 capas: $0{,}02/\sqrt{12} = 0{,}0058$. Puedes
verificarlo en tu propio modelo:

```
   std de q_proj.weight    ≈ 0.0200      <- el resto del modelo
   std de out_proj.weight  ≈ 0.0058      <- las que escriben en la corriente
   ratio                   ≈ 3.47        <- que es √12
```

**Y qué pasa si no lo haces.** Midiendo la norma media del vector de cada token al salir de cada
bloque, en el modelo real recién construido, con y sin escalar esas dos proyecciones:

| | con escalado | sin escalado |
|---|---|---|
| tras los embeddings | 0,356 | 0,356 |
| tras el bloque 2 | 0,565 | 1,844 |
| tras el bloque 4 | 0,888 | 3,516 |
| **tras el bloque 6** | **1,194** (×3,4) | **4,782** (×13,4) |

Sin escalar, la corriente sale del modelo trece veces más grande de lo que entró, y eso con seis
capas. Imagina con sesenta.

Una nota de honestidad sobre el argumento de arriba: predice un crecimiento de
$\sqrt{12} \approx 3{,}46$ en el caso sin escalar, y lo medido es 13,4. Las contribuciones **no
son independientes** de la corriente —cada bloque lee de ella, así que su salida ya escala con lo
que hay dentro— y el crecimiento acaba siendo multiplicativo en vez de aditivo. El argumento
explica bien la *dirección* del problema; no clava la magnitud.

**Y el 0,02 tampoco es arbitrario.** Es lo que hace que la pérdida del paso 0 valga $\ln(V)$: con
logits casi idénticos, el softmax sale casi uniforme. Con la normal estándar de PyTorch
($\sigma = 1$, el defecto de `nn.Embedding`) el modelo arrancaría con opiniones fuertes y
aleatorias y la pérdida saldría por encima — exactamente lo que viste medido en el módulo 05,
donde el bigrama neuronal arrancaba medio nat por encima del suelo.

### El `forward`

```python
B, T = idx.shape
if T > self.cfg.context_length:                   # 1. validar
    raise ValueError(...)

x = self.token_embedding(idx)                     # 2. ids -> vectores
x = self.drop(x)                                  # 3.
cos, sin = self.rope_cos, self.rope_sin           # 4. las tablas de la parte 3
mask = causal_mask(T, device=idx.device)          # 5. UNA vez, fuera del bucle
for block in self.blocks:                         # 6. los seis bloques
    x = block(x, cos=cos, sin=sin, mask=mask)
x = self.norm_f(x)                                # 7. la norma final
logits = self.lm_head(x)                          # 8. -> (B, T, 4096)

loss = None                                       # 9. la pérdida, si hay targets
if targets is not None:
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )
return logits, loss
```

Es el esquema del principio, línea a línea. Cuatro cosas que explicar:

**El paso 5, la máscara, va fuera del bucle.** Es exactamente la misma para las seis capas, así
que calcularla dentro funciona y desperdicia trabajo en cada uno de los 10.172 pasos del
entrenamiento final.

**El paso 7 es la tercera decisión del módulo**, `norm_f`. En pre-norm la corriente residual
nunca se normaliza por el camino: acabas de ver en la tabla que llega al final con una escala
×3,4 incluso haciendo bien la inicialización. Por eso hay una normalización justo antes de la
proyección a logits. No es opcional: sin ella los logits salen con una escala que depende de la
profundidad, la pérdida del paso 0 deja de valer $\ln(V)$ y el entrenamiento es más frágil. Es el
`norm_f` que se anunciaba al final del módulo 07, y ahora ya sabes de dónde sale.

**El `reshape(-1, vocab)` del paso 9** es el patrón del módulo 05 por tercera vez:
`F.cross_entropy` quiere `(N, V)` y `(N,)`, así que se aplanan batch y tiempo en una dimensión.

**El `ignore_index=-100`** no hace nada ahora mismo, porque ningún target vale −100. Lo
necesitarás en el módulo 16: en el fine-tuning por instrucciones se marca el prompt con −100 para
que el modelo no aprenda a predecirlo, sólo a predecir la respuesta. Déjalo puesto y te ahorras
volver.

Y fíjate en que devuelve `(logits, loss)` con `loss=None` si no hay targets, igual que el
`NeuralBigram` del módulo 05. La razón es la misma: al generar texto (módulo 14) no hay respuesta
correcta que valga, y quieres los logits para muestrear de ellos.

---

## Las tres comprobaciones

Al terminar tienes un modelo de nueve millones de parámetros que no ha entrenado ni un paso.
Éstas son las tres cosas que puedes verificar **antes** de gastar una hora de GPU, y las tres las
hace la demo.

**1. El número.** Fórmula, conteo y objetivo tienen que dar 8.933.440. Es la auditoría de los
ejercicios 1 y 2.

**2. La pérdida del paso 0.**

```
   pérdida del modelo sin entrenar : 8,3747
   ln(4096)                        : 8,3178
   desvío                          : +0,0569
```

El detector de bugs del módulo 05, aplicado al modelo de verdad. Fíjate en que **no da exacto,
sino un pelo por encima**, y eso es lo correcto: la inicialización con $\sigma = 0{,}02$ produce
logits casi idénticos, no idénticos, y con una muestra finita el promedio tampoco cae justo en la
media teórica. Un desvío de centésimas es normal; uno de varios nats no. Lo que importa es la
dirección:

```
   ≈ ln(V)      correcto, el modelo arranca sin opiniones
   bastante MÁS  la init es demasiado agresiva
   MÁS BAJA      fuga de información: mira la máscara, y luego cómo se monta el batch
```

Ese "y luego cómo se monta el batch" no es retórico. Los targets van **desplazados un token**
(`x = seq[:, :-1]`, `y = seq[:, 1:]`). Si pasaras `modelo(idx, idx)`, en la posición `t` el
modelo vería el token que tiene que predecir y la pérdida saldría por debajo de $\ln(V)$. Parece
un bug del modelo y es un bug de quien monta el batch, con un síntoma idéntico al de una máscara
rota.

**3. Es causal de verdad.** Cambia el token de la posición 6 y mira cuánto se mueven los logits
de cada posición:

```
   posición 0-5:  0.00e+00    <- cero EXACTO
   posición 6:    1.46e+00    <- el token cambiado
   posición 7:    2.52e-01
   posición 8:    2.32e-01
```

Cero exacto antes de la posición 6, no "muy pequeño": esas predicciones no pueden ver el token 6
de ninguna manera. A partir de ahí sí cambian. Eso es la máscara causal funcionando, y no hay
forma más limpia de demostrar que no hay fuga.

---

## Dónde se va la memoria

Una cuenta que te hará falta en el módulo 13, cuando la RTX 2060 se quede sin memoria y haya que
decidir dónde recortar:

| qué | MB |
|---|---|
| pesos del modelo (fp32) | 35,7 |
| gradientes (fp32) | 35,7 |
| estados de AdamW (dos momentos por parámetro) | 71,5 |
| **logits** en fp16, batch 48 × ctx 512 × vocab 4096 | 201,3 |
| + su versión fp32 (`cross_entropy` promociona) | 402,7 |
| + su gradiente | 402,7 |

**Los logits solos ocupan siete veces más que el modelo, los gradientes y el optimizador juntos**
(1007 MB frente a 143 MB). Es contraintuitivo y es la consecuencia directa de aquel
`(B, T, 4096)` del esquema: un número por cada token del vocabulario, en cada posición, de cada
secuencia del batch.

Cuando te quedes sin memoria, éste es el primer sitio donde mirar, no las activaciones del
modelo. La solución habitual es calcular la pérdida por trozos en vez de materializar el tensor
entero.

## Dónde está el debate

Acabas de montar un modelo con una docena de decisiones de diseño, y no todas tienen el mismo
respaldo. Merece la pena separarlas:

**Bien fundamentado:** el escalado por $\sqrt{d_k}$ de la atención (hay un argumento de varianza
claro, módulo 06), la necesidad de residuales, la normalización final en pre-norm.

**Convención con apoyo empírico:** pre-norm sobre post-norm, RMSNorm sobre LayerNorm, SwiGLU
sobre GELU, el weight tying. Funcionan mejor en los benchmarks; no hay teoría.

**Prácticamente arbitrario:** el 0,02 de la inicialización (viene de GPT-2 y nadie lo ha vuelto a
justificar), el factor 4x del FFN, el $\theta = 10000$ de RoPE, la proporción entre profundidad y
anchura.

Y una honesta sobre este modelo concreto: **6 capas de 320 dimensiones no es una elección óptima
derivada de nada.** Es un punto razonable para que quepa en una RTX 2060 y entrene en horas. Con
los mismos 9M de parámetros podrías hacer 12 capas de 224, o 3 de 512, y funcionarían de forma
parecida. La relación entre profundidad y anchura está poco explorada a esta escala, y ahora
tienes exactamente las herramientas para probarlo: cambia el YAML, corre `expected_param_count` y
entrena.

---

**Para ampliar:** Radford et al. 2019,
[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
(de donde salen la init escalada y el 0,02) · Press & Wolf 2017,
[Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
(weight tying) · [nanoGPT](https://github.com/karpathy/nanoGPT). Términos sueltos, en
[GLOSARIO.md](../../GLOSARIO.md).
