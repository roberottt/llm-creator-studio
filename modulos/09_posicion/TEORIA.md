# 09 — Información posicional y RoPE: decirle al modelo qué va antes

## Por qué importa este módulo

**Porque la atención no sabe qué palabra va antes.**

Suena a detalle y es un defecto fatal. Vuelve a mirar la fórmula del módulo 06: es una suma
ponderada, y una suma no tiene orden. Para el mecanismo de atención, *"el perro muerde al
hombre"* y *"el hombre muerde al perro"* producen exactamente lo mismo.

No es una forma de hablar; lo comprobaste con números en el módulo 06. Cogiendo el ejemplo de
tres tokens, intercambiando dos de ellos y mirando la salida de la última posición:

```
   orden original:  [0.4708, 0.5040]
   orden cambiado:  [0.4708, 0.5040]     idénticas
```

Aquí se arregla, y se arregla con una idea bastante bonita: en vez de sumarle al vector una
etiqueta que diga "soy la posición 7", se le aplica una **rotación** cuyo ángulo depende de la
posición. Eso hace que el modelo aprenda distancias relativas —"el token de dos posiciones
atrás"— en vez de posiciones absolutas.

Es la técnica que usan Llama, Mistral y prácticamente todo lo moderno. Y tiene una propiedad
que la hace todavía más llamativa: **no añade ni un solo parámetro al modelo.** Las tablas que
vas a calcular son fijas, no se entrenan. La alternativa clásica —una tabla aprendida— habría
costado 163.840 parámetros, el 1,8% del modelo; RoPE cuesta cero.

### Qué sabrás al terminar

- Por qué sin esto tu modelo no distingue el orden de las palabras
- Tres formas de resolverlo, en orden histórico, y qué falla en cada una
- Qué es RoPE y **la propiedad matemática que lo justifica**, comprobada con números que
  puedes reproducir
- **Cómo se emparejan las dimensiones para rotarlas**, que es lo que hace que el ejercicio 2
  parezca magia si nadie te lo enseña con una tabla delante
- Qué pasa de verdad cuando le pides a un modelo un contexto más largo del que entrenó, medido

### Qué vas a escribir

Tres ejercicios. Esta teoría está ordenada para que los leas en este orden, y **cada uno tiene
su propia sección con su ejemplo numérico**:

| Ejercicio | Qué hace | Dónde se explica |
|---|---|---|
| 1. `sinusoidal_embeddings` | La tabla del paper de 2017 (histórico) | [§ Senos y cosenos](#ejercicio-1-senos-y-cosenos-sinusoidal_embeddings) |
| 2. `rope_frequencies` | Precalcular los ángulos de rotación | [§ Las tablas de RoPE](#ejercicio-2-las-tablas-de-ángulos-rope_frequencies) |
| 3. `apply_rope` | Rotar Q y K | [§ Rotar](#ejercicio-3-rotar-de-verdad-apply_rope) |

El ejercicio 2 es el que cuesta, y cuesta por un solo paso: el que duplica las frecuencias. El
3 es una línea, pero sólo tiene sentido después de entender el 2. El 1 no lo usa nuestro
modelo y aun así merece la pena — su sección explica por qué.

### Cuánto cuesta

2,5 horas.

---

## Qué parte del LLM es esta

Con el módulo 08 cerraste el bloque del Transformer. Éste no añade una caja nueva al dibujo:
**se mete dentro de una que ya escribiste.**

Y de hecho ya lo has llamado. Vuelve al ejercicio 3 del módulo 06, el `MultiHeadAttention`.
Hay un paso que copiaste sin saber muy bien qué hacía:

```python
if cos is not None and sin is not None:
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
```

Ése `apply_rope` venía dado desde `llmfs.reference` con un comentario que decía "es del módulo
09, ignóralo por ahora". **Hoy lo escribes tú**, y los argumentos `cos` y `sin` de aquella
firma son exactamente lo que produce el ejercicio 2.

Dónde encaja, entonces:

```
   x  ──> q_proj ──> _split_heads ──>  q  (B, 8, T, 40)
                                        │
                                        ▼   apply_rope(q, cos, sin)      ← ejercicio 3
                                        │
                                     q rotada
                                        │
                                        ▼
                              q @ kᵀ / √d_k  ...  el resto del módulo 06
```

Tres cosas de ese esquema que conviene fijar antes de seguir:

- **Se aplica dentro de cada cabeza**, sobre `head_dim` dimensiones (40 en nuestro caso, o sea
  20 pares), no sobre las 320 de `d_model`. Por eso va *después* de `_split_heads`.
- **Sólo a Q y a K, nunca a V.** Lo que debe depender de la posición son las *puntuaciones* de
  atención, no el contenido que se transporta. Si rotaras también los valores, estarías
  metiendo la posición dos veces y ensuciando lo que el token aporta.
- **No es una capa.** No hay `nn.Module`, no hay pesos, no hay nada que entrenar. Las tablas
  `cos` y `sin` se calculan una vez al construir el modelo y se guardan como *buffers* — que
  es como PyTorch llama a los tensores que acompañan al modelo pero no son parámetros. En el
  GPT del módulo 10 los verás como `rope_cos` y `rope_sin`, de forma `(512, 40)`.

---

## El problema: la atención no sabe qué va antes

Vuelve a mirar la fórmula de la atención del módulo 06. Es una suma ponderada de los valores, y
los pesos salen de productos escalares entre queries y keys.

En ningún sitio aparece la **posición**.

La consecuencia es la del principio: si barajas los tokens de entrada, la salida se baraja
igual y nada más cambia. A esa propiedad se le llama **equivariancia a permutaciones**, y en
casi cualquier otro contexto sería una virtud (procesar un conjunto de cosas sin que el orden
importe). Aquí es un defecto fatal, porque el orden de las palabras es la mitad del significado.

Hay que meter la posición de alguna forma. Vamos a ver tres, en orden histórico, porque cada
una arregla un problema de la anterior y así se entiende de dónde sale RoPE.

## Opción 1: aprender una tabla

La más simple. Una tabla con una fila por posición, que se entrena como cualquier otro
parámetro, y se **suma** al embedding del token:

```
   entrada = embedding_de_token[id] + embedding_de_posición[i]
```

Es lo que hace GPT-2. Funciona bien y no tiene misterio: si la fila 7 acaba conteniendo lo que
haga falta para que el modelo sepa que está en la posición 7, problema resuelto.

Tiene dos pegas, y las dos importan.

La primera es un **techo duro**: si entrenaste con 1024 posiciones, para la posición 1025 no
hay fila que consultar. El modelo no puede procesarla de ninguna manera, ni mal. Lo verás
literalmente en la demo, donde la columna de esta opción pone "no puede" en cuanto la secuencia
pasa del contexto entrenado.

La segunda es más sutil: el modelo aprende posiciones **absolutas** — "esto es el token número
7" — cuando lo que suele importar es la relación: "esto está dos palabras antes del verbo". Y
como cada fila se entrena por separado, lo que aprenda sobre la posición 7 no le sirve para la
300, aunque la relación que representen sea la misma.

## Opción 2: senos y cosenos

El paper de 2017 propuso una tabla **fija**, sin parámetros, hecha de senos y cosenos de
distintas frecuencias:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Sigue siendo algo que se **suma** al embedding, como la opción 1. Lo que cambia es de dónde
salen los números: en vez de aprenderlos, se calculan.

### La intuición: un contador binario

Fíjate en cómo se cuenta en binario:

```
   0000    el bit de la derecha cambia en cada paso
   0001    el siguiente, cada dos
   0010    el siguiente, cada cuatro
   0011    ...
```

Cada bit oscila a un ritmo distinto, y la combinación de todos identifica un número de forma
única sin que ningún bit por sí solo tenga que hacer todo el trabajo. Las sinusoidales hacen lo
mismo pero con ondas continuas: los primeros pares de dimensiones oscilan rápido y distinguen
posiciones vecinas; los últimos oscilan lentísimo y distinguen el principio del final de la
secuencia.

Ventaja sobre la tabla aprendida: está definida para **cualquier** posición, no hay techo. Le
pides la fila 100.000 y te la calcula. En la práctica la extrapolación tampoco funciona muy
bien —eso lo mediremos al final— pero al menos existe.

---

## Ejercicio 1: senos y cosenos (`sinusoidal_embeddings`)

### Primero, por qué escribes algo que el modelo no usa

Conviene decirlo de entrada para que no te quedes con la mosca detrás de la oreja: **nuestro
modelo no usa esta función.** Usa RoPE, que es el ejercicio 2 y el 3. Ésta es la opción 2, la
de 2017, y está aquí por tres razones:

1. Es la que introduce la **escalera de frecuencias**, que es exactamente la misma idea que
   RoPE reutiliza. Si entiendes esta tabla, el ejercicio 2 deja de ser magia.
2. Aparece en muchísimo código y en todos los papers de la época; vas a encontrártela.
3. La demo la entrena de verdad y la compara con las otras dos, así que necesitas la
   implementación para que esa comparación exista.

### El ejemplo, con la tabla entera

Con `seq_len = 5` y `d_model = 4`, tu función tiene que devolver exactamente esto:

```
   posición 0:  [ 0.0000,  1.0000,  0.0000,  1.0000]
   posición 1:  [ 0.8415,  0.5403,  0.0100,  0.9999]
   posición 2:  [ 0.9093, -0.4161,  0.0200,  0.9998]
   posición 3:  [ 0.1411, -0.9900,  0.0300,  0.9996]
   posición 4:  [-0.7568, -0.6536,  0.0400,  0.9992]
                 └───┬───┘└───┬────┘└───┬───┘└──┬──┘
                   sin      cos       sin      cos
                  rápido   rápido    lento    lento
```

Léelo por columnas y verás la escalera. Las dos primeras columnas son $\sin(pos)$ y
$\cos(pos)$: oscilan deprisa, y de la posición 0 a la 4 ya han recorrido más de media vuelta.
Las dos últimas son $\sin(0{,}01 \cdot pos)$ y $\cos(0{,}01 \cdot pos)$: en cinco posiciones
apenas se han movido de `[0, 1]`. Con `d_model = 4` sólo hay dos velocidades; con 320 hay 160,
repartidas entre esos dos extremos.

Ese `0.01` sale de `div_term`, que para `d_model=4` vale `[1.0, 0.01]`. Es el paso 2 del
docstring, y merece la pena entender por qué está escrito con exponenciales:

```python
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model))
```

`exp(-log(base) · 2i/d)` es **matemáticamente idéntico** a `base ** (-2i/d)`, pero mucho más
estable numéricamente: elevar 10000 a una potencia negativa grande pierde precisión en coma
flotante, y hacerlo pasando por logaritmos no. Es una regla general que te servirá en otros
sitios: si ves una potencia con exponente grande, `exp(log(...))` suele ser mejor.

### El intercalado

El otro paso que puede despistar es el 4:

```python
embeddings[:, 0::2] = torch.sin(position * div_term)
embeddings[:, 1::2] = torch.cos(position * div_term)
```

`[:, 0::2]` significa "todas las filas, columnas desde 0 de dos en dos", o sea las **pares**;
`[:, 1::2]`, las impares. Así es como se intercalan seno y coseno sin escribir un bucle: cada
frecuencia ocupa dos columnas contiguas, una con el seno y otra con el coseno. En la tabla de
arriba se ve: columnas 0 y 1 comparten frecuencia, y 2 y 3 comparten la otra.

Y el `position * div_term` de dentro es un broadcast de `(T,1)` por `(d/2,)` que da `(T, d/2)`:
todos los ángulos de todas las posiciones de golpe, sin bucles. Es el mismo patrón que ya usaste
en el módulo 05 con las probabilidades del bigrama.

---

## Opción 3: RoPE, rotar en vez de sumar

Aquí está la idea que usa nuestro modelo, y Llama, y casi todo lo moderno.

**En lugar de sumar algo al vector, se le aplica una rotación cuyo ángulo depende de la
posición.**

Toma un vector de 2 dimensiones y rótalo un ángulo $\theta$. La matriz de rotación de toda la
vida:

$$\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} =
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

RoPE parte el vector de cada cabeza en pares y rota cada par un ángulo proporcional a la
posición. Como en las sinusoidales, cada par tiene su propia velocidad de giro: los primeros
giran deprisa, los últimos lentísimo. Con `head_dim = 40` son 20 pares, o sea 20 velocidades.

### Por qué esto es tan buena idea

Y aquí viene la propiedad que lo justifica todo. Las rotaciones tienen una particularidad:
**el producto escalar de dos vectores rotados depende sólo de la diferencia de ángulos.**

$$\langle R(m)\,q,\; R(n)\,k \rangle = \langle q,\; R(n-m)\,k \rangle$$

Traducido a lo que importa: la puntuación de atención entre el token de la posición 5 y el de
la 3 es **idéntica** a la que habría entre el 105 y el 103. Lo que el modelo aprende no es "el
token número 3" sino **"el token de dos posiciones atrás"**, y eso lo puede aplicar en
cualquier parte de la secuencia.

No hace falta que te lo creas. Cogiendo un mismo par de vectores `q` y `k`, colocándolos en
posiciones distintas y midiendo la puntuación entre ellos:

| posiciones (q, k) | distancia | puntuación |
|---|---|---|
| (0, 3) | 3 | −5,9859375954 |
| (2, 5) | 3 | −5,9859371185 |
| (10, 13) | 3 | −5,9859361649 |
| (100, 103) | 3 | −5,9859242439 |
| (200, 203) | 3 | −5,9859414101 |
| (0, 7) | 7 | −0,7609109879 |
| (50, 57) | 7 | −0,7609119415 |

Las cinco primeras filas están a distancia 3 y dan el mismo número hasta la séptima cifra
—las diferencias del final son redondeo de coma flotante, no el mecanismo—, aunque una esté al
principio de la secuencia y otra en la posición 200. Las dos últimas, a distancia 7, dan otro
valor pero también igual entre sí. Cuando termines el ejercicio 3 puedes reproducir esta tabla
tú mismo; hay un test que la comprueba.

Y hay un segundo beneficio, más discreto pero importante: **rotar no cambia la longitud del
vector**. Sumar un embedding posicional sí altera la magnitud, y como las puntuaciones de
atención son productos escalares —que dependen de las longitudes—, esa alteración se cuela en
las puntuaciones sin que nadie la haya pedido. Una rotación sólo cambia la dirección.

---

## Ejercicio 2: las tablas de ángulos (`rope_frequencies`)

Este ejercicio no rota nada: precalcula, para cada posición y cada par de dimensiones, el
coseno y el seno del ángulo que tocará. Se hace una vez al construir el modelo y se reutiliza
en los seis bloques y en los miles de pasos de entrenamiento.

Cuatro pasos, y sólo uno de ellos es raro. Vamos con un ejemplo diminuto, `head_dim = 4`, que
son dos pares.

### Paso 2: las frecuencias

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
```

Con `head_dim=4` y `theta=10000`: exponentes `[0/4, 2/4]`, o sea `inv_freq = [1.0, 0.01]`. Dos
velocidades, la misma escalera del ejercicio 1.

Para el modelo de verdad, con `head_dim=40`, la escalera queda así (medido):

| par | frecuencia (rad/posición) | da una vuelta completa cada |
|---|---|---|
| 0 | 1,000000 | 6 posiciones |
| 4 | 0,316228 | 20 posiciones |
| 8 | 0,100000 | 63 posiciones |
| 16 | 0,010001 | 628 posiciones |
| 24 | 0,000977 | 6.434 posiciones |
| 31 | 0,000000 | nunca |

Guárdate esa última fila: es el origen de todas las limitaciones de RoPE, y volvemos a ella al
final.

### Paso 3: todos los ángulos

```python
angles = torch.outer(positions, inv_freq)      # (T, head_dim/2)
```

`torch.outer(a, b)[i,j] = a[i] * b[j]`, que es justo lo que hace falta: todas las combinaciones
de posición por frecuencia. Con nuestro ejemplo, la fila de la posición 1 es `[1.0, 0.01]` y la
de la posición 2 es `[2.0, 0.02]`.

### Paso 4: el que confunde, y por qué

```python
angles = torch.cat([angles, angles], dim=-1)   # (T, head_dim)
```

Duplicar la tabla pegándola consigo misma. La primera vez que se ve, esto parece un error: ¿por
qué tener 4 columnas de ángulos si sólo hay 2 frecuencias?

La respuesta está en **cómo se emparejan las dimensiones para rotarlas**, y hay dos convenios:

```
   el paper original empareja CONSECUTIVAS:   (x0,x1), (x2,x3), ...
   Llama y HuggingFace, por MITADES:          (x0,x2), (x1,x3), ...   ← usamos éste
```

Con el convenio de mitades y `head_dim=4`, el par que rota junto es `(x0, x2)` y el otro es
`(x1, x3)`. Las dos componentes de un par necesitan **el mismo ángulo**, así que el ángulo de
la columna 0 tiene que repetirse en la columna 2, y el de la 1 en la 3. Y eso es exactamente lo
que hace el `cat`: convierte `[a, b]` en `[a, b, a, b]`.

Míralo en las tablas que tiene que devolver tu función con `head_dim=4`:

```
   cos = [[ 1.0000,  1.0000,  1.0000,  1.0000],     posición 0: ángulo 0, sin rotación
          [ 0.5403,  0.9999,  0.5403,  0.9999],     posición 1: cos(1.0) y cos(0.01)
          [-0.4161,  0.9998, -0.4161,  0.9998],     posición 2: cos(2.0) y cos(0.02)
          [-0.9900,  0.9996, -0.9900,  0.9996]]
                └── repetido ──┘

   sin = [[ 0.0000,  0.0000,  0.0000,  0.0000],
          [ 0.8415,  0.0100,  0.8415,  0.0100],
          [ 0.9093,  0.0200,  0.9093,  0.0200],
          [ 0.1411,  0.0300,  0.1411,  0.0300]]
```

Las columnas 0 y 2 son idénticas, y las 1 y 3 también. Si tu función devuelve algo así, el paso
4 está bien. Hay un test dedicado (`test_rope_duplica_las_frecuencias_por_mitades`).

Fíjate además en la primera fila: en la posición 0 el coseno vale 1 y el seno 0, es decir,
**ninguna rotación**. Tiene sentido: el primer token es el origen y no se mueve. Hay otro test
que lo comprueba.

Los dos convenios son equivalentes salvo una permutación de las dimensiones, que la red aprende
sin enterarse de nada. El de mitades ganó porque hace que el ejercicio 3 sea una línea sin
reordenar nada, y eso lo verás enseguida.

---

## Ejercicio 3: rotar de verdad (`apply_rope`)

Ahora sí. Y es una línea:

```python
return x * cos + rotate_half(x) * sin
```

con un ayudante de tres líneas:

```python
def rotate_half(x):
    mitad = x.shape[-1] // 2
    x1, x2 = x[..., :mitad], x[..., mitad:]
    return torch.cat([-x2, x1], dim=-1)
```

### Por qué esa línea es la matriz de rotación

Rotar un par `(x1, x2)` un ángulo `t` es lo de siempre:

```
   x1' = x1·cos(t) - x2·sin(t)
   x2' = x2·cos(t) + x1·sin(t)
```

Y ahora comprueba que la línea produce eso. Con `head_dim=4`,
`rotate_half([x0, x1, x2, x3]) = [-x2, -x3, x0, x1]`. Componente a componente:

```
   salida[0] = x0·cos[0] + (-x2)·sin[0]  =  x0·cos - x2·sin      el par (x0,x2), primera comp.
   salida[2] = x2·cos[2] + ( x0)·sin[2]  =  x2·cos + x0·sin      el par (x0,x2), segunda comp.
```

Y como `cos[0] == cos[2]` gracias al paso 4 del ejercicio anterior, las dos líneas usan **el
mismo ángulo**. Ahí está la razón de ser del `cat`: sin él, `salida[2]` habría usado la
frecuencia equivocada. Los dos convenios de emparejamiento, la duplicación y el `rotate_half`
son tres piezas de un mismo mecanismo, y por eso el ejercicio 2 hay que entenderlo antes que
éste.

### Compruébalo con números

Coge el vector `x = [1.0, 0.0, 0.0, 1.0]` y las tablas de `head_dim=4` de la sección anterior.
Tu función tiene que devolver:

```
   posición 0:  [ 1.0000,  0.0000,  0.0000,  1.0000]     intacto: ángulo 0
   posición 1:  [ 0.5403, -0.0100,  0.8415,  0.9999]
   posición 2:  [-0.4161, -0.0200,  0.9093,  0.9998]

   norma en las tres:  1.4142   (o sea √2, la del vector original)
```

Sigue la posición 1 a mano y se ve todo el mecanismo:

- **par (x0, x2) = (1, 0)**, ángulo 1,0 rad → `(cos·1 − sin·0, sin·1 + cos·0)` = `(0.5403,
  0.8415)`, que son las salidas 0 y 2. ✓
- **par (x1, x3) = (0, 1)**, ángulo 0,01 rad → `(cos·0 − sin·1, sin·0 + cos·1)` = `(−0.0100,
  0.9999)`, que son las salidas 1 y 3. ✓

Y la norma sigue valiendo √2 en las tres posiciones, que es la propiedad de la que hablábamos:
rotar no cambia la longitud.

### Los dos detalles que fallan si los saltas

**El recorte `cos[:seq_len]`.** Las tablas se precalculan hasta `max_seq_len` (512 en el modelo
final) y tu secuencia casi nunca mide eso exacto. Sin recortar, el broadcast falla o —peor—
acierta por casualidad con las formas equivocadas. Hay un test que pasa una secuencia más corta
que las tablas precisamente por esto.

**El `.to(dtype=x.dtype)`.** Bajo AMP las tablas están en fp32 y `x` llega en fp16. Mezclarlos
hace que PyTorch promocione al tipo más ancho, y acabas calculando en la precisión que no
querías, más lento y gastando más memoria. Es el mismo tipo de detalle que el `.float()` de
RMSNorm en el módulo 07, aquí en la dirección contraria. Hay un test que corre en fp16.

**Y no hace falta ningún `unsqueeze`.** Ésta es la duda que le entra a todo el mundo: `x` es
`(B, n_heads, T, head_dim)` y `cos` es `(T, head_dim)`. ¿No habría que alinearlas? No: el
broadcasting de PyTorch alinea **desde la derecha**, empareja `(T, head_dim)` con las dos
últimas dimensiones de `x` y repite el resto solo. Es lo correcto, además: la rotación de la
posición 5 es la misma para todas las cabezas y para todas las secuencias del batch.

---

## Lo que RoPE no arregla

Se dice mucho que RoPE "extrapola a contextos más largos". Es verdad a medias y conviene saber
exactamente dónde acaba, porque es de las afirmaciones que más se repiten sin matizar.

La demo entrena tres modelos idénticos salvo por la codificación posicional, todos con contexto
32, y los evalúa con contextos de 8 a 128:

| contexto | aprendida | sinusoidal | RoPE |
|---|---|---|---|
| 8 | 2,1924 | 2,1306 | 2,1168 |
| 16 | 2,1139 | 2,1088 | 2,0665 |
| **32 (entrenado)** | 2,1296 | 2,0823 | **2,0376** |
| 48 | *no puede* | 2,3490 | 2,1049 |
| 64 | *no puede* | 2,5117 | 2,2527 |
| 96 | *no puede* | 2,6723 | 2,4748 |
| 128 | *no puede* | 2,7601 | 2,6324 |

Tres lecturas:

1. **La tabla aprendida tiene un techo duro literal.** No es que lo haga mal: no hay fila que
   consultar y el modelo no puede procesar la secuencia en absoluto.
2. **Sinusoidal y RoPE sí producen una respuesta** para cualquier longitud, y RoPE gana en
   todas las filas.
3. **Y aquí la parte honesta: "poder procesar" no es lo mismo que "funcionar bien".** Los dos
   se degradan mucho. De contexto 32 a 128, sinusoidal empeora un 33% y RoPE un 29%. Es una
   ventaja, no una solución.

La razón está en la escalera de frecuencias del ejercicio 2, en aquella fila que dije que te
guardaras: las frecuencias lentas apenas completan una fracción de vuelta dentro del rango
entrenado. Si el modelo sólo ha visto ángulos entre 0 y 0,03 en el par 24, los ángulos que
aparecen en la posición 2000 son **territorio no visto**, y no hay ninguna razón para que
sepa qué hacer con ellos.

Por eso existe toda una familia de técnicas para extender el contexto *después* de entrenar
—interpolación de posiciones, NTK-aware scaling, YaRN—: porque la extrapolación directa no
basta. Cuando leas que "RoPE extrapola", esto es lo que hay detrás.

## Dónde está el debate

Además de lo de la extrapolación, que ya está medido arriba, hay algo más de fondo: **no está
claro por qué la codificación posicional relativa funciona mejor que la absoluta.** Hay
argumentos razonables sobre generalización —lo aprendido en una parte de la secuencia sirve en
otra— pero no un resultado que lo establezca.

Y hay un hallazgo que complica el asunto todavía más. Los transformers con máscara causal
**infieren cierta información posicional por su cuenta**, incluso sin ninguna codificación
explícita, porque la propia máscara rompe la simetría: el token de la posición 0 ve un token, el
de la posición 5 ve seis, y de ese número de vecinos visibles se puede deducir dónde estás. Hay
trabajos que entrenan modelos causales **sin ninguna codificación posicional** y funcionan
sorprendentemente bien.

O sea que ni siquiera está claro cuánto de necesario es todo este módulo. Lo que sí está claro
es que con RoPE los modelos salen mejores que sin él, y por eso lo usa todo el mundo.

---

**Para ampliar:** Su et al. 2021, [RoFormer](https://arxiv.org/abs/2104.09864) (RoPE) ·
Press et al. 2021, [ALiBi](https://arxiv.org/abs/2108.12409) (otra alternativa, que sesga las
puntuaciones en vez de rotar) · Vaswani et al. 2017 (las sinusoidales originales).
Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
