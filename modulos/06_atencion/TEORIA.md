# 06 — Self-attention: dejar que cada token elija a qué hacer caso

## Por qué importa este módulo

**Si sólo pudieras entender un módulo del curso, sería éste.**

Todo lo anterior —tokenizar, embeddings, el MLP de Bengio— existía ya en 2003 y daba modelos
mediocres. Lo que cambió en 2017 y acabó produciendo ChatGPT es exactamente lo que vas a
programar aquí, y el núcleo cabe en cuatro líneas de código.

La idea es sencilla de enunciar: **dejar que cada token mire a los anteriores y decida a
cuáles hacer caso**. Lo difícil es creer que con eso baste. Al terminar el módulo lo habrás
visto funcionar en un modelo que entrenas tú, con un mapa de calor que muestra literalmente
a qué mira cada letra.

Y conviene decir de entrada lo que este módulo *no* es: no es teoría que luego aplicarás. Las
tres funciones que escribes aquí son, sin cambiar una línea, las que ejecuta el modelo final.
El 27,5% de los 8.933.440 parámetros que vas a entrenar viven dentro del ejercicio 3.

### Qué sabrás al terminar

- Por qué un modelo puede "recordar" algo que leyó 300 palabras antes
- Qué son Q, K y V, y **por qué hacen falta tres cosas y no una** (con el ejemplo numérico en
  el que usar una sola da la respuesta equivocada)
- De dónde salen las formas `(B, T, d_k)` y `(B, n_heads, T, head_dim)`, que es donde se
  atasca casi todo el mundo en el ejercicio 3
- Por qué se divide por `√d_k`, y qué se rompe exactamente si lo quitas (con números medidos)
- Qué es la máscara causal y por qué es **el bug más caro del curso** si la pones mal
- Qué hace la cuarta proyección, `out_proj`, que no aparece en la fórmula del paper y sin la
  cual el multi-head no sirve de nada
- Qué es lo que la atención **no** sabe hacer, y qué módulo lo arregla
- Habrás visto un heatmap de atención de un modelo entrenado por ti

### Qué vas a escribir

Tres ejercicios, y encajan uno dentro del siguiente. Esta teoría está ordenada para que los
leas en este orden, y **cada uno tiene su propia sección con su ejemplo numérico**:

| Ejercicio | Qué hace | Dónde se explica |
|---|---|---|
| 1. `causal_mask` | La matriz triangular que prohíbe mirar al futuro | [§ La máscara causal](#ejercicio-1-la-máscara-causal-causal_mask) |
| 2. `single_head_attention` | La fórmula entera, con una cabeza | [§ La atención de una cabeza](#ejercicio-2-la-atención-de-una-cabeza-single_head_attention) |
| 3. `MultiHeadAttention` | Ocho en paralelo, que es lo que usa el modelo | [§ Multi-head](#ejercicio-3-ocho-cabezas-en-paralelo-multiheadattention) |

```
    causal_mask  ──────────┐
                           ▼
    q, k, v  ────>  single_head_attention  ────>  salida, pesos
                           │
                           │  el mismo cálculo, con una dimensión más
                           ▼
                   MultiHeadAttention   ← esto es lo que va dentro del GPT
```

El ejercicio 1 es una línea. El 2 son cuatro, y cada una tiene una trampa. El 3 es el mismo
cálculo del 2 con una dimensión extra, más la fontanería de partir y volver a juntar las
cabezas: es largo, pero no es más difícil, y la sección correspondiente lo desmonta pieza a
pieza.

### Cuánto cuesta

4 horas, empatado con el 03 como el más largo. Es el que más merece la pena.

---

## Qué parte del LLM es esta

Construir un LLM son cuatro trabajos distintos, y el curso los recorre en este orden:

```
   0. FUNDAMENTOS      qué es un LLM, PyTorch, autograd        módulos 00-02   ✔ hecho
   1. TOKENIZADOR      texto  ->  números                      módulo 03       ✔ hecho
   2. DATOS            números  ->  tarea de aprendizaje       módulo 04       ✔ hecho
   3. MODELO           la arquitectura que hace la predicción  módulos 05-10   ← ESTÁS AQUÍ
   4. ENTRENAMIENTO    ajustar los pesos hasta que acierte     módulos 11-13
```

En el módulo 05 los tres baselines eran callejones sin salida a propósito. Éste no: aquí
empieza el modelo de verdad. Y para situar la pieza, así es como se ve un bloque del
Transformer que montarás entero en el módulo 10:

```
    x ──┬──> norma ──> ATENCIÓN (este módulo) ──┐
        │                                       ├──> +  ──┬──> norma ──> MLP ──┐
        └───────────────────────────────────────┘         │                    ├──> +
                                                          └────────────────────┘

         módulo 07: la norma y esas dos sumas (conexiones residuales)
         módulo 08: el MLP
         módulo 09: cómo se le dice al modelo en qué posición está cada token
         módulo 10: apilar seis de estos bloques y montar el GPT
```

Seis bloques como ése, cada uno con su atención. En parámetros:

```
   una capa de atención:  4 proyecciones de 320x320  =    409.600
   las seis capas:                                       2.457.600
   el modelo entero:                                     8.933.440   -> el 27,5%
```

Ese 27,5% no es lo que más ocupa (el MLP del módulo 08 tiene más), pero es lo que hace que el
modelo pueda relacionar cosas separadas en el texto. Sin ello tendrías una red que procesa
cada posición por su cuenta, que es exactamente el MLP de Bengio del módulo anterior.

---

## El problema que resuelve

Frase: *"el gato que vi ayer dormía"*.

Para acertar `dormía` hay que saber que el sujeto es `gato`, cuatro palabras atrás. El MLP
del módulo 05 no puede, y merece la pena tener fresco por qué, porque la atención está
diseñada exactamente contra esas dos limitaciones:

- **Concatenaba** los `k` vectores anteriores y los metía en una capa lineal. El peso de "la
  posición cuatro tokens atrás" era un número fijo en la matriz, el mismo para todas las
  frases del corpus. No había forma de decir *"en esta frase concreta, el que me importa es
  el primero"*.
- Y el tamaño de esa capa **crecía con la longitud del contexto**, así que 512 tokens era
  inalcanzable.

La atención resuelve las dos cosas de un golpe: los pesos se **calculan a partir del
contenido** en cada frase, y el número de parámetros no depende de la longitud del contexto
en absoluto. Las cuatro matrices de 320×320 del ejercicio 3 son las mismas tanto si le pasas
10 tokens como si le pasas 512.

## La idea, con números que puedes seguir a mano

Vamos a hacerlo con 3 palabras y vectores de 2 dimensiones. Después de los embeddings
tenemos:

```
   gato   = [1.0, 0.2]
   ayer   = [0.1, 0.9]
   dormía = [0.3, 0.4]
```

`dormía` quiere saber a quién mirar. Lo hace con un **producto escalar**, que mide cuánto se
parecen dos vectores: se multiplican componente a componente y se suma. Cuanto más alineados,
mayor el número.

```
   dormía · gato   = 0.3×1.0 + 0.4×0.2 = 0.38
   dormía · ayer   = 0.3×0.1 + 0.4×0.9 = 0.39
   dormía · dormía = 0.3×0.3 + 0.4×0.4 = 0.25
```

Esos números se llaman **puntuaciones** (*scores*). Se dividen por `√d_k` —aquí `√2 = 1.414`,
y en la sección del escalado verás por qué— y se convierten en pesos que sumen 1 con el
softmax, que ya conoces del módulo 05: exponenciar y normalizar.

```
   escaladas:  [0.269, 0.276, 0.177]
   softmax:    [0.343, 0.345, 0.313]
```

Y con esos pesos se mezclan los vectores:

```
   salida = 0.343×gato + 0.345×ayer + 0.313×dormía
```

Eso es la atención: **una media ponderada donde los pesos los decide el propio contenido.**

Ahora mira bien el resultado, porque es malo: los tres pesos son casi iguales y `ayer` gana
por un pelo a `gato`. `dormía` se lleva una papilla de las tres palabras en la que la que
importaba no destaca. Y con los embeddings tal cual no hay nada que hacer al respecto, porque
la similitud entre dos vectores es la que es. Esto lleva directo a la siguiente sección, que
es donde el mecanismo se vuelve utilizable.

## Q, K, V: por qué tres proyecciones y no una

En el ejemplo he usado el mismo vector para todo, y es demasiado rígido. Un token necesita
hacer tres cosas distintas, y con un solo vector las tres están atadas entre sí:

- **preguntar** algo ("busco un sujeto singular")
- **anunciarse** ante los demás ("soy un sustantivo singular")
- **aportar** contenido si resulta elegido ("el concepto de gato")

Son tres papeles diferentes, así que se aprenden **tres proyecciones lineales distintas** del
mismo vector de entrada:

$$Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V$$

**Query** (pregunta), **Key** (etiqueta) y **Value** (contenido). La similitud se calcula
entre queries y keys; lo que se mezcla son los values.

### El mismo ejemplo, ahora con proyecciones

Interpretemos las dos dimensiones de nuestros vectores como "cuánto tiene de sustantivo" y
"cuánto tiene de referencia temporal":

```
   gato   = [1.0, 0.2]     mucho sustantivo, poco tiempo
   ayer   = [0.1, 0.9]     poco sustantivo, mucho tiempo
   dormía = [0.3, 0.4]     un verbo: ni una cosa ni la otra
```

Lo que el modelo aprende en $W_Q$ es, literalmente, "cuando seas un verbo, pregunta por
sustantivos". Una matriz que hace eso es:

$$W_Q = \begin{pmatrix} 0 & 0 \\ 2.5 & 0 \end{pmatrix}, \qquad W_K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

Con $W_K$ dejando las keys como estaban, la query de `dormía` sale
$[0.3, 0.4] W_Q = [1.0,\ 0.0]$: un vector que apunta puramente en la dirección "sustantivo".
Y las puntuaciones cambian por completo:

```
                     sin proyecciones      con Q y K aprendidas
   dormía -> gato        0.38                    1.00
   dormía -> ayer        0.39                    0.10
   dormía -> dormía      0.25                    0.30

   softmax (÷√2):    [0.343, 0.345, 0.313]   [0.468, 0.247, 0.285]
```

De un empate técnico en el que ganaba la palabra equivocada, a `gato` llevándose casi el
doble que ninguna otra. **Ése es todo el trabajo que hacen $W_Q$ y $W_K$**: no cambian la
información, cambian *quién encaja con quién*. Y no están escritas a mano, se aprenden por
descenso de gradiente igual que cualquier otro peso.

La tercera, $W_V$, existe para desacoplar una cosa más: **lo que un token responde a una
pregunta no tiene por qué ser lo mismo que lo que aporta cuando lo eligen**. `gato` puede
anunciarse como "sustantivo singular" (su key) y aportar el concepto de felino (su value).
Con un solo vector, mejorar una cosa estropearía la otra.

## La fórmula

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

Es exactamente lo que acabamos de hacer, con cada símbolo en su sitio:

| símbolo | qué es en el ejemplo |
|---|---|
| $QK^\top$ | los productos escalares de golpe: la casilla $(i,j)$ es cuánto le interesa a $i$ el token $j$ |
| $\sqrt{d_k}$ | el `1.414` por el que dividimos |
| $M$ | la máscara causal, que en el ejemplo no puse |
| softmax | el paso que convirtió `[0.269, 0.276, 0.177]` en pesos que suman 1 |
| $\cdots V$ | la mezcla final |

El salto mental que hay que dar es que $QK^\top$ hace **todas las preguntas a la vez**. En el
ejemplo sólo miré la fila de `dormía`, pero la matriz tiene una fila por token: `gato`
preguntando por su cuenta, `ayer` por la suya. Una multiplicación de matrices, `T` preguntas
resueltas en paralelo. Ésa es la razón de fondo de que el Transformer entrene rápido en GPU y
las redes recurrentes que dominaban antes de 2017 no: aquellas tenían que recorrer la frase
token a token.

## Las formas: qué son B, T, S y d_k

La firma del ejercicio 2 lleva cuatro formas distintas y conviene tenerlas claras antes de
abrirlo, porque el 90% de los errores del módulo son de forma:

```
   q     (B, T, d_k)      las preguntas
   k     (B, S, d_k)      las etiquetas
   v     (B, S, d_v)      los contenidos
   mask  (T, S)           qué está permitido mirar
   ───────────────────────────────────────────────
   out   (B, T, d_v)      la salida: una fila por pregunta
   pesos (B, T, S)        la matriz de "quién mira a quién"
```

- **B** es el batch, el mismo del módulo 04: varias secuencias procesadas a la vez. La
  atención no las mezcla nunca; cada una va por su cuenta.
- **T** es cuántas *preguntas* hay, y **S** cuántas *cosas hay para mirar*. En este curso
  siempre son lo mismo, porque cada token pregunta y a la vez está disponible para ser
  mirado: eso es lo que quiere decir el "self" de *self-attention*. Se escriben con letras
  distintas porque el cálculo no exige que coincidan —en un traductor, las preguntas vienen
  del texto en español y las keys del texto en inglés— y hay un test que comprueba que tu
  función aguanta ese caso.
- **d_k** es el tamaño de los vectores de query y key. Tiene que ser el mismo en los dos, o
  el producto escalar no está definido. **d_v** puede ser distinto, aunque en la práctica
  nunca lo es.

Fíjate en un detalle del que se sigue casi todo: `d_k` desaparece en el producto
$QK^\top$ (se suma sobre él) y `S` desaparece en la multiplicación por $V$. Lo que sale tiene
`T` filas y `d_v` columnas: **la salida tiene la misma forma que la entrada**, un vector por
token. Es lo que permite apilar seis bloques de éstos uno detrás de otro sin que las piezas
dejen de encajar.

---

## Ejercicio 1: la máscara causal (`causal_mask`)

**El problema.** Al entrenar le pasamos al modelo la secuencia entera de golpe y le pedimos
que prediga cada token a partir de los anteriores — las 512 predicciones de una ventana en
una sola pasada, que es lo que viste montar en el módulo 04. Pero $QK^\top$ calcula *todos*
los pares, así que la posición 3 puede mirar a la 4, que es literalmente la respuesta que
tiene que dar.

**La solución.** Una matriz booleana triangular inferior que dice qué se puede mirar. Para
`seq_len = 4`, con el convenio `True = SÍ se puede mirar`:

```
        j=0    j=1    j=2    j=3
 i=0  [ True, False, False, False]     el token 0 sólo se ve a sí mismo
 i=1  [ True,  True, False, False]     el 1 ve al 0 y a sí mismo
 i=2  [ True,  True,  True, False]
 i=3  [ True,  True,  True,  True]     el último lo ve todo
```

La fila `i` es "a quién puede mirar el token `i`". La diagonal va incluida: un token sí puede
mirarse a sí mismo, y de hecho lo hace mucho. Eso es todo el ejercicio, una línea con
`torch.ones(...).tril()`.

**Cómo se usa, que es lo que de verdad importa.** La máscara no borra pesos: pone $-\infty$
en las puntuaciones prohibidas **antes** del softmax. Como $e^{-\infty} = 0$, esas posiciones
reciben peso exactamente cero. En el ejercicio 2 lo escribes así:

```python
scores = scores.masked_fill(~mask, float("-inf"))
```

El `~` invierte el booleano: donde la máscara dice `False` (prohibido), pon `-inf`.

**Y por qué antes y no después.** Ésta es la parte que parece un detalle de estilo y no lo
es. Si dejaras que el softmax normalizase con el futuro dentro y borraras esos pesos después,
las filas ya no sumarían 1: cada posición quedaría escalada por un factor arbitrario y
distinto, más pequeño cuanto más al principio de la frase. Poniendo $-\infty$ antes, el
softmax normaliza sólo sobre lo permitido y **cada fila sigue sumando exactamente 1**. La
demo lo comprueba:

```
   peso total sobre el futuro CON máscara:  0.000000
   peso total sobre el futuro SIN máscara:  3.901440
   suma de cada fila con máscara:           1.000000
```

**Este es el bug más caro del curso.** Si la máscara está mal, la pérdida baja
espectacularmente, todo parece ir de maravilla, y el modelo entrenado no sirve para nada
porque en generación ese futuro no existe: cuando le pides que escriba, no hay tokens
posteriores que mirar y el modelo se queda sin la muleta con la que aprendió. Por eso el
módulo 05 insiste en comparar la pérdida del paso 0 contra $\ln(V)$. **Si sale más baja, mira
la máscara.**

**Un aviso que vas a agradecer.** Usamos `True = permitido` porque es el convenio de
`F.scaled_dot_product_attention`, que usarás en el ejercicio 3. Pero `nn.MultiheadAttention`
de PyTorch usa el contrario: en su `attn_mask` booleana, `True` marca lo que hay que
**prohibir**. Por eso el test que compara contra ella le pasa `~mask`. No es un error del
curso: es una inconsistencia real dentro de la propia librería, y saberla te ahorra una tarde.

---

## Ejercicio 2: la atención de una cabeza (`single_head_attention`)

Aquí escribes la fórmula. Son cuatro líneas y cada una es un trozo de la ecuación:

```
   1.  scores  = q @ k.transpose(-2, -1) / math.sqrt(d_k)     ->  QKᵀ/√d_k
   2.  scores  = scores.masked_fill(~mask, -inf)              ->  + M
   3.  weights = F.softmax(scores, dim=-1)                    ->  softmax(...)
   4.  out     = weights @ v                                  ->  ... V
```

El `transpose(-2, -1)` es lo que convierte `k` de `(B, S, d_k)` en `(B, d_k, S)` para que el
matmul case: `(B, T, d_k) @ (B, d_k, S)` da `(B, T, S)`, la matriz de puntuaciones.

### Compruébalo con números

Si le pasas los tres vectores del ejemplo como `q`, `k` y `v` a la vez (que es lo que hace la
self-attention: los tres salen del mismo sitio) junto con la máscara de 3×3, tu función tiene
que devolver exactamente esto:

```python
X = torch.tensor([[[1.0, 0.2], [0.1, 0.9], [0.3, 0.4]]])
out, w = single_head_attention(X, X, X, causal_mask(3))
```

```
   pesos = [[1.0000, 0.0000, 0.0000]       el token 0 sólo puede mirarse a sí mismo,
            [0.4057, 0.5943, 0.0000]       así que se lleva todo el peso
            [0.3832, 0.2672, 0.3496]]

   salida = [[1.0000, 0.2000]              y por eso su salida es él mismo, intacto
             [0.4651, 0.6160]
             [0.6896, 0.4220]]
```

Dos cosas que mirar ahí. La primera fila de pesos es `[1, 0, 0]` **siempre**, pase lo que
pase con los embeddings: el primer token no tiene a nadie a quien mirar, así que la atención
no le aporta absolutamente nada. La segunda es que la salida del token 0 es su propio vector
sin tocar, `[1.0, 0.2]`, que es una buena forma de verificar de un vistazo que tu media
ponderada está bien montada.

### Las tres trampas

Ninguna de las tres da un error donde la cometes.

**`transpose(-2, -1)` con índices negativos.** Cuentan desde el final, así que funcionan igual
con `(B, T, d)` que con `(B, heads, T, d)`. Si escribes `transpose(1, 2)`, este ejercicio pasa
y el **ejercicio 3 se rompe** con un error de formas que cuesta relacionar con la causa,
porque allí la dimensión 1 son las cabezas.

**`dim=-1` en el softmax.** Estás normalizando sobre *a quién se mira*, de forma que cada
fila sume 1. Con `dim=-2` normalizarías sobre *quién mira*, que no significa nada: sería
repartir la atención que recibe un token entre los que le miran, en vez de repartir la
atención que da un token entre los que mira. Y no da error: las formas son idénticas, el
modelo entrena, y aprende peor. Hay un test que comprueba que cada fila suma 1.

**La máscara va antes del softmax**, por lo de la sección anterior.

## El escalado por √d_k: qué pasa exactamente si lo quitas

Este divisor parece un detalle arbitrario y es de las pocas decisiones del Transformer con
una justificación matemática limpia.

**De dónde sale el número.** El producto escalar de dos vectores de dimensión $d_k$ con
componentes independientes de media 0 y varianza 1 es una suma de $d_k$ términos
independientes, así que tiene **varianza $d_k$** y desviación típica $\sqrt{d_k}$. Con
$d_k = 40$ (nuestro caso) las puntuaciones se mueven en un rango de unos $\pm 6$; con
$d_k = 512$, de unos $\pm 22$. Dividir por $\sqrt{d_k}$ las devuelve a varianza 1
independientemente de la dimensión, que es justo lo que quieres: que la temperatura del
softmax no dependa de una decisión de arquitectura.

**Por qué importa.** Softmax es exponencial. Si una puntuación destaca 20 unidades sobre el
resto, $e^{20}$ frente a $e^{0}$ son 485 millones a uno: el softmax devuelve prácticamente
`[0, 0, ..., 1, ..., 0]` y la atención deja de ser una media ponderada para convertirse en
una selección dura de un único token.

La demo lo mide con la **entropía** de la distribución de atención, que es lo mismo que la
perplejidad del módulo 05 pero sin exponenciar: alta significa "reparte entre muchos", cerca
de cero significa "se fija en uno solo". Con 16 posiciones, el máximo posible es
$\ln(16) = 2{,}773$:

| d_k | entropía CON escalado | entropía SIN escalado | peso máximo sin escalar |
|---|---|---|---|
| 8 | 2,502 | 1,629 | 0,8004 |
| 32 | 2,421 | 0,692 | 0,9984 |
| 128 | 2,318 | 0,085 | 1,0000 |
| 512 | 2,346 | 0,216 | 1,0000 |
| 2048 | 2,279 | 0,007 | 1,0000 |

Con escalado la entropía se mantiene alta pase lo que pase con `d_k`. Sin él, a partir de
`d_k = 128` el peso máximo es 1,0000 redondeando: el token se fija en uno solo e ignora todo
lo demás.

**Y el problema de verdad no es ése, es el gradiente.** La derivada del softmax es $p(1-p)$;
con $p$ pegado a 0 o a 1, la derivada es prácticamente cero. Los pesos que producen esas
puntuaciones dejan de recibir señal y la capa deja de aprender. No es que el modelo atienda
mal: es que **no puede corregirse**, porque el mecanismo que le diría cómo hacerlo se ha
apagado. Sin el $\sqrt{d_k}$, un Transformer grande sencillamente no entrena.

---

## Ejercicio 3: ocho cabezas en paralelo (`MultiHeadAttention`)

**El problema.** Una sola atención tiene que resolver todas las relaciones de la frase con un
único patrón. Pero en *"el gato que vi ayer dormía"* hay varias relaciones a la vez: la
concordancia sujeto-verbo, el `que` que se refiere a `gato`, el `ayer` que sitúa la acción.
Una única distribución de pesos por token no da para todo eso: si le da peso a `gato` no se
lo da a `ayer`.

**La solución.** Hacer varias atenciones en paralelo, cada una con sus propias
$W_Q, W_K, W_V$, y concatenar los resultados. Con $d_{\text{model}} = 320$ y 8 cabezas, cada
una trabaja en $40$ dimensiones ($320/8$).

Y aquí está lo bonito: **no cuesta más**. En vez de una atención de 320 dimensiones haces
ocho de 40, y el total de parámetros es idéntico. Lo que ganas es que ocho distribuciones de
atención distintas pueden coexistir.

### Las cabezas se especializan solas

En el modelo que entrena la demo —una capa, cuatro cabezas, 400 pasos sobre Shakespeare— ya
se ve, midiendo a qué distancia media mira cada cabeza:

```
   cabeza 0: distancia media 2.10 posiciones
   cabeza 1: distancia media 3.39 posiciones
   cabeza 2: distancia media 2.93 posiciones
   cabeza 3: distancia media 4.05 posiciones
```

Son cuatro rangos de contexto distintos y **nadie se lo ha dicho**: sale de inicializar cada
cabeza al azar y dejar que el gradiente haga el resto. En modelos grandes esto llega mucho
más lejos: se han identificado cabezas que miran siempre al token anterior, cabezas que
emparejan comillas de apertura y cierre, y las llamadas *induction heads*, que detectan el
patrón "…A B … A" y predicen B. Ninguna está programada.

### El baile de las formas, que es lo que cuesta

Ésta es la parte del módulo donde de verdad se atasca la gente, y es pura contabilidad de
dimensiones. El recorrido completo, con `B=2`, `T=5` y la configuración del modelo final:

```
   x                 (2, 5, 320)      entrada
     │  q_proj / k_proj / v_proj      tres Linear(320, 320)
     ▼
   q, k, v           (2, 5, 320)      todavía sin partir
     │  _split_heads
     ▼
   q, k, v      (2, 8, 5, 40)         8 cabezas, cada una con 40 dimensiones
     │  el cálculo del ejercicio 2, idéntico
     ▼
   scores       (2, 8, 5, 5)          una matriz de atención POR CABEZA
   out          (2, 8, 5, 40)
     │  _merge_heads
     ▼
   out               (2, 5, 320)      las cabezas otra vez pegadas
     │  out_proj                      un Linear(320, 320) más
     ▼
   salida            (2, 5, 320)      misma forma que la entrada
```

Que la entrada y la salida tengan la misma forma no es casualidad: es lo que permite apilar
bloques y lo que hace posibles las conexiones residuales del módulo 07.

**`_split_heads` en concreto.** Se hace en dos pasos, y el orden importa:

```python
x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
```

El `view` parte el vector de cada token en trozos; el `transpose` mueve las cabezas delante
de las posiciones. Con un ejemplo diminuto (`d_model=4`, 2 cabezas de 2, dos tokens):

```
   entrada:  token 0 = [1, 2, 3, 4]        token 1 = [5, 6, 7, 8]

   BIEN, view(B,T,2,2) y luego transpose(1,2):
      cabeza 0 = [[1, 2],      <- la primera mitad de CADA token
                  [5, 6]]
      cabeza 1 = [[3, 4],      <- la segunda mitad de CADA token
                  [7, 8]]

   MAL, view(B,2,T,2) directamente:
      cabeza 0 = [[1, 2],      <- el token 0 entero
                  [3, 4]]
      cabeza 1 = [[5, 6],      <- el token 1 entero
                  [7, 8]]
```

La versión mala tiene la forma correcta y los datos mezclados: ha repartido *posiciones*
entre las cabezas en vez de *dimensiones*. No da ningún error. Hay un test que la caza
comprobando que las cabezas no producen patrones idénticos.

**`_merge_heads` deshace exactamente eso**, y necesita un `.contiguous()` en medio:

```python
x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
```

`transpose` no mueve datos, sólo cambia cómo se recorren en memoria (los *strides*), y `view`
exige memoria contigua. Sin el `.contiguous()`, PyTorch lanza un error que habla de strides y
no dice claramente qué hacer.

### La cuarta proyección: `out_proj`

La fórmula del paper no la menciona y sin ella el multi-head no sirve de nada, así que vale
la pena entender qué hace.

Después de `_merge_heads` tienes un vector de 320 números que son ocho trozos de 40 pegados
uno detrás de otro. Cada trozo es lo que sacó una cabeza **por su cuenta, sin saber nada de
las otras**. Si eso fuera directamente la salida de la capa, las ocho cabezas serían ocho
canales estancos: la 3 nunca podría combinar lo suyo con lo que encontró la 7.

`out_proj` es un `nn.Linear(320, 320)` cuyo trabajo es precisamente **mezclar los resultados
de las cabezas** y decidir cuánto pesa cada una. Es la que convierte ocho respuestas
independientes en una sola. Es también, dicho sea de paso, la que hace que el recuento de
parámetros de la capa sean cuatro matrices y no tres:

```
   q_proj    320 × 320 = 102.400
   k_proj    320 × 320 = 102.400
   v_proj    320 × 320 = 102.400
   out_proj  320 × 320 = 102.400
   ─────────────────────────────
                        409.600     por capa, sin sesgos
```

Sin sesgos porque `ModelConfig` usa `bias=False`: los LLM modernos los quitan de las
proyecciones porque aportan poco y complican el weight decay. Se explica en el módulo 09.

### Las tres cosas del enunciado que aún no has visto

El ejercicio 3 tiene tres argumentos que vienen de módulos que todavía no has hecho. No hace
falta entenderlos a fondo para escribirlo, pero sí saber qué son para no quedarte pensando
que te has perdido algo:

- **`dropout`, y en dos sitios.** Dropout es apagar al azar una fracción de los números
  durante el entrenamiento, para que el modelo no dependa demasiado de ninguno en concreto;
  se ve en el módulo 11. Aquí va en dos puntos distintos: `attn_dropout` sobre los pesos de
  atención (apaga algunas conexiones) y `resid_dropout` sobre la salida final. En el modelo
  del curso vale 0.
- **`cos` y `sin`: RoPE.** La forma de codificar la posición del módulo 09. Van *después* de
  partir en cabezas, porque la rotación depende de `head_dim` y no de `d_model`, y **sólo a
  q y k, nunca a v**: lo que debe depender de la posición son las puntuaciones, no el
  contenido que se transporta. Los tests de este módulo pasan `cos=None, sin=None`.
- **`use_sdpa`.** `F.scaled_dot_product_attention` es la implementación fusionada de PyTorch,
  que hace los cuatro pasos en un solo kernel sin materializar la matriz `(T, T)` completa.
  Es la que usa el entrenamiento de verdad. Tú escribes las dos ramas: la explícita, que es
  la que enseña y la que devuelve los pesos para el heatmap, y la llamada a SDPA. Un test
  comprueba que dan el mismo resultado. Ojo con el `dropout_p`: hay que pasarle
  `self.dropout if self.training else 0.0`, porque SDPA no consulta el modo por su cuenta y
  aplicaría dropout también en evaluación, con lo que tus muestras saldrían ruidosas y no
  reproducibles.

**Y por qué una proyección grande y no ocho pequeñas.** `nn.Linear(320, 320)` seguido de un
`view` es matemáticamente idéntico a ocho `nn.Linear(320, 40)` cuyos resultados se concatenan.
Pero es *un* matmul grande en vez de ocho pequeños, y como mediste en el módulo 01, las
matrices grandes aprovechan muchísimo mejor la GPU. Es el mismo razonamiento que el de
`nn.Embedding` frente a `nn.Linear` del módulo 05: misma matemática, coste distinto.

---

## Lo que la atención no hace

Tan importante como saber qué resuelve es saber qué no, porque los tres módulos siguientes
son en buena medida los parches de esta lista.

**No sabe nada del orden.** Esto sorprende y es fácil de comprobar. La atención es una suma
ponderada, y una suma no distingue el orden de sus términos: para el token que pregunta, los
anteriores son una **bolsa**, no una secuencia. Coge el ejemplo de tres palabras, intercambia
`gato` y `ayer`, y mira la salida de la última posición:

```
   orden original:  [0.4708, 0.5040]
   orden cambiado:  [0.4708, 0.5040]     idénticas
```

La máscara causal impone *quién puede mirar a quién*, pero no le dice al modelo a qué
distancia está cada cual. Sin arreglar eso, *"el gato mordió al perro"* y *"el perro mordió
al gato"* serían indistinguibles para la capa. Ése es el módulo 09.

**No hace ningún procesamiento por posición.** Todo lo que hace la atención es mover
información entre tokens; combinar y transformar esa información dentro de cada token es
trabajo del MLP del módulo 08. De hecho, sin él, apilar capas de atención no serviría de gran
cosa: son operaciones lineales salvo por el softmax. Las dos piezas se alternan por eso, y no
por costumbre.

**Cuesta el cuadrado de la longitud del contexto.** La matriz de puntuaciones es `(T, T)`:
si doblas el contexto, cuadruplicas ese coste. Con nuestro contexto de 512 y 8 cabezas, en
fp32:

```
   T =  512:   512×512 × 8 cabezas × 4 bytes  =    8,4 MB por secuencia
   T = 1024:  1024×1024 × 8 × 4               =   33,5 MB por secuencia
```

Multiplica eso por el tamaño del batch y verás por qué la longitud del contexto es la
decisión que más memoria cuesta de un Transformer. Es también el motivo de que SDPA exista:
evita materializar esa matriz entera.

---

## Dónde está el debate

Se sabe *qué* calcula la atención. Por qué funciona tan bien es harina de otro costal.

La explicación intuitiva —"cada token recupera información relevante"— es la que te acabo de
contar, es una historia razonable y no está demostrada. Hay resultados que la complican:
modelos con patrones de atención **fijos y aleatorios** funcionan sorprendentemente bien en
algunas tareas, lo que sugiere que parte del mérito está en la arquitectura general
(residuales, normalización, profundidad) y no sólo en el mecanismo de atención. Conviene
tenerlo presente cuando leas explicaciones muy seguras de sí mismas, incluida ésta.

La línea de trabajo más seria en esta dirección es la de interpretabilidad mecanicista, que
trata de leer los circuitos que se forman dentro. Ha conseguido explicar componentes
concretos —las *induction heads* son el caso de éxito— pero está muy lejos de dar cuenta de
un modelo entero.

Y hay una limitación estructural que sigue sin resolverse: el coste crece con el **cuadrado**
de la longitud del contexto. Se han propuesto decenas de alternativas subcuadráticas
(Linformer, Performer, Mamba y familia). Ninguna ha desplazado a la atención estándar en
modelos de propósito general, y no está claro si es porque la atención completa es necesaria
o porque tiene veinte años de ventaja en optimización de kernels.

---

**Para ampliar:** Vaswani et al. 2017,
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) · Elhage et al. 2021,
[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
(las *induction heads*). Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
