# 08 — El MLP de cada bloque: FFN, GELU y SwiGLU

## Por qué importa este módulo

**Porque la mayor parte de tu modelo está aquí, y casi nadie lo cuenta.**

Cuando alguien dice que un modelo tiene N parámetros, la intuición de todo el mundo es que
están en la atención. No es así. En nuestro modelo:

```
   el MLP (este módulo)       5.160.960     57,8%
   atención (módulo 06)       2.457.600     27,5%
   embeddings                 1.310.720     14,7%
   normalización (módulo 07)      4.160      0,05%
   ───────────────────────────────────────────────
                              8.933.440
```

Dentro de cada bloque, dejando los embeddings aparte, la proporción es todavía más clara:
**el 68% de un bloque del Transformer es esta pieza y el 32% es la atención.** El módulo que
todo el mundo explica es el pequeño.

Y hay una razón más profunda que el tamaño. La atención es una media ponderada, o sea una
**operación lineal**, y apilar operaciones lineales no sirve de nada: cien capas equivalen a
una. Lo que impide que el Transformer entero se derrumbe a una sola multiplicación de matrices
es precisamente este módulo. La demo lo mide, y el número es contundente: cinco capas lineales
sin activación dan el mismo resultado que una sola matriz con una diferencia de $5{,}6 \times
10^{-8}$, que es ruido de coma flotante.

### Qué sabrás al terminar

- **Qué es exactamente un FFN**, por qué tiene tres nombres distintos y por qué este módulo se
  llama "MLP y activaciones"
- Por qué sin una no-linealidad la profundidad de una red es una ilusión, con la medición
- Qué le pasa a una neurona con ReLU cuando se va a la zona negativa (se muere, literalmente)
  y por qué GELU la deja volver
- Qué hacen de verdad las $d_{ff}$ neuronas intermedias, que es la parte que casi nunca se
  explica
- Qué es una **puerta** y en qué se diferencia de una activación normal, con el ejemplo
  numérico
- **De dónde sale el 896** del config del modelo final, y por qué el ajuste que lo produce no
  cuadra tan bien como suele contarse
- Un caso donde el propio autor del paper escribe que no sabe por qué funciona

### Qué vas a escribir

Tres ejercicios. Esta teoría está ordenada para que los leas en este orden, y **cada uno tiene
su propia sección con su ejemplo numérico**:

| Ejercicio | Qué hace | Dónde se explica |
|---|---|---|
| 1. `gelu` | La no-linealidad clásica | [§ GELU](#ejercicio-1-un-corte-suave-gelu) |
| 2. `swiglu_hidden_dim` | Aritmética: de aquí sale el 896 del config | [§ El 896](#ejercicio-2-de-dónde-sale-el-896-swiglu_hidden_dim) |
| 3. `SwiGLU` | El FFN con puerta que usa el modelo | [§ SwiGLU](#ejercicio-3-añadir-una-puerta-swiglu) |

El ejercicio 1 es una línea transcribiendo una fórmula. El 2 es **el más corto del curso**,
tres líneas de aritmética entera, y produce un número que ya has visto en el YAML del config.
El 3 son cinco líneas. Como en el módulo 07, el trabajo no está en teclear: está en entender
qué hace cada pieza y por qué está ahí.

### Cuánto cuesta

1,5 horas.

---

## Qué es un FFN, y por qué el módulo se llama "MLP y activaciones"

Antes de nada, el vocabulario, porque este módulo arrastra un lío de nombres que no es culpa
tuya: **tres términos distintos para la misma caja.**

### Feed-forward: qué significa

**FFN** son las siglas de *feed-forward network*: red **hacia delante**. Es la clase de red más
antigua y más simple que existe, y el nombre describe literalmente cómo circula la información:
entra por un lado, atraviesa las capas en orden, y sale por el otro. Sin bucles, sin volver
atrás y sin mirar a los lados.

Ese "sin mirar a los lados" es la parte que importa aquí, y se entiende mejor por contraste con
lo que ya has visto:

```
   red RECURRENTE       la salida vuelve a entrar; procesa la frase token a token
                        (lo que dominaba antes de 2017)

   ATENCIÓN (mód. 06)   cada token mira a los otros tokens
                        la información se mueve DE LADO

   FEED-FORWARD (hoy)   cada token se procesa solo, sin enterarse de que existen
                        los demás. La información sólo va HACIA DELANTE
```

Por eso el nombre completo en el paper de 2017 es *position-wise feed-forward network*:
"position-wise" quiere decir que se aplica **por posición**, la misma función a cada token por
separado. Si le pasas un tensor `(B, T, 320)`, el FFN hace `B × T` cálculos independientes.

### Es el MLP del módulo 02, literalmente

Y ahora la buena noticia: **ya has construido uno.** En el módulo 02 montaste un `MLP` a mano
con tu propio motor de derivadas, y era exactamente esto:

- una **neurona** = suma ponderada de sus entradas + un sesgo, pasada por una función no lineal
- una **capa** = varias neuronas mirando a la misma entrada
- un **MLP** (*multi-layer perceptron*) = capas encadenadas, la salida de una es la entrada de
  la siguiente

El FFN clásico de un Transformer **es un MLP de dos capas**. Ni más ni menos. Lo único que
cambia respecto al del módulo 02 son los tamaños y la función no lineal:

```
   módulo 02:   MLP(3, [8, 8, 1])        tanh    113 parámetros en total
   módulo 08:   MLP(320, [1280, 320])    GELU    819.200 parámetros por bloque
```

(Ese `MLP(320, [1280, 320])` es el FFN clásico, el del paper de 2017. El de nuestro modelo es
una variante con una matriz más, SwiGLU, y sale en el ejercicio 3: mismo esqueleto, 860.160
parámetros. Pero la forma de pensarlo es la de arriba.)

Aquel `tanh` que pusiste porque el módulo 02 lo pedía es el antepasado directo del GELU que vas
a escribir hoy, y el módulo 02 ya te lo anunció: *"en el módulo 08 verás por qué los
transformers usan GELU en vez de `tanh`"*. Ésa es la segunda mitad del nombre del módulo, las
**activaciones**: la función no lineal que va dentro, entre las dos capas. El ejercicio 1 es
una de ellas y el ejercicio 3 usa otra.

### Los tres nombres

| nombre | de dónde viene | dónde lo vas a ver |
|---|---|---|
| **FFN** | *feed-forward network*, cómo circula la información | los papers de Transformers |
| **MLP** | *multi-layer perceptron*, cómo está construido | el código (`llmfs`, nanoGPT, Llama), y el nombre de este módulo |
| **feed-forward** | lo mismo que FFN, sin abreviar | los diagramas |

Los tres son la misma caja del dibujo. En este curso verás sobre todo "FFN" en la teoría y
"MLP" en los nombres de ficheros y directorios, y no hay ninguna diferencia entre los dos.

**Cuidado con una colisión de nombres**, que es la razón de que esto confunda a todo el mundo:
en el módulo 02, "MLP" era **la red entera**. En un Transformer, "MLP" es **un sub-bloque de
cada capa**, la caja que hay al lado de la atención. El mismo término para dos cosas de escala
muy distinta, según el contexto. Está avisado también en el [GLOSARIO.md](../../GLOSARIO.md).

Con eso claro, el resto del módulo va de tres cosas: por qué esta caja tiene que existir, qué
función no lineal se pone dentro, y qué variante concreta (SwiGLU) usa nuestro modelo.

---

## Qué parte del LLM es esta

Con este módulo se cierra el bloque del Transformer. Mira el dibujo del módulo 07 con la
última caja rellena:

```
    x ──┬──> norma ──> atención ──┐
        │   (mód 07)   (mód 06)   ├──> +  ──┬──> norma ──> FFN ──┐
        └─────────────────────────┘         │   (mód 07)  (HOY)  ├──> +
                                            └────────────────────┘

    lo que queda:  módulo 09  ->  decirle al modelo en qué posición está cada token
                   módulo 10  ->  apilar seis de estos bloques y montar el GPT
```

Y conviene tener presente el reparto de trabajo entre las dos cajas grandes, porque es la
estructura de fondo de todo Transformer:

```
   ATENCIÓN                          FFN
   mueve información ENTRE tokens    procesa la información DENTRO de cada token
   mezcla posiciones                 no mira a los demás tokens en absoluto
   lineal (una media ponderada)      no lineal
   32% de los parámetros del bloque  68%
```

Se alternan por eso: mover, procesar, mover, procesar. Seis veces. Que el FFN no mire a los
demás tokens tiene una consecuencia práctica que agradecerás al escribir el ejercicio 3: **no
hay ninguna máscara ni nada parecido**. Si tu tensor es `(B, T, d_model)`, el FFN aplica
exactamente la misma función a cada uno de los `B × T` vectores por separado. Podrías aplanarlo
a `(B*T, d_model)`, pasarlo, y deshacerlo, y el resultado sería idéntico. Hay un test que lo
comprueba.

---

## El problema: la atención sola no basta

Fíjate en lo que hace la atención: mezcla vectores con pesos. Una media ponderada. Y una media
ponderada es una **operación lineal**.

Eso es un problema serio, y se ve con dos líneas de álgebra. Imagina que apilas dos capas
lineales sin nada en medio:

```
   capa 1:  y = W₁ · x
   capa 2:  z = W₂ · y = W₂ · (W₁ · x) = (W₂ · W₁) · x
```

$W_2 W_1$ es **una sola matriz**. Cien capas lineales apiladas equivalen exactamente a una
capa lineal, con muchos más parámetros y ni una pizca más de capacidad. Toda la profundidad se
derrumba.

La demo lo comprueba en vez de afirmarlo: apila 5 capas lineales, multiplica sus cinco matrices
para obtener una sola, y compara.

```
   5 capas apiladas  vs  1 sola matriz           diferencia máxima: 5,59e-08
   las mismas 5 capas CON GELU  vs  1 matriz     diferencia:        0,298
```

El primer número no es "parecido", es **cero**: $5{,}6\times10^{-8}$ es lo que se acumula al
multiplicar floats de 32 bits. El segundo dice que en cuanto metes una no-linealidad entre las
capas, la red deja de ser reducible.

Para que apilar sirva de algo hace falta algo que no sea lineal entre capa y capa. Ése es el
trabajo del FFN, y es la razón de que exista.

## La forma clásica: expandir, doblar, contraer

$$\text{FFN}(x) = W_2 \cdot \text{activación}(W_1 x)$$

Con $W_1$ de $d \to 4d$ y $W_2$ de $4d \to d$. Se expande a cuatro veces el tamaño, se aplica
la no-linealidad, y se vuelve a comprimir. Con nuestras dimensiones y un token cualquiera:

```
   x              (320,)      el vector del token, saliendo de la norma
     │  W₁
     ▼
   h             (1280,)      expandido a 4×
     │  activación            <- aquí está la no-linealidad, y sólo aquí
     ▼
   h'            (1280,)
     │  W₂
     ▼
   salida         (320,)      misma forma que la entrada, lista para el residual
```

**¿Por qué 4x?** Honestamente: porque lo puso el paper de 2017 y funcionó. No hay una
derivación. Se han probado otros factores y 4 sigue siendo un punto razonable, pero es
convención, no teorema.

### Qué hacen esas 1280 neuronas del medio

Ésta es la parte que rara vez se explica y que hace que el FFN deje de parecer "una capa más".

Mira la operación por filas y columnas en vez de como dos matrices:

- **Cada fila de $W_1$ es un detector.** El número $i$ del vector expandido es el producto
  escalar de la fila $i$ de $W_1$ con el token. Es exactamente la misma operación que en el
  módulo 06: mide **cuánto se parece** el token a un patrón concreto guardado en esa fila. Si
  se parece, sale un número grande; si no, sale pequeño o negativo, y la activación lo aplasta.
- **Cada columna de $W_2$ es lo que se escribe de vuelta.** Si el detector $i$ se activa, se
  suma a la salida la columna $i$ de $W_2$, escalada por cuánto se activó.

Junta las dos cosas y el FFN se lee como: *"si el token se parece al patrón $i$, súmale a la
corriente residual el vector $i$"*. Con 1280 pares patrón-respuesta por capa, y seis capas.

A esto se le llama la interpretación del FFN como **memoria clave-valor**: las filas de $W_1$
son las claves, las columnas de $W_2$ los valores. Es una hipótesis con evidencia detrás —hay
trabajos que localizan afirmaciones concretas en neuronas concretas y las editan cambiando esos
pesos— pero **no es un resultado establecido**, y conviene leerla como una forma útil de pensar
en la operación y no como lo que la red realmente hace. Lo que sí es literal es la mecánica de
filas y columnas del párrafo anterior: eso es aritmética, no interpretación.

---

## Ejercicio 1: un corte suave (`gelu`)

### El problema: ReLU y las neuronas muertas

La no-linealidad más simple es ReLU: $\max(0, x)$. Deja pasar lo positivo y pone a cero lo
negativo. Funciona, es baratísima, y tiene un defecto que se ve mirando la derivada:

| x | ReLU | dReLU/dx | GELU | dGELU/dx |
|---|---|---|---|---|
| −3,0 | 0,0000 | **0,0000** | −0,0036 | −0,0116 |
| −1,0 | 0,0000 | **0,0000** | −0,1588 | −0,0830 |
| 0,0 | 0,0000 | — | 0,0000 | 0,5000 |
| +1,0 | 1,0000 | 1,0000 | 0,8412 | 1,0830 |
| +3,0 | 3,0000 | 1,0000 | 2,9964 | 1,0116 |

Con ReLU, la derivada en toda la zona negativa es **cero exacto**. Y ahora recuerda lo que
significa una derivada cero, que lo viste en el módulo 07: no llega gradiente. Si durante el
entrenamiento una neurona acaba dando siempre valores negativos, deja de recibir señal **para
siempre**: nunca se actualiza, así que nunca sale de ahí. Está muerta y no hay forma de
resucitarla. Es un fenómeno con nombre propio, *dying ReLU*, y en redes grandes puede matar una
fracción nada despreciable de las neuronas.

### La solución: multiplicar por una probabilidad

$$\text{GELU}(x) = x \cdot \Phi(x)$$

donde $\Phi(x)$ es la probabilidad de que una normal estándar salga menor que $x$.

La intuición: en vez de decidir con un corte duro si dejar pasar $x$, lo multiplica por la
probabilidad de que $x$ "destaque". Con números:

```
   x = -3   ->  Φ(-3) = 0.001   ->  GELU = -0.0036    casi anulado
   x = -1   ->  Φ(-1) = 0.159   ->  GELU = -0.1588    parcialmente
   x =  0   ->  Φ(0)  = 0.5     ->  GELU =  0.0000
   x =  1   ->  Φ(1)  = 0.841   ->  GELU =  0.8412    casi entero
   x =  3   ->  Φ(3)  = 0.999   ->  GELU =  2.9964    entero
```

Ésos son exactamente los cinco valores que tiene que devolver tu función, y el test los
comprueba uno a uno.

La transición es suave, así que la derivada nunca es exactamente cero: mira otra vez la columna
`dGELU/dx` de la tabla, −0,0116 en $x=-3$. Es un gradiente diminuto, pero no nulo, y eso basta
para que una neurona hundida en la zona negativa pueda volver.

### Lo que de verdad escribes

En la práctica se usa una aproximación con `tanh`, porque `erf` (la función que hace falta para
$\Phi$ exacta) era lenta en las GPU de 2016:

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}\,(x + 0.044715x^3)\right]\right)$$

Y eso es el ejercicio: una línea transcribiendo esa fórmula, sin bucles ni condicionales. Los
dos errores posibles son teclear mal una constante ($\sqrt{2/\pi} \approx 0{,}7978$) o reagrupar
la expresión de forma que cambie el orden de operaciones.

**Ojo con contra qué comparas.** Tu resultado tiene que coincidir con
`F.gelu(x, approximate="tanh")`, **no** con `F.gelu(x)` a secas: son funciones distintas —la
segunda usa `erf`— y el test compara contra la primera. Hoy la diferencia de velocidad entre
las dos es irrelevante, pero GPT-2 se entrenó con la aproximación y por compatibilidad se sigue
usando en todas partes.

### Un aviso sobre la tabla de la demo

La demo dibuja las tres activaciones sobre una rejilla de 400 puntos entre −4 y 4 y luego
imprime la fila más cercana a cada valor redondo. Por eso su tabla dice cosas como
`GELU(+0.0) = -0.0050`, que descoloca si vienes de leer aquí que GELU(0) = 0 exacto. No es una
contradicción: esa fila es en realidad $x = -0{,}01$, el punto de la rejilla más próximo a cero.
Los valores exactos son los de arriba.

### GELU y Swish, casi la misma curva

En la demo verás también **Swish** (que PyTorch llama `SiLU`), $z \cdot \sigma(z)$, y es la que
usarás en el ejercicio 3. Merece la pena fijarse en que las dos curvas son casi idénticas
teniendo orígenes completamente distintos: GELU sale de un argumento probabilístico y Swish de
una búsqueda automática de funciones de activación. Acabaron prácticamente en el mismo sitio, lo
cual sugiere que lo que importa es la *forma* —suave, casi identidad en positivo, atenuación no
nula en negativo— y no la fórmula concreta.

---

## Ejercicio 2: de dónde sale el 896 (`swiglu_hidden_dim`)

Éste es el ejercicio más corto del curso: tres líneas de aritmética entera, ningún tensor. Y
produce un número que ya has visto, el `d_ff: 896` del config. Es de esos ejercicios cuyo valor
está entero en entender **por qué** ese número es ése.

Hay dos ajustes encadenados, y son independientes entre sí.

### Ajuste 1: el factor 2/3

SwiGLU (el ejercicio 3) tiene **tres** matrices donde el FFN clásico tiene dos. Con el mismo
$d_{ff}$ eso sería un 50% más de parámetros, y entonces cualquier comparación entre las dos
arquitecturas sería tramposa: no sabrías si SwiGLU gana por ser mejor o por ser más grande.

La solución es reducir $d_{ff}$ a dos tercios para que el presupuesto cuadre:

```
   FFN clásico:  2 matrices × d × 4d           = 8d²
   SwiGLU:       3 matrices × d × (2/3 · 4d)   = 3 · d · (8/3)d = 8d²   ✓
   SwiGLU sin ajustar: 3 × d × 4d              = 12d²                   +50%
```

Con nuestro $d_{\text{model}} = 320$: $(2/3) \times 4 \times 320 = 853{,}33$, que truncado a
entero es 853.

### Ajuste 2: redondear hacia arriba a múltiplo de 64

$853 \to 896$. Y esto no es cosmética: las dimensiones alineadas a potencias de dos dejan que
los tensor cores de la GPU usen sus rutas rápidas. Una matriz de 853 columnas es notablemente
más lenta que una de 896 **teniendo menos parámetros**, que es de las cosas más contraintuitivas
del módulo 01 y aquí la ves aplicada.

El redondeo se escribe sin `math.ceil` ni floats:

```python
multiple_of * ((hidden + multiple_of - 1) // multiple_of)
```

Sumar `multiple_of - 1` antes de la división entera fuerza el redondeo hacia arriba, y si el
valor ya era múltiplo exacto no lo toca. Es el idioma estándar para esto y conviene reconocerlo.

Los dos casos del curso, para que compruebes:

```
   d_model = 320:  int(2*1280/3) = 853  ->  64 * ((853+63)//64) = 64 * 14 = 896
   d_model = 128:  int(2*512/3)  = 341  ->  64 * ((341+63)//64) = 64 *  6 = 384
```

El 896 es el `d_ff` del modelo final; el 384, el del config de juguete.

### El 2/3 no cuadra tan bien como suele contarse

Aquí va un detalle que casi nunca se menciona y que sale solo si haces la cuenta para varios
tamaños. El redondeo del ajuste 2 rompe el equilibrio del ajuste 1, y cuánto lo rompe depende
del tamaño del modelo:

| d_model | d_ff | FFN clásico | SwiGLU | de más |
|---|---|---|---|---|
| 128 | 384 | 131.072 | 147.456 | **+12,5%** |
| 320 | 896 | 819.200 | 860.160 | **+5,0%** |
| 768 | 2048 | 4.718.592 | 4.718.592 | 0,0% |
| 4096 | 10944 | 134.217.728 | 134.479.872 | +0,2% |

O sea: **el 2/3 iguala los presupuestos de forma asintótica, no exacta.** En modelos grandes el
redondeo a 64 es un ajuste marginal y la igualdad se cumple casi perfectamente; en un modelo
pequeño como el nuestro, 64 es una fracción apreciable de $d_{ff}$ y acabamos gastando un 5% de
más. No es un problema —un 5% no cambia ninguna conclusión— pero sí conviene saberlo antes de
leer por ahí que SwiGLU "cuesta exactamente lo mismo".

---

## Ejercicio 3: añadir una puerta (`SwiGLU`)

Aquí viene el cambio que usa nuestro modelo, y todos los modernos.

La idea de las variantes **GLU** (*Gated Linear Unit*) es tener **dos** proyecciones en vez de
una, saliendo las dos de la misma entrada. Una de ellas actúa como **puerta**: multiplica a la
otra elemento a elemento y decide cuánta señal pasa por cada dimensión.

$$\text{SwiGLU}(x) = \big(\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}}\big) W_{\text{down}}$$

con $\text{Swish}(z) = z \cdot \sigma(z)$, que es la curva casi idéntica a GELU de la sección
anterior. En PyTorch se llama `F.silu`, y puedes escribirla a mano como `x * torch.sigmoid(x)`
sin que cambie nada salvo que pierdes el kernel fusionado.

### El recorrido, con las formas

```
   x                    (B, T, 320)
     ├── gate_proj ──> (B, T, 896) ── Swish ──┐
     │                                        ⊙   multiplicación ELEMENTO A ELEMENTO
     └── up_proj   ──> (B, T, 896) ───────────┘
                                              │
                                          (B, T, 896)
                                              │  down_proj
                                              ▼
                                        (B, T, 320)      misma forma que la entrada
```

Las dos ramas salen con la misma forma, así que el `⊙` es un `*` de Python, **no** un `@`. Si
pusieras `@` las formas ni siquiera cuadrarían, que es de los pocos errores de este módulo que
sí dan un error inmediato.

### Qué hace la puerta, con números

Ésta es la diferencia conceptual del módulo. Imagina que en una posición concreta las dos ramas
producen estos tres números (en el modelo real serían 896):

```
   up    = [ 2.0,  -1.0,   4.0]      el contenido
   gate  = [ 3.0,  -3.0,   0.0]      la puerta, antes de Swish

   Swish(gate) = [2.8577, -0.1423, 0.0000]

   producto    = [5.7154,  0.1423, 0.0000]
                    │         │       │
                    │         │       └── dimensión CERRADA: no pasa nada
                    │         └────────── casi cerrada, y con el signo cambiado
                    └──────────────────── abierta de par en par, amplificada ×2,9
```

Compáralo con una activación normal, que aplicaría la misma función a `up` y ya está. Aquí la
red decide, **para cada dimensión y para cada token**, cuánto deja pasar — y esa decisión se
calcula a partir de la propia entrada, con pesos aprendidos. La tercera dimensión se apaga
entera no porque su contenido fuera pequeño (era 4,0, el mayor de los tres) sino porque la
puerta decidió que en este contexto no venía a cuento.

Es el mismo tipo de idea que la atención —dejar que el contenido decida qué pasa— aplicada
dentro de un token en vez de entre tokens.

### Los detalles que fallan si no los cuidas

**La activación va en `gate_proj`, no en `up_proj`.** Numéricamente el módulo funcionaría igual
de bien con la asignación invertida: es simétrico salvo por qué pesos aprenden qué. Pero **no
coincidiría con la referencia** al copiar pesos y el test fallaría con una diferencia difícil de
interpretar. Hay un test dedicado a señalarlo.

**Los nombres importan**: `gate_proj`, `up_proj`, `down_proj`. El test copia pesos por nombre,
igual que en los módulos 05 y 06.

**`bias=False` por defecto**, que es la config del modelo final. Es lo que hace que el conteo de
parámetros dé exactamente $3 \cdot d_{\text{model}} \cdot d_{ff} = 3 \times 320 \times 896 =
860.160$, sin sumandos sueltos. Hay un test que comprueba ese número exacto.

**El `dropout` va al final**, sobre la salida de `down_proj`, y en el modelo del curso vale 0.
Es la misma pieza del módulo 06 y se explica en el 11.

---

## Dónde están los parámetros

Con el config final, por capa:

| componente | parámetros | % del bloque |
|---|---|---|
| atención ($4d^2$) | 409.600 | 32% |
| SwiGLU ($3 \cdot d \cdot d_{ff}$) | 860.160 | **68%** |

Y la proporción se mantiene sea cual sea el tamaño, porque las dos crecen con $d^2$:

| d_model | d_ff | atención | FFN | % FFN |
|---|---|---|---|---|
| 128 | 384 | 65.536 | 147.456 | 69% |
| 320 | 896 | 409.600 | 860.160 | 68% |
| 768 | 2048 | 2.359.296 | 4.718.592 | 67% |
| 4096 | 10944 | 67.108.864 | 134.479.872 | 67% |

Dos tercios de cada bloque son FFN. Sobre el modelo entero la cifra baja al 57,8% porque los
embeddings se llevan su parte, pero la idea aguanta: **cuando leas que un modelo tiene N
parámetros, la mayoría están aquí, no en la atención.**

## Dónde está el debate

Este módulo es probablemente donde el "no sabemos por qué" es más explícito, y viene del propio
autor.

Shazeer (2020) probó sistemáticamente todas las variantes GLU y SwiGLU salió la mejor de forma
consistente. Su conclusión, citada literalmente del paper:

> *"We offer no explanation as to why these architectures seem to work; we attribute their
> success, as all else, to divine benevolence."*

No es una boutade: es honestidad sobre el estado del asunto. SwiGLU se usa hoy en Llama,
Mistral, PaLM y casi todo lo demás, y la justificación es que funciona mejor en los benchmarks.
No hay una teoría.

Lo mismo pasa con el 4x, que es una constante que nadie ha derivado, y con la interpretación del
FFN como memoria clave-valor, que es una hipótesis razonable con evidencia parcial. Y ojo
también con lo que enseña la demo: su cuarto experimento entrena un FFN clásico y un SwiGLU a
igualdad aproximada de parámetros sobre una tarea inventada, y gana SwiGLU por un margen amplio.
Eso **no demuestra nada** sobre modelos de lenguaje, y la propia demo lo dice. Un experimento de
juguete que confirma lo que ya creías es la forma más fácil de engañarse.

Conviene tener todo esto presente cuando leas explicaciones que suenan muy seguras de sí mismas,
incluidas las de este fichero.

---

**Para ampliar:** Hendrycks & Gimpel 2016,
[Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) · Shazeer 2020,
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (el paper de la cita) ·
Geva et al. 2021, [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913).
Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
