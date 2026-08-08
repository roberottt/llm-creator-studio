# 05 — Baselines: contra qué compites, y cómo se mide "lo hace mal"

## Por qué importa este módulo

**Porque necesitas saber contra qué compites.**

Vas a entrenar un modelo de 8,9 millones de parámetros. Cuando termine te dará un número
—la pérdida— y tendrás que decidir si eso es bueno. Sin una referencia, ese número no
significa nada. Un 2,49 no es bueno ni malo hasta que sabes que el suelo está en 4,13 y que
un modelo de tres líneas ya llega a 2,49.

Aquí construyes dos cosas distintas, y conviene no mezclarlas:

- **La regla de medir.** Cross-entropy y perplejidad, que no son fórmulas arbitrarias: tienen
  una interpretación exacta que conviene entender antes de mirar una curva de entrenamiento
  durante horas. Y sobre todo **el suelo**, `ln(V)`, que vas a usar el resto del curso como
  detector de bugs.
- **Tres modelos previos al Transformer**, cada uno menos malo que el anterior. No los vas a
  usar para nada después. Existen para dos cosas: para tener contra qué comparar, y porque
  el punto exacto donde el tercero se queda atascado es el problema que la atención viene a
  resolver en el módulo 06. Si te saltas esto, la atención parece magia arbitraria; si lo
  haces, aparece como la respuesta obvia a algo que has visto fallar con tus propios ojos.

### Qué sabrás al terminar

- Cómo se mide "se ha equivocado" en un número, y **por qué con un logaritmo**
- Qué es la perplejidad y cómo leerla de un vistazo
- El número `ln(V)` que te va a decir, en el paso 0 de cualquier entrenamiento, si hay un bug
- Que contar y aprender por gradiente dan **exactamente lo mismo** cuando el modelo es simple
- Cómo se escribe un modelo en PyTorch: `nn.Module`, `forward`, y de qué forma son los
  tensores que entran y salen. Es tu primera vez, y el patrón no cambia hasta el GPT final
- Por qué mirar más contexto ayuda, y por qué la forma ingenua de hacerlo se rompe

### Qué vas a escribir

Cinco ejercicios. Esta teoría está ordenada para que los leas en este orden, y **cada uno
tiene su propia sección con su ejemplo numérico**:

| Ejercicio | Qué hace | Dónde se explica |
|---|---|---|
| 1. `uniform_baseline_loss` | El suelo: `ln(V)` | [§ El suelo](#el-suelo-lo-que-saca-un-modelo-que-no-sabe-nada) |
| 2. `bigram_counts` | Contar qué token sigue a cada token | [§ Contar](#ejercicio-2-contar-los-pares-bigram_counts) |
| 3. `bigram_nll` | Medir cómo de bien predice esa tabla | [§ Medir la tabla](#ejercicio-3-medir-la-tabla-bigram_nll) |
| 4. `NeuralBigram` | El mismo modelo, aprendido por gradiente | [§ El bigrama neuronal](#ejercicio-4-el-mismo-modelo-pero-aprendido-neuralbigram) |
| 5. `BengioMLP` | El abuelo de los LLM (2003) | [§ El MLP de Bengio](#ejercicio-5-mirar-más-atrás-bengiomlp) |

Los tres primeros son funciones sueltas y cortas. Los dos últimos son **tus dos primeros
modelos en PyTorch**, y entre el ejercicio 3 y el 4 hay una sección aparte
([§ Qué es un modelo en PyTorch](#un-alto-en-el-camino-qué-es-un-modelo-en-pytorch)) que
traduce lo que ya hiciste a mano en el módulo 02 al vocabulario de `torch.nn`. Léela antes
de abrir el ejercicio 4; sin ella, `nn.Embedding` y `F.cross_entropy` parecen dos nombres
que hay que copiar sin saber qué hacen.

### Cuánto cuesta

2 horas. El código es poco y casi todo está dictado línea a línea en los docstrings. El
tiempo se va en entender qué significan los números que salen, que es de lo que va el
módulo.

---

## Qué parte del LLM es esta

Construir un LLM son cuatro trabajos distintos, y el curso los recorre en este orden:

```
   0. FUNDAMENTOS      qué es un LLM, PyTorch, autograd        módulos 00-02   ✔ hecho
   1. TOKENIZADOR      texto  ->  números                      módulo 03       ✔ hecho
   2. DATOS            números  ->  tarea de aprendizaje       módulo 04       ✔ hecho
   3. MODELO           la arquitectura que hace la predicción  módulos 05-10   ← ESTÁS AQUÍ
   4. ENTRENAMIENTO    ajustar los pesos hasta que acierte     módulos 11-13
   ────────────────────────────────────────────────────────────────────────────
      y después: generar texto (14), evaluar (15), afinar a instrucciones (16)
```

Entras en la parte del modelo, y esta es la primera parada. Pero ojo con una cosa, porque
si no se dice explícitamente confunde: **ninguno de los tres modelos de este módulo forma
parte del GPT final.** No estás construyendo una pieza que luego encaje; estás construyendo
tres callejones sin salida, a propósito.

Merece la pena porque los tres fallan de formas distintas y muy informativas:

```
   modelo               qué mira para predecir        dónde se rompe
   ───────────────────────────────────────────────────────────────────────────────
   uniforme             nada                          no es un modelo, es el suelo
   bigrama              1 token atrás                 el contexto es ridículo
   MLP de Bengio        k tokens atrás, fijos         los parámetros crecen con k,
                                                      y trata todas las posiciones
                                                      por igual
   ───────────────────────────────────────────────────────────────────────────────
   atención (mód. 06)   todo el contexto, y ELIGE     ← lo que arregla las dos cosas
```

La última fila es el módulo siguiente. Todo este módulo existe para que esa fila se lea
como una solución y no como una ocurrencia.

---

## El problema: poner un número a "se ha equivocado"

Un modelo de lenguaje no da una respuesta, da una **distribución de probabilidad** sobre
todo el vocabulario: un número por cada token posible, todos positivos y sumando 1. ¿Cómo
puntúas eso?

Imagina un vocabulario de solo 4 palabras: `[gato, perro, casa, azul]`. Ante `"el "`, el
modelo dice:

```
gato   0.70
perro  0.16
casa   0.03
azul   0.11
```

Y la palabra que venía de verdad era `gato`. Lo ha hecho bien. Si hubiera venido `azul`, lo
habría hecho mal. Pero *cuánto* de mal, en un número.

La respuesta que se usa en todo el campo:

$$\text{pérdida} = -\ln(\text{probabilidad que el modelo dio al token correcto})$$

Con los números de arriba:

```
si vino gato:  -ln(0.70) = 0.352     bien
si vino azul:  -ln(0.11) = 2.207     mal
```

Fíjate en lo que **no** se mira: no importa nada lo que el modelo dijera sobre `perro` o
sobre `casa`. Solo cuenta la probabilidad que le dio al token que de verdad vino. Y en el
extremo: si hubiera dado 0,99 a `gato`, la pérdida sería 0,010; si hubiera dado 0,001, sería
6,908.

### Antes de la probabilidad están los logits

Un detalle que hay que tener claro antes del ejercicio 4, porque es la forma real en la que
esto ocurre dentro de una red: **el modelo no produce probabilidades, produce logits.** Un
logit es una puntuación en bruto, un número cualquiera, positivo o negativo, sin ninguna
restricción. La red escupe cuatro números y ya está:

```
   logits:   gato  2.0     perro  0.5     casa  -1.0     azul  0.1
```

Eso no es una distribución: hay un negativo y no suman 1. Convertirlos en probabilidades es
el trabajo del **softmax**, que exponencia cada uno (con lo que todos se vuelven positivos)
y divide entre la suma (con lo que suman 1):

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

```
   exponenciar:   e^2.0 = 7.389    e^0.5 = 1.649    e^-1.0 = 0.368    e^0.1 = 1.105
   la suma:       7.389 + 1.649 + 0.368 + 1.105 = 10.511
   dividir:       0.7030           0.1569          0.0350            0.1051
```

Que son, con dos decimales más, los números del ejemplo de arriba. Y la pérdida si vino
`gato` es `-ln(0.7030) = 0.3524`.

**Los dos pasos —softmax y luego `-ln`— son lo que hace `F.cross_entropy` de una sola vez.**
Por eso en los ejercicios 4 y 5 le pasas los *logits* directamente y nunca llamas a softmax:
si lo hicieras, estarías aplicándolo dos veces. Es un error clásico y silencioso, porque el
modelo sigue entrenando, solo que peor. PyTorch los fusiona en una única operación por
estabilidad numérica: exponenciar un logit de 50 desborda en float32, y el truco para
evitarlo (restar el máximo antes de exponenciar) vive dentro de `cross_entropy`.

## Por qué el logaritmo, y no otra cosa

Tres razones, y las tres importan.

**1. Convierte productos en sumas.** La probabilidad de una frase entera es el producto de
las probabilidades de cada token: $P(w_1)P(w_2|w_1)P(w_3|w_1w_2)\cdots$. Con 500 tokens de
probabilidad ~0,1 cada uno, ese producto es $10^{-500}$: cero exacto en coma flotante. En
logaritmos es una suma de 500 números en torno a $-2{,}3$, perfectamente manejable.

**2. Castiga muy duro estar seguro y equivocarse.** La curva de $-\ln(p)$ se dispara a
infinito cuando $p \to 0$. Un modelo que dice "estoy segurísimo" y falla recibe una
penalización enorme; uno que reparte sus apuestas recibe una moderada. Esto empuja a los
modelos hacia la calibración, no solo hacia acertar.

**3. Tiene una interpretación exacta.** $-\log_2(p)$ es el número de *bits* que necesitarías
para transmitir ese token si codificaras el mensaje usando las probabilidades del modelo. Un
modelo de lenguaje **es** un compresor: cuanto mejor predice, menos bits necesita. Esta
equivalencia entre predicción y compresión viene de Shannon (1948) y no es una analogía, es
una identidad.

Promediando sobre todos los tokens se obtiene la **cross-entropy**, que es la función que
minimiza cualquier LLM:

$$L = -\frac{1}{N}\sum_{i=1}^{N} \ln P(\text{token}_i \mid \text{contexto}_i)$$

Léela con el ejemplo delante: cada sumando es un `-ln(0.70)` de los de antes, uno por token
del corpus, y `N` es cuántos tokens has evaluado. La media es lo que hace que el número sea
comparable entre corpus de tamaños distintos: **la pérdida son nats por token**, no nats
totales.

Un *nat* es la unidad que sale de usar logaritmo natural en vez de logaritmo en base 2. Un
nat son 1,44 bits. Todo el curso trabaja en nats porque es lo que devuelve `torch.log`, y no
hay más misterio que ese.

## Perplejidad: la misma cosa en unidades legibles

Una pérdida de 2,49 no dice mucho a simple vista. La **perplejidad** es simplemente $e^L$, y
sí se interpreta:

$$\text{PPL} = e^{L}$$

Significa **entre cuántas opciones equiprobables está dudando el modelo, efectivamente**.
Perplejidad 10 quiere decir que, en promedio, el modelo está tan indeciso como si eligiera
al azar entre 10 palabras. Perplejidad 1 sería un modelo perfecto.

Con los números que sacarás en la demo de este módulo, sobre Shakespeare a nivel carácter
con 62 caracteres distintos:

| modelo | pérdida | perplejidad | cómo se lee |
|---|---|---|---|
| uniforme | 4,1271 | 62,0 | duda entre los 62 caracteres, o sea no sabe nada |
| bigrama | 2,4916 | 12,1 | duda entre 12: ya ha descartado 50 |
| MLP de Bengio (ctx 4) | 2,0939 | 8,1 | duda entre 8 |

La perplejidad es la magnitud que se publica en los papers, y la pérdida es la que se mira
mientras se entrena. Son el mismo número.

## El suelo: lo que saca un modelo que no sabe nada

Aquí está el número más útil de todo el entrenamiento. Un modelo que reparte la probabilidad
por igual entre las $V$ palabras del vocabulario da $P = 1/V$ a todas, así que:

$$L_{\text{uniforme}} = -\ln(1/V) = \ln(V)$$

Con nuestro vocabulario final de 4096: $\ln(4096) = 8{,}317$. Perplejidad 4096, que era de
esperar: si dudas por igual entre 4096 opciones, estás dudando entre 4096 opciones.

**Úsalo así:** cuando arranques cualquier entrenamiento, la pérdida del primer paso tiene que
valer casi exactamente `ln(V)`. Ni más ni menos.

- Si sale **mucho más alta** (12, 20), la inicialización está mal: el modelo empieza con
  opiniones fuertes y equivocadas en vez de con ignorancia honesta.
- Si sale **más baja**, hay fuga de información: casi siempre, la máscara causal mal puesta
  y el modelo viendo la respuesta.

El segundo caso parece una buena noticia y es el bug más caro del curso. La pérdida baja
espectacularmente, todo parece ir de maravilla, y el modelo entrenado no sirve para nada
porque en el momento de generar texto ese futuro que estaba mirando no existe.

**Y el primer caso lo vas a ver en vivo en la demo de este módulo**, lo cual es una suerte
porque enseña de dónde sale. El `NeuralBigram` arranca en 4,6434 cuando el suelo es 4,1271:
medio nat de más. La causa es que `nn.Embedding` inicializa sus pesos con una normal de
desviación 1, y como en ese modelo las filas de la tabla *son* directamente los logits, el
modelo empieza con apuestas fuertes y aleatorias. Un logit de +2 frente a otro de −2 es
apostar 55 a 1 antes de haber visto un solo dato, y acertar por azar es improbable. Ese medio
nat es literalmente el precio de opinar sin información.

Por eso el GPT del módulo 10 inicializa todo con `std=0.02`: con logits casi idénticos el
softmax sale casi uniforme y el paso 0 cae justo sobre `ln(V)`.

### Ejercicio 1: el suelo en una línea (`uniform_baseline_loss`)

Es una línea: `return math.log(vocab_size)`, más una comprobación de que el vocabulario es
positivo. El ejercicio no tiene ninguna dificultad y es el más importante del módulo, porque
lo que estás escribiendo no es una función sino **la comprobación que vas a hacer en el paso
0 de todos los entrenamientos que queden**.

Los dos números que te vas a encontrar de verdad:

```
   vocab   62   ->  ln(62)   = 4.1271     Shakespeare a nivel carácter (las demos)
   vocab 4096   ->  ln(4096) = 8.3178     el modelo final sobre TinyStories
```

(El fichero entero de Shakespeare tiene 65 caracteres distintos y `ln(65) = 4.1744`; la demo
de este módulo se queda con los primeros 200.000 caracteres y ahí aparecen 62. Si ves los
dos números por el curso, es por eso: el suelo depende del vocabulario que uses de verdad, no
de una constante universal.)

---

## Ejercicio 2: contar los pares (`bigram_counts`)

**El problema.** Quieres un modelo de lenguaje sin entrenar nada. La forma más tonta que
funciona: mira el corpus, apunta cuántas veces cada carácter siguió a cada carácter, y para
predecir consulta la tabla.

Un **bigrama** es exactamente eso: un modelo cuyo contexto es un solo token. Para predecir
el carácter 500 solo mira el 499 e ignora los 498 anteriores. Es un modelo malísimo y es
sorprendente lo lejos que llega.

**El ejemplo con números.** Vocabulario de tres caracteres, `a→0`, `b→1`, `c→2`, y el corpus
es `"ababc"`, o sea `ids = [0, 1, 0, 1, 2]`. Los pares consecutivos son:

```
   a b a b c
   └─┘         (0,1)
     └─┘       (1,0)
       └─┘     (0,1)
         └─┘   (1,2)
```

Cuatro pares para cinco tokens: siempre uno menos que la longitud, porque el último carácter
no tiene sucesor. Y la matriz de conteos, donde la **fila es el "desde"** y la **columna es
el "hasta"**:

```
             hasta a   hasta b   hasta c
   desde a       0         2         0        <- la b siguió a la a dos veces
   desde b       1         0         1
   desde c       0         0         0        <- la c nunca tuvo sucesor
```

Ese es exactamente el resultado que comprueba el test del ejercicio.

**La forma.** La matriz es `(V, V)`: una fila por cada token que puede estar delante, una
columna por cada token que puede venir detrás. Con `V = 4096` son 16,7 millones de casillas,
y esa es la primera pista de por qué contar no escala: para trigramas necesitarías `V³`, que
son 68 mil millones de casillas, casi todas a cero.

**Lo que dicta el docstring**, y por qué. La forma obvia de rellenar la tabla es un `for`
sobre los pares. Funciona y es más legible, pero con 500 millones de tokens son 500 millones
de iteraciones de Python. La versión vectorizada usa que `tokens[:-1]` es la lista de todos
los "desde" y `tokens[1:]` la de todos los "hasta", y le pide a PyTorch que sume 1 en cada
posición `(desde, hasta)` de golpe con `index_put_(..., accumulate=True)`.

**El `accumulate=True` no es opcional y es la trampa del ejercicio.** Sin él, `index_put_`
*asigna* en vez de sumar: cada par repetido pisa al anterior y todos los conteos acaban
valiendo 1 en vez de su frecuencia real. Con el corpus `[0,0,0,0,0]`, lo correcto es
`counts[0][0] = 4`; sin `accumulate` sale 1. Hay un test dedicado exactamente a eso.

---

## Ejercicio 3: medir la tabla (`bigram_nll`)

Tienes conteos. Para que sean un modelo hacen falta probabilidades, y para saber si el
modelo es bueno hace falta evaluarlo **sobre texto que no usaste para contar** (la partición
de validación del módulo 04; medir sobre el mismo texto que contaste solo te dice cuánto ha
memorizado).

**De conteos a probabilidades: normalizar por filas.** La fila `a` de la tabla de arriba es
`[0, 2, 0]`, que suma 2. Dividiendo: `P(a|a)=0`, `P(b|a)=1`, `P(c|a)=0`. El modelo dice que
después de una `a` viene una `b` con certeza absoluta.

Y ahí está el desastre. Si en validación aparece el par `a→c`, que nunca viste, su
probabilidad es 0, su logaritmo es $-\infty$, y **como la pérdida es una media, ese único
$-\infty$ se lleva por delante el resultado entero**. La perplejidad de todo tu conjunto de
validación se va a infinito por un par que no viste. No es una hipótesis: pásale
`alpha=0` a tu función con esos datos y sale `inf`.

**El arreglo: suavizado de Laplace.** Sumas una constante $\alpha$ a *todos* los conteos
antes de normalizar. Es admitir que "no lo he visto" no es lo mismo que "es imposible".

$$P(b \mid a) = \frac{C_{ab} + \alpha}{\sum_{b'} C_{ab'} + \alpha V}$$

Con $\alpha = 1$ y $V = 3$, la tabla del ejercicio 2 se convierte en:

```
   fila a:  [0,2,0] + 1  =  [1,3,1]  suma 5   ->  [0.200, 0.600, 0.200]
   fila b:  [1,0,1] + 1  =  [2,1,2]  suma 5   ->  [0.400, 0.200, 0.400]
   fila c:  [0,0,0] + 1  =  [1,1,1]  suma 3   ->  [0.333, 0.333, 0.333]
```

Mira la fila `c`: un token del que no viste ni un solo sucesor acaba con una distribución
uniforme. Eso es exactamente lo que quieres que diga un modelo que no tiene información.
Y mira el denominador de la fila `a`: es 5, no 2. Al sumar $\alpha$ a las $V$ entradas de la
fila, el total creció en $\alpha V$. **Ese $\alpha V$ de la fórmula no lo escribes tú**:
aparece solo si sumas primero y sumas la fila después, que es el orden que dicta el
docstring. Si dividieras por `suma_original + alpha`, las probabilidades no sumarían 1.

**Y ahora la pérdida.** Evalúa sobre `"abc"`, o sea `[0, 1, 2]`. Los pares son `(a,b)` y
`(b,c)`, y sus probabilidades en la tabla suavizada son 0,600 y 0,400:

```
   -ln(0.600) = 0.5108
   -ln(0.400) = 0.9163
   media      = 0.7136   <- la pérdida, en nats por token
   e^0.7136   = 2.041    <- la perplejidad
```

Y el suelo de este vocabulario es `ln(3) = 1.0986`. El modelo baja de él, o sea que ha
aprendido algo. Esos son los números exactos que devuelve la función si la implementas bien.

**Cuánto suavizar.** $\alpha$ es un mando con dos extremos, y los dos son malos:

| alpha | pérdida en validación (Shakespeare) | qué pasa |
|---|---|---|
| 0,0001 | 2,4892 | casi sin suavizar; con un par no visto, `inf` |
| 0,01 | 2,4834 | el mejor de la tabla |
| 1,0 | 2,4916 | el valor clásico, razonable |
| 100 | 2,9337 | los conteos reales empiezan a ahogarse |
| 10000 | 4,0430 | prácticamente uniforme: casi el suelo (4,1271) |

Subir $\alpha$ empuja al modelo hacia la ignorancia. Con $\alpha$ enorme, los conteos reales
son ruido frente a la constante que sumaste y todas las filas salen casi uniformes. Es la
primera vez en el curso que ves un **hiperparámetro**: un número que eliges tú, que no se
aprende, y cuyo valor óptimo se busca probando.

**Las dos trampas silenciosas del ejercicio.** Ninguna da error; las dos dan números
plausibles y equivocados.

- **`keepdim=True` al sumar la fila.** Sin él, `sum(dim=1)` devuelve forma `(V,)` en vez de
  `(V, 1)`, y las reglas de broadcasting de PyTorch acaban dividiendo por **columnas** en
  lugar de por filas. Sale un número perfectamente creíble y completamente incorrecto. Hay un
  test que lo caza comprobando que cada fila suma 1.
- **`.double()` y no `.float()`.** Con corpus grandes se suman millones de conteos, y float32
  tiene 24 bits de mantisa: empieza a perder precisión antes de lo que uno espera.

---

## Un alto en el camino: qué es un modelo en PyTorch

Los dos ejercicios que quedan son tus primeros modelos en `torch.nn`, y todo lo que aprendas
aquí lo vas a repetir sin cambios hasta el GPT del módulo 10. Merece la pena parar y traducir
el vocabulario, porque lo conceptual **ya lo hiciste a mano en el módulo 02**: allí montaste
un MLP con tu propio motor de autodiff y escribiste el bucle de entrenamiento entero. Esto
es lo mismo con las piezas ya hechas.

### `nn.Module` y `forward`

Un modelo es una clase que hereda de `nn.Module` y define dos cosas:

```python
class MiModelo(nn.Module):
    def __init__(self, ...):
        super().__init__()          # esta línea nunca se olvida
        self.capa = nn.Linear(...)  # aquí se CREAN los pesos

    def forward(self, x):
        return self.capa(x)         # aquí se USAN
```

Lo único que hay que saber de fondo: cuando asignas un `nn.Linear` o un `nn.Embedding` a un
atributo, `nn.Module` **lo registra**. Eso es lo que hace que `modelo.parameters()` los
encuentre todos, que `modelo.to(device)` los mueva a la GPU y que guardar el modelo guarde
los pesos. Es el equivalente automático del `parameters()` que en el módulo 02 escribiste tú
recorriendo neuronas a mano.

Y una convención que sorprende la primera vez: **el modelo se llama como `modelo(x)`, nunca
como `modelo.forward(x)`**. Son casi lo mismo, pero la primera forma pasa por los ganchos
internos de PyTorch y la segunda se los salta. En este curso da igual; en cuanto uses hooks
o `DataParallel`, no.

### La forma de los tensores: `(B, T, V)`

Tres letras que vas a ver en todos los comentarios del curso:

```
   B  batch       cuántas secuencias procesas a la vez        (paralelismo)
   T  time        cuántos tokens tiene cada secuencia         (el contexto)
   V  vocab       cuántos tokens distintos existen
```

El recorrido de un modelo de lenguaje es siempre este:

```
   idx      (B, T)        enteros: los ids de los tokens de entrada
     │
     │  el modelo
     ▼
   logits   (B, T, V)     floats: una puntuación por cada token posible,
                          en cada posición, de cada secuencia
```

O sea: por cada una de las `B × T` posiciones, el modelo emite `V` números. Con `B=32`,
`T=512` y `V=4096` eso son 67 millones de floats en un solo tensor, y por eso los logits son
el mayor consumidor de memoria del entrenamiento final, por encima incluso de las
activaciones.

### `F.cross_entropy` y el `reshape`

`F.cross_entropy` espera exactamente dos cosas: los logits en forma `(N, V)` y los targets
en forma `(N,)`, donde `N` es "cuántas predicciones estás puntuando" y cada target es el
**índice** del token correcto (no un vector one-hot).

Tú tienes `(B, T, V)` y `(B, T)`. La traducción es aplanar batch y tiempo en una sola
dimensión, porque a la pérdida le da exactamente igual de qué secuencia venía cada
predicción:

```python
loss = F.cross_entropy(
    logits.reshape(-1, self.vocab_size),   # (B, T, V) -> (B*T, V)
    targets.reshape(-1),                   # (B, T)    -> (B*T,)
)
```

Este par de líneas es idéntico en el `NeuralBigram` y en el GPT final. Vale la pena
reconocerlo ahora.

### Por qué `forward` devuelve `(logits, loss)`

Los dos modelos de este módulo devuelven una tupla, y `loss` es `None` cuando no le pasas
targets. La razón es que hay dos situaciones distintas:

- **Entrenando** tienes la respuesta correcta, quieres la pérdida, y de los logits no haces
  nada.
- **Generando** (módulo 14) no hay respuesta correcta que valga: quieres los logits para
  muestrear de ellos el siguiente token.

Devolver `None` y no `0` en el segundo caso es deliberado: un `0` se sumaría alegremente a
cualquier cosa y el bug pasaría desapercibido; un `None` revienta en el acto.

---

## Ejercicio 4: el mismo modelo, pero aprendido (`NeuralBigram`)

**La idea.** Coge el bigrama del ejercicio 2 y, en vez de rellenar la tabla contando,
inicialízala al azar y deja que el descenso de gradiente la ajuste. El modelo entero es una
`nn.Embedding(V, V)`: una tabla de `V` filas por `V` columnas donde **la fila `i` son
directamente los logits del token que sigue al token `i`**.

Parece un truco de escritura y es literalmente el mismo modelo. Lo interesante es el
resultado: entrenado con cross-entropy, converge a los conteos normalizados. Los números
medidos sobre Shakespeare:

```
   contando   (ejercicio 2 + 3):  2.4916
   aprendiendo (este ejercicio):  2.4838
```

**El mismo modelo llega al mismo sitio por dos caminos completamente distintos**, y la
diferencia de 0,008 es ruido de suavizado y de cuántos pasos entrenaste. Contar es
instantáneo y aprender tarda unos segundos, así que en este punto contar gana. La cuestión
es que contar se acaba aquí y aprender no: para el modelo del ejercicio 5 ya no existe
ninguna tabla que rellenar, y para el GPT del módulo 10 mucho menos.

**Por qué `nn.Embedding` y no `nn.Linear`.** Son la misma operación. Un embedding es un
`Linear` cuya entrada es un vector one-hot: multiplicar una matriz por un vector que es todo
ceros y un único 1 en la posición `i` da, exactamente, la fila `i` de la matriz. La
diferencia es puramente de coste: el embedding **lee** la fila que necesita en lugar de
hacer la multiplicación. Con `V=4096`, leer 4096 números frente a hacer 16,7 millones de
multiplicaciones que en su inmensa mayoría son por cero.

Esa equivalencia one-hot ↔ fila conviene tenerla masticada: reaparece en el módulo 09 con
los embeddings posicionales y en el módulo 10 con el *weight tying*.

**Cuidado con el nombre.** `self.token_embedding` no es opcional: el test copia pesos por
nombre para comparar tu modelo con la referencia, y si lo llamas de otra forma falla sin
que el modelo tenga nada malo.

---

## Ejercicio 5: mirar más atrás (`BengioMLP`)

Este es el modelo de Bengio et al. (2003), *A Neural Probabilistic Language Model*, y es el
abuelo directo de todo esto: el primer modelo de lenguaje neuronal que funcionó de verdad,
trece años antes del Transformer.

**El problema que resuelve.** El bigrama mira un token. Predecir el final de *"el gato se
subió al ___"* mirando solo `al` es desesperado. Quieres mirar `k` tokens atrás. Con conteos
no puedes: la tabla necesitaría $V^k$ casillas y estarían casi todas vacías (la *maldición de
la dimensionalidad* del módulo 00). Con una red, sí.

**Las dos ideas de Bengio**, que siguen vivas veinte años después y las dos están en tu GPT
final:

1. **Cada token se representa como un vector denso aprendido**, no como un id sin estructura.
   Un id es una etiqueta: el 47 y el 48 no se parecen en nada por ser consecutivos. Un vector
   de 24 números sí puede parecerse a otro, y ahí está toda la capacidad de generalizar que
   una tabla de conteos no tiene. Si `perro` y `gato` acaban con vectores parecidos, lo que
   el modelo aprenda sobre uno le sirve para el otro **aunque la combinación exacta nunca
   apareciera en el corpus**. Es la diferencia entre memorizar y aprender.
2. **La probabilidad del siguiente token la calcula una red**, no una tabla.

**El recorrido, con las formas.** Con `block_size=4` (mira 4 caracteres atrás), `d_embed=24`
y `n_hidden=128`, sobre Shakespeare con `V=62`:

```
   idx       (B, 4)          los ids de los 4 caracteres anteriores
     │  embedding
     ▼
   emb       (B, 4, 24)      cada uno convertido en su vector de 24 números
     │  reshape(B, -1)       ← CONCATENAR: pegar los 4 vectores en fila
     ▼
   flat      (B, 96)         4 × 24 = 96
     │  hidden + tanh
     ▼
   h         (B, 128)        la capa oculta
     │  output
     ▼
   logits    (B, 62)         un logit por carácter posible
```

Y fíjate en la última fila comparada con el `NeuralBigram`: aquí los logits son `(B, V)`, sin
la dimensión `T`. Este modelo hace **una sola predicción por muestra**, no una por posición.
Por eso `targets` es `(B,)` y por eso `cross_entropy` se llama sin ningún `reshape`: las
formas ya le encajan. Es la diferencia estructural más importante entre los dos ejercicios y
la fuente más probable de confusión al escribirlos seguidos.

El `tanh` es el del paper original. La razón de que haya una no linealidad ahí la viste en el
módulo 02: sin ella, apilar dos capas lineales da otra capa lineal y la capa oculta no
serviría para nada.

**Concatenar, no promediar.** El `reshape(batch, -1)` pega los embeddings uno detrás de otro,
y eso es lo que conserva **el orden**. Si hicieras `emb.mean(dim=1)` obtendrías un vector de
24 números perfectamente válido... en el que `[el, gato, come]` y `[come, gato, el]` dan
exactamente lo mismo. El modelo perdería la noción de qué iba antes. Hay un test que lo
comprueba pasándole el contexto al revés y exigiendo que la salida cambie.

Y vigila dónde va el `-1`: `reshape(batch, -1)`, no `reshape(-1, batch)`. El segundo compila,
no da ningún error y produce basura.

### Dónde se rompe, que es la razón de estar aquí

Los números medidos en la demo, entrenando los tres con el mismo presupuesto de 400 pasos:

| modelo | contexto | pérdida (val) | parámetros |
|---|---|---|---|
| bigrama | 1 | 2,4916 | 3.844 |
| Bengio MLP | 2 | 2,1940 | 15.758 |
| Bengio MLP | 4 | **2,0939** | 21.902 |
| Bengio MLP | 8 | 2,1928 | 34.190 |

Dos cosas que sacar de ahí, y las dos son el módulo 06 asomando.

**Primera: los parámetros crecen con el contexto.** La capa oculta es
`Linear(block_size * d_embed, n_hidden)`, así que su tamaño es *lineal* en la longitud del
contexto. En la tabla se ve directamente:

```
   ctx 2:  Linear(48,  128)  =   6.272 pesos
   ctx 4:  Linear(96,  128)  =  12.416
   ctx 8:  Linear(192, 128)  =  24.704
   ...
   ctx 512 con d_embed 320:  Linear(163840, 128)  -> imposible
```

Un contexto de 512 tokens, que es el del modelo que vas a entrenar, es sencillamente
inalcanzable por este camino. Y peor: el contexto es **fijo**. Está cableado en la forma de
la capa. No puedes darle 3 tokens a un modelo entrenado con 4, ni 5.

**Segunda, y más profunda: el modelo trata cada posición como una entrada independiente.**
Los 96 números de `flat` son 96 entradas sin relación entre sí para la capa `hidden`. No hay
forma de que el modelo diga "de estos 512 tokens los que me importan ahora son el 3 y el 47".
El peso de cada posición está fijado en la matriz y es el mismo para todas las frases del
corpus, cuando lo que hace falta es que dependa de *qué* hay escrito en cada posición.

Resolver las dos cosas a la vez —contexto largo sin que exploten los parámetros, y decidir
sobre la marcha a qué posiciones hacer caso— es exactamente lo que hace la atención. Ese es
el módulo 06.

**Y una tercera lección, gratis, que no va del modelo sino de cómo se comparan modelos.**
Mira otra vez la tabla: el contexto 8 sale *peor* que el contexto 4. No es un fallo de la
demo. Los tres han entrenado los mismos 400 pasos, y el de contexto 8 tiene más del doble de
parámetros que el de contexto 2: con el mismo presupuesto de pasos, el modelo grande se queda
a medio entrenar. **Comparar arquitecturas a igualdad de pasos no es compararlas a igualdad
de cómputo, y casi siempre favorece injustamente al modelo pequeño.** Es exactamente el error
que las leyes de escala del módulo 12 vienen a corregir, y lo vas a ver aquí con tus propios
números antes de que nadie te lo cuente en abstracto.

---

## Dónde está el debate

La perplejidad es una métrica de sustitución, no el objetivo. Mide lo bien que el modelo
predice el corpus de validación; nadie quiere un modelo que prediga corpus, se quiere uno
que sea útil.

Se sabe que la correlación entre perplejidad y utilidad se rompe en los extremos. Un modelo
puede bajar la perplejidad memorizando, o especializándose en las peculiaridades de un
dataset concreto. Y comparar perplejidades entre modelos con **tokenizadores distintos no
tiene ningún sentido**: si tu vocabulario parte las palabras en trozos más pequeños, cada
token individual es más fácil de predecir y tu perplejidad sale mejor sin que el modelo sea
mejor. Por eso en el módulo 15 usaremos *bits por byte*, que sí es comparable.

Aun así, dentro de un mismo tokenizador y un mismo dataset, la perplejidad de validación
sigue siendo la señal más fiable que hay para saber si un entrenamiento va bien. Es una
herramienta buena con un alcance limitado, y conviene tener claro dónde acaba.

Sobre el suavizado hay una discusión más vieja y ya casi cerrada: Laplace es el método más
simple y no es el mejor. Kneser-Ney, que reparte la masa sobrante mirando en cuántos
contextos distintos aparece cada token en vez de por igual, gana con claridad en modelos de
n-gramas. Aquí usamos Laplace porque el modelo de n-gramas es un baseline que vas a abandonar
en el módulo siguiente, y gastar esfuerzo en afinarlo sería invertir en el callejón sin
salida. Pero merece la pena saber que existe medio siglo de literatura sobre cómo repartir la
probabilidad de lo que no se ha visto, y que toda ella dejó de importar cuando los modelos
neuronales empezaron a generalizar en vez de a suavizar.

---

**Para ampliar:** Bengio et al. 2003,
[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
· Shannon 1948, *A Mathematical Theory of Communication* (la equivalencia entre predicción y
compresión). Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
