# 05 — Baselines: cómo se mide "cómo de mal lo hace"

Antes de construir un Transformer hay que responder a dos preguntas que van juntas: **¿cómo
se mide si un modelo es bueno?** y **¿contra qué hay que compararlo?**

Si tu modelo de 9 millones de parámetros no le gana a una tabla de conteos, no tienes un
modelo: tienes un bug.

## El problema: poner un número a "se ha equivocado"

Un modelo de lenguaje no da una respuesta, da una distribución de probabilidad sobre todo
el vocabulario. ¿Cómo puntúas eso?

Imagina un vocabulario de solo 4 palabras: `[gato, perro, casa, azul]`. Ante `"el "`, el
modelo dice:

```
gato   0.70
perro  0.10
casa   0.10
azul   0.10
```

Y la palabra que venía de verdad era `gato`. Lo ha hecho bien. Si hubiera venido `azul`, lo
habría hecho mal. Pero *cuánto* de mal, en un número.

La respuesta que se usa en todo el campo:

$$\text{pérdida} = -\ln(\text{probabilidad que el modelo dio al token correcto})$$

Con los números de arriba:

```
si vino gato:  -ln(0.70) = 0.357     bien
si vino azul:  -ln(0.10) = 2.303     mal
```

Y en el extremo: si el modelo hubiera dado 0,99 a `gato`, la pérdida sería 0,010. Si hubiera
dado 0,001, sería 6,908.

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

## Perplejidad: la misma cosa en unidades legibles

Una pérdida de 3,2 no dice mucho a simple vista. La **perplejidad** es simplemente
$e^L$, y sí se interpreta:

$$\text{PPL} = e^{L}$$

Significa **entre cuántas opciones equiprobables está dudando el modelo, efectivamente**.
Perplejidad 10 quiere decir que, en promedio, el modelo está tan indeciso como si eligiera
al azar entre 10 palabras. Perplejidad 1 sería un modelo perfecto.

## El suelo: lo que saca un modelo que no sabe nada

Aquí está el número más útil de todo el entrenamiento. Un modelo que reparte la probabilidad
por igual entre las $V$ palabras del vocabulario da $P = 1/V$ a todas, así que:

$$L_{\text{uniforme}} = -\ln(1/V) = \ln(V)$$

Con nuestro vocabulario de 4096: $\ln(4096) = 8{,}317$. Perplejidad 4096, que era de
esperar.

**Úsalo así:** cuando arranques el entrenamiento en el módulo 11, la pérdida del primer paso
tiene que valer casi exactamente 8,317. Ni más ni menos.

- Si sale **mucho más alta** (12, 20), la inicialización está mal: el modelo empieza con
  opiniones fuertes y equivocadas en vez de con ignorancia honesta.
- Si sale **más baja**, hay fuga de información: casi siempre, la máscara causal mal puesta
  y el modelo viendo la respuesta.

Es la comprobación más barata y más informativa que existe, y aparece ya en el módulo 10.

## Los tres baselines que vas a construir

**Bigrama por conteo.** Cuentas cuántas veces sigue cada token a cada token, normalizas, y
ya tienes un modelo. Es el módulo 00 con más rigor.

Aquí aparece un problema serio: si un par nunca apareció en entrenamiento, su probabilidad
es 0, su logaritmo es $-\infty$, y **la perplejidad de todo el conjunto de validación se va
a infinito por un solo par no visto**. La solución clásica es el suavizado de Laplace:
sumar $\alpha$ a todos los conteos antes de normalizar.

$$P(b \mid a) = \frac{C_{ab} + \alpha}{\sum_{b'} C_{ab'} + \alpha V}$$

Es admitir que "no lo he visto" no es lo mismo que "es imposible".

**Bigrama neuronal.** El mismo modelo, escrito como red: una `nn.Embedding(V, V)` donde la
fila $i$ son directamente los logits del token que sigue a $i$. Entrenado con descenso de
gradiente converge a los conteos normalizados. Sirve para ver que *contar* y *aprender* dan
lo mismo cuando el modelo es lo bastante simple — y que a partir de ahí, aprender escala y
contar no.

**MLP de Bengio (2003).** El abuelo de todo esto. Concatena los embeddings de los $k$ tokens
anteriores y los pasa por un MLP. Dos ideas suyas siguen vivas veinte años después:
representar palabras como vectores densos aprendidos, y modelar la probabilidad con una red.
Su limitación es exactamente lo que la atención viene a resolver: el contexto es de tamaño
fijo, y como concatena, el número de parámetros crece linealmente con la longitud del
contexto.

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

---

**Para ampliar:** Bengio et al. 2003,
[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
· Shannon 1948, *A Mathematical Theory of Communication* (la equivalencia entre predicción y
compresión). Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
