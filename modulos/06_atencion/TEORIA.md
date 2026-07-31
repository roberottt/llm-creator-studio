# 06 — Self-attention

## Por qué importa este módulo

**Si sólo pudieras entender un módulo del curso, sería éste.**

Todo lo anterior —tokenizar, embeddings, el MLP de Bengio— existía ya en 2003 y daba
modelos mediocres. Lo que cambió en 2017 y acabó produciendo ChatGPT es exactamente lo que
vas a programar aquí, y cabe en cuatro líneas de código.

La idea es sencilla de enunciar: **dejar que cada palabra mire a las anteriores y decida a
cuáles hacer caso**. Lo difícil es creer que con eso baste. Al terminar el módulo lo habrás
visto funcionar en un modelo que entrenas tú, con un mapa de calor que muestra literalmente
a qué mira cada letra.

### Qué sabrás al terminar

- Por qué un modelo puede "recordar" algo que leyó 300 palabras antes
- Qué son Q, K y V, y **por qué hacen falta tres cosas y no una**
- Por qué se divide por `√d_k`, y qué se rompe exactamente si lo quitas (con números)
- Qué es la máscara causal y por qué es **el bug más caro del curso** si la pones mal
- Habrás visto un heatmap de atención de un modelo entrenado por ti

### Cuánto cuesta

4 horas, empatado con el 03 como el más largo. Es el que más merece la pena.

---

## ¿Qué problema resuelve la atención?

Frase: *"el gato que vi ayer dormía"*.

Para acertar `dormía` hay que saber que el sujeto es `gato`, cuatro palabras atrás. El MLP
del módulo 05 no puede: mira una ventana fija y trata todas las posiciones igual, sin forma
de decir "de estos tokens, el que me importa ahora es el primero".

La atención deja que **cada palabra mire a las anteriores y decida a cuáles hacer caso**.
No con una regla fija, sino calculándolo a partir del contenido.

## Con números de verdad

Vamos a hacerlo a mano con 3 palabras y vectores de 2 dimensiones. Digamos que después de
los embeddings tenemos:

```
gato   = [1.0, 0.2]
ayer   = [0.1, 0.9]
dormía = [0.8, 0.3]
```

`dormía` quiere saber a quién mirar. Lo hace con un **producto escalar**: mide cuánto se
parecen dos vectores. Cuanto más alineados, mayor el número.

```
gato · dormía = 1.0×0.8 + 0.2×0.3 = 0.86      -> mucho
ayer · dormía = 0.1×0.8 + 0.9×0.3 = 0.35      -> poco
dormía · dormía = 0.8×0.8 + 0.3×0.3 = 0.73    -> a sí misma
```

Esos números se llaman **puntuaciones** (*scores*). Ahora hay que convertirlos en pesos que
sumen 1, y para eso se usa softmax (exponenciar y normalizar, como el módulo 00 pero
admitiendo negativos):

```
softmax([0.86, 0.35, 0.73]) = [0.40, 0.24, 0.36]
```

Y con esos pesos se mezclan los vectores:

```
salida = 0.40×gato + 0.24×ayer + 0.36×dormía
```

Eso es la atención. **Una media ponderada donde los pesos los decide el propio contenido.**
La representación de `dormía` ahora lleva dentro un 40% de `gato`, que es exactamente la
información que necesitaba.

## Q, K, V: por qué tres proyecciones y no una

En el ejemplo he usado el mismo vector para todo, y eso es demasiado rígido. Un token
necesita hacer tres cosas distintas:

- **preguntar** algo ("busco un sujeto singular")
- **anunciarse** ante los demás ("soy un sustantivo singular")
- **aportar** contenido si resulta elegido ("el concepto de gato")

Son tres papeles diferentes, así que se aprenden tres proyecciones lineales distintas del
mismo vector de entrada:

$$Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V$$

**Query** (pregunta), **Key** (etiqueta) y **Value** (contenido). La similitud se calcula
entre queries y keys; lo que se mezcla son los values. Así el modelo puede aprender que
`gato` *responde bien* a una pregunta sobre sujetos sin que eso condicione *qué información
aporta* cuando lo eligen.

## La fórmula

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

Es exactamente lo que acabamos de hacer:

- $QK^\top$ son todos los productos escalares de golpe: una matriz $T \times T$ donde la
  casilla $(i,j)$ dice cuánto le interesa a $i$ el token $j$.
- $\sqrt{d_k}$ es el escalado (ahora lo vemos).
- $M$ es la máscara causal.
- softmax convierte cada fila en pesos que suman 1.
- multiplicar por $V$ hace la mezcla.

## El escalado por √d_k: qué pasa si lo quitas

Este divisor parece un detalle arbitrario y no lo es.

Un producto escalar de dos vectores de dimensión $d_k$ con componentes independientes de
media 0 y varianza 1 tiene **varianza $d_k$**. Con $d_k = 40$ (nuestro caso), las
puntuaciones se mueven en un rango de $\pm 6$ aproximadamente. Con $d_k = 512$, en $\pm 22$.

¿Y qué? Que softmax es exponencial. Si una puntuación destaca 20 unidades sobre el resto,
$e^{20}$ frente a $e^{0}$ son 485 millones a uno: el softmax devuelve prácticamente
`[0, 0, ..., 1, ..., 0]`. La atención se vuelve una selección dura de un único token.

Y el problema de verdad no es el forward, es el **gradiente**. La derivada del softmax es
$p(1-p)$; con $p$ pegado a 0 o a 1, la derivada es prácticamente cero. La capa deja de
aprender. Dividir por $\sqrt{d_k}$ devuelve las puntuaciones a varianza 1, el softmax se
queda en una zona blanda, y el gradiente fluye.

La demo del módulo te lo enseña midiendo la entropía de la distribución con y sin escalado.

## La máscara causal

Al entrenar le pasamos toda la secuencia de golpe y le pedimos que prediga cada token a
partir de los anteriores. Sin más, la posición 3 podría mirar a la 4 — o sea, ver la
respuesta.

La máscara pone $-\infty$ en las puntuaciones prohibidas *antes* del softmax. Como
$e^{-\infty} = 0$, esas posiciones reciben peso exactamente cero:

```
[[ ✓  ·  ·  · ]      el token 0 solo se ve a sí mismo
 [ ✓  ✓  ·  · ]      el 1 ve al 0 y a sí mismo
 [ ✓  ✓  ✓  · ]
 [ ✓  ✓  ✓  ✓ ]]
```

Se pone antes del softmax y no después por una razón concreta: si borraras los pesos
después, las filas ya no sumarían 1. Poniendo $-\infty$ antes, el softmax normaliza solo
sobre lo permitido.

**Este es el bug más caro del curso.** Si la máscara está mal, la pérdida baja
espectacularmente, todo parece ir de maravilla, y el modelo entrenado no sirve para nada
porque en generación ese futuro no existe. Por eso el módulo 05 insiste en comparar la
pérdida del paso 0 contra $\ln(V)$: si sale *más baja*, mira la máscara.

## Multi-head: varias en paralelo

Una sola atención tiene que resolver todas las relaciones de la frase con un único patrón.
La solución es hacer varias en paralelo, cada una con sus propias $W_Q, W_K, W_V$, y
concatenar los resultados.

Con $d_{\text{model}} = 320$ y 8 cabezas, cada una trabaja en $40$ dimensiones
($320/8$). **No cuesta más**: en vez de una atención de 320 dimensiones haces ocho de 40, y
el total de parámetros es idéntico.

Lo interesante es que las cabezas se especializan solas. En modelos entrenados se han
identificado cabezas que miran al token anterior, cabezas que emparejan comillas de
apertura y cierre, y las llamadas *induction heads*, que detectan el patrón "…A B … A" y
predicen B. Nadie las programó.

El truco de implementación: no se hacen 8 proyecciones separadas. Se hace una de
$320 \to 320$ y se parte el resultado en 8 trozos de 40. Es matemáticamente equivalente y
mucho más rápido, porque es un matmul grande en lugar de ocho pequeños.

## Dónde está el debate

Se sabe *qué* calcula la atención. Por qué funciona tan bien es harina de otro costal.

La explicación intuitiva —"cada token recupera información relevante"— es una historia
razonable y no está demostrada. Hay resultados que la complican: modelos con patrones de
atención **fijos y aleatorios** funcionan sorprendentemente bien en algunas tareas, lo que
sugiere que parte del mérito está en la arquitectura general (residuales, normalización,
profundidad) y no solo en el mecanismo de atención.

La línea de trabajo más seria en esta dirección es la de interpretabilidad mecanicista, que
trata de leer los circuitos que se forman dentro. Ha conseguido explicar componentes
concretos —las *induction heads* son el caso de éxito— pero está muy lejos de dar cuenta de
un modelo entero.

Y hay una limitación estructural que sigue sin resolverse: el coste crece con el **cuadrado**
de la longitud del contexto. Se han propuesto decenas de alternativas subcuadráticas
(Linformer, Performer, Mamba y familia). Ninguna ha desplazado a la atención estándar en
modelos de propósito general, y no está claro si es porque la atención completa es
necesaria o porque tiene veinte años de ventaja en optimización de kernels.

---

**Para ampliar:** Vaswani et al. 2017,
[Attention Is All You Need](https://arxiv.org/abs/1706.03762) · Elhage et al. 2021,
[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
(las *induction heads*). Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
