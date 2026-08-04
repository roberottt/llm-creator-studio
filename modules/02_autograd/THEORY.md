# 02 — Autodiferenciación desde cero

## Por qué importa este módulo

**Para que `loss.backward()` deje de ser magia.**

Esa línea es la que hace que una red aprenda. Calcula, de golpe, cómo hay que mover cada uno
de los 8,9 millones de parámetros de tu modelo para que se equivoque menos. Y casi todo el
mundo que la escribe no tiene ni idea de qué hace por dentro.

En este módulo la escribes tú, en unas 100 líneas y sin usar PyTorch. Al terminar, cuando
tu entrenamiento no converja, sabrás dónde mirar en vez de probar cosas al azar.

Es el módulo más "de matemáticas" del curso, y también el que más te va a servir para
depurar todo lo demás.

### Qué sabrás al terminar

- Qué es un gradiente y por qué la red los necesita
- Cómo se calculan **todos** los gradientes de golpe, en el tiempo de dos forwards
- Por qué existe `optimizer.zero_grad()` y qué pasa exactamente si lo olvidas
- Qué hace PyTorch por dentro cuando llamas a `.backward()`

### Cuánto cuesta

3 horas. Es denso, pero es la base de todo lo que viene.

---

## El problema: ¿hacia dónde muevo cada peso?

Entrenar es esto: tienes una función que mide cómo de mal lo estás haciendo (la pérdida), y
quieres cambiar los pesos para que ese número baje.

Con **un** peso es fácil. Súbelo un poco y mira si la pérdida sube o baja. Si baja, sigue
subiéndolo. A la derivada de la pérdida respecto a un peso se le llama **gradiente**, y es
literalmente eso: cuánto cambia la pérdida si muevo este peso un poquito.

Con 8,9 millones de pesos, la fuerza bruta no vale. Probar peso a peso significa evaluar
la red entera 8,9 millones de veces por cada paso de entrenamiento. Un solo paso tardaría
días.

## Tres formas de derivar, y por qué solo sirve una

**Numérica.** Lo que acabo de describir: mover el peso un poco y ver qué pasa. Además de
carísima, es imprecisa: si mueves poco, la diferencia se pierde en el redondeo de la coma
flotante; si mueves mucho, la aproximación es mala. Sirve para *comprobar* que tus
gradientes están bien, no para entrenar.

**Simbólica.** Sacar la fórmula cerrada de la derivada, como en clase de cálculo. En cuanto
compones unas cuantas funciones, la expresión crece hasta ser inmanejable.

**Automática en modo inverso.** Lo que vas a implementar. Descompone el cálculo en
operaciones elementales (sumas, multiplicaciones) y aplica la regla de la cadena hacia
atrás. Coste: **un forward más un backward que cuesta el doble, y ya está** — da igual que
haya 10 parámetros o 10.000 millones. Ese resultado es lo que hace posible el deep learning.

## La regla de la cadena, con números

Toda la maquinaria descansa en algo que ya sabes del instituto. Si $y$ depende de $u$ y
$u$ depende de $x$:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

Vamos a verlo funcionando. Sea $a = 2$, $b = -3$, y:

```
c = a · b        ->  c = -6
d = c + 10       ->  d = 4
```

Queremos $\partial d/\partial a$: si muevo $a$, ¿cuánto se mueve $d$?

Empezamos por el final y vamos hacia atrás:

```
∂d/∂d = 1                      (algo respecto a sí mismo)
∂d/∂c = 1                      (d = c + 10, sumar una constante no cambia la pendiente)
∂c/∂a = b = -3                 (c = a·b, la derivada respecto a 'a' es 'b')

∂d/∂a = ∂d/∂c · ∂c/∂a = 1 · (-3) = -3
```

Si subo $a$ en 0,1, $d$ baja 0,3. Compruébalo: con $a = 2{,}1$ sale $c = -6{,}3$ y
$d = 3{,}7$. Exacto.

**Eso es todo el algoritmo.** Ir de atrás hacia delante multiplicando derivadas locales.
Lo único que falta es organizarlo para que funcione con miles de operaciones.

## Por qué hacia atrás y no hacia delante

Podrías recorrer la cadena en el otro sentido y también saldría. Pero hay una asimetría
que lo cambia todo: **muchas entradas, una sola salida**. Tienes millones de parámetros y
una única pérdida.

Yendo hacia atrás, empiezas en la pérdida (un número) y vas repartiendo. Cada valor
intermedio que llevas es un solo número por nodo. Yendo hacia delante tendrías que arrastrar
un valor por cada parámetro respecto al que derivas, o sea millones de cálculos en paralelo.

Regla general: modo inverso gana cuando hay muchas entradas y pocas salidas. Que es
exactamente el caso de cualquier función de pérdida.

## El grafo: cada operación se acuerda de sí misma

La estructura es esta: cada vez que haces una operación, en lugar de devolver solo el
resultado devuelves un objeto que además guarda **de dónde salió**.

Para `c = a * b`, el objeto `c` guarda:

- su valor (`-6`),
- quiénes son sus hijos (`a` y `b`),
- y una función que sabe repartir hacia atrás el gradiente que le llegue.

Esa última parte, para una multiplicación, es:

$$\frac{\partial L}{\partial a} \mathrel{+}= b \cdot \frac{\partial L}{\partial c},
\qquad
\frac{\partial L}{\partial b} \mathrel{+}= a \cdot \frac{\partial L}{\partial c}$$

Encadenando operaciones se construye un grafo, y el backward lo recorre entero.

## El detalle que rompe todo si lo haces mal: el `+=`

Fíjate en que arriba pone `+=` y no `=`. No es un capricho.

Prueba el caso más simple posible: $y = x + x$, con $x = 3$.

- Con `=`: la primera rama pone `x.grad = 1`, la segunda lo **pisa** y vuelve a poner
  `x.grad = 1`. Resultado: 1. **Mal.**
- Con `+=`: `x.grad = 1 + 1 = 2`. Y es correcto, porque $y = 2x$ y su derivada es 2.

La razón matemática es la regla de la cadena multivariable: si una variable influye en el
resultado por varios caminos, su derivada total es la **suma** de lo que aporta cada camino.

$$\frac{\partial L}{\partial x} = \sum_{k} \frac{\partial L}{\partial u_k} \frac{\partial u_k}{\partial x}$$

En una red esto pasa constantemente: cada conexión residual, cada peso compartido, cada
embedding que aparece varias veces en una frase.

**Consecuencia práctica que te va a morder:** como los gradientes se suman, hay que
**ponerlos a cero antes de cada paso**. Ese `optimizer.zero_grad()` que verás en todos los
bucles de entrenamiento es exactamente esto. Si se te olvida, no salta ningún error: cada
paso usa la suma de todos los gradientes desde el principio, y el modelo simplemente no
aprende. Es uno de los bugs más frustrantes que existen porque no da ninguna señal.

## El orden importa

Cuando un nodo reparte su gradiente hacia abajo, tiene que haber recibido ya **todo** lo
que le corresponde de arriba. Si lo reparte antes de tiempo, envía un gradiente incompleto
y todo lo que hay debajo sale mal.

Eso obliga a recorrer los nodos en un orden concreto: **cada nodo solo se procesa cuando
todos los que dependen de él ya están procesados**. A ese orden se le llama *orden
topológico*, y se calcula con un recorrido en profundidad.

Con un grafo simple, en forma de árbol, cualquier orden razonable funciona y el fallo pasa
desapercibido. En cuanto hay un nodo reutilizado, el orden mal calculado produce gradientes
incorrectos **sin ningún síntoma visible**: la pérdida baja algo, el modelo aprende peor, y
no hay nada que apunte al culpable.

Un apunte de implementación: hazlo **iterativo, con una pila**, no recursivo. El grafo de
un MLP de unos cientos de neuronas ya supera el límite de recursión de Python.

## Qué hace PyTorch distinto

Conceptualmente, nada. Un `torch.Tensor` con `requires_grad=True` construye el mismo grafo,
guarda las mismas funciones de backward (ahí se llaman `grad_fn`) y hace el mismo recorrido
topológico. Las diferencias son de ingeniería:

- Opera sobre **tensores** en vez de números sueltos: una operación es un kernel de GPU en
  lugar de un millón de objetos de Python.
- El grafo vive en C++.
- Libera los resultados intermedios según los recorre, para ahorrar memoria. Por eso llamar
  a `backward()` dos veces falla salvo que pidas `retain_graph=True`.

Al terminar este módulo, `loss.backward()` es código que entiendes línea por línea. Esa es
la única razón por la que existe el módulo.

## Dónde está el debate

Aquí, poco. La autodiferenciación en modo inverso es matemática asentada desde los años 70,
y su redescubrimiento como "backpropagation" (Rumelhart, Hinton y Williams, 1986) es un
caso clásico de reinvención en la literatura científica. Lo que sí sigue abierto es si el
cerebro hace algo parecido: el *weight transport problem* —que el backward necesita los
mismos pesos que el forward, transpuestos— no tiene ningún análogo biológico conocido, y
hay una línea de investigación entera buscando alternativas plausibles.

---

**Para ampliar:** Karpathy, [micrograd](https://github.com/karpathy/micrograd) (150 líneas,
merece la pena leerlo entero después de hacer el ejercicio) · Baydin et al. 2018,
[Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767).
Términos sueltos, en [GLOSSARY.md](../../GLOSSARY.md).
