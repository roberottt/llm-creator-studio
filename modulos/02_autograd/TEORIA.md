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
- Qué es una neurona, qué es un MLP, y qué es *entrenar*, en su versión más desnuda

### Qué vas a escribir

Tres ejercicios, y esta teoría está ordenada para que los leas en este orden:

| Ejercicio | Qué es | Dónde se explica |
|---|---|---|
| 1. `Value` | Un número que recuerda de dónde salió | [§ De la regla de la cadena a una clase](#de-la-regla-de-la-cadena-a-una-clase-de-python) |
| 2. `topological_order` | En qué orden recorrer el grafo hacia atrás | [§ El orden importa](#el-orden-importa-y-por-eso-existe-el-ejercicio-2) |
| 3. `train_scalar_mlp` | Entrenar una red con tu motor | [§ Qué es una neurona](#qué-es-una-neurona-y-qué-es-un-mlp) y [§ El bucle](#el-bucle-de-entrenamiento-paso-a-paso) |

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
Lo único que falta es organizarlo para que funcione con miles de operaciones. Ese
"organizarlo" es la clase `Value`, y es lo que viene ahora.

## Por qué hacia atrás y no hacia delante

Podrías recorrer la cadena en el otro sentido y también saldría. Pero hay una asimetría
que lo cambia todo: **muchas entradas, una sola salida**. Tienes millones de parámetros y
una única pérdida.

Yendo hacia atrás, empiezas en la pérdida (un número) y vas repartiendo. Cada valor
intermedio que llevas es un solo número por nodo. Yendo hacia delante tendrías que arrastrar
un valor por cada parámetro respecto al que derivas, o sea millones de cálculos en paralelo.

Regla general: modo inverso gana cuando hay muchas entradas y pocas salidas. Que es
exactamente el caso de cualquier función de pérdida.

---

## De la regla de la cadena a una clase de Python

Aquí es donde aparece `Value`, y conviene decir de dónde sale: **no es un concepto de
machine learning, es una decisión de ingeniería**. Tenemos un algoritmo (ir hacia atrás
multiplicando derivadas locales) y necesitamos una estructura de datos que lo soporte. El
nombre viene de [micrograd](https://github.com/karpathy/micrograd), de Karpathy, que es de
donde toma prestada la idea este módulo.

### Por qué un número normal no basta

Escribe esto en Python:

```python
a = 2.0
b = -3.0
c = a * b        # -6.0
```

Después de esa línea, `c` es `-6.0` y nada más. Python ha tirado a la basura las dos cosas
que necesitas para el backward:

- que `c` salió de `a` y `b` (no de otros dos números cualesquiera), y
- que salió por una **multiplicación** (no por una suma, que tendría otra derivada).

El backward no puede reconstruir eso. Así que la idea es: **en vez de devolver un número,
devuelve un objeto que además guarde su origen**. Ese objeto es `Value`.

### Los cinco campos, y por qué está cada uno

```python
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)          # el número de siempre
        self.grad = 0.0                  # ∂L/∂(este nodo). Empieza a cero.
        self._prev = tuple(_children)    # de quién salió
        self._op = _op                   # por qué operación salió
        self._backward = lambda: None    # cómo repartir el gradiente hacia abajo
```

Uno a uno, contra el ejemplo de arriba (`c = a * b`):

- **`data`** es `-6.0`. Lo que ya tenías.
- **`grad`** es donde se irá acumulando $\partial L/\partial c$. Durante el forward vale 0;
  el backward lo va rellenando de arriba abajo. Ojo con la asimetría: `data` se llena
  yendo hacia delante, `grad` yendo hacia atrás.
- **`_prev`** es `(a, b)`. Las **aristas del grafo**, guardadas hacia atrás. Es lo que
  permite que `backward()` sepa a quién repartir.
- **`_op`** es `'*'`. Solo sirve para depurar y para dibujar el grafo bonito en la demo. Si
  lo quitaras, todo seguiría funcionando.
- **`_backward`** es la pieza rara, y tiene su propia sección más abajo.

El guion bajo de `_prev`, `_op` y `_backward` es la convención de Python de "esto es
interna de la clase". Quien usa `Value` toca `data` y `grad`, nada más.

### Por qué se sobrecargan los operadores

Podrías escribir el grafo a mano: `c = multiplicar(a, b)`. Funcionaría, pero al llegar al
ejercicio 3 tendrías que escribir el MLP entero en esa notación y sería ilegible.

Definiendo `__mul__`, Python traduce `a * b` a `a.__mul__(b)` automáticamente. Es decir:
**escribes matemáticas normales y el grafo se construye solo, como efecto secundario**. Eso
es exactamente lo que hace PyTorch cuando escribes `x @ w + b` sobre tensores con
`requires_grad=True`. No hay más truco que ese.

Los métodos `__radd__`, `__rmul__` y compañía —que ya vienen escritos en `ejercicios.py`—
cubren el caso `2 * a`: Python prueba primero `(2).__mul__(a)`, el `int` no sabe qué hacer
con un `Value`, devuelve `NotImplemented`, y entonces Python llama a `a.__rmul__(2)`.

### El molde: todas las operaciones se escriben igual

Aprende este molde y siete de los nueve métodos son copiar y cambiar dos líneas:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)   # (a)
    out = Value(self.data * other.data, (self, other), '*')       # (b)

    def _backward():                                              # (c)
        self.grad  += other.data * out.grad
        other.grad += self.data  * out.grad

    out._backward = _backward                                     # (d)
    return out
```

- **(a)** envuelve el número suelto, para que `a * 3` funcione.
- **(b)** crea el nodo resultado y le dice quiénes son sus hijos. El forward, ya está.
- **(c)** define cómo repartir el gradiente. **No se ejecuta ahora.** Solo se define.
- **(d)** se cuelga del nodo, para ejecutarla en el backward.

### La closure: lo que más cuesta ver

`_backward` es una función definida dentro de otra función. Al salir de `__mul__`, esa
función interna sigue viva y **sigue viendo las variables `self`, `other` y `out`**, por
referencia. A eso se le llama *closure*, y es lo que hace que todo esto funcione.

La consecuencia es la clave del módulo: cuando `_backward()` se ejecute —minutos de código
después, ya en el backward— la línea `self.grad += other.data * out.grad` leerá el
`out.grad` **de ese momento**, no el que había cuando se definió (que era 0). Para entonces
los padres de `out` ya habrán aportado su parte.

Otra forma de verlo: durante el forward estás construyendo una **lista de tareas
pendientes**, una por operación. El backward las ejecuta en orden inverso.

### Traza completa, con los números del ejemplo

Vamos a seguir `d = a * b + 10` con $a = 2$, $b = -3$, campo a campo.

**Forward** — se crean cuatro nodos:

| nodo | `data` | `_op` | `_prev` | `grad` |
|---|---|---|---|---|
| `a` | 2.0 | `''` (hoja) | `()` | 0.0 |
| `b` | -3.0 | `''` (hoja) | `()` | 0.0 |
| `c` | -6.0 | `'*'` | `(a, b)` | 0.0 |
| `10` | 10.0 | `''` (hoja) | `()` | 0.0 |
| `d` | 4.0 | `'+'` | `(c, 10)` | 0.0 |

Todos los gradientes valen 0. Todavía no se ha derivado nada: solo se ha construido el
grafo mientras se calculaba el resultado.

**Backward** — `d.backward()` hace tres cosas:

1. `d.grad = 1.0` — la semilla. La derivada de algo respecto a sí mismo. **Sin esta línea
   todo se queda a cero y no pasa absolutamente nada**, porque cada `_backward` multiplica
   por el gradiente del nodo de arriba.
2. Pide el orden topológico: `[10, b, a, c, d]` (ese es el orden real que devuelve la
   implementación; lo importante es que `d` queda el último).
3. Lo recorre **al revés** llamando a `node._backward()`:

```
d._backward()  ->  c.grad  += 1.0        ->  c.grad  = 1.0
                   10.grad += 1.0
c._backward()  ->  a.grad  += b.data * c.grad = -3 * 1 = -3
                   b.grad  += a.data * c.grad =  2 * 1 =  2
a._backward()  ->  no hace nada: es una hoja, su _backward es `lambda: None`
b._backward()  ->  idem
```

Resultado: `a.grad = -3.0`, `b.grad = 2.0`. Es lo que salía a mano en la sección de la
regla de la cadena. Puedes verificarlo tú mismo: `llmfs demo 02` imprime esta tabla al lado
de lo que dice `torch.autograd` para la misma expresión.

### Las derivadas locales: una línea por método

Lo único que cambia entre operaciones es el paso (c). Esta tabla **es** el ejercicio 1:

| operación | qué va en `_backward` |
|---|---|
| `a + b` | `self.grad += out.grad` ; `other.grad += out.grad` |
| `a * b` | `self.grad += other.data * out.grad` ; `other.grad += self.data * out.grad` |
| `a ** n` | `self.grad += n * self.data**(n-1) * out.grad` |
| `exp(a)` | `self.grad += out.data * out.grad` (porque $e^a$ es su propia derivada) |
| `log(a)` | `self.grad += (1/self.data) * out.grad` |
| `tanh(a)` | `self.grad += (1 - out.data**2) * out.grad` |
| `relu(a)` | `self.grad += (out.data > 0) * out.grad` |

Fíjate en el patrón: **derivada local × gradiente que llega de arriba**. Eso es literalmente
la regla de la cadena, escrita una vez por operación. La suma reparte el gradiente
intacto (derivada local 1) y por eso se dice que "la suma es un enrutador de gradientes".

### El azúcar no necesita derivadas nuevas

`__neg__`, `__sub__` y `__truediv__` se apoyan en lo que ya tienes:

```python
-a       ->  self * -1
a - b    ->  self + (-otro)
a / b    ->  self * otro ** -1
```

No hace falta escribir la derivada del cociente: sale sola de componer `*` y `**-1`, y la
regla de la cadena la reconstruye. Esta es la razón de que un motor de autodiff necesite tan
pocas operaciones primitivas.

## El detalle que rompe todo si lo haces mal: el `+=`

Fíjate en que en toda la tabla pone `+=` y no `=`. No es un capricho.

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

---

## El orden importa (y por eso existe el ejercicio 2)

Cuando un nodo reparte su gradiente hacia abajo, tiene que haber recibido ya **todo** lo
que le corresponde de arriba. Si lo reparte antes de tiempo, envía un gradiente incompleto
y todo lo que hay debajo sale mal.

Míralo en el grafo más pequeño donde se nota, un **diamante**: $x$ influye en $L$ por dos
caminos.

```
x = 3
u = x * 2        ->  6        (camino 1)
v = x + 1        ->  4        (camino 2)
L = u * v        ->  24
```

La respuesta correcta: $L = 2x(x+1) = 2x^2 + 2x$, así que $dL/dx = 4x + 2 = 14$.

Y por el grafo: $\partial L/\partial u = v = 4$, $\partial L/\partial v = u = 6$, con lo que
$\partial L/\partial x = 4 \cdot 2 + 6 \cdot 1 = 14$. Coincide.

Ahora, **el orden**. Si procesas los nodos así:

```
L  ->  reparte a u y v          u.grad = 4,  v.grad = 6      bien
u  ->  reparte a x              x.grad += 4*2 = 8
v  ->  reparte a x              x.grad += 6*1 = 6            total 14   ✓
```

Pero si procesas `u` **antes** que `L`, `u.grad` todavía vale 0 y `x` recibe 0 por ese
camino: sale 6 en vez de 14. Silenciosamente mal.

La regla es: **cada nodo solo se procesa cuando todos los que dependen de él ya están
procesados**. A ese orden se le llama *orden topológico*, y se consigue con un recorrido en
profundidad que apunta cada nodo **después** de sus hijos (post-orden). `backward()` recorre
esa lista al revés, y así la raíz —la pérdida— va primera.

Con un grafo simple, en forma de árbol, cualquier orden razonable funciona y el fallo pasa
desapercibido. En cuanto hay un nodo reutilizado, el orden mal calculado produce gradientes
incorrectos **sin ningún síntoma visible**: la pérdida baja algo, el modelo aprende peor, y
no hay nada que apunte al culpable.

Un apunte de implementación: hazlo **iterativo, con una pila**, no recursivo. El grafo de
un MLP de unos cientos de neuronas ya supera el límite de recursión de Python — el del
ejercicio 3, con solo 113 parámetros, tiene **1.068 nodos**. El docstring del ejercicio 2
tiene el esqueleto con la bandera de "hijos ya expandidos", que es el truco estándar para
hacer un post-orden sin recursión.

---

## Qué es una neurona y qué es un MLP

Hasta aquí has construido un motor de derivadas. El ejercicio 3 lo usa para entrenar algo,
y ese algo es un **MLP**. Si nunca has visto uno, esta sección es la que faltaba.

La clase `MLP` **ya está escrita** y se importa en `ejercicios.py`; no es el objetivo del
módulo. Pero no debería ser una caja negra, así que aquí está por dentro. El código vive en
`llmfs/reference/autograd.py` y son 60 líneas: ábrelo mientras lees esto.

### Una neurona

Una neurona hace **dos cosas**, en este orden:

1. una **suma ponderada** de sus entradas, más un término independiente:
   $\text{act} = w_1 x_1 + w_2 x_2 + \dots + b$
2. le pasa el resultado por una función **no lineal**, aquí `tanh`.

Los $w_i$ son los **pesos** y $b$ es el **sesgo** (*bias*). Son los parámetros: los números
que el entrenamiento va a mover. Las $x_i$ son la entrada, y no se tocan.

Con números. Entrada $x = [1{,}0,\ -2{,}0]$, pesos $w = [0{,}5,\ 0{,}25]$, sesgo $b = 0{,}1$:

```
act = 0.5·1.0 + 0.25·(-2.0) + 0.1 = 0.5 - 0.5 + 0.1 = 0.1
o   = tanh(0.1) = 0.099668
```

En código son cuatro líneas, y fíjate en que **está escrito con tus `Value`**: cada `wi * xi`
crea un nodo, cada suma crea otro, y el `tanh` uno más. Al terminar el forward de la neurona
ya tienes su trocito de grafo montado.

```python
class Neuron:
    def __init__(self, nin, nonlin=True, value_cls=Value, rng=None):
        self.w = [value_cls(rng.uniform(-1, 1)) for _ in range(nin)]
        self.b = value_cls(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return [*self.w, self.b]
```

El parámetro `value_cls` es el que hace que puedas montar la red con **tu** clase en vez de
con la de referencia. Por eso `train_scalar_mlp` lo recibe y lo pasa hacia abajo.

**Por qué hace falta el `tanh`.** Si quitas la no linealidad, cada capa es una combinación
lineal de la anterior, y componer funciones lineales da otra función lineal: una red de 50
capas colapsaría a una sola. La no linealidad es lo único que permite que apilar capas
sirva de algo. `tanh` aplasta cualquier número al intervalo $(-1, 1)$: `tanh(0.1) = 0.0997`,
`tanh(3) = 0.995`, `tanh(-3) = -0.995`. En el módulo 08 verás por qué los transformers usan
GELU en vez de `tanh`.

### Una capa y un MLP

Una **capa** son varias neuronas mirando a la misma entrada, cada una con sus propios pesos.
Un **MLP** (*multi-layer perceptron*, perceptrón multicapa) son capas encadenadas: la salida
de una es la entrada de la siguiente.

```python
class MLP:
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

El del ejercicio 3 es `MLP(3, [8, 8, 1])`: entra un vector de 3 números, pasa por dos capas
de 8 neuronas y sale **un** número. Cuenta los parámetros: la primera capa tiene 8 neuronas
× (3 pesos + 1 sesgo) = 32; la segunda 8 × (8 + 1) = 72; la de salida 1 × (8 + 1) = 9. Total
**113 parámetros**, que es lo que devuelve `len(model.parameters())`.

Dos detalles del constructor que conviene notar:

- **La última capa es lineal**, sin `tanh` (`nonlin=i < len(nouts) - 1`). Si la salida tiene
  que poder valer 2,7 o -40, aplastarla a $(-1,1)$ te lo impide. Es la misma razón por la
  que el GPT del módulo 10 termina en una capa lineal que produce logits sin acotar.
- **Los pesos arrancan aleatorios** (`rng.uniform(-1, 1)`) y los sesgos a cero. Si arrancaran
  todos iguales, todas las neuronas de una capa calcularían exactamente lo mismo y
  recibirían el mismo gradiente para siempre: la capa entera se comportaría como una sola
  neurona. Se llama *ruptura de simetría*, y en el módulo 10 volverá con más cuidado.

Los otros dos métodos son los que usa el bucle de entrenamiento:

- `parameters()` aplana toda la red en una lista de 113 `Value`. Es a lo que se le aplica el
  descenso de gradiente.
- `zero_grad()` pone los 113 `.grad` a cero. Ya sabes por qué existe.

Nada de esto es específico de los MLP: `parameters()` y `zero_grad()` se llaman igual en
PyTorch y hacen exactamente lo mismo.

---

## El bucle de entrenamiento, paso a paso

El ejercicio 3 tiene cuatro puntos de entrada y sus cuatro objetivos:

```
xs = [[2,3,-1], [3,-1,0.5], [0.5,1,1], [1,1,-1]]
ys = [   1,        -1,         -1,        1    ]
```

Se le pide a la red que devuelva $+1$ para el primero y el cuarto, y $-1$ para los otros
dos. Al principio no lo hace: con los pesos aleatorios, la predicción para el primer punto
es 0,939 cuando debería ser 1, y las demás están peor.

### Medir el error: el MSE

Necesitamos un número que diga cómo de mal va. El **error cuadrático medio** (MSE) es la
media de los errores al cuadrado:

$$L = \frac{1}{N}\sum_{i=1}^{N} (p_i - y_i)^2$$

Se eleva al cuadrado por dos razones: para que los errores de signo contrario no se
cancelen, y porque el cuadrado es derivable en todas partes (el valor absoluto no lo es en
el cero). Con los pesos iniciales de este ejercicio sale $L = 1{,}4334$.

En código, con tus `Value`:

```python
loss = sum(((p - y)**2 for p, y in zip(preds, ys)), value_cls(0.0)) * (1.0 / len(ys))
```

Ese `value_cls(0.0)` es el valor inicial del `sum`: si no lo pones, Python empieza a
acumular desde el entero `0` y aunque funcionaría por `__radd__`, es más limpio ser
explícito. Y fíjate en lo importante: **`loss` es un `Value`**, la raíz de un grafo de 1.068
nodos que llega hasta cada uno de los 113 parámetros. Por eso puedes llamar a
`loss.backward()`.

### El descenso de gradiente

Después del backward, cada parámetro tiene su `.grad`: cuánto sube la pérdida si subo ese
parámetro. Como quieres que la pérdida **baje**, te mueves en contra:

```python
for p in model.parameters():
    p.data -= lr * p.grad
```

`lr` es el **learning rate** (tasa de aprendizaje): cuánto caso le haces al gradiente. El
gradiente te da la dirección, no la distancia — es una pendiente local, y solo es fiable muy
cerca del punto en el que la calculaste.

Un paso completo, con la neurona de antes ($w = [0{,}5,\ 0{,}25]$, $b = 0{,}1$,
$x = [1,\ -2]$) y objetivo $y = 1$:

```
forward:   o = tanh(0.1) = 0.099668
           L = (o - 1)² = 0.810598
backward:  dL/dw0 = -1.782777
           dL/dw1 = +3.565553
           dL/db  = -1.782777
paso:      w0 = 0.5 - 0.1·(-1.782777) = 0.678278      (lr = 0.1)
```

El gradiente de $w_0$ es negativo, así que $w_0$ **sube**, y con eso `act` sube, `o` se
acerca a 1 y la pérdida baja. Los signos cuadran: $x_1$ vale $-2$, negativo, y por eso su
peso se mueve en sentido contrario.

Esos tres gradientes están comprobados contra `torch.autograd` — coinciden en los seis
decimales.

### Los seis pasos, y por qué en ese orden

```python
for _ in range(steps):
    preds = [model(x) for x in xs]                    # 1. forward
    loss  = mse(preds, ys)                            # 2. pérdida
    model.zero_grad()                                 # 3. limpiar gradientes
    loss.backward()                                   # 4. backward
    for p in model.parameters():                      # 5. mover los pesos
        p.data -= lr * p.grad
    history.append(loss.data)                         # 6. registrar
```

**El paso 3 es el que se olvida todo el mundo**, y va **antes** del backward. Los gradientes
se acumulan (lo escribiste así en el ejercicio 1), así que sin limpiarlos el paso 50 usa la
suma de los gradientes de los pasos 1 a 50. Y no da ningún error: la pérdida baja un poco al
principio y luego se estanca o explota.

Ponerlo *después* del backward también funciona, pero solo por casualidad —deja los
gradientes limpios para la siguiente vuelta—. Es frágil: en cuanto el bucle tenga un
`continue` o una rama, deja de valer.

**`p.data -= ...` y no `p -= ...`.** Estás modificando el número de dentro del nodo, no
creando un nodo nuevo. Si crearas nodos nuevos, el grafo del paso siguiente colgaría del
anterior y crecería sin parar hasta agotar la memoria. En PyTorch el equivalente exacto de
esta distinción es el `with torch.no_grad():` que envuelve el paso del optimizador.

### Qué tiene que salir

Con `steps=100, lr=0.05, seed=0`, la pérdida que devuelve `train_scalar_mlp`:

| paso | 0 | 1 | 5 | 10 | 25 | 50 | 99 |
|---|---|---|---|---|---|---|---|
| pérdida | 1,4334 | 0,9573 | 0,4246 | 0,1760 | 0,0024 | 6,7·10⁻⁷ | ~10⁻¹¹ |

Si tu primer valor es 1,4334 y baja monótonamente, tu motor funciona. Y con `lr=0.005`, diez
veces más pequeño, después de 100 pasos la pérdida sigue en 0,1701: no le ha dado tiempo. Es
el mismo modelo y la misma inicialización; solo cambia el learning rate. Ejecuta
`llmfs demo 02` para ver las dos curvas juntas.

---

## Qué hace PyTorch distinto

Conceptualmente, nada. Un `torch.Tensor` con `requires_grad=True` construye el mismo grafo,
guarda las mismas funciones de backward y hace el mismo recorrido topológico:

| tu código | PyTorch |
|---|---|
| `Value` | `torch.Tensor` con `requires_grad=True` |
| `.data` | `.data` |
| `.grad` | `.grad` |
| `_prev` | las aristas internas del grafo, en C++ |
| `_backward` | `.grad_fn` (`<MulBackward0>`, `<TanhBackward0>`…) |
| `topological_order` | el recorrido del motor de autograd |
| `model.zero_grad()` | `optimizer.zero_grad()` |
| `p.data -= lr * p.grad` | `optimizer.step()` dentro de `torch.no_grad()` |

Las diferencias son de ingeniería:

- Opera sobre **tensores** en vez de números sueltos: una operación es un kernel de GPU en
  lugar de un millón de objetos de Python. Tu motor tarda un segundo largo en 100 pasos de
  una red de 113 parámetros; el módulo 13 entrena 8,9 millones.
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
Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
