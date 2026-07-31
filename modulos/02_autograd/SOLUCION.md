# 02 — Solución comentada

## Ejercicio 1 — La clase `Value`

### El molde

Todas las operaciones tienen la misma forma. Si entiendes una, entiendes las siete:

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)   # 1
    out = Value(self.data * other.data, (self, other), '*')       # 2

    def _backward():                                              # 3
        self.grad  += other.data * out.grad
        other.grad += self.data  * out.grad

    out._backward = _backward                                     # 4
    return out
```

1. **Envolver el número suelto.** Así `a * 3` funciona sin escribir `a * Value(3)`.
2. **Crear el nodo resultado**, diciéndole quiénes son sus hijos. Esa tupla es lo que
   permite recorrer el grafo después.
3. **La closure.** Esto es lo único conceptualmente nuevo. No se ejecuta ahora: se guarda
   para más tarde. Cuando se ejecute, `out.grad` ya tendrá el valor que le hayan puesto sus
   padres, porque la closure captura `out` **por referencia**, no por valor.
4. **Colgarla del nodo**, para que `backward()` pueda llamarla.

Que el paso 3 no se ejecute en el momento es lo que hace que todo esto funcione, y es la
parte que más cuesta ver la primera vez. Estás construyendo una lista de tareas pendientes
mientras haces el forward, y el backward las ejecuta en orden inverso.

### Las derivadas, una a una

| operación | derivada local | por qué |
|---|---|---|
| `a + b` | `1` para ambos | subir `a` en 1 sube la suma en 1 |
| `a * b` | `b` para `a`, `a` para `b` | subir `a` en 1 sube el producto en `b` |
| `a ** n` | `n * a^(n-1)` | la regla de siempre |
| `exp(a)` | `out.data` | la exponencial es su propia derivada, y ya la tienes calculada |
| `log(a)` | `1 / a.data` | |
| `tanh(a)` | `1 - out.data²` | identidad conocida, y reutiliza el valor del forward |
| `relu(a)` | `1` si salió positivo, `0` si no | |

Dos de ellas —`exp` y `tanh`— usan `out.data` en vez de recalcular. No es solo por
velocidad: es el mismo truco que usa PyTorch, guardar en el forward lo que hará falta en el
backward.

Sobre `relu`: en el 0 exacto la derivada no existe (hay un pico). Por convención se toma 0.
Da igual cuál elijas: la probabilidad de caer exactamente en 0 con floats es despreciable.

### El azúcar sintáctico

No hace falta escribir ninguna derivada nueva. Todo se apoya en las anteriores:

```python
-a       ->  a * -1
a - b    ->  a + (-b)
a / b    ->  a * b**-1
```

Y las versiones `__r*__` son para cuando el `Value` está a la derecha. Cuando Python
evalúa `2 * a`, primero prueba `(2).__mul__(a)`, que devuelve `NotImplemented` porque `int`
no sabe qué hacer con un `Value`. Entonces prueba `a.__rmul__(2)`. Por eso `__radd__` y
`__rmul__` pueden delegar directamente (la suma y el producto son conmutativos) pero
`__rsub__` y `__rtruediv__` no: `2 - a` no es `a - 2`.

### `backward()`

```python
def backward(self):
    self.grad = 1.0
    for node in reversed(topological_order(self)):
        node._backward()
```

Tres líneas. El `self.grad = 1.0` es la semilla: la derivada de la pérdida respecto a sí
misma. Sin ella todos los gradientes serían 0 y no pasaría nada.

## Ejercicio 2 — `topological_order`

DFS post-orden. La versión recursiva es de cinco líneas y **no sirve**: con unos cientos de
neuronas revienta con `RecursionError`. Iterativa:

```python
order, visited = [], set()
stack = [(root, False)]          # (nodo, ¿ya expandí sus hijos?)

while stack:
    node, expanded = stack.pop()
    if expanded:                  # segunda visita: ya están todos sus hijos
        order.append(node)
        continue
    if id(node) in visited:
        continue
    visited.add(id(node))
    stack.append((node, True))    # me reencolo para después de mis hijos
    for child in node._prev:
        if id(child) not in visited:
            stack.append((child, False))
```

El truco es la bandera. Cada nodo entra dos veces en la pila: la primera para expandir sus
hijos, la segunda —que se procesa **después** de todos ellos— para añadirse al resultado.
Eso es exactamente lo que produce el post-orden sin recursión.

**Usa `id(node)` y no `node` en el conjunto de visitados.** Si sobrecargas operadores en la
clase, apoyarse en el hash o la igualdad por defecto es pedir problemas. `id()` es la
identidad del objeto y es justo lo que quieres aquí.

**El resultado tiene a `root` al final**, y `backward()` lo recorre al revés. Si te sale
al principio, tienes el orden invertido: el síntoma será que los gradientes salen mal en
cuanto haya un nodo reutilizado, pero *bien* en grafos simples. Que un test pase con un
árbol y falle con un rombo casi siempre es esto.

## Ejercicio 3 — `train_scalar_mlp`

```python
model = MLP(len(xs[0]), [*hidden, 1], value_cls=value_cls, seed=seed)
history = []

for _ in range(steps):
    preds = [model(x) for x in xs]
    loss = sum(((p - y)**2 for p, y in zip(preds, ys)), value_cls(0.0)) * (1.0/len(ys))

    model.zero_grad()      # ANTES del backward
    loss.backward()

    for p in model.parameters():
        p.data -= lr * p.grad

    history.append(loss.data)
```

**El `value_cls(0.0)` en el `sum()`.** El `sum()` de Python empieza a acumular desde el
entero `0`. La primera suma sería `0 + Value(...)`, que funciona por `__radd__`, pero es
más limpio y más explícito darle el valor inicial correcto.

**El orden de `zero_grad()` y `backward()`.** Los gradientes se acumulan (ejercicio 1). Si
no los limpias antes de cada backward, el paso 50 usa la suma de los gradientes de los
pasos 1 a 50. El síntoma es que la pérdida baja un poco al principio y luego se estanca o
explota, y **no hay ningún mensaje de error**. Es probablemente el bug más frecuente de
todo el deep learning.

Ponerlo *después* del `backward()` en lugar de antes también funciona, pero solo por
casualidad (los deja limpios para el siguiente paso). Es frágil: en cuanto el bucle tenga
un `continue` o una rama, deja de valer. Ponlo antes.

**`p.data -= lr * p.grad`, no `p -= lr * p.grad`.** Estás modificando el número que hay
dentro del nodo, no creando un nodo nuevo. Si crearas nodos nuevos, el grafo del siguiente
paso colgaría del anterior y crecería sin parar hasta agotar la memoria. En PyTorch, el
equivalente de esta distinción es el bloque `with torch.no_grad():` alrededor del paso del
optimizador.

## Lo que deberías ver en la demo

Los gradientes de tu motor idénticos a los de `torch.autograd` hasta el último decimal. No
parecidos: **idénticos**, porque estás haciendo literalmente las mismas operaciones en el
mismo orden.

Y las dos curvas de pérdida con `lr` distinto. La lección de esa gráfica es la que se
repetirá en el módulo 11: el learning rate es el hiperparámetro que más entrenamientos
arruina, y no hay forma de saber el bueno sin probar.

---

## El código completo

Si te has atascado, aquí está la implementación entera. **Cópiala, pégala y ejecuta los
tests**: verlos pasar con código que entiendes es mejor que quedarte bloqueado.

Y después vuelve al ejercicio y escríbela tú. Leer una solución que ya has peleado funciona
muy bien; leerla en frío, no funciona nada.

```python
class Value:

    __slots__ = ("data", "grad", "_prev", "_op", "_backward", "label")

    def __init__(
        self,
        data: float,
        _children: Iterable["Value"] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self.label = label
        self._backward: Callable[[], None] = lambda: None

    # ------------------------------------------------------------------ operaciones

    def __add__(self, other: Any) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Any) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> "Value":
        if not isinstance(exponent, (int, float)):
            raise TypeError("solo se admiten exponentes int o float constantes")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            # d(a^n)/da = n * a^(n-1)
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            # d(e^a)/da = e^a, que es justamente out.data
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            # d(tanh)/dx = 1 - tanh^2
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward() -> None:
            # La derivada en 0 no existe; por convencion se toma 0.
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Value":
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ azucar

    def __neg__(self) -> "Value":
        return self * -1

    def __radd__(self, other: Any) -> "Value":
        return self + other

    def __sub__(self, other: Any) -> "Value":
        return self + (-(other if isinstance(other, Value) else Value(other)))

    def __rsub__(self, other: Any) -> "Value":
        return (other if isinstance(other, Value) else Value(other)) + (-self)

    def __rmul__(self, other: Any) -> "Value":
        return self * other

    def __truediv__(self, other: Any) -> "Value":
        return self * (other if isinstance(other, Value) else Value(other)) ** -1

    def __rtruediv__(self, other: Any) -> "Value":
        return (other if isinstance(other, Value) else Value(other)) * self**-1

    def __repr__(self) -> str:
        etiqueta = f" {self.label}" if self.label else ""
        return f"Value({self.data:.6g}, grad={self.grad:.6g}{etiqueta})"

    # ------------------------------------------------------------------ backward

    def backward(self) -> None:
        self.grad = 1.0
        for node in reversed(topological_order(self)):
            node._backward()


def topological_order(root: Value) -> list[Value]:
    order: list[Value] = []
    visited: set[int] = set()
    # (nodo, hijos_ya_expandidos)
    stack: list[tuple[Value, bool]] = [(root, False)]

    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))
        for child in node._prev:
            if id(child) not in visited:
                stack.append((child, False))

    return order


def train_scalar_mlp(
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    hidden: Sequence[int] = (8, 8),
    steps: int = 100,
    lr: float = 0.05,
    seed: int = 0,
    value_cls: type = Value,
) -> list[float]:
    model = MLP(len(xs[0]), [*hidden, 1], value_cls=value_cls, seed=seed)
    history: list[float] = []

    for _ in range(steps):
        # forward: predicciones y perdida
        preds = [model(x) for x in xs]
        loss = sum(((p - y) ** 2 for p, y in zip(preds, ys)), value_cls(0.0)) * (1.0 / len(ys))

        # backward: SIEMPRE poner los gradientes a cero antes, porque se acumulan.
        # Olvidarse de esto es el bug numero uno de quien empieza.
        model.zero_grad()
        loss.backward()

        # descenso: un paso en contra del gradiente
        for p in model.parameters():
            p.data -= lr * p.grad

        history.append(loss.data)

    return history
```

Los imports que hacen falta ya están en el `ejercicios.py` del módulo, salvo los que
aparezcan arriba del bloque.
