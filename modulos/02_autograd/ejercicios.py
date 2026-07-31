"""Modulo 02 - Autodiferenciacion desde cero.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa -> `llmfs check 02` -> `llmfs hint 02 -e N` si te atascas
-> `SOLUCION.md` tiene el codigo completo si lo necesitas.

Este es el modulo mas denso del curso. Si algo no encaja, no sigas adelante: todo lo demas
se apoya en esto.

QUÉ VAS A CONSTRUIR
===================

El motor que hace que `loss.backward()` funcione. En unas 100 lineas y sin usar PyTorch.

    Value               (ejercicio 1) un numero que recuerda de donde salio
    topological_order   (ejercicio 2) en que orden recorrer el grafo hacia atras
    train_scalar_mlp    (ejercicio 3) entrenar una red usando solo tu motor

Al terminar, `loss.backward()` sera codigo que entiendes linea por linea.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **gradiente**: la derivada de la perdida respecto a un parametro. Dice hacia donde mover
  ese parametro para que el modelo se equivoque menos.
- **backward / backpropagation**: el algoritmo que calcula TODOS los gradientes de golpe,
  recorriendo la red hacia atras.
- **grafo de computo**: el registro de que operaciones se hicieron y en que orden. Es lo
  que permite recorrerlo hacia atras.
- **regla de la cadena**: si y depende de u y u depende de x, entonces
  dy/dx = (dy/du) x (du/dx). Toda la maquinaria descansa en esto.

    llmfs demo 02     dibuja el grafo por dentro y compara con torch.autograd
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Sequence

# El MLP ya esta escrito: no es el objetivo del modulo. Lo importante es que acepta
# `value_cls`, asi que lo puedes montar con TU clase Value.
from llmfs.reference import MLP


class Value:
    """Un escalar que recuerda de donde salio. Es el ejercicio mas largo del curso.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Nueve metodos, pero SIETE de ellos siguen el mismo molde. Aprende el molde y el resto
    es copiar y cambiar dos lineas.

    **Paso 1 - `__init__`.** Guarda cuatro cosas:

        self.data = float(data)      # el valor hacia delante
        self.grad = 0.0              # el gradiente, EMPIEZA EN CERO
        self._prev = tuple(_children)  # de quien depende este nodo
        self._op = _op               # etiqueta, solo para depurar
        self._backward = lambda: None  # de momento, no hace nada

    **Paso 2 - el molde de las operaciones.** Todas se escriben asi:

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)   # (a)
            out = Value(self.data * other.data, (self, other), '*')        # (b)

            def _backward():                                               # (c)
                self.grad  += other.data * out.grad
                other.grad += self.data  * out.grad

            out._backward = _backward                                      # (d)
            return out

        (a) envuelve el numero suelto, para que `a * 3` funcione
        (b) crea el nodo resultado, diciendole quienes son sus hijos
        (c) la closure: NO se ejecuta ahora, se guarda para el backward
        (d) se cuelga del nodo

    **Paso 3 - las derivadas locales.** Solo cambia el paso (c) en cada operacion:

        a + b     ->  self.grad += out.grad             ;  other.grad += out.grad
        a * b     ->  self.grad += other.data*out.grad  ;  other.grad += self.data*out.grad
        a ** n    ->  self.grad += n * self.data**(n-1) * out.grad
        exp(a)    ->  self.grad += out.data * out.grad        (e^a es su propia derivada)
        log(a)    ->  self.grad += (1/self.data) * out.grad
        tanh(a)   ->  self.grad += (1 - out.data**2) * out.grad
        relu(a)   ->  self.grad += (out.data > 0) * out.grad

    **Paso 4 - el azucar.** No necesita derivadas nuevas, se apoya en lo anterior:

        -a          ->  return self * -1
        a - b       ->  return self + (-otro)
        a / b       ->  return self * otro ** -1

    **Paso 5 - `backward()`.** Tres lineas:

        self.grad = 1.0
        for node in reversed(topological_order(self)):
            node._backward()

    LO QUE MÁS CUESTA VER
    ---------------------
    La closure del paso (c) NO se ejecuta cuando la escribes: se guarda. Cuando el backward
    la ejecute, `out.grad` ya tendra el valor que le hayan puesto sus padres, porque la
    closure captura `out` POR REFERENCIA.

    Estas construyendo una lista de tareas pendientes durante el forward, y el backward las
    ejecuta en orden inverso.

    EL ERROR QUE HAY QUE EVITAR: `+=` Y NUNCA `=`
    ---------------------------------------------
    Pruebalo con `y = x + x`:

        con `=`  : la primera rama pone x.grad = 1, la segunda lo PISA -> 1.   MAL
        con `+=` : x.grad = 1 + 1 = 2.                                        BIEN

    Es correcto porque y = 2x y su derivada es 2. Si una variable influye por varios caminos,
    su derivada total es la SUMA de lo que aporta cada camino, y en una red eso pasa
    constantemente.

    Y de ahi sale el `optimizer.zero_grad()` que veras en el modulo 11: como los gradientes se
    acumulan, hay que ponerlos a cero antes de cada paso.

    DOS AVISOS
    ----------
    - El `self.grad = 1.0` de `backward()` es la semilla: la derivada de algo respecto a si
      mismo. Sin ella todos los gradientes salen 0 y no pasa nada.
    - No uses `__slots__` si no sabes exactamente lo que hace.
    - `_prev` puede ser una tupla o un set de los hijos.

    Los metodos `__radd__`, `__rmul__`, `__rsub__` y `__rtruediv__` ya estan escritos abajo:
    son los que hacen que `2 * a` funcione (python prueba `(2).__mul__(a)`, falla, y llama a
    `a.__rmul__(2)`).
    """

    def __init__(
        self,
        data: float,
        _children: Iterable["Value"] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        raise NotImplementedError("TODO: modulo 02, ejercicio 1 - Value.__init__")

    def __add__(self, other: Any) -> "Value":
        raise NotImplementedError("TODO: modulo 02, ejercicio 1 - Value.__add__")

    def __mul__(self, other: Any) -> "Value":
        raise NotImplementedError("TODO: modulo 02, ejercicio 1 - Value.__mul__")

    def __pow__(self, exponent: float) -> "Value":
        raise NotImplementedError("TODO: modulo 02, ejercicio 1 - Value.__pow__")

    def exp(self) -> "Value":
        raise NotImplementedError("TODO: modulo 02, ejercicio 1 - Value.exp")

    def log(self) -> "Value":
        raise NotImplementedError("TODO: modulo 02, ejercicio 1 - Value.log")

    def tanh(self) -> "Value":
        raise NotImplementedError("TODO: modulo 02, ejercicio 1 - Value.tanh")

    def relu(self) -> "Value":
        raise NotImplementedError("TODO: modulo 02, ejercicio 1 - Value.relu")

    def backward(self) -> None:
        """Propaga el gradiente desde este nodo hasta las hojas.

        Dos pasos:
          1. `self.grad = 1.0`  (la derivada de algo respecto a si mismo es 1)
          2. Recorrer `topological_order(self)` AL REVES llamando a `node._backward()`.
        """
        raise NotImplementedError("TODO: modulo 02, ejercicio 1 - Value.backward")

    # -------------------------------------------------- azucar (implementalo tambien)

    def __neg__(self) -> "Value":
        raise NotImplementedError("TODO: modulo 02, ejercicio 1 - Value.__neg__")

    def __sub__(self, other: Any) -> "Value":
        raise NotImplementedError("TODO: modulo 02, ejercicio 1 - Value.__sub__")

    def __truediv__(self, other: Any) -> "Value":
        raise NotImplementedError("TODO: modulo 02, ejercicio 1 - Value.__truediv__")

    def __radd__(self, other: Any) -> "Value":
        return self + other

    def __rmul__(self, other: Any) -> "Value":
        return self * other

    def __rsub__(self, other: Any) -> "Value":
        return (other if isinstance(other, Value) else Value(other)) + (-self)

    def __rtruediv__(self, other: Any) -> "Value":
        return (other if isinstance(other, Value) else Value(other)) * self**-1

    def __repr__(self) -> str:
        return f"Value({getattr(self, 'data', '?')}, grad={getattr(self, 'grad', '?')})"


def topological_order(root: "Value") -> list["Value"]:
    """Ordena los nodos del grafo para poder recorrerlo hacia atras.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Un recorrido en profundidad con una PILA EXPLICITA (no recursivo). El truco es una
    bandera que dice si ya expandiste los hijos de ese nodo.

        order, visited = [], set()
        stack = [(root, False)]          # (nodo, ¿ya expandi sus hijos?)

        while stack:
            node, expanded = stack.pop()

            if expanded:                  # segunda visita: sus hijos ya estan dentro
                order.append(node)
                continue

            if id(node) in visited:
                continue
            visited.add(id(node))

            stack.append((node, True))    # me reencolo para DESPUES de mis hijos
            for child in node._prev:
                if id(child) not in visited:
                    stack.append((child, False))

        return order

    CÓMO FUNCIONA EL TRUCO
    ----------------------
    Cada nodo entra DOS veces en la pila:
      - la primera con `expanded=False`, para meter a sus hijos
      - la segunda con `expanded=True`, y esa se procesa DESPUES de todos ellos, porque los
        hijos se apilaron encima

    Eso es exactamente lo que produce el orden post-orden sin usar recursion.

    QUÉ TIENE QUE CUMPLIR EL RESULTADO
    ----------------------------------
    **Cada nodo aparece DESPUES de todos sus hijos.** O sea, `root` queda el ULTIMO de la
    lista, y `backward()` la recorre al reves.

    POR QUÉ IMPORTA
    ---------------
    `node._backward()` reparte el gradiente de ese nodo a sus hijos. Si lo llamas antes de que
    todos los PADRES de ese nodo hayan aportado su parte, estaras repartiendo un gradiente
    incompleto.

    Con un grafo en forma de arbol, cualquier orden razonable funciona y el bug pasa
    desapercibido. En cuanto hay un nodo reutilizado —y en una red los hay a miles— un orden
    mal calculado da gradientes silenciosamente incorrectos.

    DOS AVISOS
    ----------
    **Iterativo, no recursivo.** El grafo de un MLP de unos cientos de neuronas ya supera el
    limite de recursion de python. Hay un test con 3000 nodos que lo comprueba.

    **Usa `id(nodo)` en el set de visitados, no el nodo.** Si sobrecargas operadores en una
    clase, apoyarte en su hash por defecto es pedir problemas. `id()` es la identidad del
    objeto, que es justo lo que quieres.

    Si te sale `root` el PRIMERO, tienes el orden invertido: el sintoma sera que los gradientes
    salen bien en grafos simples y mal en cuanto haya un nodo reutilizado.

    Args:
        root: el nodo raiz, tipicamente la perdida.

    Returns:
        Lista de todos los nodos alcanzables desde `root`, en orden topologico.
    """
    raise NotImplementedError("TODO: modulo 02, ejercicio 2 - topological_order")


def train_scalar_mlp(
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    hidden: Sequence[int] = (8, 8),
    steps: int = 100,
    lr: float = 0.05,
    seed: int = 0,
    value_cls: type = Value,
) -> list[float]:
    """Entrena un MLP con TU motor de autodiff.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    El bucle de entrenamiento en su forma mas desnuda. Seis pasos, y el orden importa.

        1. Construye el modelo (la clase MLP ya esta hecha, se importa arriba):

               model = MLP(len(xs[0]), [*hidden, 1], value_cls=value_cls, seed=seed)
               history = []

        2. Repite `steps` veces:

             a. FORWARD - predice para cada entrada:

                    preds = [model(x) for x in xs]

             b. PERDIDA - el error cuadratico medio:

                    loss = sum(((p - y)**2 for p, y in zip(preds, ys)),
                               value_cls(0.0)) * (1.0 / len(ys))

             c. LIMPIAR LOS GRADIENTES, **antes** del backward:

                    model.zero_grad()

             d. BACKWARD:

                    loss.backward()

             e. MOVER LOS PESOS, en contra del gradiente:

                    for p in model.parameters():
                        p.data -= lr * p.grad

             f. REGISTRAR:

                    history.append(loss.data)

        3. Devuelve `history`.

    EL PASO 2c ES EL QUE SE OLVIDA TODO EL MUNDO
    --------------------------------------------
    Los gradientes se ACUMULAN (lo hiciste asi en el ejercicio 1). Sin ponerlos a cero, el
    paso 50 usa la suma de los gradientes de los pasos 1 a 50.

    Y NO da ningun error: la perdida baja un poco al principio y luego se estanca o explota.
    Es probablemente el bug mas frustrante de todo el deep learning, porque no hay ninguna
    senyal que apunte a la causa.

    Ponerlo DESPUES del backward tambien funciona, pero solo por casualidad (los deja limpios
    para el siguiente paso). Es fragil: en cuanto el bucle tenga un `continue` o una rama,
    deja de valer.

    DOS DETALLES DE PYTHON
    ----------------------
    **El `value_cls(0.0)` del paso 2b.** El `sum()` de python empieza a acumular desde el
    entero `0`. Funcionaria por `__radd__`, pero darle el valor inicial correcto es mas limpio
    y mas explicito.

    **`p.data -= ...` y no `p -= ...`.** Estas modificando el numero de dentro del nodo, no
    creando un nodo nuevo. Si crearas nodos nuevos, el grafo del paso siguiente colgaria del
    anterior y creceria sin parar hasta agotar la memoria. En PyTorch, el equivalente de esta
    distincion es el bloque `with torch.no_grad():` alrededor del paso del optimizador.

    Args:
        xs: lista de vectores de entrada, todos de la misma longitud.
        ys: los objetivos escalares, uno por entrada.
        hidden: los tamanyos de las capas ocultas. La de salida (1 neurona) se anyade sola.
        steps: cuantos pasos de descenso de gradiente.
        lr: la tasa de aprendizaje.
        seed: la semilla de la inicializacion, para reproducibilidad.
        value_cls: la clase escalar a usar. Por defecto la tuya.

    Returns:
        Lista de `steps` perdidas (floats), una por paso.
    """
    raise NotImplementedError("TODO: modulo 02, ejercicio 3 - train_scalar_mlp")
