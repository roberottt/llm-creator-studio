"""Modulo 02 - Autodiferenciacion desde cero.

Vas a construir un motor de autodiff escalar. Al terminar, `loss.backward()` de PyTorch
sera codigo que entiendes, no una caja negra.

    llmfs check 02
    llmfs demo 02     # dibuja el grafo de computo y compara gradientes con torch
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Sequence

# El MLP ya esta escrito: no es el objetivo del modulo. Lo importante es que acepta
# `value_cls`, asi que lo puedes montar con TU clase Value.
from llmfs.reference import MLP


class Value:
    """Un escalar que recuerda de donde ha salido.

    Cada operacion sobre `Value` devuelve un `Value` nuevo que guarda:
      - `data`: el resultado hacia delante.
      - `_prev`: los nodos de los que depende.
      - `_op`: etiqueta de la operacion (solo para depurar y dibujar).
      - `_backward`: una closure que, cuando se la llama, reparte `self.grad` entre
        `_prev` segun la derivada local de la operacion.

    Y ademas:
      - `grad`: derivada de la salida final respecto a este nodo. Empieza en 0.0 y se
        ACUMULA (`+=`), nunca se asigna (`=`). Ver TEORIA.md.

    Lo que tienes que implementar:

        __init__      guardar data, grad=0.0, _prev, _op y un _backward que no haga nada
        __add__       out = a + b
        __mul__       out = a * b
        __pow__       out = a ** n   (n constante int/float, NO otro Value)
        exp           out = e^a
        log           out = ln(a)
        tanh          out = tanh(a)
        relu          out = max(0, a)
        backward      recorrer el grafo y propagar

    Y el azucar sintactico, que se resuelve todo en terminos de los cinco anteriores:

        __neg__       -a          =  a * -1
        __sub__       a - b       =  a + (-b)
        __truediv__   a / b       =  a * b**-1
        __radd__, __rmul__, __rsub__, __rtruediv__   para que `2 * a` funcione

    DERIVADAS LOCALES (esto es todo el contenido matematico del modulo):

        out = a + b        ->  da += 1 * out.grad          db += 1 * out.grad
        out = a * b        ->  da += b.data * out.grad     db += a.data * out.grad
        out = a ** n       ->  da += n * a.data**(n-1) * out.grad
        out = exp(a)       ->  da += out.data * out.grad          (porque d(e^a)/da = e^a)
        out = log(a)       ->  da += (1/a.data) * out.grad
        out = tanh(a)      ->  da += (1 - out.data**2) * out.grad
        out = relu(a)      ->  da += (out.data > 0) * out.grad

    PATRON DE CADA OPERACION (usa este molde):

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data + other.data, (self, other), '+')

            def _backward():
                ...   # aqui las derivadas locales, con +=
            out._backward = _backward
            return out

    La closure captura `self`, `other` y `out` por referencia. Cuando se ejecute,
    `out.grad` ya tendra el valor que le hayan puesto sus padres. Eso es lo que hace que
    esto funcione y por lo que el orden del recorrido importa.

    ATENCION: no uses `__slots__` si no sabes exactamente lo que hace, y define `_prev`
    como tupla o set de los hijos.
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
    """Orden topologico del grafo que cuelga de `root`.

    Contrato: cada nodo aparece en la lista DESPUES de todos sus hijos (`_prev`). Es
    decir, `root` es el ultimo elemento. `backward()` recorre la lista al reves.

    Por que importa: `node._backward()` reparte el gradiente de `node` a sus hijos. Si lo
    llamas antes de que todos los PADRES de `node` hayan aportado su parte a `node.grad`,
    estaras repartiendo un gradiente incompleto. En un grafo con nodos reutilizados eso da
    gradientes incorrectos sin ningun sintoma visible.

    Implementalo ITERATIVO con una pila explicita, no con recursion: el grafo de un MLP de
    unos cientos de neuronas ya supera el limite de recursion de python.

    Pista de estructura: DFS post-orden. Una pila de tuplas `(nodo, hijos_ya_expandidos)`
    resuelve el post-orden sin recursion. Usa `id(nodo)` para el conjunto de visitados,
    no el nodo (`Value` no define `__hash__` de forma fiable si sobrecargas operadores).

    Args:
        root: nodo raiz, tipicamente la perdida.

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
    """Entrena un MLP con TU motor de autodiff y devuelve el historial de perdida.

    Este es el bucle de entrenamiento en su forma mas desnuda. El del modulo 10 sera el
    mismo con AMP, scheduler y checkpointing por encima; el nucleo no cambia.

    Pasos, en este orden exacto:

        model = MLP(len(xs[0]), [*hidden, 1], value_cls=value_cls, seed=seed)

        repetir `steps` veces:
            1. forward   : preds = [model(x) for x in xs]
            2. perdida   : MSE = (1/N) * sum((pred - y)**2)
                           OJO: `sum()` de python empieza en el int 0; pasale
                           `value_cls(0.0)` como valor inicial o mezclaras tipos.
            3. zero grad : model.zero_grad()   <-- ANTES del backward, no despues
            4. backward  : loss.backward()
            5. paso      : para cada p en model.parameters(): p.data -= lr * p.grad
            6. registrar : history.append(loss.data)

    El paso 3 es el que se olvida todo el mundo. Los gradientes se acumulan (`+=`), asi
    que sin ponerlos a cero el paso N usa la suma de los gradientes de los pasos 1..N.
    El sintoma es que la perdida baja un poco y luego se estanca o explota.

    Args:
        xs: lista de vectores de entrada, todos de la misma longitud.
        ys: objetivos escalares, uno por entrada.
        hidden: tamanyos de las capas ocultas. La de salida (1 neurona) se anyade sola.
        steps: pasos de descenso de gradiente.
        lr: tasa de aprendizaje.
        seed: semilla de la inicializacion, para reproducibilidad.
        value_cls: la clase escalar a usar. Por defecto la tuya.

    Returns:
        Lista de `steps` perdidas (floats), una por paso.
    """
    raise NotImplementedError("TODO: modulo 02, ejercicio 3 - train_scalar_mlp")
