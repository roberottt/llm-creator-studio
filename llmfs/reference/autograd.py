"""Referencia del modulo 01: motor de autodiferenciacion escalar.

Un `Value` es un numero que ademas recuerda de donde ha salido. Al encadenar operaciones
se construye un grafo dirigido aciclico; `backward()` lo recorre en orden topologico
inverso aplicando la regla de la cadena.

Es exactamente lo que hace PyTorch, solo que PyTorch lo hace sobre tensores y en C++.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Sequence


class Value:
    """Un escalar con gradiente.

    Attributes:
        data: el valor hacia delante.
        grad: derivada de la salida final respecto a este nodo. Se acumula.
        _prev: nodos de los que depende este.
        _op: etiqueta de la operacion, solo para depurar y dibujar el grafo.
        _backward: closure que propaga el gradiente de este nodo a sus hijos.
    """

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
        """Propaga el gradiente desde este nodo hasta las hojas.

        Dos detalles que importan:

        1. Se arranca con `self.grad = 1.0`, porque la derivada de la salida respecto a
           si misma es 1.
        2. Hay que recorrer en orden topologico inverso. Si propagases el gradiente de un
           nodo antes de que TODOS sus padres hayan aportado el suyo, ese nodo enviaria
           hacia abajo un gradiente incompleto. Con un grafo en forma de arbol no se nota;
           en cuanto hay un nodo reutilizado (y en una red neuronal los hay a miles), da
           resultados mal.
        """
        self.grad = 1.0
        for node in reversed(topological_order(self)):
            node._backward()


def topological_order(root: Value) -> list[Value]:
    """Orden topologico del grafo: cada nodo aparece despues de todos sus hijos.

    DFS post-orden iterativo. Iterativo y no recursivo porque el grafo de un MLP con
    unos cuantos cientos de neuronas ya se acerca al limite de recursion de python.
    """
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


# ---------------------------------------------------------------------------- red neuronal


class Neuron:
    """Una neurona: `act(w . x + b)`."""

    def __init__(self, nin: int, nonlin: bool = True, value_cls: type = Value, rng: Any = None) -> None:
        rng = rng or random.Random(0)
        self.w = [value_cls(rng.uniform(-1, 1)) for _ in range(nin)]
        self.b = value_cls(0.0)
        self.nonlin = nonlin

    def __call__(self, x: Sequence[Any]) -> Any:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self) -> list[Any]:
        return [*self.w, self.b]


class Layer:
    def __init__(self, nin: int, nout: int, value_cls: type = Value, rng: Any = None, **kw: Any) -> None:
        self.neurons = [Neuron(nin, value_cls=value_cls, rng=rng, **kw) for _ in range(nout)]

    def __call__(self, x: Sequence[Any]) -> Any:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Any]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Perceptron multicapa construido sobre `value_cls`.

    `value_cls` se pasa por parametro para que puedas montarlo con TU clase `Value` sin
    tocar este codigo. Es lo que hace el ejercicio `train_scalar_mlp`.
    """

    def __init__(
        self,
        nin: int,
        nouts: Sequence[int],
        value_cls: type = Value,
        seed: int = 0,
    ) -> None:
        rng = random.Random(seed)
        sizes = [nin, *nouts]
        self.layers = [
            Layer(
                sizes[i],
                sizes[i + 1],
                value_cls=value_cls,
                rng=rng,
                nonlin=i < len(nouts) - 1,  # la ultima capa es lineal
            )
            for i in range(len(nouts))
        ]

    def __call__(self, x: Sequence[Any]) -> Any:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[Any]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0


def train_scalar_mlp(
    xs: Sequence[Sequence[float]],
    ys: Sequence[float],
    hidden: Sequence[int] = (8, 8),
    steps: int = 100,
    lr: float = 0.05,
    seed: int = 0,
    value_cls: type = Value,
) -> list[float]:
    """Entrena un MLP con descenso de gradiente y devuelve el historial de perdida.

    Perdida: error cuadratico medio. El bucle es el mismo que usaras en el modulo 10 con
    PyTorch, y conviene verlo aqui sin ninguna abstraccion por medio.

    Returns:
        Lista de `steps` perdidas, una por paso.
    """
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
