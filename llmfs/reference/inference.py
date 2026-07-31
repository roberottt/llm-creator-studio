"""Referencia del modulo 14: muestreo y KV cache."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float = 1.1
) -> torch.Tensor:
    """Penaliza los tokens que ya han salido, para romper bucles.

    El detalle que casi todo el mundo implementa mal: hay que DIVIDIR si el logit es
    positivo y MULTIPLICAR si es negativo.

        logit > 0  ->  logit / penalty     lo acerca a cero
        logit < 0  ->  logit * penalty     lo aleja de cero, hacia abajo

    Si dividieras siempre, un logit de -5 pasaria a -4,5, o sea que el token se volveria
    MAS probable: justo lo contrario de penalizarlo.

    Con `penalty=1.0` no hace nada. Valores tipicos: 1.05 a 1.2.
    """
    if penalty == 1.0:
        return logits

    salida = logits.clone()
    for fila in range(logits.shape[0]):
        vistos = torch.unique(generated[fila])
        valores = salida[fila, vistos]
        salida[fila, vistos] = torch.where(valores > 0, valores / penalty, valores * penalty)
    return salida


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Deja solo los `k` logits mayores y pone el resto a -inf.

    Corta la cola larga de la distribucion. Con vocabulario de 4096, hay miles de tokens
    con probabilidad diminuta pero no nula; sumadas, esa cola puede llevarse un 20% de la
    masa, y de vez en cuando sale una de ellas y descarrila la frase.

    `k <= 0` o `k >= vocab_size` no filtra nada.
    """
    if k <= 0 or k >= logits.shape[-1]:
        return logits

    umbral = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < umbral, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: se queda con los tokens que acumulan una masa `p`.

    A diferencia de top-k, el numero de candidatos es VARIABLE:

    - si el modelo esta muy seguro, un solo token puede acumular el 90% y se queda solo el
    - si duda entre muchos, se quedan muchos

    Eso es lo que lo hace mejor que top-k en la practica (Holtzman et al. 2020): se adapta
    a lo seguro que este el modelo en cada posicion.

    El detalle que importa: el primer token SIEMPRE se conserva, aunque el solo ya supere
    `p`. Si no, con p=0.5 y un token de probabilidad 0.9 te quedarias sin candidatos.
    """
    if p >= 1.0:
        return logits

    ordenados, indices = torch.sort(logits, descending=True, dim=-1)
    acumulada = torch.cumsum(F.softmax(ordenados, dim=-1), dim=-1)

    quitar = acumulada - F.softmax(ordenados, dim=-1) > p
    quitar[..., 0] = False  # el mas probable siempre se queda

    a_quitar = quitar.scatter(-1, indices, quitar)
    return logits.masked_fill(a_quitar, float("-inf"))


class KVCache:
    """Guarda las claves y valores ya calculados para no recalcularlos.

    EL PROBLEMA. Al generar el token 100, la version ingenua vuelve a pasar los 100 tokens
    por el modelo, aunque los 99 primeros no han cambiado. Generar N tokens cuesta
    O(N^2) en vez de O(N).

    LA SOLUCION. Guardar las claves y valores de cada capa y, en cada paso, procesar SOLO
    el token nuevo, concatenando sus K y V a lo guardado.

    LO QUE NO SE PUEDE CACHEAR: las queries. Cada token nuevo necesita su propia query para
    preguntar; lo que se reutiliza son las respuestas (K) y los contenidos (V) de los
    anteriores.

    LA MEMORIA:  2 * n_layers * T * d_model * bytes

    Para nuestro modelo con 512 tokens en fp16 son 3,9 MB. Para un modelo de 70B con
    contexto de 100.000, decenas de gigabytes, y de ahi que existan tecnicas como
    grouped-query attention.
    """

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self.keys: list[torch.Tensor | None] = [None] * n_layers
        self.values: list[torch.Tensor | None] = [None] * n_layers

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Anyade las K y V del token nuevo y devuelve la secuencia completa.

        Args:
            layer: el indice de capa.
            k, v: `(B, n_heads, T_nuevo, head_dim)`.

        Returns:
            `(K, V)` completas, `(B, n_heads, T_total, head_dim)`.
        """
        if self.keys[layer] is None:
            self.keys[layer] = k
            self.values[layer] = v
        else:
            # dim=-2 es la dimension de tiempo con la forma (B, heads, T, head_dim)
            self.keys[layer] = torch.cat([self.keys[layer], k], dim=-2)
            self.values[layer] = torch.cat([self.values[layer], v], dim=-2)
        return self.keys[layer], self.values[layer]

    @property
    def seq_len(self) -> int:
        """Cuantos tokens hay guardados."""
        return 0 if self.keys[0] is None else self.keys[0].shape[-2]

    def reset(self) -> None:
        self.keys = [None] * self.n_layers
        self.values = [None] * self.n_layers

    def memory_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in [*self.keys, *self.values]
            if t is not None
        )


@torch.no_grad()
def generate_with_cache(
    model: Any,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    eos_token: int | None = None,
) -> torch.Tensor:
    """Genera texto usando la KV cache.

    El bucle tiene dos fases:

    1. **Prefill**: se pasa el prompt entero de golpe y se llena la cache.
    2. **Decode**: en cada paso se pasa SOLO el ultimo token, se lee la cache, y se anyade
       el token nuevo.

    El orden de los filtros importa y es este: penalizacion -> temperatura -> top-k ->
    top-p. La penalizacion va primero porque opera sobre los logits crudos; la temperatura
    antes que los filtros porque cambia las probabilidades acumuladas que mira top-p.

    LIMITE DE CONTEXTO. Esta implementacion PARA al llegar al contexto maximo del modelo,
    en vez de recortar como hace `model.generate`.

    No es pereza: recortar con cache es genuinamente mas complicado. Habria que descartar
    las entradas antiguas Y remapear las posiciones de RoPE de todo lo que queda, porque
    los tokens que sobreviven pasarian a ocupar posiciones distintas. Se llama sliding
    window attention y da para un modulo entero.

    Parar es lo honesto: la alternativa silenciosa seria generar texto incorrecto sin
    avisar.
    """
    model.eval()
    cache = KVCache(model.cfg.n_layers)
    contexto_max = model.cfg.context_length

    if idx.shape[1] >= contexto_max:
        raise ValueError(
            f"el prompt ya tiene {idx.shape[1]} tokens y el contexto del modelo es "
            f"{contexto_max}: no queda sitio para generar"
        )

    logits, _ = model(idx, use_cache=True, cache=cache)

    for _ in range(max_new_tokens):
        if idx.shape[1] >= contexto_max:
            break
        siguiente = logits[:, -1, :].float()

        if repetition_penalty != 1.0:
            siguiente = apply_repetition_penalty(siguiente, idx, repetition_penalty)
        if temperature != 1.0:
            siguiente = siguiente / max(temperature, 1e-8)
        if top_k is not None:
            siguiente = top_k_filter(siguiente, top_k)
        if top_p is not None:
            siguiente = top_p_filter(siguiente, top_p)

        if temperature == 0.0:
            nuevo = siguiente.argmax(dim=-1, keepdim=True)
        else:
            nuevo = torch.multinomial(F.softmax(siguiente, dim=-1), num_samples=1)

        idx = torch.cat([idx, nuevo], dim=1)
        if eos_token is not None and bool((nuevo == eos_token).all()):
            break

        # Solo el token nuevo: la cache tiene el resto.
        logits, _ = model(nuevo, use_cache=True, cache=cache)

    return idx
