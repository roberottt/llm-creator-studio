"""Referencia del modulo 06: self-attention.

La pieza central del Transformer. Cada token mira a los anteriores, decide a cuales hacer
caso, y se lleva una mezcla ponderada de lo que aportan.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Mascara triangular que impide mirar hacia el futuro.

    Convenio: `True` = SI se puede mirar. Es el mismo que usa
    `F.scaled_dot_product_attention` con `attn_mask` booleana.

    Para `seq_len=4`:

        [[ True, False, False, False],     el token 0 solo se ve a si mismo
         [ True,  True, False, False],     el token 1 ve al 0 y a si mismo
         [ True,  True,  True, False],
         [ True,  True,  True,  True]]

    La diagonal va incluida: un token si puede mirarse a si mismo.
    """
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def single_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Atencion de una cabeza: `softmax(Q K^T / sqrt(d_k) + mascara) V`.

    Args:
        q: `(B, T, d_k)` las preguntas.
        k: `(B, S, d_k)` las etiquetas.
        v: `(B, S, d_v)` los contenidos.
        mask: `(T, S)` o `(B, T, S)` booleana, `True` = permitido. `None` = sin mascara.

    Returns:
        `(salida, pesos)` con salida `(B, T, d_v)` y pesos `(B, T, S)`. Los pesos se
        devuelven porque son lo que dibuja el heatmap de la demo.
    """
    d_k = q.shape[-1]

    # (B, T, d_k) @ (B, d_k, S) -> (B, T, S). scores[b,i,j] = cuanto le interesa a i el j.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        # -inf antes del softmax se convierte en probabilidad 0 despues.
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """Varias atenciones en paralelo, cada una con sus propias proyecciones.

    Una sola cabeza tiene que resolver con un unico patron de atencion todas las
    relaciones de la frase. Con varias, cada una puede especializarse: hay cabezas que
    miran al token anterior, otras que buscan el sujeto del verbo, otras que copian.

    El truco de implementacion es que NO se hacen `n_heads` proyecciones separadas: se
    hace una proyeccion grande de `d_model -> d_model` y se parte el resultado en
    `n_heads` trozos de `head_dim`. Es matematicamente equivalente y muchisimo mas rapido,
    porque es un matmul grande en vez de ocho pequenyos.

    Submodulos (los tests copian pesos por nombre):
        q_proj, k_proj, v_proj, out_proj: nn.Linear(d_model, d_model, bias=bias)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_sdpa: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) no es divisible por n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        #: Si `True`, usa el kernel fusionado de PyTorch. Se activa en el modulo 12, donde
        #: se mide cuanto gana. La salida es la misma; cambia la memoria y la velocidad.
        self.use_sdpa = use_sdpa

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, T, d_model)` -> `(B, n_heads, T, head_dim)`."""
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, n_heads, T, head_dim)` -> `(B, T, d_model)`."""
        batch, _, seq_len, _ = x.shape
        # contiguous() hace falta porque transpose deja el tensor con strides raros y
        # view() exige memoria contigua.
        return x.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        return_weights: bool = False,
        cache: Any = None,
        layer_idx: int = 0,
        pos_offset: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: `(B, T, d_model)`.
            mask: `(T, T)` booleana, `True` = permitido. Si es `None` se genera causal.
            cos, sin: tablas de RoPE (modulo 09). Si se pasan, se rotan Q y K antes de
                calcular las puntuaciones. `None` = sin informacion posicional aqui.
            return_weights: devolver tambien los pesos de atencion, para visualizarlos.
            cache: `KVCache` del modulo 14, o `None`.
            layer_idx: que capa es, para indexar la cache.
            pos_offset: cuantos tokens hay ya en la cache. Se usa para que RoPE rote el
                token nuevo con el angulo de su posicion REAL, no de la 0.

        Returns:
            `(B, T, d_model)`, o la tupla `(salida, pesos)` si `return_weights`.
        """
        seq_len = x.shape[1]

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if cos is not None and sin is not None:
            # Solo a Q y K, nunca a V: lo que debe depender de la posicion son las
            # puntuaciones de atencion, no el contenido que se transporta.
            #
            # El recorte por `pos_offset` es lo que hace que la cache sea correcta: al
            # generar el token 50 se le pasa solo el, pero tiene que rotarse con el angulo
            # de la posicion 50, no de la 0.
            from llmfs.reference.position import apply_rope

            cos_t = cos[pos_offset : pos_offset + seq_len]
            sin_t = sin[pos_offset : pos_offset + seq_len]
            q, k = apply_rope(q, cos_t, sin_t), apply_rope(k, cos_t, sin_t)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        total_len = k.shape[-2]
        if mask is None:
            if cache is not None:
                # Con cache, el token nuevo puede mirar a TODO lo anterior mas a si mismo.
                # No hace falta triangular nada: la propia cache solo contiene el pasado.
                mask = torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device)
            else:
                mask = causal_mask(seq_len, device=x.device)

        if self.use_sdpa and not return_weights:
            # Kernel fusionado: no materializa la matriz T x T. En Turing usa el backend
            # memory_efficient porque FlashAttention-2 pide sm_80+.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
            )
            weights = None
        else:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            out = self.attn_dropout(weights) @ v

        out = self.resid_dropout(self.out_proj(self._merge_heads(out)))
        if return_weights:
            return out, weights  # type: ignore[return-value]
        return out
