"""Reference for module 06: self-attention.

The centerpiece of the Transformer. Each token looks at the previous ones, decides which
to pay attention to, and takes away a weighted mixture of what they contribute.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Triangular mask that blocks looking into the future.

    Convention: `True` = looking IS allowed. It is the same one
    `F.scaled_dot_product_attention` uses with a boolean `attn_mask`.

    For `seq_len=4`:

        [[ True, False, False, False],     token 0 only sees itself
         [ True,  True, False, False],     token 1 sees token 0 and itself
         [ True,  True,  True, False],
         [ True,  True,  True,  True]]

    The diagonal is included: a token can look at itself.
    """
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def single_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-head attention: `softmax(Q K^T / sqrt(d_k) + mask) V`.

    Args:
        q: `(B, T, d_k)` the queries.
        k: `(B, S, d_k)` the keys.
        v: `(B, S, d_v)` the values.
        mask: `(T, S)` or `(B, T, S)` boolean, `True` = allowed. `None` = no mask.

    Returns:
        `(output, weights)` with output `(B, T, d_v)` and weights `(B, T, S)`. The weights
        are returned because they are what the demo's heatmap draws.
    """
    d_k = q.shape[-1]

    # (B, T, d_k) @ (B, d_k, S) -> (B, T, S). scores[b,i,j] = how interested i is in j.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        # -inf before the softmax becomes probability 0 after it.
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):
    """Several attentions in parallel, each with its own projections.

    A single head has to resolve every relationship in the sentence with one attention
    pattern. With several, each can specialize: there are heads that look at the previous
    token, others that hunt for the verb's subject, others that copy.

    The implementation trick is that you do NOT make `n_heads` separate projections: you
    make one big `d_model -> d_model` projection and split the result into `n_heads` chunks
    of `head_dim`. It is mathematically equivalent and vastly faster, because it is one big
    matmul instead of eight small ones.

    Submodules (the tests copy weights by name):
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
            raise ValueError(f"d_model ({d_model}) is not divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        #: If `True`, use PyTorch's fused kernel. It is switched on in module 12, where
        #: the gain is measured. The output is the same; memory and speed change.
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
        # contiguous() is needed because transpose leaves the tensor with odd strides and
        # view() demands contiguous memory.
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
            mask: `(T, T)` boolean, `True` = allowed. If `None`, a causal one is built.
            cos, sin: RoPE tables (module 09). If given, Q and K are rotated before the
                scores are computed. `None` = no positional information here.
            return_weights: also return the attention weights, for visualization.
            cache: the `KVCache` from module 14, or `None`.
            layer_idx: which layer this is, for indexing the cache.
            pos_offset: how many tokens are already in the cache. Used so RoPE rotates the
                new token by the angle of its REAL position, not position 0.

        Returns:
            `(B, T, d_model)`, or the tuple `(output, weights)` if `return_weights`.
        """
        seq_len = x.shape[1]

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if cos is not None and sin is not None:
            # To Q and K only, never to V: what should depend on position is the attention
            # scores, not the content being transported.
            #
            # The slice by `pos_offset` is what makes the cache correct: when generating
            # token 50 only that token is passed in, but it has to be rotated by the angle
            # of position 50, not position 0.
            from llmfs.reference.position import apply_rope

            cos_t = cos[pos_offset : pos_offset + seq_len]
            sin_t = sin[pos_offset : pos_offset + seq_len]
            q, k = apply_rope(q, cos_t, sin_t), apply_rope(k, cos_t, sin_t)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        total_len = k.shape[-2]
        if mask is None:
            if cache is not None:
                # With a cache, the new token can look at EVERYTHING before it plus
                # itself. Nothing needs triangulating: the cache only holds the past.
                mask = torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device)
            else:
                mask = causal_mask(seq_len, device=x.device)

        if self.use_sdpa and not return_weights:
            # Fused kernel: it does not materialize the T x T matrix. On Turing it uses
            # the memory_efficient backend because FlashAttention-2 needs sm_80+.
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
