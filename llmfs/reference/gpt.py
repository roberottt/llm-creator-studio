"""Reference for module 10: the complete model.

This is where all the pieces from modules 06-09 come together into the 8,933,440-parameter
GPT.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmfs.config import ModelConfig
from llmfs.reference.attention import MultiHeadAttention, causal_mask
from llmfs.reference.mlp import MLP, SwiGLU
from llmfs.reference.norm import RMSNorm
from llmfs.reference.position import rope_frequencies, sinusoidal_embeddings


def make_norm(cfg: ModelConfig) -> nn.Module:
    """Whichever normalization layer the config asks for."""
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.d_model)
    if cfg.norm == "layernorm":
        return nn.LayerNorm(cfg.d_model, bias=cfg.bias)
    raise ValueError(f"unknown norm: {cfg.norm}")


def make_ffn(cfg: ModelConfig) -> nn.Module:
    """Whichever feed-forward network the config asks for."""
    if cfg.activation == "swiglu":
        return SwiGLU(cfg.d_model, cfg.d_ff, dropout=cfg.dropout, bias=cfg.bias)
    return MLP(cfg.d_model, cfg.d_ff, dropout=cfg.dropout, bias=cfg.bias)


class TransformerBlock(nn.Module):
    """One block: attention and FFN, each with its own normalization and residual.

        x = x + attention(norm1(x))
        x = x + ffn(norm2(x))

    This is pre-norm: the normalization goes INSIDE the branch, not wrapped around the sum.
    That leaves the residual path `x -> x` free of any operation, and the gradient reaches
    the first layer intact.

    The two residuals are independent on purpose: each sub-block can contribute a little or
    a lot to the residual stream without constraining the other.

    Submodules:
        attn_norm, attn, ffn_norm, ffn
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = make_norm(cfg)
        self.attn = MultiHeadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, bias=cfg.bias
        )
        self.ffn_norm = make_norm(cfg)
        self.ffn = make_ffn(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: object = None,
        layer_idx: int = 0,
        pos_offset: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            mask=mask,
            cos=cos,
            sin=sin,
            cache=cache,
            layer_idx=layer_idx,
            pos_offset=pos_offset,
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x


class GPT(nn.Module):
    """The complete model.

    Structure:

        tokens -> embeddings -> [block] x n_layers -> final norm -> logits

    Three design decisions that matter, developed in module 10:

    1. **Weight tying.** `lm_head.weight` IS the same tensor as `token_embedding.weight`.
       The matrix that turns an id into a vector is reused, transposed, to turn a vector
       into scores over the vocabulary. It saves 1.31 M parameters (15% of the model) and
       usually improves quality too, because each weight receives gradient along two paths.

    2. **Depth-scaled initialization.** The projections that WRITE into the residual stream
       (attention's `out_proj` and the FFN's `down_proj`) are initialized with standard
       deviation `0.02 / sqrt(2 * n_layers)` instead of `0.02`. Without that, the variance
       of the residual stream grows linearly with depth, because every layer adds its
       contribution. The 2 is because each block writes twice (attention and FFN).

    3. **Final norm.** Mandatory in pre-norm. Since the residual stream is never normalized
       along the way, it reaches the output at an arbitrary scale.

    Submodules:
        token_embedding, blocks (ModuleList), norm_f, lm_head
        pos_embedding only if cfg.pos == "learned"
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embedding: nn.Embedding | None = None
        if cfg.pos == "learned":
            self.pos_embedding = nn.Embedding(cfg.context_length, cfg.d_model)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        if cfg.pos == "rope":
            cos, sin = rope_frequencies(cfg.head_dim, cfg.context_length, cfg.rope_theta)
            # persistent=False: they are recomputed on construction, so there is no need
            # to store them in the checkpoint or have them take up space in the file.
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        elif cfg.pos == "sinusoidal":
            self.register_buffer(
                "pos_table", sinusoidal_embeddings(cfg.context_length, cfg.d_model),
                persistent=False,
            )

        self.apply(self._init_weights)
        # The scaled init is applied AFTER the general apply, to override it.
        scale = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, param in self.named_parameters():
            if name.endswith(("out_proj.weight", "down_proj.weight", "fc_out.weight")):
                nn.init.normal_(param, mean=0.0, std=scale)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_cache: bool = False,
        cache: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            idx: `(B, T)` int64 with the token ids.
            targets: `(B, T)` int64, the next token at each position. `None` at inference
                time.
            use_cache: if `True`, use the KV cache from module 14.
            cache: the `KVCache` object.

        Returns:
            `(logits, loss)`. Logits `(B, T, vocab_size)`; `loss` is `None` with no targets.
        """
        batch, seq_len = idx.shape
        pos_offset = cache.seq_len if (use_cache and cache is not None) else 0

        if seq_len + pos_offset > self.cfg.context_length:
            raise ValueError(
                f"sequence of {seq_len + pos_offset} tokens, but the model's context "
                f"is {self.cfg.context_length}"
            )

        x = self.token_embedding(idx)
        positions = torch.arange(pos_offset, pos_offset + seq_len, device=idx.device)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(positions)
        elif self.cfg.pos == "sinusoidal":
            x = x + self.pos_table[positions]
        x = self.drop(x)

        cos = sin = None
        if self.cfg.pos == "rope":
            cos, sin = self.rope_cos, self.rope_sin

        # The mask can only be skipped when ONE token comes in (decode with cache): there the
        # query is a single one and sees the whole past, which is exactly right. In prefill
        # the entire prompt comes in, `seq_len > 1`, and without a mask the prompt tokens see
        # each other forwards: an information leak that corrupts the K/V stored in the cache
        # and, with them, everything generated afterwards.
        mask = None if seq_len == 1 else causal_mask(seq_len, device=idx.device)
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                cos=cos,
                sin=sin,
                mask=mask,
                cache=cache if use_cache else None,
                layer_idx=i,
                pos_offset=pos_offset,
            )

        x = self.norm_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Naive generation: it recomputes the whole context for every token.

        It is correct but slow: module 14 implements the KV cache, which avoids recomputing
        what was already there and gives the same output N times faster.
        """
        self.eval()
        for _ in range(max_new_tokens):
            window = idx[:, -self.cfg.context_length :]
            logits, _ = self(window)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


# ---------------------------------------------------------------------------- counting


def expected_param_count(cfg: ModelConfig) -> int:
    """The parameter count, computed from the formula instead of by counting.

    It is good for two things: checking that the model you assembled is the one you thought
    you assembled, and being able to design an architecture BEFORE building it.

        embeddings      = vocab_size * d_model
        attention/layer = 4 * d_model^2            (Wq, Wk, Wv, Wo, no bias)
        ffn/layer       = 3 * d_model * d_ff       (SwiGLU) or 2 * d_model * d_ff (MLP)
        norms/layer     = 2 * d_model              (RMSNorm: scale only)
        final norm      = d_model
        lm_head         = 0 with tying, otherwise vocab_size * d_model

    With the final config:
        1,310,720 + 6 * (409,600 + 860,160 + 640) + 320 = 8,933,440
    """
    d, v, ff = cfg.d_model, cfg.vocab_size, cfg.d_ff

    total = v * d  # token embeddings
    if cfg.pos == "learned":
        total += cfg.context_length * d

    attention = 4 * d * d + (4 * d if cfg.bias else 0)
    ffn_matrices = 3 if cfg.activation == "swiglu" else 2
    ffn = ffn_matrices * d * ff
    if cfg.bias:
        ffn += 2 * ff + d if ffn_matrices == 3 else ff + d

    # RMSNorm has scale only; LayerNorm has scale and (optionally) bias.
    per_norm = d if cfg.norm == "rmsnorm" else (2 * d if cfg.bias else d)

    total += cfg.n_layers * (attention + ffn + 2 * per_norm)
    total += per_norm  # the final norm

    if not cfg.tie_embeddings:
        total += v * d

    return total


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count the parameters for real, broken down by component.

    With weight tying, `lm_head.weight` and `token_embedding.weight` are THE SAME tensor.
    `model.parameters()` deduplicates by identity, so the total comes out right on its own;
    but when breaking it down you have to be careful not to count it twice. That is why a
    set of already-seen `id()`s is kept.

    Returns:
        A dict with `embeddings`, `attention`, `ffn`, `norms`, `lm_head`, `other`,
        `total` and `non_embedding`.
    """
    breakdown = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "norms": 0,
        "lm_head": 0,
        "other": 0,
    }
    seen: set[int] = set()

    for name, param in model.named_parameters():
        if id(param) in seen:
            continue  # tying: already counted
        seen.add(id(param))
        n = param.numel()

        if "token_embedding" in name or "pos_embedding" in name:
            breakdown["embeddings"] += n
        elif "attn." in name or "attention" in name:
            breakdown["attention"] += n
        elif any(k in name for k in ("gate_proj", "up_proj", "down_proj", "fc_in", "fc_out")):
            breakdown["ffn"] += n
        elif "norm" in name:
            breakdown["norms"] += n
        elif "lm_head" in name:
            breakdown["lm_head"] += n
        else:
            breakdown["other"] += n

    breakdown["total"] = sum(breakdown.values())
    breakdown["non_embedding"] = breakdown["total"] - breakdown["embeddings"]
    return breakdown
