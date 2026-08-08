"""Referencia del modulo 10: el modelo completo.

Aqui se juntan todas las piezas de los modulos 06-09 en el GPT de 8.933.440 parametros.
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
    """La capa de normalizacion que pida el config."""
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.d_model)
    if cfg.norm == "layernorm":
        return nn.LayerNorm(cfg.d_model, bias=cfg.bias)
    raise ValueError(f"norm desconocida: {cfg.norm}")


def make_ffn(cfg: ModelConfig) -> nn.Module:
    """La red feed-forward que pida el config."""
    if cfg.activation == "swiglu":
        return SwiGLU(cfg.d_model, cfg.d_ff, dropout=cfg.dropout, bias=cfg.bias)
    return MLP(cfg.d_model, cfg.d_ff, dropout=cfg.dropout, bias=cfg.bias)


class TransformerBlock(nn.Module):
    """Un bloque: atencion y FFN, cada uno con su normalizacion y su residual.

        x = x + atencion(norm1(x))
        x = x + ffn(norm2(x))

    Es pre-norm: la normalizacion va DENTRO de la rama, no envolviendo a la suma. Asi el
    camino residual `x -> x` queda libre de cualquier operacion, y el gradiente llega
    intacto hasta la primera capa.

    Los dos residuales son independientes a proposito: cada sub-bloque puede aportar poco o
    mucho a la corriente residual sin condicionar al otro.

    Submodulos:
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
    """El modelo completo.

    Estructura:

        tokens -> embeddings -> [bloque] x n_layers -> norm final -> logits

    Tres decisiones de disenyo que importan y que el modulo 10 desarrolla:

    1. **Weight tying.** `lm_head.weight` ES el mismo tensor que `token_embedding.weight`.
       La matriz que convierte un id en vector se reutiliza, transpuesta, para convertir un
       vector en puntuaciones sobre el vocabulario. Ahorra 1,31 M de parametros (un 15% del
       modelo) y ademas suele mejorar la calidad, porque cada peso recibe gradiente por dos
       caminos.

    2. **Inicializacion escalada por profundidad.** Las proyecciones que ESCRIBEN en la
       corriente residual (`out_proj` de la atencion y `down_proj` del FFN) se inicializan
       con desviacion `0.02 / sqrt(2 * n_layers)` en vez de `0.02`. Sin eso, la varianza de
       la corriente residual crece linealmente con la profundidad, porque cada capa suma su
       contribucion. El 2 es porque cada bloque escribe dos veces (atencion y FFN).

    3. **Norma final.** Obligatoria en pre-norm. Como la corriente residual nunca se
       normaliza por el camino, llega a la salida con una escala arbitraria.

    Submodulos:
        token_embedding, blocks (ModuleList), norm_f, lm_head
        pos_embedding solo si cfg.pos == "learned"
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
            # persistent=False: se recalculan al construir, no hace falta guardarlas en el
            # checkpoint ni que ocupen sitio en el fichero.
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        elif cfg.pos == "sinusoidal":
            self.register_buffer(
                "pos_table", sinusoidal_embeddings(cfg.context_length, cfg.d_model),
                persistent=False,
            )

        self.apply(self._init_weights)
        # La init escalada se aplica DESPUES del apply general, para pisarla.
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
            idx: `(B, T)` int64 con los ids de token.
            targets: `(B, T)` int64, el token siguiente en cada posicion. `None` en
                inferencia.
            use_cache: si `True`, usa la KV cache del modulo 14.
            cache: el objeto `KVCache`.

        Returns:
            `(logits, loss)`. Logits `(B, T, vocab_size)`; `loss` es `None` sin targets.
        """
        batch, seq_len = idx.shape
        pos_offset = cache.seq_len if (use_cache and cache is not None) else 0

        if seq_len + pos_offset > self.cfg.context_length:
            raise ValueError(
                f"secuencia de {seq_len + pos_offset} tokens, pero el contexto del modelo "
                f"es {self.cfg.context_length}"
            )

        x = self.token_embedding(idx)
        posiciones = torch.arange(pos_offset, pos_offset + seq_len, device=idx.device)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(posiciones)
        elif self.cfg.pos == "sinusoidal":
            x = x + self.pos_table[posiciones]
        x = self.drop(x)

        cos = sin = None
        if self.cfg.pos == "rope":
            cos, sin = self.rope_cos, self.rope_sin

        # La mascara solo se puede omitir cuando entra UN token (decode con cache): ahi la
        # query es una sola y ve todo el pasado, que es justo lo que debe. En prefill entra
        # el prompt entero, `seq_len > 1`, y sin mascara los tokens del prompt se ven entre
        # si hacia delante: fuga de informacion que corrompe las K/V guardadas en la cache
        # y, con ellas, todo lo que se genere despues.
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
        """Generacion ingenua: recalcula todo el contexto en cada token.

        Es correcta pero lenta: en el modulo 14 se implementa la KV cache, que evita
        recalcular lo que ya estaba y da la misma salida N veces mas rapido.
        """
        self.eval()
        for _ in range(max_new_tokens):
            recorte = idx[:, -self.cfg.context_length :]
            logits, _ = self(recorte)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


# ---------------------------------------------------------------------------- conteo


def expected_param_count(cfg: ModelConfig) -> int:
    """El numero de parametros, calculado con la formula en vez de contando.

    Sirve para dos cosas: comprobar que el modelo que has montado es el que creias, y
    poder disenyar una arquitectura ANTES de construirla.

        embeddings     = vocab_size * d_model
        atencion/capa  = 4 * d_model^2            (Wq, Wk, Wv, Wo, sin sesgo)
        ffn/capa       = 3 * d_model * d_ff       (SwiGLU) o 2 * d_model * d_ff (MLP)
        normas/capa    = 2 * d_model              (RMSNorm: solo escala)
        norma final    = d_model
        lm_head        = 0 si hay tying, si no vocab_size * d_model

    Con el config final:
        1.310.720 + 6 * (409.600 + 860.160 + 640) + 320 = 8.933.440
    """
    d, v, ff = cfg.d_model, cfg.vocab_size, cfg.d_ff

    total = v * d  # embeddings de token
    if cfg.pos == "learned":
        total += cfg.context_length * d

    atencion = 4 * d * d + (4 * d if cfg.bias else 0)
    ffn_matrices = 3 if cfg.activation == "swiglu" else 2
    ffn = ffn_matrices * d * ff
    if cfg.bias:
        ffn += 2 * ff + d if ffn_matrices == 3 else ff + d

    # RMSNorm tiene solo escala; LayerNorm tiene escala y (opcionalmente) sesgo.
    por_norma = d if cfg.norm == "rmsnorm" else (2 * d if cfg.bias else d)

    total += cfg.n_layers * (atencion + ffn + 2 * por_norma)
    total += por_norma  # la norma final

    if not cfg.tie_embeddings:
        total += v * d

    return total


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Cuenta los parametros de verdad, desglosados por componente.

    Con weight tying, `lm_head.weight` y `token_embedding.weight` son EL MISMO tensor.
    `model.parameters()` deduplica por identidad, asi que el total sale bien solo; pero al
    desglosar hay que tener cuidado de no contarlo dos veces. Por eso se lleva un conjunto
    de `id()` ya vistos.

    Returns:
        Un dict con `embeddings`, `attention`, `ffn`, `norms`, `lm_head`, `other`,
        `total` y `non_embedding`.
    """
    desglose = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "norms": 0,
        "lm_head": 0,
        "other": 0,
    }
    vistos: set[int] = set()

    for name, param in model.named_parameters():
        if id(param) in vistos:
            continue  # tying: ya contado
        vistos.add(id(param))
        n = param.numel()

        if "token_embedding" in name or "pos_embedding" in name:
            desglose["embeddings"] += n
        elif "attn." in name or "attention" in name:
            desglose["attention"] += n
        elif any(k in name for k in ("gate_proj", "up_proj", "down_proj", "fc_in", "fc_out")):
            desglose["ffn"] += n
        elif "norm" in name:
            desglose["norms"] += n
        elif "lm_head" in name:
            desglose["lm_head"] += n
        else:
            desglose["other"] += n

    desglose["total"] = sum(desglose.values())
    desglose["non_embedding"] = desglose["total"] - desglose["embeddings"]
    return desglose
