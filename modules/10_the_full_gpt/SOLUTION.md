# 10 — Solución comentada

## Ejercicio 1 — `expected_param_count`

```python
d, v, ff = cfg.d_model, cfg.vocab_size, cfg.d_ff

total = v * d                                    # embeddings de token
if cfg.pos == "learned":
    total += cfg.context_length * d

atencion = 4 * d * d + (4 * d if cfg.bias else 0)
ffn_matrices = 3 if cfg.activation == "swiglu" else 2
ffn = ffn_matrices * d * ff
por_norma = d if cfg.norm == "rmsnorm" else (2 * d if cfg.bias else d)

total += cfg.n_layers * (atencion + ffn + 2 * por_norma)
total += por_norma                               # la norma final

if not cfg.tie_embeddings:
    total += v * d

return total
```

Aritmética. Lo que importa es haberla derivado a mano antes de escribirla.

**RoPE no aporta ni un parámetro.** Sus tablas salen de una fórmula y se guardan como
*buffers*. Si tu cuenta incluye algo de RoPE, tienes un término de más.

**RMSNorm tiene $d$ parámetros, no $2d$.** Solo escala, sin sesgo. Con 6 capas × 2 normas +
1 final son 13 × 320 = 4.160 parámetros: una miseria, pero si los olvidas el total no cuadra
y el test lo dice.

## Ejercicio 2 — `count_parameters`

```python
desglose = {"embeddings": 0, "attention": 0, "ffn": 0, "norms": 0, "lm_head": 0, "other": 0}
vistos = set()

for name, param in model.named_parameters():
    if id(param) in vistos:
        continue
    vistos.add(id(param))
    n = param.numel()

    if "token_embedding" in name or "pos_embedding" in name:
        desglose["embeddings"] += n
    elif "attn." in name:
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
```

**Una corrección a lo que suele decirse sobre el weight tying.** Al escribir este módulo di
por hecho que `named_parameters()` devolvía el tensor atado dos veces, y el test lo desmintió:
**tanto `parameters()` como `named_parameters()` deduplican por identidad por defecto**
(`remove_duplicate=True`, desde PyTorch 1.13). El total sale bien sin hacer nada.

El `set` de `id()` sigue mereciendo la pena por dos motivos: deja explícito que sabes que hay
pesos compartidos, y protege el desglose si algún día recorres los parámetros con
`remove_duplicate=False`. Con `named_parameters(remove_duplicate=False)` sobre el modelo
final contarías 1.310.720 parámetros de más.

**El orden de las comprobaciones importa.** Los nombres son del estilo
`blocks.3.attn.out_proj.weight`, y `"norm"` aparece también en `attn_norm` y `ffn_norm`. Si
comprobaras `"norm"` antes que `"attn."`, la normalización de la atención acabaría en la
categoría equivocada. Comprueba de más específico a más general.

## Ejercicio 3 — `TransformerBlock`

```python
def __init__(self, cfg):
    super().__init__()
    self.attn_norm = make_norm(cfg)
    self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads,
                                   dropout=cfg.dropout, bias=cfg.bias)
    self.ffn_norm = make_norm(cfg)
    self.ffn = make_ffn(cfg)

def forward(self, x, cos=None, sin=None, mask=None):
    x = x + self.attn(self.attn_norm(x), mask=mask, cos=cos, sin=sin)
    x = x + self.ffn(self.ffn_norm(x))
    return x
```

Dos líneas de forward. **Dos residuales independientes**, no uno solo alrededor de todo el
bloque: cada sub-bloque decide por su cuenta cuánto aporta a la corriente residual.

El test `test_el_bloque_usa_residuales` lo comprueba de la forma más directa: pone a cero
los pesos de salida de las dos ramas y verifica que la salida es **exactamente** la entrada.
Si tu bloque no tiene residuales, devolvería cero.

El FFN no recibe `cos`, `sin` ni `mask`: no mira a otros tokens, así que no los necesita.

## Ejercicio 4 — `GPT`

### El tying

```python
if cfg.tie_embeddings:
    self.lm_head.weight = self.token_embedding.weight
```

Asignar el atributo hace que las dos capas apunten **al mismo objeto**. El test comprueba
`is`, no `==`: tienen que ser el mismo tensor, no dos copias con los mismos valores. Si
hicieras `self.lm_head.weight.data = self.token_embedding.weight.data.clone()` tendrías dos
tensores con los mismos números que divergirían en cuanto empezara el entrenamiento.

Va después de crear `lm_head` y antes de la inicialización.

### Los buffers de RoPE

```python
cos, sin = rope_frequencies(cfg.head_dim, cfg.context_length, cfg.rope_theta)
self.register_buffer("rope_cos", cos, persistent=False)
self.register_buffer("rope_sin", sin, persistent=False)
```

Un *buffer* es un tensor que acompaña al modelo —se mueve con `.to(device)`, aparece en
`.eval()`— pero no es un parámetro y no recibe gradiente.

`persistent=False` hace además que **no se guarde en el checkpoint**. Como se recalculan con
una fórmula al construir el modelo, guardarlas sería desperdiciar espacio y crear un problema
si algún día cambias `rope_theta`.

### La inicialización, en dos pasadas

```python
self.apply(self._init_weights)                    # todo con std=0.02

scale = 0.02 / math.sqrt(2 * cfg.n_layers)        # y luego se pisa lo que toca
for name, param in self.named_parameters():
    if name.endswith(("out_proj.weight", "down_proj.weight", "fc_out.weight")):
        nn.init.normal_(param, mean=0.0, std=scale)
```

**El orden importa**: primero todo, luego se pisa. Si lo hicieras al revés, el `apply`
sobrescribiría la inicialización escalada.

**Por qué.** Cada bloque *suma* su contribución a la corriente residual. Con contribuciones
independientes de varianza $\sigma^2$, la varianza de la suma crece linealmente con el número
de sumandos. Con 6 capas × 2 sub-bloques son 12 contribuciones: la salida tendría 12 veces
más varianza que la entrada. Reducir $\sigma$ por $\sqrt{2 n_{\text{layers}}}$ lo compensa
exactamente.

**Y el 0,02 tampoco es magia**: es lo que hace que la pérdida del paso 0 valga $\ln(V)$. Con
`std=1` (el defecto de PyTorch) el modelo arrancaría opinando fuerte y al azar, y la pérdida
saldría por encima — exactamente lo que ves en la demo del módulo 05.

### El forward

La única sutileza: **la máscara se calcula una vez**, antes del bucle de bloques, y se pasa a
todos. Calcularla dentro de cada bloque funcionaría, pero serían 6 tensores idénticos por
forward.

## Un bug que cometí escribiendo estos tests, y que te puede pasar

El test que comprueba la pérdida inicial fallaba dando **4,94 cuando debía dar 5,55**. Más
baja que $\ln(V)$, que es el síntoma clásico de fuga de información.

No había ninguna fuga en el modelo. La había en el test: yo pasaba `modelo(idx, idx)`, o sea
**los targets sin desplazar**. En la posición $t$ el modelo ve el token `idx[t]` y se le pide
que prediga `idx[t]`: puede leerlo directamente de su propia entrada. Con weight tying es
todavía más directo, porque los logits son $x W_{\text{emb}}^\top$ y el producto escalar de
un embedding consigo mismo es grande.

Lo correcto es `x = seq[:, :-1]`, `y = seq[:, 1:]`.

Merece la pena tenerlo presente porque el síntoma —pérdida sospechosamente baja— es idéntico
tanto si el bug está en el modelo como si está en cómo montas el batch, y lo segundo es más
frecuente.

## Lo que deberías ver en la demo

**El desglose**, que cierra la Parte II:

```
embeddings   1,310,720   14.7%
attention    2,457,600   27.5%
ffn          5,160,960   57.8%
norms            4,160    0.0%
lm_head              0    0.0%   (atada)
TOTAL        8,933,440
```

Fórmula, conteo y objetivo coinciden.

**La comprobación de causalidad** es la más bonita del módulo. Se cambia el token de la
posición 6 y se mira cuánto se mueven los logits:

```
posición 0-5:  0.00e+00     cero exacto
posición 6:    1.46e+00
posición 7-11: ~2.5e-01
```

Cero **exacto**, no pequeño. Las predicciones anteriores no pueden ver ese token de ninguna
manera. Es la verificación más directa que existe de que la máscara causal está bien.

**Y la memoria**, que prepara el módulo 13:

```
pesos + gradientes + AdamW :  143 MB
logits (fp16 + fp32 + grad): 1007 MB
```

El tensor de logits (`48 × 512 × 4096`) ocupa **siete veces más** que el modelo, sus
gradientes y el optimizador juntos. Cuando te quedes sin memoria en la RTX 2060, ese es el
primer sitio donde mirar, no las activaciones del modelo.

---

## El código completo

Si te has atascado, aquí está la implementación entera. **Cópiala, pégala y ejecuta los
tests**: verlos pasar con código que entiendes es mejor que quedarte bloqueado.

Y después vuelve al ejercicio y escríbela tú. Leer una solución que ya has peleado funciona
muy bien; leerla en frío, no funciona nada.

```python
from llmfs.reference import make_ffn, make_norm, sinusoidal_embeddings

def expected_param_count(cfg: ModelConfig) -> int:
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


class TransformerBlock(nn.Module):

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

        mask = None if (use_cache and cache is not None) else causal_mask(seq_len, device=idx.device)
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
```

Los imports que hacen falta ya están en el `exercises.py` del módulo, salvo los que
aparezcan arriba del bloque.
