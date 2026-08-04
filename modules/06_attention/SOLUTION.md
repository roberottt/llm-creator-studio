# 06 — Solución comentada

## Ejercicio 1 — `causal_mask`

```python
return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
```

`tril` = *triangular lower*. Por defecto usa `diagonal=0`, que **incluye** la diagonal, y
eso es lo que quieres: un token sí puede mirarse a sí mismo.

Si te sale al revés, has usado `triu`. Si la diagonal sale a `False`, has pasado
`diagonal=-1`.

**Sobre el convenio `True = permitido`.** Es el que usa
`F.scaled_dot_product_attention` con máscaras booleanas, y por eso lo seguimos. Pero
**`nn.MultiheadAttention` usa el contrario**: su `attn_mask` booleana marca con `True` lo
que hay que *prohibir*. Por eso el test que compara contra PyTorch pasa `~mask`. Es una
inconsistencia real dentro de la propia librería y una fuente clásica de bugs.

## Ejercicio 2 — `single_head_attention`

```python
d_k = q.shape[-1]
scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
if mask is not None:
    scores = scores.masked_fill(~mask, float("-inf"))
weights = F.softmax(scores, dim=-1)
return weights @ v, weights
```

Cuatro líneas, y cada una tiene una trampa.

**`transpose(-2, -1)` y no `transpose(1, 2)`.** Los índices negativos cuentan desde el
final, así que funcionan igual con `(B, T, d)` que con `(B, heads, T, d)`. Si escribes
índices positivos, el ejercicio 2 pasa y el ejercicio 3 falla con un error de formas que
cuesta relacionar con la causa.

**`dim=-1` en el softmax.** Estás normalizando sobre *a quién se mira*, de forma que cada
fila sume 1. Con `dim=-2` normalizarías sobre *quién mira*, que no significa nada. Y no da
error: las formas son idénticas, el modelo entrena, y aprende peor. El test
`test_cada_fila_de_pesos_suma_uno` es lo que lo caza.

**`masked_fill(~mask, -inf)` antes del softmax.** El `~` invierte el booleano: donde la
máscara dice `False` (prohibido), pon `-inf`. Como $e^{-\infty} = 0$, el softmax le asigna
peso exactamente cero.

Tiene que ir **antes**. Si borraras los pesos después del softmax, las filas dejarían de
sumar 1 y estarías escalando la salida por un factor arbitrario distinto en cada posición.

**El `/ math.sqrt(d_k)`.** La demo lo mide: con $d_k = 2048$ y sin escalar, la entropía de
la atención cae a 0,007 nats (de un máximo de 2,77) y el peso máximo llega a 1,00. El token
se fija en uno solo e ignora todo lo demás.

Pero lo grave no es el forward, es el gradiente. La derivada del softmax es $p(1-p)$; con
$p$ pegado a 0 o 1 vale prácticamente cero y la capa deja de aprender. Sin el $\sqrt{d_k}$,
un Transformer grande simplemente no entrena.

## Ejercicio 3 — `MultiHeadAttention`

### Partir y juntar cabezas

```python
def _split_heads(self, x):                     # (B, T, d_model) -> (B, H, T, head_dim)
    B, T, _ = x.shape
    return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

def _merge_heads(self, x):                     # (B, H, T, head_dim) -> (B, T, d_model)
    B, _, T, _ = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, self.d_model)
```

El orden importa. `view(B, T, H, head_dim)` parte la última dimensión en cabezas
**manteniendo** la correspondencia con las posiciones; el `transpose(1, 2)` pone las cabezas
delante para que la atención opere en paralelo sobre ellas.

Si hicieras `view(B, H, T, head_dim)` directamente, estarías mezclando posiciones con
cabezas: el resultado tiene la forma correcta y los datos mal. Es un bug que no da error
nunca. El test `test_mha_no_mezcla_informacion_entre_cabezas` lo detecta comprobando que
las cabezas no dan patrones idénticos.

**El `.contiguous()`.** `transpose` no mueve datos: solo cambia los *strides*, es decir,
cómo se recorre la memoria. `view` exige memoria contigua. Sin el `.contiguous()`, PyTorch
lanza un error que habla de strides y no dice claramente qué hacer. (`reshape` lo haría
solo, pero conviene ver la distinción una vez.)

### El forward

```python
seq_len = x.shape[1]
if mask is None:
    mask = causal_mask(seq_len, device=x.device)

q = self._split_heads(self.q_proj(x))
k = self._split_heads(self.k_proj(x))
v = self._split_heads(self.v_proj(x))

if cos is not None and sin is not None:
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

if self.use_sdpa and not return_weights:
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                         dropout_p=self.dropout if self.training else 0.0)
    weights = None
else:
    scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    out = self.attn_dropout(weights) @ v

out = self.resid_dropout(self.out_proj(self._merge_heads(out)))
return (out, weights) if return_weights else out
```

**RoPE se aplica después de partir en cabezas y solo a Q y K.** La rotación depende de
`head_dim`, no de `d_model`, así que tiene que ir después del split. Y nunca a V: lo que
debe depender de la posición son las *puntuaciones*, no el contenido que se transporta.

**El dropout va en dos sitios distintos.** `attn_dropout` sobre los pesos de atención (antes
de multiplicar por V) y `resid_dropout` sobre la salida de `out_proj`. Con `dropout=0.0`
—que es la config del modelo final— ambos son la identidad, pero la estructura tiene que
estar para el config `tiny_char`, que sí usa dropout.

**El `dropout_p` de SDPA solo en entrenamiento.** `F.scaled_dot_product_attention` no
consulta `self.training` por su cuenta: si le pasas `dropout_p` fijo, aplicará dropout
también en evaluación y tus muestras saldrán ruidosas y no reproducibles.

**Por qué una proyección grande y no 8 pequeñas.** `nn.Linear(320, 320)` seguido de un
`view` es matemáticamente idéntico a ocho `nn.Linear(320, 40)` cuyos resultados se
concatenan. Pero es un matmul grande en vez de ocho pequeños, y como viste en el módulo 01,
las matrices grandes aprovechan mucho mejor la GPU.

## Lo que deberías ver en la demo

**El experimento del escalado**, con la entropía media de la atención (máximo 2,77 nats con
16 posiciones):

| d_k | con escalado | sin escalar |
|---|---|---|
| 8 | 2,51 | 1,63 |
| 128 | 2,32 | 0,08 |
| 2048 | 2,28 | 0,007 |

Con escalado la entropía se mantiene alta pase lo que pase. Sin escalar, colapsa.

**Los heatmaps.** Cuatro cabezas de un modelo entrenado 400 pasos sobre Shakespeare. Tres
cosas que mirar:

1. El triángulo superior está siempre negro. Esa es la máscara causal, y verla es la mejor
   comprobación de que está bien puesta.
2. La diagonal es brillante: casi todos los tokens se prestan mucha atención a sí mismos.
3. Cada cabeza tiene un patrón distinto. En la ejecución de referencia, las distancias
   medias a las que mira cada cabeza salen 2,10 / 3,39 / 2,93 / 4,05 posiciones. **Nadie
   les ha dicho que se especialicen.**

Con un modelo de una capa entrenado 2 segundos ya se ve. En modelos grandes esto llega
mucho más lejos: hay cabezas que emparejan comillas de apertura y cierre, y las *induction
heads*, que detectan el patrón "…A B … A" y predicen B — el mecanismo que se cree
responsable del aprendizaje en contexto.

---

## El código completo

Si te has atascado, aquí está la implementación entera. **Cópiala, pégala y ejecuta los
tests**: verlos pasar con código que entiendes es mejor que quedarte bloqueado.

Y después vuelve al ejercicio y escríbela tú. Leer una solución que ya has peleado funciona
muy bien; leerla en frío, no funciona nada.

```python
from typing import Any

def causal_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def single_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = q.shape[-1]

    # (B, T, d_k) @ (B, d_k, S) -> (B, T, S). scores[b,i,j] = cuanto le interesa a i el j.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        # -inf antes del softmax se convierte en probabilidad 0 despues.
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):

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
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
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
```

Los imports que hacen falta ya están en el `exercises.py` del módulo, salvo los que
aparezcan arriba del bloque.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
from typing import Any

def causal_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


def single_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_k = q.shape[-1]

    # (B, T, d_k) @ (B, d_k, S) -> (B, T, S). scores[b,i,j] = how interested i is in j.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        # -inf before the softmax becomes probability 0 after it.
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadAttention(nn.Module):

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
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
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
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
