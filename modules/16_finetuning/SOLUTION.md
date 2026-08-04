# 16 — Solución comentada

## Ejercicio 1 — `build_chat_template`

```python
partes = []
for mensaje in messages:
    rol = mensaje["role"]
    if rol not in CHAT_MARKERS:
        raise ValueError(f"rol desconocido: {rol!r}")
    partes.append(f"{CHAT_MARKERS[rol]}{mensaje['content']}{CHAT_MARKERS['end']}")

if add_generation_prompt:
    partes.append(CHAT_MARKERS["assistant"])

return "".join(partes)
```

**El `add_generation_prompt` deja la cadena abierta**, sin `<|end|>`. Es lo que se usa en
inferencia: el modelo continúa justo ahí y lo que escriba es la respuesta. Sin esa apertura,
el modelo no sabe que le toca hablar a él.

**El `<|end|>` es lo que le enseña cuándo parar.** Sin un marcador de fin, un modelo generaría
indefinidamente. En inferencia se usa como token de parada.

**Los marcadores no son mágicos:** son texto que el modelo aprende a reconocer durante el
SFT. Cada familia de modelos usa los suyos y son incompatibles entre sí. Usar el template
equivocado degrada bastante la calidad, y es un error sorprendentemente frecuente.

## Ejercicio 2 — `mask_prompt_tokens`

```python
targets = [ignore_index] * len(input_ids)
for i in range(prompt_len - 1, len(input_ids) - 1):
    targets[i] = input_ids[i + 1]
return targets
```

### El off-by-one, que es todo el ejercicio

```
input_ids = [10, 11, 12, 20, 21, 22]      prompt_len = 3
targets   = [-100, -100, 20, 21, 22, -100]
```

**Dos posiciones ignoradas al principio, no tres.** El bucle empieza en `prompt_len - 1`.

La razón: los targets van desplazados un token. En la posición 2 —el **último** token del
prompt— el objetivo ya es `input_ids[3] = 20`, que es el primer token de la respuesta. Y ese
sí interesa: es justo la transición *"se acabó la pregunta, me toca responder"*, que es lo más
importante que el modelo tiene que aprender del SFT.

Si empezaras en `prompt_len`, te saltarías precisamente esa transición. Y no da ninguna señal:
solo desperdicias la posición más informativa del ejemplo.

**La última posición también se ignora** porque no hay `input_ids[i+1]` que predecir.

## Ejercicio 3 — `LoRALinear`

```python
def __init__(self, base_layer, r=8, alpha=16.0, dropout=0.0):
    super().__init__()
    if r <= 0:
        raise ValueError(f"el rango r tiene que ser positivo: {r}")

    self.base = base_layer
    self.r, self.alpha = r, alpha
    self.scaling = alpha / r

    for p in self.base.parameters():
        p.requires_grad = False

    d_in, d_out = base_layer.in_features, base_layer.out_features
    self.lora_A = nn.Parameter(torch.empty(r, d_in))
    self.lora_B = nn.Parameter(torch.zeros(d_out, r))
    self.lora_dropout = nn.Dropout(dropout)

    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

def forward(self, x):
    base = self.base(x)
    adaptacion = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
    return base + adaptacion * self.scaling
```

### La inicialización asimétrica es lo importante

```
A ~ Kaiming
B = ceros
```

Con `B = 0`, el producto `BA` vale cero al empezar y **la capa es exactamente la original**.
El fine-tuning arranca sin perturbar nada. El test
`test_al_inicializar_la_salida_es_identica_a_la_base` lo comprueba directamente.

Si inicializaras las dos al azar, el modelo empezaría degradado y tendría que recuperarse
antes de empezar a mejorar.

**¿Y por qué no las dos a cero?** Porque entonces el gradiente de ambas sería cero —cada una
multiplica a la otra— y nunca aprenderían nada. Hay un test para eso también.

### Congelar la base no es opcional

```python
for p in self.base.parameters():
    p.requires_grad = False
```

Es el punto entero de LoRA. Sin eso estarías entrenando todo *y además* los adaptadores.

### Las transpuestas

`lora_A` es `(r, d_in)`, así que `x @ lora_A.T` da `(..., r)`. Luego `@ lora_B.T` con
`lora_B` de `(d_out, r)` da `(..., d_out)`. Correcto.

## Ejercicio 4 — `merge_lora_weights`

```python
fundida = nn.Linear(d_in, d_out, bias=layer.base.bias is not None)
with torch.no_grad():
    delta = (layer.lora_B @ layer.lora_A) * layer.scaling
    fundida.weight.copy_(layer.base.weight + delta)
    if layer.base.bias is not None:
        fundida.bias.copy_(layer.base.bias)
return fundida
```

**El orden `B @ A`** y no al revés: `B` es `(d_out, r)` y `A` es `(r, d_in)`, así que el
producto da `(d_out, d_in)`, que es exactamente la forma de `weight` en un `nn.Linear`.

**Por qué importa fundir.** Durante el entrenamiento LoRA añade dos matmuls por capa, y eso
se nota en inferencia. Fundido, el modelo es indistinguible de uno normal: mismo coste,
mismas formas, y se puede servir sin ninguna dependencia de LoRA.

Es una ventaja de LoRA frente a otros métodos de fine-tuning eficiente: la adaptación es
**exactamente** una suma de matrices, así que se absorbe sin aproximar nada. El demo lo
verifica: error de $1{,}3 \times 10^{-6}$ entre la capa LoRA y la fundida.

## Un bug que encontré escribiendo la referencia

`apply_lora_to_model` daba **86% de parámetros entrenables** cuando LoRA debería dar ~1%.

La causa: `LoRALinear` congela su propia capa base, pero eso no toca los embeddings, los FFN
ni las normalizaciones, que se quedaban entrenables. Congelar el modelo **entero primero** y
añadir los adaptadores después:

```python
for p in model.parameters():
    p.requires_grad = False
# ... y ahora sí, sustituir las capas objetivo
```

Con eso los números salen como deben:

| r | entrenables | % del modelo |
|---|---|---|
| 4 | 30.720 | **0,34%** |
| 8 | 61.440 | **0,68%** |
| 16 | 122.880 | **1,36%** |

Es un error fácil de cometer y silencioso: el entrenamiento funciona, simplemente no estás
haciendo LoRA.

## Lo que deberías ver en la demo

El SFT sobre el modelo de Shakespeare, con 96 ejemplos y 150 pasos:

```
ANTES:    Q: Who is the king?
          A:
          I have the courtesy? I do hear thee a dealth,
          Company, but

DESPUÉS:  Q: Who is the king?
          A:
          I say we must go.

          MARIANA:
          The castle.
```

**Lo que hay que mirar no es si la respuesta es correcta.** Con 0,8M de parámetros y 96
ejemplos, no lo va a ser: *"I say we must go"* es literalmente una de las respuestas del
conjunto de entrenamiento, memorizada.

Lo que hay que mirar es el **formato**: antes divagaba en Shakespeare indefinidamente,
después produce una respuesta corta. El post-entrenamiento ha hecho su trabajo.

Y ésa es la lección del módulo: **el post-entrenamiento no añade conocimiento**. Saca a la
superficie un comportamiento que ya estaba latente. Un modelo que no sabe algo tras el
pretraining no lo aprende con mil ejemplos de conversación.

---

## El código completo

Si te has atascado, aquí está la implementación entera. **Cópiala, pégala y ejecuta los
tests**: verlos pasar con código que entiendes es mejor que quedarte bloqueado.

Y después vuelve al ejercicio y escríbela tú. Leer una solución que ya has peleado funciona
muy bien; leerla en frío, no funciona nada.

```python
def build_chat_template(
    messages: Sequence[dict[str, str]], add_generation_prompt: bool = False
) -> str:
    partes: list[str] = []
    for mensaje in messages:
        rol = mensaje["role"]
        if rol not in CHAT_MARKERS:
            raise ValueError(f"rol desconocido: {rol!r}. Validos: system, user, assistant")
        partes.append(f"{CHAT_MARKERS[rol]}{mensaje['content']}{CHAT_MARKERS['end']}")

    if add_generation_prompt:
        partes.append(CHAT_MARKERS["assistant"])

    return "".join(partes)


def mask_prompt_tokens(
    input_ids: Sequence[int], prompt_len: int, ignore_index: int = -100
) -> list[int]:
    if prompt_len < 1:
        raise ValueError("prompt_len tiene que ser al menos 1")
    if prompt_len > len(input_ids):
        raise ValueError(
            f"prompt_len ({prompt_len}) mayor que la secuencia ({len(input_ids)})"
        )

    targets = [ignore_index] * len(input_ids)
    for i in range(prompt_len - 1, len(input_ids) - 1):
        targets[i] = input_ids[i + 1]
    return targets


class LoRALinear(nn.Module):

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"el rango r tiene que ser positivo: {r}")

        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # La capa base se congela: es el punto de LoRA.
        for p in self.base.parameters():
            p.requires_grad = False

        d_in, d_out = base_layer.in_features, base_layer.out_features
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        self.lora_dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B se queda en ceros: al arrancar, la capa es identica a la original.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        adaptacion = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + adaptacion * self.scaling


def merge_lora_weights(layer: LoRALinear) -> nn.Linear:
    d_in = layer.base.in_features
    d_out = layer.base.out_features
    fundida = nn.Linear(d_in, d_out, bias=layer.base.bias is not None)

    with torch.no_grad():
        delta = (layer.lora_B @ layer.lora_A) * layer.scaling
        fundida.weight.copy_(layer.base.weight + delta)
        if layer.base.bias is not None:
            fundida.bias.copy_(layer.base.bias)

    return fundida
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
def build_chat_template(
    messages: Sequence[dict[str, str]], add_generation_prompt: bool = False
) -> str:
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        if role not in CHAT_MARKERS:
            raise ValueError(f"unknown role: {role!r}. Valid ones: system, user, assistant")
        parts.append(f"{CHAT_MARKERS[role]}{message['content']}{CHAT_MARKERS['end']}")

    if add_generation_prompt:
        parts.append(CHAT_MARKERS["assistant"])

    return "".join(parts)


def mask_prompt_tokens(
    input_ids: Sequence[int], prompt_len: int, ignore_index: int = -100
) -> list[int]:
    if prompt_len < 1:
        raise ValueError("prompt_len has to be at least 1")
    if prompt_len > len(input_ids):
        raise ValueError(
            f"prompt_len ({prompt_len}) is larger than the sequence ({len(input_ids)})"
        )

    targets = [ignore_index] * len(input_ids)
    for i in range(prompt_len - 1, len(input_ids) - 1):
        targets[i] = input_ids[i + 1]
    return targets


class LoRALinear(nn.Module):

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"the rank r has to be positive: {r}")

        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # The base layer is frozen: that is the whole point of LoRA.
        for p in self.base.parameters():
            p.requires_grad = False

        d_in, d_out = base_layer.in_features, base_layer.out_features
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))
        self.lora_dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays at zeros: at the start, the layer is identical to the original.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        adaptation = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + adaptation * self.scaling


def merge_lora_weights(layer: LoRALinear) -> nn.Linear:
    d_in = layer.base.in_features
    d_out = layer.base.out_features
    merged = nn.Linear(d_in, d_out, bias=layer.base.bias is not None)

    with torch.no_grad():
        delta = (layer.lora_B @ layer.lora_A) * layer.scaling
        merged.weight.copy_(layer.base.weight + delta)
        if layer.base.bias is not None:
            merged.bias.copy_(layer.base.bias)

    return merged
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
