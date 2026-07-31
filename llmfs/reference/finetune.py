"""Referencia del modulo 16: post-training (SFT y LoRA)."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Los marcadores del chat template. Delimitan los turnos para que el modelo aprenda
#: DONDE empieza y acaba cada intervencion. Sin ellos no puede saber cuando parar.
CHAT_MARKERS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "end": "<|end|>",
}


def build_chat_template(
    messages: Sequence[dict[str, str]], add_generation_prompt: bool = False
) -> str:
    """Serializa una conversacion a texto plano con marcadores.

    Un modelo preentrenado solo sabe continuar texto. Para que responda en vez de seguir
    escribiendo, hay que enseñarle un FORMATO: marcadores que delimitan los turnos.

    Ejemplo:

        [{"role": "user", "content": "Hola"},
         {"role": "assistant", "content": "Que tal"}]

        ->  <|user|>Hola<|end|><|assistant|>Que tal<|end|>

    `add_generation_prompt=True` deja la cadena abierta en `<|assistant|>` para que el
    modelo continue. Es lo que se usa en inferencia.

    Los marcadores no tienen nada de magico: son texto que el modelo aprende a reconocer
    durante el SFT. Cada familia de modelos usa los suyos, y son incompatibles entre si.
    """
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
    """Construye los targets ignorando el prompt: solo se aprende de la RESPUESTA.

    LA IDEA. En SFT no quieres que el modelo aprenda a generar las preguntas del usuario:
    quieres que aprenda a responderlas. Poniendo `-100` en las posiciones del prompt,
    `F.cross_entropy(..., ignore_index=-100)` las salta.

    Ejemplo con `prompt_len=3`:

        input_ids = [10, 11, 12, 20, 21, 22]
        targets   = [-100, -100, 20, 21, 22, -100]

    Fijate en DOS cosas:

    1. Los targets van DESPLAZADOS un token, como siempre: en la posicion `i` el objetivo
       es `input_ids[i+1]`.
    2. Por eso hay `prompt_len - 1` posiciones ignoradas al principio, no `prompt_len`: en
       la posicion `prompt_len - 1` (el ultimo token del prompt) el objetivo ya es el
       primer token de la respuesta, y ESE si interesa.

    Ese off-by-one es el error tipico del ejercicio, y no da ningun error visible: solo
    desperdicia (o aprovecha de mas) una posicion.
    """
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
    """Una capa lineal con adaptadores de rango bajo.

    LA IDEA (Hu et al. 2021). En vez de entrenar la matriz `W` de d_in x d_out entera, se
    congela y se le suma el producto de DOS matrices flacas:

        salida = x @ W^T  +  (alpha/r) * x @ A^T @ B^T

    con `A` de r x d_in y `B` de d_out x r, y `r` pequenyo (4, 8, 16).

    LA ARITMETICA que lo justifica. Con d_in = d_out = 320 y r = 8:

        W entera :  320 x 320       = 102.400 parametros
        A y B    :  8x320 + 320x8   =   5.120 parametros    (el 5%)

    Se entrena el 5% de los parametros, y como el gradiente solo tiene que calcularse para
    esas dos matrices, el estado del optimizador tambien es un 5%. En modelos grandes eso
    es la diferencia entre necesitar 8 GPUs o una.

    LA INICIALIZACION IMPORTA Y NO ES SIMETRICA:
        A ~ normal (Kaiming)
        B = CEROS

    Con `B = 0`, el producto `BA` vale cero al empezar y la capa es EXACTAMENTE la original.
    El fine-tuning arranca sin perturbar nada. Si inicializaras las dos al azar, el modelo
    empezaria degradado y tendria que recuperarse antes de mejorar.

    La escala `alpha/r` existe para que cambiar `r` no obligue a reajustar el learning rate.
    """

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
    """Funde los adaptadores en la matriz base y devuelve un `nn.Linear` normal.

        W_nueva = W + (alpha/r) * B @ A

    POR QUE IMPORTA. Durante el entrenamiento, LoRA anyade dos matmuls por capa, y eso se
    nota en inferencia. Fundiendo los pesos, el modelo resultante es indistinguible de uno
    normal: mismo coste, mismas formas, y se puede servir sin ninguna dependencia de LoRA.

    Es una de las ventajas de LoRA frente a otros metodos de fine-tuning eficiente: la
    adaptacion es EXACTAMENTE una suma de matrices, asi que se puede absorber.

    La salida tiene que coincidir con la de la capa LoRA hasta el error de coma flotante.
    """
    d_in = layer.base.in_features
    d_out = layer.base.out_features
    fundida = nn.Linear(d_in, d_out, bias=layer.base.bias is not None)

    with torch.no_grad():
        delta = (layer.lora_B @ layer.lora_A) * layer.scaling
        fundida.weight.copy_(layer.base.weight + delta)
        if layer.base.bias is not None:
            fundida.bias.copy_(layer.base.bias)

    return fundida


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    target_names: Sequence[str] = ("q_proj", "v_proj"),
) -> nn.Module:
    """Sustituye las capas objetivo por versiones con LoRA. No es un ejercicio.

    Por defecto solo `q_proj` y `v_proj`, que es lo que hace el paper original: son las que
    mejor relacion dan entre parametros entrenados y calidad.

    OJO CON EL ORDEN, que es donde esta el unico detalle no obvio: primero se congela el
    modelo ENTERO y despues se anyaden los adaptadores. `LoRALinear` congela su propia capa
    base, pero eso no toca los embeddings, los FFN ni las normalizaciones, que se quedarian
    entrenables y arruinarian el proposito: con el GPT de 9M pasarias de entrenar el 1% a
    entrenar el 86%.
    """
    for p in model.parameters():
        p.requires_grad = False

    for nombre, modulo in list(model.named_modules()):
        for hijo_nombre, hijo in list(modulo.named_children()):
            if hijo_nombre in target_names and isinstance(hijo, nn.Linear):
                setattr(modulo, hijo_nombre, LoRALinear(hijo, r=r, alpha=alpha))
    return model


def count_trainable(model: nn.Module) -> dict[str, int]:
    """Cuantos parametros se entrenan de verdad frente al total."""
    entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "entrenables": entrenables,
        "total": total,
        "porcentaje": round(100 * entrenables / max(1, total), 4),
    }
