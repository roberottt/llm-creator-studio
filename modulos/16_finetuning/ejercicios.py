"""Modulo 16 - Post-training: SFT y LoRA.

Cuatro ejercicios. Los dos primeros son de formato; los dos ultimos son LoRA.

    llmfs check 16
    llmfs demo 16     hace SFT de verdad sobre el modelo del modulo 13 y compara antes/despues
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

# Los marcadores del chat template. Delimitan los turnos para que el modelo aprenda DONDE
# empieza y acaba cada intervencion.
from llmfs.reference import CHAT_MARKERS


def build_chat_template(
    messages: Sequence[dict[str, str]], add_generation_prompt: bool = False
) -> str:
    """Serializa una conversacion a texto plano con marcadores.

    QUE ES ESTO
        Un modelo preentrenado solo sabe continuar texto. Si le escribes una pregunta, lo
        mas probable es que responda con mas preguntas: un documento que empieza asi suele
        seguir asi.

        Para que RESPONDA hay que enseñarle un FORMATO. Eso son los marcadores.

    EL FORMATO

        [{"role": "user", "content": "Hola"},
         {"role": "assistant", "content": "Que tal"}]

        ->  <|user|>Hola<|end|><|assistant|>Que tal<|end|>

        Cada mensaje se envuelve en el marcador de su rol y se cierra con `<|end|>`. Los
        marcadores estan en `CHAT_MARKERS`, que ya tienes importado.

    EL `add_generation_prompt`
        Con `True`, se anyade `<|assistant|>` al final y se deja la cadena ABIERTA:

            <|user|>Hola<|end|><|assistant|>

        Es lo que se usa en inferencia: el modelo continua justo ahi y lo que escriba es la
        respuesta. Sin esa apertura, el modelo no sabe que le toca hablar a el.

    POR QUE IMPORTA EL `<|end|>`
        Es lo que le enseña al modelo CUANDO PARAR. Sin un marcador de fin, un modelo
        generaria indefinidamente. En inferencia se usa como token de parada.

    LOS MARCADORES NO SON MAGICOS
        Son texto que el modelo aprende a reconocer durante el SFT. Cada familia de modelos
        usa los suyos y son incompatibles entre si: usar el template equivocado con un
        modelo degrada bastante su calidad, y es un error sorprendentemente frecuente.

    Args:
        messages: lista de `{"role": ..., "content": ...}`. Roles validos: los de
            `CHAT_MARKERS` menos "end".
        add_generation_prompt: dejar la cadena abierta en `<|assistant|>`.

    Returns:
        La conversacion serializada.

    Raises:
        ValueError: si algun rol no esta en `CHAT_MARKERS`. Mejor un error claro que
            generar texto con un marcador inventado que el modelo no ha visto nunca.
    """
    raise NotImplementedError("TODO: modulo 16, ejercicio 1 - build_chat_template")


def mask_prompt_tokens(
    input_ids: Sequence[int], prompt_len: int, ignore_index: int = -100
) -> list[int]:
    """Construye los targets ignorando el prompt: solo se aprende de la RESPUESTA.

    LA IDEA
        En SFT no quieres que el modelo aprenda a GENERAR las preguntas del usuario:
        quieres que aprenda a RESPONDERLAS.

        Poniendo `-100` en las posiciones del prompt, `F.cross_entropy(...,
        ignore_index=-100)` las salta y no contribuyen a la perdida.

    EL EJEMPLO, y leelo con cuidado

        input_ids = [10, 11, 12, 20, 21, 22]     con prompt_len = 3
        targets   = [-100, -100, 20, 21, 22, -100]

        Hay DOS posiciones ignoradas al principio, no tres. Y una al final.

    POR QUE DOS Y NO TRES
        Los targets van DESPLAZADOS un token, como siempre en este curso: en la posicion
        `i` el objetivo es `input_ids[i+1]`.

        Asi que en la posicion 2 (el ULTIMO token del prompt) el objetivo ya es
        `input_ids[3] = 20`, que es el primer token de la RESPUESTA. Y ese si interesa: es
        justo la transicion de "se acabo la pregunta" a "empieza mi respuesta", que es lo
        mas importante que tiene que aprender.

        Ese off-by-one es EL error del ejercicio, y no da ninguna senyal: solo desperdicia
        (o aprovecha de mas) una posicion.

    POR QUE LA ULTIMA TAMBIEN SE IGNORA
        En la ultima posicion no hay `input_ids[i+1]`: no hay nada que predecir.

    LA ESTRUCTURA
        targets = [ignore_index] * len(input_ids)
        para i desde prompt_len - 1 hasta len(input_ids) - 2:
            targets[i] = input_ids[i + 1]

    Args:
        input_ids: la secuencia completa, prompt y respuesta juntos.
        prompt_len: cuantos tokens ocupa el prompt.
        ignore_index: el valor que `cross_entropy` ignora. -100 es la convencion de PyTorch.

    Returns:
        La lista de targets, de la misma longitud que `input_ids`.

    Raises:
        ValueError: si `prompt_len` es menor que 1 o mayor que la secuencia.
    """
    raise NotImplementedError("TODO: modulo 16, ejercicio 2 - mask_prompt_tokens")


class LoRALinear(nn.Module):
    """Una capa lineal con adaptadores de rango bajo.

    LA IDEA (Hu et al. 2021)
        Hacer fine-tuning completo de un modelo grande necesita memoria para los pesos, los
        gradientes Y los estados de Adam: unos 12 bytes por parametro.

        LoRA parte de una observacion: los cambios que hace el fine-tuning tienen RANGO
        BAJO. No hace falta poder modificar la matriz en cualquier direccion; bastan unas
        pocas.

        Asi que se CONGELA W y se le suma el producto de dos matrices flacas:

            salida = x @ W^T  +  (alpha/r) * x @ A^T @ B^T

        con A de (r, d_in) y B de (d_out, r), y r pequenyo (4, 8, 16).

    LA ARITMETICA
        Con d_in = d_out = 320 y r = 8:

            W entera:  320 x 320       = 102.400 parametros
            A y B:     8x320 + 320x8   =   5.120 parametros    (el 5%)

        Aplicado al GPT de 9M solo en q_proj y v_proj, con r=8: 61.440 parametros
        entrenables, o sea el 0,68% del modelo.

    LA INICIALIZACION, QUE NO ES SIMETRICA Y ES LO IMPORTANTE

            self.lora_A -> nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_B -> CEROS

        Con B = 0, el producto BA vale cero al empezar y la capa es EXACTAMENTE la
        original. El fine-tuning arranca sin perturbar nada.

        Si inicializaras las dos al azar, el modelo empezaria degradado y tendria que
        recuperarse antes de empezar a mejorar. Hay un test que comprueba que al construir
        la capa, su salida es identica a la de la base.

    NO OLVIDES CONGELAR LA BASE
        for p in self.base.parameters():
            p.requires_grad = False

        Es el punto de LoRA. Sin eso estarias entrenando todo y ademas los adaptadores.

    LA ESCALA `alpha/r`
        Existe para que cambiar `r` no obligue a reajustar el learning rate. Guardala en
        `self.scaling`.

    SUBMODULOS (respeta los nombres)
        base:         la capa original, congelada
        lora_A:       nn.Parameter de forma (r, d_in)
        lora_B:       nn.Parameter de forma (d_out, r), a ceros
        lora_dropout: nn.Dropout(dropout)

    __init__(self, base_layer, r=8, alpha=16.0, dropout=0.0)
        Valida que `r` sea positivo y lanza ValueError si no.
        Las dimensiones salen de `base_layer.in_features` y `base_layer.out_features`.

    forward(self, x):
        return self.base(x) + self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

        Cuidado con las transpuestas: `lora_A` es (r, d_in), asi que `x @ lora_A.T` da
        (..., r), y luego `@ lora_B.T` con lora_B de (d_out, r) da (..., d_out). Correcto.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        raise NotImplementedError("TODO: modulo 16, ejercicio 3 - LoRALinear.__init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TODO: modulo 16, ejercicio 3 - LoRALinear.forward")


def merge_lora_weights(layer: LoRALinear) -> nn.Linear:
    """Funde los adaptadores en la matriz base y devuelve un `nn.Linear` normal.

    LA FORMULA

        W_nueva = W + (alpha/r) * (B @ A)

        Fijate en el orden: `B @ A` con B de (d_out, r) y A de (r, d_in) da (d_out, d_in),
        que es justo la forma de `weight` en un `nn.Linear`. Si lo pusieras al reves las
        formas no cuadrarian.

    POR QUE IMPORTA
        Durante el entrenamiento, LoRA anyade dos matmuls por capa, y eso se nota en
        inferencia. Fundiendo los pesos, el modelo resultante es INDISTINGUIBLE de uno
        normal: mismo coste, mismas formas, y se puede servir sin ninguna dependencia de
        LoRA.

        Es una ventaja de LoRA frente a otros metodos de fine-tuning eficiente: la
        adaptacion es EXACTAMENTE una suma de matrices, asi que se absorbe sin aproximar
        nada.

    LA COMPROBACION
        La capa fundida tiene que dar la MISMA salida que la capa LoRA, hasta el error de
        coma flotante. Hay un test que lo verifica.

    DETALLES
        - Hazlo dentro de `with torch.no_grad():`.
        - Conserva el sesgo si lo habia: `bias=layer.base.bias is not None`, y copialo.
        - `fundida.weight.copy_(...)` en vez de asignar: asi conservas el tensor que ya
          creo `nn.Linear` con sus metadatos.

    Returns:
        Un `nn.Linear` normal con los pesos ya fundidos.
    """
    raise NotImplementedError("TODO: modulo 16, ejercicio 4 - merge_lora_weights")
