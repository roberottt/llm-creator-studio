"""Modulo 16 - Post-training: SFT y LoRA.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa -> `llmfs check 16` -> `llmfs hint 16 -e N`
-> `SOLUCION.md` tiene el codigo completo.

QUÉ VAS A CONSTRUIR
===================

Como convertir un modelo que continua texto en uno que responde:

    build_chat_template  (ej. 1)  el formato que le ensenya donde empieza cada turno
    mask_prompt_tokens   (ej. 2)  que aprenda a RESPONDER, no a preguntar
    LoRALinear           (ej. 3)  entrenar el 0,7% de los parametros
    merge_lora_weights   (ej. 4)  fundir los cambios sin dejar rastro

Los dos primeros son de formato y son cortos. Los dos ultimos son LoRA.

EL PROBLEMA
===========

Escribele a tu modelo entrenado "¿Cual es la capital de Francia?" y lo mas probable es que
responda con MAS preguntas. No esta roto: esta haciendo exactamente lo que le ensenyaste,
que es continuar texto plausible.

`TEORIA.md` sigue este mismo orden y cada docstring de aqui te dice que seccion le toca. Los
ejercicios van en DOS BLOQUES independientes: 1 y 2 son el SFT (uno da formato, el otro decide de
que se aprende) y 3 y 4 son LoRA. Puedes hacer SFT sin LoRA y LoRA sin SFT; se combinan porque en
la practica es lo que se hace.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **pretraining**: la fase larga, aprender lenguaje prediciendo el siguiente token.
- **post-training / SFT**: seguir entrenando sobre ejemplos de instruccion y respuesta.
- **chat template**: los marcadores (`<|user|>`, `<|end|>`) que delimitan los turnos.
- **ignore_index**: el valor (-100) que hace que `cross_entropy` salte una posicion sin
  contarla en la perdida.
- **LoRA**: entrenar dos matrices pequenyas anyadidas al modelo en vez de todos sus pesos.
- **rango** (r) de LoRA: la dimension interna de esas matrices. Tipicamente 4, 8 o 16.
- **congelar** un parametro: ponerle `requires_grad = False` para que no se entrene.

    llmfs demo 16     hace SFT de verdad y compara el antes y el despues
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

    Contexto en `TEORIA.md`: seccion "Ejercicio 1: el chat template", con la diferencia entre
    la version de entrenamiento y la de inferencia, que es de donde sale el flag
    add_generation_prompt.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Un bucle y un `join`.

        1. La lista donde acumular:

               partes = []

        2. Un mensaje cada vez, validando el rol:

               for mensaje in messages:
                   rol = mensaje["role"]
                   if rol not in CHAT_MARKERS:
                       raise ValueError(f"rol desconocido: {rol!r}. Validos: system, user, assistant")
                   partes.append(
                       f"{CHAT_MARKERS[rol]}{mensaje['content']}{CHAT_MARKERS['end']}"
                   )

        3. La apertura para la respuesta, si se pide:

               if add_generation_prompt:
                   partes.append(CHAT_MARKERS["assistant"])

        4. `return "".join(partes)`

    Fijate en que se une con `""` y no con espacios ni saltos de linea: los marcadores ya
    separan, y cualquier caracter extra seria uno mas que el modelo tiene que aprender a
    predecir.

    QUÉ TIENE QUE SALIR
    -------------------
        [{"role": "user", "content": "Hola"},
         {"role": "assistant", "content": "Que tal"}]

        ->  <|user|>Hola<|end|><|assistant|>Que tal<|end|>

    Y con `add_generation_prompt=True` sobre solo el primer mensaje:

        ->  <|user|>Hola<|end|><|assistant|>

    La cadena queda ABIERTA a proposito.

    QUÉ PROBLEMA RESUELVE
    ---------------------
    Un modelo preentrenado solo sabe continuar texto. Si le escribes "¿Cual es la capital de
    Francia?" lo mas probable es que responda con MAS preguntas: un documento que empieza asi
    suele seguir asi. No esta roto, esta haciendo exactamente lo que le ensenyaste.

    Para que RESPONDA hay que ensenyarle un FORMATO, y eso son los marcadores.

    EL `add_generation_prompt`
    --------------------------
    Es lo que se usa en INFERENCIA. Dejas la cadena abierta en `<|assistant|>`, el modelo
    continua justo ahi, y lo que escriba es la respuesta. Sin esa apertura el modelo no sabe que
    le toca hablar a el, y es bastante probable que genere otro `<|user|>` y se ponga a
    inventarse tu siguiente pregunta.

    En ENTRENAMIENTO va a False: ahi la respuesta del assistant ya esta en los datos.

    POR QUÉ IMPORTA EL `<|end|>`
    ----------------------------
    Es lo que le ensenya al modelo CUANDO PARAR. Sin un marcador de fin, el modelo generaria
    indefinidamente. En inferencia se usa como token de parada (el `eos_token` del modulo 14).

    LOS MARCADORES NO SON MÁGICOS
    -----------------------------
    Son texto normal y corriente que el modelo aprende a reconocer durante el SFT. Cada familia
    de modelos usa los suyos y son incompatibles entre si: usar el template equivocado con un
    modelo degrada bastante su calidad, y es un error sorprendentemente frecuente porque no da
    ningun aviso, solo respuestas peores.

    Args:
        messages: lista de `{"role": ..., "content": ...}`. Roles validos: los de
            `CHAT_MARKERS` menos "end".
        add_generation_prompt: dejar la cadena abierta en `<|assistant|>`.

    Returns:
        La conversacion serializada.

    Raises:
        ValueError: si algun rol no esta en `CHAT_MARKERS`. Mejor un error claro que generar
            texto con un marcador inventado que el modelo no ha visto nunca.
    """
    raise NotImplementedError("TODO: modulo 16, ejercicio 1 - build_chat_template")


def mask_prompt_tokens(
    input_ids: Sequence[int], prompt_len: int, ignore_index: int = -100
) -> list[int]:
    """Construye los targets ignorando el prompt: solo se aprende de la RESPUESTA.

    Contexto en `TEORIA.md`: seccion "Ejercicio 2: enmascarar el prompt", con la tabla posicion a
    posicion donde se ve que hay DOS posiciones ignoradas y no tres, y por que la transicion
    "se acabo la pregunta, me toca hablar" es lo que NO hay que enmascarar.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Cuatro lineas, y el rango del bucle es TODO el ejercicio.

        1. Las validaciones:

               if prompt_len < 1:
                   raise ValueError("prompt_len tiene que ser al menos 1")
               if prompt_len > len(input_ids):
                   raise ValueError(
                       f"prompt_len ({prompt_len}) mayor que la secuencia ({len(input_ids)})"
                   )

        2. Todo ignorado de partida:

               targets = [ignore_index] * len(input_ids)

        3. Rellena solo el tramo que si cuenta:

               for i in range(prompt_len - 1, len(input_ids) - 1):
                   targets[i] = input_ids[i + 1]

        4. `return targets`

    EL EJEMPLO, Y LÉELO CON CUIDADO
    -------------------------------
        input_ids = [10, 11, 12, 20, 21, 22]     con prompt_len = 3
        targets   = [-100, -100, 20, 21, 22, -100]

    Hay DOS posiciones ignoradas al principio, no tres. Y una al final.

    POR QUÉ DOS Y NO TRES
    ---------------------
    Los targets van DESPLAZADOS un token, como siempre en este curso: en la posicion `i` el
    objetivo es `input_ids[i+1]`.

    Asi que en la posicion 2 —el ULTIMO token del prompt— el objetivo ya es `input_ids[3] = 20`,
    que es el PRIMER token de la respuesta. Y ese si interesa muchisimo: es justo la transicion
    de "se acabo la pregunta" a "empieza mi respuesta", lo mas importante que tiene que
    aprender el modelo en todo el SFT.

    De ahi el `prompt_len - 1` del `range`. Ese off-by-one es EL error del ejercicio, y no da
    ninguna senyal: solo desperdicia la posicion mas valiosa que hay.

    POR QUÉ LA ÚLTIMA TAMBIÉN SE IGNORA
    -----------------------------------
    En la ultima posicion no existe `input_ids[i+1]`: no hay nada que predecir. De ahi el
    `len(input_ids) - 1` como final del `range`.

    QUÉ HACE EL -100
    ----------------
    `F.cross_entropy(..., ignore_index=-100)` SALTA esas posiciones: no contribuyen a la
    perdida y no generan gradiente. El -100 es una convencion de PyTorch, no un numero magico;
    podria ser cualquier valor imposible de token.

    Y por eso el `forward` de tu GPT (modulo 10) ya lleva `ignore_index=-100` puesto desde
    entonces. Este es el modulo donde por fin sirve para algo.

    LA IDEA
    -------
    En SFT no quieres que el modelo aprenda a GENERAR las preguntas del usuario: quieres que
    aprenda a RESPONDERLAS. Si contases la perdida sobre el prompt, estarias gastando capacidad
    en aprender a imitar al usuario, que es exactamente lo contrario de lo que buscas.

    Args:
        input_ids: la secuencia completa, prompt y respuesta juntos.
        prompt_len: cuantos tokens ocupa el prompt.
        ignore_index: el valor que `cross_entropy` ignora. -100 es la convencion de PyTorch.

    Returns:
        La lista de targets, de la MISMA longitud que `input_ids`.

    Raises:
        ValueError: si `prompt_len` es menor que 1 o mayor que la secuencia.
    """
    raise NotImplementedError("TODO: modulo 16, ejercicio 2 - mask_prompt_tokens")


class LoRALinear(nn.Module):
    """Una capa lineal con adaptadores de rango bajo.

    Contexto en `TEORIA.md`: seccion "Ejercicio 3: LoRA", con la aritmetica de por que r=8 entrena el
    0,68% del modelo y por que lora_B se inicializa a CEROS y lora_A no.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    **En `__init__`**, seis pasos:

        1. Valida el rango:

               if r <= 0:
                   raise ValueError(f"r tiene que ser positivo: {r}")

        2. Guarda la base y CONGÉLALA. Esto es el punto entero de LoRA:

               self.base = base_layer
               for p in self.base.parameters():
                   p.requires_grad = False

        3. Las dimensiones salen de la propia capa:

               d_in = base_layer.in_features
               d_out = base_layer.out_features

        4. Los dos adaptadores:

               self.lora_A = nn.Parameter(torch.empty(r, d_in))
               self.lora_B = nn.Parameter(torch.zeros(d_out, r))

        5. La inicializacion, que NO es simetrica (ver abajo):

               nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
               # lora_B se queda a CEROS

        6. La escala y el dropout:

               self.r = r
               self.alpha = alpha
               self.scaling = alpha / r
               self.lora_dropout = nn.Dropout(dropout)

    **En `forward`**, una linea:

        return (
            self.base(x)
            + self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        )

    SIGUE LAS FORMAS DE LA LÍNEA DEL `forward`
    ------------------------------------------
        x            (..., d_in)
        @ lora_A.T   con lora_A de (r, d_in), su T es (d_in, r)   ->  (..., r)
        @ lora_B.T   con lora_B de (d_out, r), su T es (r, d_out) ->  (..., d_out)

    Cuadra con la salida de `self.base(x)`, que es `(..., d_out)`. Si te lias con las
    transpuestas, sigue las formas asi: es mas rapido que pensarlo.

    LA INICIALIZACIÓN ASIMÉTRICA ES LO IMPORTANTE DEL EJERCICIO
    ------------------------------------------------------------
    `lora_B` empieza a CEROS. Por tanto `B @ A = 0` al construir la capa, y la salida es
    EXACTAMENTE la de la capa original. El fine-tuning arranca sin perturbar absolutamente nada.

    Si inicializaras las dos al azar, el modelo empezaria degradado y tendria que gastar los
    primeros pasos recuperando lo que ya sabia antes de empezar a mejorar.

    ¿Y por que no las DOS a cero? Porque entonces el gradiente de ambas seria cero para siempre
    (el gradiente de A pasa por B y viceversa) y nunca aprenderian nada. Una a cero rompe la
    simetria, las dos a cero la congelan.

    Hay un test que construye la capa y comprueba que su salida es identica a la de la base.

    LA IDEA (Hu et al. 2021)
    ------------------------
    Hacer fine-tuning completo de un modelo grande necesita memoria para los pesos, los
    gradientes Y los dos estados de Adam: unos 12 bytes por parametro. Con 7B de parametros eso
    son 84 GB, y no cabe en ninguna GPU de consumo.

    LoRA parte de una observacion: los cambios que hace el fine-tuning tienen RANGO BAJO. No
    hace falta poder modificar la matriz en cualquier direccion posible; bastan unas pocas
    direcciones. Asi que se CONGELA W y se le suma el producto de dos matrices flacas:

        salida = x @ W^T  +  (alpha/r) * x @ A^T @ B^T

    con A de `(r, d_in)`, B de `(d_out, r)` y r pequenyo (4, 8, 16).

    LA ARITMÉTICA, CON NUESTROS NÚMEROS
    -----------------------------------
    Con d_in = d_out = 320 y r = 8:

        W entera:  320 x 320        = 102.400 parametros
        A y B:     8x320 + 320x8    =   5.120 parametros     (el 5%)

    Aplicado al GPT de 9M solo en `q_proj` y `v_proj`, con r=8: 61.440 parametros entrenables,
    o sea el 0,68% del modelo.

    LA ESCALA `alpha/r`
    -------------------
    Existe para que cambiar `r` no obligue a reajustar el learning rate. Con r mas alto, `B @ A`
    tiene mas terminos y su magnitud crece; dividir por r lo compensa. Guardala en
    `self.scaling` porque el ejercicio 4 la necesita.

    CONGELAR LA BASE NO ES OPCIONAL
    -------------------------------
    Sin el `requires_grad = False` del paso 2 estarias entrenando el modelo entero Y ademas los
    adaptadores: lo peor de los dos mundos. Y no da ningun error, solo consume la memoria que
    querias ahorrar. Si al aplicar LoRA a tu modelo ves que el 86% de los parametros son
    entrenables en vez del 0,7%, esto es lo que ha pasado.

    SUBMÓDULOS (respeta los nombres, el ejercicio 4 los usa)
        base:         la capa original, congelada
        lora_A:       `nn.Parameter` de forma `(r, d_in)`
        lora_B:       `nn.Parameter` de forma `(d_out, r)`, a ceros
        lora_dropout: `nn.Dropout(dropout)`
        scaling:      el float `alpha / r`

    __init__(self, base_layer, r=8, alpha=16.0, dropout=0.0)
        Raises:
            ValueError: si `r` no es positivo.

    forward(self, x):
        Args:
            x: `(..., d_in)`.
        Returns:
            `(..., d_out)`, la misma forma que daria la capa base.
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

    Contexto en `TEORIA.md`: seccion "Ejercicio 4: fundir los pesos", con el error medido de la
    fusion (1,31e-06, redondeo de fp32) y por que esto es la ventaja de LoRA frente a otros
    metodos de fine-tuning eficiente.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Cuatro lineas.

        1. La capa vacia, con las mismas dimensiones y el mismo sesgo (o su ausencia):

               d_in = layer.base.in_features
               d_out = layer.base.out_features
               fundida = nn.Linear(d_in, d_out, bias=layer.base.bias is not None)

        2. La fusion, dentro de `no_grad`:

               with torch.no_grad():
                   delta = layer.lora_B @ layer.lora_A * layer.scaling
                   fundida.weight.copy_(layer.base.weight + delta)
                   if layer.base.bias is not None:
                       fundida.bias.copy_(layer.base.bias)

        3. `return fundida`

    EL ORDEN `B @ A` NO ES INTERCAMBIABLE
    -------------------------------------
        B es (d_out, r)  y  A es (r, d_in)
        B @ A            ->  (d_out, d_in)

    Que es justo la forma de `weight` en un `nn.Linear`. Al reves (`A @ B`) las formas ni
    siquiera cuadran salvo que d_in == d_out, y en ese caso cuadrarian dando el resultado
    equivocado, que es peor.

    POR QUÉ `copy_` Y NO UNA ASIGNACIÓN
    -----------------------------------
    `fundida.weight.copy_(...)` escribe DENTRO del tensor que ya creo `nn.Linear`, conservando
    su identidad y sus metadatos (que esta registrado como parametro, su `requires_grad`, su
    sitio en el `state_dict`).

    `fundida.weight = ...` con un tensor normal fallaria: `nn.Module` solo acepta `nn.Parameter`
    en ese atributo. Es el mismo tipo de distincion que ya viste con el weight tying del modulo
    10, donde ahi si querias reasignar.

    EL `no_grad`
    ------------
    Estas escribiendo sobre parametros. Sin el, cada `copy_` construiria grafo de autograd que
    no sirve para nada y ademas mantendria vivos los tensores originales en memoria.

    POR QUÉ IMPORTA ESTO
    --------------------
    Durante el entrenamiento, LoRA anyade dos multiplicaciones de matriz por capa, y eso se nota
    en inferencia: mas kernels que lanzar, mas latencia por token.

    Fundiendo los pesos, el modelo resultante es INDISTINGUIBLE de uno normal: mismo coste,
    mismas formas, y se puede servir sin ninguna dependencia del codigo de LoRA.

    Es una ventaja de LoRA frente a otros metodos de fine-tuning eficiente: la adaptacion es
    EXACTAMENTE una suma de matrices, asi que se absorbe sin aproximar nada. No pierdes ni un
    decimal.

    Y tiene una consecuencia practica bonita: puedes guardar varios adaptadores de 60 KB para
    tareas distintas sobre un unico modelo base, y fundir el que toque en cada momento.

    LA COMPROBACIÓN
    ---------------
    La capa fundida tiene que dar la MISMA salida que la capa LoRA, hasta el error de coma
    flotante. Hay un test que lo verifica con `torch.allclose`.

    (Con `dropout > 0` compara en modo `eval()`: en `train()` el dropout aleatoriza la salida de
    la capa LoRA y no habria nada que comparar.)

    Args:
        layer: la capa LoRA ya entrenada.

    Returns:
        Un `nn.Linear` normal con los pesos ya fundidos.
    """
    raise NotImplementedError("TODO: modulo 16, ejercicio 4 - merge_lora_weights")
