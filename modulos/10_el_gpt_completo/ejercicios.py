"""Modulo 10 - El GPT completo.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> haz el ejercicio 1 CON PAPEL antes de escribir codigo -> implementa el
resto -> `llmfs check 10` -> `llmfs hint 10 -e N` -> `SOLUCION.md` tiene el codigo completo.

QUÉ VAS A CONSTRUIR
===================

El modelo que vas a entrenar. Cuatro ejercicios:

    expected_param_count  (ej. 1)  la formula de cuantos parametros tendra
    count_parameters      (ej. 2)  contarlos de verdad, desglosados
    TransformerBlock      (ej. 3)  un bloque: atencion + FFN, con sus residuales
    GPT                   (ej. 4)  el modelo entero

Los dos primeros son de contar y tienen que dar el MISMO numero: 8.933.440. Si no cuadran,
tu formula o tu modelo mienten. Es una auditoria cruzada, no un ejercicio de aritmetica.

Puedes hacerlos en este orden: los tests de los ejercicios 1 y 2 usan el modelo de REFERENCIA,
asi que no hace falta haber escrito el 4 para contar. `TEORIA.md` los sigue en el mismo orden y
cada docstring de aqui te dice que seccion le toca.

Lo unico que conviene hacer de otra manera es el ejercicio 1: HAZLO CON PAPEL antes de teclear.

La seccion "El modelo entero, de un vistazo" es la que conviene tener delante mientras escribes
el ejercicio 4: es el modelo completo con la forma de los tensores en cada punto.

LA ESTRUCTURA
=============

    ids de token
        |  tabla de embeddings
    vectores
        |  bloque x 6
    vectores
        |  normalizacion final
    vectores
        |  proyeccion a logits
    puntuaciones sobre los 4096 tokens

VOCABULARIO QUE VAS A NECESITAR
===============================

- **weight tying**: reutilizar la matriz de embeddings, transpuesta, como capa de salida.
  Ahorra 1,3 millones de parametros.
- **buffer**: un tensor que acompanya al modelo (se mueve con `.to(device)`) pero NO es un
  parametro y no recibe gradiente. Las tablas de RoPE son buffers.
- **inicializacion**: los valores con los que arrancan los pesos antes de entrenar. No es
  un detalle: decide si el modelo entrena bien.
- **logits**: la salida final del modelo, una puntuacion por cada token del vocabulario.
- **causal**: que un token no puede ver a los que vienen despues.

    llmfs demo 10     desglosa los parametros y verifica que el modelo es causal
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmfs.config import ModelConfig

# Las piezas de los modulos 06-09. Si no las has hecho, el bridge usa la referencia y esto
# funciona igual: puedes montar el GPT sin haber terminado los modulos anteriores.
from llmfs.bridge import resolve

MultiHeadAttention = resolve("06_atencion", "MultiHeadAttention")
RMSNorm = resolve("07_normalizacion", "RMSNorm")
SwiGLU = resolve("08_mlp_y_activaciones", "SwiGLU")
rope_frequencies = resolve("09_posicion", "rope_frequencies")
causal_mask = resolve("06_atencion", "causal_mask")


def expected_param_count(cfg: ModelConfig) -> int:
    """El numero de parametros, calculado con la formula en vez de contando.

    Contexto en `TEORIA.md`: seccion "Ejercicios 1 y 2: el conteo exacto", con la tabla del
    desglose termino a termino y los dos errores tipicos identificados por el numero que dan.

    HAZLO CON PAPEL PRIMERO
    -----------------------
    En serio. Coge el desglose del TEORIA.md y escribe la formula a mano. Solo despues la
    traduces a codigo. Si vas directo al codigo acabaras probando numeros hasta que cuadre, y
    eso no ensenya nada.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Una suma. Ve acumulando en una variable `total` y devuelvela.

        1. Los embeddings de token:

               total = cfg.vocab_size * cfg.d_model

        2. Si `cfg.pos == "learned"`, suma tambien `cfg.context_length * cfg.d_model`.
           (Con RoPE no se suma NADA aqui: ver mas abajo.)

        3. Lo de UNA capa, y multiplicalo por `cfg.n_layers`:

               atencion = 4 * cfg.d_model**2                 # Wq, Wk, Wv, Wo
               if cfg.bias:
                   atencion += 4 * cfg.d_model

               if cfg.activation == "swiglu":
                   ffn = 3 * cfg.d_model * cfg.d_ff          # gate, up, down
               else:
                   ffn = 2 * cfg.d_model * cfg.d_ff          # el MLP clasico

               normas = 2 * cfg.d_model                      # dos RMSNorm por bloque
               # si fuese LayerNorm CON sesgo, serian 2 * (2 * cfg.d_model)

               total += cfg.n_layers * (atencion + ffn + normas)

        4. La norma final: `total += cfg.d_model`.

        5. La capa de salida:

               if not cfg.tie_embeddings:
                   total += cfg.vocab_size * cfg.d_model

        6. `return total`

    LA COMPROBACIÓN
    ---------------
    Con el config por defecto (el modelo final) tiene que dar EXACTAMENTE 8.933.440:

        1.310.720                    <- 4096 * 320, los embeddings
        + 6 * (409.600               <- 4 * 320²,   la atencion de un bloque
             + 860.160               <- 3 * 320 * 896, el SwiGLU de un bloque
             +     640)              <- 2 * 320,   las dos normas de un bloque
        +       320                  <- la norma final
        = 8.933.440

    Si te sale 10.244.160 has olvidado el weight tying (esa es la diferencia exacta:
    1.310.720). Si te sale de mas por un pelo, probablemente cuentas sesgos que no existen.

    LO QUE NO CUENTA, Y CONVIENE ENTENDER POR QUÉ
    ---------------------------------------------
    RoPE no aporta NI UN parametro. Sus tablas de cos/sin salen de una formula cerrada y se
    guardan como buffers, no como parametros: nadie las entrena. Si tu cuenta incluye algo de
    RoPE, esta mal.

    PARA QUÉ SIRVE ESTO DE VERDAD
    -----------------------------
    Para disenyar. Cambias `d_model` en el YAML y ves al instante si el modelo te cabe en la
    GPU, sin esperar a construirlo. Y para verificar: si esta formula y el ejercicio 2 no dan
    el mismo numero, o tu formula o tu modelo mienten, y hay que averiguar cual.

    Args:
        cfg: la configuracion del modelo.

    Returns:
        El numero total de parametros, como entero.
    """
    raise NotImplementedError("TODO: modulo 10, ejercicio 1 - expected_param_count")


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Cuenta los parametros de verdad, desglosados por componente.

    Contexto en `TEORIA.md`: seccion "Ejercicios 1 y 2: el conteo exacto", con el desglose que
    tiene que salir (embeddings 14,7%, atencion 27,5%, FFN 57,8%) y por que el weight tying
    obliga al `set` de ids.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Un recorrido por `named_parameters()` clasificando por el nombre.

        1. El dict de salida, a cero:

               out = {"embeddings": 0, "attention": 0, "ffn": 0,
                      "norms": 0, "lm_head": 0, "other": 0}

        2. El recorrido, saltandote los tensores ya vistos:

               vistos = set()
               for name, param in model.named_parameters():
                   if id(param) in vistos:
                       continue
                   vistos.add(id(param))
                   n = param.numel()

        3. Dentro del bucle, clasifica mirando SUBCADENAS del nombre. Los nombres son del
           estilo `blocks.3.attn.q_proj.weight`, asi que basta con `in`:

               "token_embedding" o "pos_embedding"          -> out["embeddings"] += n
               "attn."                                      -> out["attention"]  += n
               "gate_proj"/"up_proj"/"down_proj"/"fc_"      -> out["ffn"]        += n
               "norm"                                       -> out["norms"]      += n
               "lm_head"                                    -> out["lm_head"]    += n
               lo que no encaje en nada                     -> out["other"]      += n

           Cuidado con el ORDEN de los `if/elif`: `attn.` tiene que comprobarse antes que
           `norm`, porque `blocks.0.attn_norm.weight` contiene las dos cosas y quieres que
           cuente como norma. Ordena de mas especifico a mas general y compruebalo.

        4. Los dos totales:

               out["total"] = sum(v for k, v in out.items())    # antes de anyadir estos dos
               out["non_embedding"] = out["total"] - out["embeddings"]
               return out

    ANTES DE ESCRIBIR NADA, IMPRIME LOS NOMBRES
    -------------------------------------------
        print([n for n, _ in model.named_parameters()])

    Vale mucho la pena: en treinta segundos ves como esta montado el modelo entero y sabes
    exactamente que subcadenas hay que buscar. No adivines.

    EL WEIGHT TYING Y EL `set` DE ids
    ---------------------------------
    Con `tie_embeddings=True`, `lm_head.weight` y `token_embedding.weight` son EL MISMO tensor,
    no dos copias. Aparece bajo dos nombres distintos.

    Dato que contradice la creencia habitual: tanto `parameters()` como `named_parameters()`
    DEDUPLICAN por identidad por defecto (`remove_duplicate=True`), asi que el total sale bien
    aunque no hagas nada.

    Aun asi lleva el `set` de `id(param)`. Dos motivos: deja explicito que sabes que hay pesos
    compartidos, y protege el desglose si algun dia recorres con `remove_duplicate=False`. Es
    una linea y evita un error de 1.310.720 parametros.

    Y ojo con una trampa de Python: `if param in vistos` NO vale. El operador `in` usa `==`,
    que en tensores es elemento a elemento y revienta con "Boolean value of Tensor is
    ambiguous". Por eso se compara por `id()`.

    QUÉ ES `non_embedding` Y POR QUÉ SE DEVUELVE APARTE
    ---------------------------------------------------
    Es `total - embeddings`. Ese es el numero que usan las leyes de escala del modulo 12,
    porque los embeddings escalan de forma distinta al resto del modelo: crecen con el
    vocabulario, no con la profundidad, y no participan del computo por token igual que las
    capas.

    Args:
        model: el modelo ya construido.

    Returns:
        Un dict con las claves `embeddings`, `attention`, `ffn`, `norms`, `lm_head`, `other`,
        `total` y `non_embedding`.
    """
    raise NotImplementedError("TODO: modulo 10, ejercicio 2 - count_parameters")


class TransformerBlock(nn.Module):
    """Un bloque: atencion y FFN, cada uno con su normalizacion y su residual.

    Contexto en `TEORIA.md`: seccion "Ejercicio 3: el bloque", con lo que dicen de verdad esas
    dos lineas y los cuatro sitios donde se falla, que son los cuatro silenciosos.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    **En `__init__`** (cuatro lineas ademas del `super()`). Los nombres importan: el test copia
    pesos por nombre, y el ejercicio 2 clasifica por nombre.

        from llmfs.reference import make_norm, make_ffn

        self.attn_norm = make_norm(cfg)
        self.attn = MultiHeadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, bias=cfg.bias
        )
        self.ffn_norm = make_norm(cfg)
        self.ffn = make_ffn(cfg)

    `make_norm` y `make_ffn` ya estan hechos y miran `cfg.norm` y `cfg.activation` para decidir
    si construyen RMSNorm o LayerNorm, SwiGLU o el MLP clasico. No los reimplementes.

    **En `forward`** (dos lineas y un return):

        x = x + self.attn(self.attn_norm(x), cos=cos, sin=sin, mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    Y ya esta. El bloque entero son dos lineas.

    LO QUE ESTÁN DICIENDO ESAS DOS LÍNEAS
    -------------------------------------
    Cada una es "pre-norm + residual" (modulo 07): normaliza, calcula algo, y SUMA el resultado
    a lo que ya habia. La `x` de la izquierda del `+` es la corriente residual, la autopista que
    recorre el modelo de arriba abajo sin que nada la interrumpa.

    La atencion MUEVE informacion entre tokens. El FFN la PROCESA token a token. Alternan, y
    esa alternancia es todo el Transformer.

    Los dos residuales son independientes a proposito: cada sub-bloque puede aportar poco o
    mucho sin condicionar al otro.

    CUATRO SITIOS DONDE SE FALLA
    ----------------------------
    **Normalizar la corriente en vez de la rama.** Es `x + attn(norm(x))`, no `norm(x + attn(x))`.
    Lo segundo es post-norm y rompe la propiedad que hace que el gradiente llegue limpio a la
    capa 1. Entrena, pero peor y con mas warmup.

    **Reutilizar la misma norma para los dos sub-bloques.** Son DOS objetos distintos con pesos
    propios. `self.ffn_norm = self.attn_norm` compila y da resultados plausibles, y esta mal.

    **Olvidar pasarle `cos`, `sin` y `mask` a la atencion.** Sin `cos`/`sin` el modelo pierde
    toda la informacion posicional y aun asi entrena (mal). Sin `mask`, cada token ve el futuro
    y la perdida baja de forma sospechosamente buena. Los dos fallos son silenciosos.

    **Pasarle `cos`/`sin`/`mask` al FFN.** No los acepta, y es que no los necesita: el FFN no
    mira a otros tokens, asi que no hay nada que enmascarar ni ninguna posicion que inyectar.

    forward(self, x, cos=None, sin=None, mask=None):
        Args:
            x: `(B, T, d_model)`.
            cos, sin: tablas de RoPE, o None.
            mask: la mascara causal, o None para que la genere la atencion.
        Returns:
            `(B, T, d_model)`, exactamente la misma forma que entro.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError("TODO: modulo 10, ejercicio 3 - TransformerBlock.__init__")

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("TODO: modulo 10, ejercicio 3 - TransformerBlock.forward")


class GPT(nn.Module):
    """El modelo completo. 8.933.440 parametros cuando termines.

    Contexto en `TEORIA.md`: seccion "Ejercicio 4: el modelo entero", que va por las cuatro
    partes del __init__ en el mismo orden en que se escriben, explicando cada decision de disenyo
    donde toca escribirla: el weight tying en la parte 2, que es un buffer y por que RoPE vive en
    uno en la parte 3, y la inicializacion escalada en la parte 4. Y antes de esa seccion, "El
    modelo entero, de un vistazo": el recorrido de (B, T) a (B, T, 4096) con las formas, que es lo
    que conviene tener delante mientras escribes esto.

    LA ESTRUCTURA
    -------------
        ids -> embeddings -> [bloque] x n_layers -> norma final -> logits

    Con RoPE no hay embedding posicional que sumar al principio: la posicion se inyecta dentro
    de la atencion. Por eso la primera capa es solo la tabla de tokens.

    QUÉ TIENES QUE ESCRIBIR EN `__init__`
    -------------------------------------
        1. Guarda el config y crea los submodulos. Los nombres importan:

               self.cfg = cfg
               self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
               if cfg.pos == "learned":
                   self.pos_embedding = nn.Embedding(cfg.context_length, cfg.d_model)
               self.drop = nn.Dropout(cfg.dropout)
               self.blocks = nn.ModuleList(
                   [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
               )
               self.norm_f = make_norm(cfg)
               self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        2. WEIGHT TYING:

               if cfg.tie_embeddings:
                   self.lm_head.weight = self.token_embedding.weight

        3. LAS TABLAS DE RoPE, como buffer:

               if cfg.pos == "rope":
                   cos, sin = rope_frequencies(
                       cfg.head_dim, cfg.context_length, cfg.rope_theta
                   )
                   self.register_buffer("rope_cos", cos, persistent=False)
                   self.register_buffer("rope_sin", sin, persistent=False)

        4. LA INICIALIZACION, en dos pasadas. Primero todo:

               self.apply(self._init_weights)

           con `_init_weights` poniendo `nn.init.normal_(m.weight, mean=0.0, std=0.02)` en los
           `nn.Linear` y `nn.Embedding`, y `nn.init.zeros_` en los sesgos que haya.

           Y despues, PISANDO lo anterior solo en las proyecciones que escriben en la corriente
           residual:

               scale = 0.02 / math.sqrt(2 * cfg.n_layers)
               for name, param in self.named_parameters():
                   if name.endswith(("out_proj.weight", "down_proj.weight", "fc_out.weight")):
                       nn.init.normal_(param, mean=0.0, std=scale)

        El orden de los pasos 2 y 4 importa: el tying va DESPUES de crear `lm_head` y ANTES de
        la inicializacion.

    QUÉ TIENES QUE ESCRIBIR EN `forward(idx, targets=None)`
    -------------------------------------------------------
        1. Valida el tamanyo, con los dos numeros en el mensaje:

               B, T = idx.shape
               if T > self.cfg.context_length:
                   raise ValueError(f"secuencia de {T} > contexto {self.cfg.context_length}")

        2. x = self.token_embedding(idx)
        3. si `cfg.pos == "learned"`, sumale
           `self.pos_embedding(torch.arange(T, device=idx.device))`
        4. x = self.drop(x)
        5. cos, sin = (self.rope_cos, self.rope_sin) si pos=="rope", si no (None, None)
        6. mask = causal_mask(T, device=idx.device)      <- UNA vez, no dentro del bucle
        7. for block in self.blocks: x = block(x, cos=cos, sin=sin, mask=mask)
        8. x = self.norm_f(x)
        9. logits = self.lm_head(x)
       10. la perdida, si hay targets:

               loss = None
               if targets is not None:
                   loss = F.cross_entropy(
                       logits.reshape(-1, logits.size(-1)),
                       targets.reshape(-1),
                       ignore_index=-100,
                   )
               return logits, loss

    POR QUÉ CADA UNA DE LAS COSAS RARAS
    -----------------------------------
    **`self.lm_head.weight = self.token_embedding.weight`** no COPIA nada: hace que las dos
    capas apunten al mismo tensor en memoria. Ahorra 1.310.720 parametros (el 15% del modelo) y
    ademas mejora la calidad, porque cada peso recibe gradiente por dos caminos.

    **`register_buffer`** guarda un tensor que acompanya al modelo —se mueve con `.to(device)`,
    sale en el `state_dict`— pero NO es un parametro: no recibe gradiente y el optimizador ni lo
    ve. `persistent=False` hace ademas que no se guarde en el checkpoint, porque se recalcula
    solo al construir el modelo y guardarlo seria desperdiciar espacio.

    **La inicializacion escalada.** Cada bloque SUMA su contribucion a la corriente residual,
    asi que con 6 capas y 2 sub-bloques cada una la varianza de la salida seria 12 veces la de
    la entrada. Reducir la desviacion de las proyecciones que escriben ahi lo compensa. El 2 del
    denominador es porque cada bloque escribe dos veces.

    Y el 0.02 tampoco es arbitrario: es lo que hace que la perdida del paso 0 valga ln(V).
    Con `std=1` (el defecto de `nn.Embedding`) el modelo arrancaria opinando fuerte y al azar, y
    la perdida saldria por ENCIMA de ln(V). Lo viste en la demo del modulo 05.

    **`mask` fuera del bucle.** Es la misma para las 6 capas. Calcularla dentro funciona y
    desperdicia trabajo en cada paso de entrenamiento.

    **`ignore_index=-100`** no hace nada ahora mismo, pero lo necesitaras en el modulo 16 para
    enmascarar el prompt en el fine-tuning. Dejalo puesto.

    CÓMO SABER SI ESTÁ BIEN
    -----------------------
        - `count_parameters(GPT(ModelConfig()))["total"] == 8_933_440`
        - la perdida del primer forward, sin entrenar, ronda `ln(4096) = 8.32`
        - cambiar el token de la posicion 6 no mueve NI UN logit de las posiciones 0-5
          (la demo lo comprueba: sale 0.00e+00 exacto)

    Si la perdida inicial sale MUY por debajo de 8.32, no celebres: casi siempre significa que
    el modelo esta viendo el futuro. Mira la mascara, y despues mira como se monta el batch.

    forward(self, idx, targets=None):
        Args:
            idx: `(B, T)` int64, los ids de token.
            targets: `(B, T)` int64 desplazados una posicion, o None.
        Returns:
            `(logits, loss)` con logits `(B, T, vocab_size)`. `loss` es None sin targets.

        Raises:
            ValueError: si `T` supera `cfg.context_length`.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError("TODO: modulo 10, ejercicio 4 - GPT.__init__")

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("TODO: modulo 10, ejercicio 4 - GPT.forward")
