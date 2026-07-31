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
tu formula o tu modelo mienten.

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

    QUE ES ESTO
        Derivar a mano cuantos parametros tendra un modelo ANTES de construirlo. Sirve para
        disenyar (cambias d_model y ves al instante si te cabe en la GPU) y para verificar
        que el modelo que has montado es el que creias.

    HAZLO CON PAPEL PRIMERO
        En serio. Coge el desglose del TEORIA.md y escribe la formula. Solo despues lo
        traduces a codigo. Si vas directo al codigo acabaras probando numeros hasta que
        cuadre, y eso no ensenya nada.

    LOS TERMINOS

        embeddings de token   = vocab_size * d_model
        + si pos == "learned" = context_length * d_model    (con RoPE no hay nada)

        por capa:
          atencion  = 4 * d_model^2                (Wq, Wk, Wv, Wo)
                      + 4*d_model si hay sesgos
          ffn       = 3 * d_model * d_ff           (SwiGLU: gate, up, down)
                      o 2 * d_model * d_ff         (MLP clasico)
          normas    = 2 * (d_model)                (dos RMSNorm por bloque)
                      o 2 * (2*d_model) si es LayerNorm CON sesgo

        norma final = d_model      (o 2*d_model con LayerNorm y sesgo)

        lm_head     = 0 si tie_embeddings, si no vocab_size * d_model

    LO QUE NO CUENTA
        RoPE no aporta NI UN parametro. Sus tablas de cos/sin salen de una formula y se
        guardan como buffers, no como parametros. Si tu cuenta incluye algo de RoPE, esta
        mal.

    LA COMPROBACION
        Con el config por defecto (el modelo final) tiene que dar exactamente 8.933.440.

            1.310.720 + 6 * (409.600 + 860.160 + 640) + 320 = 8.933.440

    Args:
        cfg: la configuracion del modelo.

    Returns:
        El numero total de parametros, como entero.
    """
    raise NotImplementedError("TODO: modulo 10, ejercicio 1 - expected_param_count")


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Cuenta los parametros de verdad, desglosados por componente.

    QUE ES ESTO
        Lo contrario del ejercicio 1: en vez de calcular, recorrer el modelo y sumar. Si
        los dos numeros coinciden, tu formula y tu modelo dicen lo mismo.

    EL WEIGHT TYING
        Con `tie_embeddings=True`, `lm_head.weight` y `token_embedding.weight` son EL MISMO
        tensor, no dos copias. Aparece bajo dos nombres distintos.

        Dato util: tanto `parameters()` como `named_parameters()` DEDUPLICAN por identidad
        por defecto (`remove_duplicate=True`), asi que el total sale bien sin hacer nada.
        Con `named_parameters(remove_duplicate=False)` si verias el tensor repetido.

        Aun asi, lleva un `set` de `id(param)` ya vistos. Dos motivos: deja explicito que
        sabes que hay pesos compartidos, y protege el desglose si algun dia recorres los
        parametros con `remove_duplicate=False`. Es una linea y evita un fallo que seria
        de 1.310.720 parametros.

            vistos = set()
            for name, param in model.named_parameters():
                if id(param) in vistos:
                    continue
                vistos.add(id(param))
                ...

    COMO CLASIFICAR
        Por lo que aparezca en el nombre del parametro. Los nombres son del estilo
        `blocks.3.attn.q_proj.weight`, asi que basta con buscar subcadenas:

            "token_embedding" o "pos_embedding"  -> embeddings
            "attn."                              -> attention
            gate_proj / up_proj / down_proj      -> ffn
            "norm"                               -> norms
            "lm_head"                            -> lm_head
            el resto                             -> other

        (Imprime `[n for n, _ in model.named_parameters()]` una vez para verlos todos. Vale
        mucho la pena para entender como esta montado el modelo.)

    Returns:
        Un dict con las claves `embeddings`, `attention`, `ffn`, `norms`, `lm_head`,
        `other`, `total` y `non_embedding`.

        `total` es la suma de las seis primeras.
        `non_embedding` es `total - embeddings`. Ese es el numero que usan las leyes de
        escala del modulo 12, porque los embeddings escalan distinto al resto.
    """
    raise NotImplementedError("TODO: modulo 10, ejercicio 2 - count_parameters")


class TransformerBlock(nn.Module):
    """Un bloque: atencion y FFN, cada uno con su normalizacion y su residual.

    LA ESTRUCTURA, ENTERA

        x = x + atencion(norm1(x))
        x = x + ffn(norm2(x))

    Eso es todo. Dos sub-bloques con pre-norm (modulo 07). La atencion MUEVE informacion
    entre tokens; el FFN la PROCESA token a token. Alternan.

    Los dos residuales son independientes a proposito: cada sub-bloque puede aportar poco o
    mucho a la corriente residual sin condicionar al otro.

    SUBMODULOS (respeta los nombres: el test copia pesos por nombre y el ejercicio 2
    clasifica por nombre)
        attn_norm: la normalizacion que pida cfg.norm
        attn:      MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=..., bias=...)
        ffn_norm:  otra normalizacion
        ffn:       SwiGLU (o el MLP clasico si cfg.activation != "swiglu")

    Para construir las normas y el FFN segun el config puedes usar los ayudantes que ya
    estan hechos:

        from llmfs.reference import make_norm, make_ffn
        self.attn_norm = make_norm(cfg)
        self.ffn = make_ffn(cfg)

    forward(self, x, cos=None, sin=None, mask=None):
        Args:
            x: `(B, T, d_model)`.
            cos, sin: tablas de RoPE, o None.
            mask: la mascara causal, o None para que la genere la atencion.
        Returns:
            `(B, T, d_model)`.

        Pasale `cos`, `sin` y `mask` a la atencion. El FFN no los necesita: no mira a
        otros tokens.
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

    LA ESTRUCTURA

        ids -> embeddings -> [bloque] x n_layers -> norma final -> logits

    Con RoPE no hay embedding posicional que sumar al principio: la posicion se inyecta
    dentro de la atencion. Por eso la primera capa es solo la tabla de tokens.

    SUBMODULOS
        token_embedding: nn.Embedding(vocab_size, d_model)
        pos_embedding:   nn.Embedding(context_length, d_model), SOLO si cfg.pos=="learned"
        drop:            nn.Dropout(cfg.dropout)
        blocks:          nn.ModuleList con n_layers TransformerBlock
        norm_f:          la normalizacion final (make_norm(cfg))
        lm_head:         nn.Linear(d_model, vocab_size, bias=False)

    LAS TRES COSAS QUE HAY QUE HACER BIEN
    -------------------------------------

    1. WEIGHT TYING

           if cfg.tie_embeddings:
               self.lm_head.weight = self.token_embedding.weight

       Eso NO copia: hace que las dos capas apunten al mismo tensor. Ahorra 1.310.720
       parametros (el 15% del modelo) y ademas mejora la calidad, porque cada peso recibe
       gradiente por dos caminos.

       Tiene que ir DESPUES de crear `lm_head`, obviamente, y ANTES de la inicializacion.

    2. LAS TABLAS DE RoPE COMO BUFFER

           if cfg.pos == "rope":
               cos, sin = rope_frequencies(cfg.head_dim, cfg.context_length, cfg.rope_theta)
               self.register_buffer("rope_cos", cos, persistent=False)
               self.register_buffer("rope_sin", sin, persistent=False)

       `register_buffer` guarda un tensor que acompanya al modelo (se mueve con `.to(device)`)
       pero NO es un parametro y no recibe gradiente.

       `persistent=False` hace que ademas no se guarde en el checkpoint: se recalcula al
       construir el modelo, asi que guardarla seria desperdiciar espacio.

    3. LA INICIALIZACION, EN DOS PASADAS

       Primera pasada, todo:

           self.apply(self._init_weights)

       con `_init_weights` poniendo `normal_(std=0.02)` en los `nn.Linear` y `nn.Embedding`,
       y ceros en los sesgos que haya.

       Segunda pasada, PISANDO la anterior solo en las proyecciones que ESCRIBEN en la
       corriente residual (`out_proj` de la atencion y `down_proj` del FFN):

           scale = 0.02 / math.sqrt(2 * cfg.n_layers)
           for name, param in self.named_parameters():
               if name.endswith(("out_proj.weight", "down_proj.weight", "fc_out.weight")):
                   nn.init.normal_(param, mean=0.0, std=scale)

       POR QUE: cada bloque SUMA su contribucion a la corriente residual, asi que con 6
       capas y 2 sub-bloques cada una la varianza de la salida seria 12 veces la de la
       entrada. Reducir la desviacion de esas proyecciones lo compensa. El 2 del
       denominador es porque cada bloque escribe dos veces.

       Y el 0.02 tampoco es arbitrario: es lo que hace que la perdida del paso 0 valga
       ln(V). Con std=1 (el defecto de PyTorch) el modelo arrancaria opinando fuerte y al
       azar, y la perdida saldria por encima.

    forward(self, idx, targets=None):
        Args:
            idx: `(B, T)` int64.
            targets: `(B, T)` int64, o None.
        Returns:
            `(logits, loss)` con logits `(B, T, vocab_size)`; loss es None sin targets.

        Los pasos:
          1. validar que T <= cfg.context_length y lanzar ValueError con los dos numeros
          2. x = token_embedding(idx)
          3. si pos == "learned": sumar pos_embedding(arange(T))
          4. x = drop(x)
          5. si pos == "rope": cos, sin = self.rope_cos, self.rope_sin  (si no, None)
          6. mask = causal_mask(T, device=idx.device)   <- se calcula UNA vez, no por bloque
          7. pasar x por todos los bloques con cos, sin y mask
          8. x = norm_f(x)   <- obligatoria en pre-norm, ver TEORIA.md
          9. logits = lm_head(x)
         10. si hay targets:
                 F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)

        El `ignore_index=-100` no hace nada ahora, pero lo necesitaras en el modulo 16 para
        enmascarar el prompt en el fine-tuning. Dejalo puesto.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError("TODO: modulo 10, ejercicio 4 - GPT.__init__")

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("TODO: modulo 10, ejercicio 4 - GPT.forward")
