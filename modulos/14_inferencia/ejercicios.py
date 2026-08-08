"""Modulo 14 - Inferencia y muestreo.

CÓMO SE HACE ESTE MÓDULO
========================

Lee `TEORIA.md` -> implementa en orden -> `llmfs check 14` -> `llmfs hint 14 -e N`
-> `SOLUCION.md` tiene el codigo completo.

QUÉ VAS A CONSTRUIR
===================

Como se saca texto de un modelo entrenado, y como hacerlo rapido:

    apply_repetition_penalty  (ej. 1)  romper los bucles
    top_k_filter              (ej. 2)  quedarse con los k mejores
    top_p_filter              (ej. 3)  quedarse con los que suman p
            |
    KVCache                   (ej. 4)  guardar lo ya calculado
            |
            v
    generate_with_cache       (ej. 5)  el bucle que junta todo

Los tres primeros son cortos. El 5 es donde esta la dificultad, y tiene una comprobacion
implacable: con la cache tiene que salir EXACTAMENTE el mismo texto que sin ella.

`TEORIA.md` los sigue en este mismo orden y cada docstring de aqui te dice que seccion le toca.
Los tres filtros son independientes entre si; la cache y el bucle encadenan.

Y hay algo que la teoria explica y no esta en ningun docstring: EL ORDEN en que se aplican los
filtros dentro del ejercicio 5, que es penalizacion -> temperatura -> top-k -> top-p, y por que
ese y no otro.

VOCABULARIO QUE VAS A NECESITAR
===============================

- **muestrear** (sample): elegir el siguiente token al azar respetando sus probabilidades,
  en vez de coger siempre el mas probable.
- **greedy**: coger siempre el mas probable. Es determinista y se mete en bucles.
- **temperatura**: dividir los logits antes del softmax. Menor de 1 afila la distribucion,
  mayor de 1 la aplana.
- **top-k / top-p**: dos formas de descartar los tokens malos. Top-k coge un numero fijo,
  top-p un numero variable segun lo seguro que este el modelo.
- **KV cache**: guardar las claves y valores ya calculados para no recalcularlos en cada
  token. Convierte un coste O(N^2) en O(N).
- **prefill / decode**: las dos fases de la generacion. Prefill procesa el prompt entero;
  decode va token a token.

    llmfs demo 14     compara estrategias de muestreo y mide el speedup de la cache
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def apply_repetition_penalty(
    logits: torch.Tensor, generated: torch.Tensor, penalty: float = 1.1
) -> torch.Tensor:
    """Penaliza los tokens que ya han salido, para romper bucles.

    Contexto en `TEORIA.md`: seccion "Ejercicio 1: la penalizacion de repeticion", con los
    numeros de por que hay que dividir si el logit es positivo y multiplicar si es negativo.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Un bucle por filas del batch. Cuatro lineas dentro.

        1. Copia, para no modificar la entrada:

               out = logits.clone()

        2. Para cada fila:

               for fila in range(logits.shape[0]):
                   vistos = torch.unique(generated[fila])
                   valores = out[fila, vistos]
                   out[fila, vistos] = torch.where(
                       valores > 0, valores / penalty, valores * penalty
                   )

        3. `return out`

    Si `penalty == 1.0` puedes devolver `logits` tal cual y ahorrarte todo: dividir y
    multiplicar por 1 no hace nada.

    EL DETALLE QUE CASI TODO EL MUNDO IMPLEMENTA MAL
    ------------------------------------------------
        logit > 0  ->  logit / penalty      lo acerca a cero
        logit < 0  ->  logit * penalty      lo aleja de cero, HACIA ABAJO

    Con penalty=1.1:

        +3.0  ->   3.0 / 1.1 =  2.73    menos probable   OK
        -3.0  ->  -3.0 * 1.1 = -3.30    menos probable   OK

    Si dividieras SIEMPRE, el -3.0 pasaria a -2.73, o sea que el token se volveria MAS
    probable: justo lo contrario de penalizarlo. Y como los logits negativos son la mayoria
    aplastante, estarias premiando casi todo lo que ya salio. El bug es silencioso: el texto
    sale repetitivo y parece que el parametro no hace nada.

    Por eso hace falta el `torch.where` y no vale una sola operacion.

    POR QUÉ `torch.unique`
    ----------------------
    Evita penalizar dos veces un token que aparecio dos veces. Si escribieses
    `out[fila, generated[fila]] = ...` con indices repetidos, PyTorch aplica la asignacion una
    sola vez de forma no determinista, asi que ni siquiera acumularia bien. Hay
    implementaciones que si acumulan a proposito; nosotros no, para que el efecto sea
    predecible.

    QUÉ ES ESTO EN REALIDAD
    -----------------------
    Un parche. Un modelo bien entrenado no deberia meterse en bucles, y el nuestro, siendo de
    9M, si lo hace. Esto lo tapa: si un token ya ha aparecido, se le baja el logit.

    No es un metodo con base teorica —a diferencia de top-p, que sale de un paper con
    experimentos— sino un truco practico que funciona. Con valores altos (>1.3) empieza a
    notarse: el modelo evita palabras necesarias, como articulos y preposiciones, que por su
    naturaleza tienen que repetirse.

    Args:
        logits: `(B, vocab_size)`, los logits del siguiente token.
        generated: `(B, T)`, los tokens generados hasta ahora.
        penalty: 1.0 no hace nada. Valores tipicos: 1.05 a 1.2.

    Returns:
        Los logits penalizados. NO modifica la entrada.
    """
    raise NotImplementedError("TODO: modulo 14, ejercicio 1 - apply_repetition_penalty")


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Deja solo los `k` logits mayores y pone el resto a -inf.

    Contexto en `TEORIA.md`: seccion "Ejercicio 2: top-k", que es corta y explica sobre todo el
    defecto que motiva el ejercicio siguiente: la k es fija.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Tres lineas.

        1. El caso borde:

               if k <= 0 or k >= logits.shape[-1]:
                   return logits

        2. El umbral, que es el k-esimo logit mayor:

               umbral = torch.topk(logits, k, dim=-1).values[..., -1:]

        3. El filtro:

               return logits.masked_fill(logits < umbral, float("-inf"))

    DOS COSAS QUE MIRAR CON LUPA
    ----------------------------
    **El `[..., -1:]` con dos puntos.** `torch.topk` devuelve los k valores ordenados de mayor a
    menor, asi que el ultimo es el umbral. Con `[..., -1]` (sin los dos puntos) perderias esa
    dimension y el broadcast del paso 3 no cuadraria. Con `[..., -1:]` te queda `(B, 1)`, que se
    emite bien contra `(B, vocab_size)`.

    **`<` y no `<=`.** El propio umbral tiene que SOBREVIVIR: es el k-esimo mejor y forma parte
    de los k que quieres. Con `<=` te quedarias con k-1 candidatos (o con menos si hay empates).

    POR QUÉ -inf Y NO 0
    -------------------
    Porque estos numeros van a pasar por un softmax, y `exp(-inf) = 0` exacto. Poner 0 no
    descartaria nada: 0 es un logit perfectamente normal, y `exp(0) = 1` es una probabilidad
    BASTANTE alta comparada con un logit de -5.

    QUÉ PROBLEMA RESUELVE
    ---------------------
    Con vocabulario de 4096 hay miles de tokens con probabilidad diminuta pero no nula.
    Sumadas, esa cola larga puede llevarse un 20% de la masa total, y de vez en cuando sale una
    de esas y descarrila la frase entera. Como el modelo es autorregresivo, no hay vuelta atras:
    sigue generando sobre el error.

    Top-k corta la cola en seco.

    SU DEFECTO, QUE ARREGLA EL EJERCICIO SIGUIENTE
    ----------------------------------------------
    `k` es FIJO. Si el modelo esta segurisimo del siguiente token (despues de "Once upon a",
    "time" se lleva el 99%), k=40 mete 39 alternativas malas. Si el modelo duda genuinamente
    entre 100 continuaciones validas, k=40 corta opciones buenas.

    Eso lo resuelve top-p, con un numero variable de candidatos.

    Args:
        logits: `(B, vocab_size)`.
        k: cuantos candidatos dejar. Si es <= 0 o >= vocab_size, no filtra nada.

    Returns:
        Los logits con todo menos los k mayores puestos a -inf.
    """
    raise NotImplementedError("TODO: modulo 14, ejercicio 2 - top_k_filter")


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: se queda con los tokens que acumulan una masa `p`.

    Contexto en `TEORIA.md`: seccion "Ejercicio 3: top-p o nucleus", con la tabla de seis tokens
    donde se ve que el token que CRUZA el umbral entra, que es el off-by-one del ejercicio.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Seis lineas, y la 5 es la que cuesta ver.

        1. El caso borde:

               if p >= 1.0:
                   return logits

        2. Ordena de mayor a menor, guardando de donde venia cada uno:

               ordenados, indices = torch.sort(logits, descending=True, dim=-1)

        3. Las probabilidades y su acumulada:

               probs = F.softmax(ordenados, dim=-1)
               acumulada = torch.cumsum(probs, dim=-1)

        4. Marca lo que sobra:

               quitar = acumulada - probs > p

        5. El mas probable SIEMPRE se queda:

               quitar[..., 0] = False

        6. Devuelve las marcas al orden original y filtra:

               a_quitar = quitar.scatter(-1, indices, quitar)
               return logits.masked_fill(a_quitar, float("-inf"))

    EL EJEMPLO, SEGUIDO A MANO
    --------------------------
    Con probs = [0.60, 0.25, 0.10, 0.03, 0.02] y p = 0.9. La columna que decide es la acumulada
    ANTES de incluir el token actual, o sea `acumulada - probs`:

        prob    acumulada    acumulada - prob    ¿> 0.9?
        0.60      0.60             0.00            no    -> entra
        0.25      0.85             0.60            no    -> entra
        0.10      0.95             0.85            no    -> entra   <- el que CRUZA
        0.03      0.98             0.95            SI    -> fuera
        0.02      1.00             0.98            SI    -> fuera

    Se queda con 3 candidatos, que suman 0.95.

    Con probs = [0.2]*5 se quedaria con los 5. EL NUMERO DE CANDIDATOS SE ADAPTA A LO SEGURO
    QUE ESTE EL MODELO, y eso es lo que lo hace mejor que top-k en la practica.

    EL `- probs` DEL PASO 4 ES UN OFF-BY-ONE FÁCIL DE EQUIVOCAR
    -----------------------------------------------------------
    La definicion de Holtzman es "el conjunto mas pequenyo cuya probabilidad acumulada EXCEDE
    p". Fijate en el ejemplo: 0.60 + 0.25 = 0.85, que NO excede 0.9. Hace falta el tercero.

    Si compararas `acumulada > p` a secas, el token que cruza el umbral se quedaria fuera y
    cortarias uno de menos. Es un error de un solo candidato que no revienta nada y que solo se
    detecta contando.

    EL PASO 5 NO ES OPCIONAL
    ------------------------
    Con p=0.5 y un token de probabilidad 0.9, la acumulada-antes del primer token es 0.00, que
    no supera 0.5... pero el segundo tiene acumulada-antes 0.90, y todos los demas tambien.
    Peor: si el primer token tuviera probabilidad 0.95 y p fuera 0.5, sin esa linea te podrias
    quedar SIN NINGUN candidato y `torch.multinomial` reventaria con un error incomprensible.
    Hay un test que lo comprueba.

    EL PASO 6 ES EL QUE MÁS CUESTA VER
    ----------------------------------
    Has ordenado los logits, asi que tus marcas de "quitar" estan en orden de PROBABILIDAD, no
    en orden de TOKEN. Hay que devolverlas a su sitio.

    `quitar.scatter(-1, indices, quitar)` hace exactamente eso: para cada posicion `j` del
    tensor ordenado, escribe `quitar[j]` en la posicion `indices[j]` del resultado. Y `indices`
    es justo lo que devolvio `torch.sort`: de donde venia cada elemento.

    Si te lo saltas, estaras poniendo a -inf tokens elegidos casi al azar, y el resultado sera
    texto malo sin ningun error visible.

    Args:
        logits: `(B, vocab_size)`.
        p: la masa a conservar. Tipico: 0.9 o 0.95. Si es >= 1.0, no filtra nada.

    Returns:
        Los logits con los tokens de la cola puestos a -inf.
    """
    raise NotImplementedError("TODO: modulo 14, ejercicio 3 - top_p_filter")


class KVCache:
    """Guarda las claves y valores ya calculados para no recalcularlos.

    Contexto en `TEORIA.md`: seccion "Ejercicio 4: la KV cache", con el problema que resuelve y por
    que se cachean K y V pero no las queries.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
    Cinco metodos, todos de una o dos lineas.

        def __init__(self, n_layers):
            self.n_layers = n_layers
            self.keys = [None] * n_layers
            self.values = [None] * n_layers

        def update(self, layer, k, v):
            if self.keys[layer] is None:
                self.keys[layer] = k
                self.values[layer] = v
            else:
                self.keys[layer] = torch.cat([self.keys[layer], k], dim=-2)
                self.values[layer] = torch.cat([self.values[layer], v], dim=-2)
            return self.keys[layer], self.values[layer]      # las COMPLETAS

        @property
        def seq_len(self):
            return 0 if self.keys[0] is None else self.keys[0].shape[-2]

        def reset(self):
            self.keys = [None] * self.n_layers
            self.values = [None] * self.n_layers

        def memory_bytes(self):
            return sum(
                t.numel() * t.element_size()
                for t in (*self.keys, *self.values)
                if t is not None
            )

    EL `dim=-2`, QUE NO ES CAPRICHO
    -------------------------------
    Con la forma `(B, n_heads, T, head_dim)`, la dimension de tiempo es la penultima. `dim=-2`
    la senyala contando desde el final. `dim=2` te funcionaria aqui exactamente igual, y se
    rompe en silencio el dia que alguien anyada o quite una dimension. Cuenta desde donde la
    forma es estable.

    QUE `update` DEVUELVA LAS TABLAS **COMPLETAS** ES EL PUNTO
    ----------------------------------------------------------
    Le pasas las K y V del token NUEVO (T=1) y te devuelve las de TODOS los tokens vistos. La
    atencion necesita las completas: la query del token nuevo tiene que poder mirar a todos los
    anteriores. Lo unico que se ahorra es volver a CALCULARLAS.

    EL PROBLEMA QUE RESUELVE
    ------------------------
    Al generar el token 100, la version ingenua vuelve a pasar los 100 tokens por el modelo
    entero, aunque los 99 primeros no han cambiado ni un bit. Generar N tokens cuesta O(N²) en
    vez de O(N).

    POR QUÉ SE CACHEAN K Y V PERO NO Q
    ----------------------------------
    Las queries NO se pueden cachear: cada token nuevo necesita hacer su propia pregunta, y esa
    pregunta no existia antes. Lo que se reutiliza son las respuestas (K) y los contenidos (V)
    de los tokens anteriores, que ya estaban calculados y no cambian. De ahi el nombre: KV
    cache, no QKV cache.

    LA MEMORIA QUE OCUPA
    --------------------
        2 * n_layers * T * d_model * bytes_por_numero

    Nuestro modelo con 512 tokens en fp16: 3,9 MB. O sea, nada.

    Un modelo de 70B con contexto de 100.000 tokens: decenas de gigabytes, MAS que los propios
    pesos del modelo. Por eso existen tecnicas como grouped-query attention, que comparte las K
    y V entre varias cabezas precisamente para que esta tabla quepa.

    Args (de `__init__`):
        n_layers: cuantas capas tiene el modelo.
    """

    def __init__(self, n_layers: int) -> None:
        raise NotImplementedError("TODO: modulo 14, ejercicio 4 - KVCache.__init__")

    def update(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TODO: modulo 14, ejercicio 4 - KVCache.update")

    @property
    def seq_len(self) -> int:
        raise NotImplementedError("TODO: modulo 14, ejercicio 4 - KVCache.seq_len")

    def reset(self) -> None:
        raise NotImplementedError("TODO: modulo 14, ejercicio 4 - KVCache.reset")

    def memory_bytes(self) -> int:
        raise NotImplementedError("TODO: modulo 14, ejercicio 4 - KVCache.memory_bytes")


@torch.no_grad()
def generate_with_cache(
    model: Any,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    eos_token: int | None = None,
) -> torch.Tensor:
    """El bucle de generacion, con cache y con todos los filtros.

    Contexto en `TEORIA.md`: seccion "Ejercicio 5: el bucle completo", con el orden de los
    filtros, el pos_offset de RoPE, la tabla de speedup medida y —si la salida no coincide con
    la de sin cache— las dos cosas que hay que mirar, en orden.

    Es el mismo bucle autorregresivo del modulo 00 —contexto, distribucion, muestrear, anyadir,
    repetir— ahora con un modelo de verdad.

    QUÉ TIENES QUE ESCRIBIR
    -----------------------
        1. Las comprobaciones y la preparacion:

               model.eval()
               ctx = model.cfg.context_length
               if idx.shape[1] >= ctx:
                   raise ValueError(
                       f"el prompt ya ocupa {idx.shape[1]} de {ctx} tokens de contexto"
                   )
               cache = KVCache(model.cfg.n_layers)

        2. PREFILL: el prompt entero de golpe, para llenar la cache:

               logits, _ = model(idx, use_cache=True, cache=cache)

        3. El bucle, `max_new_tokens` veces:

               for _ in range(max_new_tokens):
                   if idx.shape[1] >= ctx:
                       break                          # ver mas abajo por que se PARA

                   siguiente = logits[:, -1, :].float()      # solo la ultima posicion

                   # los filtros, EN ESTE ORDEN
                   if repetition_penalty != 1.0:
                       siguiente = apply_repetition_penalty(siguiente, idx, repetition_penalty)
                   if temperature != 1.0:
                       siguiente = siguiente / max(temperature, 1e-8)
                   if top_k is not None:
                       siguiente = top_k_filter(siguiente, top_k)
                   if top_p is not None:
                       siguiente = top_p_filter(siguiente, top_p)

                   # elegir
                   if temperature == 0:
                       nuevo = siguiente.argmax(dim=-1, keepdim=True)
                   else:
                       nuevo = torch.multinomial(F.softmax(siguiente, dim=-1), num_samples=1)

                   idx = torch.cat([idx, nuevo], dim=1)

                   if eos_token is not None and bool((nuevo == eos_token).all()):
                       break

                   # DECODE: solo el token nuevo
                   logits, _ = model(nuevo, use_cache=True, cache=cache)

        4. `return idx`

    LAS DOS FASES, QUE ES LO QUE HAY QUE ENTENDER
    ---------------------------------------------
    **Prefill**: el prompt entero pasa de golpe. Es un unico forward sobre `(B, T_prompt)` y
    llena la cache con las K y V de todos esos tokens.

    **Decode**: a partir de ahi, cada vuelta pasa SOLO el token nuevo, `(B, 1)`. La cache aporta
    todo lo anterior. Este es el ahorro entero.

    Fijate en el orden dentro del bucle: los `logits` con los que empiezas la vuelta N son los
    que produjo el forward del final de la vuelta N-1 (o el prefill, la primera vez). Por eso el
    `model(...)` va al FINAL y no al principio.

    EL ORDEN DE LOS FILTROS IMPORTA
    -------------------------------
        1. penalizacion de repeticion   <- sobre los logits crudos
        2. temperatura                  <- dividir
        3. top-k
        4. top-p

    La temperatura va ANTES de los filtros porque cambia las probabilidades que mira top-p. No
    cambia el RANKING —dividir por una constante positiva no reordena nada— pero si cambia las
    masas acumuladas, y con ellas cuantos candidatos sobreviven. Con temperature=0.7 el nucleo
    de top-p es mas pequenyo que con 1.0.

    EL `.float()` DE LOS LOGITS
    ---------------------------
    Bajo AMP los logits llegan en fp16, y `torch.multinomial` sobre fp16 puede dar resultados
    raros con probabilidades muy pequenyas (hay poca resolucion cerca de cero). Convertir a fp32
    antes de muestrear es barato y evita el problema.

    EL LÍMITE DE CONTEXTO: AQUÍ SE **PARA**, NO SE RECORTA
    ------------------------------------------------------
    `model.generate` (el ingenuo) recorta el contexto y sigue. Este no puede, y no es pereza.

    Recortar con cache significaria descartar las entradas antiguas Y REMAPEAR las posiciones de
    RoPE de todo lo que queda, porque los tokens supervivientes pasarian a ocupar posiciones
    distintas de las que tenian cuando se calcularon sus K. Eso se llama sliding window
    attention y da para un modulo entero.

    Parar es lo honesto: la alternativa silenciosa seria generar texto incorrecto sin avisar.

    Y el `ValueError` de arriba, cuando el prompt YA llega al limite: mejor un error claro que
    un `break` inmediato que devuelve el prompt intacto sin explicar por que.

    LA COMPROBACIÓN OBLIGATORIA
    ---------------------------
    Con `temperature=0` (greedy, determinista), esta funcion tiene que dar EXACTAMENTE la misma
    salida que la generacion sin cache. No parecida: identica, token a token. Hay un test que lo
    verifica, y es la unica forma de saber que la cache no esta corrompiendo nada.

    Si difiere a partir de cierto token, mira primero el `pos_offset` de RoPE: el token nuevo
    tiene que rotarse con la posicion que le toca, no con la 0.

    Args:
        model: el GPT entrenado. Su forward acepta `use_cache` y `cache`.
        idx: `(B, T)`, el prompt.
        max_new_tokens: cuantos tokens generar como mucho.
        temperature: 0 para greedy. 1.0 no altera nada. <1 afila, >1 aplana.
        top_k, top_p: los filtros, o None para no aplicarlos.
        repetition_penalty: 1.0 no hace nada.
        eos_token: si aparece, se para.

    Returns:
        `(B, T + n)` con el prompt y lo generado, con n <= max_new_tokens.

    Raises:
        ValueError: si el prompt ya llena el contexto del modelo.
    """
    raise NotImplementedError("TODO: modulo 14, ejercicio 5 - generate_with_cache")
