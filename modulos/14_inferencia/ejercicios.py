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

    QUE ES ESTO
        Un parche directo contra el texto repetitivo: si un token ya ha aparecido, se le
        baja el logit para que sea menos probable que vuelva a salir.

    EL DETALLE QUE CASI TODO EL MUNDO IMPLEMENTA MAL

        logit > 0  ->  logit / penalty      lo acerca a cero
        logit < 0  ->  logit * penalty      lo aleja de cero, HACIA ABAJO

        Con penalty=1.1:
            +3.0  ->  3.0 / 1.1 = 2.73     menos probable   OK
            -3.0  -> -3.0 * 1.1 = -3.30    menos probable   OK

        Si dividieras SIEMPRE, el -3.0 pasaria a -2.73, o sea que el token se volveria MAS
        probable: justo lo contrario de penalizarlo. Y como los logits negativos son la
        mayoria, estarias premiando casi todo lo que ya salio.

    COMO
        Para cada fila del batch:
          1. `vistos = torch.unique(generated[fila])` -> los ids ya generados
          2. coger `logits[fila, vistos]`
          3. aplicar la regla con `torch.where(valores > 0, valores / penalty,
             valores * penalty)`
          4. escribirlo de vuelta

        `torch.unique` evita penalizar dos veces un token que salio dos veces. (Hay
        implementaciones que si acumulan; nosotros no, para que el efecto sea predecible.)

    Args:
        logits: `(B, vocab_size)` los logits del siguiente token.
        generated: `(B, T)` los tokens generados hasta ahora.
        penalty: 1.0 no hace nada. Valores tipicos: 1.05 a 1.2.

    Returns:
        Los logits penalizados, sin modificar la entrada (usa `.clone()`).
    """
    raise NotImplementedError("TODO: modulo 14, ejercicio 1 - apply_repetition_penalty")


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Deja solo los `k` logits mayores y pone el resto a -inf.

    QUE ES ESTO
        Con vocabulario de 4096 hay miles de tokens con probabilidad diminuta pero no nula.
        Sumadas, esa cola larga puede llevarse un 20% de la masa, y de vez en cuando sale
        una y descarrila la frase.

        Top-k la corta en seco.

    COMO, EN DOS LINEAS

        umbral = torch.topk(logits, k, dim=-1).values[..., -1:]
        return logits.masked_fill(logits < umbral, float("-inf"))

        `torch.topk(...).values[..., -1:]` es el k-esimo logit mayor, o sea el umbral. El
        `[..., -1:]` (con dos puntos) conserva la dimension para que el broadcast funcione;
        con `[..., -1]` la perderias.

        Usa `<` y no `<=`: el propio umbral tiene que sobrevivir.

    CASOS BORDE
        Si `k <= 0` o `k >= vocab_size`, devuelve los logits sin tocar.

    SU DEFECTO
        `k` es fijo. Si el modelo esta segurisimo del siguiente token, k=40 mete 39
        alternativas malas. Si duda genuinamente entre 100, corta opciones buenas. Eso lo
        resuelve top-p, en el ejercicio siguiente.
    """
    raise NotImplementedError("TODO: modulo 14, ejercicio 2 - top_k_filter")


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: se queda con los tokens que acumulan una masa `p`.

    QUE ES ESTO
        Como top-k, pero con un numero VARIABLE de candidatos. Se ordenan las
        probabilidades de mayor a menor, se van acumulando, y se corta al llegar a `p`.

        Con probs = [0.60, 0.25, 0.10, 0.03, 0.02] y p=0.9, mirando la acumulada ANTES de
        cada token:

            0.60  ->  acumulado previo 0.00  <= 0.9  -> entra
            0.25  ->  acumulado previo 0.60  <= 0.9  -> entra
            0.10  ->  acumulado previo 0.85  <= 0.9  -> entra   <- el que CRUZA entra
            0.03  ->  acumulado previo 0.95  >  0.9  -> fuera
            0.02  ->  acumulado previo 0.98  >  0.9  -> fuera

        Se queda con 3, que suman 0.95.

        OJO: el token que cruza el umbral ENTRA. La definicion de Holtzman es "el conjunto
        mas pequenyo cuya probabilidad acumulada EXCEDE p", y [0.60, 0.25] suma 0.85, que
        no excede 0.9. Es un off-by-one facil de equivocar.

        Pero si las probs fueran [0.2]*5, se quedaria con 5.

        EL NUMERO DE CANDIDATOS SE ADAPTA A LO SEGURO QUE ESTE EL MODELO. Eso es lo que lo
        hace mejor que top-k en la practica.

    COMO
        1. `ordenados, indices = torch.sort(logits, descending=True, dim=-1)`
        2. `probs = F.softmax(ordenados, dim=-1)` y `acumulada = torch.cumsum(probs, -1)`
        3. Marcar para quitar: `quitar = acumulada - probs > p`

           Fijate en el `- probs`: se compara la acumulada ANTES de incluir el token
           actual. Asi el token que hace pasar de `p` todavia entra. Si compararas
           `acumulada > p` a secas, cortarias uno de mas.

        4. `quitar[..., 0] = False`  <- EL MAS PROBABLE SIEMPRE SE QUEDA
        5. Devolver los indices al orden original:
           `a_quitar = quitar.scatter(-1, indices, quitar)`
        6. `logits.masked_fill(a_quitar, float("-inf"))`

    EL PASO 4 NO ES OPCIONAL
        Con p=0.5 y un token de probabilidad 0.9, sin esa linea te quedarias sin ningun
        candidato y `torch.multinomial` reventaria. Hay un test que lo comprueba.

    EL PASO 5 ES EL QUE MAS CUESTA VER
        Has ordenado los logits, asi que las marcas de "quitar" estan en orden de
        probabilidad, no en orden de token. `scatter` las devuelve a su sitio: para cada
        posicion `j` del tensor ordenado, escribe su marca en la posicion `indices[j]` del
        resultado.

    Si `p >= 1.0`, no filtra nada.
    """
    raise NotImplementedError("TODO: modulo 14, ejercicio 3 - top_p_filter")


class KVCache:
    """Guarda las claves y valores ya calculados para no recalcularlos.

    EL PROBLEMA
        Al generar el token 100, la version ingenua vuelve a pasar los 100 tokens por el
        modelo, aunque los 99 primeros no han cambiado. Generar N tokens cuesta O(N^2) en
        vez de O(N).

    LA SOLUCION
        Guardar las K y V de cada capa y, en cada paso, procesar SOLO el token nuevo,
        concatenando sus K y V a lo guardado.

        Lo que NO se puede cachear son las queries: cada token nuevo necesita su propia
        pregunta. Lo que se reutiliza son las respuestas (K) y los contenidos (V) de los
        anteriores. De ahi el nombre.

    LA CLASE, que es sencilla
        __init__(self, n_layers): dos listas de `n_layers` elementos, todos None.
            `self.keys` y `self.values`.

        update(self, layer, k, v) -> (K, V):
            Si `self.keys[layer]` es None, guarda k y v tal cual.
            Si no, concatena:  torch.cat([self.keys[layer], k], dim=-2)
            Devuelve las K y V COMPLETAS.

            El `dim=-2` es la dimension de tiempo con la forma (B, n_heads, T, head_dim).
            Usa indice negativo: con dim=2 te funcionaria aqui pero se rompe si algun dia
            cambia el numero de dimensiones.

        seq_len (property): cuantos tokens hay guardados.
            `0 if self.keys[0] is None else self.keys[0].shape[-2]`

        reset(self): vuelve a dejarlo todo a None.

        memory_bytes(self): suma `t.numel() * t.element_size()` de todos los tensores no
            nulos. Para poder ensenyar en la demo cuanto ocupa.

    LA MEMORIA
        2 * n_layers * T * d_model * bytes

        Nuestro modelo con 512 tokens en fp16: 3,9 MB, o sea nada. Un modelo de 70B con
        contexto de 100.000: decenas de gigabytes, mas que los propios pesos. De ahi que
        existan tecnicas como grouped-query attention.
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

    Es el mismo bucle autorregresivo del modulo 00 (contexto -> distribucion -> muestrear
    -> anyadir -> repetir), ahora con un modelo de verdad.

    LAS DOS FASES
        1. PREFILL: `logits, _ = model(idx, use_cache=True, cache=cache)`
           Se pasa el prompt ENTERO de golpe y se llena la cache.

        2. DECODE: en cada vuelta se pasa SOLO el token nuevo:
           `logits, _ = model(nuevo, use_cache=True, cache=cache)`

    EL ORDEN DE LOS FILTROS, que importa

        1. penalizacion de repeticion   <- sobre los logits crudos
        2. temperatura                  <- dividir
        3. top-k
        4. top-p

        La temperatura va ANTES de los filtros porque cambia las probabilidades que mira
        top-p. (No cambia el ranking, porque dividir por una constante positiva no reordena
        nada, pero si cambia las masas acumuladas.)

    EL BUCLE
        model.eval()
        cache = KVCache(model.cfg.n_layers)
        logits, _ = model(idx, use_cache=True, cache=cache)      # prefill

        repetir max_new_tokens veces:
            siguiente = logits[:, -1, :].float()     # solo la ultima posicion
            aplicar los filtros en orden
            si temperature == 0: nuevo = argmax        (greedy determinista)
            si no:               nuevo = torch.multinomial(softmax(siguiente), 1)
            idx = torch.cat([idx, nuevo], dim=1)
            si eos_token y nuevo == eos_token: break
            logits, _ = model(nuevo, use_cache=True, cache=cache)   # decode

    EL `.float()` DE LOS LOGITS
        Bajo AMP los logits llegan en fp16, y `torch.multinomial` sobre fp16 puede dar
        resultados raros con probabilidades muy pequenyas. Convertir a fp32 antes de
        muestrear es barato y evita el problema.

    EL LIMITE DE CONTEXTO
        PARA al llegar al contexto maximo del modelo (`model.cfg.context_length`), en vez
        de recortar como hace `model.generate`.

        No es pereza: recortar con cache es genuinamente mas complicado. Habria que
        descartar las entradas antiguas Y REMAPEAR las posiciones de RoPE de todo lo que
        queda, porque los tokens supervivientes pasarian a ocupar posiciones distintas. Eso
        se llama sliding window attention y da para un modulo entero.

        Parar es lo honesto: la alternativa silenciosa seria generar texto incorrecto sin
        avisar.

        Anyade tambien un `ValueError` si el prompt YA llega al limite: mejor un error claro
        que un `break` inmediato que devuelve el prompt sin explicacion.

    LA COMPROBACION OBLIGATORIA
        Con `temperature=0` (greedy, determinista), esta funcion tiene que dar EXACTAMENTE
        la misma salida que la generacion sin cache. No parecida: identica, token a token.
        Hay un test que lo verifica.

    Returns:
        `(B, T + max_new_tokens)` con el prompt y lo generado.
    """
    raise NotImplementedError("TODO: modulo 14, ejercicio 5 - generate_with_cache")
