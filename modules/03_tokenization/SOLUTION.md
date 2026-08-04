# 03 — Solución comentada

## Ejercicio 1 — `get_stats`

```python
counts = {} if counts is None else counts
for pair in zip(ids, ids[1:]):
    counts[pair] = counts.get(pair, 0) + 1
return counts
```

`zip(ids, ids[1:])` produce todos los pares de vecinos: `(ids[0],ids[1])`, `(ids[1],ids[2])`…
Es la forma idiomática y evita el `range(len(ids)-1)` con índices a mano.

**El `counts` mutable como parámetro** es lo que permite a `train_bpe` acumular las
estadísticas de todos los chunks sin concatenarlos. Y devolverlo, además de mutarlo, hace
que la función sirva para los dos usos: `stats = get_stats(ids)` y
`get_stats(chunk, stats)`.

Si te preocupa que un argumento mutable por defecto sea peligroso: aquí el valor por
defecto es `None`, no `{}`. Esa distinción importa — un `{}` como valor por defecto se
crearía **una sola vez** al definir la función y se compartiría entre todas las llamadas,
que es el clásico bug de Python. Con `None` y la comprobación dentro, cada llamada tiene
su diccionario.

## Ejercicio 2 — `merge`

```python
out, i, n = [], 0, len(ids)
while i < n:
    if i < n - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
        out.append(new_id)
        i += 2
    else:
        out.append(ids[i])
        i += 1
return out
```

**El `while` con índice manual no es opcional.** Un `for` avanza de uno en uno siempre, y
aquí necesitas saltar dos posiciones cuando hay coincidencia. Esa es exactamente la
diferencia entre contar pares (que sí solapan) y fusionarlos (que no).

Compruébalo con `[1,1,1]`:
- Contar: `(1,1)` sale **2** veces (posiciones 0-1 y 1-2).
- Fusionar: sale **una** sustitución → `[256, 1]`. Al consumir las posiciones 0 y 1, el
  `1` que queda en la posición 2 ya no tiene con quién emparejarse.

**El `i < n - 1`** evita mirar `ids[i+1]` cuando estás en el último elemento. Sin él, un
`IndexError` en cuanto la lista acabe justo en el primer elemento del par.

## Ejercicio 3 — `train_bpe`

```python
if vocab_size < 256:
    raise ValueError(...)

chunks = [text] if pattern is None else regex.findall(pattern, text)
ids = [list(chunk.encode("utf-8")) for chunk in chunks if chunk]

merges, vocab = {}, {i: bytes([i]) for i in range(256)}

for i in range(vocab_size - 256):
    stats = {}
    for chunk_ids in ids:
        get_stats(chunk_ids, stats)      # acumula sobre el mismo dict
    if not stats:
        break

    pair = max(stats, key=lambda p: (stats[p], p))
    new_id = 256 + i

    ids = [merge(c, pair, new_id) for c in ids]
    merges[pair] = new_id
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

return merges, vocab
```

**El `break` cuando no quedan pares.** Si pides 4096 merges sobre un texto de 20 caracteres,
en algún momento cada chunk queda reducido a un único token y no hay pares que contar. Sin
el `break`, `max()` sobre un diccionario vacío lanza `ValueError`. El test
`test_para_si_se_queda_sin_pares_que_fusionar` cubre este caso, y no es teórico: pasa en
cuanto experimentas con textos cortos.

**El desempate `(stats[p], p)`.** Python compara tuplas elemento a elemento: primero la
frecuencia y, si empata, el par. Cuál gane da igual para la calidad del tokenizador, pero
tiene que ser **determinista y el mismo que la referencia**. Si dejaras `max(stats,
key=stats.get)`, el ganador dependería del orden de inserción del diccionario, que a su vez
depende del orden en que recorriste los chunks. Funcionaría, pero cualquier cambio
inofensivo rompería la reproducibilidad.

**`vocab[new_id] = vocab[a] + vocab[b]`** es concatenación de `bytes`, no de `str`. Por eso
el vocabulario se construye a la vez que los merges: cada token nuevo se define en términos
de los dos que lo forman, y de forma recursiva acaba siendo una secuencia de bytes concreta.

**El filtro `if chunk`** descarta los trozos vacíos que la regex puede producir. Un chunk
vacío da una lista vacía que no aporta nada pero se recorre en cada uno de los miles de
merges.

### Sobre el rendimiento

Esta implementación es $O(\text{merges} \times \text{longitud del texto})$: en cada merge
recorre el corpus entero dos veces. Para 4096 merges sobre 2 GB serían días.

Es una decisión consciente: el código está escrito para entenderse. Las implementaciones
serias mantienen índices incrementales de dónde aparece cada par y solo actualizan lo que
cambia. Por eso el módulo 04 entrena los merges sobre una **muestra** (~150 MB) y luego
codifica el corpus completo con multiprocessing. Entrenar sobre una muestra apenas cambia
los merges resultantes: las frecuencias relativas de los pares se estabilizan mucho antes
de haber visto todo el texto.

## Ejercicio 4 — `bpe_encode`

```python
def _encode_chunk(ids, merges):
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids
```

**El `min` con `float("inf")`** es el corazón del ejercicio. Los ids de merge son
`256, 257, 258…` en orden de aprendizaje, así que "el id más bajo" equivale a "el que se
aprendió antes". A los pares que no están en `merges` se les asigna infinito, de modo que
nunca ganan el mínimo.

Si el ganador resulta no estar en `merges`, significa que **ninguno** de los pares
presentes es fusionable, y hay que parar.

**Por qué el orden importa tanto.** Es la parte que más cuesta ver. El tokenizador no es
"parte el texto en los trozos más largos posibles": es "reproduce exactamente el proceso de
entrenamiento". Dos tokenizaciones distintas del mismo texto pueden ser ambas válidas como
secuencias de ids, pero solo una es la que el modelo vio millones de veces al entrenar. La
otra le resulta tan extraña como texto en otro idioma.

Hay una consecuencia que sorprende y que el test recoge: con merges `(a,a)→256` y
`(256,a)→257`, la cadena `"aaaa"` da `[256, 256]` y no `[257, a]`. El primer merge se
aplica **a toda la secuencia de golpe** y se lleva las cuatro `a` de dos en dos, así que el
par `(256, a)` nunca llega a formarse. Con tres `a` sí sale `[257]`. No es un bug: es cómo
funciona BPE, y lo mismo pasa en tiktoken.

## Ejercicio 5 — `bpe_decode`

```python
raw = b"".join(vocab[i] for i in ids)
return raw.decode("utf-8", errors="replace")
```

Dos líneas, y las dos tienen su motivo.

**Juntar antes de decodificar.** UTF-8 codifica los caracteres no-ASCII en varios bytes: la
`ñ` es `0xC3 0xB1`. A BPE eso le da completamente igual — trabaja con bytes y no sabe nada
de caracteres — así que puede haber aprendido un token que termina en `0xC3` y otro que
empieza por `0xB1`. Decodificados por separado, ninguno de los dos es válido; juntos, son
una `ñ`. El test `test_decodificar_junta_los_bytes_antes_de_decodificar` construye
exactamente ese caso.

**`errors="replace"`, el bytes fallback.** Un modelo recién inicializado genera ids al azar,
y muchas de esas secuencias no forman UTF-8 válido. Sin el `errors="replace"`, una
`UnicodeDecodeError` tumbaría el bucle de generación entero. Con él sale un `�` y la
generación continúa. Cuando en el módulo 14 veas caracteres raros en las primeras muestras,
ya sabes qué son.

## Lo que deberías ver en la demo

La misma frase, tokenizada con vocabularios crecientes:

```
vocab 300  -> 35 tokens:  T | h | e  | k | ing |   | s | ha | ll  | ...
vocab 1024 -> 20 tokens:  The  | k | ing |  shall  | speak |   | to  | ...
tiktoken   -> 10 tokens:  The |  king |  shall |  speak |  to |  his | ...
```

Y el detalle que más dice: fíjate en que los tokens de tiktoken **empiezan por espacio**
(`" king"`, `" shall"`). No es casualidad, es el pre-tokenizador: el patrón asigna el
espacio previo a la palabra que viene, de forma que `"king"` a principio de frase y
`" king"` en medio son tokens distintos. Es una de las razones por las que los LLM son
sensibles a si tu prompt termina o no en espacio.

---

## El código completo

Si te has atascado, aquí está la implementación entera. **Cópiala, pégala y ejecuta los
tests**: verlos pasar con código que entiendes es mejor que quedarte bloqueado.

Y después vuelve al ejercicio y escríbela tú. Leer una solución que ya has peleado funciona
muy bien; leerla en frío, no funciona nada.

```python
def _encode_chunk(ids: list[int], merges: Merges) -> list[int]:
    while len(ids) >= 2:
        stats = get_stats(ids)
        # El par cuyo merge se aprendio antes (id mas bajo). `inf` para los que no existen.
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def get_stats(ids: Sequence[int], counts: dict[Pair, int] | None = None) -> dict[Pair, int]:
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(
    text: str,
    vocab_size: int,
    pattern: str | None = None,
    verbose: bool = False,
) -> tuple[Merges, Vocab]:
    if vocab_size < 256:
        raise ValueError(f"vocab_size ({vocab_size}) no puede bajar de 256: son los bytes.")

    chunks = [text] if pattern is None else regex.findall(pattern, text)
    ids: list[list[int]] = [list(chunk.encode("utf-8")) for chunk in chunks if chunk]

    merges: Merges = {}
    vocab: Vocab = {i: bytes([i]) for i in range(256)}

    for i in range(vocab_size - 256):
        stats: dict[Pair, int] = {}
        for chunk_ids in ids:
            get_stats(chunk_ids, stats)
        if not stats:
            break  # ya no quedan pares que fusionar

        pair = max(stats, key=lambda p: (stats[p], p))
        new_id = 256 + i

        ids = [merge(chunk_ids, pair, new_id) for chunk_ids in ids]
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

        if verbose:
            print(f"merge {i + 1}/{vocab_size - 256}: {pair} -> {new_id} "
                  f"({vocab[new_id]!r}) x{stats[pair]}")

    return merges, vocab


def bpe_encode(text: str, merges: Merges, pattern: str | None = None) -> list[int]:
    chunks = [text] if pattern is None else regex.findall(pattern, text)
    out: list[int] = []
    for chunk in chunks:
        out.extend(_encode_chunk(list(chunk.encode("utf-8")), merges))
    return out


def bpe_decode(ids: Iterable[int], vocab: Vocab) -> str:
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")
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
def _encode_chunk(ids: list[int], merges: Merges) -> list[int]:
    while len(ids) >= 2:
        stats = get_stats(ids)
        # The pair whose merge was learned first (lowest id). `inf` for those that do not exist.
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def get_stats(ids: Sequence[int], counts: dict[Pair, int] | None = None) -> dict[Pair, int]:
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: Sequence[int], pair: Pair, new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(
    text: str,
    vocab_size: int,
    pattern: str | None = None,
    verbose: bool = False,
) -> tuple[Merges, Vocab]:
    if vocab_size < 256:
        raise ValueError(f"vocab_size ({vocab_size}) cannot go below 256: those are the bytes.")

    chunks = [text] if pattern is None else regex.findall(pattern, text)
    ids: list[list[int]] = [list(chunk.encode("utf-8")) for chunk in chunks if chunk]

    merges: Merges = {}
    vocab: Vocab = {i: bytes([i]) for i in range(256)}

    for i in range(vocab_size - 256):
        stats: dict[Pair, int] = {}
        for chunk_ids in ids:
            get_stats(chunk_ids, stats)
        if not stats:
            break  # there are no pairs left to merge

        pair = max(stats, key=lambda p: (stats[p], p))
        new_id = 256 + i

        ids = [merge(chunk_ids, pair, new_id) for chunk_ids in ids]
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

        if verbose:
            print(f"merge {i + 1}/{vocab_size - 256}: {pair} -> {new_id} "
                  f"({vocab[new_id]!r}) x{stats[pair]}")

    return merges, vocab


def bpe_encode(text: str, merges: Merges, pattern: str | None = None) -> list[int]:
    chunks = [text] if pattern is None else regex.findall(pattern, text)
    out: list[int] = []
    for chunk in chunks:
        out.extend(_encode_chunk(list(chunk.encode("utf-8")), merges))
    return out


def bpe_decode(ids: Iterable[int], vocab: Vocab) -> str:
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
