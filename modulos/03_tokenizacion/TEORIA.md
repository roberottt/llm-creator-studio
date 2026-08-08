# 03 — Tokenización y BPE

## Por qué importa este módulo

**Porque una red neuronal no sabe leer.**

Sólo hace cuentas con números, así que el texto hay que convertirlo en una lista de enteros
antes de que el modelo lo vea. Y cómo lo trocees no es un detalle de fontanería: decide
cuántos parámetros tendrá tu modelo, cuánto texto le cabe en la ventana, y qué cosas le
saldrán raras.

Aquí construyes el tokenizador que usa el modelo final: BPE desde cero, con un vocabulario
de 4096 tokens entrenado sobre el corpus. No es una librería que llamas: son 60 líneas que
escribes tú.

Y de paso entenderás por qué los LLM fallan contando las letras de una palabra, por qué se
les da mal la aritmética, y por qué el español sale más caro que el inglés.

### Qué sabrás al terminar

- Por qué no se usan ni caracteres ni palabras, sino trozos de palabra
- Cómo BPE **aprende solo** qué trozos merecen ser un token, sin que nadie se lo diga
- Por qué se trabaja con bytes y no con caracteres (spoiler: para que no exista el token
  desconocido)
- **Por qué 4096 y no 50.000**, con los números que lo justifican

### Qué vas a escribir

Cinco funciones. Esta teoría está ordenada para que las leas en este orden, y cada una
tiene su propia sección con el ejemplo numérico correspondiente:

| Ejercicio | Qué hace | Dónde se explica |
|---|---|---|
| 1. `get_stats` | Contar qué par de vecinos se repite más | [§ Contar los pares](#ejercicio-1-contar-los-pares-get_stats) |
| 2. `merge` | Sustituir un par por un token nuevo | [§ Fusionar el par ganador](#ejercicio-2-fusionar-el-par-ganador-merge) |
| 3. `train_bpe` | Repetir 1 y 2 hasta llenar el vocabulario | [§ El bucle de entrenamiento](#ejercicio-3-el-bucle-de-entrenamiento-train_bpe) |
| 4. `bpe_encode` | Texto → ids, reproduciendo el entrenamiento | [§ Codificar](#ejercicio-4-codificar-bpe_encode) |
| 5. `bpe_decode` | ids → texto | [§ Decodificar](#ejercicio-5-decodificar-bpe_decode) |

Los dos primeros son cortos y mecánicos, y son los ladrillos de todo lo demás. El tercero
es el central. Los dos últimos usan lo que aprendió el tercero.

### Cuánto cuesta

4 horas. Es el módulo más largo de la Parte I, y el tokenizador que salga de aquí es el que
usarás el resto del curso.

---

## El problema: una red no sabe leer

Una red neuronal solo hace cuentas con números. El texto hay que convertirlo en una lista
de enteros antes de que el modelo lo vea, y **cómo lo trocees condiciona todo lo demás**:
cuántos parámetros tendrá el modelo, cuánto texto le cabe en la ventana y qué cosas le
saldrán raras.

La pregunta es: ¿cuál es la unidad? ¿Letras? ¿Palabras? ¿Algo intermedio?

## Opción A: por caracteres

Asignas un número a cada carácter distinto. Con Shakespeare salen 65 símbolos:

```
"gato"  ->  'g'=45, 'a'=34, 't'=58, 'o'=52  ->  [45, 34, 58, 52]
```

**A favor:** el vocabulario es diminuto y nunca hay un carácter desconocido.

**En contra:** las secuencias se vuelven larguísimas. Una historia de 200 palabras son unos
1000 caracteres. Y aquí está el problema serio: el coste de la atención crece con el
**cuadrado** de la longitud de la ventana (módulo 06). Doblar la longitud cuadruplica el
coste. Trocear fino sale caro.

Además obligas al modelo a gastar capacidad en aprender a deletrear antes de poder aprender
nada sobre el significado.

## Opción B: por palabras

Una entrada por palabra del diccionario.

**A favor:** secuencias cortas, y cada token ya significa algo.

**En contra:** dos problemas gordos. El primero, el vocabulario se dispara: 50.000 palabras
como mínimo para inglés, y muchas más para español con su conjugación. El segundo es peor:
**¿qué haces con una palabra que no está?** Nombres propios, erratas, palabras nuevas. La
respuesta clásica era un token `<UNK>` que destruye información sin remedio.

## Opción C: trozos de palabra, aprendidos de los datos

La idea de **BPE** (*Byte Pair Encoding*): que las palabras frecuentes sean un solo token y
las raras se partan en piezas. Ni caracteres ni palabras: lo que los datos digan.

Y lo interesante es que nadie escribe la lista de trozos. **Se descubre contando.** El
algoritmo empieza con las unidades más pequeñas posibles y va **fusionando el par de
vecinos que más se repite**, una y otra vez, hasta tener tantos tokens como le pidas.

### El ejemplo, paso a paso

Texto: `aaabdaaabac`. Empezamos con los bytes (`a`=97, `b`=98, `c`=99, `d`=100):

```
[97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]
```

**Paso 1.** Contamos cada par de vecinos:

```
(a,a) -> 4 veces      (b,d) -> 1
(a,b) -> 2            (d,a) -> 1
                      (b,a) -> 1     (a,c) -> 1
```

Gana `(a,a)`. Le damos el número 256 (los del 0 al 255 ya están cogidos por los bytes) y
sustituimos, de izquierda a derecha y sin solapar:

```
aaabdaaabac  ->  [256] a b d [256] a b a c
```

**Paso 2.** Volvemos a contar **sobre el resultado**, no sobre el texto original. Esto es
importante: los pares se cuentan siempre sobre la secuencia tal como ha quedado. Ahora
`(256, a)` sale 2 veces y `(a, b)` también 2. Empate, que se resuelve con una regla fija
(aquí gana el par mayor). Sale `(256, a)`, que pasa a ser el 257 y representa `"aaa"`:

```
[257] b d [257] b a c   ->   [257, 98, 100, 257, 98, 97, 99]
```

De 11 números hemos pasado a 7, y hemos aprendido dos "palabras" que nadie nos dijo:
`"aa"` y `"aaa"`. Puedes comprobar los dos pasos ejecutando la referencia con
`verbose=True`; imprime exactamente esto:

```
merge 1/2: (97, 97) -> 256 (b'aa') x4
merge 2/2: (256, 97) -> 257 (b'aaa') x2
```

Con un texto de verdad y miles de merges, lo que aprende son cosas como `" the"`, `"ing"`
o `" que"`. Aquí están los **quince primeros merges reales** sobre 200.000 caracteres de
Shakespeare, salidos de ejecutar `train_bpe`:

| # | id | token | se forma con |
|---|---|---|---|
| 1 | 256 | `'e '` | `'e'` + `' '` |
| 2 | 257 | `'th'` | `'t'` + `'h'` |
| 3 | 258 | `'t '` | `'t'` + `' '` |
| 4 | 259 | `'s '` | `'s'` + `' '` |
| 5 | 260 | `'ou'` | `'o'` + `'u'` |
| 6 | 261 | `'d '` | `'d'` + `' '` |
| 7 | 262 | `', '` | `','` + `' '` |
| 8 | 263 | `'er'` | `'e'` + `'r'` |
| 9 | 264 | `'an'` | `'a'` + `'n'` |
| 10 | 265 | `'in'` | `'i'` + `'n'` |
| … | | | |
| 251 | 506 | `'rom'` | `'r'` + `'om'` |
| 252 | 507 | `"'ll "` | `"'"` + `'ll '` |
| 253 | 508 | `'itiz'` | `'iti'` + `'z'` |
| 254 | 509 | `'itizen'` | `'itiz'` + `'en'` |

(Este entrenamiento va sin pre-tokenizador —lo que hace `llmfs demo 03`— y por eso salen
tokens con el espacio pegado **detrás**, como `'e '` o `'no '`. Con pre-tokenizador el
espacio se queda pegado a la palabra que sigue; lo verás dentro de un momento.)

Fíjate en la progresión: los primeros son pares de letras muy frecuentes, y hacia el final
ya aparecen trozos largos como `'itizen'` — construido a partir de merges anteriores, no de
letras sueltas. Cada token nuevo puede servir de material para el siguiente, y así es como
se llega de las letras a las palabras sin que nadie le explique al algoritmo que las
palabras existen.

---

## Del algoritmo a las cinco funciones

Antes de meterte en los ejercicios conviene tener claro **con qué estructuras de datos se
trabaja**, porque son tres y aparecen en todas las firmas. Si te sientas a programar sin
esto claro, el módulo se hace cuesta arriba.

**1. Los ids: `list[int]`.** El texto, ya convertido en números. Al principio son bytes
(0-255) y según se aplican merges van apareciendo números más altos:

```python
list("gato".encode("utf-8"))     # [103, 97, 116, 111]
```

Nada más. No hay clases, no hay tensores: en este módulo un texto es una lista de enteros
de Python.

**2. `merges: dict[(int, int), int]`.** Las reglas aprendidas, en el orden en que se
aprendieron. La clave es el par que se fusiona, el valor es el id nuevo:

```python
{(97, 97): 256, (256, 97): 257}
```

Se lee: «cuando veas un 97 seguido de otro 97, cámbialos por un 256». Y como los ids nuevos
se reparten en orden (256, 257, 258…), **el id dice también cuándo se aprendió la regla**.
Ese detalle, que parece contable, es lo que hace funcionar el ejercicio 4.

**3. `vocab: dict[int, bytes]`.** La tabla de significados: qué bytes representa cada id.
Arranca con los 256 bytes y crece un token por merge:

```python
{0: b'\x00', ..., 97: b'a', ..., 256: b'aa', 257: b'aaa'}
```

`merges` sirve para **codificar** y `vocab` para **decodificar**. Son dos vistas de lo
mismo y por eso `train_bpe` devuelve las dos.

Con eso, las cinco funciones encajan así:

```
    get_stats  (ej. 1) ─┐
                        ├─> train_bpe (ej. 3) ──> merges ──> bpe_encode (ej. 4)  texto -> ids
    merge      (ej. 2) ─┘                    └──> vocab  ──> bpe_decode (ej. 5)  ids -> texto
```

Y hay una asimetría que conviene tener presente desde el principio: **entrenar se hace una
vez, codificar se hace millones de veces**. `train_bpe` puede tardar minutos porque solo se
ejecuta al preparar los datos; `bpe_encode` se aplicará a cada texto que entre al modelo.

---

## Ejercicio 1: contar los pares (`get_stats`)

Es la parte "medir" del algoritmo: recorrer la secuencia y apuntar cuántas veces aparece
cada par de vecinos.

```python
get_stats([97, 97, 97, 98])  ->  {(97, 97): 2, (97, 98): 1}
```

**El detalle que hay que interiorizar: al contar, los pares se solapan.** En `[97, 97, 97]`
el par `(97, 97)` sale **dos** veces, una en las posiciones 0-1 y otra en las 1-2. En
`[1, 1, 1, 1]` sale tres veces. No estás partiendo la lista en parejas: estás mirando por
una ventanita de dos elementos que avanza de uno en uno.

```
[1, 1, 1, 1]
 └──┘             (0,1)
    └──┘          (1,2)
       └──┘       (2,3)      ->  {(1,1): 3}
```

Guarda esa imagen, porque en el ejercicio 2 la ventana avanza de otra manera y esa
diferencia es la trampa clásica del módulo.

**Por qué la función recibe un `counts` opcional.** Porque `train_bpe` no cuenta sobre un
texto seguido, sino sobre una lista de trozos, y quiere la suma de todos **sin** contar los
pares que cruzan de un trozo al siguiente (ya verás por qué en el ejercicio 3). Con un
acumulador es trivial:

```python
stats = {}
for chunk in chunks:
    get_stats(chunk, stats)     # va sumando sobre el mismo diccionario
```

Si `get_stats` solo supiera devolver diccionarios nuevos, habría que fusionarlos a mano
cada vuelta. Devuelve el diccionario *además* de mutarlo, así vale para los dos usos.

## Ejercicio 2: fusionar el par ganador (`merge`)

Es la parte "actuar": recorrer la secuencia y sustituir cada aparición del par por un solo
número nuevo.

```python
merge([97, 97, 97, 98, 97, 97], (97, 97), 256)  ->  [256, 97, 98, 256]
```

**Y aquí, al contrario que al contar, las apariciones NO se solapan.** Cuando encuentras el
par lo consumes entero y saltas dos posiciones. Míralo en `[1, 1, 1]` fusionando `(1,1)`:

```
[1, 1, 1]
 └──┘        coincide -> escribes 256 y saltas a la posicion 2
       ↑     queda un 1 suelto, sin pareja
                                              ->  [256, 1]   y NO [256, 256]
```

De ahí sale la única decisión de diseño del ejercicio: **un `while` con un índice tuyo, no
un `for`**. Un `for` avanza de uno en uno siempre; tú necesitas avanzar de uno *o* de dos
según haya coincidencia.

Ese comportamiento no es un capricho de la implementación. Es lo que hace que el conteo del
paso siguiente sea coherente: después de fusionar, la secuencia es más corta y hay pares
nuevos que antes no existían (el 256 ahora tiene vecinos). Volver a contar sobre esa
secuencia es exactamente el paso 2 del ejemplo de `aaabdaaabac`.

## Ejercicio 3: el bucle de entrenamiento (`train_bpe`)

Aquí se junta todo. El cuerpo del bucle es literalmente el ejemplo de `aaabdaaabac`
repetido `vocab_size - 256` veces:

```
repetir hasta llenar el vocabulario:
    1. contar todos los pares                    (get_stats)
    2. quedarse con el más frecuente
    3. asignarle el siguiente id libre
    4. sustituirlo en toda la secuencia          (merge)
    5. apuntar la regla en `merges` y el significado en `vocab`
```

Hay cuatro cosas que no se ven en ese esquema y que son las que dan problemas.

### Por qué el texto se parte antes en trozos

Si dejas a BPE contar libremente sobre el texto entero, aprende tokens que cruzan de una
palabra a la siguiente. No es una preocupación teórica: esto es lo que aprende de verdad
sobre `"el gato come pescado. el perro come carne. el gato duerme."` repetido ocho veces,
**sin** pre-tokenizador, mirando los veinte primeros merges:

```
'o ', 'me', 'l ', 'el ', '. ', '. el ', 'me ', 'o c', 'o co', 'o come ',
'pe', 'ga', 'gat', 'ca', '. el gat', '. el gato ', '. el gato d',
'. el gato du', '. el gato due', '. el gato duer'
```

Siete de los veinte tokens son prefijos de la misma frase, `'. el gato duer'`. El algoritmo
está memorizando una cadena concreta del corpus y gastando el vocabulario en ella.

Y esto es lo que aprende sobre el mismo texto **con** el pre-tokenizador:

```
'me', 'el', ' c', ' el', ' co', ' come', 'to', 'pe', 'ga', 'gato',
' gato', ' pe', ' pes', ' pesc', ' pesca', ' pescad', ' pescado',
' per', ' perr', ' perro'
```

Palabras. `' gato'`, `' come'`, `' pescado'`, `' perro'`. Es el mismo algoritmo, el mismo
texto y el mismo número de merges: lo único que cambia es que ahora los pares **no se
cuentan a través de las fronteras entre trozos**.

Por eso el paso 5a del docstring cuenta trozo a trozo sobre un mismo diccionario en vez de
concatenarlo todo. Y por eso `get_stats` recibe un acumulador.

Un detalle que conviene notar de esos ejemplos: los tokens empiezan por espacio
(`' gato'`, no `'gato'`). El pre-tokenizador deja el espacio pegado a la palabra que
*sigue*, así que en el vocabulario `' gato'` y `'gato'` son dos tokens distintos. Es la
razón de que en las demos de tokenizadores veas siempre esos espacios delante.

### El desempate tiene que ser determinista

```python
pair = max(stats, key=lambda p: (stats[p], p))
```

Ese `key` devuelve una **tupla**, y Python compara tuplas elemento a elemento: primero la
frecuencia y, si empata, el par en sí. En el ejemplo de `aaabdaaabac` había un empate real
a 2 entre `(256, 97)` y `(97, 98)`, y esta regla es la que decide que gane el primero.

Cuál gane da igual para la calidad del tokenizador. Lo que **no** da igual es que sea
siempre el mismo. Si usaras `max(stats, key=stats.get)`, el ganador dependería del orden de
inserción del diccionario, y en cuanto hubiera un empate tus merges divergirían de los de
la referencia — y con ellos todos los ids, y el test fallaría por un motivo que no tiene
nada que ver con haber entendido BPE.

### El `break` cuando no quedan pares

Si pides 4096 merges sobre un texto de veinte caracteres, llega un momento en que cada
trozo queda reducido a un solo token y ya no hay ningún par que contar. `stats` sale vacío
y `max()` sobre un diccionario vacío lanza `ValueError`. Hay un test que cubre justo ese
caso.

### El coste

Esta implementación recorre el corpus **entero** en cada merge. Los tiempos reales, sobre
200.000 caracteres de Shakespeare:

| vocabulario | merges | tiempo |
|---|---|---|
| 300 | 44 | 1 s |
| 512 | 256 | 3 s |
| 1.024 | 768 | 7 s |
| 2.048 | 1.792 | 14 s |

Crece linealmente con el número de merges y linealmente con el tamaño del corpus: 4096
merges sobre los gigabytes de TinyStories serían días. Es una decisión consciente —el
código está escrito para entenderse, no para correr—, y por eso el módulo 04 entrena los
merges sobre una muestra y luego codifica el corpus completo con ellos. Los tokenizadores
de producción hacen lo mismo, pero con la parte cara escrita en Rust.

## Ejercicio 4: codificar (`bpe_encode`)

Ya tienes las reglas. Ahora, texto nuevo → ids.

La tentación es pensar que codificar es «busca los trozos más largos que estén en el
vocabulario». **No lo es.** Codificar es *reproducir el proceso de entrenamiento*: aplicar
los merges aprendidos, en el mismo orden en que se aprendieron.

### Por qué el orden

Los merges se aprendieron encadenados: el 257 puede estar hecho con el 256. Si aplicas
primero el 257 sin haber creado los 256 que hacen falta, el resultado no es el mismo texto
mal codificado — es una tokenización *distinta*, perfectamente válida como lista de ids,
pero que el modelo no ha visto nunca. Y al modelo eso le resulta tan extraño como texto en
otro idioma: los mismos caracteres repartidos en piezas que no reconoce.

Por eso el bucle busca, en cada vuelta, **el par presente cuyo merge se aprendió antes**:

```python
pair = min(stats, key=lambda p: merges.get(p, float("inf")))
if pair not in merges:
    break
```

Aquí es donde se cobra aquel detalle de que los ids se reparten en orden. «Aprendido antes»
es lo mismo que «id más bajo», así que el orden de aprendizaje se recupera con un `min` y no
hace falta guardar ninguna marca de tiempo. Y `merges.get(p, float("inf"))` le da infinito a
los pares que no son fusionables, de forma que nunca ganan el `min`. Si aun así gana uno de
ellos, significa que **ninguno** de los pares presentes se puede fusionar: hemos terminado.

### Una consecuencia que sorprende

Con los merges `(a,a) -> 256` y `(256,a) -> 257`, ¿cuántos tokens da `"aaaa"`?

Lo intuitivo sería `[257, a]`: coge el trozo más largo que puedas, `"aaa"`, y deja la
sobrante. Pero lo que sale es `[256, 256]`, y puedes comprobarlo:

```python
>>> bpe_encode("aaa",  {(97,97): 256, (256,97): 257})
[257]
>>> bpe_encode("aaaa", {(97,97): 256, (256,97): 257})
[256, 256]
```

El motivo es que **cada merge se aplica a toda la secuencia de golpe**. El 256 es el más
antiguo, así que va primero, y se lleva las cuatro `a` de dos en dos. Cuando le toca el
turno al `(256, a)`, ya no queda ninguna `a` suelta con la que formar el par. Con tres `a`
sí sale `[257]`, porque el primer merge deja una suelta.

No es un bug del curso: es cómo funciona BPE, y tiktoken hace exactamente lo mismo. Hay un
test que lo documenta.

### Y no olvides el patrón

Codificar tiene que trocear el texto con **el mismo** patrón con el que entrenaste. Si
entrenas con pre-tokenizador y codificas sin él (o al revés), los pares que se te forman no
son los mismos y los ids no cuadran. No da error: da resultados peores por un motivo
invisible.

Así queda una frase real, con un vocabulario de 512 entrenado sobre Shakespeare **sin**
pre-tokenizador (por eso los espacios van pegados detrás, `'he '`, y no delante):

```
The king shall speak to his people tomorrow morning.

'T' | 'he ' | 'k' | 'ing ' | 'shall ' | 'sp' | 'ea' | 'k ' | 'to ' | 'his ' |
'pe' | 'op' | 'l' | 'e ' | 't' | 'om' | 'or' | 'r' | 'ow' | ' m' | 'or' | 'n' | 'ing' | '.'
```

52 caracteres, **24 tokens**. Palabras frecuentes en la obra (`'shall '`, `'his '`) salen
enteras; `'tomorrow'` se parte en cinco piezas. Con 4096 tokens entrenados sobre el corpus
de verdad la mayoría de palabras comunes serán un solo token.

## Ejercicio 5: decodificar (`bpe_decode`)

Dos líneas, y el orden de las dos líneas es todo el ejercicio:

```python
raw = b"".join(vocab[i] for i in ids)
return raw.decode("utf-8", errors="replace")
```

**Primero juntar todos los bytes, después decodificar una sola vez.** Lo que no puedes
hacer es esto:

```python
"".join(vocab[i].decode("utf-8") for i in ids)      # MAL
```

Motivo: UTF-8 codifica los caracteres no-ASCII en varios bytes. Una `n` es un byte, pero
una `ñ` son dos. Mira los bytes reales de `"ñandú café"`:

```
[195, 177, 97, 110, 100, 195, 186, 32, 99, 97, 102, 195, 169]
  └───┬───┘                └───┬───┘              └───┬───┘
      ñ                        ú                      é
```

A BPE eso le da completamente igual: trabaja con bytes y no sabe nada de caracteres. Puede
perfectamente haber aprendido un token que **acaba** en 195 y otro que **empieza** por 177.
Decodificados por separado, ninguno de los dos es UTF-8 válido y `.decode()` revienta.
Juntos, son una `ñ`. Hay un test que construye exactamente ese caso.

**Y por qué `errors="replace"`.** Es lo que se llama *bytes fallback*. Un modelo a medio
entrenar genera secuencias de ids cualesquiera, y muchas no forman UTF-8 válido. Con
`errors="replace"` sale un `�` donde no se pudo decodificar y la generación continúa; sin
él, una excepción tumbaría el bucle de generación entero por un byte suelto. Cuando en el
módulo 14 veas caracteres raros en las primeras muestras, ya sabes lo que son.

---

## Por qué bytes y no caracteres

Ya has visto la mecánica; queda la razón de fondo de empezar por los 256 bytes en lugar de
por los caracteres Unicode: **no existe el texto no codificable**. Cualquier cosa —un emoji,
chino, un binario mal pegado— es una secuencia de bytes, y los 256 bytes están todos en el
vocabulario desde el primer momento. El token `<UNK>` desaparece del problema, y con él una
familia entera de fallos.

Si en cambio arrancaras con los caracteres Unicode, tendrías que decidir cuáles caben (son
más de 150.000) y qué hacer con el resto.

El precio es el que ya has visto: un carácter no-ASCII cuesta varios tokens en el peor caso,
y por eso los idiomas con acentos, y más aún los que no usan alfabeto latino, salen más
caros.

## El pre-tokenizador, por dentro

El patrón que usamos es el de GPT-4 (`cl100k_base`), y está en
`llmfs/reference/tokenizer.py` como `GPT4_SPLIT_PATTERN`. No hace falta entenderlo entero,
pero sí ver qué hace:

```python
regex.findall(GPT4_SPLIT_PATTERN, "Hola, mundo!")
['Hola', ',', ' mundo', '!']

regex.findall(GPT4_SPLIT_PATTERN, "El gato come pescado.")
['El', ' gato', ' come', ' pescado', '.']

regex.findall(GPT4_SPLIT_PATTERN, "en 2026 habia 1234 gatos")
['en', ' ', '202', '6', ' habia', ' ', '123', '4', ' gatos']
```

Separa palabras (con su espacio delante), signos de puntuación y espacios sueltos. Y los
números en grupos **de tres dígitos como máximo**: fíjate en que `2026` sale como `'202'` y
`'6'`. Es deliberado — evita que el tokenizador aprenda un token para cada año o cada
cantidad frecuente — y de paso es una de las razones por las que a los modelos se les da mal
la aritmética: `1234` no es «mil doscientos treinta y cuatro» para el modelo, son dos piezas
partidas por un sitio que no significa nada.

Necesita el módulo `regex` y no el `re` de la biblioteca estándar, porque usa clases Unicode
(`\p{L}` = «cualquier letra», `\p{N}` = «cualquier dígito») y cuantificadores posesivos
(`++`, `?+`) que `re` no soporta.

## Por qué 4096 y no 50.000

Aquí es donde el tokenizador se convierte en una decisión de arquitectura. La tabla de
embeddings —la que traduce cada id a un vector; aparece en el módulo 05 y la montas entera
en el módulo 10— tiene
`vocab_size × d_model` parámetros. Con nuestro modelo, `d_model = 320`:

| vocabulario | parámetros en embeddings | % del modelo |
|---|---|---|
| 4.096 | 4096 × 320 = **1,31 M** | 15% de 8,9M |
| 32.000 | 32000 × 320 = **10,2 M** | más que todo el resto del modelo |
| 50.257 (GPT-2) | 50257 × 320 = **16,1 M** | el modelo sería casi solo embeddings |

Con un modelo pequeño, un vocabulario grande es un desastre por dos motivos. El obvio: te
gastas los parámetros en una tabla de consulta en vez de en las capas que razonan. Y el que
se ve menos: cada fila de esa tabla solo se entrena cuando su token aparece en el texto, así
que con 50.000 filas y un corpus modesto habría miles de tokens vistos cuatro veces, con
vectores prácticamente aleatorios.

El precio de un vocabulario pequeño es la **compresión**. Medida sobre Shakespeare, con
nuestro código:

| vocabulario | bytes por token |
|---|---|
| 300 | 1,42 |
| 512 | 2,05 |
| 1.024 | 2,74 |
| 2.048 | 3,48 |

Cuanto más pequeño el vocabulario, más tokens necesitas para el mismo texto, y por tanto más
pasos de entrenamiento y menos texto real cabe en la ventana de 512.

Fíjate también en que la curva se aplana rápido, y esto es lo que más tranquiliza a la hora
de elegir 4096: sobre ese mismo texto, `cl100k_base` —el tokenizador de GPT-4, con **100.277
tokens**— consigue 3,67 bytes por token. Nuestro vocabulario de 2.048 llega a 3,48. Cincuenta
veces más vocabulario para un 5% más de compresión. Los rendimientos decrecientes son
brutales, y el precio en parámetros no lo es. `llmfs demo 03` dibuja esas dos curvas juntas.

Es un intercambio directo: **parámetros en la tabla contra longitud de las secuencias**. Con
9M de parámetros, 4096 es un punto razonable; no es la única respuesta defendible.

Un aviso práctico que verás en el módulo 04: como TinyStories tokenizado con 4096 comprime
peor que con los 50k de GPT-2, el corpus dará bastantes más tokens de los ~470M que se citan
habitualmente. Ese número se mide, no se supone.

## Qué hacen los tokenizadores de verdad que tú no

Lo que escribes aquí es BPE completo y correcto, y es el que usa el modelo final. Las
diferencias con `tiktoken` o `sentencepiece` son de ingeniería y de acabado:

| tu código | un tokenizador de producción |
|---|---|
| `train_bpe` recorre el corpus en cada merge | índices incrementales; solo recuenta lo que cambió |
| Python puro | el núcleo en Rust o C++ |
| `bpe_encode` sin caché | caché por palabra: la mayoría de textos repiten las mismas |
| sin tokens especiales | `<|endoftext|>`, `<|im_start|>`… reservados fuera del BPE |
| vocabulario de 4096 | 100.000 o más |

El único de esos que te va a hacer falta en el curso es la caché, y aparece en el módulo 04
al preparar TinyStories.

## Dónde está el debate

La tokenización es probablemente la parte más fea de los LLM modernos, y hay bastante gente
que piensa que debería desaparecer.

Muchas rarezas conocidas salen de aquí. Que los modelos fallen contando letras de una
palabra: no ven letras, ven trozos, y `'itizen'` es para el modelo un símbolo tan indivisible
como lo es `'a'` para nosotros. Que la aritmética se les dé mal: ya has visto lo que el
pre-tokenizador hace con `1234`. Que los idiomas distintos del inglés cuesten más caros: el
mismo texto necesita más tokens, y a igualdad de ventana cabe menos.

Hay líneas de investigación activas hacia modelos que trabajen directamente sobre bytes, sin
tokenizador. Todavía no han desplazado a BPE, entre otras cosas por el coste cuadrático de
la atención sobre secuencias tan largas. Es un problema abierto de verdad.

---

**Para ampliar:** Sennrich et al. 2016,
[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
(el paper que trajo BPE al lenguaje) · Karpathy,
[minbpe](https://github.com/karpathy/minbpe) y su vídeo, muy recomendable después de hacer
los ejercicios. Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
