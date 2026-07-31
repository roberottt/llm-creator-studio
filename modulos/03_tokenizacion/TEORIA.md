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
"gato"  ->  [45, 34, macarrones...]   en realidad:  'g'=45, 'a'=34, 't'=58, 'o'=52
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

El algoritmo es sorprendentemente simple. Empiezas con las unidades más pequeñas posibles y
vas **fusionando el par que más se repite**, una y otra vez.

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

**Paso 2.** Volvemos a contar sobre el resultado. Ahora `(256, a)` sale 2 veces y `(a, b)`
también 2. Empate, que se resuelve con una regla fija (aquí gana el par mayor). Sale
`(256, a)`, que pasa a ser el 257 y representa `"aaa"`:

```
[257] b d [257] b a c   ->   [257, 98, 100, 257, 98, 97, 99]
```

De 11 números hemos pasado a 7, y hemos aprendido dos "palabras" que nadie nos dijo:
`"aa"` y `"aaa"`. Con un texto de verdad y 4096 merges, lo que aprende son cosas como
`" the"`, `"ing"` o `" que"`.

### Codificar y decodificar

Para **codificar** texto nuevo, aplicas los merges aprendidos **en el orden en que se
aprendieron**. Ese detalle importa: si los aplicas en otro orden sale una tokenización
distinta, válida pero incompatible con la que vio el modelo al entrenar.

Para **decodificar**, concatenas los bytes de cada token y decodificas al final. No token a
token: un token puede cortar un carácter multibyte por la mitad (una `ñ` son dos bytes y
BPE no sabe nada de eso), así que decodificar por separado fallaría.

## Por qué bytes y no caracteres

Empezar por los 256 bytes en lugar de por los caracteres Unicode tiene una consecuencia
enorme: **no existe el texto no codificable**. Cualquier cosa —emoji, chino, un binario mal
pegado— es una secuencia de bytes, y los 256 bytes están todos en el vocabulario. El token
`<UNK>` desaparece del problema.

Al decodificar sí puede pasar que salga una secuencia de bytes que no es UTF-8 válido (un
modelo a medio entrenar los produce constantemente). Para eso está
`errors="replace"`: sale un `�` en vez de una excepción que tumbe la generación.

## El pre-tokenizador: por qué no se cuenta sobre el texto entero

Si dejas a BPE contar libremente sobre todo el texto, aprende tokens como `"perro."` o
`" el gato"`, que mezclan puntuación y palabras y desperdician vocabulario en
combinaciones que no significan nada.

La solución es partir el texto antes con una expresión regular, y **contar los pares solo
dentro de cada trozo**, nunca a través de las fronteras. El patrón que usamos es el de
GPT-4: separa palabras, números (en grupos de 3 dígitos como máximo), signos de puntuación
y espacios. Necesita el módulo `regex` y no el `re` de la biblioteca estándar, porque usa
clases Unicode (`\p{L}` = "cualquier letra") y cuantificadores posesivos.

## Por qué 4096 y no 50.000

Aquí es donde el tokenizador se convierte en una decisión de arquitectura. La tabla de
embeddings tiene `vocab_size × d_model` parámetros. Con nuestro modelo:

| vocabulario | parámetros en embeddings | % del modelo |
|---|---|---|
| 4.096 | 4096 × 320 = **1,31 M** | 15% de 8,9M |
| 32.000 | 32000 × 320 = **10,2 M** | más que todo el resto del modelo |
| 50.257 (GPT-2) | 50257 × 320 = **16,1 M** | el modelo sería casi solo embeddings |

Con un modelo pequeño, un vocabulario grande es un desastre: te gastas los parámetros en
una tabla de consulta en vez de en las capas que razonan. Y encima cada fila se vería poquísimas
veces durante el entrenamiento, así que aprendería mal.

El precio es la **compresión**. Medida sobre Shakespeare, con nuestro código:

| vocabulario | bytes por token |
|---|---|
| 300 | 1,42 |
| 512 | 2,05 |
| 1.024 | 2,74 |

Cuanto más pequeño el vocabulario, más tokens necesitas para el mismo texto, y por tanto
más pasos de entrenamiento y menos texto real cabe en la ventana de 512. Es un intercambio
directo: **parámetros en la tabla contra longitud de las secuencias**. Con 9M de parámetros,
4096 es un punto razonable; no es la única respuesta defendible.

Un aviso práctico que verás en el módulo 04: como TinyStories tokenizado con 4096 comprime
peor que con los 50k de GPT-2, el corpus dará bastantes más tokens de los ~470M que se
citan habitualmente. Ese número se mide, no se supone.

## Dónde está el debate

La tokenización es probablemente la parte más fea de los LLM modernos, y hay bastante gente
que piensa que debería desaparecer.

Muchas rarezas conocidas salen de aquí. Que los modelos fallen contando letras de una
palabra: no ven letras, ven trozos. Que la aritmética se les dé mal: `327` puede ser un
token y `328` tres. Que los idiomas distintos del inglés cuesten más caros: el mismo texto
necesita más tokens, y a igualdad de ventana cabe menos.

Hay líneas de investigación activas hacia modelos que trabajen directamente sobre bytes,
sin tokenizador. Todavía no han desplazado a BPE, entre otras cosas por el coste cuadrático
de la atención sobre secuencias tan largas. Es un problema abierto de verdad.

---

**Para ampliar:** Sennrich et al. 2016,
[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
(el paper que trajo BPE al lenguaje) · Karpathy,
[minbpe](https://github.com/karpathy/minbpe) y su vídeo, muy recomendable después de hacer
los ejercicios. Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
