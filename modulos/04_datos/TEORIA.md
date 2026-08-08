# 04 — Datos: de texto a batches en la GPU

## Por qué importa este módulo

**Porque aquí es donde se decide qué aprende el modelo.**

Hasta ahora tienes texto convertido en una lista de números (módulo 03). Lo que no tienes
es una *tarea*: nadie le ha dicho todavía al modelo qué se supone que tiene que hacer con
esos números. Ese salto —de "una tira de enteros" a "una pregunta con su respuesta"— se da
en este módulo, y se da en una sola línea de código.

La línea es tan corta que es fácil pasarla por alto y no ver lo que hay dentro. Lo que hay
dentro es la razón por la que un modelo de lenguaje se puede entrenar con texto que nadie
ha etiquetado: el texto **ya trae la respuesta puesta**, porque la respuesta a "¿qué viene
después?" es, literalmente, lo que viene después. Es la idea que hizo posible entrenar con
internet entero en vez de con un corpus anotado a mano.

Lo demás del módulo es fontanería: cómo guardar 500 millones de tokens sin gastar 4 GB,
cómo leerlos sin cargarlos, y qué trozo apartas para saber si el modelo está aprendiendo o
haciendo trampa. Fontanería que decide cosas reales.

### Qué sabrás al terminar

- Qué es el **aprendizaje autosupervisado** y por qué es lo que hizo despegar a los LLM
- **Por qué una sola ventana de 512 tokens son 512 ejemplos de entrenamiento**, no uno
- Qué parte de un LLM real estás construyendo, y qué hace un laboratorio de verdad en esta
  misma fase que aquí nos saltamos
- Por qué 500 millones de tokens ocupan 1 GB y no 4
- Un bug silencioso de NumPy que te corrompe los datos sin dar ningún error, ni uno
- Por qué el conjunto de validación NO se coge al azar, con el número que lo demuestra
- Si el disco es de verdad tu cuello de botella o es una leyenda que se repite sin medir

### Qué vas a escribir

Tres funciones. Esta teoría está ordenada para que las leas en este orden, y cada una tiene
su propia sección con el ejemplo numérico correspondiente:

| Ejercicio | Qué hace | Dónde se explica |
|---|---|---|
| 1. `pack_tokens_uint16` | Lista de ids → array de 2 bytes por token, validando | [§ Empaquetar el corpus](#ejercicio-1-empaquetar-el-corpus-pack_tokens_uint16) |
| 2. `train_val_split` | Apartar el final del corpus para validación | [§ Apartar la validación](#ejercicio-2-apartar-la-validación-train_val_split) |
| 3. `get_batch` | Sacar un lote de ventanas al azar y subirlo a la GPU | [§ Sacar un batch](#ejercicio-3-sacar-un-batch-get_batch) |

Las dos primeras son cortas y casi todo lo que tienen es validación de errores. La tercera
es el módulo: la va a ejecutar tu entrenamiento decenas de miles de veces y es donde está
la idea.

Pero no saltes directo a ellas. Las dos secciones que vienen ahora —qué parte de un LLM es
esta y cuál es la idea de fondo— son las que hacen que los tres ejercicios dejen de parecer
fontanería suelta.

### Cuánto cuesta

2 horas. Poco código, pero el ejercicio 3 tiene un *off-by-one* que, si lo fallas, no da un
error donde lo cometes sino tres líneas más abajo y con un mensaje que no dice nada.

---

## Qué parte del LLM es esta

Antes de entrar en el código conviene situarse, porque este módulo es de los que se
entienden mal si se leen aislados: no construye ninguna pieza del modelo, y aun así decide
qué acaba sabiendo el modelo.

Construir un LLM son cuatro trabajos distintos, y el curso los recorre en este orden:

```
   0. FUNDAMENTOS      qué es un LLM, PyTorch, autograd        módulos 00-02   ✔ hecho
   1. TOKENIZADOR      texto  ->  números                      módulo 03       ✔ hecho
   2. DATOS            números  ->  tarea de aprendizaje       módulo 04       ← ESTÁS AQUÍ
   3. MODELO           la arquitectura que hace la predicción  módulos 05-10
   4. ENTRENAMIENTO    ajustar los pesos hasta que acierte     módulos 11-13
   ────────────────────────────────────────────────────────────────────────────
      y después: generar texto (14), evaluar (15), afinar a instrucciones (16)
```

La pieza 2 es la de hoy, y su nombre completo en la literatura es **el pipeline de datos de
preentrenamiento**. Es la parte que los papers despachan en un párrafo y que en un
laboratorio de verdad ocupa a un equipo entero durante meses.

### Preentrenamiento: la fase para la que preparas los datos

Un LLM comercial se construye en dos fases muy distintas, y merece la pena tenerlas
separadas en la cabeza desde ya:

- **Preentrenamiento.** Le das cantidades brutales de texto y una única tarea: predecir el
  siguiente token. No hay conversación, no hay instrucciones, no hay nadie corrigiéndole. Es
  aquí donde el modelo aprende gramática, hechos, estilo y las regularidades del lenguaje, y
  es donde se va la inmensa mayoría del cómputo. **Los datos de este módulo son los de esta
  fase**, y es la única que hace este curso hasta el módulo 13.
- **Post-entrenamiento** (SFT, RLHF, módulo 16). Un pulido comparativamente diminuto sobre
  ejemplos de instrucción y respuesta, que convierte «un modelo que continúa texto» en «un
  modelo que obedece». Esos datos son otros, son caros y llevan humanos detrás.

Cuando alguien dice que GPT-4 «se entrenó con internet», habla de la primera fase. Y las
cifras de la primera fase son todas de este tipo:

| modelo | parámetros (el tamaño del modelo) | tokens de preentrenamiento (el texto que se le da) |
|---|---|---|
| GPT-3 (2020) | 175.000 M | 300.000 M |
| Chinchilla (2022) | 70.000 M | 1.400.000 M |
| Llama 3 405B (2024) | 405.000 M | 15.000.000 M |
| **el tuyo** | **8,93 M** | **500 M** |

Ojo con no mezclar las dos columnas, porque se confunden con facilidad y significan cosas
muy distintas. **Los parámetros son el modelo**: los números que la red ajusta y que acabas
guardando en un fichero de pesos — los 8.933.440 del tuyo, que montas en el módulo 10. **Los
tokens son el texto que le pasas por delante durante el entrenamiento**, y no se guardan en
el modelo: se leen, se aprende de ellos y se descartan. Un modelo de 8,93 M de parámetros
entrenado sobre 500 M de tokens ve **cincuenta y seis veces su propio tamaño en texto**, y
sigue ocupando lo que ocupa: unos 36 MB en fp32. (En el módulo 12 verás esa misma proporción
como 65 tokens por parámetro; allí se descuentan los embeddings del recuento, que es lo
habitual en las leyes de escala. Misma cuenta, distinto denominador.)

Con eso claro, mira la tabla otra vez: tu modelo es cuarenta y cinco mil veces más pequeño
que Llama 3 y ve treinta mil veces menos texto. Pero el tipo de objeto que estás preparando
y la tarea que estás definiendo son exactamente los mismos, y el `get_batch` que vas a
escribir se diferencia del suyo en detalles de ingeniería, no en la idea.

Que 500 M de tokens sea la cifra correcta *para un modelo de 8,93 M de parámetros* tampoco
es arbitrario: esa proporción entre las dos columnas es justo lo que estudian las leyes de
escala, y el módulo 12 las deduce y calcula la nuestra.

### La idea de fondo: el texto se supervisa a sí mismo

Aquí está el concepto que hay que llevarse de este módulo, y es un concepto de machine
learning, no de fontanería.

**El problema, en llano.** El aprendizaje automático clásico necesita datos *etiquetados*:
mil fotos con un humano que haya escrito «gato» o «perro» debajo de cada una. Esa etiqueta
es cara. Es el cuello de botella histórico del campo: no hay dinero para etiquetar a mano
los millones de ejemplos que hacen falta para algo grande.

Un texto no viene etiquetado. Nadie ha anotado *Don Quijote* diciendo qué es correcto y qué
no. Entonces, ¿de dónde sale la respuesta con la que comparar la predicción del modelo?

**El truco: la respuesta ya está escrita, un token más allá.** Si la tarea es «predecir qué
viene después», el texto original *es* la solución. La etiqueta no hay que fabricarla: hay
que taparla y destaparla.

```
    frase:      el   gato   duerme   en   el   sofá

    ejemplo 1   entrada: "el"                     respuesta: "gato"
    ejemplo 2   entrada: "el gato"                respuesta: "duerme"
    ejemplo 3   entrada: "el gato duerme"         respuesta: "en"
    ejemplo 4   entrada: "el gato duerme en"      respuesta: "el"
    ejemplo 5   entrada: "el gato duerme en el"   respuesta: "sofá"
```

Seis palabras, cinco ejemplos de entrenamiento con su respuesta correcta, y cero humanos
implicados. A esto se le llama **aprendizaje autosupervisado**: la supervisión existe, pero
sale del propio dato en lugar de un anotador. Es lo que permitió dejar de entrenar con
corpus anotados de miles de ejemplos y empezar a entrenar con la web entera, y es —más que
ninguna innovación de arquitectura— la razón de que los LLM despegaran.

**La fórmula.** Lo que ese esquema define formalmente es el objetivo de entrenamiento. Si el
corpus es una secuencia de tokens $x_1, x_2, \dots, x_n$, el modelo tiene que maximizar la
probabilidad que le asigna a cada token dado todo lo anterior:

$$\max_\theta \sum_{t=1}^{n} \log P_\theta(x_t \mid x_1, \dots, x_{t-1})$$

Léelo con el ejemplo delante: cada sumando es una fila de la tabla de arriba. $\theta$ son
los pesos del modelo, y $P_\theta(x_t \mid \cdots)$ es la probabilidad que el modelo le da a
la palabra que de verdad venía. Maximizar eso es exactamente minimizar la **cross-entropy**,
que es la pérdida que implementas en el módulo 05 y el número que mirarás durante horas en
el 13.

Y ahí está el papel de este módulo: **`get_batch` es lo que materializa esa suma.** El
sumatorio de la fórmula no lo escribe nadie a mano; aparece porque le pasas al modelo un `x`
y un `y` desplazado un token, y él calcula las 512 predicciones de golpe.

### Lo que este módulo deliberadamente no hace

Para que quede claro el perímetro, porque un pipeline de datos de verdad tiene más piezas de
las que hay aquí:

- **No hay modelo ni gradientes.** No se importa nada de `torch.nn`. Todo esto ocurre antes
  de que el modelo exista.
- **No se limpia el corpus.** Un pipeline real dedica la mayor parte del esfuerzo a filtrar
  basura, deduplicar (documentos repetidos hacen que el modelo memorice), quitar contenido
  tóxico y decidir en qué proporción se mezclan las fuentes. Nosotros nos saltamos todo eso
  porque TinyStories viene limpio de fábrica, y esa es la razón principal de que un modelo
  de nueve millones de parámetros pueda escribir algo coherente.
- **No se tokeniza aquí.** Eso fue el módulo 03. Este módulo empieza donde termina aquel:
  con la lista de ids ya hecha.

## Dónde encaja este módulo, en concreto

Con el mapa general claro, este es el recorrido exacto de los datos por el código que vas a
escribir:

```
   módulo 03      texto crudo          "Once upon a time…"
                       │
                       │  tu tokenizador
                       ▼
                  lista de ids         [271, 4, 88, 1902, …]
   ───────────────────┼──────────────────────────────────────────
                      │  pack_tokens_uint16   (ej. 1)
                      ▼
   módulo 04     fichero .bin          1 GB de uint16 en disco
                      │
                      │  train_val_split      (ej. 2)
                      ▼
              train  /  val            dos vistas del mismo array
                      │
                      │  get_batch            (ej. 3)   ← en cada paso
                      ▼
   ───────────────────┼──────────────────────────────────────────
                   (x, y)              dos tensores int64 de 48×512
                      │
   módulo 11          ▼                el bucle de entrenamiento
```

A la izquierda de este módulo está tu tokenizador. A la derecha está el bucle de
entrenamiento. Lo que hay en medio —lo que escribes hoy— se ejecuta en dos momentos muy
distintos, y esa diferencia explica por qué las tres funciones tienen prioridades opuestas:

- **Los ejercicios 1 y 2 se ejecutan una sola vez**, al preparar el corpus. Pueden tardar.
  Lo que no pueden es equivocarse en silencio, porque el error se queda grabado en el
  fichero y todo lo que venga después entrena con datos corruptos sin saberlo. Por eso son
  casi todo validaciones.
- **El ejercicio 3 se ejecuta en cada paso de entrenamiento**, decenas de miles de veces.
  Ahí no hay validaciones caras que valgan.

## Las tres cosas con las que se trabaja

Antes de teclear nada, ten claras las tres estructuras de datos que aparecen en todas las
firmas de `ejercicios.py`. Son solo tres y no hay ninguna clase de por medio.

**1. La lista de ids: `list[int]`.** Lo que te da el módulo 03. Números de Python normales
y corrientes, cada uno entre 0 y 4095:

```python
[271, 4, 88, 1902, 33, 4, 271]
```

**2. El array de tokens: `np.ndarray` de `uint16`.** Lo mismo, pero como array de NumPy y
ocupando 2 bytes por número en vez de los 8 que gasta Python. Esto es lo que se escribe a
disco y lo que leerá el entrenamiento. La conversión es el ejercicio 1.

```python
array([ 271,    4,   88, 1902,   33,    4,  271], dtype=uint16)
```

Cuando ese array vive en un fichero en lugar de en RAM se llama `np.memmap`, pero se usa
exactamente igual: `data[100:200]` funciona sin más. Hay una sección al final sobre por qué.

**3. El par `(x, y)`: dos tensores de PyTorch de forma `(batch_size, context_length)`.** Lo
que se le da al modelo. `x` es lo que ve y `y` es lo que tiene que predecir, y la única
diferencia entre los dos es que `y` está desplazado un token. Producirlos es el ejercicio 3.

```python
x.shape   # torch.Size([48, 512])   dtype=torch.int64
y.shape   # torch.Size([48, 512])   dtype=torch.int64
```

Fíjate en que el tipo cambia dos veces por el camino: `int` de Python → `uint16` para
guardarlo → `int64` para dárselo al modelo. No es capricho, y las dos conversiones tienen
su motivo. Las dos están explicadas más abajo, en el ejercicio que las hace.

---

## Ejercicio 1: empaquetar el corpus (`pack_tokens_uint16`)

Entra una lista de ids, sale un array de NumPy de tipo `uint16`. Cuatro líneas, y tres son
comprobaciones.

### Por qué `uint16` y no el tipo por defecto

Un token de nuestro modelo es un número entre 0 y 4095. La pregunta es cuántos bytes le
dedicas a guardar cada uno, y la respuesta la decide el corpus completo:

| tipo | rango que aguanta | 500M tokens ocupan |
|---|---|---|
| `int64` (lo que usa Python) | ±9 · 10¹⁸ | **4,0 GB** |
| `uint32` | 0 a 4.294.967.295 | 2,0 GB |
| `uint16` | **0 a 65.535** | **1,0 GB** |

`uint16` llega hasta 65.535, dieciséis veces más de lo que necesitamos, y ocupa la cuarta
parte que el `int64` que Python usaría por su cuenta. Tres gigabytes de diferencia por
escribir `dtype=np.uint16` en el sitio correcto.

(`u` es de *unsigned*, sin signo: no gasta un bit en representar negativos, que en ids de
token no existen. Y `16` son los bits: 2 bytes, 2¹⁶ = 65.536 valores distintos.)

Podrías apurar más: con un vocabulario de 4.096 bastarían 12 bits por token. No se hace,
porque los tipos de NumPy van de byte en byte y empaquetar a mano costaría más CPU en cada
lectura de lo que ahorra en disco. `uint16` es el punto razonable.

### La trampa: NumPy no avisa cuando un número no cabe

Esto es lo que justifica que el ejercicio exista. Si conviertes a `uint16` un número que no
cabe, NumPy **no lanza una excepción y no imprime un aviso**. Da la vuelta al contador y
sigue como si nada. Ejecutado, no estimado:

```python
np.array([65535], dtype=np.int64).astype(np.uint16)   # -> 65535   bien
np.array([65536], dtype=np.int64).astype(np.uint16)   # -> 0
np.array([65537], dtype=np.int64).astype(np.uint16)   # -> 1
np.array([66536], dtype=np.int64).astype(np.uint16)   # -> 1000
np.array([   -1], dtype=np.int64).astype(np.uint16)   # -> 65535
```

Mira el `66536 -> 1000`. Ese es el caso verdaderamente feo, y no el del 0. Un id que se
salió de rango se ha convertido en **1.000, que es un id perfectamente válido** de nuestro
vocabulario de 4.096: no hay nada en el fichero que delate que ese token está mal. Se
escribe, se entrena con él, y lo único que notas es que el modelo aprende algo peor de lo
que debería. No hay traza, no hay excepción, no hay nada que mirar. Es de los bugs que se
pasan días buscando en el sitio equivocado.

Y el `-1 -> 65535` es la misma historia por el otro extremo: un id negativo (que solo puede
venir de un fallo tuyo en el tokenizador) se convierte en el número más grande posible.

### Por eso el orden es: convertir a `int64`, validar, y solo entonces empaquetar

La secuencia del ejercicio no es arbitraria:

```python
array = np.asarray(ids, dtype=np.int64)      # 1. a un tipo donde TODO cabe
if array.size and (...):                     # 2. comprobar el rango
    raise ValueError(...)
return array.astype(np.uint16)               # 3. y ahora sí, empaquetar
```

Si validaras después de convertir, estarías comprobando datos **que ya se han corrompido**:
el 65.536 ya sería un 0, y un 0 pasa cualquier validación de rango con nota. La comprobación
tiene que hacerse mientras los números siguen siendo ellos mismos.

### Los dos detalles pequeños

**El `array.size and ...` del paso 2.** Sobre un array vacío, `.min()` no devuelve nada
sensato: lanza `ValueError: zero-size array to reduction operation minimum`. Es un error de
verdad, pero no tiene nada que ver con lo que estás validando y despista a quien lo lea. El
cortocircuito del `and` evita entrar ahí. Un corpus vacío es raro pero legítimo (un fichero
que resultó no tener nada), y no debería reventar.

**Los valores concretos en el mensaje de error.** «ids fuera de rango» no le sirve a nadie.
`minimo=0, maximo=9999` te dice al instante que tu tokenizador está emitiendo ids que no
debería y con qué magnitud, que es justo lo que necesitas para saber dónde mirar. Hay un
test que comprueba que el número aparece en el mensaje.

### Qué se hace con el array después

Fuera de este ejercicio, en el pipeline real, el array se vuelca a disco tal cual:

```python
tokens.tofile("train.bin")
```

Y eso escribe exactamente los bytes del array, sin cabecera, sin metadatos, sin nada. Un
fichero de 1 GB que es literalmente la tira de números. Por eso al leerlo hay que decirle a
NumPy de qué tipo era (`dtype=np.uint16`): el fichero no lo sabe, no hay dónde guardarlo. Si
te equivocas de `dtype` al leer, obtienes basura ordenada — otro fallo silencioso más de esta
familia.

---

## Ejercicio 2: apartar la validación (`train_val_split`)

Entra el corpus entero, salen dos trozos. Cuatro líneas.

### Para qué sirve el conjunto de validación

Si mides cómo de bien lo hace el modelo sobre el mismo texto con el que lo has entrenado,
la respuesta no significa nada: un modelo con memoria suficiente puede memorizar el texto y
sacar una nota perfecta sin haber aprendido nada que sirva para otra cosa. Eso es el
**sobreajuste** (*overfitting*).

La forma de detectarlo es apartar un trozo de texto, no entrenar jamás con él, y medir ahí
de vez en cuando. Lo que verás durante el entrenamiento es esto:

```
  pérdida
    │
    │ \
    │  \                       validación: baja, toca fondo y empieza a SUBIR
    │   \        ___..--''
    │    '-.__.-'
    │     \__
    │        '--..___          entrenamiento: baja y baja y baja
    └────────────┬──────────────  pasos
                 │
        aquí empezó a memorizar
```

Mientras las dos bajan juntas, el modelo está aprendiendo cosas generales. Cuando la de
entrenamiento sigue bajando y la de validación se estanca o sube, está memorizando. Sin
conjunto de validación **no tienes forma de ver ese momento**, y es información que
necesitarás en el módulo 11.

### Por qué el corte es contiguo y por el final

El reflejo de cualquiera que haya hecho machine learning tabular es barajar y repartir:
`train_test_split(shuffle=True)`. Aquí eso está **mal**, y no un poco: está completamente
roto. La razón es que las ventanas de entrenamiento se solapan.

Con `context_length=512`, la ventana que empieza en la posición 100 y la que empieza en la
101 comparten 511 de sus 512 tokens. Cada token del corpus aparece en 512 ventanas
distintas. Así que si repartes **ventanas** al azar, tus ventanas de validación son casi
idénticas a ventanas que están en entrenamiento.

Cuánto de casi. Esto es una simulación ejecutada, corpus de 100.000 tokens, contexto 512,
reparto aleatorio de ventanas 99,5% / 0,5%:

| forma de repartir | tokens de validación que el modelo ya vio entrenando |
|---|---|
| ventanas barajadas al azar | **100,00 %** |
| corte contiguo por el final | **0,0 %** |

No es «un poco de fuga»: es que el conjunto de validación es un subconjunto exacto de lo que
ya vio. El 100% de las ventanas de validación estaban íntegramente cubiertas por ventanas de
entrenamiento.

Y lo peor es cómo se manifiesta, porque no se manifiesta como un fallo. Se manifiesta como
un éxito:

- la pérdida de validación baja pegadita a la de entrenamiento,
- nunca sube, nunca se estanca,
- no ves sobreajuste jamás, por mucho que entrenes.

Un gráfico precioso que dice exactamente nada. Estarías midiendo memorización y llamándolo
generalización. Es el tipo de error que se descubre tarde, cuando el modelo sale al mundo y
resulta que no era tan bueno.

**La solución es cortar un bloque contiguo por el final** y no tocarlo. Como TinyStories son
historias independientes unas de otras, ese último 0,5% son historias enteras que el modelo
no ha visto nunca. Ni siquiera hay fuga en la frontera del corte, porque las dos mitades
pasan a ser arrays separados: ninguna ventana de entrenamiento puede cruzar al otro lado.

Detrás hay un principio general que vale para mucho más que este módulo: **el conjunto de
validación tiene que ser independiente del de entrenamiento en la unidad que importa**. Aquí
la unidad no es el token: es la historia. En datos temporales sería el día; en datos
médicos, el paciente. Barajar filas es correcto solo cuando las filas son de verdad
independientes, y aquí no lo son ni de lejos.

### Los dos detalles del código

**El `max(1, ...)`.** Con un corpus de 50 tokens y `val_fraction=0.005`, `int(50 * 0.005)`
es `int(0.25)`, o sea 0: te quedarías sin conjunto de validación y sin ningún aviso. El
`max(1, ...)` garantiza al menos un token. Es un caso que solo aparece en los tests y en
pruebas rápidas, pero aparece.

**Devuelve vistas, no copias.** Esto:

```python
return tokens[:-n_val], tokens[-n_val:]
```

no copia ni un byte. El *slicing* de NumPy devuelve una **vista**: un array nuevo que apunta
a la misma memoria que el original. Con 500M tokens, hacer un `.copy()` por costumbre serían
1 GB de RAM tirados a cambio de nada. Y si el original es un `memmap`, la vista sigue siendo
un `memmap` y sigue sin cargar nada. Hay un test que lo verifica con `np.shares_memory`.

Lo que esta función **no** hace, y conviene que quede claro: no baraja nada, no reordena, no
copia, no toca los datos. Corta y devuelve dos vistas. Eso es todo.

---

## Ejercicio 3: sacar un batch (`get_batch`)

Aquí está el módulo. Entra el corpus, salen dos tensores listos para el modelo.

### La idea, ahora en números

Esta es la traducción a código del aprendizaje autosupervisado que veíamos al principio: los
pares «entrada → respuesta» de la frase del gato, pero con ids y en dos matrices.

Tienes el corpus como una tira larguísima de números. Eliges una posición al azar y coges
una ventana. **La entrada es la ventana. El objetivo es la misma ventana desplazada un
token.**

```
corpus =  [ 5,  8,  2,  9,  1,  7, ... ]

     x =  [ 5,  8,  2,  9]                 la ventana tal cual
     y =      [ 8,  2,  9,  1]             la misma, corrida una posición
```

Y ahora léelo columna a columna, que es donde está lo interesante:

| lo que ve el modelo | lo que tiene que predecir |
|---|---|
| `[5]` | `8` |
| `[5, 8]` | `2` |
| `[5, 8, 2]` | `9` |
| `[5, 8, 2, 9]` | `1` |

**Una ventana de 4 tokens no es un ejemplo de entrenamiento: son cuatro.** Con contexto 512
son 512 predicciones por ventana, y un batch de 48×512 son **24.576 predicciones en una sola
pasada**. Todas se calculan a la vez, en el mismo *forward*, y todas contribuyen a la
pérdida.

Cada una de esas 24.576 predicciones es un sumando de la fórmula del objetivo que veíamos
arriba. El sumatorio no lo escribe nadie: sale de aquí, de haber pasado un `y` desplazado un
token.

Hay una condición para que esto funcione, y no está en este módulo: el modelo tiene que ser
incapaz de mirar hacia adelante. Si al predecir la posición 2 pudiera ver el token 3, lo
copiaría y no aprendería absolutamente nada — pérdida cero y modelo inútil. Lo que lo impide
es la **máscara causal** del módulo 06. Ten presente que existe, porque la construcción de
`y` que estás escribiendo hoy depende de ella.

### Qué hace la función, con números de verdad

Esto es una ejecución real: corpus `[0, 1, 2, …, 999]` (para que los valores digan de dónde
salen), `batch_size=4`, `context_length=8`, semilla 0.

```python
data = np.arange(1000, dtype=np.uint16)
x, y = get_batch(data, batch_size=4, context_length=8, rng=np.random.default_rng(0))
```

Primero calcula hasta dónde puede empezar una ventana:

```
max_start = len(data) - context_length - 1 = 1000 - 8 - 1 = 991
```

Luego saca 4 posiciones de inicio al azar. Con la semilla 0 salen exactamente estas:

```
starts = [842, 631, 506, 267]
```

Y de cada una toma dos ventanas, la de entrada y la desplazada:

```
x = [[842, 843, 844, 845, 846, 847, 848, 849],
     [631, 632, 633, 634, 635, 636, 637, 638],
     [506, 507, 508, 509, 510, 511, 512, 513],
     [267, 268, 269, 270, 271, 272, 273, 274]]

y = [[843, 844, 845, 846, 847, 848, 849, 850],
     [632, 633, 634, 635, 636, 637, 638, 639],
     [507, 508, 509, 510, 511, 512, 513, 514],
     [268, 269, 270, 271, 272, 273, 274, 275]]
```

Ahí está todo lo que hace la función. Cada fila es una ventana contigua del corpus, las
cuatro empiezan en sitios distintos y sin relación, y `y` es `x` corrido una posición. Si tu
implementación produce esto, has terminado.

Fíjate además en una propiedad que usan los tests y que te sirve a ti para depurar:
`x[:, 1:]` y `y[:, :-1]` contienen exactamente lo mismo. Si eso no se cumple, tu `y` no es el
desplazamiento de tu `x` y el modelo estaría aprendiendo a predecir cualquier otra cosa.

### El `-1` es el error del ejercicio

`max_start = len(data) - context_length - 1`. Ese `-1` es donde más gente se equivoca, así
que merece la pena verlo despacio.

`x` necesita tokens desde `i` hasta `i + context_length - 1`. Pero `y` necesita **uno más**:
llega hasta `i + context_length`. Si calcularas `max_start` pensando solo en `x`, la última
ventana posible dejaría a `y` pidiendo un token que no existe.

Y aquí viene lo desagradable: **NumPy no lanza ningún error al hacer slicing fuera de
rango**. Simplemente te devuelve menos elementos de los que pediste.

```python
data = np.arange(10)
data[8:14]        # -> array([8, 9])    dos elementos, ningun error
```

Así que tu ventana corta se cuela sin protestar y el fallo salta tres líneas más abajo, en
el `np.stack`, con un mensaje sobre formas incompatibles que no menciona ni el índice ni el
final del corpus ni nada que te lleve a la causa. Si te sale ese error, ya sabes dónde
mirar.

### El `.astype(np.int64)` hace dos cosas, y las dos importan

```python
x_np = np.stack([data[i : i + context_length] for i in starts]).astype(np.int64)
```

**La obvia:** `nn.Embedding` indexa su tabla con los tokens, y PyTorch exige que los índices
sean `int64`. Tus datos están en `uint16`. Sin la conversión, el modelo lanza un error de
tipo en cuanto lo intentas.

**La menos obvia:** `astype` **copia**. Y esa copia es justo lo que quieres cuando `data` es
un `memmap`, porque `torch.from_numpy` no copia nada: envuelve la memoria que le des. Si le
dieras la ventana del `memmap` directamente, tendrías un tensor apuntando a memoria mapeada
de un fichero, y cada vez que el modelo lo leyera sería potencialmente una lectura de disco
en mitad del forward. El `astype` materializa los datos en RAM normal y corta esa
dependencia.

### El `device` y el `pin_memory`, solo en CUDA

El último paso sube los tensores a donde esté el modelo. En CUDA hay un truco que en CPU y
MPS no aplica:

```python
if device.type == "cuda":
    x = x.pin_memory().to(device, non_blocking=True)
```

**Memoria fijada** (*pinned*, o *page-locked*) es memoria que el sistema operativo se
compromete a no mover de sitio ni mandar al fichero de intercambio. Eso permite a la GPU
leerla por DMA —acceso directo, sin que la CPU haga de intermediaria copiando bytes—, y es
lo que hace posible el segundo ingrediente: con `non_blocking=True` la llamada `.to(device)`
**vuelve inmediatamente**, sin esperar a que la copia termine. La transferencia del siguiente
batch se solapa con el cálculo del actual en vez de sumarse a él.

En MPS no tiene sentido porque la memoria es unificada: la CPU y la GPU miran la misma RAM y
no hay copia que solapar. En CPU, obviamente, tampoco. De ahí el `if`.

### Por qué al azar y con repeticiones

Cada llamada elige posiciones al azar de todo el corpus, sin llevar cuenta de las que ya
salieron. Eso significa que algunas ventanas saldrán varias veces y otras ninguna: **no es
una época** en el sentido clásico de «una pasada completa por los datos».

A cambio, la función no tiene estado. No hay índice que mantener, ni permutación que
guardar, ni que decidir qué pasa al llegar al final. Reanudar un entrenamiento desde un
checkpoint es trivial porque no hay nada que reanudar. Es lo que hace nanoGPT y funciona
bien. En la sección del debate está la otra cara.

Un apunte de escala, para que se vea por qué preocuparse por las repeticiones es
innecesario: un corpus de 500M tokens con contexto 512 tiene **499.999.487 posiciones de
inicio distintas**. Una tirada de entrenamiento de 50.000 pasos con batch 48 consume
2,4 millones de ventanas. Estás muestreando el 0,5% del espacio posible: los choques son
anecdóticos.

---

## `memmap`: qué es y por qué se usa aquí

Un `np.memmap` es un array de NumPy cuyos datos viven en un fichero en lugar de en RAM. Lo
importante es que **se usa exactamente igual que un array normal**: `data[100:200]` funciona
sin más, `len(data)` funciona, el slicing funciona. Por debajo, el sistema operativo carga
en memoria solo las páginas que tocas y las descarta cuando necesita sitio.

```python
mm = np.memmap("train.bin", dtype=np.uint16, mode="r")
mm[1000:1512]        # el SO lee solo esa pagina del disco
```

Ahora la parte honesta, porque esto se suele explicar mal. Nuestro fichero de 1 GB **cabría
perfectamente en tus 16 GB de RAM**. La razón de usar `memmap` no es que no quepa:

1. **Arranque instantáneo.** Cargar 1 GB de disco a RAM tarda lo que tarde, y lo pagas cada
   vez que lanzas el script. Con `memmap` es inmediato: no se lee nada hasta que se toca.
2. **La caché del sistema operativo hace el trabajo por ti.** Como accedes a posiciones
   aleatorias una y otra vez, el SO acaba reteniendo en RAM lo que más usas. Gratis, y mejor
   gestionado de lo que lo harías tú.
3. **Escala sin cambiar nada.** Si mañana entrenas con un corpus de 50 GB, el mismo código
   sigue funcionando igual. Sin `memmap` habría que reescribirlo.

Si tu corpus es pequeño, cargarlo entero con `np.fromfile` es igual de válido y más simple.
Aquí no hay magia. Y ojo con medirlo: cronometrar `memmap` contra `fromfile` sobre un
fichero que ya está en la caché de páginas del sistema no mide el disco, mide la caché. El
`demo.py` de este módulo hace esa medición y te avisa explícitamente de que no leas nada en
ella.

---

## ¿Es el disco tu cuello de botella? Mídelo

La versión que se repite en todas partes es que la GPU se queda parada esperando datos. Es
verdad a veces. Vamos a ver si es verdad **aquí**, porque medirlo cuesta treinta segundos y
repetirlo sin medir cuesta decisiones equivocadas.

Medido en el MacBook Pro M5 (MPS), corpus de 50M tokens en un fichero de 100 MB abierto con
`memmap`, con `get_batch` de la referencia:

| batch × contexto | tokens/batch | ms por batch | tokens/s |
|---|---|---|---|
| 8 × 64 | 512 | 0,56 | 0,9 M |
| 16 × 128 | 2.048 | 0,36 | 5,7 M |
| 32 × 256 | 8.192 | 0,40 | 20,4 M |
| **48 × 512** | **24.576** | **0,45** | **54,6 M** |

Y ahora el otro lado de la balanza: un paso completo de entrenamiento del GPT de 8,9M
parámetros que construyes en el módulo 10, con ese mismo batch de 48×512, en esa misma
máquina:

```
get_batch                                 0,47 ms      (misma tirada que la línea de abajo;
paso completo (forward + backward + opt)  1.342 ms      de ahí los 0,47 y no los 0,45)
                                          ─────────
el dato es el 0,04 % del paso
```

**El disco no es tu cuello de botella, ni de lejos.** Podrías hacer `get_batch` dos mil veces
más lento y seguiría sin notarse. Esto es lo que hay que quedarse: a esta escala, la
fontanería de datos ya es lo bastante rápida y optimizarla más es tiempo perdido.

¿Cuándo dejaría de ser cierto? Cuando el corpus no quepa en la caché del sistema y cada
batch tenga que ir al disco de verdad; cuando el modelo sea tan pequeño que el paso dure
microsegundos; o —el caso realista y por eso lo primero que hace este módulo— **cuando
tokenizas al vuelo en lugar de una vez**. Ese último número también está medido: nuestro
`bpe_encode` del módulo 03, en Python puro, procesa unos **14,5 kB de texto por segundo**.

```
1 MB de texto           ≈  70 segundos
1,7 GB (500M tokens)    ≈  33 horas
```

Treinta y tres horas cada vez que arrancas un entrenamiento. *Ese* sí es un cuello de
botella, y es exactamente el que el ejercicio 1 elimina: tokenizas una vez, guardas los ids,
y a partir de ahí lees enteros de un fichero.

---

## Dónde está el debate

**El muestreo aleatorio con reemplazo no es lo único razonable.** Como no lleva cuenta de
las ventanas usadas, no da ninguna garantía de cobertura: puede que haya trozos del corpus
que el modelo no vea nunca. Un recorrido ordenado y barajado por bloques sí la da, a cambio
de mantener estado y complicar la reanudación. Con 500M tokens y una sola pasada la
diferencia es despreciable; con muchas épocas sobre un corpus pequeño importaría bastante
más, y ahí la mayoría de la gente usa un `DataLoader` con permutación.

**Las ventanas de tamaño fijo cortan los documentos por la mitad.** Nuestra ventana empieza
en una posición aleatoria, así que casi siempre arranca a mitad de una historia y termina a
mitad de otra. El modelo se pasa el entrenamiento viendo fragmentos descabezados. Las
alternativas —rellenar con *padding* hasta el final del documento, o concatenar documentos
separándolos con un token `<|endoftext|>` y enseñar al modelo a reiniciarse ahí— tienen sus
propios costes: la primera desperdicia cómputo en tokens vacíos, la segunda deja que la
atención cruce de un documento al siguiente salvo que compliques la máscara. La mayoría de
los modelos grandes concatenan y aceptan el cruce. Aquí, con historias cortas, cortar es
suficiente.

**Y lo más discutido: qué debería haber dentro del corpus.** El paper de TinyStories sostiene
que un dataset pequeño y muy limpio, con vocabulario de niño de cuatro años, permite a
modelos diminutos generar texto coherente — algo que no se consigue entrenando el mismo
modelo con un fragmento de internet del mismo tamaño. Que la calidad y la *distribución* de
los datos importen tanto o más que la cantidad es hoy una de las líneas más activas del
campo, y también una de las peor documentadas: los laboratorios grandes publican
arquitecturas y se guardan los datos. Cuando leas que un modelo nuevo es mejor, ten presente
que buena parte de la diferencia puede estar aquí, en el módulo menos glamuroso de todos.

---

**Para ampliar:** Eldan & Li 2023, [TinyStories](https://arxiv.org/abs/2305.07759) ·
[nanoGPT](https://github.com/karpathy/nanoGPT) (su `get_batch` es prácticamente el de este
módulo) · [Data movement is all you need](https://arxiv.org/abs/2007.00072), sobre cuánto
del tiempo de entrenamiento se va en mover datos y no en calcular. Términos sueltos, en
[GLOSARIO.md](../../GLOSARIO.md).
