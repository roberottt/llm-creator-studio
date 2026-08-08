# 00 — Qué es un LLM, en realidad

## Por qué importa este módulo

**Empieza aquí aunque tengas prisa.** Es el único módulo sin PyTorch, sin matrices y sin
derivadas, y es el que hace que todo lo demás tenga sentido.

La razón es esta: los diecisiete módulos siguientes construyen piezas cada vez más
sofisticadas para hacer **una sola cosa**. Si no tienes clarísimo cuál es esa cosa, lo que
viene después es ingeniería sin propósito: implementarás una atención multi-cabeza que pasa
los tests sin saber para qué sirve.

En una hora vas a escribir un generador de texto que funciona de verdad, con diccionarios y
una división. No es un juguete pedagógico que luego se tira: el bucle que lo mueve es
*literalmente* el mismo que ejecuta ChatGPT, y lo vas a reescribir casi idéntico en el
módulo 14 sobre tu propio GPT de nueve millones de parámetros. Lo único que cambia entre
uno y otro es de dónde salen los números.

Y vas a ver, con datos medidos y no con hand-waving, **por qué ese modelo tan simple se
estrella**. Ese choque es el motivo por el que existen las redes neuronales. Quien no lo ha
visto en carne propia se pasa el resto del curso creyendo que los transformers son
complicados porque sí.

### Qué sabrás al terminar

- Qué es exactamente un modelo de lenguaje, con una definición que cabe en una línea
  (spoiler: mucho menos místico de lo que parece).
- Por qué se dice que "solo predice el siguiente token", y qué significa eso de verdad.
- Cómo se elige ese token, y por qué coger siempre el más probable es mala idea.
- **Cómo se mide si un modelo de lenguaje es bueno**: la pérdida, que va a ser el número que
  mires durante horas en el módulo 13.
- **Por qué hacen falta redes neuronales**, viendo con números reales por qué la alternativa
  obvia —contar— se estrella contra un muro que no se puede rodear.

### Cuánto cuesta

Una hora: unos veinte minutos de lectura y cuarenta de código. Es el módulo más corto del
curso y el que más rentabilidad da por minuto invertido.

---

## 1. La idea, en una frase

**Un modelo de lenguaje es una función que, dado un texto, te dice qué probabilidad tiene
cada posible continuación.**

Nada más. No "entiende", no "razona", no "sabe". Recibe un trozo de texto y devuelve una
lista de números: uno por cada palabra o carácter que podría venir a continuación.

Si le das *"El cielo es de color "*, un modelo decente devolverá algo así:

```
azul      0.72
gris      0.11
negro     0.04
rosa      0.02
patata    0.0000003
...
```

Y ya está. Eso es el modelo entero. Fíjate en lo que **no** hay ahí: no hay una decisión, no
hay una frase, no hay una respuesta. Hay una distribución sobre el vocabulario. Lo que ves
cuando hablas con ChatGPT es este paso repetido miles de veces: se elige una palabra según
esas probabilidades, se pega al final del texto, y se vuelve a preguntar. Una y otra vez.

A ese bucle se le llama **generación autorregresiva** («auto» = a sí mismo, «regresivo» =
se realimenta). Merece la pena parar en la consecuencia, porque es contraintuitiva: **el
modelo no planifica la frase entera**. Escribe un token, se lo lee a sí mismo como si se lo
hubiera dado otro, y decide el siguiente. Cuando un LLM te da una respuesta bien
estructurada de tres párrafos, no la había pensado antes de empezar. Fue eligiendo token a
token, y cada elección condicionó la siguiente.

Esto también explica de dónde salen dos comportamientos que la gente encuentra raros:

- **Un LLM no tiene memoria entre conversaciones.** La función solo ve el texto que le
  pasas. Si la aplicación no le vuelve a mandar la conversación entera en cada turno, para
  el modelo no ha pasado nada. En el módulo 14 verás el bucle real, que le vuelve a pasar
  todo el texto acumulado en cada paso (y el truco que evita recalcularlo entero).
- **Las alucinaciones no son un fallo del sistema, son el sistema.** Si el modelo asigna
  0.03 a una continuación falsa y se muestrea, sale la continuación falsa. No hay una base
  de datos que consultar ni un paso de verificación. Solo hay una distribución.

## 2. De dónde salen esas probabilidades: contar

El modelo tiene que sacar esos números de algún sitio. La forma más tonta que existe, y que
funciona lo suficiente como para que Shannon la publicara en 1948, es **contar**.

Coge un texto y apunta, para cada carácter, cuáles le siguieron y cuántas veces. Con el
texto `"banana"` recorres las parejas `ba`, `an`, `na`, `an`, `na`:

```
tras 'b'  ->  'a' 1 vez
tras 'a'  ->  'n' 2 veces
tras 'n'  ->  'a' 2 veces
```

Ahora conviértelo en probabilidades dividiendo cada conteo entre el total de esa fila:

```
tras 'a'  ->  total = 2  ->  'n' con probabilidad 2/2 = 1.0
```

Con `"banana"` la tabla es aburrida porque todo sale 1.0. Con un texto de verdad no. Estos
números son reales, salen de contar sobre las 1.115.394 letras de Tiny Shakespeare, el
corpus que vas a usar en el `demo`:

```
tras 'a'  ->  'n' 10197 veces,  't' 8339,  'r' 7081,  'l' 4149,  's' 3893,  ' ' 2685, ...
             total = 55507
          ->  'n' 0.1837,  't' 0.1502,  'r' 0.1276,  'l' 0.0747,  's' 0.0701,  ' ' 0.0484
```

Y hay una fila del corpus que es una pequeña joya:

```
tras 'q'  ->  'u' 609 veces, y nada más.  ->  'u' con probabilidad 1.0
```

Seiscientas nueve `q` en el corpus y las 609 seguidas de `u`. El modelo ha aprendido una
regla ortográfica del inglés sin que nadie se la enseñe, solo contando. Ese es el mecanismo
entero del curso en miniatura: **de los datos sale la estructura**.

### La fórmula

Lo que acabas de hacer a mano se escribe así:

$$P(x_t = c \mid x_{t-1} = a) = \frac{\text{count}(a, c)}{\sum_{c'} \text{count}(a, c')}$$

Léelo despacio: la probabilidad de que el siguiente carácter sea `c`, sabiendo que el
anterior fue `a`, es las veces que viste `ac` juntos dividido entre las veces que viste `a`
seguida de cualquier cosa. Con `a = 'a'` y `c = 'n'`: 10197 / 55507 = 0.1837. Es
exactamente la cuenta de arriba, solo que con símbolos.

Esta receta tiene nombre, y te lo doy porque lo vas a reencontrar: es el **estimador de
máxima verosimilitud**. "Verosimilitud" es la probabilidad que tu modelo le asigna a los
datos que has observado; el estimador de máxima verosimilitud es el conjunto de parámetros
que la hace lo más alta posible. Y resulta que, para un modelo de conteo, esa elección
óptima son justo las frecuencias observadas. Cuando en el módulo 05 minimices la
cross-entropy con gradientes estarás haciendo lo mismo por otro camino: buscar los
parámetros que le dan la máxima probabilidad al texto real.

El resultado de esa división es una **distribución de probabilidad**: una lista de números
no negativos que suman 1. Es el objeto central de todo el curso. Cambiará radicalmente
*cómo* la producimos —de una división a nueve millones de parámetros— pero no *qué* es. Si
en el módulo 10 te pierdes, vuelve a esta frase: al final de todo el transformer hay una
lista de 4096 números que suman 1.

## 3. Elegir uno

Ya tienes `{'n': 0.40, 'r': 0.25, ' ': 0.20, 's': 0.15}`. ¿Cuál eliges?

La opción evidente es coger el más probable, `'n'`. Se llama **greedy** (o *argmax*), y
tiene dos problemas. El primero es que el modelo se vuelve determinista: con la misma
entrada escribe siempre exactamente lo mismo, palabra por palabra. El segundo es peor, y lo
vas a medir en el módulo 14: greedy se mete en bucles. Produce cosas como *"the cat sat on
the mat. the cat sat on the mat."* La lógica del bucle es fácil de ver: si tras un contexto
el token más probable te devuelve a un contexto que ya visitaste, no hay nada que rompa el
ciclo, porque no hay azar en ninguna parte.

Por eso se **muestrea**: se tira un dado trucado en el que `'n'` sale el 40% de las veces,
`'r'` el 25%, y así.

El método es el de la ruleta, y es lo que vas a programar en el ejercicio 2. Reparte la
recta del 0 al 1 en trozos proporcionales a cada probabilidad:

```
|----'n'----|--'r'--|--' '--|-'s'-|
0          0.40    0.65    0.85   1.0
```

Saca un número aleatorio en `[0, 1)` y mira en qué trozo cae, acumulando:

```
r = 0.61

'n'  acumulado = 0.40    ¿0.61 < 0.40?  no, sigo
'r'  acumulado = 0.65    ¿0.61 < 0.65?  sí  ->  sale 'r'
```

Que esto es correcto se ve así: el trozo de `'r'` mide 0.65 − 0.40 = 0.25 de largo, y un
número uniforme entre 0 y 1 cae dentro de él exactamente el 25% de las veces. Cada token
sale con su probabilidad, que es justo lo que queríamos. En el módulo 14 verás cómo se
deforma esta ruleta a propósito —temperatura, top-k, top-p— para que el texto salga más
creativo o más conservador, pero el mecanismo de base es este.

## 4. Cómo se sabe si un modelo es bueno

Aquí aparece el número que vas a mirar durante horas en el módulo 13, así que vale la pena
entenderlo ahora que el modelo cabe en una servilleta.

La intuición primero: un modelo es bueno si **no le sorprende** el texto real. Coges texto
que el modelo no ha visto, y en cada posición le preguntas qué probabilidad le daba al
carácter que de verdad venía. Si le daba mucha, bien. Si le daba poquísima, mal.

Para convertir "poquísima" en un número que se pueda promediar se usa `-ln(p)`. Con
números concretos:

```
p = 1.00   ->  -ln(1.00) = 0.00     acertó del todo, no hay sorpresa
p = 0.50   ->  -ln(0.50) = 0.69
p = 0.10   ->  -ln(0.10) = 2.30
p = 0.01   ->  -ln(0.01) = 4.61     le daba 1 entre 100 y salió: sorpresa grande
p -> 0     ->  -ln(p) -> infinito   creía que era imposible y ocurrió
```

La **pérdida** (*loss*) es el promedio de ese número sobre todas las posiciones:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \ln P(x_t \mid \text{contexto})$$

Entrenar es, literalmente, hacer bajar este número. Nada más.

Dos anclas para saber si una pérdida es buena o mala. La primera: un modelo que no sepa
absolutamente nada y reparta la probabilidad por igual entre los 65 caracteres distintos de
Tiny Shakespeare da `-ln(1/65) = ln(65) = 4.174`. Ese es el listón del cero absoluto; si tu
modelo saca más de 4.17, está peor que tirar los dados. La segunda: el modelo de conteo que
programas hoy, con un solo carácter de contexto, da **2.470** sobre texto que no ha visto.
Ha aprendido algo de verdad, y lo ha hecho contando.

Hay una segunda forma de leer el mismo número, la **perplejidad**, que es `e` elevado a la
pérdida. `e^2.470 = 11.8` se interpreta como "el modelo está tan indeciso como si eligiera
al azar entre 11.8 caracteres". Es el mismo dato con otra escala; la usa más la literatura
que el código.

## 5. Por qué esto no basta

El modelo del ejercicio 3 mira **un solo carácter hacia atrás**. Para decidir qué viene
después de *"el gat"* solo mira la `'t'` y tira los dados. Es desesperante, y se nota:

```
contexto de 1 carácter (65 contextos posibles, todos vistos)
  FRirmpavet wis, an wok, therongushy t t atheand nturorofouceir, m tatevete ar aterd
```

Reconoce que existen las vocales, los espacios y las comas. No sabe nada más.

La reacción obvia es mirar más atrás: en vez de contar pares, contar tríos, o ventanas de
cuatro o seis caracteres. A eso se le llama un modelo de **n-gramas**, y **funciona**. Estas
salidas son reales, generadas con el mismo código que vas a escribir hoy, cambiando solo el
tamaño del contexto:

```
contexto de 2
  Fin tis fall mounto degiver he of or he were menth I to herriand my lough mord whe hat

contexto de 3
  First perange is ther, rumous the had to did reseralic beford,
  Why, to my back, I hair lain!

contexto de 4
  First Camiliar,
  And hear'd his now him in his way, 'almost thy chainstruchio is in shown your women

contexto de 6
  First Gentleman:
  The senator:
  No more spices of my colour half way thee,
  I have shame:
  Upon him.
```

Con seis caracteres de contexto sale texto con nombres de personaje, saltos de línea en su
sitio y palabras inglesas de verdad. La pérdida acompaña: baja de 2.470 a 0.880. Parece que
la receta está clara: sube el contexto y sigue contando.

Pues no. Aquí es donde se acaba el camino, y por dos muros distintos.

### Muro 1: la tabla explota

Cada carácter más de contexto multiplica por 65 el número de filas *posibles* de la tabla.
Esta tabla está medida sobre el corpus real:

| contexto | contextos vistos | combinaciones posibles | % del espacio cubierto |
|---|---|---|---|
| 1 carácter  | 65      | 65        | 100 % |
| 2 caracteres | 1.403   | 4.225     | 33 % |
| 3 caracteres | 11.556  | 274.625   | 4,2 % |
| 4 caracteres | 50.712  | 17.850.625 | 0,28 % |
| 6 caracteres | 283.313 | 7,5 · 10¹⁰ | 0,00038 % |

Los contextos vistos crecen despacio —no pueden crecer más rápido que el corpus, que tiene
un millón de caracteres— mientras que los posibles crecen exponencialmente. Y esto es a
nivel de **carácter**, con un vocabulario ridículo de 65. Tu modelo final va a trabajar con
4096 tokens distintos y una ventana de 512. La tabla equivalente tendría $4096^{512}$
filas: un número con más de 1800 cifras. No hay disco en el planeta, ni lo habrá.

### Muro 2: casi todo tiene probabilidad cero

El primer muro es de espacio y suena a problema de ingeniería. El segundo es peor, porque
es de datos y no se arregla con hardware.

Si un contexto no apareció nunca en el texto de entrenamiento, la tabla **no tiene fila para
él**. No es que dé una probabilidad mala: es que no da ninguna. El modelo se queda
literalmente mudo, y por eso el ejercicio 3 necesita un `break` para ese caso.

Esto, medido sobre el 10% del corpus que reservamos como validación:

| contexto | pérdida en entrenamiento | pérdida en validación | predicciones imposibles |
|---|---|---|---|
| 1 carácter  | 2.452 | 2.470 | 0,17 % |
| 2 caracteres | 1.903 | 1.967 | 1,4 % |
| 3 caracteres | 1.491 | 1.571 | 4,2 % |
| 4 caracteres | 1.216 | 1.286 | 10,1 % |
| 6 caracteres | 0.761 | 0.880 | **34,8 %** |

Lee la última columna. Con seis caracteres de contexto, **en más de un tercio de las
posiciones del texto nuevo el modelo le da probabilidad cero a lo que de verdad pasó** (esas
posiciones ni siquiera entran en el cálculo de la pérdida; si entraran, sería infinita).
Aumentar el contexto mejora las columnas de pérdida y arruina la de al lado. El modelo no
está aprendiendo inglés: está memorizando Shakespeare, y cuanto más contexto le das, más
memoriza y menos generaliza. Eso tiene nombre y lo vas a volver a ver: **overfitting**.

Fíjate también en la brecha entre las dos columnas de pérdida. Con contexto 1 es de 0.018;
con contexto 6, de 0.119. Esa distancia creciente entre lo que el modelo hace sobre datos
vistos y sobre datos nuevos es la señal de alarma que vigilarás en el módulo 13.

Existen parches para las probabilidades cero —se llaman **suavizado** (*smoothing*): repartir
un poquito de probabilidad entre lo nunca visto, o mezclar el modelo de 6 caracteres con el
de 3 cuando el primero no sabe qué decir. Funcionan, se usaron durante décadas, y no
resuelven el problema de fondo, que es este: para una tabla, `"gato"` y `"perro"` son dos
claves distintas **sin ninguna relación entre sí**, tan ajenas la una a la otra como
`"gato"` y `"paraguas"`. Lo aprendido sobre una no ayuda absolutamente nada con la otra. Y
si tienes que ver cada combinación al menos una vez para saber algo de ella, no hay corpus
suficientemente grande.

Este es **el problema central del modelado del lenguaje**, y se llama la **maldición de la
dimensionalidad**.

## 6. Lo que hace una red neuronal

La solución no es contar mejor. Es **generalizar**: si el modelo ha visto *"el gato negro
duerme"*, tiene que poder decir algo sensato sobre *"el perro negro duerme"* aunque esa
frase no aparezca en ningún sitio.

La idea, que Bengio y sus coautores publicaron en 2003 y que es el antepasado directo de
todo lo que viene, es dejar de usar el token como una clave de diccionario y representarlo
como un **vector de números aprendido de los datos**. A ese vector se le llama *embedding*.

Con números pequeños, para que se vea. Imagina que a cada palabra le tocan solo dos números,
y que tras entrenar sobre mucho texto salen estos:

```
             animal   objeto
gato          0.91     0.05
perro         0.88     0.09
paraguas      0.02     0.95
```

Nadie escribió las etiquetas "animal" y "objeto"; se las he puesto yo al mirar el resultado.
El modelo solo ajustó números hasta predecir mejor, y `gato` y `perro` acabaron cerca porque
aparecen en contextos parecidos. La consecuencia es la que buscábamos: lo que el modelo
calcula para `gato` y lo que calcula para `perro` sale casi igual, **porque las entradas son
casi iguales**. Lo aprendido sobre uno se transfiere al otro gratis, sin haber visto nunca
la frase del perro. Una tabla de conteos no puede hacer esto ni en principio.

De ahí salen dos propiedades que resuelven los dos muros de golpe:

- **No hay tabla que explote.** En vez de $4096^{512}$ filas, el modelo guarda unos pocos
  millones de números y *calcula* la respuesta. El tuyo tendrá exactamente 8.933.440.
- **Nunca se queda mudo.** Le des el contexto que le des, incluido uno que no ha visto
  jamás, el cálculo produce una distribución. Puede ser mala, pero existe. No hay `break`
  que valga: no existe el caso "no tengo fila para esto".

Lo que falta, y es todo el resto del curso, es *cómo* se combinan esos vectores. Porque no
basta con tener un vector por palabra: hace falta que cada posición decida a cuáles de las
anteriores hacer caso —en *"el gato que vimos ayer en el parque duerme"*, quien manda sobre
`duerme` es `gato`, no `parque`—. Ese mecanismo es la **atención**, y es el módulo 06.

## 7. El mapa: qué sustituye a qué

Todo lo que viene son piezas que reemplazan una parte de lo que has hecho hoy. Vuelve a esta
tabla cuando un módulo te parezca gratuito:

| Lo que haces hoy | Lo que lo sustituye | Módulo |
|---|---|---|
| Un carácter = un token | Trozos de palabra, con BPE | 03 |
| El carácter como clave de un dict | Un vector aprendido (embedding) | 05 |
| Mirar 1 carácter atrás | Mirar 512 tokens, decidiendo a cuáles hacer caso | 06 |
| `conteo / total` | `softmax(logits)` | 05, 06 |
| Contar (una pasada por el texto) | Bajar la pérdida con gradientes | 02, 05, 11 |
| El `break` cuando no hay fila | Nada: siempre hay salida | — |
| Muestrear de la ruleta tal cual | Temperatura, top-k, top-p | 14 |
| El bucle de `generate_naive` | El mismo bucle, con un GPT dentro | 14 |

La última fila es la importante. El bucle no cambia. Nunca cambia.

## 8. Los tres números que vas a ver todo el rato

**Token**: la unidad de texto que maneja el modelo. Hoy será un carácter; a partir del
módulo 03 serán trozos de palabra. Nuestro modelo final conocerá 4096 tokens distintos.

**Parámetros**: los números que la red aprende. El nuestro tendrá 8.933.440 —el conteo es
exacto y hay un test que lo comprueba—. Los modelos comerciales grandes andan del orden de
cien mil veces por encima.

**Pérdida** (*loss*): `-ln(probabilidad que el modelo le dio al token correcto)`, promediada.
Hoy has visto 4.174 como listón del que no sabe nada y 2.470 como la del modelo por conteo
más simple. Cuando entrenes de verdad sobre TinyStories, la pérdida bajará de unos 8,3 al
empezar (`ln(4096)`, el modelo sin entrenar reparte por igual) hasta la zona en la que
empiezan a salir historias legibles. Ese descenso, en una gráfica, es el módulo 13 entero.

## Dónde está el debate

Que un LLM «solo predice el siguiente token» es cierto y a la vez engañoso, y conviene que
sepas que la discusión está viva antes de que alguien te la venda cerrada.

La afirmación mecánica no la discute nadie: el objetivo de entrenamiento es predecir el
siguiente token, y ya está. La pregunta abierta —de verdad abierta, no retórica— es **qué
estructura interna necesita construir un sistema para predecir bien**. Hay evidencia de que
modelos entrenados solo con predicción de texto acaban desarrollando representaciones
internas de cosas que nadie les enseñó explícitamente: se han encontrado direcciones
interpretables asociadas a propiedades del mundo, y en modelos entrenados sobre partidas de
juegos de tablero se han recuperado representaciones del estado del tablero a partir de las
activaciones, cuando el modelo solo había visto listas de movimientos.

Hay quien lee eso como comprensión emergente y quien lo lee como estadística muy
sofisticada, y la discusión se enreda porque las dos partes usan "entender" con
definiciones distintas y no siempre explícitas. Lo honesto es decir que no está resuelto, y
desconfiar de quien te lo afirme con seguridad en cualquiera de las dos direcciones.

Un segundo debate, más práctico y más cercano a lo que vas a tocar: **hasta dónde llega la
predicción del siguiente token como objetivo**. Hay quien sostiene que basta con escalarla y
quien sostiene que hay capacidades —planificación a largo plazo, corrección de errores
propios— que necesitan un objetivo de entrenamiento distinto. En el módulo 16 verás la
primera grieta de forma concreta: para que un modelo siga instrucciones no basta el
preentrenamiento, hace falta una fase adicional con otro tipo de datos.

---

**Para ampliar:**

- Shannon 1948,
  [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
  — el artículo que inventó todo esto. En la sección 3 ya aparecen los modelos por conteo
  que vas a programar hoy, generados a mano con un libro y un lápiz.
- Bengio et al. 2003,
  [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
  — el paper que plantea la maldición de la dimensionalidad exactamente como está contada
  aquí y propone los embeddings como salida. Todo lo que viene después es descendiente suyo.

Si un término no te suena, está en [GLOSARIO.md](../../GLOSARIO.md).
