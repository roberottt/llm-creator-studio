# 15 — Evaluación: ¿es bueno mi modelo, y comparado con qué?

## Por qué importa este módulo

**Porque "¿es bueno mi modelo?" es más difícil de responder de lo que parece.**

Tienes un modelo entrenado y un número: la pérdida. ¿Y ahora qué? Ese número no te dice si el
modelo escribe historias que alguien querría leer, y comparar tu número con el de otro modelo
puede ser directamente engañoso.

Aquí aprendes a usar tres herramientas distintas y, sobre todo, **qué mide cada una y dónde
falla**. Incluida una métrica que sí es comparable entre modelos distintos y que casi nadie usa,
y la parte que ninguna métrica automática sustituye: leer lo que escribe.

Es también el módulo donde se pone en contexto lo que has construido, con expectativas concretas
sobre qué puede y qué no puede hacer un modelo de 9M.

### Qué sabrás al terminar

- Por qué comparar perplejidades entre modelos con tokenizadores distintos **no significa nada**,
  y se hace constantemente
- Una métrica que sí es comparable, y su interpretación exacta: tu modelo es un compresor
- Cómo evaluar cualitativamente con una batería fija de prompts, y por qué se hace a mano
- Qué esperar realmente de un modelo de 9M, para no llevarte una decepción injusta

### Qué vas a escribir

Tres funciones, y esta teoría las sigue en orden:

| Ejercicio | Qué hace |
|---|---|
| 1. `perplexity_from_loss` | La métrica de siempre, y sus límites |
| 2. `bits_per_byte` | La que **sí** se puede comparar entre modelos |
| 3. `run_prompt_battery` | La que ninguna métrica automática sustituye |

Las tres son cortas —dos líneas las dos primeras— y el tercero ni siquiera calcula nada:
organiza el trabajo para que la parte que de verdad importa, **leer lo que escribe el modelo**,
sea cómoda. Ese es el módulo entero: tres funciones diminutas cuyo valor está en saber cuándo
usar cada una.

### Cuánto cuesta

2 horas, y buena parte se va leyendo el informe que generas.

---

## Ejercicio 1: perplejidad (`perplexity_from_loss`)

Ya la conoces del módulo 05. Es $e^L$, con $L$ la pérdida media en nats, y se interpreta como
**entre cuántas opciones equiprobables está dudando el modelo**. La función son dos líneas: una
guarda para valores no finitos y `math.exp(loss)`.

Los tres casos que hay que reconocer:

```
   pérdida 8,317  ->  perplejidad 4096,0   sin entrenar: duda entre TODO el vocabulario
   pérdida 1,60   ->  perplejidad    4,95  duda entre unas 5 opciones
   pérdida 0,0    ->  perplejidad    1,0   perfecto, no duda
```

El primero merece la comprobación: `ln(4096) = 8,317`, así que `exp(8,317) = 4096`. Un modelo
recién inicializado reparte la probabilidad por igual entre los 4096 tokens y su perplejidad es
exactamente el tamaño del vocabulario. Es el mismo suelo del módulo 05, visto desde el otro lado.

Y así se lee sobre el modelo del curso, medido:

| conjunto | pérdida | perplejidad | |
|---|---|---|---|
| azar (el suelo) | 4,1744 | 65,0 | lo que saca un modelo sin entrenar |
| train | 1,2746 | 3,58 | |
| val | 1,5497 | 4,71 | |

De dudar entre 65 caracteres a dudar entre 4,7. Y la brecha train/val de +0,275 es pequeña, que
es lo que quieres ver: **si fuera grande, el modelo estaría memorizando** en vez de aprendiendo.
Ésa es la lectura útil de tener las dos cifras juntas.

### Y su problema, que es serio

**La perplejidad depende del tokenizador.** Si tu vocabulario parte las palabras en trozos más
pequeños, cada token individual es más fácil de predecir y tu perplejidad sale mejor sin que el
modelo sea mejor.

Un ejemplo extremo para verlo: un modelo que predijera bit a bit tendría perplejidad cercana a 2
y sería inútil. Uno a nivel de palabra, sobre el mismo texto, tendría perplejidad de cientos.

**Comparar perplejidades entre modelos con tokenizadores distintos no significa nada**, y se hace
constantemente en papers y en posts de blog. De ahí el ejercicio siguiente.

---

## Ejercicio 2: bits por byte (`bits_per_byte`)

La solución: normalizar por **bytes de texto original** en vez de por tokens. Los bytes no
dependen de cómo trocees.

$$\text{bits/byte} = \frac{L_{\text{total}} / \ln 2}{n_{\text{bytes}}}$$

También dos líneas. El $\ln 2$ es lo único con truco: convierte nats a bits, porque toda la
pérdida del curso está en nats —logaritmo natural, módulo 05— y esta métrica se expresa en bits
por convención.

Fíjate en que la entrada es la pérdida **total**, no la media: si le pasaras la media por token
estarías mezclando una normalización por tokens con otra por bytes, que es justo lo que esta
métrica viene a evitar.

### Lo bonito: tu modelo es un compresor

Esta métrica tiene una interpretación exacta: **es cuántos bits necesitarías para transmitir el
texto usando el modelo como compresor**. Y no es una analogía, es una identidad que viene de
Shannon (1948).

Medido sobre el modelo del curso:

| compresor | bits/byte |
|---|---|
| sin comprimir | 8,00 |
| gzip (texto en inglés) | ~2,50 |
| **tu modelo** | **2,236** |
| los mejores LLM | 0,60 – 0,80 |

O sea que **tu modelo de juguete comprime mejor que gzip**. A 2,236 bits/byte reduce el texto a
1/3,6 de su tamaño. No es un truco de presentación: si conectaras un codificador aritmético a
sus probabilidades, comprimiría de verdad a ese ratio.

Y a diferencia de la perplejidad, esta cifra **sí se puede comparar** con la de cualquier otro
modelo, tokenice como tokenice.

---

## Ejercicio 3: la batería cualitativa (`run_prompt_battery`)

Ninguna de las dos métricas anteriores te dice si el modelo escribe algo que un humano querría
leer. Para eso hay que leerlo.

El ejercicio no calcula nada: coge una lista de prompts fijos, genera una continuación para cada
uno y las devuelve organizadas para que las leas. **Los prompts son fijos a propósito**: si
cambias los prompts entre dos evaluaciones no estás comparando modelos, estás comparando prompts.
Es la misma razón por la que el conjunto de validación no se toca.

El paper de TinyStories propone mirar tres cosas por separado:

**Gramática.** ¿Las frases están bien construidas? ¿Concuerdan sujeto y verbo?

**Coherencia.** ¿La historia se contradice? Si en la primera frase el gato es negro, ¿sigue
siendo negro tres frases después?

**Creatividad.** ¿Aporta algo o repite plantillas?

Lo interesante del paper es que estas tres capacidades **aparecen a escalas distintas**. Un
modelo de 1M de parámetros ya hace gramática decente; la coherencia necesita más; la creatividad,
todavía más. No es una escalera única: son habilidades que emergen por separado, y por eso se
puntúan por separado.

Los seis prompts de la batería del curso no son aleatorios: cada uno prueba una cosa distinta
—continuación básica, coherencia causal, seguimiento de un objeto, resolución, uso de un objeto,
cierre de historia—. Al leer las continuaciones, léelas contra lo que cada prompt pretendía
probar.

**Un aviso sobre la demo:** corre con el modelo de Shakespeare a nivel carácter, no con uno
entrenado sobre TinyStories. Los prompts en inglés moderno le quedan completamente fuera de
distribución y las continuaciones salen raras. El ejercicio de leerlas es el mismo, y de hecho se
aprende bastante viendo cómo un modelo intenta continuar algo que no ha visto nunca.

---

## Por qué no hay una métrica automática buena

Se han intentado muchas y todas fallan por el mismo sitio.

**BLEU, ROUGE** y compañía comparan contra una respuesta de referencia. Para generación libre no
hay una respuesta correcta: hay infinitas, y todas distintas de la de referencia.

**Usar otro LLM como juez** (*LLM-as-a-judge*) es lo que se hace ahora, y funciona razonablemente
para modelos grandes. Tiene sesgos conocidos y bien documentados: prefiere respuestas largas,
prefiere el estilo del propio modelo juez, y es sensible al orden en que se le presentan las
opciones.

**La evaluación humana** es el patrón oro y es cara, lenta y ruidosa: dos anotadores discrepan
más de lo que uno esperaría.

Para tu modelo de 9M, leer seis continuaciones es perfectamente razonable y probablemente más
informativo que cualquier número.

## Qué esperar de un modelo de 9M sobre TinyStories

Conviene ser concreto con las expectativas, porque el paper original entrenaba modelos parecidos
y sabemos qué sale:

- **Gramática correcta la mayor parte del tiempo.** Frases bien formadas.
- **Coherencia local, no global.** Dos o tres frases seguidas tienen sentido; una historia de
  diez, probablemente no.
- **Vocabulario limitado**, que es lo esperado: TinyStories está escrito a propósito con
  vocabulario de niño de 4 años.
- **Nada de razonamiento.** Ni aritmética, ni conocimiento del mundo, ni seguir instrucciones.

Si tu modelo hace eso, ha funcionado. Si esperabas algo parecido a un asistente, la diferencia no
es de entrenamiento: son tres o cuatro órdenes de magnitud en parámetros y datos, más el
post-entrenamiento del módulo 16, que es literalmente lo que convierte "un modelo que continúa
texto" en "un modelo que obedece".

## Dónde está el debate

**La relación entre perplejidad y capacidades es más floja de lo que se asume.** Se sabe que bajar
la pérdida mejora el modelo, pero no cuánto ni en qué. Dos modelos con la misma perplejidad pueden
comportarse muy distinto en tareas concretas, y la perplejidad puede bajar por memorización sin
que mejore nada útil.

**La contaminación de datos ha arruinado buena parte de la evaluación por benchmarks.** Los
conjuntos de test están en internet, y los modelos se entrenan con internet. Cuando un modelo saca
buena nota en un benchmark, distinguir "ha aprendido" de "lo ha visto" es técnicamente difícil y
comercialmente incómodo. Es uno de los problemas metodológicos más serios del campo ahora mismo.

**Y las capacidades emergentes están en discusión activa.** Se documentaron saltos bruscos de
capacidad al aumentar la escala, y en 2023 se publicó un análisis convincente argumentando que
muchos de esos saltos son **artefactos de métricas discontinuas**: si mides con una métrica de
todo-o-nada, ves saltos donde con una continua verías una curva suave. La discusión sigue.

---

**Para ampliar:** Eldan & Li 2023, [TinyStories](https://arxiv.org/abs/2305.07759) (la batería
cualitativa) · Shannon 1948, *A Mathematical Theory of Communication* (predicción y compresión) ·
Schaeffer et al. 2023, [Are Emergent Abilities a Mirage?](https://arxiv.org/abs/2304.15004).
Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
