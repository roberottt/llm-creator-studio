# 13 — La tirada real: lanzarla, mirarla y saber si va bien

## Por qué importa este módulo

**Porque aquí lo entrenas de verdad.**

No hay concepto nuevo. Lo que se aprende aquí sólo se aprende haciéndolo: qué mirar mientras un
entrenamiento corre, qué es normal, y qué significa que algo vaya mal. Es el módulo menos
teórico del curso y el que más cambia lo que sabes hacer.

Y hay una técnica concreta que merece por sí sola el módulo: **el overfit a un batch**. Treinta
segundos que cazan casi cualquier bug del modelo o del bucle. Es el consejo con mejor relación
coste/beneficio de todo el deep learning, y casi nadie lo aplica.

También es donde vas a ver, en el fichero de muestras, el modelo aprendiendo a escribir paso a
paso. Eso es más informativo que cualquier curva de pérdida.

### Qué sabrás al terminar

- Una comprobación de 30 segundos que caza bugs que tardarías cuatro horas en descubrir
- **Cómo se lee la línea del log**, campo a campo — cada uno viene de un módulo distinto y
  juntos son el curso entero
- Los tres números que hay que mirar en el paso 0 de cualquier entrenamiento
- Qué es normal durante una tirada y qué significa un pico que no se recupera
- Qué hay que guardar en un checkpoint para poder reanudar **sin que el modelo pegue un
  bandazo**

### Qué vas a escribir

Dos funciones, y son las más pequeñas del curso:

| Ejercicio | Qué hace |
|---|---|
| 1. `overfit_single_batch` | La comprobación de 30 segundos que caza casi todo |
| 2. `format_eta` | Cuánto falta, en algo legible |

El ejercicio 1 es el bucle de entrenamiento más simple que existe, cuatro pasos, y es el que
importa. El 2 es una cadena de `if` que formatea segundos — parece cosmético y explico más abajo
por qué no lo es.

**Y luego viene lo que de verdad hace este módulo, que no es un ejercicio: entrenar.** El código
del entrenamiento ya está escrito, montado sobre las piezas de los módulos 04 a 12. Tu trabajo
aquí es lanzarlo, saber leerlo y decidir si va bien.

### Cuánto cuesta

1 hora de ejercicios, más lo que tarde tu entrenamiento. `tiny_char` son unos 70 segundos y
corre en cualquier máquina, incluida CPU.

---

## Ejercicio 1: la comprobación de los 30 segundos (`overfit_single_batch`)

Coges cuatro secuencias, se las das al modelo **una y otra vez**, y compruebas que la pérdida
baja casi a cero.

La idea es que un modelo con millones de parámetros tiene capacidad de sobra para memorizar
cuatro secuencias. No hay nada que generalizar: sólo memorizar. Si no lo consigue, hay un bug. Y
lo sabes en 30 segundos en vez de en cuatro horas.

Lo que escribes es el bucle de entrenamiento más desnudo posible: sin scheduler, sin acumulación
de gradientes, sin AMP. **A propósito**: cuantas menos piezas haya, menos sitios donde se pueda
esconder un bug. Este bucle es el patrón de referencia contra el que comparar el bucle de verdad
cuando algo falle.

### Qué tiene que salir

Medido con el modelo juguete (`tiny_char`, vocabulario 65):

| paso | pérdida | |
|---|---|---|
| 0 | 4,2902 | ← debería rondar `ln(65) = 4,174` |
| 10 | 3,1004 | |
| 50 | 2,7296 | |
| 100 | 1,4448 | |
| 200 | 0,1558 | |
| 299 | 0,0627 | ← debería estar casi en cero |

Dos segundos de reloj. Si `historial[-1]` no está muy por debajo de `historial[0]`, **para y
busca el bug**; no lances el entrenamiento largo.

### Qué caza y qué no

**Caza:** gradientes que no llegan a alguna parte del modelo (un `detach()` de más), el
`zero_grad()` olvidado —los gradientes se **acumulan** por defecto en PyTorch—, un learning rate
absurdo por arriba o por abajo, una capa desconectada del grafo, el optimizador construido sobre
los parámetros equivocados.

**No caza:** nada relacionado con la generalización. Un modelo que memoriza un batch
perfectamente puede seguir siendo completamente inútil con datos nuevos. Esto comprueba que la
**maquinaria** funciona, no que el modelo sea bueno.

### Si baja demasiado deprisa, también sospecha

Si la pérdida se planta en cero en cinco pasos, no lo celebres: mira si hay una fuga de
información. Comprueba que `y` va desplazado **un** token respecto a `x`. Si pasaras
`model(x, x)` el modelo sólo tendría que copiar la entrada y la pérdida se desplomaría. El
síntoma es idéntico al de una máscara causal rota, y ya te lo has encontrado dos veces en el
curso —módulos 05 y 10—: cuando la pérdida es sospechosamente buena, mira la máscara y después
mira quién monta el batch.

---

## Ejercicio 2: cuánto falta (`format_eta`)

Una cadena de `if` que convierte segundos en `"1h 2m"`. Cuatro tramos y un caso raro.

**Y no es cosmético, por dos razones.**

La primera es que vas a mirar ese número muchas veces durante una tirada de horas. `"1h 2m"` se
lee al instante; `"3725s"` hay que dividirlo mentalmente cada vez.

La segunda es más de fondo: a partir de una hora **se dejan de mostrar los segundos**. Cuando
faltan dos horas, los segundos son ruido: cambian todo el rato, no aportan información y hacen
que el número baile en pantalla. La precisión útil de una estimación siempre es proporcional a
su magnitud, y ésa es una regla que sirve mucho más allá de este ejercicio.

**Y el `"?"` en vez de un 0.** Devolver `"?"` es lo honesto cuando todavía no hay datos
suficientes para estimar: en los primeros pasos la velocidad media no significa nada. Además
evita imprimir cosas como `"-1s"` o `"infd 0h"`, que además de feas te hacen dudar de si el
entrenamiento va bien. `math.isfinite(x)` es `False` para `inf`, `-inf` y `nan`, y los tres salen
de dividir por cero al calcular el ritmo en el primer paso.

Comprueba el 3725 a mano: `3725 // 3600 = 1` y `(3725 % 3600) // 60 = 125 // 60 = 2`, o sea
`"1h 2m"`. Los cinco segundos sobrantes se pierden, que es justo lo que se quiere.

---

## Y ahora lo importante: lanzar la tirada

Con los dos ejercicios en verde, esto es lo que se hace, y en este orden:

```bash
llmfs check 13                              # los dos ejercicios en verde
llmfs train --config tiny_char              # el juguete: ~70 s, cualquier máquina
llmfs train --config tiny_char --max-steps 100   # sonda de ritmo antes de lo grande
llmfs train --config tinystories_9m         # la de verdad
llmfs train --config tinystories_9m --resume     # si se corta
```

**El juguete primero, siempre.** `tiny_char` es un GPT de 861.440 parámetros a nivel carácter
sobre Shakespeare, y existe exactamente para esto: validar el pipeline entero —datos, modelo,
bucle, muestreo, checkpoints— en menos de un minuto y en cualquier máquina. Si algo está roto se
ve ahí, y no cuatro horas después de lanzar la tirada de verdad.

Y antes de la grande, esa **sonda de 100 pasos** con `--max-steps`. No es para entrenar nada: es
para ver el ritmo real y el ETA. Si el ETA dice 40 horas cuando esperabas 4, algo va mal y más
vale saberlo antes de irse a dormir.

---

## Cómo se lee la línea del log

Ésta es la habilidad que de verdad te llevas del módulo. Cuando lanzas el entrenamiento, cada N
pasos aparece una línea como ésta (real, de una tirada de `tiny_char`):

```
paso  100/600   perdida 2.2693   lr 3.00e-03   |g| 0.70   112.1k tok/s   MFU 4.8%   faltan 18s
```

Seis campos, y **cada uno viene de un módulo distinto del curso**:

| campo | qué es | de dónde sale | qué mirar |
|---|---|---|---|
| `perdida` | cross-entropy del último batch | módulo 05 | que baje; ruidosa es normal |
| `lr` | el learning rate de este paso | módulo 11, ej. 2 | que suba en el warmup y baje después |
| `\|g\|` | norma global del gradiente **antes** de recortar | módulo 11, ej. 3 | entre 0,1 y 10 |
| `tok/s` | rendimiento medido | módulo 12 | que sea estable |
| `MFU` | fracción del pico del hardware | módulo 12, ej. 2 | que no se hunda de repente |
| `faltan` | el ETA | módulo 13, ej. 2 | que cuadre con lo previsto |

Por eso el `|g|` se registra **antes** del recorte y no después: si registraras la norma posterior
verías `1.00` clavado para siempre y no te enterarías de nada. Registrándola antes, un ascenso
sostenido te avisa de que el entrenamiento se está desestabilizando **antes** de que reviente. Es
exactamente lo que se decía en el módulo 11 y aquí es donde se cobra.

Y la MFU está en esa línea por lo mismo: no por su valor absoluto —el 4,8% de esa tirada es bajo
porque el modelo es diminuto— sino porque **si cae de golpe a mitad de la tirada, algo ha
cambiado**: otro proceso compitiendo por la GPU, throttling térmico, un dataloader que se ha
quedado sin caché.

---

## Los tres números del paso 0

Antes de la primera línea de log aparece la pérdida inicial. Míralos antes de irte a hacer otra
cosa:

**La pérdida inicial** tiene que valer $\ln(V)$. En la tirada de arriba salió `4.2633` frente a
`ln(65) = 4.1744`, un desvío de +0,089 — normal, por lo que viste en el módulo 10: la
inicialización con `std=0.02` da logits casi idénticos, no idénticos. Más alta significa
inicialización demasiado agresiva; más baja, fuga de información.

**La norma del gradiente** debería estar en el orden de 0,1 a 10. En la tirada real arrancó en
1,10. Si sale $10^5$, algo está explotando; si sale $10^{-8}$, algo se está desvaneciendo.

**Los tokens por segundo.** Multiplica por la duración prevista y comprueba que el ETA cuadra con
lo que esperabas.

Los tres son gratis y los tres te ahorran horas.

---

## Qué es normal durante la tirada

**La curva baja deprisa al principio y luego se aplana.** Es lo esperado: aprender que existen
los espacios y las vocales es fácil; aprender gramática, no. En escala logarítmica la caída es
aproximadamente una recta, que es lo que dicen las leyes de escala del módulo 12.

**La pérdida de entrenamiento es ruidosa y la de validación es suave.** La primera se mide sobre
un solo batch; la segunda, sobre cien. El ruido no significa nada — en la tirada de la demo, la
pérdida de entrenamiento sube de 1,6050 a 1,6362 entre los pasos 500 y 600 mientras la de
validación baja de 1,7903 a 1,7131. No es que el modelo empeore: es que el batch del paso 600 era
más difícil que el del 500.

**La brecha entre ambas crece un poco.** Es sobreajuste incipiente y es normal. Con TinyStories y
una sola pasada por los datos debería quedarse pequeña; si se dispara, el modelo está memorizando.

**Picos ocasionales.** Un batch raro produce un pico y el modelo se recupera en unos pasos. Con
`grad_clip` deberían ser pequeños — mediste en el módulo 11 exactamente cuánto: sin recorte, un
batch envenenado subía la pérdida 3×; con recorte, ni se enteraba. **Si un pico no se recupera,
el entrenamiento se ha roto**: para, y reanuda desde el último checkpoint bueno.

---

## Las muestras de texto: la parte que importa

Cada N pasos el script genera texto y lo añade a `samples.md`. Ese fichero, leído de arriba abajo
cuando termine, **es el modelo aprendiendo a escribir**. Esto es lo que salió en una tirada de
600 pasos del juguete:

```
paso 0     kUU$sbpKKMMbbbPcxfffffTjjfNLL --TJ??333OOqIwTGG33m'T.B--tuq
           D'sSSOOMBiPtB'''''wEvgRRR.vUUUHgJ;OXD3xxExqVOX$J-DUUHIiit&!

paso 300   MAPCHASTING Yrace not be town, bunders.  CAMILLY: Mare striset
           mist and be doth bare Enay?  First Larry a thee slay, to I pine

paso 600   Which begane of schame a loved, this show as friar, But there
           appos bementes that that will down, And my tell are whity it here
```

Mira lo que ha aprendido en el paso 300 sin que nadie se lo dijera: que las palabras se separan
con espacios, que las frases llevan puntuación, que hay nombres en mayúsculas seguidos de dos
puntos porque Shakespeare se publica en formato de obra de teatro. Las palabras casi todas están
mal, pero **la forma** es correcta.

**Es más informativo que la curva de pérdida.** Un salto de 1,6 a 1,5 no te dice mucho; ver que
el modelo ha empezado a cerrar los paréntesis, sí.

---

## Checkpoints: qué se guarda y por qué

Esto ya está escrito en `llmfs/train/checkpoint.py`, pero merece la pena abrirlo y ver qué mete
dentro, porque la lista no es obvia. Un checkpoint reanudable necesita cuatro cosas:

- los **pesos** del modelo
- el estado del **optimizador**, o sea los momentos de Adam del módulo 11
- el estado del **GradScaler**
- el **número de paso** y los tokens vistos

Si reanudas sólo con los pesos, Adam arranca con sus momentos a cero y el modelo **pega un
bandazo** justo al reanudar. Se ve clarísimo como un pico en la curva, exactamente en el punto
donde reanudaste. Es el mismo problema que resuelve la corrección de sesgo al principio de un
entrenamiento, sólo que ahora ocurre a mitad y sin que nadie lo esperase.

**Y un detalle de implementación que importa:** se escribe primero en un fichero temporal y se
renombra al final. Si el proceso muere a mitad de la escritura, el checkpoint anterior sigue
intacto. **Un checkpoint a medias es peor que no tener checkpoint**, porque parece bueno.

El entrenamiento guarda dos: `last.pt` para reanudar y `best.pt` con la mejor pérdida de
validación vista. No son el mismo fichero y conviene saber cuál quieres.

---

## La tirada de TinyStories en tu hardware

```
   modelo    : 8.933.440 parámetros
   tokens    : 500.000.000
   FLOPs     : 6 × 8,93M × 500M ≈ 2,7·10¹⁶
```

(El 8,93M de ahí es `params_matmul`, el recuento del módulo 12: todo menos las escalas de
normalización. No es el no-embedding de 7,62M, que es el que va en Chinchilla y no en los FLOPs.
Es justo la distinción de la que avisa aquel módulo, y aquí es donde toca aplicarla.)

Con la RTX 2060 a 51,6 TFLOPS de pico y una MFU realista del 10-15%, salen **entre 2 y 5 horas**.

Y aquí toca ser explícito: **eso es una estimación de servilleta, no una medición.** Esta tirada
no se ha ejecutado al escribir el curso, porque la máquina de desarrollo no tiene CUDA. Todo lo
demás que hay en este fichero está medido; esto no. El número real lo dará tu propia sonda de 100
pasos en los primeros minutos, y si no cuadra con esta estimación, fíate de la tuya.

---

## Dónde está el debate

**Cuándo parar es una decisión con menos ciencia de la que parece.** Lo estándar es entrenar
hasta agotar el presupuesto de tokens, pero no está claro que sea óptimo: hay evidencia de que
seguir entrenando más allá del punto de Chinchilla sigue mejorando el modelo, con rendimientos
decrecientes que nadie ha caracterizado bien. Es la otra cara de lo que viste en el módulo 12.

**La reproducibilidad exacta es más difícil de lo que parece.** Aunque fijes todas las semillas,
cuDNN elige algoritmos no deterministas por rendimiento y las reducciones en GPU suman en orden
no determinista. Dos tiradas idénticas divergen.
`torch.use_deterministic_algorithms(True)` lo arregla a costa de velocidad; para experimentos de
investigación merece la pena, para entrenar casi nunca.

Y una **sobre este curso**: nuestra tirada de 500M tokens con un solo conjunto de
hiperparámetros **no es un experimento controlado**. Si al terminar el modelo genera historias
decentes, no sabrás cuánto se debe a la arquitectura, cuánto al learning rate y cuánto al
dataset. Sacar conclusiones de una sola tirada es el error metodológico más común del campo, y
este curso no es una excepción: es un ejercicio de aprendizaje, no un experimento.

---

**Para ampliar:** Eldan & Li 2023, [TinyStories](https://arxiv.org/abs/2305.07759) ·
Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
(de donde viene el consejo del overfit a un batch). Términos sueltos, en
[GLOSARIO.md](../../GLOSARIO.md).
