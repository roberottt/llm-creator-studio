# 12 — Eficiencia y leyes de escala: ¿voy rápido, y estoy gastando bien?

## Por qué importa este módulo

**Porque "más grande es mejor" resultó ser falso, y eso cambió el campo.**

Dos preguntas que parecen distintas y son la misma: ¿estoy aprovechando mi GPU? y ¿cómo
debería repartir mi presupuesto entre tamaño de modelo y cantidad de datos?

La segunda tiene una respuesta concreta, y en 2022 resultó que la industria entera la estaba
haciendo mal. GPT-3 tenía 175.000 millones de parámetros y estaba **doce veces
infra-entrenado**: con su mismo presupuesto de cómputo, un modelo tres veces más pequeño
entrenado con más datos habría sido mejor. Lo demostraron entrenando uno.

En este módulo vas a reproducir esa fórmula y comprobar que predice el tamaño real de
Chinchilla con tres cifras de precisión. Y vas a medir la eficiencia de tu propio entrenamiento
con la métrica estándar del campo, que además es la que te dirá dónde tocar cuando el módulo 13
vaya lento.

### Qué sabrás al terminar

- Dónde se van los FLOPs de tu modelo, y por qué alargar el contexto sale caro
- **Cuál de los tres recuentos de parámetros del curso va en cada fórmula**, que es donde se
  pierde todo el mundo
- Qué es la MFU, qué valor es razonable, y por qué el tuyo va a ser bajo sin que sea culpa tuya
- Dónde mirar cuando un entrenamiento va más lento de lo que debería
- La fórmula de Chinchilla, **verificada contra modelos históricos reales**
- Por qué nuestro modelo está sobreentrenado a propósito, y por qué Llama-3 lo está 90 veces

### Qué vas a escribir

Tres funciones, y esta teoría las sigue en orden:

| Ejercicio | Qué hace |
|---|---|
| 1. `model_flops_per_token` | Cuánto cuesta un token, desglosado |
| 2. `compute_mfu` | Qué fracción de tu GPU aprovechas |
| 3. `chinchilla_optimal_allocation` | Cómo repartir el presupuesto de cómputo |

**Ninguna pasa de cinco líneas de código** y no hay tensores, ni modelos, ni entrenamiento:
sólo aritmética con los campos del config. El módulo es corto de teclear y largo de entender,
que es justo lo contrario del 11. Toda la dificultad está en saber qué significan los números
que salen y qué decisiones se toman con ellos.

Los ejercicios 1 y 2 encadenan: el `total` que devuelve el primero es el `flops_per_token` que
come el segundo. El tercero es independiente.

### Cuánto cuesta

2 horas.

---

## Ejercicio 1: dónde se van los FLOPs (`model_flops_per_token`)

En el módulo 01 estimaste el coste de un token con la regla rápida. Aquí lo desglosas de
verdad, y el desglose tiene una consecuencia de diseño.

La función separa el coste en dos términos porque **crecen con cosas distintas**:

```
   matmul     crece con el TAMAÑO del modelo   (d_model, n_layers, d_ff, vocab)
   attention  crece con el CONTEXTO            (context_length)
```

Y ésa es toda la gracia. Con nuestra config el reparto es 82/18, pero mira qué pasa al alargar
el contexto dejando el modelo igual:

| contexto | matmul | atención | total | % atención |
|---|---|---|---|---|
| 128 | 53,6M | 2,9M | 56,5M | 5% |
| **512 (el nuestro)** | **53,6M** | **11,8M** | **65,4M** | **18%** |
| 1024 | 53,6M | 23,6M | 77,2M | 31% |
| 2048 | 53,6M | 47,2M | 100,8M | 47% |
| 4096 | 53,6M | 94,4M | 147,9M | 64% |
| 8192 | 53,6M | 188,7M | 242,3M | 78% |

La columna de matmul **no se mueve**: no depende del contexto en absoluto. La de atención crece
linealmente y a partir de 2048 ya domina. Esto te dice al instante si alargar el contexto te va
a salir caro, sin tener que probarlo — y enlaza con el coste cuadrático en memoria que viste en
el módulo 06.

### De dónde sale cada constante

Son tres números y ninguno es arbitrario.

**El 2 de `matmul = 2 * params_matmul`.** Multiplicar una matriz por un vector hace, por cada
peso, una multiplicación y una suma: 2 operaciones por parámetro.

**El 4 de `attention = 4 * n_layers * T * d_model`.** Son los dos matmuls de la atención que
**no involucran parámetros**: $QK^\top$ y $\text{pesos} \times V$. Cada uno cuesta $2 T d$ por
token, y son dos. Por eso no aparecen en el término de matmul: no hay pesos que contar, pero el
cálculo se hace igual.

**El 3 del backward.** El paso hacia atrás cuesta aproximadamente el **doble** que el hacia
delante, porque hay que calcular el gradiente respecto a la entrada *y* respecto a los pesos:
dos matmuls donde el forward hacía uno. Forward + backward = 1 + 2 = 3. Puedes comprobarlo
llamando a la función con `include_backward=False` y dividiendo: sale 3,0 exacto.

### Los tres recuentos de parámetros, que es donde se pierde todo el mundo

Aquí va lo que más confunde de este módulo, y es culpa de que el curso arrastra **tres números
distintos** que se parecen mucho:

| recuento | valor | qué incluye | dónde se usa |
|---|---|---|---|
| **total** | 8.933.440 | todo | el módulo 10, «cuántos parámetros tiene» |
| **no-embedding** | 7.622.720 | total − tabla de embeddings | Chinchilla, ejercicio 3 |
| **params_matmul** | 8.929.280 | todo menos las escalas de normalización | los FLOPs, ejercicio 1 |

Se parecen y no son intercambiables. Dos observaciones que los ordenan:

**`params_matmul` incluye la proyección final aunque haya weight tying.** Atar los pesos ahorra
**memoria**, no **cálculo**: la multiplicación por la matriz de $320 \times 4096$ se hace igual
en cada token. Por eso el término suma `d_model * vocab_size` pese a que el módulo 10 contara
esa `lm_head` como 0 parámetros. Hay un test dedicado
(`test_la_proyeccion_final_cuenta_aunque_haya_tying`).

**Y `params_matmul` es exactamente el grupo *con decay* del módulo 11**, los 8.929.280 de 43
tensores. No es casualidad: los dos criterios son el mismo, «todo lo que es una matriz». Lo que
tiene 2 dimensiones o más participa en multiplicaciones de matriz y recibe weight decay; las
escalas de RMSNorm, que tienen 1 dimensión, ni lo uno ni lo otro.

**Cuidado entonces con el `6N` del módulo 01.** La regla dice «coste por token ≈ 6 × número de
parámetros», y el `N` que hay que meter ahí es `params_matmul`:

```
   6 × 8.929.280  = 53.575.680   = el término matmul  ✓
   6 × 7.622.720  = 45.736.320   ✗ no cuadra
```

Y sin embargo el `N` de Chinchilla, en el ejercicio 3, **sí** es el no-embedding. Son dos
fórmulas distintas con dos convenios distintos, y mezclarlos es el error silencioso de este
módulo.

---

## Ejercicio 2: cuánto de tu GPU estás usando (`compute_mfu`)

Ya tienes lo que cuesta un token y en el módulo 01 mediste el pico de tu hardware. La **MFU**
(*Model FLOPs Utilization*) junta ambas cosas:

$$\text{MFU} = \frac{\text{tokens/s} \times C_{\text{token}}}{\text{FLOPS pico}}$$

Son dos líneas de código: una validación y una división. El único sitio donde se puede meter la
pata es el `1e12` que convierte TeraFLOPS en FLOPS, para que numerador y denominador estén en
las mismas unidades.

### Qué sale de verdad

La demo entrena unos pasos y cronometra. Esto es lo medido en la máquina de desarrollo (M5,
pico estimado 14,0 TFLOPS):

| batch | tokens/paso | ms/paso | tokens/s | MFU |
|---|---|---|---|---|
| 1 | 256 | 12,2 | 21,0k | 7,9% |
| 2 | 512 | 21,5 | 23,8k | 8,9% |
| 4 | 1.024 | 40,9 | 25,1k | 9,4% |
| 8 | 2.048 | 80,8 | 25,4k | 9,5% |
| 16 | 4.096 | 162,6 | 25,2k | 9,5% |

Lo interesante de esa tabla no es el número final, es **la forma de la curva**: sube con el
batch y luego se estanca. Ese punto de estancamiento es donde dejas de estar limitado por el
lanzamiento de kernels y pasas a estarlo por el cálculo. Subir el batch más allá ya no compra
nada de eficiencia — sólo memoria.

### Qué valor es razonable

| situación | MFU típica |
|---|---|
| modelos grandes bien optimizados en A100/H100 | 0,4 – 0,5 |
| modelos medianos | 0,2 – 0,3 |
| **nuestro modelo de 9M** | 0,1 – 0,2 |
| algo va mal | < 0,05 |

**Nadie llega a 1.** El pico teórico sólo se alcanza con matmuls enormes perfectamente alineados
y absolutamente nada más de por medio.

Y con un modelo pequeño la MFU baja es **inevitable**, no un fallo tuyo: las matrices de
320×320 no dan para saturar los tensor cores, y una parte importante del tiempo se va en lanzar
kernels en vez de en calcular. Es el mismo fenómeno que mediste en la demo del módulo 01, donde
las matrices de 128 daban menos de 2 TFLOPS y las de 2048 diez veces más.

### Para qué sirve de verdad

No por su valor absoluto, sino porque es **comparable**: no depende del modelo ni del hardware.
«25.000 tokens por segundo» no te dice nada; «9,5% de MFU» sí. Cambias el batch size, activas
`torch.compile`, mueves el dataloader a otro hilo, y miras si el número sube. Es el termómetro
de las optimizaciones, y lo vas a usar en el módulo 13.

### Dónde se va el tiempo cuando la MFU es baja

Cuatro sospechosos, en orden de frecuencia:

1. **El dataloader.** Si preparar el siguiente batch tarda más que procesarlo, la GPU espera. Se
   detecta cronometrando `get_batch` por separado, que ya lo hiciste en el módulo 04 — allí
   salió que era el 0,04% del paso, así que en nuestro caso no es esto.
2. **El batch es pequeño.** Menos trabajo por lanzamiento de kernel. Es lo primero que hay que
   probar, y la tabla de arriba te dice exactamente dónde deja de compensar.
3. **Sincronizaciones accidentales.** Cualquier `.item()`, `float(tensor)` o `print` de un
   tensor obliga a la CPU a esperar a que la GPU termine. Dentro del bucle de entrenamiento eso
   mata el rendimiento, y es fácil colarlo sin darse cuenta al añadir logging.
4. **Operaciones memory-bound.** Normalizaciones y activaciones no aparecen en el conteo de
   FLOPs pero sí consumen tiempo. En un modelo pequeño son una fracción importante — es el mismo
   asunto que la tabla de tiempos de RMSNorm del módulo 07.

---

## Ejercicio 3: cómo repartir el presupuesto (`chinchilla_optimal_allocation`)

Ahora la pregunta de diseño. Tienes un presupuesto fijo de cómputo —una GPU y dos semanas,
digamos—. Puedes gastarlo en un **modelo grande con pocos datos** o en un **modelo pequeño con
muchos datos**. ¿Cuál acaba con menos pérdida?

Durante años se asumió que había que hacer los modelos más grandes, y punto. GPT-3 tenía
175.000 millones de parámetros entrenados con 300.000 millones de tokens.

En 2022, Hoffmann et al. midieron esto en serio: entrenaron más de 400 modelos de distintos
tamaños con distintas cantidades de datos y ajustaron una superficie. Su conclusión:

> **Parámetros y datos deben crecer proporcionalmente. Unos 20 tokens por parámetro.**

GPT-3 tenía **1,7 tokens por parámetro**, doce veces por debajo del óptimo. Para demostrarlo
entrenaron **Chinchilla**: 70.000 millones de parámetros y 1,4 billones de tokens, con el mismo
presupuesto de cómputo que Gopher, que tenía 280.000 millones. Chinchilla ganó en casi todos los
benchmarks **con la cuarta parte de parámetros**.

### La aritmética, que son tres líneas

Partiendo del $C = 6ND$ del módulo 01 y de la restricción $D = kN$ (con $k$ los tokens por
parámetro):

$$C = 6N(kN) = 6k\,N^2 \quad \Longrightarrow \quad N = \sqrt{\frac{C}{6k}}, \qquad D = kN$$

Y de ahí sale una consecuencia que conviene interiorizar: **si duplicas el presupuesto, el
modelo óptimo no se duplica, crece un 41%** ($\sqrt{2}$). Y los datos, otro 41%. Los dos a la
vez, nunca uno solo. Para cuadruplicar el modelo hay que multiplicar el cómputo por dieciséis.

### Compruébalo contra la realidad

Ésta es la parte que da confianza en la fórmula, y es lo que hace la demo. Con el presupuesto
real de Chinchilla, $5{,}88 \times 10^{23}$ FLOPs:

```
   N = √(5,88·10²³ / 120) = 7,0·10¹⁰ = 70.000 millones de parámetros
```

El modelo real tenía 70.000 millones. Verla acertar sobre un caso histórico da bastante más
confianza que leer la derivación.

Y aplicada a los modelos que conoces:

| modelo | parámetros | tokens | tok/param | óptimo Chinchilla | veredicto |
|---|---|---|---|---|---|
| GPT-3 | 1,75e11 | 3e11 | 2 | 5,12e10 | infraentrenado |
| Gopher | 2,8e11 | 3e11 | 1 | 6,48e10 | infraentrenado |
| Chinchilla | 7e10 | 1,4e12 | 20 | 7,0e10 | en el punto |
| Llama-2 7B | 7e9 | 2e12 | 286 | 2,65e10 | sobreentrenado a propósito |
| Llama-3 8B | 8e9 | 1,5e13 | 1875 | 7,75e10 | sobreentrenado a propósito |
| **el nuestro** | **7,62e6** | **5e8** | **66** | **1,38e7** | **sobreentrenado a propósito** |

Fíjate en la fila de Chinchilla: es la única que cae justo en su propio óptimo, que es
exactamente lo que el paper se propuso demostrar.

### Y por qué el nuestro está tres veces por encima

```
   parámetros no-embedding : 7,62 M
   tokens                  : 500 M
   tokens por parámetro    : 66        (el "óptimo" serían 20)
```

Es deliberado, y por dos razones.

**La primera: Chinchilla optimiza el cómputo de *entrenamiento*, no el de uso.** Si el modelo se
va a ejecutar muchas veces después, conviene uno más pequeño y más entrenado: el entrenamiento
se paga una vez y la inferencia, cada vez. Llama-3 lleva esto al extremo con ~1.800 tokens por
parámetro, noventa veces por encima de Chinchilla, y no es un error: es que su función objetivo
es otra. Un modelo de 8B que se ejecuta mil millones de veces sale rentabilísimo aunque
entrenarlo haya costado de más.

**La segunda: a esta escala entrenar de más es barato.** Horas, no meses. Y da un modelo
notablemente mejor. La optimalidad de Chinchilla importa cuando el cómputo es el recurso escaso;
aquí el recurso escaso es tu paciencia.

O sea que «sobreentrenado» no es un insulto ni «infraentrenado» un diagnóstico automático de
error: son posiciones en un compromiso, y cuál te conviene depende de qué vas a hacer con el
modelo.

---

## KV cache: por qué generar es distinto de entrenar

Un apunte que prepara el módulo 14 y que se entiende justo ahora, con el desglose del ejercicio
1 fresco.

Al entrenar procesas los 512 tokens de golpe y aprovechas la paralelización. Al **generar**,
produces un token cada vez, y en cada paso el modelo recalcula las claves y valores de *todos*
los tokens anteriores, que no han cambiado desde el paso anterior.

Guardarlos convierte un coste cuadrático en lineal. El precio es memoria:

$$\text{memoria KV} = 2 \times n_{\text{layers}} \times T \times d_{\text{model}} \times \text{bytes}$$

Para nuestro modelo con 512 tokens en fp16: $2 \times 6 \times 512 \times 320 \times 2 = 3{,}9$
MB. Nada, comparado con los 1007 MB de logits que contaste en el módulo 10. Pero para un modelo
de 70B con contexto de 100.000 serían decenas de gigabytes, y por eso existen técnicas como
*grouped-query attention*.

## Dónde está el debate

Las leyes de escala están **peor establecidas de lo que su nombre sugiere**, y merece la pena
saberlo antes de citarlas con demasiada seguridad.

Los coeficientes de Chinchilla se ajustaron a un rango concreto de escalas y a un dataset
concreto, y **extrapolar fuera de ahí no está justificado**. De hecho, en 2024 un grupo reanalizó
los datos originales y encontró que el ajuste tenía problemas metodológicos y que los intervalos
de confianza eran mucho más amplios de lo reportado. La conclusión cualitativa —«hay que entrenar
con más datos de los que se creía»— se sostiene; los números exactos, con más cautela. El 20 que
usas por defecto en el ejercicio 3 es una cifra redonda cómoda, no una constante de la
naturaleza.

Además, las leyes de escala predicen **pérdida**, no capacidades. La relación entre bajar la
pérdida y «razonar mejor» no es directa ni está bien entendida, y es una de las discusiones
abiertas más importantes del campo.

Y hay algo que ninguna ley de escala captura: **la calidad de los datos**. El paper de
TinyStories —el dataset que vas a usar— muestra que un corpus pequeño y muy limpio permite a
modelos diminutos generar texto coherente, algo que no se consigue con la misma cantidad de
texto de internet. Ningún $N$ ni $D$ recoge eso, y es literalmente la razón de que un modelo de
nueve millones de parámetros vaya a escribir algo legible al final del módulo 13.

---

**Para ampliar:** Hoffmann et al. 2022,
[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
(Chinchilla) · Kaplan et al. 2020,
[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) · Chowdhery et al.
2022, [PaLM](https://arxiv.org/abs/2204.02311) (definición de MFU) · Besiroglu et al. 2024,
[Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102).
Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
