# 01 — Entorno y hardware

## Por qué importa este módulo

**Este módulo evita que pierdas una semana.**

Vas a entrenar un modelo en tu ordenador. Antes de escribir una sola línea de red neuronal
conviene saber si eso va a tardar dos horas o dos semanas, porque la respuesta cambia todas
las decisiones que vienen después: cuántas capas, qué contexto, qué tamaño de batch, y si
la tirada final se lanza un viernes por la tarde o hay que replantearla entera.

Hay una segunda razón, menos obvia y más importante a largo plazo. Casi todo el mundo que
entrena modelos copia los números de un tutorial sin saber de dónde salen, y cuando algo no
encaja —no cabe en memoria, va cuatro veces más lento de lo esperado— no tiene con qué
diagnosticarlo. Al terminar este módulo vas a saber **calcular** lo que cuesta un modelo
antes de construirlo. Eso es lo que separa diseñar de copiar.

Aviso de expectativas, porque este módulo es distinto a los demás: aquí no se construye
nada del modelo. Se construye el instrumento de medida. Son tres funciones cortas y una
demo que interroga a tu máquina de verdad.

### Qué sabrás al terminar

- Cuántas operaciones cuesta procesar un token, y **de dónde sale la fórmula**, término a
  término.
- Cuántos TFLOPS da tu GPU **de verdad**, no lo que dice la ficha técnica —y por qué la
  diferencia entre las dos cifras es de un factor de 3, no de un 5%.
- Por qué las matrices pequeñas dejan la GPU muerta de risa, con la curva medida.
- Por qué una GPU sin bf16 está obligada a usar `float16` y qué problema silencioso trae eso.
- **Dónde se va la memoria** de una tirada de entrenamiento, que casi nunca es donde crees.
- Estimar cuánto va a durar un entrenamiento antes de lanzarlo.

### Cuánto cuesta

45 minutos. Tres funciones cortas, y la demo mide tu hardware de verdad.

---

## 1. La pregunta: ¿cómo se mide "cuánto cuesta"?

Un ordenador no tarda lo mismo en dos tareas distintas, así que hace falta una unidad común.
Se usa el **FLOP**: una operación con números decimales, una suma o una multiplicación.
Entrenar un modelo son un montón de FLOPs, y una GPU puede hacer unos cuantos billones por
segundo.

Dos números y una división:

```
tiempo = FLOPs totales que hay que hacer / FLOPs por segundo que da mi máquina
```

Todo el módulo va de estimar bien esos dos números. El numerador se calcula con papel y
lápiz —es aritmética exacta sobre la arquitectura— y el denominador **hay que medirlo**,
porque el que viene en la caja no se parece al que vas a conseguir.

## 2. De dónde salen los FLOPs de una red

Casi todo lo que hace una red neuronal es **multiplicar matrices**. Así que empecemos por
contar exactamente lo que cuesta una, con números que puedas seguir a mano.

Multiplica una matriz de 2×3 por otra de 3×2. El resultado es 2×2, o sea 4 números. Cada uno
de esos 4 sale de emparejar 3 valores con otros 3, multiplicarlos y sumarlos: 3
multiplicaciones y 2 sumas, que por convención se redondean a 6 operaciones (2 por cada
pareja). Total:

```
4 números de salida × 6 operaciones = 24 FLOPs
```

En general, multiplicar una matriz $m \times k$ por una $k \times n$ cuesta $2mnk$ FLOPs.

Ahora el paso que convierte esto en una regla que se usa a diario. Una capa de la red guarda
sus pesos en una matriz. Si esa matriz tiene $P$ números dentro, pasar **un token** por ella
cuesta $2P$ FLOPs. Y se entiende sin fórmulas: cada peso se toca exactamente una vez, en una
multiplicación y una suma.

Con eso ya sabes estimar el forward de cualquier red: cuenta sus parámetros y multiplica por
dos.

### Y el backward

Entrenar no es solo pasar los datos hacia delante. Hay que averiguar cómo ajustar cada peso,
y eso es el *backward* (módulo 02). Cuesta aproximadamente el **doble** que el forward,
porque por cada multiplicación que hizo el forward tiene que hacer dos: una para saber cómo
cambiar la entrada de la capa —el gradiente que hay que pasarle a la capa anterior— y otra
para saber cómo cambiar sus pesos.

Forward más backward, uno más dos, sale la regla que verás citada en todas partes:

$$C_{\text{token}} \approx 6N$$

donde $N$ son los parámetros del modelo. Seis FLOPs por parámetro y por token. Ya está: esa
es la fórmula con la que se presupuestan las tiradas de millones de euros.

### La atención se cuenta aparte

Hay una parte del Transformer que no encaja en la regla, porque no sale de multiplicar por
pesos sino de multiplicar tokens **entre sí**: cada token se compara con todos los
anteriores. Es la atención (módulo 06), y su coste no depende del tamaño del modelo sino de
cuántos tokens haya en la ventana:

$$C_{\text{token}} \approx 6N + 12 \cdot n_{\text{capas}} \cdot T \cdot d_{\text{model}}$$

Con los números de nuestro modelo ($T=512$, 6 capas, $d_{\text{model}}=320$, $d_{ff}=896$,
vocabulario 4096), el desglose exacto es:

```
por parámetros (6N)   53.575.680 FLOPs/token     82%
por atención          11.796.480 FLOPs/token     18%
                      ----------
total                 65.372.160 FLOPs/token
```

Ese `65.372.160` es el número exacto que tiene que devolver tu ejercicio 2. Y ahora fíjate
en lo que pasa si dejas el modelo igual y solo alargas la ventana a 4096 tokens: el primer
término no se mueve —los parámetros son los mismos— y el segundo se multiplica por ocho.
La atención pasa del 18% al **64%** del coste total. Por eso los modelos de contexto largo
son caros, y por eso hay un campo entero de investigación dedicado a ese término.

### Lo que este cálculo ignora, y por qué importa

La fórmula no cuenta las normalizaciones, las activaciones ni el softmax. No es que sean
gratis: es que su coste no está en calcular, sino en **mover datos entre la memoria y el
procesador**. Una GELU hace una operación por número, pero para hacerla tiene que traerse el
número desde la memoria y devolverlo, y eso —a escala de GPU— es lentísimo comparado con
multiplicar.

De ahí sale una distinción que conviene tener en la cabeza el resto del curso:

- una operación **compute-bound** está limitada por la potencia de cálculo (un matmul
  grande);
- una operación **memory-bound** está limitada por el ancho de banda de la memoria (una
  activación, una normalización, un dropout).

En un modelo pequeño como el nuestro, las memory-bound se llevan una parte nada despreciable
del tiempo real. Esa diferencia entre "los FLOPs que cuento" y "los segundos que tardo" es
exactamente lo que mide la MFU, que es lo siguiente.

## 3. MFU: cuánto de tu máquina estás usando de verdad

$$\text{MFU} = \frac{\text{tokens/s} \times C_{\text{token}}}{\text{FLOPS pico del hardware}}$$

Arriba, los FLOPs útiles que estás haciendo por segundo. Abajo, los que la máquina podría
hacer. El cociente es la fracción que aprovechas.

Si tu GPU puede hacer 50 billones de FLOPs por segundo y tú solo le estás sacando 10, tu MFU
es 0,2. **Nadie llega a 1.** Un modelo grande bien optimizado anda por 0,4-0,5. El nuestro,
de 9 millones de parámetros, se va a quedar mucho más abajo, y no es culpa tuya.

La razón es de tamaño. Una GPU tiene miles de unidades de cálculo, y para tenerlas todas
ocupadas necesita matrices grandes. Las nuestras son de 320×320, que en términos de GPU es
diminuto: pasa más tiempo recibiendo instrucciones y esperando a la memoria que
multiplicando.

Esto no hay que creérselo, se mide. Esta es una curva real, medida con `llmfs demo 01` sobre
una máquina de referencia —un portátil con GPU integrada, backend MPS—. La tuya dará otras
cifras; lo que importa es la **forma** de la curva, que es la misma en cualquier GPU:

| lado de la matriz | fp32 | fp16 |
|---|---|---|
| 320 (el de nuestro modelo) | 0,46 TFLOPS | 0,84 TFLOPS |
| 640 | 0,74 | 2,24 |
| 1024 | 3,51 | 2,99 |
| 2048 | 3,77 | 15,37 |
| 4096 | 3,61 | **15,90** |

Lee la primera fila y la última. **El mismo chip, la misma operación, el mismo tipo de dato,
y una diferencia de 8 veces en fp32 y de 19 veces en fp16** solo por el tamaño de la
matriz. Con matrices de 320 —las de nuestro modelo— esa máquina está dando el 12% de lo que
sabe hacer en fp32, y el 5% de lo que sabe hacer en fp16.

Ejecuta la demo y mira tu propia tabla: los valores absolutos cambiarán mucho entre una GPU
discreta y una integrada, pero la fila de 320 seguirá siendo una fracción pequeña de la
última.

Esa tabla explica de golpe tres cosas que si no parecen arbitrarias: por qué los modelos
grandes son *más* eficientes por FLOP que los pequeños, por qué merece la pena subir el
batch size hasta donde quepa, y por qué tu modelo de 9M no va a acercarse ni de lejos a las
cifras que se leen en los papers.

## 4. Por eso el ejercicio 1 te hace medir

Las fichas técnicas de las GPU de gama media citan cifras de decenas de TFLOPS. Esos números
son reales en el sentido de que se pueden alcanzar: con matrices enormes, en 16 bits, con
las unidades de cálculo perfectamente alimentadas y sin hacer nada más. En un entrenamiento
de verdad no los vas a ver nunca.

Esto es lo que ocurre al entrenar de verdad la config `tiny_char` en la misma máquina de
referencia (una tirada completa, ~70 segundos):

```
paso 2,925/2,929  perdida 1.3288  110.6k tok/s  MFU 4.7%
```

110.600 tokens por segundo, y `tiny_char` cuesta 5.948.160 FLOPs por token. Multiplicando:
**0,66 TFLOPS efectivos**. Ahora divide eso por un "pico" y mira lo que pasa según cuál
elijas:

```
0,66 / 14,0  (el pico nominal del backend, que es lo que usa el logger)  =  4,7 %
0,66 /  3,77 (el pico fp32 MEDIDO en esa máquina, y la tirada va en fp32) = 17,5 %
```

**Es la misma tirada.** La MFU no ha cambiado; ha cambiado el denominador. El 4,7% invita a
pensar que hay algo roto y a perder una tarde optimizando; el 17,5% dice la verdad, que para
un modelo de 0,8M de parámetros con matrices de 128×128 eso está más o menos donde tiene que
estar.

Esa es toda la moraleja del ejercicio 1, y es la razón de que te haga cronometrar en vez de
consultar: **el único número que sirve de denominador es el de tu máquina, medido con tu
dtype y tus tamaños.**

### Las tres trampas de cronometrar una GPU

Medir esto mal es facilísimo, y el ejercicio está construido alrededor de las tres formas de
equivocarse:

1. **No calentar.** La primera multiplicación de un tamaño dado es entre 10 y 100 veces más
   lenta que las siguientes: la GPU está eligiendo qué kernel usar y reservando memoria. Con
   diez iteraciones cronometradas, esa primera domina la media y arruina el resultado.
2. **No sincronizar.** Esta es la que muerde. `a @ b` en GPU **no espera a nada**: encola el
   trabajo y devuelve el control inmediatamente. Si cronometras sin sincronizar, estás
   midiendo lo que tarda la CPU en encolar una orden —unos 20 microsegundos— y te salen
   miles de TFLOPS. El resultado es tan absurdo que hay un test puesto expresamente para
   cazarlo.
3. **Medir un solo tamaño y creer que es "el pico".** Ya has visto la tabla: el pico depende
   del tamaño. Por eso la demo barre seis.

## 5. Precisión: por qué 16 bits y no 32

Los números decimales se guardan repartiendo bits entre el *exponente* —cuán grande o
pequeño puede ser el número— y la *mantisa* —cuántas cifras significativas tiene—:

| formato | exponente | mantisa | rango |
|---|---|---|---|
| fp32 | 8 bits | 23 bits | $10^{\pm 38}$ |
| fp16 | 5 bits | 10 bits | $6\times10^{-5}$ a $65504$ |
| bf16 | 8 bits | 7 bits | $10^{\pm 38}$ |

Usar 16 bits en lugar de 32 ocupa la mitad de memoria y va aproximadamente el doble de
rápido —en la tabla de arriba, más de cuatro veces—. La pega está en el rango.

**fp16 tiene un rango minúsculo.** Durante el entrenamiento, los gradientes de las capas
profundas son números muy pequeños, del orden de $10^{-7}$. Mira la tabla: el número
positivo más pequeño que fp16 representa con normalidad es $6\times10^{-5}$. Un gradiente de
$10^{-7}$ **es cero en fp16**. Y cuando un gradiente es cero, el peso correspondiente no se
mueve: esa capa deja de aprender. Sin mensaje de error, sin excepción, sin nada. Solo una
curva de pérdida que se queda plana y una tarde intentando entender por qué.

La solución tiene nombre y es más simple de lo que parece: **`GradScaler`**. Antes de
calcular los gradientes, multiplica la pérdida por un número grande (del orden de 65.000).
Como el gradiente es una derivada y la derivada es lineal, **todos** los gradientes quedan
multiplicados por ese mismo factor, y suben al rango representable. Justo antes de
actualizar los pesos, se divide otra vez por el mismo número. El resultado matemático es
idéntico; lo único que ha cambiado es que los números han hecho el viaje por una zona donde
fp16 sabe contar. Si algún valor se pasa por arriba y sale infinito, se descarta ese paso
entero y se baja el factor.

**bf16 no necesita nada de esto**, porque conserva los 8 bits de exponente de fp32. Paga el
precio en mantisa —solo 7 bits de precisión— y resulta que en deep learning el rango importa
muchísimo más que la precisión. Por eso bf16 se comió el mundo en cuanto el hardware lo
soportó.

### Qué toca en cada backend

Nada de esto tienes que decidirlo tú: `llmfs/device.py` detecta el hardware y elige. Pero
conviene saber qué está eligiendo y por qué, porque explica bastantes rarezas.

**En GPUs NVIDIA anteriores a Ampere** (`sm_75` y por debajo: la serie RTX 20, las GTX)
**no hay bf16 en hardware**, así que toca fp16 + GradScaler. Hay tres trampas conocidas ahí:

- `torch.cuda.is_bf16_supported()` **devuelve `True` en esas tarjetas**. Es cierto y es
  inútil: cuenta una emulación por software que da el resultado correcto y va lentísima. Por
  eso el código mira directamente la *compute capability* en vez de fiarse de esa función;
  bf16 de verdad empieza en `sm_80`.
- **FlashAttention-2 tampoco funciona** por debajo de `sm_80`. No pasa nada:
  `F.scaled_dot_product_attention` detecta la GPU y cae a otro algoritmo
  (*memory-efficient*) que sí va y que también evita el pico de memoria del método ingenuo.
- **`torch.compile` está desactivado por defecto** en esas generaciones, porque falla a
  compilar con bastante frecuencia y cuando compila no siempre gana. Es un flag opcional,
  nunca el valor por defecto.

**En GPUs Ampere o posteriores** (`sm_80`+) hay bf16 nativo, y entonces no hace falta
GradScaler: es el camino cómodo.

**En Apple Silicon (backend MPS)** el valor por defecto es fp32. La memoria es unificada, así
que no hay trasiego por PCIe que amortizar y 16 bits gana menos que en una GPU discreta. Hay
además un detalle que causa lentitudes inexplicables: `PYTORCH_ENABLE_MPS_FALLBACK=1` está
activo —`llmfs/__init__.py` lo pone antes de importar torch— y hace que las operaciones sin
kernel Metal caigan a CPU **en silencio**. Si algo va cien veces más lento de lo que debería
en un Mac, mira ahí primero.

`uv run python -m llmfs device` te dice qué ha detectado y qué ha decidido en tu caso.

## 6. Dónde se va la memoria

Los FLOPs deciden cuánto tardas; la memoria decide si arrancas siquiera. Y el reparto casi
nunca es el que uno espera, así que hagamos la cuenta para nuestro modelo de 8.933.440
parámetros con la config real de la tirada final (batch 48, contexto 512, vocabulario 4096).

Lo que ocupa el modelo, en fp32, 4 bytes por número:

```
pesos                       8,93M × 4 B  =   35,7 MB
gradientes (uno por peso)   8,93M × 4 B  =   35,7 MB
estados de AdamW (m y v)    8,93M × 8 B  =   71,5 MB
                                            --------
                                             142,9 MB
```

143 MB. En cualquier tarjeta actual, nada. La intuición dice que el modelo es lo que llena la
GPU, y con un modelo pequeño la intuición está completamente equivocada.

Ahora el tensor de logits, que es la salida del modelo antes del softmax: un número por cada
token del batch y por cada palabra del vocabulario.

```
48 × 512 = 24.576 tokens por micro-batch
24.576 × 4096 = 100.663.296 logits
en fp32:  402,7 MB     ...y su gradiente, otros 402,7 MB
```

**Ochocientos megas para un tensor, frente a 143 MB para el modelo entero y todo su
optimizador.** Ese tensor es el mayor consumidor de memoria de la tirada final, por encima
incluso de las activaciones intermedias. Si en algún momento te quedas sin memoria, ahí es
donde hay que mirar primero, y la palanca que funciona es bajar el batch size o el contexto
—no adelgazar el modelo.

La razón de que sea tan grande es estructural y merece la pena verla: ese tensor escala con
`batch × contexto × vocabulario`, y el vocabulario (4096) es mucho más grande que el
`d_model` (320) con el que trabajan todas las capas internas. La última proyección infla los
datos por un factor de 12,8 justo antes del final.

## 7. Juntándolo todo: cuánto tarda la tirada final

Ya tienes las tres piezas. La estimación completa, con el pico medido de la máquina de
referencia (15,9 TFLOPS):

```
FLOPs totales = 65.372.160 FLOPs/token × 500.000.000 tokens = 3,27 × 10^16 FLOPs

a MFU 0,10  ->  5,7 horas
a MFU 0,20  ->  2,9 horas
a MFU 0,40 (que no vas a ver con un modelo de este tamaño)  ->  1,4 horas
```

Eso es lo que imprime `llmfs demo 01`, ya con el pico de **tu** máquina. Y es una estimación
honesta porque lleva su rango de incertidumbre explícito: entre tres y seis horas, no "unas
horas". Con ese número en la mano ya puedes decidir si la tirada se lanza esta noche o si
hay que recortar tokens.

Un aviso de honestidad sobre este repositorio: los tiempos que aparecen en el README para la
tirada de 500M tokens son **estimaciones calculadas con esta misma fórmula**, no medidas,
porque esa tirada todavía no se ha ejecutado en CUDA. Cuando alguien la ejecute, el número
real será lo primero que haya que corregir ahí.

## Dónde está el debate

La regla del $6N$ se cita como si fuera física, y no lo es: es un modelo del coste, con
supuestos discutibles y una convención pegada encima.

Asume que el backward cuesta exactamente el doble que el forward, lo cual depende de qué
activaciones guardes y cuáles recalcules: con *gradient checkpointing* —recalcular el
forward de algunos bloques en vez de guardarlo, para ahorrar memoria— el factor sube a 4 y
la fórmula pasa a ser $8N$. Ignora por completo todo lo que es memory-bound, que en modelos
pequeños no es un detalle. Y hay una decisión arbitraria en el término de la atención: como
la máscara causal solo necesita calcular medio triángulo, se podría dividir por dos, pero la
convención (nanoGPT, los papers) es no hacerlo.

Nosotros seguimos la convención para que tu MFU sea comparable con la de todo el mundo, no
porque sea más correcta. Si un día ves una MFU que parece el doble de buena que la tuya con
hardware parecido, comprueba antes de nada si el otro contaba la atención igual.

Sobre la MFU en sí hay una crítica más de fondo, y es que se ha convertido en una métrica
que se optimiza por sí misma. Una MFU alta significa que aprovechas el hardware, no que
entrenes bien: se puede subir la MFU con un modelo peor —matrices más gordas, menos capas—
y quedarse con una pérdida más alta. Es un diagnóstico de eficiencia, no un objetivo. El
objetivo sigue siendo la pérdida de validación.

---

**Para ampliar:** Kaplan et al. 2020,
[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (apéndice B, de
donde sale el $6N$) · Micikevicius et al. 2018,
[Mixed Precision Training](https://arxiv.org/abs/1710.03740) (el paper del GradScaler) ·
Chowdhery et al. 2022, [PaLM](https://arxiv.org/abs/2204.02311) (donde se define la MFU).
Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
