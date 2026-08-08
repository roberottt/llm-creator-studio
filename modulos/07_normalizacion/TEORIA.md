# 07 — Normalización y conexiones residuales: la fontanería que hace que entrene

## Por qué importa este módulo

**Porque sin esto, una red profunda no entrena. Punto.**

Dos piezas que no calculan nada interesante y que son la diferencia entre un modelo que
aprende y uno que devuelve `NaN` a los tres pasos. Son la fontanería del Transformer: nadie
las menciona en los titulares y sin ellas no hay nada.

Para que veas hasta qué punto es fontanería, un número: las trece capas de normalización del
modelo final suman **4.160 parámetros de 8.933.440**, el 0,047%. Y son irrenunciables. Este
módulo va de las dos piezas más baratas de todo el modelo y de por qué sin ellas no hay
modelo.

El problema que resuelven es concreto y lo vas a ver medido: los números que atraviesan una
red profunda tienden a explotar o a desvanecerse, y con 64 capas el gradiente que llega abajo
es **cero exacto**. No "muy pequeño": cero, por underflow de la coma flotante. La demo lo
mide y la tabla está más abajo.

Además, aquí está una de las decisiones de diseño donde más se aprende comparando: dónde
poner la normalización cambia si tu red necesita warmup o no.

### Qué sabrás al terminar

- Por qué los números se descontrolan al apilar capas, con la cuenta que lo explica y con la
  medición que lo confirma
- Qué hace exactamente LayerNorm, y **qué le sobra** (eso es RMSNorm)
- Por qué `x + f(x)` es una de las ideas más importantes del deep learning
- Por qué el ejercicio 1 es una función suelta y el 2 es una clase: es la primera vez que
  escribes una capa **con pesos propios**
- Pre-norm contra post-norm, medido: cuánto gradiente llega a la primera capa en cada caso, y
  por qué el argumento habitual a favor de pre-norm está a medias mal
- Dos trampas numéricas que no dan error y estropean el resultado: `unbiased` y el fp16

### Qué vas a escribir

Tres ejercicios. Esta teoría está ordenada para que los leas en este orden, y **cada uno
tiene su propia sección con su ejemplo numérico**:

| Ejercicio | Qué hace | Dónde se explica |
|---|---|---|
| 1. `layer_norm` | Centrar en 0 y escalar a varianza 1 | [§ LayerNorm](#ejercicio-1-layernorm-a-mano-layer_norm) |
| 2. `RMSNorm` | Lo mismo sin la media: lo que usa Llama y lo que usamos nosotros | [§ RMSNorm](#ejercicio-2-quitarle-la-mitad-rmsnorm) |
| 3. `prenorm_residual` | Una línea, y es la más importante del módulo | [§ Pre-norm](#ejercicio-3-dónde-van-los-paréntesis-prenorm_residual) |

Los tres son cortos: el primero son cinco líneas, el segundo tres, y el tercero **una**. Es el
módulo con menos código del curso y no es el más fácil, porque lo que se aprende aquí no está
en el código sino en entender qué se rompe sin él. Las 1,5 horas se van en la teoría y en la
demo, no en teclear.

Una nota sobre el ejercicio 2, porque es donde más gente se pregunta si se ha perdido algo:
`layer_norm` es una **función** y `RMSNorm` es una **clase que hereda de `nn.Module`**. No es
capricho ni inconsistencia; la razón está en [su sección](#por-qué-el-1-es-una-función-y-el-2-una-clase).

### Cuánto cuesta

1,5 horas.

---

## Qué parte del LLM es esta

Éste es el bloque del Transformer que empezaste en el módulo 06. Lo que traes hecho es la
caja del medio; lo que montas hoy es **todo lo demás del dibujo**:

```
    x ──┬──> NORMA ──> atención (módulo 06) ──┐
        │   (ej. 1-2)                         ├──> +  ──┬──> NORMA ──> MLP ──┐
        └─────────────────────────────────────┘  (ej.3)│   (ej. 1-2)  (mód 8)├──> +
                                                       └─────────────────────┘
                                                                        (ej. 3)
```

Las dos normas y las dos sumas de cada bloque. Seis bloques, más una normalización final
antes de los logits que se llamará `norm_f` y de la que hablaremos en el ejercicio 3. Trece
capas de normalización en total:

```
   6 bloques × 2 normas   =  12
   la final (norm_f)      =   1
   ─────────────────────────────
                             13  ×  320 parámetros  =  4.160
```

Y las sumas residuales no tienen ni un parámetro: son un `+`.

Compáralo con los 409.600 parámetros por capa de la atención del módulo anterior y verás la
desproporción entre lo que cuesta esta pieza y lo que aporta. Ésa es la idea que hay que
llevarse del módulo: **no todo lo que importa en una arquitectura tiene parámetros.**

---

## El problema: los números se descontrolan

Una red profunda es una composición de funciones. Cada capa multiplica por una matriz, y esas
multiplicaciones se acumulan.

Imagina que cada capa multiplica la magnitud de sus entradas por 1,2. Parece inofensivo:

```
   capa 1:  ×1,2  ->  1,2
   capa 2:  ×1,2  ->  1,44
   capa 3:  ×1,2  ->  1,73
   ...
   capa 40: ×1,2  ->  1470
```

Y si el factor fuera 0,8 en lugar de 1,2, después de 40 capas quedaría 0,00013. En un caso los
números explotan, en el otro se desvanecen. **Y lo mismo le pasa al gradiente hacia atrás**,
que es lo que de verdad hace daño: si el gradiente se desvanece, las capas de abajo no reciben
señal y no aprenden nada. Se quedan con los pesos aleatorios con los que nacieron.

Con fp16, que sólo llega hasta 65.504 por arriba y hasta $6\times10^{-5}$ por abajo, esto deja
de ser una molestia y se convierte en `inf` y en ceros.

### La medición, que es más brutal que la cuenta

La demo apila N bloques idénticos y mide la **norma del gradiente que llega a la entrada**.
Ésta es la tabla, y conviene mirarla despacio porque contiene el módulo entero:

| capas | nada | sólo norma | post-norm | pre-norm |
|---|---|---|---|---|
| 4 | 3,230e-01 | 1,379e+01 | 7,062e+01 | 7,876e+01 |
| 8 | 2,709e-03 | 1,892e+01 | 6,750e+01 | 8,678e+01 |
| 16 | 1,094e-07 | 1,947e+01 | 6,387e+01 | 9,707e+01 |
| 32 | 2,086e-16 | 1,343e+01 | 5,689e+01 | 1,236e+02 |
| 64 | **0,000e+00** | 5,103e+00 | 5,941e+01 | 1,467e+02 |

La primera columna es una red sin nada: capas lineales encadenadas. Con 4 capas el gradiente
ya ha perdido dos tercios; con 16 es $10^{-7}$; con 64 es **cero exacto**, no una
aproximación. Ha bajado por debajo del número más pequeño representable y la coma flotante lo
ha redondeado a cero. Esa red no es que aprenda despacio: sus primeras capas **no se mueven en
absoluto**, hagas los pasos que hagas.

Las otras tres columnas son las tres cosas que vas a construir. Vuelve a esta tabla cuando
termines cada ejercicio.

---

## Solución 1: normalizar

La idea es brutal en su simplicidad: **antes de cada bloque, vuelve a poner los números en una
escala conocida.** No importa lo que haya hecho la capa anterior; al entrar en la siguiente,
renormalizas. Así la cadena de factores del ejemplo se corta: da igual que una capa multiplique
por 1,2, porque el siguiente paso vuelve a dejarlo todo en escala 1.

### LayerNorm, con números

Toma el vector de un token, digamos de 4 dimensiones:

```
   x = [2.0, 8.0, 4.0, 6.0]
```

Calcula su media y su varianza:

```
   media    = (2+8+4+6)/4 = 5.0
   varianza = ((2-5)² + (8-5)² + (4-5)² + (6-5)²)/4 = (9+9+1+1)/4 = 5.0
   desviación = √5 = 2.236
```

Resta la media y divide por la desviación:

```
   x_norm = [(2-5)/2.236, (8-5)/2.236, (4-5)/2.236, (6-5)/2.236]
          = [-1.342, 1.342, -0.447, 0.447]
```

Ahora media 0 y varianza 1, venga de donde venga la entrada.

Pero forzar siempre media 0 y varianza 1 le quita libertad al modelo —a lo mejor a esa capa le
venía bien que la dimensión 7 fuera sistemáticamente grande—, así que se le devuelve con dos
parámetros aprendidos, $\gamma$ (escala) y $\beta$ (desplazamiento):

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

Son un vector de tamaño `d` cada uno, uno por dimensión, y salen del entrenamiento como
cualquier otro peso. El $\epsilon$ (típicamente $10^{-5}$) evita dividir por cero cuando todas
las componentes son iguales.

### Sobre qué se calcula la media: el punto que hay que tener claro

**La media y la varianza se calculan sobre las dimensiones de cada token, por separado.** Si
tu tensor es `(B, T, d)`, se normaliza a lo largo de `d`, y hay `B × T` normalizaciones
independientes.

```
   tensor (2, 3, 4):  2 secuencias × 3 tokens × 4 dimensiones

   token (0,0): [2.0, 8.0, 4.0, 6.0]   ──> su propia media y su propia varianza
   token (0,1): [1.0, 1.0, 1.0, 9.0]   ──> las suyas, sin mirar al de arriba
   token (0,2): ...
```

Nada que ver con BatchNorm, que normaliza a lo largo del batch: allí cada dimensión se
normaliza usando los valores de *esa misma dimensión en todos los ejemplos del lote*. Eso trae
dos problemas que en un modelo de lenguaje son inaceptables: el resultado de un ejemplo
depende de con quién le haya tocado compartir batch, y en inferencia, con un solo ejemplo, no
hay lote del que sacar estadísticas, así que hay que guardar medias móviles del entrenamiento
y rezar para que valgan.

LayerNorm no tiene ninguno de los dos: trata cada token independientemente, funciona igual con
batch de 1 que de 1000, y no guarda nada. Por eso es la que usan los Transformers.

---

## Ejercicio 1: LayerNorm a mano (`layer_norm`)

Son cinco líneas y son la fórmula traducida directamente:

```
   1.  mean = x.mean(dim=-1, keepdim=True)                 la μ
   2.  var  = x.var(dim=-1, keepdim=True, unbiased=False)  la σ²
   3.  normalized = (x - mean) / torch.sqrt(var + eps)     la fracción entera
   4.  si hay weight: multiplicar;  si hay bias: sumar     γ y β
   5.  return normalized
```

Los argumentos `weight` y `bias` son opcionales (`None`) porque la función tiene que servir
para las dos cosas: comprobar la normalización pura contra el ejemplo de arriba, y hacer de
capa completa cuando le pasas los parámetros. Con `weight=None` y `bias=None`, tu función
sobre `[2.0, 8.0, 4.0, 6.0]` tiene que devolver exactamente `[-1.3416, 1.3416, -0.4472,
0.4472]`, que es lo que devuelve `F.layer_norm` de PyTorch.

### La trampa: `unbiased=False`

`torch.var` divide por $n-1$ por defecto (la varianza *muestral*, con la corrección de
Bessel), porque la usa habitualmente quien estima la varianza de una población a partir de una
muestra. LayerNorm no estima nada: normaliza los números que tiene. Usa la **poblacional**,
que divide por $n$.

Con nuestro vector de 4 componentes:

```
   correcta (÷4):    varianza 5.000  ->  [-1.3416,  1.3416, -0.4472,  0.4472]
   con Bessel (÷3):  varianza 6.667  ->  [-1.1619,  1.1619, -0.3873,  0.3873]
```

Un 13,4% de diferencia en el resultado, y **ningún error**. Además el tamaño del fallo depende
de la dimensión: con `d=4` la varianza sale un 33% más alta, pero con `d=320` —el tamaño de
verdad— sale sólo un 0,3% más alta y la diferencia en el resultado es del 0,16%. Es decir: si
lo pruebas con el modelo real, el resultado parece bien. Éste es el prototipo de bug que este
curso intenta enseñarte a temer: el que sólo se ve si lo buscas con un ejemplo pequeño. El
test compara tu resultado contra las dos versiones y te dice a cuál te pareces.

### Los otros dos detalles

**`keepdim=True`.** Sin él, `mean(dim=-1)` sobre `(4, 8, 32)` devuelve `(4, 8)` en vez de
`(4, 8, 1)`, y la resta `x - mean` intenta emitir mal las dimensiones. A veces lanza error y a
veces —cuando las formas casualmente encajan— produce basura en silencio. Es exactamente la
misma trampa del `keepdim` del módulo 05, y no será la última vez.

**El `eps` va dentro de la raíz:** `sqrt(var + eps)`, no `sqrt(var) + eps`. Es lo que hace
`F.layer_norm`, y con varianza pequeña la diferencia importa: si la varianza vale $10^{-8}$,
la primera forma da $\sqrt{10^{-5}}\approx 0{,}0032$ y la segunda da
$10^{-4} + 10^{-5}$, unas treinta veces menos, con lo que multiplicas el resultado por treinta.

---

## Ejercicio 2: quitarle la mitad (`RMSNorm`)

En 2019, Zhang y Sennrich probaron algo: **¿y si no restamos la media?**

$$y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

Sólo se reescala por la raíz del cuadrado medio (*root mean square*, de ahí el nombre). Sin
restar media y sin $\beta$. Con el mismo vector:

```
   RMS = √((4+64+16+36)/4) = √30 = 5.477
   y   = [2/5.477, 8/5.477, 4/5.477, 6/5.477]
       = [0.3651, 1.4606, 0.7303, 1.0954]
```

Fíjate en que los cuatro números siguen siendo positivos: RMSNorm **no centra nada**, sólo
ajusta el tamaño. Si tu vector estaba desplazado, sigue desplazado.

### ¿Y eso no importa?

Es la pregunta correcta, y la respuesta honesta es: en la práctica no, y nadie sabe demostrar
por qué. La demo lo mide pasándoles a las dos el mismo tensor, primero centrado y luego
desplazado a propósito:

| entrada | media tras LayerNorm | media tras RMSNorm | diferencia máxima |
|---|---|---|---|
| media 0 | +0,0000 | −0,0135 | 0,171 |
| media +5 | −0,0000 | +0,9806 | 3,839 |
| media +50 | −0,0000 | +0,9998 | 4,924 |

Con datos ya centrados las dos hacen prácticamente lo mismo. Con un desplazamiento grande
divergen: LayerNorm lo elimina y RMSNorm lo conserva casi entero. Lo que pasa es que **dentro
de una red las activaciones suelen estar centradas**, así que el caso en el que difieren casi
no ocurre.

Zhang y Sennrich observaron que casi todo el beneficio de LayerNorm viene de **reescalar**, no
de **recentrar**, y que quitando la parte de la media se ahorra una pasada por los datos y un
tensor intermedio: entre un 7% y un 64% más rápido según el caso, sin pérdida de calidad
medible. Eso es un resultado **empírico**, comprobado entrenando, no un teorema. Pero es lo
bastante robusto como para que lo usen Llama, Mistral y prácticamente todo lo moderno, y por
eso es lo que usa nuestro modelo.

La otra mitad de la ganancia son los parámetros: `320` en vez de `640` por capa, porque no hay
$\beta$.

### Cuidado al leer los tiempos de la demo

La demo cronometra las tres implementaciones y saca esto, que a primera vista contradice todo
lo que acabo de contarte:

```
   F.layer_norm (PyTorch)   0,158 ms      640 parámetros
   tu layer_norm            0,445 ms      640
   RMSNorm                  0,512 ms      320       <- ¿la más lenta?
```

No hay que sacar de ahí que RMSNorm sea peor. Estas capas son **memory-bound**, el término del
módulo 01: no están limitadas por cuántas cuentas hacen sino por mover los datos entre la
memoria y el procesador. A esta escala lo que domina el cronómetro es el coste fijo de lanzar
cada kernel de PyTorch, y RMSNorm tal como la escribes son varias operaciones sueltas
(`pow`, `mean`, `rsqrt`, dos multiplicaciones) frente al kernel único y compilado de
`F.layer_norm`. Estás comparando tu implementación didáctica contra código C++ optimizado, no
un algoritmo contra otro.

La ganancia real que reportan Zhang y Sennrich se mide sobre el entrenamiento entero y con
implementaciones fusionadas de las dos. Lo que sí es un hecho comprobable aquí es la mitad de
parámetros.

### Por qué el 1 es una función y el 2 una clase

Aquí está la diferencia conceptual del módulo, y es de PyTorch, no de normalización.

`layer_norm` recibe sus parámetros como argumentos: `weight` y `bias` entran por la puerta y
la función no recuerda nada entre llamadas. Es una operación pura.

`RMSNorm` **es dueña de su parámetro**. En `__init__` escribes:

```python
self.weight = nn.Parameter(torch.ones(dim))
```

`nn.Parameter` es un tensor marcado con un cartel que dice "esto se entrena". Eso hace tres
cosas de golpe, todas las que viste en el módulo 05: aparece en `modelo.parameters()`, así que
el optimizador lo actualizará; se mueve a la GPU con `modelo.to(device)`; y se guarda con el
modelo. Un `torch.ones(dim)` a secas no haría nada de eso — sería una constante.

De ahí que el ejercicio 2 sea una clase: **es la primera capa del curso que tiene pesos
propios que no vienen de un `nn.Linear` o un `nn.Embedding`**, y son 4.160 de los 8.933.440
del modelo final.

**Y por eso se inicializa con `torch.ones` y no con `torch.randn`.** Al arrancar, la capa tiene
que ser la normalización pura, es decir, multiplicar por 1. Si `weight` empezara aleatorio
estarías escalando cada dimensión por un factor arbitrario antes de haber aprendido nada, y la
pérdida del paso 0 no cuadraría con `ln(V)` — el detector de bugs del módulo 05, que ya te ha
pillado un caso en el bigrama neuronal.

### El `.float()` no es paranoia

El `forward` que dicta el docstring lleva una conversión que parece de más:

```python
return self._norm(x.float()).type_as(x) * self.weight
```

Con activaciones en fp16, elevar al cuadrado desborda antes de lo que uno espera:
`300² = 90.000` y fp16 se acaba en 65.504. Lo puedes comprobar en dos líneas:

```python
>>> h = torch.tensor([300.0]).half()
>>> (h * h).item()
inf
```

Y a partir de ahí la cadena entera se cae: la media sería `inf`, `rsqrt(inf)` es 0, y **la capa
devolvería ceros**. No `NaN`, no un error: ceros, en silencio. Hay un test que reproduce ese
caso exacto. Como la RTX 2060 del curso entrena en fp16 obligatoriamente (no tiene bf16 en
hardware, módulo 01), esto no es un caso hipotético.

`torch.rsqrt(z)` calcula $1/\sqrt{z}$ de una vez: es un kernel menos que dividir y algo más
estable numéricamente.

### Un detalle que sorprende

Aunque hagas `.type_as(x)` para volver a fp16, la salida acaba siendo **fp32**, porque después
multiplicas por `self.weight`, que es un parámetro fp32, y PyTorch promociona el resultado al
tipo más ancho de los dos.

No es un bug: es exactamente lo que hace la implementación de Llama y es lo deseable. Bajo
autocast los pesos se mantienen en fp32 y las operaciones siguientes convierten lo que
necesiten. Hay un test que lo documenta, y lo menciono aquí para que no pierdas media hora
persiguiendo un `dtype` que está bien.

---

## Solución 2: conexiones residuales

La segunda pieza, y la más importante de las dos.

En vez de que cada bloque *sustituya* la representación, se le pide que la *modifique*:

$$x_{\text{salida}} = x + f(x)$$

El bloque calcula una corrección, no un reemplazo. A esa suma acumulada que atraviesa toda la
red se le llama **corriente residual** (*residual stream*), y es una forma muy útil de pensar
en un Transformer: hay un canal principal por el que viaja la representación de cada token, y
cada bloque lee de él, calcula algo y **suma** su aportación de vuelta.

**Por qué esto lo cambia todo:** derivando esa expresión respecto a $x$,

$$\frac{\partial x_{\text{salida}}}{\partial x} = 1 + \frac{\partial f(x)}{\partial x}$$

Ese **1** es una autopista. Aunque $\partial f/\partial x$ sea diminuto, el gradiente que llega
a las capas de abajo no baja de 1 por ese camino. Sin residuales, los factores de cada capa se
multiplican entre sí y el producto se desvanece; con residuales, siempre hay una ruta directa
por la que el gradiente pasa sin tocarse.

---

## Ejercicio 3: dónde van los paréntesis (`prenorm_residual`)

El ejercicio entero es esto:

```python
return x + fn(norm(x))
```

Una línea. Y ahora la parte que importa: hay dos formas de colocar la normalización, y sólo
cambian los paréntesis de sitio.

```
   post-norm (el paper de 2017):   norm(x + fn(x))
   pre-norm  (todo lo moderno):     x  + fn(norm(x))
```

Parece cosmético. No lo es, y lo que cambia es **por dónde pasa el gradiente**.

En **post-norm**, la normalización está *encima* del camino residual: el gradiente la atraviesa
en cada capa y se va reescalando. La autopista tiene un peaje en cada salida. Con 6 capas se
nota poco; con 40, hace falta un warmup cuidadoso del learning rate para que el entrenamiento
no explote en los primeros pasos.

En **pre-norm**, la normalización está *dentro* de la rama. El camino $x \to x$ queda
completamente libre y el gradiente llega intacto hasta la primera capa. Se puede entrenar sin
warmup y con learning rates más altos.

Si escribes `norm(x + fn(x))` has hecho post-norm, y hay un test que lo detecta.

### Cómo se comprueba que está bien

Hay un test que merece la pena entender porque es el módulo entero en tres líneas: anula por
completo el gradiente de la rama —`fn` devuelve un tensor desconectado del grafo, multiplicado
por cero— y verifica que el gradiente en la entrada sigue siendo **exactamente 1,0**.

Ese 1 es toda la razón de ser del residual. Aunque el bloque entero fuera inútil y no aportara
ninguna señal, la información sigue pasando y el gradiente sigue llegando. Un bloque de un
Transformer nunca puede hacer que la red vaya *peor*, porque siempre tiene la opción de no
aportar nada y dejar pasar la corriente. Es la razón de fondo por la que se pueden apilar 100
capas.

### Lo que la tabla dice de verdad, que no es lo que suele contarse

Vuelve ahora a la tabla del principio, a las dos últimas columnas, con 64 capas:

```
   nada         0,000e+00      <- muerto
   sólo norma   5,103e+00      <- vivo
   post-norm    5,941e+01
   pre-norm     1,467e+02
```

La lectura habitual es "pre-norm evita que el gradiente se desvanezca". Mirando los números,
eso está a medias mal: **la normalización sola ya rescata el problema**. Pasar de cero exacto a
5,1 es el salto grande, y lo da la columna que ni siquiera tiene residuales.

Lo que distingue de verdad a pre-norm es otra cosa, y se ve en la forma de la columna: es la
única de las cuatro en la que el gradiente **crece** con la profundidad (78 → 146) en vez de
decrecer. El camino $x \to x$ no tiene ningún peaje, así que cada capa que añades suma su
contribución en lugar de atenuar la de las anteriores. Normalización y residuales atacan el
mismo problema por caminos distintos y no son alternativas, son complementos.

### La consecuencia: `norm_f`, y por qué existe

Pre-norm tiene un precio. Como la corriente residual **nunca se normaliza por el camino**
—cada capa le suma su contribución y nadie vuelve a ajustarla—, llega al final con una escala
que crece con la profundidad. Se mide en el modelo real, midiendo la norma media del vector de
cada token al salir de cada bloque (modelo recién inicializado, sin entrenar):

```
   tras los embeddings:   0,357
   tras el bloque 1:      0,450
   tras el bloque 2:      0,559
   tras el bloque 3:      0,692
   tras el bloque 4:      0,868
   tras el bloque 5:      1,002
   tras el bloque 6:      1,183     <- 3,3 veces lo que entró
   ─────────────────────────────
   tras norm_f:          17,886     <- √320, o sea RMS 1 por componente
```

Si esos vectores de escala arbitraria fueran directos a la capa de logits, la pérdida del paso
0 no valdría `ln(V)` y el modelo arrancaría con opiniones fuertes por accidente — el caso
"inicialización mal" del módulo 05, otra vez.

Por eso **los modelos pre-norm llevan siempre una normalización final** antes de la capa de
salida. En el módulo 10 se llamará `norm_f`, y ahora ya sabes de dónde sale: no es un adorno,
es el parche obligatorio del diseño que acabas de escribir.

---

## Dónde está el debate

Aquí hay más de lo que parece.

**Por qué funciona la normalización sigue sin estar claro.** La explicación original de
BatchNorm (2015) fue el *internal covariate shift*: que normalizar estabiliza la distribución
de las entradas de cada capa. Santurkar et al. (2018) lo pusieron a prueba inyectando ruido
*después* de normalizar —destruyendo deliberadamente esa estabilidad— y la red seguía
entrenando igual de bien. La explicación original está hoy largamente descartada, y la
sustituta —que suaviza el paisaje de la función de pérdida— es más una observación empírica que
una teoría. Es decir: usas dos piezas imprescindibles cuyo mecanismo nadie sabe explicar del
todo, y eso es más normal en este campo de lo que a nadie le gustaría.

**Que RMSNorm baste es empírico.** Lo dice la sección de arriba y conviene repetirlo: no hay
ninguna razón teórica por la que recentrar deba ser prescindible. Se comprobó entrenando, se
sostiene desde 2019 y se ha extendido a casi todos los modelos grandes, que es la mejor
evidencia disponible; pero es evidencia, no demostración.

**Pre-norm no es gratis.** Está bastante aceptado que pre-norm entrena más fácil, pero hay
evidencia de que post-norm, cuando converge, alcanza mejor calidad final. Hay arquitecturas
recientes que usan variantes híbridas por esto mismo. Nosotros usamos pre-norm porque es lo
robusto y lo que hace todo el mundo, no porque esté demostrado que sea superior.

---

**Para ampliar:** Ba et al. 2016, [Layer Normalization](https://arxiv.org/abs/1607.06450) ·
Zhang & Sennrich 2019, [RMSNorm](https://arxiv.org/abs/1910.07467) · Xiong et al. 2020,
[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
(el análisis pre/post-norm) · He et al. 2015,
[Deep Residual Learning](https://arxiv.org/abs/1512.03385). Términos sueltos, en
[GLOSARIO.md](../../GLOSARIO.md).
