# 11 — El bucle de entrenamiento: las cuatro piezas que evitan que reviente

## Por qué importa este módulo

**Porque tener un modelo no es tenerlo entrenado.**

Ya sabes construir un GPT que produce logits y medir cuánto se equivoca. Falta la parte que
convierte eso en aprendizaje: cómo se mueven 8.933.440 parámetros para que la pérdida baje.

Y el bucle en sí **ya lo escribiste**, en el módulo 02, con tu propio motor de derivadas:
predecir, medir, gradientes, mover, poner a cero, repetir. Eso no cambia. Lo que se añade aquí
son cuatro piezas que hacen que ese bucle funcione **a escala** en vez de divergir a los
cincuenta pasos.

Cada una resuelve un problema concreto que verías si no estuviera, y la demo te enseña los
cuatro problemas ocurriendo. Son también las cuatro cosas que te van a permitir depurar un
entrenamiento que va mal en vez de cambiar números al azar, que es lo que hace todo el mundo.

### Qué sabrás al terminar

- Por qué un solo learning rate vale para toda la red, y qué hace Adam para conseguirlo
- Cómo se escribe un optimizador en PyTorch: `param_groups`, `state`, y por qué el `step` tiene
  la forma que tiene
- Qué es el warmup y por qué sin él el modelo a veces no se recupera nunca
- Cómo evitar que **un solo batch raro** destruya horas de entrenamiento, medido
- Qué parámetros NO deben decaer, y por qué aplicárselo a todos es un error silencioso
- Un detalle de AMP que se olvida siempre y hace que el entrenamiento se arrastre

### Qué vas a escribir

Cuatro ejercicios, y esta teoría los sigue en orden:

| Ejercicio | Qué hace |
|---|---|
| 1. `AdamWScratch` | El optimizador, desde cero |
| 2. `lr_at_step` | Cómo cambia el learning rate durante la tirada |
| 3. `clip_grad_norm` | Que un batch raro no destruya horas de trabajo |
| 4. `build_param_groups` | Qué parámetros decaen y cuáles no |

El ejercicio 1 es **el más largo del curso** y los otros tres son cortos: el 4 son cinco líneas.
Hay una pequeña dependencia circular entre el 1 y el 4 —el `step` que escribes en el 1 recorre
los grupos que construye el 4— pero ninguno necesita al otro para funcionar ni para pasar sus
tests. Si el ejercicio 1 se te atraganta, haz el 2, 3 y 4 primero y vuelve.

Cuando los cuatro estén en verde, **el modelo final entrenará con tu optimizador**.

### Cuánto cuesta

4 horas. Abre la Parte III: aquí se pasa de tener un modelo a entrenarlo.

---

## El bucle, y qué le falta

Empecemos por lo que ya conoces. El bucle de entrenamiento, desnudo, es esto:

```
   para cada paso:
       predecir y medir la pérdida       (forward)
       calcular los gradientes           (backward)
       mover los parámetros              (paso del optimizador)
       poner los gradientes a cero
```

Cuatro líneas, y las escribiste en el módulo 02 sobre 113 parámetros. Con 8,9 millones y diez
mil pasos, ese bucle tal cual **no llega al final**. Lo que le falta es lo que vas a escribir
hoy, y cada pieza va en un sitio concreto:

```
   para cada paso:
       lr = lr_at_step(paso, ...)          ← ejercicio 2: qué lr toca ahora
       ajustar el lr en los grupos del optimizador

       forward, backward                    (esto ya lo tienes)

       clip_grad_norm(params, 1.0)         ← ejercicio 3: acotar el daño de un batch raro
       optimizador.step()                  ← ejercicio 1: TU AdamW
       optimizador.zero_grad()

   y una sola vez, al construir el optimizador:
       build_param_groups(modelo, 0.1)     ← ejercicio 4: quién decae y quién no
```

Ésa es la foto completa. El resto de la teoría explica cada pieza en ese orden.

---

## Ejercicio 1: el optimizador (`AdamWScratch`)

### El problema: un solo learning rate no vale para todos

La versión más simple de "mover los parámetros" es el descenso de gradiente, la del módulo 02:

```
   p ← p − lr · gradiente
```

Funciona, y tiene un problema serio. Piensa en dos parámetros de tu modelo: uno del embedding de
la palabra `the`, que aparece en casi todas las frases, y otro del embedding de una palabra rara
que sale una vez cada mil. El primero recibe gradientes grandes constantemente; el segundo, casi
nunca. Con un único `lr` para ambos, o el primero da saltos absurdos o el segundo no se mueve en
toda la tirada.

Y no puedes poner un `lr` por parámetro a mano: son 8,9 millones.

### Las dos ideas de Adam

**Momento.** En vez de moverse según el gradiente de este paso, se usa una media móvil de los
recientes:

```
   m = 0,9·m + 0,1·gradiente
```

Como cada batch es una muestra distinta, sus gradientes son ruidosos. Promediar cancela el ruido
y deja la dirección consistente.

**Escalado por dimensión.** Se lleva también una media móvil del gradiente **al cuadrado**, y se
divide por su raíz:

```
   v = 0,95·v + 0,05·gradiente²
   paso = m / √v
```

Un parámetro con gradientes consistentemente grandes tiene $v$ grande y se mueve poco. Uno que
casi nunca recibe señal tiene $v$ pequeño y se mueve mucho cuando la recibe. **Cada parámetro
acaba con su propio learning rate efectivo**, calculado solo, y por eso un único `lr` global
funciona para todo el modelo.

La demo lo mide sobre la misma tarea, mismo `lr` y mismos pasos:

```
   SGD (sin momento ni escalado)   pérdida final 0,309866
   AdamW completo                  pérdida final 0,000200
```

Tres órdenes de magnitud, y la única diferencia es el escalado por dimensión.

### La corrección de sesgo

$m$ y $v$ empiezan en cero, así que los primeros pasos subestiman las magnitudes reales. Con
$\beta_2 = 0{,}95$, tras un paso $v$ vale sólo el 5% de $g^2$: dividir por su raíz daría un paso
4,5 veces mayor de lo debido.

La corrección lo arregla exactamente:

$$\hat{m} = \frac{m}{1-\beta_1^t}, \qquad \hat{v} = \frac{v}{1-\beta_2^t}$$

En el paso 1 con $\beta_2 = 0{,}95$: $1 - 0{,}95 = 0{,}05$, y dividir por 0,05 multiplica por 20,
que es justo el factor que faltaba. Según avanza $t$, $\beta^t \to 0$ y la corrección se
desvanece sola.

**Y ojo con el `t`: empieza en 1, no en 0.** Con $t=0$, $1 - \beta^0 = 0$ y estás dividiendo por
cero. Incrementa el contador **antes** de usarlo. Es el primer error clásico del ejercicio.

### La W de AdamW

*Weight decay* es empujar los pesos hacia cero para que no crezcan sin control. Hay dos formas de
hacerlo y la diferencia es toda la letra W:

```
   Adam + L2:   g ← g + λ·p       y luego Adam procesa ese g
   AdamW:       p ← p − lr·λ·p    directamente sobre el parámetro, aparte de Adam
```

En la primera, el decaimiento pasa por la división por $\sqrt{v}$, así que su efecto real acaba
dependiendo de la magnitud de los gradientes de ese parámetro: un peso con gradientes grandes
apenas decae, uno con gradientes pequeños decae muchísimo. Nadie quiere eso. Loshchilov y Hutter
(2019) lo desacoplaron y funcionó mejor de forma consistente.

**Éste es el segundo error clásico**: sumar el weight decay al gradiente en vez de aplicarlo al
parámetro. Hay un test que distingue las dos versiones
(`test_el_weight_decay_esta_desacoplado`).

### Cómo se escribe un optimizador en PyTorch

El `__init__` **ya está hecho**. Tu único trabajo es el método `step()`, y tiene una estructura
fija: dos bucles anidados y dentro seis operaciones. Tres cosas que hay que saber de la API antes
de escribirlo:

**`self.param_groups`** son los grupos del ejercicio 4: cada uno con sus propios parámetros y su
propio `weight_decay`. Por eso los hiperparámetros se leen **dentro** del bucle de grupos y no
una vez al principio — si los leyeras fuera, todos los parámetros compartirían el mismo decay y
el ejercicio 4 no serviría de nada.

**`self.state[p]`** es un diccionario por parámetro donde guardas $m$, $v$ y el contador de
pasos. La primera vez que tocas un parámetro está vacío (`len(state) == 0`) y hay que
inicializarlo. PyTorch lo serializa solo en `optimizer.state_dict()`, que es lo que permite
reanudar un entrenamiento a mitad — y con 8,9 millones de parámetros eso son dos tensores
adicionales por parámetro, los 71,5 MB del desglose de memoria del módulo 10.

**El `@torch.no_grad()`** es obligatorio. Estás modificando parámetros que tienen
`requires_grad=True`; sin él estarías construyendo grafo de autograd sobre las propias
actualizaciones, lo cual además de estar conceptualmente mal se comería la memoria.

### Las operaciones in-place

El docstring escribe las actualizaciones así:

```python
m.mul_(beta1).add_(grad, alpha=1 - beta1)            # m = beta1*m + (1-beta1)*g
v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v = beta2*v + (1-beta2)*g²
p.addcdiv_(m, denom, value=-step_size)               # p -= step_size * m/denom
```

El guion bajo final significa **in-place**: modifica el tensor en vez de crear uno nuevo. Con 8,9
millones de parámetros, reservar tensores nuevos en cada uno de los 10.172 pasos se nota.

Si te resultan crípticas, escríbelo con operaciones normales primero (`m = beta1*m + ...`) y
optimiza después: el test compara resultados, no estilo. Pero **cuidado con una cosa**: con la
versión no in-place estás creando tensores nuevos, así que tienes que volver a guardarlos en
`state["exp_avg"]` a mano. Si no, el estado se queda en ceros para siempre y el optimizador se
comporta como si no tuviera memoria.

### Cómo saber si está bien

El test entrena el mismo problema 50 pasos con tu optimizador y con `torch.optim.AdamW`, y
compara los pesos finales con `torch.allclose`. Es un oráculo externo: o coincides con la
implementación de referencia del mundo, o no.

**Un aviso sobre la demo, para que no te vuelvas loco.** La demo entrena 200 pasos para dibujar
la curva, pero la comparación de pesos la hace a 50, igual que el test. La razón es numérica y
merece la pena entenderla: con esa tarea la pérdida ya está prácticamente convergida hacia el
paso 100, y entonces $m$ y $v$ son los dos casi cero. El cociente $m/(\sqrt{v}+\epsilon)$ tiene
numerador y denominador diminutos, así que cualquier diferencia de último bit entre dos
implementaciones se amplifica sin parar. Medido con la referencia:

```
    50 pasos  ->  error 8e-07     (la pérdida todavía es 2,3e-01)
   200 pasos  ->  error 1,5e-04
   400 pasos  ->  error 4,2e-02
```

Dos implementaciones **idénticas** se separan así. No es un bug tuyo ni de nadie: es lo que pasa
al dividir números diminutos entre números diminutos.

---

## Ejercicio 2: el planificador del learning rate (`lr_at_step`)

El `lr` no es constante durante el entrenamiento. Tiene dos tramos, y la función que escribes
devuelve el que toca en cada paso.

**Warmup: subir despacio al principio.** En los primeros pasos los momentos de Adam están casi
vacíos y sus estimaciones son ruidosísimas — es el mismo problema que ataca la corrección de
sesgo, pero la corrección no lo resuelve del todo. Y además los pesos recién inicializados
producen gradientes grandes. Arrancar a `lr` completo suele producir un pico de pérdida del que a
veces el modelo no se recupera nunca. Se sube linealmente de casi 0 a `lr` durante 500 pasos.

**Coseno: bajar al final.** Al principio interesa moverse rápido y explorar; al final, afinar en
una zona buena. El coseno baja despacio, luego deprisa, luego despacio otra vez:

$$\text{lr}(t) = \text{lr}_{\min} + (\text{lr} - \text{lr}_{\min}) \cdot \frac{1 + \cos(\pi \cdot \text{progreso})}{2}$$

La diferencia frente a una recta es pequeña pero consistente en todos los papers que la han
medido.

Y **no se decae hasta cero**, sino hasta el 10% del `lr` inicial: por debajo de cierto punto el
modelo deja de aprender del todo y cada paso extra es cómputo tirado. Si vas a parar, mejor
parar.

### Los números de la tirada final

Así queda con los valores del config (`lr=1e-3`, `warmup=500`, 10.172 pasos):

| paso | lr | tramo |
|---|---|---|
| 0 | 2,000e-06 | warmup, arranca casi en cero |
| 250 | 5,020e-04 | warmup, a mitad |
| 500 | 1,000e-03 | fin del warmup: el máximo |
| 2.500 | 9,083e-04 | coseno |
| 5.086 | 5,865e-04 | coseno, a mitad de entrenamiento |
| 10.172 | 1,000e-04 | fin: el suelo del 10% |
| 12.000 | 1,000e-04 | pasado el final: se queda en el suelo |

### Comprueba la fórmula a mano antes de ejecutar nada

Es la forma de saber que el coseno está bien puesto sin correr los tests:

```
   progreso = 0  ->  cos(0) = 1    ->  coef = 1  ->  devuelve lr
   progreso = 1  ->  cos(π) = −1   ->  coef = 0  ->  devuelve min_lr
```

Si te sale al revés, te has dejado el `0.5 * (1 + ...)` y estás usando el coseno crudo, que va de
1 a −1 en vez de 1 a 0.

### Tres detalles que parecen decorativos y no lo son

**El `+1` del warmup.** Es `lr * (step + 1) / warmup_steps`, no `lr * step / warmup_steps`. Sin
él, el paso 0 tendría `lr` exactamente cero: un paso que no aprende nada, desperdiciado. Por eso
la tabla empieza en 2e-06 y no en 0.

**El `max(1, ...)` del denominador** evita dividir por cero si `max_steps <= warmup_steps`.

**El acotado del progreso a [0, 1]** es el que produce la última fila de la tabla. El coseno es
**periódico**: con progreso mayor que 1 empezaría a *subir* otra vez, y un entrenamiento que se
pasa de los pasos previstos vería el `lr` remontando de la nada. Acotado, se queda en el suelo.

---

## Ejercicio 3: el recorte de gradientes (`clip_grad_norm`)

Ocasionalmente un batch produce gradientes enormes: una secuencia rara, un token muy poco
frecuente, una línea corrupta del dataset. Sin protección, ese **único** batch puede dar un salto
que destruya horas de entrenamiento, y lo verás como un pico vertical en la curva de pérdida del
que el modelo tarda mucho en recuperarse, o no se recupera.

La solución: calcular la norma **global** de todos los gradientes juntos, como si fueran un solo
vector gigante, y si supera un umbral, multiplicarlos todos por el mismo factor.

```
   norma = √(Σ ‖g_i‖²)
   si norma > max_norm:  todos los g ×= max_norm / norma
```

### Global, no por tensor, y esto es el corazón del ejercicio

Recortar cada tensor por separado cambiaría la **dirección** del gradiente conjunto, que es justo
lo que no quieres tocar. El gradiente te dice *hacia dónde* ir; tú sólo estás limitando *cuánto*
avanzas en esa dirección. Multiplicando todos los tensores por el mismo escalar, la dirección se
conserva exactamente.

La demo lo comprueba midiendo el coseno entre las dos direcciones:

```
   norma ANTES de recortar : 112.858,7
   norma DESPUÉS           : 1,0000
   coseno entre las dos direcciones: 0,99999994
```

Ese coseno de 1 es lo importante. Y el efecto sobre un entrenamiento real, con un batch
envenenado en el paso 50:

```
   sin recortar        paso 49: 0,0561  ->  paso 55: 0,1698     la pérdida SUBE 3×
   con grad_clip=1.0   paso 49: 0,0501  ->  paso 55: 0,0409     ni se entera
```

### Por qué se devuelve la norma antes de recortar

Es lo que hace `torch.nn.utils.clip_grad_norm_`, y es lo útil. Si la registras en el log y ves
que sube de forma sostenida, el entrenamiento se está desestabilizando y te enteras **antes** de
que reviente. Si devolvieras la norma posterior verías `max_norm` clavado y no te enterarías de
nada.

### La trampa del generador

`parameters` puede ser un generador —`model.parameters()` lo es— y **un generador se agota al
recorrerlo**. Si lo recorres una vez para calcular la norma y otra para multiplicar, la segunda
vez está vacío: la función devuelve la norma correcta y no recorta nada. Sin error, sin aviso.

Por eso se materializa la lista de gradientes **una** vez, al principio, y se trabaja siempre
sobre ella.

---

## Ejercicio 4: qué parámetros decaen (`build_param_groups`)

Cinco líneas, y la regla es sorprendentemente simple:

```
   parámetros de 2 dimensiones o más   ->  CON weight decay
   parámetros de 1 dimensión           ->  SIN weight decay
```

O sea: las matrices decaen, y los sesgos y las escalas de normalización no. `param.dim()` da el
número de dimensiones: una matriz de pesos tiene 2, la escala de un RMSNorm tiene 1.

**Por qué.** El weight decay empuja los pesos hacia cero. En una matriz de proyección tiene
sentido: penalizar magnitudes grandes reduce el sobreajuste. En la escala de un RMSNorm no tiene
ninguno: ese parámetro arranca en 1 —lo escribiste tú en el módulo 07— y su trabajo es reescalar
la salida de la capa. Empujarlo hacia cero es empujar la salida hacia cero, que es exactamente lo
contrario de lo que hace falta.

Así queda repartido nuestro modelo:

| grupo | weight_decay | tensores | parámetros | qué hay dentro |
|---|---|---|---|---|
| con decay | 0,1 | 43 | 8.929.280 | matrices: embeddings y proyecciones |
| sin decay | 0,0 | 13 | 4.160 | escalas de RMSNorm |

Esos 13 tensores y 4.160 parámetros son exactamente las trece capas de normalización que contaste
en el módulo 10. Aquí es donde importaba tenerlas identificadas.

**Aplicar decay a todo es un error frecuente, no da ningún error visible, y degrada el
resultado.** Sólo se detecta comparando dos entrenamientos completos, que es caro. Por eso merece
la pena tenerlo bien de entrada.

### Dos detalles

**Saltar `requires_grad=False`.** Esos parámetros no se van a actualizar; meterlos en el
optimizador sólo gasta memoria de estado, dos tensores por parámetro. Ahora es una optimización
menor; en el módulo 16, con LoRA, pasa a ser esencial porque casi todo el modelo está congelado.

**Los pesos atados.** `model.parameters()` ya deduplica por identidad, así que el embedding atado
del módulo 10 aparece **una** sola vez y va al grupo con decay, porque tiene 2 dimensiones. No
hay que hacer nada especial.

**Y el formato:** una lista de diccionarios, cada uno con al menos la clave `"params"`. Cualquier
clave adicional (`lr`, `weight_decay`...) sobreescribe el valor por defecto del optimizador **sólo
para ese grupo**. Es el mecanismo estándar de PyTorch, y es exactamente lo que lee tu
`AdamWScratch.step` cuando hace `for group in self.param_groups`. Ahí se cierra el círculo entre
el ejercicio 1 y éste.

---

## La pieza que no escribes: la precisión mixta

No hay ejercicio para esto porque `GradScaler` viene hecho en PyTorch, pero tiene un detalle que
te va a morder en el módulo 13 si no lo sabes ahora.

En la RTX 2060 (Turing, sin bf16 en hardware, módulo 01) el entrenamiento va en **fp16**, cuyo
rango se acaba por abajo en $6\times10^{-5}$. Los gradientes de las capas profundas son menores
que eso y se convierten en cero: el modelo deja de aprender por abajo sin dar ningún error.

`GradScaler` lo resuelve multiplicando la pérdida por ~65.000 antes del backward, con lo que
todos los gradientes suben al rango representable, y dividiendo antes del paso del optimizador.
Si algún valor se desborda, descarta ese paso y baja el factor.

**Y aquí está el detalle silencioso:** con AMP hay que **desescalar los gradientes antes de
recortarlos**. Si no, su norma está multiplicada por 65.000, y tu `clip_grad_norm(params, 1.0)`
del ejercicio 3 estaría recortando a un umbral efectivo 65.000 veces más pequeño del que crees.
El entrenamiento se arrastra y nada lo indica.

```python
scaler.unscale_(optimizer)      # esto primero
clip_grad_norm(params, 1.0)     # y ahora sí
```

---

## Dónde está el debate

Adam **domina sin que nadie sepa bien por qué**. La justificación habitual —que aproxima
información de segundo orden— no resiste el análisis: $\sqrt{v}$ no es la diagonal del Hessiano
ni nada parecido. Hay trabajos que sugieren que su ventaja real está en la invariancia de escala,
o en cómo interactúa con la normalización. Sigue abierto.

Sobre el **warmup** pasa algo parecido: es imprescindible en la práctica y las explicaciones son
post-hoc. Hay resultados que sugieren que con pre-norm y una inicialización cuidadosa se puede
prescindir de él, lo que apunta a que compensa problemas de otras partes de la arquitectura — y
tú tienes pre-norm y una inicialización cuidadosa, así que la pregunta no es retórica.

Y sobre los **hiperparámetros** de nuestro config: `lr=1e-3`, `betas=(0.9, 0.95)`,
`weight_decay=0.1`, `grad_clip=1.0`, `warmup=500`. Son valores estándar heredados de GPT-2/GPT-3
y ajustados a ojo para esta escala. No son óptimos; son razonables. Un barrido de
hiperparámetros probablemente encontraría algo mejor, y costaría más cómputo que el propio
entrenamiento — que es, por cierto, la razón por la que casi nadie los ajusta y todo el mundo
copia los mismos números de un paper a otro.

---

**Para ampliar:** Loshchilov & Hutter 2019,
[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) · Kingma & Ba 2015,
[Adam](https://arxiv.org/abs/1412.6980) · Micikevicius et al. 2018,
[Mixed Precision Training](https://arxiv.org/abs/1710.03740). Términos sueltos, en
[GLOSARIO.md](../../GLOSARIO.md).
