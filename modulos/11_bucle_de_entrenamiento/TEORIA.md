# 11 — El bucle de entrenamiento

## Por qué importa este módulo

**Porque tener un modelo no es tenerlo entrenado.**

Ya sabes construir un GPT que produce logits y medir cuánto se equivoca. Falta la parte que
convierte eso en aprendizaje: cómo se mueven 8,9 millones de parámetros para que la pérdida
baje.

El bucle en sí lo escribiste en el módulo 02 con tu motor de autodiff: predecir, medir,
gradientes, mover, repetir. Lo que se añade aquí son cuatro piezas que hacen que ese bucle
funcione **a escala** en vez de divergir a los cincuenta pasos.

Cada una resuelve un problema concreto que verías si no estuviera. Y las cuatro son las que
te van a permitir depurar un entrenamiento que va mal en vez de cambiar números al azar.

### Qué sabrás al terminar

- Por qué un solo learning rate vale para toda la red (y qué hace Adam para conseguirlo)
- Qué es el warmup y por qué sin él el modelo a veces no se recupera nunca
- Cómo evitar que **un solo batch raro** destruya horas de entrenamiento
- Qué parámetros NO deben decaer, y por qué aplicárselo a todos es un error silencioso
- Un detalle de AMP que se olvida siempre y hace que el entrenamiento se arrastre

### Cuánto cuesta

4 horas. El primer ejercicio (AdamW desde cero) es el más largo del curso; los otros tres
son cortos.

---

## El bucle, en cuatro líneas

```
para cada paso:
    predecir y medir la pérdida       (forward)
    calcular los gradientes           (backward)
    mover los parámetros              (paso del optimizador)
    poner los gradientes a cero
```

Eso es todo. Ya lo escribiste en el módulo 02 con tu motor de autodiff. El resto del
módulo son cuatro piezas que hacen que ese bucle funcione a escala.

## Pieza 1: el optimizador

La versión más simple de "mover los parámetros" es el descenso de gradiente:

```
p ← p − lr · gradiente
```

Funciona, y tiene un problema serio. Piensa en dos parámetros de tu modelo: uno del
embedding de la palabra `the`, que aparece en casi todas las frases, y otro del embedding
de una palabra rara. El primero recibe gradientes grandes constantemente; el segundo, casi
nunca. Con un único `lr` para ambos, o el primero da saltos absurdos o el segundo no se
mueve nunca.

**Adam** resuelve esto con dos ideas.

**Momento.** En vez de moverse según el gradiente de este paso, se usa una media móvil de
los recientes:

```
m = 0,9·m + 0,1·gradiente
```

Como cada batch es una muestra distinta, sus gradientes son ruidosos. Promediar cancela el
ruido y deja la dirección consistente.

**Escalado por dimensión.** Se lleva también una media móvil del gradiente **al cuadrado**,
y se divide por su raíz:

```
v = 0,95·v + 0,05·gradiente²
paso = m / √v
```

Un parámetro con gradientes consistentemente grandes tiene $v$ grande y se mueve poco. Uno
que casi nunca recibe señal tiene $v$ pequeño y se mueve mucho cuando la recibe. **Cada
parámetro acaba con su propio learning rate efectivo**, y por eso un único `lr` global
funciona.

### La corrección de sesgo

$m$ y $v$ empiezan en cero, así que los primeros pasos subestiman las magnitudes reales.
Con $\beta_2 = 0{,}95$, tras un paso $v$ vale solo el 5% de $g^2$: dividir por su raíz daría
un paso 4,5 veces mayor de lo debido.

La corrección lo arregla exactamente:

$$\hat{m} = \frac{m}{1-\beta_1^t}, \qquad \hat{v} = \frac{v}{1-\beta_2^t}$$

En el paso 1 con $\beta_2 = 0{,}95$: $1 - 0{,}95 = 0{,}05$, y dividir por 0,05 multiplica
por 20, que es justo el factor que faltaba. Según avanza $t$, $\beta^t \to 0$ y la corrección
se desvanece sola.

Sin ella, los primeros pasos dan saltos enormes y el entrenamiento puede diverger antes de
empezar.

### La W de AdamW

*Weight decay* es empujar los pesos hacia cero para que no crezcan sin control. Hay dos
formas de hacerlo, y la diferencia importa:

```
Adam + L2:   g ← g + λ·p       luego Adam procesa g
AdamW:       p ← p − lr·λ·p    directamente, aparte de Adam
```

En la primera, el decaimiento pasa por la división por $\sqrt{v}$, así que su efecto real
depende de la magnitud de los gradientes de ese parámetro. Un peso con gradientes grandes
apenas decae; uno con gradientes pequeños decae muchísimo. Nadie quiere eso.

Loshchilov y Hutter (2019) lo desacoplaron y funcionó mejor de forma consistente. De ahí la
W.

## Pieza 2: el planificador del learning rate

El `lr` no es constante durante el entrenamiento. Tiene dos tramos.

**Warmup: subir despacio al principio.** En los primeros pasos, los momentos de Adam están
casi vacíos y sus estimaciones son ruidosas; además los pesos están recién inicializados y
los gradientes son grandes. Arrancar a `lr` completo suele producir un pico de pérdida del
que a veces el modelo no se recupera. Se sube linealmente de 0 a `lr` durante 500 pasos.

**Coseno: bajar al final.** Al principio interesa moverse rápido; al final, afinar. El
coseno baja despacio, luego deprisa, luego despacio otra vez:

$$\text{lr}(t) = \text{lr}_{\min} + (\text{lr} - \text{lr}_{\min}) \cdot \frac{1 + \cos(\pi \cdot \text{progreso})}{2}$$

No se decae hasta cero, sino hasta el 10% del `lr` inicial: por debajo de cierto punto el
modelo deja de aprender del todo y se desperdicia cómputo.

## Pieza 3: el recorte de gradientes

Ocasionalmente un batch produce gradientes enormes — una secuencia rara, un token muy poco
frecuente. Sin protección, ese único batch puede dar un salto que destruya horas de
entrenamiento.

La solución: calcular la norma **global** de todos los gradientes juntos, como si fueran un
solo vector, y si supera un umbral, multiplicarlos todos por el mismo factor.

```
norma = √(Σ ‖g_i‖²)
si norma > max_norm:  todos los g ×= max_norm / norma
```

**Global, no por tensor.** Recortar cada tensor por separado cambiaría la *dirección* del
gradiente conjunto, que es justo lo que no quieres: el gradiente apunta a dónde ir, y solo
estás limitando cuánto avanzas. Con la norma global la dirección se conserva exactamente.

## Pieza 4: qué parámetros decaen y cuáles no

Weight decay **solo en las matrices** (parámetros de 2 dimensiones o más). Sesgos y escalas
de normalización, no.

Piénsalo: la escala de un RMSNorm arranca en 1 y su trabajo es reescalar la salida.
Empujarla hacia cero es empujar la salida de la capa hacia cero, que es exactamente lo
contrario de lo que hace falta.

Aplicar decay a todo es un error frecuente, **no da ningún error visible** y degrada el
resultado. Solo se detecta comparando dos entrenamientos completos.

## Y la precisión mixta

En la RTX 2060 (Turing, sin bf16) el entrenamiento va en fp16, cuyo rango se acaba por abajo
en $6\times10^{-5}$. Los gradientes de las capas profundas son menores que eso y se
convierten en cero.

`GradScaler` lo resuelve: multiplica la pérdida por ~65.000 antes del backward, con lo que
todos los gradientes suben al rango representable, y divide antes del paso del optimizador.
Si algún valor se desborda, descarta ese paso y baja el factor.

**Hay un detalle que se olvida y es silencioso:** con AMP hay que **desescalar los
gradientes antes de recortarlos**. Si no, su norma está multiplicada por 65.000 y estarías
recortando a un umbral 65.000 veces más pequeño del que crees. El entrenamiento se arrastra
sin que nada lo indique.

```python
scaler.unscale_(optimizer)      # esto primero
clip_grad_norm(params, 1.0)     # y ahora sí
```

## Dónde está el debate

Adam **domina sin que nadie sepa bien por qué**. La justificación habitual —que aproxima
información de segundo orden— no resiste el análisis: $\sqrt{v}$ no es la diagonal del
Hessiano ni nada parecido. Hay trabajos que sugieren que su ventaja real está en la
invariancia de escala, o en cómo interactúa con la normalización. Sigue abierto.

Sobre el **warmup** pasa algo parecido: es imprescindible en la práctica y las explicaciones
son post-hoc. Hay resultados que sugieren que con pre-norm y una inicialización cuidadosa se
puede prescindir de él, lo que apunta a que compensa problemas de otras partes de la
arquitectura.

Y sobre los **hiperparámetros** de nuestro config: `lr=1e-3`, `betas=(0.9, 0.95)`,
`weight_decay=0.1`, `warmup=500`. Son valores estándar heredados de GPT-2/GPT-3 y ajustados
a ojo para esta escala. No son óptimos; son razonables. Un barrido de hiperparámetros
probablemente encontraría algo mejor, y costaría más cómputo que el propio entrenamiento.

---

**Para ampliar:** Loshchilov & Hutter 2019,
[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) · Kingma & Ba
2015, [Adam](https://arxiv.org/abs/1412.6980) · Micikevicius et al. 2018,
[Mixed Precision Training](https://arxiv.org/abs/1710.03740). Términos sueltos, en
[GLOSARIO.md](../../GLOSARIO.md).
