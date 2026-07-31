# 07 — Normalización y conexiones residuales

## Por qué importa este módulo

**Porque sin esto, una red profunda no entrena. Punto.**

Dos piezas que no calculan nada interesante y que son la diferencia entre un modelo que
aprende y uno que devuelve `NaN` a los tres pasos. Son la fontanería del Transformer: nadie
las menciona en los titulares y sin ellas no hay nada.

El problema que resuelven es concreto y lo vas a ver medido: los números que atraviesan una
red profunda tienden a explotar o a desvanecerse, y con 40 capas el gradiente llega a cero
**exacto**. La demo lo mide.

Además, aquí está una de las decisiones de diseño donde más se aprende comparando: dónde
poner la normalización cambia si tu red necesita warmup o no.

### Qué sabrás al terminar

- Por qué los números se descontrolan al apilar capas, con la cuenta que lo explica
- Qué hace exactamente LayerNorm, y **qué le sobra** (eso es RMSNorm)
- Por qué `x + f(x)` es una de las ideas más importantes del deep learning
- Pre-norm contra post-norm, medido: cuánto gradiente llega a la primera capa en cada caso

### Cuánto cuesta

1,5 horas. Tres ejercicios cortos, y el tercero es literalmente una línea.

---

## El problema: los números se descontrolan

Una red profunda es una composición de funciones. Cada capa multiplica por una matriz, y
esas multiplicaciones se acumulan.

Imagina que cada capa multiplica la magnitud de sus entradas por 1,2. Parece inofensivo:

```
capa 1:  ×1,2  ->  1,2
capa 2:  ×1,2  ->  1,44
capa 3:  ×1,2  ->  1,73
...
capa 40: ×1,2  ->  1470
```

Y si el factor fuera 0,8 en lugar de 1,2, después de 40 capas quedaría 0,00013. En un caso
los números explotan, en el otro se desvanecen. **Y lo mismo le pasa al gradiente hacia
atrás**, que es lo que de verdad hace daño: si el gradiente se desvanece, las capas de abajo
no reciben señal y no aprenden nada.

Con fp16, que solo llega hasta 65504 por arriba y hasta $6\times10^{-5}$ por abajo, esto
deja de ser una molestia y se convierte en `inf` y en ceros.

## La solución 1: normalizar

La idea es brutal en su simplicidad: **después de cada bloque, vuelve a poner los números en
una escala conocida.** No importa lo que haya hecho la capa; al salir, renormalizas.

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

Pero forzar siempre media 0 y varianza 1 le quita libertad al modelo, así que se le devuelve
con dos parámetros aprendidos, $\gamma$ (escala) y $\beta$ (desplazamiento):

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

El $\epsilon$ (típicamente $10^{-5}$) evita dividir por cero cuando todas las componentes
son iguales.

**Importante: la media y la varianza se calculan sobre las dimensiones de cada token, por
separado.** Nada que ver con BatchNorm, que normaliza a lo largo del batch. LayerNorm trata
cada token independientemente, y por eso funciona igual con batch de 1 que de 1000 y no
necesita guardar estadísticas para la inferencia.

### RMSNorm: quitar la mitad

En 2019, Zhang y Sennrich probaron algo: ¿y si no restamos la media?

$$y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma$$

Solo se reescala por la raíz del cuadrado medio (*root mean square*). Sin restar media, sin
$\beta$. Con el mismo vector:

```
RMS = √((4+64+16+36)/4) = √30 = 5.477
x_norm = [0.365, 1.461, 0.730, 1.096]
```

Resultado: **entre un 7% y un 64% más rápido según el caso, y sin pérdida de calidad
medible**. Se ahorra una pasada por los datos y un tensor intermedio. Por eso lo usan Llama,
Mistral y prácticamente todo lo moderno, y por eso es lo que usa nuestro modelo.

Un detalle de implementación que importa: **el cálculo se hace en float32 aunque la entrada
venga en float16**. Elevar al cuadrado activaciones grandes puede desbordar el rango de fp16
y dar `inf`.

## La solución 2: conexiones residuales

La segunda pieza, y la más importante de las dos.

En vez de que cada bloque *sustituya* la representación, se le pide que la *modifique*:

$$x_{\text{salida}} = x + f(x)$$

El bloque calcula una corrección, no un reemplazo. A esa suma acumulada que atraviesa toda
la red se le llama **corriente residual** (*residual stream*).

**Por qué esto lo cambia todo:** derivando esa expresión respecto a $x$,

$$\frac{\partial x_{\text{salida}}}{\partial x} = 1 + \frac{\partial f(x)}{\partial x}$$

Ese **1** es una autopista. Aunque $\partial f/\partial x$ sea diminuto, el gradiente que
llega a las capas de abajo nunca baja de 1 por ese camino. Sin residuales, los factores se
multiplican y se desvanecen; con residuales, hay siempre una ruta directa.

## Pre-norm contra post-norm

Ahora la pregunta que decide si tu red entrena: **¿dónde va la normalización?**

```
post-norm (el paper de 2017):   x = norm(x + f(x))
pre-norm  (todo lo moderno):    x = x + f(norm(x))
```

Parece cosmético. No lo es.

En **post-norm**, la normalización está *encima* del camino residual. El gradiente la
atraviesa en cada capa y se va reescalando: la autopista tiene un peaje en cada salida. Con
6 capas se nota poco; con 40, hace falta un warmup cuidadoso del learning rate para que el
entrenamiento no explote en los primeros pasos.

En **pre-norm**, la normalización está *dentro* de la rama. El camino $x \to x$ queda
completamente libre y el gradiente llega intacto hasta la primera capa. Se puede entrenar
sin warmup y con learning rates más altos.

El precio de pre-norm: la corriente residual crece con la profundidad, porque cada capa le
suma su contribución sin que nadie la vuelva a normalizar. Por eso los modelos pre-norm
llevan **siempre** una normalización final antes de la capa de salida. Si se te olvida, los
logits salen con una escala arbitraria.

La demo del módulo mide esto empíricamente: entrena la misma red con las dos variantes y
compara la norma del gradiente que llega a la primera capa.

## Dónde está el debate

Aquí hay más de lo que parece.

**Por qué funciona la normalización sigue sin estar claro.** La explicación original de
BatchNorm (2015) fue el *internal covariate shift*: que normalizar estabiliza la
distribución de las entradas de cada capa. Santurkar et al. (2018) lo pusieron a prueba
inyectando ruido *después* de normalizar —destruyendo deliberadamente esa estabilidad— y la
red seguía entrenando igual de bien. La explicación original está hoy largamente
descartada, y la sustituta —que suaviza el paisaje de la función de pérdida— es más una
observación empírica que una teoría.

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
