# 08 — FFN, GELU y SwiGLU

## Por qué importa este módulo

**Porque dos tercios de tu modelo están aquí, y casi nadie lo sabe.**

Cuando alguien dice que un modelo tiene N parámetros, la mayoría no están en la atención:
están en esta parte, que recibe muchísima menos atención en las explicaciones. En nuestro
modelo son 5,16 millones de 8,93.

Y hay una razón más profunda. La atención es una media ponderada, o sea una **operación
lineal**, y apilar operaciones lineales no sirve de nada: cien capas equivalen a una. Lo que
impide que el Transformer entero se derrumbe a una sola multiplicación de matrices es
precisamente este módulo. La demo lo mide: cinco capas lineales sin activación dan
exactamente el mismo resultado que una sola matriz.

### Qué sabrás al terminar

- Por qué sin una no-linealidad la profundidad de una red es una ilusión
- Qué le pasa a una neurona con ReLU cuando se va a la zona negativa (se muere, literalmente)
- Qué es SwiGLU y **de dónde sale el 896** del config del modelo final
- Un caso donde el propio autor del paper escribe que no sabe por qué funciona

### Cuánto cuesta

1,5 horas. El segundo ejercicio es aritmética pura y produce un número del config.

---

## El problema: la atención sola no basta

Fíjate en lo que hace la atención: mezcla vectores con pesos. Una media ponderada. Y una
media ponderada es una **operación lineal**.

Eso es un problema serio, y se ve con números. Imagina que apilas dos capas lineales sin
nada en medio:

```
capa 1:  y = W₁ · x
capa 2:  z = W₂ · y = W₂ · (W₁ · x) = (W₂ · W₁) · x
```

$W_2 W_1$ es **una sola matriz**. Cien capas lineales apiladas equivalen exactamente a una
capa lineal. Toda la profundidad se derrumba.

Para que apilar sirva de algo hace falta algo que no sea lineal entre capa y capa. Ese es el
trabajo del FFN.

## La forma clásica: expandir, doblar, contraer

$$\text{FFN}(x) = W_2 \cdot \text{activación}(W_1 x)$$

Con $W_1$ de $d \to 4d$ y $W_2$ de $4d \to d$. Se expande a 4 veces el tamaño, se aplica la
no-linealidad, y se vuelve a comprimir.

**¿Por qué 4x?** Honestamente: porque lo puso el paper de 2017 y funcionó. No hay una
derivación. Se han probado otros factores y 4 sigue siendo un punto razonable, pero es
convención, no teorema. Lo que sí tiene sentido es *expandir*: la no-linealidad tiene más
espacio donde operar, y hay una interpretación —discutida— de que el FFN funciona como una
memoria de tipo clave-valor, donde cada una de las $4d$ neuronas intermedias reconoce un
patrón concreto.

Una diferencia importante con la atención: **el FFN procesa cada token por separado**. No
mezcla información entre posiciones. La atención mueve información entre tokens; el FFN la
procesa. Alternan.

## ReLU, y por qué no basta

La no-linealidad más simple es ReLU: $\max(0, x)$. Funciona, pero tiene un defecto. Su
derivada es exactamente **0** para toda entrada negativa. Si una neurona acaba dando
siempre valores negativos, deja de recibir gradiente para siempre. Está muerta y no hay
forma de recuperarla.

## GELU: un corte suave

$$\text{GELU}(x) = x \cdot \Phi(x)$$

donde $\Phi(x)$ es la probabilidad de que una normal estándar salga menor que $x$.

La intuición: en vez de decidir con un corte duro si dejar pasar $x$, lo multiplica por la
probabilidad de que $x$ "destaque". Con números:

```
x = -3   ->  Φ(-3) = 0.001   ->  GELU = -0.003    casi anulado
x = -1   ->  Φ(-1) = 0.159   ->  GELU = -0.159    parcialmente
x =  0   ->  Φ(0)  = 0.5     ->  GELU =  0
x =  1   ->  Φ(1)  = 0.841   ->  GELU =  0.841    casi entero
x =  3   ->  Φ(3)  = 0.999   ->  GELU =  2.996    entero
```

La transición es suave, así que la derivada nunca es exactamente cero: una neurona en la
zona negativa puede recuperarse.

En la práctica se usa una aproximación con tanh, porque `erf` era lento en las GPU de 2016:

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{2/\pi}\,(x + 0.044715x^3)\right]\right)$$

Hoy la diferencia de velocidad es irrelevante, pero GPT-2 se entrenó con la aproximación y
por compatibilidad se sigue usando. Es lo que hace `F.gelu(x, approximate="tanh")`.

## SwiGLU: añadir una puerta

Aquí viene el cambio que usa nuestro modelo, y todos los modernos.

La idea de las variantes **GLU** (*Gated Linear Unit*) es tener **dos** proyecciones en vez
de una. Una de ellas actúa como **puerta**: multiplica a la otra elemento a elemento y
decide cuánta señal pasa por cada dimensión.

$$\text{SwiGLU}(x) = \big(\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}}\big) W_{\text{down}}$$

con $\text{Swish}(z) = z \cdot \sigma(z)$, que es prácticamente GELU con otra fórmula.

Lo interesante es que ese filtrado **depende de la entrada**. Una activación normal aplica
la misma función a todo; una puerta decide, para cada dimensión y cada token, cuánto deja
pasar.

### El factor 2/3, con la aritmética

SwiGLU tiene **tres** matrices en lugar de dos. Con el mismo $d_{ff}$ eso sería un 50% más
de parámetros. Para gastar lo mismo se reduce $d_{ff}$ a dos tercios:

```
FFN clásico:  2 matrices × d × 4d           = 8d²
SwiGLU:       3 matrices × d × (2/3 · 4d)   = 3 · d · (8/3)d = 8d²   ✓
```

Con nuestro $d_{\text{model}} = 320$:

```
(2/3) × 4 × 320 = 853,33
```

Y después se redondea **hacia arriba al siguiente múltiplo de 64**: $853{,}33 \to 896$. Eso
da el `d_ff: 896` del config.

El redondeo no es cosmético. Las dimensiones alineadas a potencias de dos permiten a los
tensor cores usar sus rutas rápidas; una matriz de 853 columnas es notablemente más lenta
que una de 896, con más parámetros y todo.

## Dónde están los parámetros

Con el config final, por capa:

| componente | parámetros | % |
|---|---|---|
| atención ($4d^2$) | 409.600 | 32% |
| SwiGLU ($3 \cdot d \cdot d_{ff}$) | 860.160 | 68% |

**Dos tercios del modelo son FFN.** Cuando leas que un modelo tiene N parámetros, la
mayoría están aquí, no en la atención. Es también donde la investigación en
interpretabilidad ha encontrado el almacenamiento de hechos concretos: hay trabajos que
localizan y editan afirmaciones específicas modificando pesos del FFN de capas concretas.

## Dónde está el debate

Este módulo es probablemente donde el "no sabemos por qué" es más explícito, y viene del
propio autor.

Shazeer (2020) probó sistemáticamente todas las variantes GLU y SwiGLU salió la mejor de
forma consistente. Su conclusión, citada literalmente del paper:

> *"We offer no explanation as to why these architectures seem to work; we attribute their
> success, as all else, to divine benevolence."*

No es una boutade: es honestidad sobre el estado del asunto. SwiGLU se usa hoy en Llama,
Mistral, PaLM y casi todo lo demás, y la justificación es que funciona mejor en los
benchmarks. No hay una teoría.

Lo mismo pasa con el 4x y con la interpretación del FFN como memoria clave-valor: son
observaciones e hipótesis razonables, no resultados establecidos. Conviene tenerlo presente
cuando leas explicaciones que suenan muy seguras de sí mismas.

---

**Para ampliar:** Hendrycks & Gimpel 2016,
[Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) · Shazeer 2020,
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (el paper de la cita) ·
Geva et al. 2021, [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913).
Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
