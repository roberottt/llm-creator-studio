# 12 — Eficiencia y leyes de escala

Dos preguntas que parecen distintas y son la misma: **¿estoy aprovechando mi GPU?** y
**¿cómo debería gastar mi presupuesto de cómputo?**

## Parte 1: MFU, o cuánto de tu GPU estás usando

Ya viste en el módulo 01 cuánto cuesta un token: unos 65,4 millones de FLOPs para nuestro
modelo. Y mediste el pico de tu hardware. La MFU junta ambas cosas:

$$\text{MFU} = \frac{\text{tokens/s} \times C_{\text{token}}}{\text{FLOPS pico}}$$

Ejemplo con números de la RTX 2060. Si entrenando ves 3.000 tokens/s:

```
3.000 × 65,4·10⁶ = 1,96·10¹¹ FLOPS reales
pico de la 2060  = 5,16·10¹³ FLOPS
MFU = 1,96·10¹¹ / 5,16·10¹³ = 0,004 = 0,4%
```

Menos del 1%. Eso suena a desastre y hay que saber leerlo.

### Qué MFU es razonable

| situación | MFU típica |
|---|---|
| modelos grandes bien optimizados en A100/H100 | 0,4 – 0,5 |
| modelos medianos | 0,2 – 0,3 |
| **nuestro modelo de 9M** | 0,1 – 0,2 |
| algo va mal | < 0,05 |

**Nadie llega a 1.** El pico teórico solo se alcanza con matmuls enormes perfectamente
alineados y nada más de por medio.

Con un modelo pequeño la MFU baja es inevitable. Las matrices de 320×320 no dan para saturar
los tensor cores, y el tiempo se va en lanzar kernels y mover memoria. Es el mismo fenómeno
que mediste en la demo del módulo 01: matrices de 128 daban menos de 2 TFLOPS y las de 2048,
diez veces más.

**Lo importante de la MFU no es su valor absoluto, es que sea comparable.** Es independiente
del modelo y del hardware, así que puedes cambiar el batch size, activar `torch.compile` o
mover el dataloader a otro hilo, y ver si el número sube.

### Dónde se va el tiempo cuando la MFU es baja

Cuatro sospechosos, en orden de frecuencia:

1. **El dataloader.** Si preparar el siguiente batch tarda más que procesarlo, la GPU
   espera. Se detecta cronometrando `get_batch` por separado (lo hiciste en el módulo 04).
2. **El batch es pequeño.** Menos trabajo por lanzamiento de kernel. Subir `batch_size`
   hasta llenar la memoria suele ser lo primero que hay que probar.
3. **Sincronizaciones accidentales.** Cualquier `.item()`, `float(tensor)` o `print` de un
   tensor obliga a la CPU a esperar a la GPU. Dentro del bucle de entrenamiento, eso mata
   el rendimiento.
4. **Operaciones memory-bound.** Normalizaciones y activaciones no aparecen en el conteo de
   FLOPs pero sí consumen tiempo. En un modelo pequeño son una fracción importante.

## Parte 2: Chinchilla, o cómo repartir el presupuesto

Ahora la pregunta de diseño. Tienes un presupuesto fijo de cómputo. Puedes gastarlo en un
**modelo grande con pocos datos** o en un **modelo pequeño con muchos datos**. ¿Cuál da
menos pérdida?

Durante años se asumió que había que hacer los modelos más grandes. GPT-3 tenía 175.000
millones de parámetros entrenados con 300.000 millones de tokens.

En 2022, Hoffmann et al. midieron esto en serio: entrenaron más de 400 modelos de distintos
tamaños con distintas cantidades de datos y ajustaron una superficie. Su conclusión:

> **Parámetros y datos deben crecer proporcionalmente. Unos 20 tokens por parámetro.**

GPT-3 tenía **1,7 tokens por parámetro**, doce veces por debajo del óptimo. Estaba
enormemente infra-entrenado.

Para demostrarlo entrenaron **Chinchilla**: 70.000 millones de parámetros y 1,4 billones de
tokens, con el mismo presupuesto de cómputo que Gopher (280.000 millones de parámetros).
Chinchilla ganó en casi todos los benchmarks **con la cuarta parte de parámetros**.

### La aritmética

Partiendo de $C = 6ND$ (módulo 01) y de $D = 20N$:

$$C = 6N(20N) = 120N^2 \quad \Longrightarrow \quad N = \sqrt{\frac{C}{120}}, \qquad D = 20N$$

Compruébalo con el propio Chinchilla. Su presupuesto fue $5{,}76 \times 10^{23}$ FLOPs:

```
N = √(5,76·10²³ / 120) = 6,9·10¹⁰ = 69.000 millones de parámetros
```

El modelo real tenía 70.000 millones. La fórmula lo clava.

### Nuestro caso

```
parámetros no-embedding : 7,62 M
tokens                  : 500 M
tokens por parámetro    : 65
```

**Más de tres veces por encima del "óptimo" de Chinchilla.** Es deliberado, y por dos
razones.

**La primera: Chinchilla optimiza el cómputo de *entrenamiento*, no el de uso.** Si el
modelo se va a ejecutar muchas veces después, conviene uno más pequeño y más entrenado: el
entrenamiento se paga una vez y la inferencia, cada vez. Llama-3 lleva esto al extremo con
~1.800 tokens por parámetro, noventa veces por encima de Chinchilla, y no es un error: es
que su función objetivo es otra.

**La segunda: a esta escala entrenar de más es barato.** Horas, no meses. Y da un modelo
notablemente mejor. La optimalidad de Chinchilla importa cuando el cómputo es el recurso
escaso; aquí el recurso escaso es tu paciencia.

## KV cache: por qué generar es distinto de entrenar

Un apunte que prepara el módulo 14.

Al entrenar procesas los 512 tokens de golpe y aprovechas la paralelización. Al **generar**,
produces un token cada vez, y en cada paso el modelo recalcula las claves y valores de
*todos* los tokens anteriores, que no han cambiado.

Guardarlos convierte un coste cuadrático en lineal. El precio es memoria:

$$\text{memoria KV} = 2 \times n_{\text{layers}} \times T \times d_{\text{model}} \times \text{bytes}$$

Para nuestro modelo con 512 tokens en fp16: $2 \times 6 \times 512 \times 320 \times 2 =
3{,}9$ MB. Nada. Para un modelo de 70B con contexto de 100.000, serían decenas de gigabytes,
y por eso existen técnicas como *grouped-query attention*.

## Dónde está el debate

Las leyes de escala están **peor establecidas de lo que su nombre sugiere**.

Los coeficientes de Chinchilla se ajustaron a un rango concreto de escalas y a un dataset
concreto, y **extrapolar fuera de ahí no está justificado**. De hecho, en 2024 un grupo
reanalizó los datos originales y encontró que el ajuste tenía problemas metodológicos y que
los intervalos de confianza eran mucho más amplios de lo reportado. La conclusión cualitativa
—"hay que entrenar con más datos de los que se creía"— se sostiene; los números exactos, con
más cautela.

Además, las leyes de escala predicen **pérdida**, no capacidades. La relación entre bajar la
pérdida y "razonar mejor" no es directa ni está bien entendida, y es una de las discusiones
abiertas más importantes del campo.

Y hay algo que ninguna ley de escala captura: **la calidad de los datos**. El paper de
TinyStories muestra que un dataset pequeño y muy limpio permite a modelos diminutos generar
texto coherente, algo que no se consigue con la misma cantidad de texto de internet. Ningún
$N$ ni $D$ recoge eso.

---

**Para ampliar:** Hoffmann et al. 2022,
[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
(Chinchilla) · Kaplan et al. 2020,
[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) · Chowdhery et
al. 2022, [PaLM](https://arxiv.org/abs/2204.02311) (definición de MFU) · Besiroglu et al.
2024, [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102).
Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
