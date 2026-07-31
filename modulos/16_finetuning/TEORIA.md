# 16 — Post-training: SFT y LoRA

Tu modelo sabe continuar texto. Si le escribes *"¿Cuál es la capital de Francia?"*, es
bastante probable que responda con más preguntas:

```
¿Cuál es la capital de Francia? ¿Y la de Italia? ¿Cuántos habitantes tiene?
```

No está roto. Está haciendo **exactamente** lo que le enseñaste: continuar texto plausible.
Un documento que empieza con una pregunta suele seguir con más preguntas.

Convertir eso en algo que responda es el **post-entrenamiento**, y es una fase distinta con
sus propios métodos.

## Pretraining contra post-training

| | pretraining | post-training |
|---|---|---|
| **objetivo** | aprender lenguaje | aprender a comportarse |
| **datos** | todo el texto que puedas | ejemplos curados, pocos |
| **cantidad** | miles de millones de tokens | miles o decenas de miles |
| **coste** | meses y millones | horas |
| **qué cambia** | el conocimiento | el formato de la respuesta |

Lo importante, y es una idea que cuesta aceptar: **el post-entrenamiento no añade
conocimiento**. Lo que hace es sacar a la superficie un comportamiento que ya estaba
latente. Un modelo que no sabe algo tras el pretraining no lo va a aprender con mil ejemplos
de conversación.

## SFT: enseñar el formato

*Supervised Fine-Tuning* es seguir entrenando con la misma pérdida de siempre, pero sobre
pares de instrucción y respuesta.

Dos piezas lo hacen funcionar.

### El chat template

Un modelo preentrenado no tiene ni idea de dónde acaba una pregunta y empieza una respuesta.
Se le enseña con **marcadores**:

```
<|user|>¿Cuál es la capital de Francia?<|end|><|assistant|>París.<|end|>
```

Los marcadores no tienen nada de mágico: son texto que el modelo aprende a reconocer durante
el SFT. Aprende que después de `<|assistant|>` toca responder, y que `<|end|>` significa
parar — **sin eso, un modelo no sabe cuándo callarse**.

Cada familia de modelos usa los suyos y son incompatibles entre sí. Usar el template
equivocado con un modelo degrada bastante su calidad, y es un error sorprendentemente
frecuente.

### Enmascarar el prompt

Aquí está la parte sutil. No quieres que el modelo aprenda a **generar las preguntas del
usuario**: quieres que aprenda a **responderlas**.

La solución es poner `-100` en los targets de las posiciones del prompt.
`F.cross_entropy(..., ignore_index=-100)` las salta.

```
input_ids = [10, 11, 12, 20, 21, 22]      con prompt_len = 3
targets   = [-100, -100, 20, 21, 22, -100]
```

**Fíjate en que hay dos posiciones ignoradas, no tres.** Los targets van desplazados un
token, así que en la posición 2 —el último token del prompt— el objetivo ya es el primer
token de la respuesta, y ese sí interesa.

Ese off-by-one es el error típico, y no da ningún error visible: solo desperdicia (o
aprovecha de más) una posición.

## LoRA: entrenar el 1% del modelo

Hacer SFT completo sobre un modelo de 70B requiere memoria para los pesos, los gradientes y
los estados de Adam: del orden de 12 bytes por parámetro, casi un terabyte.

**LoRA** (Hu et al., 2021) resuelve esto con una observación: los cambios que hace el
fine-tuning tienen **rango bajo**. No hace falta poder modificar la matriz en cualquier
dirección; basta con unas pocas.

Así que se congela `W` y se le suma el producto de dos matrices flacas:

$$W' = W + \frac{\alpha}{r} BA$$

con $A$ de $r \times d_{in}$ y $B$ de $d_{out} \times r$, y $r$ pequeño (4, 8, 16).

### La aritmética que lo justifica

Con $d_{in} = d_{out} = 320$ y $r = 8$:

```
W entera:  320 × 320       = 102.400 parámetros
A y B:     8×320 + 320×8   =   5.120 parámetros    (el 5%)
```

Aplicado a nuestro modelo de 9M, adaptando solo `q_proj` y `v_proj`:

| r | entrenables | % del modelo |
|---|---|---|
| 4 | 30.720 | **0,34%** |
| 8 | 61.440 | **0,68%** |
| 16 | 122.880 | **1,36%** |

Y como el estado del optimizador solo existe para lo entrenable, la memoria de Adam baja en
la misma proporción. En modelos grandes es la diferencia entre necesitar ocho GPUs o una.

### La inicialización, que no es simétrica

```
A ~ normal (Kaiming)
B = CEROS
```

Con $B = 0$, el producto $BA$ vale cero al empezar y **la capa es exactamente la original**.
El fine-tuning arranca sin perturbar nada.

Si inicializaras las dos al azar, el modelo empezaría degradado y tendría que recuperarse
antes de empezar a mejorar. Es una de esas decisiones que parecen un detalle y no lo son.

### Fundir los pesos

Al terminar, los adaptadores se **absorben** en la matriz base:

$$W_{\text{nueva}} = W + \frac{\alpha}{r} BA$$

El modelo resultante es indistinguible de uno normal: mismo coste de inferencia, mismas
formas, y se puede servir sin ninguna dependencia de LoRA.

Esa es una ventaja de LoRA frente a otros métodos de fine-tuning eficiente: la adaptación es
**exactamente** una suma de matrices, así que se puede absorber sin aproximar nada.

## Lo que NO vamos a hacer, y es importante

Después del SFT, los modelos comerciales pasan por **RLHF** o **DPO**: se recogen
preferencias humanas entre pares de respuestas y se ajusta el modelo hacia las preferidas.

Eso es lo que hace que un modelo sea *útil* en lugar de solo *obediente al formato*, y es
también donde se instala buena parte del comportamiento que asocias a un asistente.

No lo haremos aquí. Requiere datos de preferencias que no tenemos, y un modelo de 9M no
tiene capacidad para aprovecharlo. Merece la pena saber que existe ese escalón.

## Dónde está el debate

**Por qué funciona LoRA no está claro.** La hipótesis del "rango bajo intrínseco" es
razonable y tiene evidencia, pero no está demostrada. Hay trabajos que muestran que LoRA
rinde peor que el fine-tuning completo en tareas que requieren aprender conocimiento nuevo,
y comparable en las que solo cambian el estilo — lo cual encaja con la hipótesis, pero es
correlación.

**Cuánta capacidad añade realmente el post-entrenamiento** es una discusión activa. La
hipótesis del *superficial alignment* sostiene que casi todo el conocimiento está en el
pretraining y el post-entrenamiento solo selecciona el formato. Hay evidencia a favor —se
consiguen resultados muy decentes con mil ejemplos— y también en contra.

Y una honesta sobre este módulo: **con un modelo de 9M entrenado sobre TinyStories, el SFT no
va a producir un asistente**. Vas a ver que aprende el formato —responde tras el marcador,
para al final— y poco más. El ejercicio enseña el mecanismo, no produce un producto.

---

**Para ampliar:** Hu et al. 2021, [LoRA](https://arxiv.org/abs/2106.09685) · Ouyang et al.
2022, [InstructGPT](https://arxiv.org/abs/2203.02155) (RLHF) · Zhou et al. 2023,
[LIMA](https://arxiv.org/abs/2305.11206) (la hipótesis del superficial alignment).
Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
