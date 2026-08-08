# 16 — Post-training: enseñarle a responder en vez de a continuar

## Por qué importa este módulo

**Porque tu modelo entrenado no responde preguntas: las continúa.**

Escríbele *"¿Cuál es la capital de Francia?"* y lo más probable es que siga con más preguntas.
No está roto: está haciendo exactamente lo que le enseñaste, que es continuar texto plausible. Un
documento que empieza con una pregunta suele seguir con más.

Convertir eso en algo que responde es una fase distinta con sus propios métodos, y es donde se
instala buena parte de lo que asocias a un asistente. En este módulo la haces: SFT de verdad
sobre tu modelo, y ves el antes y el después.

Y aprendes LoRA, que es la técnica que hace accesible el fine-tuning de modelos grandes:
entrenar el **0,7%** de los parámetros en vez del 100%.

### Qué sabrás al terminar

- La diferencia real entre pretraining y post-training, y qué añade cada uno
- Por qué un modelo necesita marcadores para saber cuándo le toca hablar y cuándo callarse
- Un off-by-one que decide si el modelo aprende a responder o a preguntar
- Cómo entrenar el 0,7% de un modelo y luego **fundir los cambios** sin dejar rastro
- Qué esperar de verdad al hacer SFT sobre un modelo de juguete, para no interpretarlo mal

### Qué vas a escribir

Cuatro ejercicios, en dos bloques independientes. Esta teoría los sigue en orden:

| Ejercicio | Qué hace | |
|---|---|---|
| 1. `build_chat_template` | Serializar la conversación con marcadores | SFT |
| 2. `mask_prompt_tokens` | Que solo se aprenda de la respuesta | SFT |
| 3. `LoRALinear` | La capa con adaptadores de rango bajo | LoRA |
| 4. `merge_lora_weights` | Fundir los adaptadores en la matriz base | LoRA |

Los dos primeros van juntos y son el SFT: uno da formato y el otro decide de qué se aprende. Los
dos últimos van juntos y son LoRA, y son independientes de los anteriores — puedes hacer SFT sin
LoRA y LoRA sin SFT; se combinan porque en la práctica es lo que se hace.

El ejercicio 2 son cuatro líneas y **el rango del bucle es todo el ejercicio**.

### Cuánto cuesta

3 horas. La demo hace SFT de verdad sobre tu modelo del módulo 13, así que verás el antes y el
después con tus propios pesos.

---

## Pretraining contra post-training

| | pretraining | post-training |
|---|---|---|
| **objetivo** | aprender lenguaje | aprender a comportarse |
| **datos** | todo el texto que puedas | ejemplos curados, pocos |
| **cantidad** | miles de millones de tokens | miles o decenas de miles |
| **coste** | meses y millones | horas |
| **qué cambia** | el conocimiento | el formato de la respuesta |

Lo importante, y es una idea que cuesta aceptar: **el post-entrenamiento no añade conocimiento**.
Lo que hace es sacar a la superficie un comportamiento que ya estaba latente. Un modelo que no
sabe algo tras el pretraining no lo va a aprender con mil ejemplos de conversación.

Es literalmente la misma pérdida de siempre —cross-entropy sobre el siguiente token, la del
módulo 05— y el mismo bucle del módulo 11. Lo único que cambia son los datos y una máscara.

---

## Ejercicio 1: el chat template (`build_chat_template`)

Un modelo preentrenado no tiene ni idea de dónde acaba una pregunta y empieza una respuesta. Se
le enseña con **marcadores**:

```
   entrenamiento: <|user|>Who is the king?<|end|><|assistant|>The king is Richard.<|end|>
   inferencia   : <|user|>Who is the king?<|end|><|assistant|>
                                                             ↑ abierta: el modelo continúa aquí
```

Fíjate en la diferencia entre las dos líneas, porque es la razón de que la función tenga un flag
`add_generation_prompt`: al **entrenar** le das la conversación completa, respuesta incluida; al
**generar** la dejas abierta justo después del marcador del asistente, y el modelo continúa desde
ahí. Es el mismo texto con distinto final.

Los marcadores no tienen nada de mágico: son texto que el modelo aprende a reconocer durante el
SFT. Aprende que después de `<|assistant|>` toca responder, y que `<|end|>` significa parar —
**sin eso, un modelo no sabe cuándo callarse** y genera hasta agotar el contexto.

Cada familia de modelos usa los suyos y son incompatibles entre sí. Usar el template equivocado
con un modelo degrada bastante su calidad, y es un error sorprendentemente frecuente.

---

## Ejercicio 2: enmascarar el prompt (`mask_prompt_tokens`)

Aquí está la parte sutil del módulo. No quieres que el modelo aprenda a **generar las preguntas
del usuario**: quieres que aprenda a **responderlas**.

La solución es poner `-100` en los targets de las posiciones del prompt.
`F.cross_entropy(..., ignore_index=-100)` las salta — y ése es el `ignore_index` que dejaste
puesto en el módulo 10 sin usarlo. Aquí es donde se cobra.

Con `input_ids = [10, 11, 12, 20, 21, 22]` y `prompt_len = 3`:

| posición | input | target | |
|---|---|---|---|
| 0 | 10 | −100 | prompt: ignorado |
| 1 | 11 | −100 | prompt: ignorado |
| 2 | 12 | **20** | **la transición: SÍ aprende** |
| 3 | 20 | 21 | respuesta |
| 4 | 21 | 22 | respuesta |
| 5 | 22 | −100 | no hay siguiente token |

**Fíjate en la posición 2: hay dos posiciones ignoradas al principio, no tres.** Es el último
token del prompt, pero como los targets van desplazados un token (módulo 04), su objetivo ya es
el primer token de la respuesta.

Y esa transición —*"se acabó la pregunta, me toca hablar"*— es lo más importante que el modelo
tiene que aprender en todo el SFT. Enmascararla sería quitarle justo la señal que necesita.

Ese off-by-one es el error típico del ejercicio y **no da ningún error visible**: simplemente
desperdicia la posición más informativa, y el modelo aprende peor sin que nada lo indique.

---

## Ejercicio 3: LoRA (`LoRALinear`)

Hacer SFT completo sobre un modelo de 70B requiere memoria para los pesos, los gradientes y los
estados de Adam: del orden de 12 bytes por parámetro, casi un terabyte. Es el mismo desglose que
hiciste en el módulo 10, escalado.

**LoRA** (Hu et al., 2021) parte de una observación: los cambios que hace el fine-tuning tienen
**rango bajo**. No hace falta poder modificar la matriz en cualquier dirección; basta con unas
pocas.

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

Y como el estado del optimizador solo existe para lo entrenable —los `requires_grad=False` que
salta `build_param_groups` en el módulo 11—, la memoria de Adam baja en la misma proporción. En
modelos grandes es la diferencia entre necesitar ocho GPUs o una.

### Congelar la base es el ejercicio

El paso que hay que entender del `__init__` es éste:

```python
self.base = base_layer
for p in self.base.parameters():
    p.requires_grad = False
```

Sin esas dos líneas tendrías una capa con adaptadores **y** la base entrenándose: ni ahorras
memoria ni tiene sentido. Todo LoRA está en congelar y adaptar por fuera.

### La inicialización, que no es simétrica

```
   A ~ normal (Kaiming)
   B = CEROS
```

Con $B = 0$, el producto $BA$ vale cero al empezar y **la capa es exactamente la original**. El
fine-tuning arranca sin perturbar nada, y la demo lo comprueba: al inicializar, la salida de la
capa LoRA es idéntica a la de la base.

Si inicializaras las dos al azar, el modelo empezaría degradado y tendría que gastar los primeros
pasos recuperándose antes de empezar a mejorar. Es una de esas decisiones que parecen un detalle
y no lo son — la misma idea que el `torch.ones` de RMSNorm en el módulo 07: al arrancar, la pieza
nueva no debe hacer nada.

---

## Ejercicio 4: fundir los pesos (`merge_lora_weights`)

Al terminar, los adaptadores se **absorben** en la matriz base:

$$W_{\text{nueva}} = W + \frac{\alpha}{r} BA$$

El modelo resultante es indistinguible de uno normal: mismo coste de inferencia, mismas formas, y
se puede servir sin ninguna dependencia de LoRA. Medido en la demo, la capa fundida da lo mismo
que la capa con adaptadores con un error de `1,31e-06`, que es redondeo de fp32.

Ésa es la ventaja de LoRA frente a otros métodos de fine-tuning eficiente: la adaptación es
**exactamente** una suma de matrices, así que se puede absorber sin aproximar nada. Otros métodos
añaden capas o cambian la topología, y entonces no hay forma de volver a un modelo estándar.

Es también lo que permite tener **varios adaptadores para un mismo modelo base** y cargar uno u
otro según la tarea, sin duplicar los pesos grandes.

---

## Qué vas a ver de verdad al hacer SFT

La demo hace SFT sobre el modelo de Shakespeare con 96 ejemplos en formato `Q: ... / A: ...` (usa
ese formato en vez de los marcadores `<|user|>` porque el tokenizador de caracteres solo conoce
los símbolos que aparecen en Shakespeare). La pérdida baja de 1,4568 a 0,0912, y esto es el antes
y el después:

```
   ANTES:    Q: Who is the king?
             A:
             I have the comptaint the headen shall do logger, To hear it...

   DESPUÉS:  Q: Who is the king?
             A:
             I say we must go.

             MARCIUS:
             A lord.
```

**Lo que hay que mirar no es si la respuesta es correcta.** Con 0,8M de parámetros y 96 ejemplos,
no lo va a ser.

Lo que hay que mirar es el **formato**: antes seguía escribiendo Shakespeare indefinidamente,
después responde algo corto y para. Eso es exactamente lo que el post-entrenamiento enseña, y es
la lección del módulo: **no añade conocimiento, saca a la superficie un comportamiento**.

Si esperabas un asistente, la distancia no es de entrenamiento: son tres o cuatro órdenes de
magnitud en parámetros y datos, más el escalón que viene ahora.

## Lo que NO vamos a hacer, y es importante

Después del SFT, los modelos comerciales pasan por **RLHF** o **DPO**: se recogen preferencias
humanas entre pares de respuestas y se ajusta el modelo hacia las preferidas.

Eso es lo que hace que un modelo sea *útil* en lugar de solo *obediente al formato*, y es también
donde se instala buena parte del comportamiento que asocias a un asistente.

No lo haremos aquí. Requiere datos de preferencias que no tenemos, y un modelo de 9M no tiene
capacidad para aprovecharlo. Pero merece la pena saber que existe ese escalón, porque explica
buena parte de la distancia entre lo que acabas de construir y lo que usas a diario.

## Dónde está el debate

**Por qué funciona LoRA no está claro.** La hipótesis del "rango bajo intrínseco" es razonable y
tiene evidencia, pero no está demostrada. Hay trabajos que muestran que LoRA rinde peor que el
fine-tuning completo en tareas que requieren aprender conocimiento nuevo, y comparable en las que
solo cambian el estilo — lo cual encaja con la hipótesis, pero es correlación.

**Cuánta capacidad añade realmente el post-entrenamiento** es una discusión activa. La hipótesis
del *superficial alignment* sostiene que casi todo el conocimiento está en el pretraining y el
post-entrenamiento solo selecciona el formato. Hay evidencia a favor —se consiguen resultados muy
decentes con mil ejemplos— y también en contra.

Y una honesta sobre este módulo: **con un modelo de 9M entrenado sobre TinyStories, el SFT no va
a producir un asistente**. Vas a ver que aprende el formato —responde tras el marcador, para al
final— y poco más. El ejercicio enseña el mecanismo, no produce un producto.

---

**Para ampliar:** Hu et al. 2021, [LoRA](https://arxiv.org/abs/2106.09685) · Ouyang et al. 2022,
[InstructGPT](https://arxiv.org/abs/2203.02155) (RLHF) · Zhou et al. 2023,
[LIMA](https://arxiv.org/abs/2305.11206) (la hipótesis del superficial alignment).
Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
