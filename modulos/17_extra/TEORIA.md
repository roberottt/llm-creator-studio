# 17 — Extras y límites honestos

## Por qué importa este módulo

**Porque conviene saber dónde estás.**

Dos cosas. La primera es práctica: cómo hacer el modelo cuatro veces más pequeño para poder
servirlo, con una técnica que se usa en producción en todas partes.

La segunda es una conversación franca. Has construido un modelo de 8,9 millones de
parámetros. Un modelo frontier tiene del orden de un billón. La distancia no es sólo de
tamaño, y merece la pena entender las **cinco** cosas que la componen, porque cuatro de ellas
se mencionan mucho menos que la primera.

Y también la otra cara: qué has conseguido de verdad, que es bastante más de lo que parece
mirando sólo los parámetros.

### Qué sabrás al terminar

- Cómo guardar el modelo en la cuarta parte de espacio, y qué se pierde exactamente
- Por qué 127 y no 128 (y por qué ese detalle importa más de lo que parece)
- Qué separa tu modelo de GPT-4, con las cinco piezas desglosadas
- Qué te llevas del curso que no sale en los tutoriales

### Cuánto cuesta

2 horas. Es el último.

---

## Cuantización: el modelo en la cuarta parte

Tu modelo ocupa 35,7 MB en fp32. Guardando los pesos en enteros de 8 bits ocuparía 8,9 MB.

La idea es sencilla: en vez de guardar cada peso como un float de 4 bytes, se guarda un
entero de 1 byte más una **escala** que permite recuperar el valor aproximado.

### Con números

Toma una fila de pesos:

```
W = [0.12, -0.45, 0.03, 0.28]
```

El mayor en valor absoluto es 0,45. Se mapea ese rango a `[-127, +127]`:

```
escala = 0.45 / 127 = 0.003543

W_int8 = round(W / escala) = [34, -127, 8, 79]
```

Y para recuperar:

```
W' = W_int8 × escala = [0.1204, -0.4500, 0.0283, 0.2799]
```

No es exacto. El error es del orden de media unidad de escala, y eso es lo que se paga.

### Por qué 127 y no 128

`int8` va de −128 a 127. Usando 127 el rango queda **simétrico** y el cero se representa
exactamente. Eso importa más de lo que parece: en una matriz con muchos valores pequeños,
que el cero sea exacto evita un sesgo sistemático que se acumularía capa tras capa.

### Por canal frente a por tensor

Se puede calcular **una escala para toda la matriz** o **una por fila**. Por fila cuesta un
vector de escalas más —despreciable— y reduce bastante el error, porque una sola fila con
valores grandes no arrastra a las demás.

Medido sobre una matriz real del modelo:

| método | error relativo |
|---|---|
| por tensor | 1,07% |
| **por canal** | **0,71%** |

Es lo que hacen todas las implementaciones serias.

### Qué se gana y qué se pierde

Se gana **4× en tamaño**. En una GPU con poca memoria, eso puede ser la diferencia entre que
el modelo quepa o no.

Se pierde precisión. Que un error del 0,7% en los pesos apenas afecte a la calidad del
modelo es un **hecho empírico**, no un teorema. Nadie predijo que las redes fueran tan
robustas a la cuantización; se descubrió probando.

Y hay un matiz que se suele omitir: **cuantizar los pesos no acelera nada por sí solo** si
después conviertes a float para multiplicar. La aceleración de verdad requiere kernels que
operen en int8 nativamente, y eso depende del hardware.

## Servir el modelo

Con el modelo entrenado y cuantizado, servirlo es un problema de ingeniería normal: un
endpoint HTTP que recibe un prompt y devuelve tokens. Con FastAPI son unas 30 líneas.

Lo único específico de LLM es que conviene **transmitir en streaming**: enviar cada token
según se genera en vez de esperar a la respuesta completa. Con generación a 30 tokens/s, una
respuesta de 200 tokens tarda 7 segundos, y esperar 7 segundos mirando una pantalla en
blanco se percibe como algo roto.

## Y ahora la parte honesta: qué te separa de un modelo frontier

Tu modelo tiene 8,9 millones de parámetros y ha visto 500 millones de tokens. Un modelo
frontier tiene del orden de un billón de parámetros y ha visto decenas de billones de
tokens. La diferencia no es de grado.

Pero **el tamaño es solo una de cinco cosas**, y las otras cuatro se mencionan menos.

### 1. Los datos

Tú usas TinyStories: 2 GB de texto sintético, limpio y homogéneo. Un modelo frontier usa
del orden de 15 billones de tokens, filtrados con clasificadores entrenados para el
propósito, deduplicados, mezclados en proporciones ajustadas experimentalmente, y con
cantidades enormes de código y matemáticas porque **mejoran el razonamiento en tareas que no
son ni código ni matemáticas** — un resultado empírico que nadie predijo y que sigue sin
explicarse bien.

La composición exacta de esos datasets es el secreto peor guardado y mejor protegido del
sector. Ningún laboratorio publica su receta.

### 2. El cómputo

```
tu modelo    : ~2,3·10¹⁶ FLOPs      unas horas en una RTX 2060
GPT-4        : ~2·10²⁵ FLOPs        miles de GPUs durante meses
```

Son **nueve órdenes de magnitud**. Y el coste no es solo de las GPUs: es el centro de datos,
la red que las conecta, y los ingenieros que mantienen todo eso funcionando durante meses sin
que una tirada se caiga.

### 3. La arquitectura

Tu modelo es denso: todos los parámetros participan en cada token. Los modelos grandes usan
**Mixture of Experts**, donde una red enrutadora activa solo una fracción de los parámetros
por token. Eso permite tener un billón de parámetros con el coste de cómputo de cien mil
millones.

Añade también atención con contexto largo, técnicas de eficiencia de memoria en la atención,
y una cantidad considerable de trabajo en que todo eso entrene de forma estable.

### 4. El post-entrenamiento

Viste el SFT en el módulo 16. Después viene RLHF o DPO: recoger preferencias humanas entre
respuestas y ajustar el modelo hacia las preferidas. Y después de eso, iteraciones de
red-teaming, evaluación y ajuste que duran meses.

**Esa fase es la que convierte un modelo que predice texto en algo que quieras usar**, y en
los laboratorios grandes emplea a más gente que el pretraining.

### 5. La infraestructura

Entrenar en miles de GPUs durante meses requiere paralelismo en varias dimensiones a la vez,
tolerancia a fallos (con miles de GPUs, alguna falla cada pocas horas), monitorización, y la
capacidad de reanudar sin perder días de trabajo. Es un problema de sistemas distribuidos
tan difícil como el problema de machine learning.

## Lo que sí has conseguido

Y ahora la otra cara, porque es igual de cierta.

**Has escrito todas las piezas.** La atención, RoPE, SwiGLU, AdamW, la KV cache, el
tokenizador. Todas validadas numéricamente contra PyTorch. Un modelo frontier usa
exactamente estas piezas: más grandes, con más ingeniería alrededor, pero las mismas.

**Sabes leer un paper de arquitectura.** Cuando salga el siguiente modelo y digan que usa
grouped-query attention o RMSNorm o SwiGLU, sabes qué son y por qué.

**Sabes depurar un entrenamiento.** La pérdida del paso 0 frente a `ln(V)`, el overfit a un
batch, la norma del gradiente, la MFU. Eso es lo que distingue a quien sabe entrenar modelos
de quien copia scripts.

**Y sabes qué no se sabe.** A lo largo del curso has visto que SwiGLU funciona sin
explicación, que Adam domina sin que nadie sepa bien por qué, que las leyes de escala tienen
intervalos de confianza más amplios de lo que se reporta, y que la evaluación por benchmarks
está contaminada. Esa parte no suele aparecer en los tutoriales, y es la que más te va a
servir para leer con criterio.

## Dónde está el debate

**Si escalar basta.** La posición de que "escalar es todo lo que hace falta" tiene defensores
serios y detractores serios. Los datos de alta calidad se están agotando, y los modelos
entrenados con datos sintéticos generados por otros modelos muestran degradación en algunos
setups. Nadie sabe si la curva sigue.

**Qué construye un modelo por dentro.** La interpretabilidad mecanicista ha conseguido
explicar componentes concretos —las *induction heads* del módulo 06 son el caso de éxito—
pero está muy lejos de dar cuenta de un modelo entero. Si estos sistemas "entienden" en algún
sentido útil de la palabra es una pregunta abierta, y desconfía de quien te dé una respuesta
tajante en cualquiera de las dos direcciones.

---

**Para ampliar:** Dettmers et al. 2022,
[LLM.int8()](https://arxiv.org/abs/2208.07339) · Shazeer et al. 2017,
[Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538) (MoE) ·
Elhage et al. 2021,
[Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html).
Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
