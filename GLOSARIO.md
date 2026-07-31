# Glosario

Los términos que aparecen en el curso, explicados en una o dos frases y en el orden en que
te los vas a encontrar. Si estás leyendo un `TEORIA.md` y algo no te suena, está aquí.

Entre paréntesis, el módulo donde se explica a fondo.

---

## Lo básico

**Token** — La unidad mínima de texto que maneja el modelo. Puede ser un carácter, una
palabra o (lo habitual) un trozo de palabra. `"tokenización"` podría ser tres tokens:
`token`, `iza`, `ción`. Nuestro modelo final maneja 4096 tokens distintos. *(módulo 03)*

**Vocabulario** (`vocab_size`) — Cuántos tokens distintos conoce el modelo. Es un número
que eliges tú al diseñarlo, no algo que se descubra.

**Modelo de lenguaje** — Una función que, dado un texto, devuelve la probabilidad de cada
token posible como continuación. Eso es todo lo que es. *(módulo 00)*

**Autorregresivo** — Que genera de uno en uno, metiendo cada salida en la entrada del
siguiente paso. Es la razón de que generar texto sea lento y no se pueda paralelizar.

**Contexto** (`context_length`, *ventana*) — Cuántos tokens hacia atrás puede mirar el
modelo. El nuestro, 512. Doblar el contexto cuadruplica el coste de la atención.

**Distribución de probabilidad** — Una lista de números no negativos que suman 1. La salida
de un modelo de lenguaje siempre es una de estas, sobre el vocabulario entero.

**Muestrear** (*sample*) — Elegir un token al azar respetando sus probabilidades, en lugar
de coger siempre el más probable. *(módulos 00 y 14)*

---

## Entrenamiento

**Parámetro** (*peso*) — Cada número que la red ajusta durante el entrenamiento. Nuestro
modelo tiene 8.933.440. GPT-4 tiene del orden de un millón de veces más.

**Embedding** — El vector de números que representa a un token. Tokens que aparecen en
contextos parecidos acaban con vectores parecidos, y ahí está la capacidad de generalizar
que una tabla de conteos no tiene. *(módulo 05)*

**Logit** — La puntuación en bruto que el modelo da a cada token antes de convertirla en
probabilidad. Puede ser cualquier número real, positivo o negativo.

**Softmax** — La función que convierte logits en probabilidades: exponencia cada uno y
divide entre la suma. Exponenciar es lo que permite trabajar con números negativos.

**Pérdida** (*loss*) — Cómo de mal lo está haciendo el modelo. Concretamente
`-ln(probabilidad que le dio al token correcto)`. Si acierta con probabilidad 1, la pérdida
es 0. Entrenar es minimizar este número. *(módulo 05)*

**Cross-entropy** — El nombre técnico de esa pérdida. *(módulo 05)*

**Perplejidad** — `e` elevado a la pérdida. Se interpreta como "entre cuántas opciones está
dudando el modelo, en la práctica". Perplejidad 10 ≈ está dudando entre 10 tokens.
*(módulo 15)*

**Gradiente** — La derivada de la pérdida respecto a un parámetro. Dice en qué dirección
mover ese parámetro para que la pérdida baje. *(módulo 02)*

**Backpropagation** (*backward*) — El algoritmo que calcula todos los gradientes de golpe,
recorriendo la red hacia atrás. Cuesta unas 2 veces lo que cuesta el forward,
independientemente de cuántos parámetros haya. *(módulo 02)*

**Forward** — Pasar los datos por la red y obtener la salida.

**Epoch** (*época*) — Una pasada completa por todo el conjunto de datos.

**Batch** — Un grupo de muestras que se procesan a la vez. Ir de una en una desaprovecha
la GPU.

**Learning rate** (`lr`, *tasa de aprendizaje*) — Cuánto se mueven los parámetros en cada
paso. El hiperparámetro que más veces arruina un entrenamiento. *(módulo 11)*

**Optimizador** — El algoritmo que decide cómo aplicar los gradientes. Usaremos AdamW.
*(módulo 11)*

**Overfitting** (*sobreajuste*) — Cuando el modelo memoriza los datos de entrenamiento en
vez de aprender patrones. Se detecta porque la pérdida de entrenamiento baja y la de
validación no.

**Hiperparámetro** — Un número que eliges tú (learning rate, número de capas), a diferencia
de un parámetro, que lo aprende el modelo.

---

## La arquitectura

**Transformer** — La arquitectura de todos los LLM modernos, publicada en 2017. Su idea
central es la atención. *(módulos 06-10)*

**Atención** (*self-attention*) — El mecanismo que deja que cada token mire a los anteriores
y decida a cuáles hacer caso. *(módulo 06)*

**Query, Key, Value** (Q, K, V) — Las tres proyecciones de la atención. Metáfora útil: la
*query* es la pregunta que lanza un token, la *key* es la etiqueta con la que cada token se
anuncia, y el *value* es el contenido que aporta si resulta elegido. *(módulo 06)*

**Cabeza** (*head*) — La atención se hace varias veces en paralelo con proyecciones
distintas, para que cada "cabeza" pueda especializarse. El nuestro tiene 8. *(módulo 06)*

**Máscara causal** — Impide que un token mire a los que vienen después. Sin ella el modelo
haría trampa: vería la respuesta. *(módulo 06)*

**Normalización** (LayerNorm, RMSNorm) — Reescala los valores dentro de la red para que no
crezcan ni se encojan descontroladamente capa a capa. *(módulo 07)*

**Conexión residual** — Sumar la entrada de un bloque a su salida (`x + f(x)`). Es lo que
permite entrenar redes profundas: da al gradiente un camino directo hasta abajo.
*(módulo 07)*

**FFN / MLP** — La parte de cada bloque que no es atención: dos o tres capas lineales con
una no-linealidad en medio. Suele tener más parámetros que la atención. *(módulo 08)*

**GELU, SwiGLU** — Funciones de activación, la parte "no lineal" sin la cual toda la red
colapsaría a una única multiplicación de matrices. *(módulo 08)*

**Embedding posicional / RoPE** — Cómo se le dice al modelo en qué posición está cada
token. La atención por sí sola no distingue el orden. *(módulo 09)*

**Weight tying** — Reutilizar la matriz de embeddings como matriz de salida. Ahorra 1,3
millones de parámetros en nuestro modelo. *(módulo 10)*

---

## Rendimiento y hardware

**FLOP** — Una operación en coma flotante. Se usa para medir cuánto cuesta entrenar algo.

**MFU** (*Model FLOPs Utilization*) — Qué fracción de la potencia teórica de tu GPU estás
aprovechando de verdad. Un modelo pequeño rara vez pasa del 20%. *(módulos 01 y 12)*

**fp32 / fp16 / bf16** — Formatos numéricos de 32 y 16 bits. fp16 ocupa la mitad y va el
doble de rápido, pero su rango es tan estrecho que los gradientes se van a cero.
*(módulo 01)*

**GradScaler** — El truco que hace viable fp16: multiplica la pérdida por un número grande
antes del backward para que los gradientes no desaparezcan. *(módulo 11)*

**AMP** (*Automatic Mixed Precision*) — Hacer algunas operaciones en 16 bits y otras en 32,
automáticamente.

**KV cache** — Guardar las keys y values ya calculadas para no recalcularlas en cada token
generado. Hace la generación varias veces más rápida. *(módulo 14)*

**Chinchilla** — El resultado de 2022 que dice cuántos tokens conviene usar para entrenar
un modelo de un tamaño dado (aproximadamente 20 por parámetro). *(módulo 12)*

**Cuantización** — Guardar los pesos con menos bits (int8 en vez de fp16) para que el
modelo ocupe menos. *(módulo 17)*

---

## Post-entrenamiento

**Pretraining** — La fase larga: aprender lenguaje prediciendo el siguiente token sobre
muchísimo texto. Es lo que hacemos hasta el módulo 13.

**SFT** (*Supervised Fine-Tuning*) — Seguir entrenando un modelo ya preentrenado sobre
ejemplos de instrucción y respuesta, para que obedezca en vez de limitarse a continuar
texto. *(módulo 16)*

**LoRA** — Entrenar solo unas matrices pequeñas añadidas al modelo en lugar de todos sus
pesos. Mucho más barato. *(módulo 16)*

**RLHF** — Ajustar el modelo con preferencias humanas. No lo haremos, pero es una de las
cosas que separa esto de un modelo comercial. *(módulo 17)*
