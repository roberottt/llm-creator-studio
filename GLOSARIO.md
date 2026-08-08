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

**Tokenizar** — Convertir texto en la lista de enteros que el modelo entiende, y al revés.
Se hace **antes** de que el modelo vea nada: no forma parte de la red. *(módulo 03)*

**BPE** (*Byte Pair Encoding*) — El algoritmo que decide cuáles son los tokens: parte de los
256 bytes y va fusionando el par de vecinos más frecuente hasta llenar el vocabulario. Nadie
escribe la lista de tokens; se descubre contando. *(módulo 03)*

**Merge** — Una de esas fusiones: la regla «cuando veas este par, cámbialo por este id
nuevo». Un tokenizador entrenado es una lista ordenada de merges, y el orden importa: al
codificar se aplican en el mismo orden en que se aprendieron. *(módulo 03)*

**Pre-tokenizador** — La expresión regular que trocea el texto en palabras, números y signos
**antes** de contar pares, para que ningún merge cruce de una palabra a la siguiente. Sin él
BPE aprende tokens como `". el gato duer"`. *(módulo 03)*

**Bytes fallback** — Trabajar sobre bytes (0-255) en vez de sobre caracteres Unicode. Como
todo texto es una secuencia de bytes y los 256 están en el vocabulario, no existe el
carácter imposible de codificar. Al decodificar, `errors="replace"` cubre el caso contrario:
bytes que no forman UTF-8 válido salen como `�` en vez de tumbar la generación.
*(módulo 03)*

**`<UNK>`** — El token «palabra desconocida» de los tokenizadores clásicos por palabras.
Destruye información sin remedio, y con bytes fallback deja de hacer falta. *(módulo 03)*

**Compresión** (*bytes por token*) — Cuánto texto cabe de media en un token. Un vocabulario
más grande comprime mejor (secuencias más cortas, menos pasos de entrenamiento) pero se come
el presupuesto de parámetros en la tabla de embeddings. Ese intercambio es lo que decide el
`vocab_size`. *(módulo 03)*

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

**Greedy** (*argmax*) — Lo contrario de muestrear: coger siempre el token más probable. Es
determinista y tiende a meterse en bucles repetitivos. *(módulos 00 y 14)*

**n-grama** — Un modelo de lenguaje que predice contando cuántas veces siguió cada token a
cada secuencia de `n` tokens anteriores. Es lo que construyes en el módulo 00. Funciona,
pero la tabla crece exponencialmente con `n`. *(módulo 00)*

**Bigrama** — El n-grama más pequeño que sirve de algo: el modelo que predice mirando **un
solo token atrás**. Su tabla de conteos es de `V × V`. Malísimo, y sorprende lo lejos que
llega. *(módulo 05)*

**Maldición de la dimensionalidad** — El motivo por el que contar no escala: al ampliar el
contexto, las combinaciones posibles crecen exponencialmente y el corpus cubre una fracción
cada vez más ridícula de ellas. Todo lo no visto se queda con probabilidad cero. Es el
problema que las redes neuronales resuelven generalizando. *(módulo 00)*

**Suavizado** (*smoothing*) — Los parches clásicos para las probabilidades cero de un
modelo de n-gramas: repartir algo de masa entre lo nunca visto, o mezclar modelos de varios
tamaños de contexto. Alivia el síntoma, no la causa. *(módulo 00)*

**Suavizado de Laplace** (*add-α*) — El suavizado más simple: sumar una constante `α` a
todos los conteos antes de normalizar, para que ninguna probabilidad sea cero. Sin él, un
único par no visto manda la pérdida a infinito, porque `-ln(0)` es infinito y la pérdida es
una media. Es admitir que «no lo he visto» no es lo mismo que «es imposible». No es el mejor
suavizado —**Kneser-Ney** reparte la masa sobrante según en cuántos contextos distintos
aparece cada token, en vez de por igual, y gana con claridad—, pero sí el más simple, y el
n-grama es un baseline que se abandona en el módulo siguiente. *(módulo 05)*

**Baseline** — Un modelo deliberadamente simple contra el que se compara el modelo de
verdad. Sin baseline, una pérdida de 2,49 no significa nada. El más importante es el
**uniforme**, que reparte la probabilidad por igual y da exactamente `ln(V)`. *(módulo 05)*

**Máxima verosimilitud** — El criterio de elegir los parámetros que hacen más probable el
texto observado. Minimizar la cross-entropy es exactamente eso. *(módulos 00 y 05)*

**Alucinación** — Que el modelo genere algo falso con total aplomo. No es un fallo añadido:
es la consecuencia directa de muestrear de una distribución sin ningún paso de
verificación. *(módulo 00)*

**Generalizar** — Acertar sobre datos que no se han visto durante el entrenamiento. Es lo
único que distingue aprender de memorizar. *(módulo 00)*

---

## Los datos

**Aprendizaje autosupervisado** — Entrenar sin etiquetas hechas por humanos, porque la
respuesta correcta se saca del propio dato: la respuesta a «¿qué token viene después?» es,
literalmente, el token que venía después. Es lo que permite entrenar con texto crudo de
internet y la razón de que los LLM despegaran. *(módulo 04)*

**Pipeline de datos** — Todo lo que pasa entre el texto crudo y el batch que entra al
modelo: tokenizar, empaquetar, guardar, partir en train/validación y muestrear. En un
laboratorio de verdad incluye además filtrar, deduplicar y decidir la mezcla de fuentes.
*(módulo 04)*

**Corpus** — Todo el texto con el que se entrena, ya tokenizado: una tira de varios cientos
de millones de enteros. El nuestro son 500M tokens de TinyStories. *(módulo 04)*

**Deduplicar** — Quitar del corpus los documentos repetidos. Sin ello el modelo ve el mismo
texto muchas veces y lo memoriza en vez de aprender de él. TinyStories viene limpio y aquí
no hace falta. *(módulo 04)*

**TinyStories** — El dataset del curso: historias cortas generadas con vocabulario de niño
de cuatro años. Su gracia es que un modelo diminuto puede aprender a escribirlas coherentes,
cosa que no pasa con un trozo de internet del mismo tamaño. *(módulo 04)*

**`uint16`** — Entero sin signo de 2 bytes, de 0 a 65.535. El tipo en el que se guarda cada
token: con `int64` el mismo corpus ocuparía cuatro veces más. *(módulo 04)*

**Wrap around** (*desbordamiento silencioso*) — Lo que hace NumPy cuando conviertes a un
tipo donde el número no cabe: da la vuelta al contador sin avisar (65.536 se convierte en
0). Corrompe los datos sin lanzar ningún error. *(módulo 04)*

**`memmap`** (*memoria mapeada*) — Un array cuyos datos viven en un fichero y no en RAM,
pero que se usa exactamente igual que uno normal. El sistema operativo carga solo las
páginas que tocas. *(módulo 04)*

**Ventana deslizante** — La forma de sacar muestras del corpus: se elige una posición al
azar y se toman `context_length` tokens seguidos. Dos ventanas contiguas comparten casi
todos sus tokens, y de ahí sale la regla de no barajar al partir train/validación.
*(módulo 04)*

**Conjunto de validación** — El trozo de corpus con el que NO se entrena, reservado para
medir si el modelo generaliza o está memorizando. Se corta contiguo y por el final, nunca al
azar. *(módulo 04)*

**Muestreo con reemplazo** — Elegir cada ventana al azar sin llevar cuenta de las que ya
salieron. Algunas saldrán repetidas y otras ninguna, así que no es una época en sentido
estricto; a cambio la función no tiene estado. *(módulo 04)*

**Memoria fijada** (*pinned*, *page-locked*) — Memoria que el sistema operativo se
compromete a no mover, lo que permite a la GPU leerla por DMA sin que la CPU haga de
intermediaria. Con `non_blocking=True`, la copia del siguiente batch se solapa con el
cálculo del actual. Solo tiene sentido en CUDA. *(módulo 04)*

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

**Nat** — La unidad en la que se mide la pérdida cuando se usa logaritmo natural, que es lo
que hace `torch.log` y por tanto todo este curso. Un nat son 1,44 bits. *(módulo 05)*

**One-hot** — Un vector de ceros con un único 1, en la posición del token. Multiplicar una
matriz por un vector one-hot da exactamente la fila correspondiente de la matriz, y por eso
`nn.Embedding` y `nn.Linear` son la misma operación: el embedding **lee** la fila en vez de
multiplicar por una matriz llena de ceros. *(módulo 05)*

**Capa oculta** (*hidden layer*) — Cualquier capa de una red que no es ni la entrada ni la
salida. «Oculta» solo significa que nadie mira sus números directamente. *(módulos 02 y 05)*

**Perplejidad** — `e` elevado a la pérdida. Se interpreta como "entre cuántas opciones está
dudando el modelo, en la práctica". Perplejidad 10 ≈ está dudando entre 10 tokens.
*(módulo 15)*

**Entropía** — Lo mismo que la perplejidad pero sin exponenciar: mide cómo de repartida está
una distribución. Máxima cuando todo es equiprobable (`ln(n)`), cerca de cero cuando toda la
masa está en una opción. En el módulo 06 se usa para medir si la atención reparte o se fija
en un solo token. *(módulos 05 y 06)*

**Dropout** — Apagar al azar una fracción de los números durante el entrenamiento, para que el
modelo no dependa demasiado de ninguno en concreto. Se desactiva en evaluación, y hay que
acordarse de hacerlo a mano cuando la operación no consulta el modo por su cuenta.
*(módulos 06 y 11)*

**Gradiente** — La derivada de la pérdida respecto a un parámetro. Dice en qué dirección
mover ese parámetro para que la pérdida baje. *(módulo 02)*

**Backpropagation** (*backward*) — El algoritmo que calcula todos los gradientes de golpe,
recorriendo la red hacia atrás. Cuesta unas 2 veces lo que cuesta el forward,
independientemente de cuántos parámetros haya. *(módulo 02)*

**Regla de la cadena** — Si `y` depende de `u` y `u` depende de `x`, entonces
`dy/dx = (dy/du)·(du/dx)`. Toda la backpropagation es esto aplicado operación a operación.
Si una variable influye por varios caminos, sus aportaciones se **suman**. *(módulo 02)*

**Grafo de cómputo** — El registro de qué operaciones se hicieron, sobre qué operandos y en
qué orden. Se construye solo durante el forward y es lo que permite recorrerlo hacia atrás.
En nuestro motor son los campos `_prev` y `_op` de cada `Value`. *(módulo 02)*

**Autodiferenciación en modo inverso** — La técnica que calcula derivadas exactas
descomponiendo el cálculo en operaciones elementales y recorriendo el grafo hacia atrás. Ni
numérica (aproximada y cara) ni simbólica (inmanejable). Es lo que hay dentro de
`torch.autograd`. *(módulo 02)*

**Orden topológico** — El orden en que hay que recorrer el grafo para que ningún nodo
reparta su gradiente antes de haber recibido el de todos sus padres. Un orden mal calculado
da gradientes incorrectos sin dar ningún error. *(módulo 02)*

**Descenso de gradiente** — La regla de aprendizaje: `p -= lr * p.grad`. Mover cada
parámetro un poco en contra de su gradiente, porque el gradiente apunta hacia donde la
pérdida sube. *(módulo 02)*

**Neurona** — La unidad mínima: una suma ponderada de sus entradas más un sesgo,
`w₁x₁ + w₂x₂ + … + b`, pasada por una función no lineal. Los `w` y el `b` son sus
parámetros. *(módulo 02)*

**Sesgo** (*bias*) — El término independiente `b` de una neurona: le permite desplazar su
salida sin depender de la entrada. *(módulo 02)*

**Función de activación** — La parte no lineal de una neurona (`tanh`, `relu`, `gelu`). Sin
ella, apilar capas no sirve de nada: la composición de funciones lineales es otra función
lineal. *(módulos 02 y 08)*

**MSE** (*error cuadrático medio*) — Una pérdida para predecir números:
`media((predicción - objetivo)²)`. Se usa en el módulo 02; para predecir tokens se usa
cross-entropy. *(módulo 02)*

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
haría trampa: vería la respuesta. Se aplica poniendo `-inf` en las puntuaciones prohibidas
**antes** del softmax, no borrando pesos después: así cada fila sigue sumando 1.
*(módulo 06)*

**Puntuaciones** (*scores*) — La matriz `T × T` de productos escalares entre queries y keys,
antes del softmax. La casilla `(i, j)` dice cuánto le interesa al token `i` el token `j`.
*(módulo 06)*

**Producto escalar** — Multiplicar dos vectores componente a componente y sumar. Mide cuánto
se parecen: cuanto más alineados, mayor el número. Es la operación con la que la atención
decide a quién hacer caso. *(módulo 06)*

**Escalado por √d_k** — Dividir las puntuaciones por la raíz del tamaño de las queries. El
producto escalar de dos vectores de dimensión `d_k` tiene varianza `d_k`, y sin corregirlo el
softmax se satura: los pesos se van a 0 y 1, su derivada `p(1-p)` se anula y la capa deja de
aprender. *(módulo 06)*

**Proyección de salida** (`out_proj`) — La cuarta matriz de una capa de atención, que no
aparece en la fórmula del paper. Mezcla los resultados de las cabezas entre sí; sin ella
serían canales estancos. *(módulo 06)*

**SDPA** (`scaled_dot_product_attention`) — La implementación fusionada de la atención que
trae PyTorch: hace los cuatro pasos en un solo kernel sin materializar la matriz `T × T`
entera. Es la que usa el entrenamiento de verdad. *(módulo 06)*

**Induction head** — Una cabeza de atención que aprende sola a detectar el patrón «…A B … A»
y a predecir B. Es el componente mejor entendido de un Transformer y el caso de éxito de la
interpretabilidad mecanicista. *(módulo 06)*

**Normalización** (LayerNorm, RMSNorm) — Reescala los valores dentro de la red para que no
crezcan ni se encojan descontroladamente capa a capa. *(módulo 07)*

**Conexión residual** — Sumar la entrada de un bloque a su salida (`x + f(x)`). Es lo que
permite entrenar redes profundas: da al gradiente un camino directo hasta abajo. Al derivar,
esa suma aporta un `1` que ninguna capa puede atenuar. *(módulo 07)*

**Corriente residual** (*residual stream*) — El canal principal por el que viaja la
representación de cada token de punta a punta del modelo. Cada bloque lee de él, calcula algo y
**suma** su aportación de vuelta, en lugar de sustituirlo. *(módulo 07)*

**Pre-norm / post-norm** — Si la normalización va dentro de la rama (`x + f(norm(x))`) o
envolviendo la suma (`norm(x + f(x))`). Sólo cambian los paréntesis, y decide si el gradiente
atraviesa una normalización por capa o llega intacto abajo. Todo lo moderno es pre-norm.
*(módulo 07)*

**BatchNorm** — La normalización que se calcula a lo largo del batch, no de las dimensiones de
cada token. No se usa en Transformers: el resultado de un ejemplo dependería de con quién
comparte lote, y en inferencia con un solo ejemplo hay que tirar de estadísticas guardadas.
*(módulo 07)*

**Gradiente que se desvanece / que explota** — Cuando el gradiente se multiplica capa tras capa
por un factor menor o mayor que 1 y acaba en cero o en infinito. Con 64 capas lineales llega a
ser cero **exacto**, por underflow de la coma flotante. *(módulo 07)*

**Warmup** — Empezar el entrenamiento con un learning rate muy pequeño y subirlo poco a poco
durante los primeros pasos. Post-norm lo necesita para no explotar; pre-norm no.
*(módulos 07 y 11)*

**FFN / MLP** — La parte de cada bloque que no es atención: dos o tres capas lineales con
una no-linealidad en medio. Suele tener más parámetros que la atención. *(módulo 08)*
Cuidado con el nombre: en el módulo 02, "MLP" es la red entera (capas de neuronas
encadenadas); en un transformer es solo ese sub-bloque de cada capa.

**GELU, SwiGLU** — Funciones de activación, la parte "no lineal" sin la cual toda la red
colapsaría a una única multiplicación de matrices. *(módulo 08)*

**Embedding posicional / RoPE** — Cómo se le dice al modelo en qué posición está cada
token. La atención por sí sola no distingue el orden. *(módulo 09)*

**Weight tying** — Reutilizar la matriz de embeddings como matriz de salida. Ahorra 1,3
millones de parámetros en nuestro modelo. *(módulo 10)*

---

## Rendimiento y hardware

**FLOP** — Una operación en coma flotante. Se usa para medir cuánto cuesta entrenar algo.

**TFLOPS** — Billones de FLOPs por segundo. La unidad en la que se mide la potencia de una
GPU. El pico de la ficha técnica y el que consigues de verdad se llevan un factor de 3 o
más. *(módulo 01)*

**MFU** (*Model FLOPs Utilization*) — Qué fracción de la potencia teórica de tu GPU estás
aprovechando de verdad. Un modelo pequeño rara vez pasa del 20%. *(módulos 01 y 12)*

**Compute-bound / memory-bound** — Si una operación está limitada por la potencia de cálculo
(un matmul grande) o por el ancho de banda de la memoria (una activación, una
normalización). La fórmula de los FLOPs solo ve las primeras, y de ahí viene buena parte de
la diferencia entre el tiempo estimado y el real. *(módulo 01)*

**Tensor cores** — Las unidades de una GPU NVIDIA especializadas en multiplicar matrices
pequeñas en 16 bits. Son las que dan las cifras grandes de la ficha técnica, y solo se
aprovechan con matrices suficientemente gordas. *(módulo 01)*

**Compute capability** (`sm_75`, `sm_80`…) — La generación de una GPU NVIDIA, que determina
qué sabe hacer. bf16 y FlashAttention-2 necesitan `sm_80` (Ampere); la serie RTX 20 se queda
en `sm_75`.
*(módulo 01)*

**Gradient checkpointing** — Recalcular el forward de algunos bloques durante el backward en
vez de guardar sus activaciones. Ahorra memoria y sube el coste de 6N a 8N por token.
*(módulos 01 y 12)*

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
