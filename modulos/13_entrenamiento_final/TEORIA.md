# 13 — La tirada real

Aquí no se aprende ningún concepto nuevo. Se lanza el entrenamiento de verdad, y se aprende
lo que solo se aprende haciéndolo: qué mirar, qué es normal, y qué significa que algo vaya
mal.

## Antes de lanzar: la comprobación de los 30 segundos

**Overfit a un solo batch.** Coges cuatro secuencias, se las das al modelo una y otra vez, y
compruebas que la pérdida baja casi a cero.

La idea es que un modelo con millones de parámetros tiene capacidad de sobra para memorizar
cuatro secuencias. Si no lo consigue, hay un bug.

Y lo sabes en **30 segundos** en vez de en cuatro horas.

### Qué caza y qué no

**Caza:** gradientes que no llegan a alguna parte del modelo (un `detach()` de más), el
`zero_grad()` olvidado, un learning rate absurdo, una capa desconectada del grafo, el
optimizador construido sobre los parámetros equivocados.

**No caza:** nada relacionado con generalización. Un modelo que memoriza un batch puede
seguir siendo completamente inútil.

**Un aviso:** si la pérdida baja a cero *demasiado* deprisa —en cinco pasos— sospecha de una
fuga de información. Revisa que los targets vayan desplazados un token respecto a la entrada.

Es el consejo con mejor relación coste/beneficio de todo el deep learning, y aun así casi
nadie lo hace.

## Los tres números del paso 0

Cuando arranca el entrenamiento, mira estos tres antes de irte a hacer otra cosa:

**La pérdida inicial** tiene que valer $\ln(V)$. Con vocabulario 4096, eso es 8,317. Más
alta significa inicialización demasiado agresiva; más baja, fuga de información. Ya lo viste
en el módulo 05 y sigue siendo la comprobación más informativa que existe.

**La norma del gradiente** debería estar en el orden de 0,1 a 10. Si sale $10^5$, algo está
explotando. Si sale $10^{-8}$, algo se está desvaneciendo.

**Los tokens por segundo.** Multiplica por la duración prevista y comprueba que el ETA
cuadra con lo que esperabas. Si son diez veces menos de lo estimado, para y averigua por qué
antes de dejarlo corriendo toda la noche.

## Qué es normal durante la tirada

**La curva de pérdida baja deprisa al principio y luego se aplana.** Eso es lo esperado:
aprender que existen los espacios y las vocales es fácil; aprender gramática, no. En escala
logarítmica la caída es aproximadamente una línea recta, que es lo que dicen las leyes de
escala.

**La pérdida de entrenamiento es ruidosa y la de validación es suave.** La primera se mide
sobre un solo batch; la segunda, sobre cien. El ruido no significa nada.

**La brecha entre ambas crece un poco.** Es sobreajuste incipiente y es normal. Con
TinyStories y una sola pasada por los datos debería quedarse pequeña; si se dispara, el
modelo está memorizando.

**Picos ocasionales.** Un batch raro produce un pico de pérdida y el modelo se recupera en
unos pasos. Con `grad_clip` deberían ser pequeños. Si un pico no se recupera, el
entrenamiento se ha roto: para y reanuda desde el último checkpoint bueno.

## Las muestras de texto: la parte que importa

Cada N pasos, el script genera texto y lo guarda en `samples.md`. Ese fichero, leído de
arriba abajo cuando termine, es el modelo aprendiendo a escribir.

Con el modelo de caracteres sobre Shakespeare el recorrido es aproximadamente este:

```
paso 0      qkxJ;zW,QQjjxk vvv         ruido puro
paso 100    the the the and the       palabras frecuentes
paso 500    I thinks crown me the      estructura de frase, algo de puntuación
paso 1500   KING RICHARD III:          nombres, formato de obra de teatro
            That's such heaven dull
```

**Es más informativo que la curva de pérdida.** Un salto de 1,6 a 1,5 no te dice mucho; ver
que el modelo ha empezado a cerrar los paréntesis, sí.

## Checkpoints: qué hay que guardar

No basta con los pesos. Un checkpoint reanudable necesita:

- los **pesos** del modelo
- el estado del **optimizador** (los momentos de Adam)
- el estado del **GradScaler**
- el **número de paso** y los tokens vistos

Si reanudas solo con los pesos, Adam arranca con sus momentos a cero y el modelo pega un
bandazo justo al reanudar. Se ve como un pico en la curva, exactamente en el punto donde
reanudaste.

**Un detalle de implementación que importa:** escribe primero en un fichero temporal y
renombra al final. Si el proceso muere a mitad de la escritura, el checkpoint anterior sigue
intacto. Un checkpoint a medias es peor que no tener checkpoint.

## La tirada de TinyStories en tu hardware

```
modelo    : 8.933.440 parámetros
tokens    : 500.000.000
FLOPs     : 6 × 7,62M × 500M ≈ 2,3·10¹⁶
```

Con la RTX 2060 a 51,6 TFLOPS de pico y una MFU realista del 10-15%, salen **entre 2 y 5
horas**. Es una estimación de servilleta; el número real lo dará tu propia medición en los
primeros minutos.

Antes de lanzarla, dos cosas: corre el overfit a un batch, y lanza 100 pasos con
`--max-steps 100` para ver el ritmo real y el ETA. Si el ETA dice 40 horas, algo va mal y
más vale saberlo antes.

## Dónde está el debate

**Cuándo parar es una decisión con menos ciencia de la que parece.** Lo estándar es entrenar
hasta agotar el presupuesto de tokens, pero no está claro que sea óptimo: hay evidencia de
que seguir entrenando más allá del punto de Chinchilla sigue mejorando el modelo, con
rendimientos decrecientes que nadie ha caracterizado bien.

**La reproducibilidad exacta es más difícil de lo que parece.** Aunque fijes todas las
semillas, cuDNN elige algoritmos no deterministas por rendimiento, y las reducciones en GPU
suman en orden no determinista. Dos tiradas idénticas divergen. `torch.use_deterministic_algorithms(True)`
lo arregla a costa de velocidad, y para experimentos de investigación merece la pena; para
entrenar, casi nunca.

Y una **sobre este curso**: nuestra tirada de 500M tokens con un solo conjunto de
hiperparámetros no es un experimento controlado. Si al terminar el modelo genera historias
decentes, no sabrás cuánto se debe a la arquitectura, cuánto al learning rate y cuánto al
dataset. Sacar conclusiones de una sola tirada es el error metodológico más común del campo,
y este curso no es una excepción: es un ejercicio de aprendizaje, no un experimento.

---

**Para ampliar:** Eldan & Li 2023, [TinyStories](https://arxiv.org/abs/2305.07759) ·
Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
(de donde viene el consejo del overfit a un batch). Términos sueltos, en
[GLOSARIO.md](../../GLOSARIO.md).
