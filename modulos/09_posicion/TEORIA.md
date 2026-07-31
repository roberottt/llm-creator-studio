# 09 — Información posicional y RoPE

## El problema: la atención no sabe qué va antes

Vuelve a mirar la fórmula de la atención del módulo 06. Es una suma ponderada de los
valores, y los pesos salen de productos escalares entre queries y keys.

En ningún sitio aparece la **posición**.

La consecuencia es brutal y conviene verla: para el mecanismo de atención,
*"el perro muerde al hombre"* y *"el hombre muerde al perro"* producen exactamente el mismo
conjunto de vectores de salida, solo que reordenados. Si barajas los tokens de entrada, la
salida se baraja igual y nada más cambia. A esa propiedad se le llama **equivariancia a
permutaciones**, y aquí es un defecto fatal: el orden de las palabras es la mitad del
significado.

Hay que meter la posición de alguna forma. Vamos a ver tres, en orden histórico.

## Opción 1: aprender una tabla

La más simple. Una tabla con una fila por posición, que se entrena como cualquier otro
parámetro, y se **suma** al embedding del token:

```
entrada = embedding_de_token[id] + embedding_de_posición[i]
```

Es lo que hace GPT-2. Funciona bien y no tiene misterio.

Tiene dos pegas. La primera es un **techo duro**: si entrenaste con 1024 posiciones, para la
posición 1025 no hay fila que consultar. El modelo no puede procesarla de ninguna manera, ni
mal. La segunda es que el modelo aprende posiciones **absolutas** — "esto es el token
número 7" — cuando lo que suele importar es la relación: "esto está dos palabras antes del
verbo".

## Opción 2: senos y cosenos

El paper de 2017 propuso una tabla fija, sin parámetros, hecha de senos y cosenos de
distintas frecuencias:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

La intuición es la de un **contador binario**. Fíjate en cómo se cuenta en binario:

```
0000    el bit de la derecha cambia en cada paso
0001    el siguiente, cada dos
0010    el siguiente, cada cuatro
0011    ...
```

Cada bit oscila a un ritmo distinto, y la combinación de todos identifica un número de forma
única. Las sinusoidales hacen lo mismo pero con ondas continuas: los primeros pares de
dimensiones oscilan rápido y distinguen posiciones vecinas; los últimos oscilan lentísimo y
distinguen el principio del final de la secuencia.

Ventaja sobre la tabla aprendida: está definida para cualquier posición, no hay techo. En la
práctica la extrapolación tampoco funciona muy bien, pero al menos existe.

## Opción 3: RoPE — rotar en vez de sumar

Aquí está la idea que usa nuestro modelo, y Llama, y casi todo lo moderno.

**En lugar de sumar algo al vector, se le aplica una rotación cuyo ángulo depende de la
posición.**

Toma un vector de 2 dimensiones y rótalo un ángulo $\theta$. La matriz de rotación de toda la
vida:

$$\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} =
\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

RoPE parte el vector de cada cabeza en pares y rota cada par un ángulo proporcional a la
posición. Como en las sinusoidales, cada par tiene su propia velocidad de giro: los primeros
giran deprisa, los últimos lentísimo.

### Por qué esto es tan buena idea

Y aquí viene la propiedad que lo justifica todo. Las rotaciones tienen una particularidad:
**el producto escalar de dos vectores rotados depende solo de la diferencia de ángulos.**

$$\langle R(m)\,q,\; R(n)\,k \rangle = \langle q,\; R(n-m)\,k \rangle$$

Traducido a lo que importa: la puntuación de atención entre el token de la posición 5 y el
de la 3 es **idéntica** a la que habría entre el 105 y el 103. Lo que el modelo aprende no es
"el token número 3" sino **"el token de dos posiciones atrás"**.

Puedes comprobarlo tú: en la demo se calcula $\langle R(2)q, R(5)k \rangle$ y
$\langle R(4)q, R(7)k \rangle$ y salen el mismo número hasta el último decimal.

Y hay un segundo beneficio: rotar **no cambia la longitud del vector**. Sumar un embedding
posicional sí altera la magnitud, y eso interfiere con los productos escalares de la
atención. Rotar solo cambia la dirección.

### Dos detalles de implementación

**Solo se aplica a Q y K, nunca a V.** Lo que debe depender de la posición son las
*puntuaciones* de atención, no el contenido que se transporta. Y como la posición ya está
codificada en las puntuaciones, meterla también en los valores sería redundante y dañino.

**Se aplica dentro de cada cabeza**, sobre `head_dim` dimensiones (40 en nuestro caso, o sea
20 pares), no sobre las 320 de `d_model`.

Sobre cómo emparejar las dimensiones hay dos convenios. El paper original empareja
consecutivas: $(x_0, x_1), (x_2, x_3)\ldots$. Llama y HuggingFace emparejan por mitades:
$(x_0, x_{d/2}), (x_1, x_{d/2+1})\ldots$. **Son equivalentes salvo una permutación de las
dimensiones**, que la red aprende sin enterarse, y el de mitades se implementa con
operaciones vectoriales mucho más limpias. Usamos ese.

## Dónde está el debate

Se dice mucho que RoPE "extrapola a contextos más largos". Es verdad a medias y conviene
saber dónde acaba.

RoPE tiene la propiedad relativa, sí, pero un modelo entrenado con contexto 512 y evaluado
con 4096 **se degrada bastante**. La razón es que las frecuencias lentas apenas completan
una fracción de vuelta dentro del rango entrenado, así que los ángulos grandes son
literalmente territorio no visto. Hay toda una familia de técnicas para extender el contexto
después de entrenar —interpolación de posiciones, NTK-aware scaling, YaRN— que existen
precisamente porque la extrapolación directa no basta.

Más de fondo: no está claro *por qué* la codificación posicional relativa funciona mejor que
la absoluta. Hay argumentos razonables sobre generalización, y hay evidencia de que los
transformers con máscara causal **infieren cierta información posicional por su cuenta**
incluso sin ninguna codificación explícita, porque la propia máscara rompe la simetría. Hay
trabajos que entrenan modelos causales sin codificación posicional alguna y funcionan
sorprendentemente bien. O sea que ni siquiera está claro cuánto de necesario es todo esto.

---

**Para ampliar:** Su et al. 2021, [RoFormer](https://arxiv.org/abs/2104.09864) (RoPE) ·
Press et al. 2021, [ALiBi](https://arxiv.org/abs/2108.12409) (otra alternativa, que sesga las
puntuaciones en vez de rotar) · Vaswani et al. 2017 (las sinusoidales originales).
Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
