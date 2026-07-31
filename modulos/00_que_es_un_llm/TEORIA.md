# 00 — Qué es un LLM, en realidad

## Por qué importa este módulo

**Empieza aquí aunque tengas prisa.** Es el único módulo sin PyTorch, sin matrices y sin
derivadas, y es el que hace que todo lo demás tenga sentido.

La razón: el resto del curso construye piezas cada vez más sofisticadas para hacer **una
sola cosa**. Si no tienes clarísimo cuál es esa cosa, los 17 módulos siguientes son
ingeniería sin propósito.

En una hora vas a escribir un generador de texto que funciona de verdad, con diccionarios y
una división. Y vas a ver que el bucle que lo mueve es *literalmente* el mismo que ejecuta
ChatGPT.

### Qué sabrás al terminar

- Qué es exactamente un modelo de lenguaje (spoiler: mucho menos místico de lo que parece)
- Por qué se dice que "solo predice el siguiente token", y qué significa eso de verdad
- Cómo se elige ese token, y por qué no se coge siempre el más probable
- **Por qué hacen falta redes neuronales**, viendo con números por qué la alternativa
  obvia se estrella contra un muro

### Cuánto cuesta

Una hora. Es el módulo más corto y el que más rentabilidad da.

---

## La idea, en una frase

**Un modelo de lenguaje es una función que, dado un texto, te dice qué probabilidad tiene
cada posible continuación.**

Nada más. No "entiende", no "razona", no "sabe". Recibe un trozo de texto y devuelve una
lista de probabilidades, una por cada palabra o carácter que podría venir a continuación.

Si le das *"El cielo es de color "*, un buen modelo devolverá algo así:

```
azul      0.72
gris      0.11
negro     0.04
rosa      0.02
patata    0.0000003
...
```

Y ya está. Eso es el modelo entero. Lo que ves cuando hablas con ChatGPT es este paso
repetido: elige una palabra según esas probabilidades, la pega al final del texto, y
vuelve a preguntar. Una y otra vez, palabra a palabra.

A ese bucle se le llama **generación autorregresiva** («auto» = a sí mismo, «regresivo» =
se realimenta). Es importante que veas la consecuencia: el modelo no planifica la frase
entera. Escribe un token, lo lee como si se lo hubiera dado otro, y decide el siguiente.

## Vamos a construir uno ahora mismo

Un modelo tiene que sacar esas probabilidades de algún sitio. La forma más tonta que
existe, y que funciona: **contar**.

Coge un texto y apunta, para cada carácter, cuáles le siguieron y cuántas veces. Con el
texto `"banana"`:

```
tras 'b'  ->  'a' 1 vez
tras 'a'  ->  'n' 2 veces
tras 'n'  ->  'a' 2 veces
```

Ahora conviértelo en probabilidades dividiendo entre el total:

```
tras 'a'  ->  'n' con probabilidad 2/2 = 1.0
```

Con un texto de verdad, la tabla de la `'a'` sería más interesante:

```
tras 'a'  ->  'n' 40 veces,  'r' 25,  ' ' 20,  's' 15
total = 100
        ->  'n' 0.40,  'r' 0.25,  ' ' 0.20,  's' 0.15
```

Eso es una **distribución de probabilidad**: una lista de números no negativos que suman 1.
Todo el curso va de producir distribuciones sobre el siguiente token. Cambiará radicalmente
*cómo* las producimos; qué son, no.

### Elegir uno

Tienes `{'n': 0.40, 'r': 0.25, ' ': 0.20, 's': 0.15}`. ¿Cuál eliges?

Si coges siempre el más probable (`'n'`), el modelo es determinista y aburridísimo: con la
misma entrada escribe siempre exactamente lo mismo, y tiende a meterse en bucles. Por eso
se **muestrea**: se tira un dado trucado en el que `'n'` sale el 40% de las veces.

El método es el de la ruleta. Saca un número aleatorio entre 0 y 1 y ve acumulando:

```
r = 0.61

'n'  acumulado = 0.40    0.61 > 0.40, sigo
'r'  acumulado = 0.65    0.61 < 0.65, ¡me paso aquí!  ->  sale 'r'
```

Cada token ocupa un trozo de la recta [0,1] proporcional a su probabilidad, y el número
aleatorio cae en uno de ellos. En el módulo 14 verás cómo se manipula esta ruleta
(temperatura, top-k, top-p) para que el texto salga más creativo o más conservador.

## Por qué esto no basta

Tu modelo del ejercicio 3 generará algo parecido a esto entrenado sobre Shakespeare:

```
QUEO: hend f th s the wive an t ourourthe
```

Reconoce que existen las vocales y los espacios. No sabe nada más. El problema es que
**solo mira un carácter hacia atrás**. Para decidir qué viene después de la `'e'` de
*"el gat"*, mirar solo la `'t'` es desesperante.

La reacción obvia es mirar más atrás: contar trigramas, o ventanas de 10 caracteres. Y
funciona un poco mejor, hasta que te estrellas contra un muro. Con un vocabulario de 4096
tokens y una ventana de 10, la tabla tendría $4096^{10} \approx 10^{36}$ entradas. No hay
disco en el planeta, y además casi todas estarían vacías: la mayoría de las combinaciones
de 10 tokens no aparecen jamás, ni siquiera en todo internet.

Este es **el problema central del modelado del lenguaje**, y tiene nombre: la maldición de
la dimensionalidad. Contar no escala.

## Lo que hace una red neuronal

La solución no es contar mejor, es **generalizar**. Si el modelo ha visto *"el gato negro
duerme"*, debería poder decir algo sensato sobre *"el perro negro duerme"* aunque no lo
haya visto nunca.

Contar no puede: para una tabla, `"gato"` y `"perro"` son dos claves distintas sin ninguna
relación, tan distintas entre sí como `"gato"` y `"paraguas"`.

Una red neuronal representa cada token como un **vector de números** aprendido de los datos
(un *embedding*). Si `"gato"` y `"perro"` acaban con vectores parecidos —porque aparecen en
contextos parecidos— entonces lo aprendido sobre uno se transfiere automáticamente al otro.
Ahí está toda la gracia. El modelo comprime miles de millones de conteos imposibles en unos
pocos millones de números que capturan *parecidos*.

Y una vez tienes vectores, necesitas una forma de que cada palabra decida a cuáles de las
anteriores hacer caso. Eso es la **atención**, y es el módulo 06.

## Los tres números que vas a ver todo el rato

**Token**: la unidad de texto que maneja el modelo. Aquí serán caracteres; a partir del
módulo 03 serán trozos de palabra. Nuestro modelo final tendrá 4096 tokens distintos.

**Parámetros**: los números que la red aprende. El nuestro tendrá 8.933.440. GPT-4 tiene
del orden de un millón de veces más.

**Pérdida** (*loss*): cómo de mal lo está haciendo. Concretamente, `-ln(probabilidad que
el modelo le dio al token correcto)`. Si le dio 1.0 al token que de verdad venía, la
pérdida es 0. Si le dio 0.01, la pérdida es 4.6. Entrenar es minimizar este número, y en
el módulo 05 verás por qué esta fórmula concreta y no otra.

## Dónde está el debate

Que un LLM «solo predice el siguiente token» es cierto y a la vez engañoso. La pregunta
abierta —de verdad abierta, no retórica— es qué estructura interna necesita construir un
sistema para predecir bien. Hay evidencia de que modelos entrenados solo con predicción de
texto acaban desarrollando representaciones internas de cosas que nadie les enseñó
explícitamente. Hay quien lo lee como comprensión emergente y quien lo lee como estadística
sofisticada. Nadie tiene la respuesta, y desconfía de quien te diga lo contrario en
cualquiera de las dos direcciones.

---

**Para ampliar:** Shannon 1948,
[A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
— el artículo que inventó esto, y donde ya aparecen los modelos por conteo que vas a
programar hoy. Si un término no te suena, está en [GLOSARIO.md](../../GLOSARIO.md).
