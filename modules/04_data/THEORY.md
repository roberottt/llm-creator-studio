# 04 — Datos: de texto a batches en la GPU

## Por qué importa este módulo

**Porque la GPU no puede estar esperando.**

Es el módulo menos glamuroso del curso y uno de los que más rendimiento decide. Si preparar
el siguiente lote de datos tarda más que procesarlo, tu GPU se pasa la mitad del tiempo
parada y tu entrenamiento dura el doble. Con un modelo pequeño como el nuestro, ese riesgo
es real.

Y hay algo más importante que la velocidad: aquí es donde se define **qué aprende el
modelo**. La forma en que emparejas entradas y objetivos es lo que convierte un montón de
texto en una tarea de aprendizaje. Es una idea de tres líneas y es la que hace que los
modelos de lenguaje sean tan eficientes con los datos.

### Qué sabrás al terminar

- Por qué 500 millones de tokens ocupan 1 GB y no 4
- Un bug silencioso de NumPy que te corrompería los datos sin dar ningún error
- **Por qué una sola ventana de 512 tokens son 512 ejemplos de entrenamiento**, no uno
- Por qué el conjunto de validación NO se coge al azar, y qué pasa si lo haces

### Cuánto cuesta

2 horas. Tres funciones cortas, pero la del batch la va a ejecutar tu entrenamiento
decenas de miles de veces.

---

## El problema: la GPU no puede estar esperando

Durante el entrenamiento, la GPU procesa un lote de datos y pide el siguiente. Si
prepararlo tarda más que procesarlo, la GPU se queda parada. Con un modelo pequeño como el
nuestro —que procesa un batch en centésimas de segundo— esto es un riesgo real: es fácil
que el cuello de botella sea leer el disco, no calcular.

Y hay un segundo problema, más tonto pero más caro: **tokenizar 2 GB de texto con tu BPE en
Python puro tarda del orden de una hora**. Hacerlo cada vez que arrancas un entrenamiento
es inaceptable. Hay que hacerlo **una vez** y guardar el resultado.

## La solución: un array de enteros en disco

El plan es: tokenizas el corpus entero una sola vez, guardas los ids en un fichero binario
plano, y a partir de ahí lees de ahí siempre.

### Elegir el tipo: `uint16`

Un token de nuestro modelo es un número entre 0 y 4095. ¿Cuántos bytes le dedicas?

| tipo | rango | 500M tokens ocupan |
|---|---|---|
| `int64` (el de Python) | ±9·10¹⁸ | **4 GB** |
| `uint32` | 0 a 4·10⁹ | 2 GB |
| `uint16` | **0 a 65.535** | **1 GB** |

`uint16` llega hasta 65.535, de sobra para nuestros 4.096. Y ocupa la cuarta parte que el
`int64` que usaría Python por defecto.

**Cuidado con una trampa muy fea de NumPy:** si un id se sale del rango, no avisa. Hace
*wrap around* en silencio. El 65.536 se convierte en 0, el 65.537 en 1. No hay excepción,
no hay warning: simplemente tus datos quedan corruptos y el modelo aprende peor sin que
nada apunte a la causa. Por eso el ejercicio 1 te obliga a validar antes de convertir. Diez
líneas de comprobación ahora contra días de depuración después.

### Guardarlo con `memmap`

Un `np.memmap` es un array de NumPy que vive en disco pero se usa **exactamente igual** que
uno normal: `data[100:200]` funciona sin más. El sistema operativo se encarga de cargar en
memoria solo las páginas que tocas, y de descartarlas cuando hace falta sitio.

Aquí conviene ser honesto, porque se suele explicar mal. Nuestro fichero de 1 GB **cabría
perfectamente en tus 16 GB de RAM**. La razón de usar `memmap` no es que no quepa:

1. **Arranque instantáneo.** Cargar 1 GB del disco a RAM son unos segundos cada vez que
   lanzas el script. Con `memmap` es inmediato: no se lee nada hasta que se toca.
2. **La caché del sistema operativo hace el trabajo.** Como accedes a posiciones aleatorias
   repetidamente, el SO acaba manteniendo en RAM lo que más usas. Gratis y mejor de lo que
   lo harías tú.
3. **Escala sin cambiar nada.** Si mañana entrenas con un corpus de 50 GB, el mismo código
   sigue funcionando.

Si tu corpus es pequeño, cargarlo en RAM con `np.fromfile` es igual de válido y más simple.
No hay magia aquí.

## Cómo se saca un batch

Aquí está la idea que hace que entrenar un modelo de lenguaje sea tan eficiente en datos.

Tienes el corpus como una tira larguísima de números. Eliges una posición al azar y coges
una ventana. La entrada es la ventana, y el objetivo es **la misma ventana desplazada un
token**:

```
corpus = [ 5, 8, 2, 9, 1, 7, ...]

x      = [ 5, 8, 2, 9]
y      = [ 8, 2, 9, 1]
```

Lee la correspondencia columna a columna:

```
viendo [5]            hay que predecir 8
viendo [5,8]          hay que predecir 2
viendo [5,8,2]        hay que predecir 9
viendo [5,8,2,9]      hay que predecir 1
```

**Una sola ventana de 4 tokens produce 4 ejemplos de entrenamiento**, no uno. Con nuestro
contexto de 512, cada muestra da 512 predicciones. Por eso los modelos de lenguaje aprenden
tanto de cada pasada: la señal de entrenamiento es densísima.

Esto es posible gracias a la máscara causal del módulo 06, que impide que la posición 2
pueda ver el token 3. Sin ella, el modelo vería la respuesta y no aprendería nada.

Un batch son varias de estas ventanas apiladas. Con `batch_size=48` y `context_length=512`,
cada `x` es una matriz de `(48, 512)` = 24.576 tokens.

## Entrenamiento y validación: por qué el corte no es aleatorio

Necesitas texto que el modelo **no** haya visto, para saber si está aprendiendo de verdad o
simplemente memorizando.

El reflejo habitual es barajar y repartir. **Aquí es un error**, y la razón es sutil: como
las ventanas se solapan, dos muestras que empiezan en las posiciones 100 y 101 comparten 511
de sus 512 tokens. Si repartieras a nivel de token o de ventana, tu conjunto de validación
estaría lleno de fragmentos que el modelo ya vio en entrenamiento. La pérdida de validación
saldría preciosa y no significaría nada.

La solución es cortar **contiguo y por el final**: el último 0,5% del corpus se reserva
entero. Como TinyStories son historias independientes, eso son historias completas que el
modelo no ha visto jamás.

Es un caso particular de un principio general: el conjunto de validación tiene que ser
independiente del de entrenamiento *en la unidad que importa*. Aquí la unidad no es el
token, es la historia.

## Detalles de rendimiento que sí notas

**`pin_memory` y `non_blocking`** (solo en CUDA). Memoria "fijada" es memoria que el
sistema operativo promete no mover, y eso permite a la GPU leerla por DMA sin que la CPU
intervenga. Combinado con `non_blocking=True`, la copia del siguiente batch se solapa con
el cálculo del actual. En un modelo pequeño, donde el cálculo dura poco, esto se nota.

**La copia con `.astype(np.int64)`.** Los índices de un `nn.Embedding` tienen que ser
`int64`, así que hay que convertir. Y esa conversión, además, materializa el `memmap`: sin
ella, PyTorch se quedaría apuntando a memoria mapeada de disco y cada acceso sería una
lectura.

## Dónde está el debate

El muestreo aleatorio con reemplazo que vamos a usar no es una época en sentido estricto:
algunas ventanas saldrán varias veces y otras ninguna. Es lo que hace nanoGPT y funciona
bien, pero no es lo único razonable — un recorrido ordenado y barajado por bloques da
garantías de cobertura que este método no da. Con 500M tokens y una sola pasada la
diferencia es pequeña; con muchas épocas sobre un corpus pequeño, importaría más.

Más discutido todavía es qué debería haber *dentro* del corpus. El paper de TinyStories
sostiene que un dataset pequeño y muy limpio, con vocabulario de niño de 4 años, permite a
modelos diminutos generar texto coherente — algo que no se consigue entrenando el mismo
modelo con un fragmento de internet del mismo tamaño. Que la calidad y la *distribución* de
los datos importen tanto o más que su cantidad es hoy una de las líneas más activas del
campo, y también una de las menos publicadas: los laboratorios grandes no cuentan qué hay
en sus datasets.

---

**Para ampliar:** Eldan & Li 2023, [TinyStories](https://arxiv.org/abs/2305.07759) ·
[nanoGPT](https://github.com/karpathy/nanoGPT) (su `get_batch` es prácticamente el de este
módulo). Términos sueltos, en [GLOSSARY.md](../../GLOSSARY.md).
