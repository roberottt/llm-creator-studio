# 04 — Solución comentada

## Ejercicio 1 — `pack_tokens_uint16`

```python
if vocab_size > MAX_UINT16:
    raise ValueError(f"vocab_size={vocab_size} no cabe en uint16...")

array = np.asarray(ids, dtype=np.int64)
if array.size and (array.min() < 0 or array.max() >= vocab_size):
    raise ValueError(
        f"ids fuera del vocabulario [0, {vocab_size}): "
        f"minimo={array.min()}, maximo={array.max()}"
    )
return array.astype(np.uint16)
```

**El orden importa: `int64` primero, comprobar, `uint16` después.** Si convirtieras
directamente a `uint16`, el desbordamiento ya habría ocurrido y estarías comprobando datos
ya corruptos. Hay que validar en un tipo donde todo quepa.

**Por qué esto merece un ejercicio propio.** El comportamiento de NumPy aquí es
genuinamente peligroso:

```python
np.array([65536], dtype=np.int64).astype(np.uint16)   # -> array([0], dtype=uint16)
```

Sin excepción, sin warning. En un pipeline de datos, un bug silencioso que corrompe una
fracción pequeña del corpus es de los peores que existen: el modelo entrena, la pérdida
baja, todo *parece* funcionar, y simplemente el resultado es peor de lo que debería sin que
nada apunte a la causa. La validación cuesta tres líneas.

**El `array.size and ...`.** Sobre un array vacío, `.min()` lanza `ValueError` con un
mensaje sobre operaciones de reducción en secuencias vacías — un error real, pero que no
tiene nada que ver con lo que estás validando y despista. El cortocircuito lo evita.

**El mensaje de error dice los valores.** `"ids fuera de rango"` no ayuda; `"maximo=9999"`
te dice inmediatamente que tu tokenizador está produciendo ids que no debería, y con qué
magnitud.

## Ejercicio 2 — `train_val_split`

```python
if not 0.0 < val_fraction < 1.0:
    raise ValueError(...)

n_val = max(1, int(len(tokens) * val_fraction))
if n_val >= len(tokens):
    raise ValueError("val_fraction deja el conjunto de entrenamiento vacio")
return tokens[:-n_val], tokens[-n_val:]
```

Dos líneas de lógica y una decisión de diseño que sí importa.

**Por qué contiguo y por el final.** Es lo único que hay que entender de este ejercicio.
Las ventanas de entrenamiento se solapan masivamente: la que empieza en la posición 100 y
la que empieza en la 101 comparten 511 de sus 512 tokens. Si repartieras al azar —a nivel
de token o incluso de ventana— el conjunto de validación estaría plagado de fragmentos que
el modelo ya vio en entrenamiento.

El síntoma sería precioso y engañoso: pérdida de validación bajísima, casi idéntica a la de
entrenamiento, y ninguna señal de sobreajuste jamás. Estarías midiendo memorización y
llamándolo generalización.

Cortando un bloque contiguo del final, lo que reservas son historias completas de
TinyStories que el modelo no ha visto nunca. Es la unidad correcta: el conjunto de
validación tiene que ser independiente del de entrenamiento *en la unidad que importa*, y
aquí la unidad no es el token, es la historia.

**Devolver vistas, no copias.** El slicing de NumPy no copia. Con 500M tokens, un `.copy()`
gratuito serían 1 GB de RAM tirado a la basura. El test `test_devuelve_vistas_y_no_copias`
lo comprueba con `np.shares_memory`.

**El `max(1, ...)`.** Con un corpus de 50 tokens y `val_fraction=0.005`, `int(50*0.005)` es
0, y te quedarías sin conjunto de validación. El `max` garantiza al menos un token.

## Ejercicio 3 — `get_batch`

```python
rng = rng or np.random.default_rng()
max_start = len(data) - context_length - 1
if max_start < 1:
    raise ValueError(...)

starts = rng.integers(0, max_start, size=batch_size)
x_np = np.stack([data[i : i+context_length]     for i in starts]).astype(np.int64)
y_np = np.stack([data[i+1 : i+1+context_length] for i in starts]).astype(np.int64)

x, y = torch.from_numpy(x_np), torch.from_numpy(y_np)

if device is not None:
    device = torch.device(device)
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
return x, y
```

**El `-1` de `max_start`.** Es el off-by-one del ejercicio. `y` necesita un token más allá
del final de `x`: si `x` llega hasta `i + context_length - 1`, `y` llega hasta
`i + context_length`. Sin el `-1`, la última ventana posible desborda el array. NumPy no
lanza error al hacer slicing fuera de rango —simplemente devuelve menos elementos— así que
lo que obtendrías es un `np.stack` fallando con un mensaje sobre formas incompatibles, tres
líneas más abajo y sin ninguna pista de la causa real.

**El `.astype(np.int64)` hace dos cosas a la vez.** La obvia: `nn.Embedding` exige índices
`int64`, y los datos están en `uint16`. La menos obvia: la conversión **copia**. Sin esa
copia, `torch.from_numpy` se quedaría apuntando a memoria mapeada de disco, y cada acceso
del modelo sería potencialmente una lectura de fichero.

**`pin_memory` + `non_blocking`, solo en CUDA.** Memoria "fijada" es memoria que el sistema
operativo se compromete a no mover de sitio, lo cual permite a la GPU leerla por DMA sin
intervención de la CPU. Con `non_blocking=True`, la llamada vuelve inmediatamente y la copia
se solapa con lo que la GPU esté calculando. En MPS no aplica —la memoria es unificada, no
hay copia que solapar— y en CPU tampoco.

**El generador `rng` como parámetro.** Es lo que hace el ejercicio testeable: dos llamadas
con `np.random.default_rng(42)` producen exactamente el mismo batch, y el test puede
compararlo con la referencia elemento a elemento. Sin él, solo podrías comprobar formas.

**Sobre el muestreo con reemplazo.** Elegir posiciones al azar en cada llamada significa
que algunas ventanas saldrán varias veces y otras ninguna. No es una "época" en sentido
estricto. Es lo que hace nanoGPT y funciona bien con una sola pasada sobre 500M tokens; con
muchas épocas sobre un corpus pequeño, un recorrido barajado daría mejores garantías de
cobertura.

## Lo que deberías ver en la demo

La correspondencia `x`/`y` sobre texto real:

```
x = 'ot accidenta'
y = 't accidental'

viendo 'o'      -> debe predecir 't'
viendo 'ot'     -> debe predecir ' '
viendo 'ot '    -> debe predecir 'a'
```

Y el número que resume el módulo: **un batch de 8×64 son 512 predicciones**, no 8. Con la
configuración final, 48×512 son 24.576 predicciones por batch. Ahí está la razón de que
entrenar un modelo de lenguaje sea tan eficiente en datos comparado con casi cualquier otra
tarea de aprendizaje supervisado.

La sección de velocidad conviene mirarla con calma en el módulo 12: si `get_batch` tarda
más que un paso de entrenamiento, la GPU se pasa el rato esperando y hay que mover la carga
de datos a un hilo aparte.
