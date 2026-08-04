# 15 — Solución comentada

## Ejercicio 1 — `perplexity_from_loss`

```python
if not math.isfinite(loss):
    return float("inf")
return math.exp(loss)
```

La guarda de `isfinite` no es decorativa: `math.exp(inf)` lanza `OverflowError`, y en medio
de una evaluación eso te deja sin saber qué pasó. Devolver `inf` es informativo.

**La comprobación útil:** con pérdida $\ln(V)$, la perplejidad es exactamente $V$. Con
vocabulario 4096 y un modelo sin entrenar, perplejidad 4096. Es el mismo detector de bugs
del módulo 05 visto desde el otro lado.

## Ejercicio 2 — `bits_per_byte`

```python
if n_bytes <= 0:
    raise ValueError("n_bytes tiene que ser positivo")
return total_loss_nats / math.log(2) / n_bytes
```

**El `/ math.log(2)`** convierte nats a bits: un nat son $1/\ln 2 = 1{,}4427$ bits.

**El primer argumento es la pérdida TOTAL, no la media.** Si le pasas la media, el resultado
sale dividido por el número de tokens y no significa nada. El parámetro `n_tokens` no se usa
en el cálculo; está en la firma justo para dejarlo claro.

**Por qué esta métrica y no la perplejidad.** La perplejidad depende del tokenizador: si tu
vocabulario parte las palabras en trozos más pequeños, cada token es más fácil de predecir y
tu número sale mejor sin que el modelo lo sea. Bits por byte normaliza por bytes del texto
original, que no dependen de cómo trocees.

Y tiene una interpretación exacta: **cuántos bits necesitarías para transmitir el texto
usando el modelo como compresor**. No es una analogía. Un modelo de lenguaje *es* un
compresor, y la equivalencia entre predicción y compresión viene de Shannon (1948).

## Ejercicio 3 — `run_prompt_battery`

```python
prompts = prompts or PROMPTS_TINYSTORIES
return [
    {"prompt": prompt, "tests": etiqueta, "completion": generate_fn(prompt)}
    for prompt, etiqueta in prompts
]
```

Tres líneas. **El valor del ejercicio no está en el código**, está en tener una batería
**fija** que puedas volver a pasar cada vez que cambies algo, y comparar.

**Por qué se pasa `generate_fn` en vez del modelo.** Encapsula el modelo *y* el tokenizador,
así que la función no sabe nada de ninguno de los dos. Es el mismo patrón que `get_batch` en
el módulo 04 y `optimizer_factory` en el 13: pasar la capacidad como función en lugar de
acoplarse a un objeto concreto. Y hace el ejercicio testeable con un generador falso.

## Lo que deberías ver en la demo

**Las métricas del modelo entrenado en el módulo 13:**

```
azar (el suelo)    pérdida 4.1744    perplejidad 65.0
train              pérdida 1.3546    perplejidad  3.88
val                pérdida 1.5973    perplejidad  4.94
```

De dudar entre 65 caracteres a dudar entre 5. La brecha train/val de +0,24 es sobreajuste
incipiente y a esta escala es normal.

**Y bits por byte, que sitúa el modelo en contexto:**

| compresor | bits/byte |
|---|---|
| sin comprimir | 8,00 |
| gzip (inglés) | ~2,50 |
| **tu modelo** | **~2,30** |
| los mejores LLM | 0,60–0,80 |

Tu modelo de 0,8M parámetros entrenado 70 segundos comprime aproximadamente como gzip. No es
poco: gzip es un algoritmo muy bueno afinado durante décadas.

## Sobre la batería, y una advertencia honesta

El demo pasa la batería de TinyStories a un modelo entrenado sobre **Shakespeare a nivel de
carácter**. Los prompts en inglés moderno le quedan completamente fuera de distribución, y el
resultado se nota:

```
Once upon a time, there was a little girl named Lily. She is A my soul,
when thy should stay for thy true.  LUCENTIO: Your true
```

Empieza a copiar el prompt, y en cuanto puede se vuelve a Shakespeare. **Esto no es un fallo
del modelo: es exactamente lo que debe pasar.** Un modelo solo sabe lo que ha visto.

El ejercicio de leer las seis continuaciones y juzgarlas es el mismo con el modelo de
TinyStories, y ahí sí verás gramática correcta y coherencia local.

## Qué esperar del modelo final de 9M sobre TinyStories

Para que las expectativas sean concretas:

- **Gramática correcta** la mayor parte del tiempo.
- **Coherencia local, no global.** Dos o tres frases seguidas tienen sentido; una historia de
  diez, probablemente no.
- **Vocabulario limitado**, que es lo buscado: TinyStories está escrito a propósito con
  vocabulario de niño de 4 años.
- **Nada de razonamiento.** Ni aritmética, ni conocimiento del mundo, ni seguir instrucciones.

Si tu modelo hace eso, ha funcionado. La distancia con un asistente no es de entrenamiento:
son tres o cuatro órdenes de magnitud en parámetros y datos, más todo el post-entrenamiento
del módulo 16.

---

## El código completo

Si te has atascado, aquí está la implementación entera. **Cópiala, pégala y ejecuta los
tests**: verlos pasar con código que entiendes es mejor que quedarte bloqueado.

Y después vuelve al ejercicio y escríbela tú. Leer una solución que ya has peleado funciona
muy bien; leerla en frío, no funciona nada.

```python
def perplexity_from_loss(loss: float) -> float:
    if not math.isfinite(loss):
        return float("inf")
    return math.exp(loss)


def bits_per_byte(total_loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    if n_bytes <= 0:
        raise ValueError("n_bytes tiene que ser positivo")
    return total_loss_nats / math.log(2) / n_bytes


def run_prompt_battery(
    generate_fn: Callable[[str], str],
    prompts: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    prompts = prompts or PROMPTS_TINYSTORIES
    return [
        {"prompt": prompt, "tests": etiqueta, "completion": generate_fn(prompt)}
        for prompt, etiqueta in prompts
    ]
```

Los imports que hacen falta ya están en el `exercises.py` del módulo, salvo los que
aparezcan arriba del bloque.

---

## The complete code

If you got stuck, here is the whole implementation. **Copy it, paste it and run the
tests**: seeing them pass with code you understand beats staying blocked.

And then go back to the exercise and write it yourself. Reading a solution you have already
wrestled with works very well; reading it cold does not work at all.

```python
def perplexity_from_loss(loss: float) -> float:
    if not math.isfinite(loss):
        return float("inf")
    return math.exp(loss)


def bits_per_byte(total_loss_nats: float, n_tokens: int, n_bytes: int) -> float:
    if n_bytes <= 0:
        raise ValueError("n_bytes has to be positive")
    return total_loss_nats / math.log(2) / n_bytes


def run_prompt_battery(
    generate_fn: Callable[[str], str],
    prompts: Sequence[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    prompts = prompts or PROMPTS_TINYSTORIES
    return [
        {"prompt": prompt, "tests": label, "completion": generate_fn(prompt)}
        for prompt, label in prompts
    ]
```

The imports you need are already in the module's `exercises.py`, except for any that appear
at the top of the block.
