# 13 — Solución comentada

## Ejercicio 1 — `overfit_single_batch`

```python
factory = optimizer_factory or (lambda params: torch.optim.AdamW(params, lr=lr))
opt = factory(model.parameters())

model.train()
historial = []
for _ in range(steps):
    _, perdida = model(x, y)
    opt.zero_grad(set_to_none=True)
    perdida.backward()
    opt.step()
    historial.append(float(perdida.detach()))

return historial
```

El bucle más simple posible: sin scheduler, sin acumulación, sin AMP. **A propósito**: cuantas
menos piezas, menos sitios donde pueda esconderse un bug.

**El `model.train()` no es decorativo.** Si el modelo venía en modo `eval`, el dropout estaría
desactivado y no estarías probando el mismo camino de código que usará el entrenamiento real.
Hay un test que lo comprueba.

**`float(perdida.detach())`** y no `float(perdida)`: sin el detach, PyTorch lanza un warning
sobre convertir tensores con gradiente a escalares. Funciona, pero ensucia la salida.

### Por qué este ejercicio es el más útil del módulo

En el demo, sobre el modelo de verdad:

```
paso   0:  4.1856   ← ln(65) = 4.1744, correcto
paso  10:  3.4090
paso 100:  0.4363
paso 299:  0.0173   ← memorizado
```

Un modelo con 800.000 parámetros memoriza cuatro secuencias de 128 caracteres sin
despeinarse. **Si no lo consigue, hay un bug**, y lo sabes en 30 segundos.

Y el aviso que va con ello: si bajara a cero en cinco pasos, sospecha de una fuga de
información. Revisa que los targets vayan desplazados un token respecto a la entrada — el
mismo bug que cometí escribiendo los tests del módulo 10.

## Ejercicio 2 — `format_eta`

```python
if not math.isfinite(seconds) or seconds < 0:
    return "?"

segundos = int(seconds)
if segundos < 60:
    return f"{segundos}s"
if segundos < 3600:
    return f"{segundos // 60}m {segundos % 60}s"
if segundos < 86400:
    return f"{segundos // 3600}h {(segundos % 3600) // 60}m"
return f"{segundos // 86400}d {(segundos % 86400) // 3600}h"
```

**A partir de una hora se dejan de mostrar los segundos.** Cuando faltan dos horas, los
segundos son ruido: `2h 1m` se lee de un vistazo y `2h 1m 5s` no aporta nada.

**Los valores no finitos devuelven `"?"`.** Es lo honesto cuando todavía no hay datos
suficientes para estimar, y evita imprimir cosas como `-1s` o `infd 0h`. El
`math.isfinite()` cubre `inf`, `-inf` y `nan` de una vez.

Parece un ejercicio cosmético y no lo es: vas a mirar ese número muchas veces durante una
tirada de horas.

## La tirada de verdad

Con todo implementado:

```bash
uv run python -m llmfs train --config tiny_char
```

En este hardware (MPS) son unos 70 segundos para 1.500 pasos. En la RTX 2060 debería ir
parecido o algo más rápido.

### Lo que deberías ver

**La pérdida del paso 0 frente a `ln(V)`.** El entrenador lo comprueba solo y lo pinta en
verde o en rojo:

```
perdida inicial: 4.2325  (ln(65) = 4.1744, desvio +0.0581)
```

**La curva.** Baja deprisa al principio y se aplana; la de validación sigue a la de
entrenamiento con una brecha que crece despacio. Eso es sobreajuste incipiente y es normal.

**Y las muestras**, que son la parte que de verdad enseña algo:

```
paso 0     kUU$sbpKKMMbbbPcxfffffTjjfNLL --TJ??333OOqIw
paso 300   MAPCHASTING Yrace not be town, bunders. CAMILLY: Mare striset mist
paso 600   Which begane of schame a loved, this show as friar, But there appos
paso 1500  KING RICHARD III: That's such heaven dull sented braw and starm
```

Ruido puro → palabras reconocibles → estructura de frase → formato de obra de teatro con
nombres de personaje. **Ese fichero leído de arriba abajo es el modelo aprendiendo a
escribir**, y es más informativo que la curva de pérdida.

### Antes de lanzar la tirada larga

Dos cosas, en este orden:

1. **El overfit a un batch.** 30 segundos.
2. **`--max-steps 100`** para medir el ritmo real y ver el ETA. Si dice 40 horas cuando
   esperabas 4, algo va mal y más vale saberlo antes de dejarlo toda la noche.

## Sobre reanudar

El checkpoint guarda los pesos, **el estado del optimizador**, el del GradScaler y el número
de paso. Si reanudaras solo con los pesos, Adam arrancaría con sus momentos a cero y el
modelo pegaría un bandazo: se ve como un pico en la curva, exactamente en el punto donde
reanudaste.

Y el detalle de implementación: se escribe en un fichero temporal y se renombra al final. Si
el proceso muere a mitad de la escritura, el checkpoint anterior sigue intacto. **Un
checkpoint a medias es peor que no tener checkpoint.**

Pruébalo: interrumpe con Ctrl+C y reanuda con `--resume`. El entrenador guarda antes de
salir.
