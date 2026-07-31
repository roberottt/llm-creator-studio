# 01 — Entorno y hardware

Antes de escribir una línea del modelo hay que contestar una pregunta muy práctica:
**¿cuánto va a tardar esto en mi ordenador?** No es curiosidad. La respuesta decide el
tamaño del modelo, cuánto texto puede ver de golpe y si el proyecto cabe en una tarde o
en una semana.

## La pregunta: ¿cómo se mide "cuánto cuesta"?

Un ordenador no tarda lo mismo en dos tareas distintas, así que necesitamos una unidad
común. Se usa el **FLOP**: una operación con números decimales (una suma o una
multiplicación). Entrenar un modelo son un montón de FLOPs, y una GPU puede hacer unos
cuantos billones por segundo.

Dos números, y una división:

```
tiempo = FLOPs totales que hay que hacer / FLOPs por segundo que da mi GPU
```

Todo este módulo va de estimar bien esos dos números.

## De dónde salen los FLOPs de una red

Casi todo lo que hace una red neuronal es **multiplicar matrices**. Vamos a contar
exactamente cuánto cuesta una.

Multiplica una matriz de 2×3 por otra de 3×2. El resultado es 2×2, o sea 4 números. Cada
uno de esos 4 números sale de emparejar 3 valores con otros 3, multiplicarlos y sumarlos:
3 multiplicaciones y 2 sumas, que redondeamos a 6 operaciones (2 por cada pareja). Total:

```
4 números de salida × 6 operaciones = 24 FLOPs
```

En general, multiplicar una matriz $m \times k$ por una $k \times n$ cuesta $2mnk$ FLOPs.

Ahora el paso que lo convierte en una regla útil. Una capa de la red guarda sus pesos en
una matriz. Si esa matriz tiene $P$ números dentro, procesar **un token** a través de ella
cuesta $2P$ FLOPs. Se entiende bien: cada peso se usa una vez, en una multiplicación y una
suma.

Con eso ya puedes estimar el forward de cualquier red: cuenta sus parámetros y multiplica
por dos.

### Y el backward

Entrenar no es solo pasar los datos hacia delante. Hay que calcular cómo ajustar cada
peso, y eso es el *backward* (módulo 02). Cuesta aproximadamente el **doble** que el
forward, porque hace dos multiplicaciones por cada una que hizo el forward: una para saber
cómo cambiar la entrada de la capa y otra para saber cómo cambiar sus pesos.

Sumando forward + backward sale el número que verás citado en todas partes:

$$C_{\text{token}} \approx 6N$$

donde $N$ son los parámetros del modelo. Seis FLOPs por parámetro y por token. Ya está.

### La atención se cuenta aparte

Hay una parte del Transformer que no encaja en la regla, porque no viene de multiplicar
por pesos sino de multiplicar tokens **entre sí**. Es la atención (módulo 06), y su coste
depende de cuántos tokens haya en la ventana:

$$C_{\text{token}} \approx 6N + 12 \cdot n_{\text{capas}} \cdot T \cdot d_{\text{model}}$$

Con nuestros números ($T=512$, 6 capas, $d_{\text{model}}=320$) el total sale **65,4
millones de FLOPs por token**, de los cuales la atención es un 18%. Con una ventana de
4096 tokens sería el 64%. Por eso los modelos con contexto muy largo son caros: ese
término crece mientras el otro se queda quieto.

### Lo que este cálculo ignora

No cuenta las normalizaciones, las activaciones ni el softmax. No es que sean gratis: es
que su coste no está en calcular, sino en **mover datos entre la memoria y el procesador**.
En un modelo pequeño como el nuestro eso puede ser una parte importante del tiempo real, y
esa diferencia entre "los FLOPs que cuento" y "los segundos que tardo" es justo lo que
mide la MFU.

## MFU: cuánto de tu GPU estás usando de verdad

$$\text{MFU} = \frac{\text{tokens/s} \times C_{\text{token}}}{\text{FLOPS pico del hardware}}$$

Si tu GPU puede hacer 50 billones de FLOPs por segundo y tú solo le estás sacando 10, tu
MFU es 0,2. **Nadie llega a 1.** Un modelo grande bien optimizado anda por 0,4-0,5. El
nuestro, de 9 millones de parámetros, se quedará en 0,1-0,2, y no es culpa tuya.

La razón es de tamaño. Una GPU tiene miles de unidades de cálculo. Para tenerlas todas
ocupadas necesita matrices grandes. Las nuestras son de 320×320, que es diminuto: la GPU
pasa más tiempo recibiendo instrucciones y esperando a la memoria que multiplicando. En la
demo verás la curva: matrices de 128 dan menos de 2 TFLOPS y matrices de 2048 dan diez
veces más, en la misma GPU y con el mismo tipo de dato.

El "pico" tampoco es un número honesto. La ficha de la RTX 2060 dice 51,6 TFLOPS, pero eso
es en el caso ideal. Por eso el ejercicio 1 te hace **medirlo** en lugar de leerlo: el
único número que sirve es el de tu máquina.

## Precisión: por qué 16 bits y no 32

Los números decimales se guardan repartiendo bits entre el *exponente* (cuán grande o
pequeño puede ser el número) y la *mantisa* (cuántas cifras significativas tiene):

| formato | exponente | mantisa | rango |
|---|---|---|---|
| fp32 | 8 bits | 23 bits | $10^{\pm 38}$ |
| fp16 | 5 bits | 10 bits | $6\times10^{-5}$ a $65504$ |
| bf16 | 8 bits | 7 bits | $10^{\pm 38}$ |

Usar 16 bits en vez de 32 ocupa la mitad de memoria y va aproximadamente el doble de
rápido. La pega está en el rango.

**fp16 tiene un rango minúsculo.** Durante el entrenamiento, los gradientes de las capas
profundas son números muy pequeños, del orden de $10^{-7}$. En fp16 eso es cero: el número
no se puede representar y se pierde. El resultado es que esas capas dejan de aprender, en
silencio y sin ningún mensaje de error.

La solución tiene nombre y es más simple de lo que parece: **`GradScaler`**. Antes de
calcular los gradientes, multiplica la pérdida por un número grande (unos 65.000). Como el
gradiente es una derivada, todos los gradientes quedan multiplicados por ese mismo número
y suben al rango representable. Justo antes de actualizar los pesos, se divide otra vez. Si
algún valor se pasa por arriba y sale infinito, se descarta ese paso y se baja el factor.

**bf16 no necesita nada de esto**, porque conserva los 8 bits de exponente de fp32 (a costa
de precisión, que en deep learning importa mucho menos que el rango).

### Tu hardware concreto

**La RTX 2060 es Turing (`sm_75`) y no tiene bf16.** Estás obligado a fp16 + GradScaler.
Además hay tres trampas que `llmfs/device.py` ya esquiva por ti:

- `torch.cuda.is_bf16_supported()` **devuelve `True` en tu GPU**, contando una emulación
  por software que es correcta y lentísima. Por eso el código mira directamente la
  *compute capability*: bf16 de verdad empieza en `sm_80`.
- **FlashAttention-2 tampoco funciona** por debajo de `sm_80`. No pasa nada:
  `F.scaled_dot_product_attention` detecta la GPU y usa otro algoritmo (*memory-efficient*)
  que sí va y que también evita el consumo de memoria del método ingenuo.
- **`torch.compile` está desactivado por defecto**, porque en Turing falla a compilar con
  bastante frecuencia y cuando compila no siempre gana.

En el MacBook (MPS) el valor por defecto es fp32. La memoria es unificada, así que no hay
un trasiego por PCIe que amortizar y fp16 gana menos que en una GPU discreta.

## Dónde está el debate

La regla del $6N$ se cita como si fuera física, y no lo es: es un modelo del coste, con
supuestos discutibles. Asume que el backward cuesta exactamente el doble que el forward,
lo cual depende de qué activaciones guardes y cuáles recalcules (con *gradient
checkpointing* el factor sube a 4). Ignora todo lo que es memory-bound. Y hay una decisión
arbitraria en la atención: como la máscara causal solo necesita medio triángulo, se podría
dividir por dos, pero la convención (nanoGPT, los papers) es no hacerlo. Nosotros
seguimos la convención para que tu MFU sea comparable con la de todo el mundo, no porque
sea más correcta.

---

**Para ampliar:** Kaplan et al. 2020,
[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (apéndice B) ·
Micikevicius et al. 2018, [Mixed Precision Training](https://arxiv.org/abs/1710.03740) ·
Chowdhery et al. 2022, [PaLM](https://arxiv.org/abs/2204.02311) (definición de MFU).
Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
