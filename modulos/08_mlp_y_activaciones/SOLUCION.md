# 08 — Solución comentada

## Ejercicio 1 — `gelu`

```python
return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
```

Una línea, transcribiendo la fórmula tal cual. Los dos errores posibles son teclear mal una
constante (`sqrt(2/pi) ≈ 0.7978`) o reagrupar la expresión de forma que cambie el orden de
operaciones.

**Por qué la aproximación y no la exacta.** La definición real es $x \cdot \Phi(x)$, con
$\Phi$ la acumulada de la normal, que se calcula con `erf`. En 2016 `erf` era lento en GPU,
así que Hendrycks y Gimpel propusieron esta aproximación con tanh. Hoy la diferencia de
velocidad es irrelevante, pero GPT-2 se entrenó con ella y por compatibilidad histórica se
sigue usando. Tu resultado tiene que coincidir con `F.gelu(x, approximate="tanh")`, no con
`F.gelu(x)` a secas — son funciones distintas y el test compara contra la primera.

**Lo que hay que llevarse del ejercicio no es la fórmula, es la derivada.** El demo lo
tabula:

| x | ReLU | dReLU/dx | GELU | dGELU/dx |
|---|---|---|---|---|
| −3,0 | 0,0000 | **0,0000** | −0,0036 | −0,0119 |
| −1,0 | 0,0000 | **0,0000** | −0,1588 | −0,0833 |

Con ReLU, la derivada en toda la zona negativa es **cero exacto**. Una neurona que acabe
dando siempre valores negativos deja de recibir gradiente para siempre: está muerta y no hay
forma de resucitarla. GELU tiene derivada pequeña pero no nula, así que puede volver.

## Ejercicio 2 — `swiglu_hidden_dim`

```python
hidden = int(2 * (4 * d_model) / 3)
if ffn_dim_multiplier is not None:
    hidden = int(ffn_dim_multiplier * hidden)
return multiple_of * ((hidden + multiple_of - 1) // multiple_of)
```

**El redondeo hacia arriba sin `math.ceil`.** Sumar `multiple_of - 1` antes de la división
entera fuerza el redondeo hacia arriba, y si el valor ya era múltiplo exacto no lo cambia. Es
el idioma estándar para esto y evita meter floats donde no hacen falta.

Compruébalo con los dos casos del curso:

```
d_model = 320:  int(2·1280/3) = 853  ->  64 · ((853+63)//64) = 64 · 14 = 896   ✓
d_model = 128:  int(2·512/3)  = 341  ->  64 · ((341+63)//64) = 64 ·  6 = 384   ✓
```

**De dónde sale el 2/3**, que es lo único conceptual del ejercicio:

```
FFN clásico:  2 matrices × d × 4d           = 8d²
SwiGLU:       3 matrices × d × (2/3 · 4d)   = 8d²      ✓ mismo presupuesto
SwiGLU sin ajustar: 3 × d × 4d              = 12d²     ✗ un 50% más
```

El demo lo muestra con los números del modelo: FFN clásico 819.200, SwiGLU sin ajustar
1.228.800 (+50%), SwiGLU con el 2/3 → 860.160 (+5%).

Ese +5% residual, y no 0%, es por el redondeo a múltiplo de 64. **La igualdad de
presupuestos es asintótica, no exacta**: a $d = 4096$ el desvío baja al 0,2%.

**Por qué se redondea.** No es cosmética. Las dimensiones alineadas dejan que los tensor
cores usen sus rutas rápidas; una matriz de 853 columnas es más lenta que una de 896 pese a
tener menos parámetros.

## Ejercicio 3 — `SwiGLU`

```python
def __init__(self, d_model, d_ff, dropout=0.0, bias=False):
    super().__init__()
    self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
    self.up_proj = nn.Linear(d_model, d_ff, bias=bias)
    self.down_proj = nn.Linear(d_ff, d_model, bias=bias)
    self.dropout = nn.Dropout(dropout)

def forward(self, x):
    return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
```

**El `*` es multiplicación elemento a elemento**, no matricial. Las dos ramas salen con la
misma forma `(B, T, d_ff)` y se multiplican punto a punto. Si pusieras `@` las formas ni
siquiera cuadrarían.

**La activación va en `gate_proj`, no en `up_proj`.** Numéricamente el módulo funcionaría
igual de bien con la asignación invertida —es simétrico salvo por qué pesos aprenden qué—
pero **no coincidiría con la referencia al copiar pesos** y el test fallaría con una
diferencia difícil de interpretar. El test
`test_swiglu_aplica_la_activacion_a_la_rama_gate` está para señalarlo directamente.

**`F.silu` es Swish.** $\text{Swish}(z) = z \cdot \sigma(z)$. Puedes escribirlo a mano
(`x * torch.sigmoid(x)`) y da lo mismo, pero `F.silu` tiene un kernel fusionado.

**Sin sesgos por defecto.** `bias=False` es la config del modelo final, y es lo que hace que
el conteo dé exactamente $3 \cdot d \cdot d_{ff}$. Los LLM modernos han ido eliminando los
sesgos: aportan poco y complican el weight decay del módulo 11.

## Lo que deberías ver en la demo

**El colapso lineal**, que es el argumento entero del módulo:

```
5 capas apiladas  vs  1 sola matriz  ->  diferencia máxima: 2.38e-07   (o sea, cero)
las mismas 5 capas CON GELU          ->  diferencia: 4.7
```

Cinco capas lineales sin activación **son** una capa. Como la atención es una media
ponderada —lineal—, el FFN es literalmente lo único que impide que el Transformer entero se
derrumbe a una sola multiplicación de matrices.

**El reparto de parámetros:**

| d_model | atención | FFN | % FFN |
|---|---|---|---|
| 320 | 409.600 | 860.160 | **68%** |
| 4096 | 67.108.864 | 134.479.872 | **67%** |

Dos tercios del modelo son FFN. Cuando leas que un modelo tiene N parámetros, la mayoría
están aquí, no en la atención.

**La comparación SwiGLU / FFN clásico** merece una nota de método. En la primera versión de
esta demo la tarea era tan fácil que ambos llegaban a pérdida `0.00000` y el código
declaraba ganador al que tuviera el último decimal más bajo: eso es leer ruido. Ahora la
tarea es más dura, los resultados se imprimen en notación científica, y **si la diferencia
queda por debajo del 10% el propio demo dice que el experimento no distingue entre las dos
arquitecturas**.

Y aun cuando sí distinga, un experimento de juguete con una tarea inventada y una sola
semilla no demuestra nada sobre modelos de lenguaje. Shazeer (2020) probó todas las
variantes GLU entrenando transformers de verdad. Su explicación de por qué SwiGLU gana,
citada literalmente:

> *"We offer no explanation as to why these architectures seem to work; we attribute their
> success, as all else, to divine benevolence."*

Es de las decisiones de arquitectura más usadas y peor entendidas del campo, y conviene
saberlo cuando leas explicaciones que suenan muy seguras.
