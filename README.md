# LLM desde cero

> **IMPORTANTE — ¿prefieres el curso en inglés?**
> La rama [`english`](https://github.com/roberottt/llm-creator-studio/tree/english) contiene
> exactamente el mismo curso, íntegramente traducido al inglés: teoría, ejercicios,
> soluciones, pistas y mensajes de la CLI. Mismo temario, mismos 18 módulos, mismos 62
> ejercicios.
>
> ```bash
> git checkout english
> ```
>
> *Looking for the English version? Everything in this course is also available, fully
> translated, on the [`english`](https://github.com/roberottt/llm-creator-studio/tree/english)
> branch.*

Un **curso-repositorio** para construir un GPT de 8.933.440 parámetros programando en
PyTorch, y entrenarlo en tu propio hardware hasta que escriba historias cortas coherentes.

No es un tutorial para leer. Abres el repo en VSCode, lees la teoría, implementas funciones
marcadas con `NotImplementedError`, y ejecutas tests hasta que pasan.

```bash
make install
uv run python -m llmfs next
```

## Cómo funciona

Cada módulo tiene cinco ficheros y el bucle es siempre el mismo:

```
TEORIA.md      →  lees (10-15 min)
ejercicios.py  →  implementas
llmfs check NN →  rojo → llmfs hint NN -e 1   →  verde → siguiente
llmfs demo NN  →  ves el concepto en gráficas y números
SOLUCION.md    →  la explicación, Y el código completo para copiar si te bloqueas
```

**Si te atascas de verdad, cada `SOLUCION.md` termina con el código entero**, listo para
copiar y pegar. Un test verifica que ese código compila, usa solo nombres que tienes
disponibles, y pasa los tests del módulo. Copiarlo no es hacer trampa: hacer trampa sería
copiarlo sin haberlo intentado.

**Los tests comparan contra referencia, no contra "no peta".** Tu `MultiHeadAttention` se
valida con `torch.allclose` contra `nn.MultiheadAttention`; tu `AdamW` contra
`torch.optim.AdamW`; tu `layer_norm` contra `F.layer_norm`.

**Nunca te quedas bloqueado.** Existe `llmfs/reference/` con todo implementado. Si tu
ejercicio del módulo 6 está a medias, los módulos 7 en adelante usan la referencia
automáticamente y te avisan por stderr. Y cuando tu ejercicio está bien, **el modelo final
entrena con tu código**.

## El currículo

**18 módulos, 62 ejercicios, ~42 h de trabajo** (sin contar tiempo de GPU).

### Parte 0 — Antes de empezar

| | módulo | qué construyes | tiempo |
|---|---|---|---|
| 00 | **Qué es un LLM** | un generador de texto por conteo, sin torch | 1 h |

### Parte I — Fundamentos

| | módulo | qué construyes | tiempo |
|---|---|---|---|
| 01 | Entorno y hardware | mides los TFLOPS reales de tu GPU | 45 min |
| 02 | Autodiferenciación | un motor de autograd escalar, estilo micrograd | 3 h |
| 03 | Tokenización | BPE desde cero, vocabulario de 4096 | 4 h |
| 04 | Datos | memmap uint16 y el dataloader de ventanas | 2 h |

### Parte II — Arquitectura

| | módulo | qué construyes | tiempo |
|---|---|---|---|
| 05 | Baselines | bigrama, MLP de Bengio, cross-entropy | 2 h |
| 06 | **Self-attention** | Q/K/V, máscara causal, multi-head | 4 h |
| 07 | Normalización | LayerNorm → RMSNorm, pre-norm vs post-norm | 1,5 h |
| 08 | FFN y activaciones | GELU, SwiGLU, el factor 2/3 | 1,5 h |
| 09 | Posición y RoPE | sinusoidales → RoPE, extrapolación | 2,5 h |
| 10 | **El GPT completo** | weight tying, init escalada, **8.933.440 params** | 3 h |

### Parte III — Entrenamiento

| | módulo | qué construyes | tiempo |
|---|---|---|---|
| 11 | El bucle | AdamW desde cero, warmup+coseno, clipping | 4 h |
| 12 | Eficiencia y escalado | MFU, Chinchilla | 2 h |
| 13 | **La tirada real** | overfit a un batch, y entrenas de verdad | 1 h |

### Parte IV — Uso y evaluación

| | módulo | qué construyes | tiempo |
|---|---|---|---|
| 14 | Inferencia | temperatura, top-k, top-p, **KV cache** | 3 h |
| 15 | Evaluación | perplejidad, bits/byte, batería TinyStories | 2 h |
| 16 | Post-training | chat template, SFT, **LoRA desde cero** | 3 h |
| 17 | Extras y límites | int8, y qué te separa de un modelo frontier | 2 h |

## El modelo final

```yaml
vocab_size: 4096      n_layers: 6       d_model: 320
n_heads: 8            d_ff: 896         context_length: 512
norm: rmsnorm         pos: rope         activation: swiglu
tie_embeddings: true  dropout: 0.0
```

| componente | parámetros |
|---|---|
| embeddings (4096 × 320) | 1.310.720 |
| atención (6 × 4 × 320²) | 2.457.600 |
| SwiGLU (6 × 3 × 320 × 896) | 5.160.960 |
| RMSNorm (13 × 320) | 4.160 |
| lm_head (atada) | 0 |
| **TOTAL** | **8.933.440** |

## Tiempos medidos

Todo lo de abajo está **medido de verdad**, no estimado. En un MacBook Pro M5 (MPS):

| | tiempo |
|---|---|
| suite de tests completa | **4,6 s** |
| `llmfs demo 06` (entrena atención y saca heatmaps) | 15 s |
| `llmfs demo 13` (entrenamiento completo) | 40 s |
| **`llmfs train --config tiny_char`** (1500 pasos) | **70 s** |
| `llmfs demo 16` (SFT real) | 30 s |

El entrenamiento de `tiny_char` va a **112k tokens/s** y baja la pérdida de 3,2 a 1,60.

### La tirada final de TinyStories

En la RTX 2060 (Turing, 51,6 TFLOPS fp16), con 500M tokens:

```
FLOPs = 6 × 7,62M × 500M ≈ 2,3·10¹⁶
```

Con una MFU realista del 10-15%: **entre 2 y 5 horas**. El número real lo dará tu propia
medición en los primeros minutos — el entrenador imprime tokens/s, MFU y ETA cada pocos
pasos.

Antes de lanzarla, dos cosas: el overfit a un batch (30 s) y `--max-steps 100` para ver el
ritmo real.

## Hardware

Todo corre en **CUDA, MPS y CPU sin cambios**. La detección y la política de precisión viven
en `llmfs/device.py` y en ningún otro sitio.

**RTX 2060 (Turing, sm_75):** sin bfloat16 en hardware, así que fp16 + `GradScaler`. Ojo:
`torch.cuda.is_bf16_supported()` devuelve `True` en Turing contando emulación por software,
por eso el código mira la compute capability directamente. Sin FlashAttention-2 (pide sm_80),
pero `F.scaled_dot_product_attention` cae solo al backend *memory-efficient*.
`torch.compile` desactivado por defecto: en Turing falla a compilar con frecuencia.

**Apple Silicon (MPS):** `PYTORCH_ENABLE_MPS_FALLBACK=1` se fija antes de importar torch.
fp32 por defecto. Algunos ops caen a CPU en silencio, y esa es la causa más común de
lentitud inexplicable en Mac.

## Comandos

```bash
make install                    # uv sync --extra compare
make test                       # tu progreso (rojo hasta que implementes: es lo normal)
make test-reference             # salud del curso (siempre verde)
make test-soluciones            # comprueba que el código de las soluciones se puede copiar

llmfs status                    # tabla de progreso, calculada ejecutando los tests
llmfs next                      # qué módulo toca y qué ejercicio
llmfs check 06                  # tests del módulo 06
llmfs hint 06 -e 2              # pista progresiva (repite para más nivel)
llmfs demo 06                   # el experimento del módulo
llmfs device                    # hardware detectado
llmfs train --config tiny_char  # entrena de verdad
```

**El estado del currículo no se declara en ningún sitio**: se calcula ejecutando los tests.

## Sobre la honestidad intelectual

Cada `TEORIA.md` cierra con una sección **"Dónde está el debate"**, y no es de adorno. A lo
largo del curso vas a leer que:

- SwiGLU funciona mejor y **su propio autor escribe** que no tiene explicación.
- Adam domina sin que nadie sepa bien por qué; la justificación habitual no resiste el
  análisis.
- Los coeficientes de Chinchilla tienen intervalos de confianza mucho más amplios de lo que
  se reportó, según un reanálisis de 2024.
- La evaluación por benchmarks está contaminada, y separar "ha aprendido" de "lo ha visto"
  es técnicamente difícil.
- La normalización sola ya rescata el gradiente casi tanto como los residuales — el
  argumento habitual es verdad a medias, y el demo del módulo 07 lo mide.

Esa parte no suele aparecer en los tutoriales, y es la que más sirve para leer papers con
criterio.

## Dependencias

`torch`, `numpy`, `datasets`, `matplotlib`, `pytest`, `tqdm`, `pyyaml`, `rich`, `regex`.

`tiktoken` va en el extra `[compare]` y solo se usa en la comparativa del módulo 03.
**Nada de `transformers` ni HuggingFace para el modelo**: `datasets` solo descarga
TinyStories.

## Empezar

```bash
make install
uv run python -m llmfs next
```

El módulo 00 no tiene torch, ni matrices, ni derivadas: construyes un generador de texto con
diccionarios y una división. Y ves el bucle autorregresivo funcionando antes de saber qué es
un transformer.

## Licencia

MIT. Ver [LICENSE](LICENSE).
