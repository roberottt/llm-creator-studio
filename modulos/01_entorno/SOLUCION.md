# 01 — Solución comentada

## Ejercicio 1 — `measure_matmul_tflops`

La estructura es: reservar tensores, calentar, sincronizar, cronometrar N iteraciones,
sincronizar, dividir.

Los dos errores que se cometen aquí son siempre los mismos.

**No sincronizar.** `a @ b` en CUDA o MPS encola trabajo y devuelve el control
inmediatamente. Si haces `t0 = perf_counter(); a @ b; t1 = perf_counter()`, mides cuánto
tarda la CPU en encolar una orden — del orden de 20 µs — y obtienes cifras de miles de
TFLOPS. El test `test_la_medicion_devuelve_un_numero_plausible` acota el resultado
precisamente para cazar esto. Hay que llamar a `cfg.synchronize()` antes de cada
`perf_counter()`.

**No calentar.** La primera invocación de un tamaño concreto dispara la heurística de
selección de kernel de cuBLAS (o la compilación del shader en Metal) y reserva memoria en
el caching allocator. Puede ser dos órdenes de magnitud más lenta. Con `iters=10` y sin
warmup, esa primera llamada domina la media.

Detalle menor: no hace falta guardar el resultado del matmul, pero tampoco lo optimices a
`None` — PyTorch no elimina código muerto, así que `a @ b` a secas se ejecuta igual.

El `dtype` por defecto sale de `cfg.amp_dtype`, que en CUDA es fp16 y en MPS es `None`;
por eso el fallback es `torch.float32`.

## Ejercicio 2 — `transformer_flops_per_token`

Aritmética pura; lo interesante es entender de dónde sale cada término.

```
params_matmul = n_layers * (4·d² + n_ffn·d·d_ff) + d·V
forward       = 2·params_matmul + 4·n_layers·T·d
total         = 3·forward
```

El `2·params` es la regla de oro: *cada parámetro contribuye una multiplicación y una
suma por token*. El `4·n_layers·T·d` es la atención, y conviene tener claro que **no es un
término de parámetros**: son los productos $QK^\top$ y $\text{softmax}\cdot V$, cuyo coste
depende de cuántos tokens hay en el contexto, no de cuántos pesos tiene el modelo.

El `3×` del backward merece un comentario. El forward calcula $y = Wx$. El backward
necesita dos cosas: $\partial L/\partial x = W^\top \, \partial L/\partial y$ para seguir
propagando, y $\partial L/\partial W = \partial L/\partial y \, x^\top$ para actualizar los
pesos. Son dos matmuls del mismo tamaño que el del forward, de ahí el factor 2 adicional.

La `lm_head` se cuenta aunque `tie_embeddings` sea `true`. Atar los pesos ahorra memoria y
parámetros, no cómputo: el matmul $(B \cdot T, d) \times (d, V)$ se ejecuta igual.

Sobre no dividir por dos en la atención causal: es cierto que solo se calcula el triángulo
inferior, y con un kernel ideal costaría la mitad. En la práctica los kernels densos
calculan la matriz entera y enmascaran, y los kernels tipo Flash se saltan bloques enteros
pero no exactamente la mitad. La convención de contarlo completo viene de nanoGPT y es la
que usan los papers al reportar MFU. Si dividieras por dos, tu MFU saldría un ~9% más baja
que la de todo el mundo con el mismo hardware.

## Ejercicio 3 — `estimate_tokens_per_second`

$$\text{tokens/s} = \frac{\text{TFLOPS} \times 10^{12} \times \text{MFU}}{C_{\text{token}}}$$

La comprobación de `flops_per_token > 0` no es decorativa: es lo que impide una división
por cero silenciosa que produciría `inf` y estimaciones absurdas.

## Qué esperar al ejecutar la demo

En la RTX 2060 verás algo entre 15 y 30 TFLOPS en fp16 con matrices de 2048–4096, y
menos de 2 TFLOPS con 128×128. Esa caída es el mensaje del módulo: **el tamaño de matriz
determina si aprovechas el hardware**, y nuestro modelo de 320 dimensiones vive en la zona
mala de esa curva. Es una decisión consciente: queremos un modelo que entrene en horas, no
uno que maximice la MFU.

En el M5 con MPS los números son más planos entre dtypes, porque la memoria es unificada y
no hay tráfico PCIe que amortizar; fp16 gana menos de lo que ganaría en una GPU discreta.

---

## El código completo

Si te has atascado, aquí está la implementación entera. **Cópiala, pégala y ejecuta los
tests**: verlos pasar con código que entiendes es mejor que quedarte bloqueado.

Y después vuelve al ejercicio y escríbela tú. Leer una solución que ya has peleado funciona
muy bien; leerla en frío, no funciona nada.

```python
def matmul_flops(size: int) -> int:
    return 2 * size**3


def measure_matmul_tflops(
    cfg: DeviceConfig | None = None,
    size: int = 2048,
    dtype: torch.dtype | None = None,
    warmup: int = 3,
    iters: int = 10,
) -> float:
    cfg = cfg or get_device()
    if dtype is None:
        dtype = cfg.amp_dtype or torch.float32

    a = torch.randn(size, size, device=cfg.device, dtype=dtype)
    b = torch.randn(size, size, device=cfg.device, dtype=dtype)

    for _ in range(warmup):
        a @ b
    cfg.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    cfg.synchronize()
    elapsed = time.perf_counter() - start

    return (matmul_flops(size) * iters) / elapsed / 1e12


def transformer_flops_per_token(
    n_layers: int,
    d_model: int,
    d_ff: int,
    context_length: int,
    vocab_size: int,
    n_ffn_matrices: int = 3,
    include_backward: bool = True,
) -> int:
    params_atencion = 4 * d_model**2
    params_ffn = n_ffn_matrices * d_model * d_ff
    params_matmul = n_layers * (params_atencion + params_ffn)
    params_matmul += d_model * vocab_size  # la proyeccion final a logits

    forward = 2 * params_matmul + 4 * n_layers * context_length * d_model
    return int(3 * forward if include_backward else forward)


def estimate_tokens_per_second(tflops: float, flops_per_token: int, mfu: float = 0.4) -> float:
    if flops_per_token <= 0:
        raise ValueError("flops_per_token debe ser positivo")
    return tflops * 1e12 * mfu / flops_per_token
```

Los imports que hacen falta ya están en el `ejercicios.py` del módulo, salvo los que
aparezcan arriba del bloque.
