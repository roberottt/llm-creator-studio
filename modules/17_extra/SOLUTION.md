# 17 — Solución comentada

## Ejercicio 1 — `quantize_int8_symmetric`

```python
if per_channel and weight.dim() >= 2:
    max_abs = weight.abs().amax(dim=-1, keepdim=True)
else:
    max_abs = weight.abs().amax()

escala = (max_abs / 127.0).clamp_min(1e-12)
cuantizado = torch.round(weight / escala).clamp(-127, 127).to(torch.int8)
return cuantizado, escala
```

**El `clamp_min(1e-12)`** evita dividir por cero si una fila es todo ceros. Pasa más de lo
que uno espera con matrices dispersas.

**El `clamp(-127, 127)`** protege del redondeo en el borde. Sin él, un valor justo en el
máximo podría redondear a 128, que no cabe en `int8` y haría *wrap around* a −128: el peso
más grande se convertiría en el más negativo. Silencioso y devastador.

**Por qué 127 y no 128.** `int8` va de −128 a 127. Usando 127 el rango queda simétrico y el
cero se representa exactamente. En una matriz con muchos valores pequeños, que el cero sea
exacto evita un sesgo sistemático que se acumularía capa tras capa.

## Ejercicio 2 — `dequantize_int8`

```python
return quantized.to(torch.float32) * scale
```

**El `.to(torch.float32)` va antes de multiplicar.** Si multiplicaras el `int8` directamente,
PyTorch haría la operación en enteros y el resultado sería basura.

## Ejercicio 3 — `quantization_error`

```python
q, escala = quantize_int8_symmetric(original, per_channel=per_channel)
recuperado = dequantize_int8(q, escala)
error = (original - recuperado).abs()

return {
    "relative_error": float(error.norm()) / max(float(original.norm()), 1e-12),
    "max_error": float(error.max()),
    "mean_error": float(error.mean()),
    "compression": original.element_size() / q.element_size(),
    "original_bytes": original.numel() * original.element_size(),
    "quantized_bytes": q.numel() * q.element_size() + escala.numel() * escala.element_size(),
}
```

**`element_size()`** da los bytes por elemento (4 para float32, 1 para int8). Con eso la
compresión sale sola, sin números mágicos.

**Los bytes cuantizados incluyen las escalas.** Con una escala por fila son despreciables,
pero contarlas es lo honesto. Hay un test que lo comprueba.

**El error relativo es la métrica que conviene mirar**, porque es independiente de la escala
de los datos: puedes comparar capas distintas. El test
`test_el_error_relativo_es_independiente_de_la_escala` multiplica los pesos por 1000 y
verifica que no cambia.

## Lo que deberías ver en la demo

**El ejemplo a mano:**

| original | int8 | recuperado | error |
|---|---|---|---|
| +0,1200 | 34 | +0,1205 | 0,0005 |
| **−0,4500** | **−127** | **−0,4500** | **0,0000** |
| +0,0300 | 8 | +0,0283 | 0,0017 |
| +0,2800 | 79 | +0,2799 | 0,0001 |

El −0,45 se recupera **exacto** porque es el máximo y se mapea justo a −127. Los demás
pierden hasta media unidad de escala.

**Sobre el modelo real:**

| matriz | por canal | por tensor |
|---|---|---|
| token_embedding | 0,711% | 1,108% |
| q_proj | 0,714% | 1,067% |
| down_proj | 0,779% | 1,116% |

**Por canal siempre gana**, y por un margen consistente: una sola fila con valores grandes no
arrastra a las demás. Cuesta un vector de escalas más, que es despreciable.

Y el resultado que importa: **35,7 MB → 9,0 MB, 4× más pequeño**, con un 0,7% de error en los
pesos.

## Dos matices que se suelen omitir

**Que un error del 0,7% apenas afecte a la calidad del modelo es un hecho empírico, no un
teorema.** Nadie predijo que las redes fueran tan robustas a la cuantización; se descubrió
probando. Y no es universal: hay capas y arquitecturas donde int8 sí degrada de forma
apreciable, y por eso existen esquemas mixtos que dejan algunas capas en más precisión.

**Cuantizar los pesos no acelera nada por sí solo** si después conviertes a float para
multiplicar, que es lo que hace este ejercicio. La aceleración de verdad requiere kernels que
operen en int8 nativamente, y eso depende del hardware. Lo que sí ganas siempre es memoria, y
en una GPU con 6 GB eso puede ser la diferencia entre que el modelo quepa o no.

---

## Fin del curso

Has escrito **todas** las piezas: la atención, RoPE, SwiGLU, RMSNorm, AdamW, la KV cache, el
tokenizador BPE, el bucle de entrenamiento. Todas validadas numéricamente contra PyTorch o
contra los papers originales.

Un modelo frontier usa exactamente estas piezas. Más grandes, con muchísima más ingeniería
alrededor, con datos que nadie publica y cómputo que cuesta cien millones. Pero las mismas.

Lo que te llevas que no sale en los tutoriales: **sabes qué no se sabe**. Que SwiGLU funciona
sin explicación y su propio autor lo dice. Que Adam domina sin que nadie entienda bien por
qué. Que las leyes de escala tienen intervalos de confianza más amplios de lo que se reporta.
Que los benchmarks están contaminados. Que la interpretabilidad ha explicado unos pocos
circuitos y ni de lejos un modelo entero.

Eso es lo que distingue leer un paper con criterio de leerlo con fe.

---

## El código completo

Si te has atascado, aquí está la implementación entera. **Cópiala, pégala y ejecuta los
tests**: verlos pasar con código que entiendes es mejor que quedarte bloqueado.

Y después vuelve al ejercicio y escríbela tú. Leer una solución que ya has peleado funciona
muy bien; leerla en frío, no funciona nada.

```python
def quantize_int8_symmetric(
    weight: torch.Tensor, per_channel: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    if per_channel and weight.dim() >= 2:
        max_abs = weight.abs().amax(dim=-1, keepdim=True)
    else:
        max_abs = weight.abs().amax()

    # clamp_min evita dividir por cero si una fila es todo ceros.
    escala = (max_abs / 127.0).clamp_min(1e-12)
    cuantizado = torch.round(weight / escala).clamp(-127, 127).to(torch.int8)
    return cuantizado, escala


def dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return quantized.to(torch.float32) * scale


def quantization_error(original: torch.Tensor, per_channel: bool = True) -> dict[str, float]:
    q, escala = quantize_int8_symmetric(original, per_channel=per_channel)
    recuperado = dequantize_int8(q, escala)

    error = (original - recuperado).abs()
    norma_original = float(original.norm())

    return {
        "relative_error": float(error.norm()) / max(norma_original, 1e-12),
        "max_error": float(error.max()),
        "mean_error": float(error.mean()),
        "compression": original.element_size() / q.element_size(),
        "original_bytes": original.numel() * original.element_size(),
        "quantized_bytes": q.numel() * q.element_size() + escala.numel() * escala.element_size(),
    }
```

Los imports que hacen falta ya están en el `exercises.py` del módulo, salvo los que
aparezcan arriba del bloque.
