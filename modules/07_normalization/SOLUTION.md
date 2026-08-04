# 07 — Solución comentada

## Ejercicio 1 — `layer_norm`

```python
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True, unbiased=False)
normalized = (x - mean) / torch.sqrt(var + eps)
if weight is not None:
    normalized = normalized * weight
if bias is not None:
    normalized = normalized + bias
return normalized
```

**`unbiased=False` es la trampa del ejercicio.** `torch.var` divide por $n-1$ por defecto
(varianza muestral, la corrección de Bessel). LayerNorm usa la **poblacional**, que divide
por $n$. Con $d = 320$ la diferencia es del 0,3% y podrías no notarla; con $d = 4$ es del
33% y se ve a simple vista. El test compara tu resultado contra las dos versiones y te dice
a cuál se parece más.

**`keepdim=True`.** Sin él, `mean(dim=-1)` sobre `(4, 8, 32)` devuelve `(4, 8)` en vez de
`(4, 8, 1)`, y la resta `x - mean` intenta emitir mal las dimensiones. A veces lanza error y
a veces —cuando las formas casualmente encajan— produce basura en silencio.

**El $\epsilon$ va dentro de la raíz**, no fuera:
$\sqrt{\sigma^2 + \epsilon}$, no $\sqrt{\sigma^2} + \epsilon$. Es lo que hace `F.layer_norm`
y con varianza pequeña la diferencia importa.

**Los parámetros opcionales.** Que `weight` y `bias` puedan ser `None` permite comparar la
normalización pura contra `F.layer_norm(x, (d,))` sin argumentos afines. No es un capricho:
es lo que hace que el test tenga un oráculo limpio.

## Ejercicio 2 — `RMSNorm`

```python
def __init__(self, dim, eps=1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

def forward(self, x):
    return self._norm(x.float()).type_as(x) * self.weight
```

**`torch.ones` y no `torch.randn`.** Al inicializar, la capa tiene que ser la normalización
pura. Si `weight` empezara aleatorio, estarías escalando cada dimensión por un factor
arbitrario antes de haber aprendido nada, y la pérdida del paso 0 no cuadraría con
$\ln(V)$ — el detector de bugs del módulo 05.

**`torch.rsqrt(z)` en lugar de `1/torch.sqrt(z)`.** Calcula el inverso de la raíz de una vez.
Es un kernel más y algo más estable numéricamente.

**El `.float()` no es paranoia.** Con activaciones en fp16, elevar al cuadrado desborda
antes de lo que uno espera: $300^2 = 90.000$ y fp16 se acaba en 65.504. El resultado sería
`inf`, luego la media sería `inf`, y `rsqrt(inf)` sería 0: la capa devolvería ceros. El test
`test_rmsnorm_calcula_en_fp32_y_no_desborda` reproduce exactamente ese caso.

**Un detalle que sorprende y conviene ver una vez.** Aunque hagas `.type_as(x)` para volver
a fp16, la salida acaba siendo **fp32**, porque después multiplicas por `self.weight`, que
es un parámetro fp32, y PyTorch promociona. No es un bug: es lo que hace la implementación
de Llama y es lo deseable. Bajo autocast los pesos se mantienen en fp32 y las operaciones
siguientes convierten lo que necesiten; dejar la salida de una normalización en precisión
alta es gratis y da margen numérico. Hay un test que lo documenta.

## Ejercicio 3 — `prenorm_residual`

```python
return x + fn(norm(x))
```

Una línea. Y es el ejercicio más importante del módulo.

La diferencia con post-norm —`norm(x + fn(x))`— parece de paréntesis y decide si una red
profunda entrena. Derivando pre-norm respecto a `x`:

$$\frac{\partial}{\partial x}\big(x + f(\text{norm}(x))\big) = 1 + \frac{\partial f(\text{norm}(x))}{\partial x}$$

Ese **1** llega intacto a las capas de abajo por muchas capas que haya. El test
`test_el_gradiente_llega_intacto_a_traves_del_residual` lo comprueba de la forma más
directa posible: anula por completo el gradiente de la rama y verifica que el gradiente en
la entrada sigue siendo exactamente 1.

## Lo que deberías ver en la demo, y una corrección a la narrativa habitual

El experimento apila $N$ bloques y mide la norma del gradiente que llega a la entrada:

| capas | nada | solo norma | post-norm | pre-norm |
|---|---|---|---|---|
| 4 | 3,2e-01 | 1,4e+01 | 7,1e+01 | 7,9e+01 |
| 16 | 1,1e-07 | 1,9e+01 | 6,4e+01 | 9,7e+01 |
| 64 | **0,0e+00** | 5,1e+00 | 5,9e+01 | **1,5e+02** |

Léelo con cuidado, porque hay un matiz que la explicación de manual suele saltarse.

**Sin nada**, el gradiente llega a **cero exacto** con 64 capas. No "pequeño": cero, por
underflow de la coma flotante. Las primeras capas no reciben ninguna señal.

**Solo con normalización** —sin residuales— el gradiente ya se recupera hasta 5,1. Esto es
importante: **la normalización por sí sola resuelve buena parte del problema del
desvanecimiento**, porque devuelve la escala a 1 en cada paso y corta la cadena de factores
multiplicativos.

Así que el argumento de "los residuales existen para que el gradiente no se desvanezca" es
verdad a medias. Normalización y residuales atacan el mismo problema por caminos distintos
y son complementos, no alternativas. Lo que distingue a **pre-norm** no es evitar el
desvanecimiento, sino que es la única configuración cuyo gradiente **crece** con la
profundidad en lugar de decrecer: el camino $x \to x$ no tiene ningún peaje.

## Sobre la comparación LayerNorm / RMSNorm

La demo mide también en qué se diferencian de verdad, y aquí hay una lección de metodología.

Mi primera versión comparaba las dos salidas con el **coeficiente de correlación**, y daba
0,998 con datos centrados y 0,999 con datos desplazados. O sea, la correlación *subía* al
desplazar los datos, justo lo contrario de lo esperado.

El error era de la métrica, no del resultado: **la correlación es invariante a
transformaciones afines**, así que da ~1 aunque una de las dos deje un desplazamiento que la
otra elimina. Es exactamente lo que estaba intentando medir, y era ciega a ello.

La métrica correcta es mirar la **media de la salida**: LayerNorm la deja en 0 siempre, y
RMSNorm la conserva. Con datos ya centrados —que es lo normal dentro de una red— las dos
hacen prácticamente lo mismo, y de ahí que se pueda prescindir de restar la media. Con un
desplazamiento grande divergen.

Que eso no perjudique en la práctica es un **resultado empírico**, no un teorema. Zhang y
Sennrich lo comprobaron entrenando modelos, no demostrándolo.

---

## El código completo

Si te has atascado, aquí está la implementación entera. **Cópiala, pégala y ejecuta los
tests**: verlos pasar con código que entiendes es mejor que quedarte bloqueado.

Y después vuelve al ejercicio y escríbela tú. Leer una solución que ya has peleado funciona
muy bien; leerla en frío, no funciona nada.

```python
def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(var + eps)

    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias
    return normalized


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


def prenorm_residual(
    x: torch.Tensor,
    fn: Callable[[torch.Tensor], torch.Tensor],
    norm: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    return x + fn(norm(x))
```

Los imports que hacen falta ya están en el `exercises.py` del módulo, salvo los que
aparezcan arriba del bloque.
