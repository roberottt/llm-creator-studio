# 11 — Solución comentada

## Ejercicio 1 — `AdamWScratch.step`

```python
@torch.no_grad()
def step(self, closure=None):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        beta1, beta2 = group["betas"]
        lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad

            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            t = state["step"]
            m, v = state["exp_avg"], state["exp_avg_sq"]

            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            bias_correction1 = 1 - beta1**t
            bias_correction2 = 1 - beta2**t

            denom = (v / bias_correction2).sqrt_().add_(eps)
            step_size = lr / bias_correction1

            if wd != 0.0:
                p.mul_(1 - lr * wd)
            p.addcdiv_(m, denom, value=-step_size)

    return loss
```

### Los tres errores que caza el test

**Empezar `t` en 0.** Con $t=0$, $1 - \beta^0 = 0$ y divides por cero. Hay que incrementar
`state["step"]` **antes** de usarlo.

**Olvidar la corrección de sesgo.** El test
`test_la_correccion_de_sesgo_esta_aplicada` lo mide de la forma más directa: un solo paso con
gradiente 1 y `lr=0.1` tiene que mover el parámetro exactamente 0,1. Sin corrección, con
$\beta_2 = 0{,}95$, $v = 0{,}05$ y $\sqrt{v} = 0{,}224$: el paso saldría $0{,}1/0{,}224 =
0{,}447$, **4,5 veces mayor de lo debido**. Con eso, los primeros pasos destrozan la
inicialización.

**Sumar el weight decay al gradiente.** Es la diferencia entre Adam+L2 y AdamW. El test
`test_el_weight_decay_esta_desacoplado` lo distingue con un truco elegante: pone el
**gradiente a cero** y comprueba que el parámetro sigue encogiendo. Con decay desacoplado,
`p ← p·(1 − lr·wd)` pasa de 2,0 a 1,9. Con L2, el "gradiente" sería `wd·p` y pasaría por la
división por $\sqrt{v}$, que con $v \approx 0$ da algo muy distinto.

### Sobre `p.mul_(1 - lr * wd)`

Es equivalente a `p -= lr * wd * p` pero en una sola operación. Fíjate en que **el decay se
aplica antes** de la actualización de Adam, igual que en la implementación de PyTorch. El
orden importa para que los pesos coincidan al último decimal.

### Sobre las operaciones in-place

`addcmul_(a, b, value=v)` calcula `self += v * a * b`, y `addcdiv_(a, b, value=v)` calcula
`self += v * a / b`. Son crípticas pero evitan reservar tensores intermedios. Con 8,9
millones de parámetros y 10.000 pasos, eso se nota.

Si te resultan incómodas, escríbelo con operaciones normales primero: el test compara
resultados, no estilo.

## Ejercicio 2 — `lr_at_step`

```python
min_lr = lr * min_lr_ratio

if warmup_steps > 0 and step < warmup_steps:
    return lr * (step + 1) / warmup_steps

if step >= max_steps:
    return min_lr

if schedule == "constant":
    return lr

progreso = (step - warmup_steps) / max(1, max_steps - warmup_steps)
progreso = min(1.0, max(0.0, progreso))

if schedule == "linear":
    return lr - (lr - min_lr) * progreso

coef = 0.5 * (1.0 + math.cos(math.pi * progreso))
return min_lr + (lr - min_lr) * coef
```

**El `+1` del warmup** evita que el paso 0 tenga `lr` exactamente cero. Un paso con lr=0 no
aprende nada: es un paso desperdiciado, y con warmup de 500 pasos serían 500 desperdiciados
al arrancar cada tirada.

**Comprueba los extremos del coseno** en vez de fiarte: con `progreso=0`, $\cos(0)=1$ y
`coef=1`, así que devuelve `lr`. Con `progreso=1`, $\cos(\pi)=-1$ y `coef=0`, así que
devuelve `min_lr`. Correcto.

**El orden de las guardas importa.** El `step >= max_steps` tiene que ir antes del cálculo
del coseno; si no, `progreso` saldría mayor que 1 y el coseno empezaría a *subir* otra vez.
El `min(1.0, max(0.0, ...))` es un cinturón adicional.

## Ejercicio 3 — `clip_grad_norm`

```python
grads = [p.grad for p in parameters if p.grad is not None]
if not grads:
    return 0.0

total = torch.sqrt(sum(g.detach().pow(2).sum() for g in grads))
total_f = float(total)

if max_norm > 0 and total_f > max_norm:
    factor = max_norm / (total_f + 1e-6)
    for g in grads:
        g.mul_(factor)

return total_f
```

**La norma es global**, no por tensor. Eso es lo único conceptual del ejercicio, y el test
`test_conserva_la_direccion_del_gradiente` lo comprueba de forma directa: calcula el coseno
entre el vector de gradientes antes y después del recorte, y exige que sea > 0,9999. Si
recortaras cada tensor por separado, cada uno se escalaría por un factor distinto y la
dirección conjunta cambiaría.

El demo lo mide: la norma pasa de **112.858 a 1,0000** con un coseno de **0,99999994**. Solo
cambia la magnitud.

**Devolver la norma antes de recortar** es lo que hace `torch.nn.utils.clip_grad_norm_` y es
lo útil: registrarla te avisa de que el entrenamiento se desestabiliza antes de que reviente.

## Ejercicio 4 — `build_param_groups`

```python
decay, no_decay = [], []
for param in model.parameters():
    if not param.requires_grad:
        continue
    (decay if param.dim() >= 2 else no_decay).append(param)

return [
    {"params": decay, "weight_decay": weight_decay},
    {"params": no_decay, "weight_decay": 0.0},
]
```

Cinco líneas, y la regla es sorprendentemente simple: **`param.dim() >= 2`**. Las matrices
decaen; los vectores no.

Con nuestro modelo la partición sale así:

| grupo | tensores | parámetros |
|---|---|---|
| con decay (matrices) | 43 | 8.929.280 |
| sin decay (RMSNorm) | 13 | 4.160 |

Los 4.160 son las 13 normalizaciones × 320. Son el 0,05% de los parámetros y aplicarles
decay **no daría ningún error visible**: simplemente el modelo entrenaría algo peor y solo lo
detectarías comparando dos tiradas completas.

## Un bug que cometí escribiendo estos tests

`assert modelo[0].weight not in todos`, donde `todos` es una lista de tensores, revienta con
`RuntimeError: Boolean value of Tensor with more than one value is ambiguous`.

El operador `in` usa `==`, y en tensores `==` devuelve comparación **elemento a elemento**, no
un booleano. Hay que comparar por identidad:

```python
ids = {id(p) for g in grupos for p in g["params"]}
assert id(modelo[0].weight) not in ids
```

Es un tropiezo clásico con PyTorch y merece la pena tenerlo presente: cualquier uso de
tensores en contexto booleano —`if tensor:`, `x in lista_de_tensores`, `assert tensor`— hace
algo distinto de lo que parece.

## Lo que deberías ver en la demo

**Tu AdamW frente al de PyTorch:** error máximo de $2 \times 10^{-7}$ en los pesos tras 200
pasos. No parecidos: idénticos salvo redondeo de fp32.

**El recorte ante un batch tóxico**, inyectado en el paso 50:

```
sin recortar     : la pérdida SUBE 3,0x tras el batch tóxico
con grad_clip=1.0: ni se entera (0,8x, sigue bajando)
```

En una tirada de 10.000 pasos, un solo batch raro puede costarte el entrenamiento entero.
`grad_clip=1.0` acota el daño máximo de cualquier batch.

---

## El código completo

Si te has atascado, aquí está la implementación entera. **Cópiala, pégala y ejecuta los
tests**: verlos pasar con código que entiendes es mejor que quedarte bloqueado.

Y después vuelve al ejercicio y escríbela tú. Leer una solución que ya has peleado funciona
muy bien; leerla en frío, no funciona nada.

```python
class AdamWScratch(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr no puede ser negativo: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"las betas deben estar en [0, 1): {betas}")
        if eps < 0.0:
            raise ValueError(f"eps no puede ser negativo: {eps}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                # Medias moviles, in-place para no reservar tensores nuevos cada paso.
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t

                denom = (v / bias_correction2).sqrt_().add_(eps)
                step_size = lr / bias_correction1

                # Weight decay DESACOPLADO: directamente sobre el parametro.
                if wd != 0.0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m, denom, value=-step_size)

        return loss


def lr_at_step(
    step: int,
    max_steps: int,
    lr: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
    schedule: str = "cosine",
) -> float:
    min_lr = lr * min_lr_ratio

    if warmup_steps > 0 and step < warmup_steps:
        # +1 para que el paso 0 no tenga lr exactamente cero (no aprenderia nada).
        return lr * (step + 1) / warmup_steps

    if step >= max_steps:
        return min_lr

    if schedule == "constant":
        return lr

    progreso = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progreso = min(1.0, max(0.0, progreso))

    if schedule == "linear":
        return lr - (lr - min_lr) * progreso

    coef = 0.5 * (1.0 + math.cos(math.pi * progreso))
    return min_lr + (lr - min_lr) * coef


def clip_grad_norm(parameters: Iterable[nn.Parameter], max_norm: float) -> float:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    total = torch.sqrt(sum(g.detach().pow(2).sum() for g in grads))
    total_f = float(total)

    if max_norm > 0 and total_f > max_norm:
        # 1e-6 para no dividir por cero si la norma es minuscula.
        factor = max_norm / (total_f + 1e-6)
        for g in grads:
            g.mul_(factor)

    return total_f


def build_param_groups(
    model: nn.Module, weight_decay: float = 0.1
) -> list[dict[str, Any]]:
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
```

Los imports que hacen falta ya están en el `exercises.py` del módulo, salvo los que
aparezcan arriba del bloque.
