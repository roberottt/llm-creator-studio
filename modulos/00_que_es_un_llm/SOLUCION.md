# 00 — Solución comentada

## Ejercicio 1 — `next_token_probs`

Una línea de trabajo real: sumar y dividir.

```
total = suma de todos los conteos
devolver {token: conteo/total para cada par}
```

**Por qué se comprueba que el total no sea cero.** Si la tabla viene vacía, `sum()` da 0 y
la división revienta con `ZeroDivisionError`. El problema no es que reviente: es *dónde*
revienta. Sin la comprobación, el error salta dentro de una comprensión de diccionario,
tres niveles por debajo de donde está la causa real, y el mensaje no menciona en ningún
sitio que el problema es una tabla vacía. Lanzar un `ValueError` con un mensaje claro
convierte media hora de depuración en cinco segundos. Esto no es manía de estilo: es la
diferencia entre un error que te ayuda y uno que te estorba.

**Por qué importa el orden del diccionario.** En Python 3.7+ los diccionarios conservan el
orden de inserción. Si construyes el resultado recorriendo `counts.items()`, el orden se
mantiene. Si lo ordenases alfabéticamente, tu ruleta del ejercicio 2 repartiría los trozos
de la recta [0,1] de otra forma, y con la misma semilla saldría un texto distinto al de la
referencia. El test `test_conserva_las_claves_y_su_orden` está ahí exactamente por eso.

**Conexión con lo que viene.** Esto es normalizar. En el módulo 06 verás `softmax`, que
hace lo mismo pero exponenciando primero:

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

¿Por qué exponenciar? Porque una red neuronal escupe números cualquiera, positivos y
negativos, y no puedes normalizar `[-2.1, 0.5, 3.0]` dividiendo entre su suma: saldrían
probabilidades negativas. La exponencial convierte cualquier número real en uno positivo
sin cambiar el orden. Aquí no hace falta porque los conteos ya son positivos.

## Ejercicio 2 — `sample_next_token`

```
r = rng.random()          # un float en [0, 1)
acumulado = 0
para cada (token, p) en probs:
    acumulado += p
    si r < acumulado:
        devolver token
devolver el último token vista
```

**El error más común es usar `<=` en lugar de `<`, o al revés.** Piénsalo con
`{'a': 0.5, 'b': 0.5}` y `r = 0.5` exacto. Con `r < acumulado`: tras `'a'` el acumulado es
0.5, y `0.5 < 0.5` es falso, así que sigue y devuelve `'b'`. Es lo correcto: `'a'` ocupa
el intervalo $[0, 0.5)$ y `'b'` el $[0.5, 1)$. Como `rng.random()` devuelve un número en
$[0, 1)$ —el 1 nunca sale, el 0 sí— este reparto es el que da exactamente 50/50.

**El `return` final no es paranoia.** Los floats no suman exacto. Prueba
`sum([0.1] * 10)` en un intérprete: da `0.9999999999999999`, no `1.0`. Si
`rng.random()` devuelve `0.99999999999999995`, el bucle termina sin haber devuelto nada y
la función devuelve `None`. Eso rompe el ejercicio 3 con un error incomprensible varios
pasos después. El test `test_nunca_devuelve_none_aunque_las_probabilidades_no_sumen_exacto`
reproduce el caso exacto.

**Alternativa que también vale.** `random.choices(list(probs), weights=list(probs.values()))`
hace lo mismo en una línea. Se pide a mano porque el objetivo es que entiendas el mecanismo:
en el módulo 14 vas a manipular esta ruleta directamente (recortarla con top-k, estirarla
o comprimirla con la temperatura) y para eso hay que saber qué hay dentro.

## Ejercicio 3 — `generate_naive`

```
salida = lista con los caracteres de start
repetir (length - len(start)) veces:
    contexto = últimos len(start) caracteres de salida
    counts   = table.get(contexto)
    si counts es None o está vacío: parar
    salida.append(sample_next_token(next_token_probs(counts), rng))
devolver "".join(salida)
```

**Lo importante de este ejercicio no es el código, es lo que representa.** Este bucle es,
literalmente, el mismo que ejecuta ChatGPT. La única diferencia con el módulo 14 es de
dónde salen las probabilidades: aquí de `table.get(contexto)`, allí de un forward de la
red. La estructura —mirar contexto, obtener distribución, muestrear, añadir, repetir— es
idéntica.

**El `break` es el punto pedagógico.** Cuando llegas a un contexto que no estaba en el
texto de entrenamiento, un modelo por conteo se queda literalmente mudo: no tiene ninguna
fila que consultar. Una red neuronal *nunca* tiene este problema, porque no consulta una
tabla: calcula. Le des lo que le des, produce una distribución. Puede ser mala, pero
existe. Esa es una de las razones profundas por las que se usan redes.

**Trabajar con lista y unir al final.** Ir haciendo `salida = salida + caracter` con
cadenas crea una cadena nueva en cada vuelta. Para 200 caracteres da igual; para los
500 millones de tokens del módulo 13 no. La costumbre de acumular en lista y hacer
`"".join()` al final es gratis y siempre correcta.

**El detalle del `length`.** Cuenta el total devuelto, incluyendo `start`. Si `start` tiene
2 caracteres y pides 5, generas 3. Por eso el bucle itera `length - len(start)` veces y no
`length`. El test lo comprueba porque es un off-by-one que se cuela solo.

## Lo que deberías ver al ejecutar la demo

Sobre Shakespeare, el porcentaje de palabras generadas que existen de verdad:

| contexto | palabras reales | pinta |
|---|---|---|
| 1 carácter | ~14% | `Wieisiopthote hashe hon ghou` |
| 2 caracteres | ~45% | `Fin tis fall mounto degiver he of` |
| 3 caracteres | ~62% | `First perange is ther, rumous the had to` |
| 4 caracteres | ~91% | `First Camiliar, And hear'd his now him in his way` |
| 6 caracteres | ~98% | `The senator: No more spices of my colour half way` |

Con 6 caracteres de contexto el texto ya parece Shakespeare de lejos. **Y sin embargo esto
no es el camino.** Mira la segunda tabla de la demo: con contexto de 6, el corpus cubre el
0,00038% de las combinaciones posibles. El modelo funciona por memorización pura, y lo que
está haciendo con 283.313 contextos es prácticamente copiar trozos literales.

Ahí está el argumento del curso entero. Contar da resultados decentes deprisa y se estrella
contra un muro exponencial. Aprender representaciones cuesta más, pero escala.

---

## El código completo

Si te has atascado, aquí está la implementación entera. **Cópiala, pégala y ejecuta los
tests**: verlos pasar con código que entiendes es mejor que quedarte bloqueado.

Y después vuelve al ejercicio y escríbela tú. Leer una solución que ya has peleado funciona
muy bien; leerla en frío, no funciona nada.

```python
def next_token_probs(counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        raise ValueError("no se puede normalizar una tabla de conteos vacia")
    return {token: count / total for token, count in counts.items()}


def sample_next_token(probs: Mapping[str, float], rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    r = rng.random()
    acumulado = 0.0
    ultimo = ""
    for token, p in probs.items():
        acumulado += p
        ultimo = token
        if r < acumulado:
            return token
    # Solo se llega aqui por error de redondeo en coma flotante (acumulado = 0.9999...).
    return ultimo


def generate_naive(
    table: dict[str, dict[str, int]],
    start: str,
    length: int = 200,
    rng: random.Random | None = None,
) -> str:
    rng = rng or random.Random()
    context_size = len(start)
    salida = list(start)

    for _ in range(max(0, length - len(start))):
        contexto = "".join(salida[-context_size:])
        counts = table.get(contexto)
        if not counts:
            break  # contexto desconocido: el modelo no sabe seguir
        salida.append(sample_next_token(next_token_probs(counts), rng))

    return "".join(salida)
```

Los imports que hacen falta ya están en el `ejercicios.py` del módulo, salvo los que
aparezcan arriba del bloque.
