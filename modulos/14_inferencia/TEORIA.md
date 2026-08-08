# 14 — Inferencia: sacarle texto al modelo, y sacárselo rápido

## Por qué importa este módulo

**Porque un modelo entrenado no sirve de nada si no sabes sacarle texto.**

Y sacar texto tiene más miga de la que parece. Si eliges siempre el token más probable —que es
lo obvio— el modelo se mete en bucles: *"the cat sat on the mat. the cat sat on the mat."* La
demo lo enseña. Resulta que **el texto humano no maximiza la probabilidad**, y entender eso es
la mitad del módulo.

La otra mitad es velocidad. La generación ingenua recalcula todo el contexto en cada token, lo
que hace que generar N tokens cueste N². La KV cache lo arregla, y es la optimización más
importante que existe en inferencia: sin ella, ningún chatbot sería usable.

### Qué sabrás al terminar

- Por qué coger siempre lo más probable produce texto malo
- Qué hacen la temperatura, top-k y top-p, **y en qué orden se aplican**, que es lo que casi
  nunca se dice
- Cómo generar N veces más rápido sin cambiar ni un token de la salida
- Por qué los modelos con contexto muy largo consumen tanta memoria en inferencia

### Qué vas a escribir

Cinco ejercicios, y esta teoría los sigue en orden. Los tres primeros son filtros sobre logits,
independientes entre sí; los dos últimos son la cache y encadenan:

| Ejercicio | Qué hace |
|---|---|
| 1. `apply_repetition_penalty` | Romper los bucles |
| 2. `top_k_filter` | Quedarse con los k mejores |
| 3. `top_p_filter` | Quedarse con los que suman p |
| 4. `KVCache` | Guardar lo ya calculado |
| 5. `generate_with_cache` | El bucle que junta todo |

Los tres primeros son cortos y cada uno cabe en cuatro o cinco líneas. **El 5 es donde está la
dificultad**, y tiene una comprobación implacable: con la cache tiene que salir *exactamente* el
mismo texto que sin ella. No parecido: idéntico, token a token.

### Cuánto cuesta

3 horas.

---

## Parte 1: cómo elegir el siguiente token

El modelo te da 4096 números, uno por token del vocabulario. ¿Cuál eliges?

### Greedy: siempre el más probable

Lo obvio, y funciona mal. Es determinista —con el mismo prompt siempre sale exactamente lo
mismo— y sobre todo **se mete en bucles**:

```
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
```

La razón es sutil y la explica bien Holtzman et al. (2020): el texto humano **no maximiza la
probabilidad**. Una persona escribe cosas sorprendentes de vez en cuando; siempre elegir lo más
probable produce texto plano y repetitivo, aunque cada token individual sea plausible.

La demo lo mide con la variedad de 4-gramas del texto generado: greedy solo saca 91%, y las
variantes con muestreo llegan al 99-100%.

### Temperatura: aplanar o afilar la distribución

Se dividen los logits por un número antes del softmax:

$$P_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

Con los logits `[3, 2, 1, 0.5]` de la demo:

| T | tok 0 | tok 1 | tok 2 | tok 3 | efecto |
|---|---|---|---|---|---|
| 0,5 | 0,862 | 0,117 | 0,016 | 0,006 | afilada: casi siempre el primero |
| 1,0 | 0,623 | 0,229 | 0,084 | 0,051 | la distribución tal cual |
| 2,0 | 0,409 | 0,248 | 0,150 | 0,117 | plana: más variedad |

Dividir por un número pequeño **separa** los logits, y como el softmax es exponencial esa
separación se amplifica. Dividir por uno grande los **junta**. Con $T \to 0$ se recupera greedy.

Valores típicos: 0,7–0,9 para texto coherente, 1,0 para variedad, por encima de 1,2 empieza a
desvariar.

---

## Ejercicio 1: la penalización de repetición (`apply_repetition_penalty`)

Un parche directo contra los bucles: bajar el logit de los tokens que ya han salido.

Aquí hay un detalle que casi todo el mundo implementa mal, y es todo el ejercicio. Hay que
**dividir si el logit es positivo y multiplicar si es negativo**:

```
   token 0, logit +3.0  ->  3.0 / 1.1 = +1.50    lo acerca a cero
   token 5, logit -3.0  -> -3.0 * 1.1 = -6.00    lo aleja de cero, hacia abajo
```

(Los números son los de la demo, con `penalty = 2.0`.)

Si dividieras siempre, el −3,0 pasaría a −1,5 y el token se volvería **más** probable: justo lo
contrario de penalizarlo. Y no es un caso raro: en un vocabulario de 4096, la mayoría de los
logits son negativos casi siempre, así que dividir sin más te penalizaría bien unos pocos tokens
y premiaría a miles.

Lo bueno de este filtro es que **rescata a greedy sin quitarle el determinismo**: sigue siendo
reproducible, pero ya no se atasca. En la demo, greedy con penalización sube su variedad de
4-gramas de 91% a un texto que no repite.

---

## Ejercicio 2: top-k (`top_k_filter`)

El problema de la temperatura sola es que **nunca elimina** los tokens malos, solo los hace menos
probables. Con 4096 tokens, la cola larga puede acumular un 20% de la masa entre miles de
opciones absurdas, y de vez en cuando sale una.

Top-k lo corta en seco: ordena, se queda con los `k` mayores, pone el resto a $-\infty$.

Su defecto es que `k` es **fijo**. Si el modelo está segurísimo del siguiente token, `k=40` mete
39 alternativas malas. Si está genuinamente indeciso entre 100, corta opciones buenas. De ahí el
ejercicio siguiente.

---

## Ejercicio 3: top-p o nucleus (`top_p_filter`)

La respuesta a ese defecto. En vez de un número fijo, se acumula probabilidad hasta llegar a `p`
y se corta ahí. Con la tabla de la demo y `p = 0.9`:

| token | logit | prob | acumulada | top-k=2 | top-p=0,9 |
|---|---|---|---|---|---|
| 0 | +3,0 | 0,623 | 0,623 | sí | sí |
| 1 | +2,0 | 0,229 | 0,852 | sí | sí |
| 2 | +1,0 | 0,084 | 0,936 | no | **sí** |
| 3 | +0,5 | 0,051 | 0,987 | no | no |
| 4 | −1,0 | 0,011 | 0,998 | no | no |
| 5 | −3,0 | 0,002 | 1,000 | no | no |

**Fíjate en el token 2, el que cruza el umbral: entra.** La definición de Holtzman es *"el
conjunto más pequeño cuya probabilidad acumulada **excede** p"*, y `[0.623, 0.229]` suma 0,852,
que no excede 0,9. Hace falta el tercero.

Es un off-by-one facilísimo de equivocar, y por eso en el código la comparación se hace sobre la
acumulada **antes** de incluir cada token, no después.

Y compara las dos últimas columnas de la tabla: con estos logits, top-k=2 deja 2 candidatos y
top-p deja 3. Si la distribución fuera `[0.2, 0.2, 0.2, 0.2, 0.2]`, top-k seguiría dejando 2 y
top-p dejaría los 5. **El número de candidatos se adapta a lo seguro que esté el modelo**, y eso
es exactamente lo que quieres.

**Un detalle de implementación:** el token más probable siempre se conserva, aunque él solo ya
supere `p`. Si no, con `p=0.5` y un token de probabilidad 0,9 te quedarías sin candidatos.

### El orden en que se aplican, que casi nunca se dice

Los tres filtros y la temperatura se combinan, y **el orden importa**. En el ejercicio 5 lo
escribes así:

```
   penalización  ->  temperatura  ->  top-k  ->  top-p  ->  muestrear
```

- La **penalización va primero** porque opera sobre los logits crudos: su regla de
  dividir-o-multiplicar depende del signo, y la temperatura no cambia los signos pero sí las
  magnitudes.
- La **temperatura va antes que los filtros** porque cambia las probabilidades, y top-p mira
  probabilidades acumuladas. Aplicar top-p antes de la temperatura filtraría con una
  distribución distinta de la que luego se usa para muestrear.

---

## Ejercicio 4: la KV cache (`KVCache`)

Ahora la parte de velocidad, y es donde está la ganancia grande.

**El problema.** Al generar el token 100, la versión ingenua pasa los 100 tokens por el modelo.
Otra vez. Aunque los 99 primeros no han cambiado nada. Generar N tokens cuesta $O(N^2)$ cuando
debería costar $O(N)$.

**La solución.** Guardar las claves y valores de cada capa. En cada paso, procesar **solo el
token nuevo** y concatenar sus K y V a lo guardado.

Lo que **no** se puede cachear son las queries: cada token nuevo necesita su propia pregunta. Lo
que se reutiliza son las etiquetas (K) y los contenidos (V) de los anteriores. De ahí el nombre,
y ahí es donde se cobra la separación en tres proyecciones del módulo 06.

El ejercicio en sí es una estructura de datos sencilla: una lista de tensores por capa, un método
para añadir y otro para saber cuánta memoria ocupa. La dificultad viene en el siguiente.

---

## Ejercicio 5: el bucle completo (`generate_with_cache`)

Aquí se junta todo. El bucle queda en dos fases:

1. **Prefill:** se pasa el prompt entero de golpe y se llena la cache.
2. **Decode:** en cada paso entra un solo token, se lee la cache y se añade lo nuevo.

### El detalle que rompe todo si lo olvidas

**RoPE tiene que rotar el token nuevo con el ángulo de su posición real.** Al generar el token 50
le pasas un tensor de longitud 1, y si aplicas RoPE tal cual lo rotará como si fuera la posición
0.

Por eso hace falta saber cuántos tokens hay ya en la cache:

```python
cos_t = cos[pos_offset : pos_offset + seq_len]
```

Sin ese recorte, la generación con cache produce texto distinto —y peor— que sin ella, y el bug
es difícil de localizar porque nada falla: simplemente el modelo escribe mal.

### La comprobación obligatoria

La cache tiene que dar **exactamente la misma salida**, no parecida. Con `temperature=0` la
generación es greedy y por tanto determinista, así que las dos secuencias deben coincidir token a
token:

```
   sin cache: [44, 1, 58, 46, 43, 1, 41, 53, 51, 51]
   con cache: [44, 1, 58, 46, 43, 1, 41, 53, 51, 51]
```

Si no coinciden, lo primero que hay que mirar es el `pos_offset` de RoPE. Lo segundo, la máscara
causal del prefill: en prefill entra el prompt entero, así que **sí hace falta máscara**; sólo se
puede omitir en decode, cuando entra un único token que legítimamente ve todo el pasado. Si se
omite en las dos fases, los tokens del prompt se ven entre sí hacia delante y esa fuga corrompe
las K y V que quedan guardadas en la cache — con lo que todo lo que generes después sale mal.
Ese bug estuvo en la referencia de este curso hasta que lo cazó justamente esta comprobación.

### La ganancia

Medido sobre un modelo de contexto 1024 (con el juguete de contexto 128 las tiradas largas topan
con el límite y la comparación se aplana justo donde empieza a ser interesante):

| tokens | sin cache | con cache | speedup | memoria de la cache |
|---|---|---|---|---|
| 50 | 153 ms | 129 ms | 1,18× | 232 KB |
| 100 | 381 ms | 246 ms | 1,55× | 432 KB |
| 200 | 765 ms | 496 ms | 1,54× | 832 KB |
| 400 | 1566 ms | 996 ms | 1,57× | 1632 KB |
| 800 | 3585 ms | 1990 ms | 1,80× | 3232 KB |

**El speedup crece con la longitud**, que es lo que hay que mirar: sin cache generar N tokens
cuesta $O(N^2)$ y con cache $O(N)$. Con las longitudes de un chatbot real la diferencia deja de
ser 1,8× y pasa a ser de órdenes de magnitud.

### La memoria

$$\text{memoria KV} = 2 \times n_{\text{layers}} \times T \times d_{\text{model}} \times \text{bytes}$$

Nuestro modelo con 512 tokens en fp16: **3,9 MB**. Nada. Un modelo de 70B con contexto de 100.000
tokens: decenas de gigabytes, más que los propios pesos. Por eso existen técnicas como
*grouped-query attention*, que comparten K y V entre varias cabezas.

### Una limitación que conviene conocer

La generación con cache **para** al llegar al contexto máximo, en vez de recortar como hace la
versión ingenua.

No es pereza. Recortar con cache exigiría descartar las entradas antiguas **y remapear las
posiciones de RoPE** de todo lo que queda, porque los tokens supervivientes pasarían a ocupar
posiciones distintas de aquellas con las que se rotaron. Eso se llama *sliding window attention*
y da para un módulo entero. Parar es lo honesto: la alternativa silenciosa sería generar texto
incorrecto sin avisar.

---

## Dónde está el debate

**Nadie sabe cuáles son los parámetros de muestreo correctos.** Los valores que se usan
—temperatura 0,8, top-p 0,9— son folclore heredado, ajustados a ojo sobre modelos concretos. No
hay teoría que los derive, y el óptimo depende del modelo, de la tarea y de a quién le preguntes.

Más de fondo: **por qué el texto humano no maximiza la probabilidad** sigue sin explicación
satisfactoria. Holtzman et al. lo documentaron empíricamente y propusieron top-p como remedio,
pero la pregunta de fondo —qué distribución genera realmente el lenguaje humano y por qué
maximizar verosimilitud se aleja de ella— está abierta.

Y hay una discusión práctica en curso sobre si el muestreo debería sustituirse por algo mejor. Se
han propuesto alternativas (typical sampling, mirostat, min-p) con argumentos razonables, y
ninguna ha desplazado a top-p. Puede que porque no sean mejores, o puede que por inercia.

---

**Para ampliar:** Holtzman et al. 2020,
[The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (top-p) ·
Fan et al. 2018, [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833)
(top-k). Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
