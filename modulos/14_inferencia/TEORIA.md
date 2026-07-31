# 14 — Inferencia y muestreo

Tienes un modelo entrenado que produce logits. Falta convertir eso en texto que alguien
quiera leer, y hacerlo deprisa.

## Parte 1: cómo elegir el siguiente token

El modelo te da 4096 números, uno por token del vocabulario. ¿Cuál eliges?

### Greedy: siempre el más probable

Lo obvio, y funciona mal. Es determinista —con el mismo prompt siempre sale exactamente lo
mismo— y sobre todo **se mete en bucles**:

```
The cat sat on the mat. The cat sat on the mat. The cat sat on the mat.
```

La razón es sutil y la explica bien Holtzman et al. (2020): el texto humano **no maximiza
la probabilidad**. Una persona escribe cosas sorprendentes de vez en cuando; siempre elegir
lo más probable produce texto plano y repetitivo, aunque cada token individual sea
plausible.

### Temperatura: aplanar o afilar la distribución

Se dividen los logits por un número antes del softmax:

$$P_i = \frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

Con números. Supón logits `[3, 2, 1]`:

```
T = 1.0   →  [0.665, 0.245, 0.090]     la distribución tal cual
T = 0.5   →  [0.867, 0.117, 0.016]     más afilada: casi siempre el primero
T = 2.0   →  [0.506, 0.307, 0.186]     más plana: más variedad
T → 0     →  [1, 0, 0]                 equivale a greedy
```

Dividir por un número pequeño **separa** los logits, y como el softmax es exponencial, esa
separación se amplifica. Dividir por uno grande los **junta**.

Valores típicos: 0,7–0,9 para texto coherente, 1,0 para variedad, por encima de 1,2 empieza
a desvariar.

### Top-k: quedarse con los k mejores

El problema de la temperatura sola es que **nunca elimina** los tokens malos, solo los hace
menos probables. Con 4096 tokens, la cola larga puede acumular un 20% de la masa entre
miles de opciones absurdas, y de vez en cuando sale una.

Top-k lo corta en seco: ordena, se queda con los `k` mayores, pone el resto a $-\infty$.

Su defecto es que `k` es **fijo**. Si el modelo está segurísimo del siguiente token, k=40
mete 39 alternativas malas. Si está genuinamente indeciso entre 100, corta opciones buenas.

### Top-p (nucleus): quedarse con los que suman p

La respuesta a ese defecto. En vez de un número fijo, se acumula probabilidad hasta llegar a
`p` y se corta ahí:

```
probs = [0.60, 0.25, 0.10, 0.03, 0.02]
p = 0.9

acumulado sin este token:
  0.60  →  0.00  ≤ 0.9  →  entra
  0.25  →  0.60  ≤ 0.9  →  entra
  0.10  →  0.85  ≤ 0.9  →  entra   ← el que CRUZA el umbral también entra
  0.03  →  0.95  > 0.9  →  fuera
  0.02  →  0.98  > 0.9  →  fuera
```

Se queda con 3 candidatos, que suman 0,95.

**Fíjate en el token que cruza el umbral: entra.** La definición de Holtzman es *"el conjunto
más pequeño cuya probabilidad acumulada **excede** p"*, y `[0.60, 0.25]` suma 0,85, que no
excede 0,9. Hace falta el tercero. Si cortaras antes, el conjunto no llegaría a la masa
pedida.

Es un off-by-one fácil de equivocar —yo lo tuve mal escribiendo este módulo— y por eso en el
código la comparación es sobre la acumulada **antes** de incluir cada token.

Y si la distribución fuera `[0.2, 0.2, 0.2, 0.2, 0.2]`, se quedaría con los 5. **El número de
candidatos se adapta a lo seguro que esté el modelo**, y eso es exactamente lo que quieres.

Un detalle de implementación: **el token más probable siempre se conserva**, aunque él solo
ya supere `p`. Si no, con `p=0.5` y un token de probabilidad 0,9 te quedarías sin candidatos.

### Penalización de repetición

Un parche directo contra los bucles: bajar el logit de los tokens que ya han salido.

Aquí hay un detalle que casi todo el mundo implementa mal. Hay que **dividir si el logit es
positivo y multiplicar si es negativo**:

```
logit = +3  →  3 / 1.1 = 2.73    lo acerca a cero
logit = -3  →  -3 * 1.1 = -3.3   lo aleja de cero, hacia abajo
```

Si dividieras siempre, un logit de −5 pasaría a −4,5, o sea que el token se volvería **más**
probable: justo lo contrario.

## Parte 2: la KV cache

Ahora la parte de velocidad, y es donde está la ganancia grande.

### El problema

Al generar el token 100, la versión ingenua pasa los 100 tokens por el modelo. Otra vez.
Aunque los 99 primeros no han cambiado nada.

Generar N tokens cuesta $O(N^2)$ cuando debería costar $O(N)$.

### La solución

Guardar las claves y valores de cada capa. En cada paso, procesar **solo el token nuevo** y
concatenar sus K y V a lo guardado.

Lo que **no** se puede cachear son las queries: cada token nuevo necesita su propia pregunta.
Lo que se reutiliza son las respuestas (K) y los contenidos (V) de los anteriores. De ahí el
nombre.

El bucle queda en dos fases:

1. **Prefill:** se pasa el prompt entero de golpe y se llena la cache.
2. **Decode:** en cada paso entra un solo token.

### El detalle que rompe todo si lo olvidas

**RoPE tiene que rotar el token nuevo con el ángulo de su posición real.** Al generar el
token 50 le pasas un tensor de longitud 1, y si aplicas RoPE tal cual, lo rotará como si
fuera la posición 0.

Por eso `apply_rope` necesita saber cuántos tokens hay ya en la cache:

```python
cos_t = cos[pos_offset : pos_offset + seq_len]
```

Sin ese recorte, la generación con cache produce texto distinto —y peor— que sin ella, y el
bug es difícil de localizar porque nada falla: simplemente el modelo escribe mal.

### La memoria

$$\text{memoria KV} = 2 \times n_{\text{layers}} \times T \times d_{\text{model}} \times \text{bytes}$$

Nuestro modelo con 512 tokens en fp16: **3,9 MB**. Nada.

Un modelo de 70B con contexto de 100.000 tokens: decenas de gigabytes, más que los propios
pesos. Por eso existen técnicas como *grouped-query attention*, que comparten K y V entre
varias cabezas.

### Una limitación que conviene conocer

La generación con cache **para** al llegar al contexto máximo, en vez de recortar como hace
la versión ingenua.

No es pereza. Recortar con cache exigiría descartar las entradas antiguas **y remapear las
posiciones de RoPE** de todo lo que queda, porque los tokens supervivientes pasarían a
ocupar posiciones distintas de aquellas con las que se rotaron. Eso se llama *sliding window
attention*, y es un tema por sí solo.

Parar es lo honesto: la alternativa silenciosa sería generar texto incorrecto sin avisar.

### La comprobación obligatoria

La cache tiene que dar **exactamente la misma salida**, no parecida. Con `temperature=0`
(greedy, determinista) las dos secuencias deben coincidir token a token. Si no, hay un bug.

En la demo verás ambas cosas: salida idéntica y un speedup de 2–3× que crece con la longitud.

## Dónde está el debate

**Nadie sabe cuáles son los parámetros de muestreo correctos.** Los valores que se usan
—temperatura 0,8, top-p 0,9— son folclore heredado, ajustados a ojo sobre modelos concretos.
No hay teoría que los derive, y el óptimo depende del modelo, de la tarea y de a quién le
preguntes.

Más de fondo: **por qué el texto humano no maximiza la probabilidad** sigue sin explicación
satisfactoria. Holtzman et al. lo documentaron empíricamente y propusieron top-p como
remedio, pero la pregunta de fondo —qué distribución genera realmente el lenguaje humano y
por qué maximizar verosimilitud se aleja de ella— está abierta.

Y hay una discusión práctica en curso sobre si el muestreo debería sustituirse por algo
mejor. Se han propuesto alternativas (typical sampling, mirostat, min-p) con argumentos
razonables, y ninguna ha desplazado a top-p. Puede que porque no sean mejores, o puede que
por inercia.

---

**Para ampliar:** Holtzman et al. 2020,
[The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (top-p) ·
Fan et al. 2018, [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833)
(top-k). Términos sueltos, en [GLOSARIO.md](../../GLOSARIO.md).
