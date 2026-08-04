# 15 — Evaluación

## Por qué importa este módulo

**Porque "¿es bueno mi modelo?" es más difícil de responder de lo que parece.**

Tienes un modelo entrenado y un número: la pérdida. ¿Y ahora qué? Ese número no te dice si
el modelo escribe historias que alguien querría leer, y comparar tu número con el de otro
modelo puede ser directamente engañoso.

Aquí aprendes a usar tres herramientas distintas y, sobre todo, **qué mide cada una y dónde
falla**. Incluida una métrica que sí es comparable entre modelos distintos y que casi nadie
usa, y la parte que ninguna métrica automática sustituye: leer lo que escribe.

Es también el módulo donde se pone en contexto lo que has construido, con expectativas
concretas sobre qué puede y qué no puede hacer un modelo de 9M.

### Qué sabrás al terminar

- Por qué comparar perplejidades entre modelos con tokenizadores distintos **no significa
  nada**, y se hace constantemente
- Una métrica que sí es comparable, y su interpretación exacta: tu modelo es un compresor
- Cómo evaluar cualitativamente con una batería fija de prompts
- Qué esperar realmente de un modelo de 9M, para no llevarte una decepción injusta

### Cuánto cuesta

2 horas. Tres funciones cortas y un informe generado que puedes leer con calma.

---

## Perplejidad: la métrica de siempre

Ya la conoces del módulo 05. Es $e^L$, con $L$ la pérdida media, y se interpreta como
**entre cuántas opciones equiprobables está dudando el modelo**:

```
pérdida 8,317  →  perplejidad 4096   (sin entrenar, dudando entre todo el vocabulario)
pérdida 1,60   →  perplejidad 4,95   (dudando entre unas 5 opciones)
```

Es la métrica más usada porque es barata, automática y correlaciona bien con la calidad
*dentro de un mismo setup*.

### Y su problema, que es serio

**La perplejidad depende del tokenizador.** Si tu vocabulario parte las palabras en trozos
más pequeños, cada token individual es más fácil de predecir y tu perplejidad sale mejor
sin que el modelo sea mejor.

Un ejemplo extremo: un modelo a nivel de bit tendría perplejidad cercana a 2 y sería
inútil. Uno a nivel de palabra, con el mismo texto, tendría perplejidad de cientos.

**Comparar perplejidades entre modelos con tokenizadores distintos no significa nada**, y se
hace constantemente en papers y en posts de blog.

## Bits por byte: la métrica que sí es comparable

La solución: normalizar por **bytes de texto original** en vez de por tokens. Los bytes no
dependen de cómo trocees.

$$\text{bits/byte} = \frac{L_{\text{total}} / \ln 2}{n_{\text{bytes}}}$$

El $\ln 2$ convierte nats a bits.

Y tiene una interpretación exacta y bonita: **es cuántos bits necesitarías para transmitir el
texto usando el modelo como compresor**. Un modelo de 1,0 bits/byte comprime a la octava
parte. Referencias:

| | bits/byte |
|---|---|
| gzip sobre texto en inglés | ~2,5 |
| un buen modelo pequeño | ~1,2 |
| los mejores LLM | 0,6–0,8 |
| el límite teórico (Shannon) | ~0,6–1,3 (discutido) |

Esta equivalencia entre predicción y compresión viene de Shannon (1948) y no es una
analogía: es una identidad. Un modelo de lenguaje **es** un compresor.

## La batería cualitativa

Ninguna de las dos métricas anteriores te dice si el modelo escribe historias que un humano
querría leer. Para eso hay que leerlas.

El paper de TinyStories propone evaluar tres cosas por separado, con prompts fijos:

**Gramática.** ¿Las frases están bien construidas? ¿Concuerdan sujeto y verbo?

**Coherencia.** ¿La historia se contradice? Si en la primera frase el gato es negro, ¿sigue
siendo negro tres frases después?

**Creatividad.** ¿Aporta algo o repite plantillas?

Lo interesante del paper es que estas tres capacidades **aparecen a escalas distintas**. Un
modelo de 1M de parámetros ya hace gramática decente; la coherencia necesita más; la
creatividad, todavía más. No es una escalera única: son habilidades que emergen por
separado.

En el módulo usarás una batería de seis prompts fijos y leerás las continuaciones tú. Sí,
a mano. No hay atajo.

## Por qué no hay una métrica automática buena

Se han intentado muchas y todas fallan por el mismo sitio.

**BLEU, ROUGE** y compañía comparan contra una respuesta de referencia. Para generación
libre no hay una respuesta correcta: hay infinitas, y todas distintas de la de referencia.

**Usar otro LLM como juez** (LLM-as-a-judge) es lo que se hace ahora, y funciona
razonablemente para modelos grandes. Tiene sesgos conocidos y bien documentados: prefiere
respuestas largas, prefiere el estilo del propio modelo juez, y es sensible al orden en que
se le presentan las opciones.

**La evaluación humana** es el patrón oro y es cara, lenta y ruidosa: dos anotadores
discrepan más de lo que uno esperaría.

Para tu modelo de 9M, leer seis continuaciones es perfectamente razonable y probablemente
más informativo que cualquier número.

## Qué esperar de un modelo de 9M sobre TinyStories

Sé concreto con las expectativas, porque el paper original entrenaba modelos parecidos:

- **Gramática correcta la mayor parte del tiempo.** Frases bien formadas.
- **Coherencia local, no global.** Dos o tres frases seguidas tienen sentido; una historia
  de diez, probablemente no.
- **Vocabulario limitado**, que es lo esperado: TinyStories está escrito a propósito con
  vocabulario de niño de 4 años.
- **Nada de razonamiento.** Ni aritmética, ni conocimiento del mundo, ni instrucciones.

Si tu modelo hace eso, ha funcionado. Si esperabas algo parecido a un asistente, la
diferencia no es de entrenamiento: es de tres o cuatro órdenes de magnitud en parámetros y
datos, y del post-entrenamiento que verás en el módulo 16.

## Dónde está el debate

**La relación entre perplejidad y capacidades es más floja de lo que se asume.** Se sabe que
bajar la pérdida mejora el modelo, pero no cuánto ni en qué. Dos modelos con la misma
perplejidad pueden comportarse muy distinto en tareas concretas, y la perplejidad puede
bajar por memorización sin que mejore nada útil.

**La contaminación de datos ha arruinado buena parte de la evaluación por benchmarks.** Los
conjuntos de test están en internet, y los modelos se entrenan con internet. Cuando un
modelo saca buena nota en un benchmark, distinguir "ha aprendido" de "lo ha visto" es
técnicamente difícil y comercialmente incómodo. Es uno de los problemas metodológicos más
serios del campo ahora mismo.

**Y las capacidades emergentes están en discusión activa.** Se documentaron saltos bruscos
de capacidad al aumentar la escala, y en 2023 se publicó un análisis convincente
argumentando que muchos de esos saltos son **artefactos de métricas discontinuas**: si mides
con una métrica de todo-o-nada, ves saltos donde con una continua verías una curva suave. La
discusión sigue.

---

**Para ampliar:** Eldan & Li 2023, [TinyStories](https://arxiv.org/abs/2305.07759) (la
batería cualitativa) · Shannon 1948, *A Mathematical Theory of Communication* (predicción y
compresión) · Schaeffer et al. 2023,
[Are Emergent Abilities a Mirage?](https://arxiv.org/abs/2304.15004). Términos sueltos, en
[GLOSSARY.md](../../GLOSSARY.md).
