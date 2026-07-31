"""Pistas progresivas por ejercicio.

Tres niveles, pensados para desbloquearte sin quemarte el ejercicio:

  1. **Conceptual.** Que estas intentando conseguir y por que. Sin codigo, sin formulas.
     Muchas veces con esto basta: el atasco no era tecnico, era que no estaba claro el
     objetivo.
  2. **Tecnico.** La formula exacta, las formas de los tensores, o que funcion de PyTorch
     es la adecuada. Todavia sin codigo.
  3. **Estructural.** El esqueleto en pseudocodigo, incluidos los sitios donde la gente se
     equivoca (ejes al reves, broadcasting, el `-1` que sobra). Sigue sin ser la solucion
     escrita: eso lo pones tu.

Si el nivel 3 no te desbloquea, `modulos/NN_*/SOLUCION.md` explica la solucion entera y
`llmfs/reference/` tiene el codigo. Mirarlo no es hacer trampa: hacer trampa seria mirarlo
sin haber intentado nada. Aprender leyendo una solucion que ya has peleado funciona muy
bien; leerla en frio no funciona nada.
"""

from __future__ import annotations

HINTS: dict[str, dict[str, tuple[str, ...]]] = {
    # ------------------------------------------------------------------ modulo 00
    "00_que_es_un_llm": {
        "next_token_probs": (
            "Tienes cuentas: la 'n' salio 40 veces, la 'r' 25... Son numeros enteros que "
            "no significan nada por si solos, porque dependen de lo largo que fuera el "
            "texto.\n\n"
            "Lo que necesitas es la PROPORCION: de todas las veces que miraste, "
            "que fraccion fue una 'n'. Y esas fracciones tienen que sumar exactamente 1, "
            "porque algo tuvo que venir siempre.",

            "Probabilidad de cada token = su conteo dividido entre la suma de TODOS los "
            "conteos.\n\n"
            "    total = 40 + 25 + 20 + 15 = 100\n"
            "    P(n) = 40/100 = 0.40\n\n"
            "Dos avisos:\n"
            "- Comprueba que el total no sea 0 ANTES de dividir, y lanza ValueError si lo "
            "es. Si no, el ZeroDivisionError saltara mas tarde y en otro sitio.\n"
            "- Conserva el orden original de las claves. El ejercicio 2 recorre este "
            "diccionario y el orden cambia el resultado del muestreo.",

            "    total = sum(counts.values())\n"
            "    si total == 0: raise ValueError(...)\n"
            "    return {token: conteo / total  para cada (token, conteo) en counts.items()}\n\n"
            "Una comprension de diccionario sobre `counts.items()` conserva el orden sola. "
            "No ordenes las claves.",
        ),
        "sample_next_token": (
            "Piensa en una ruleta. Cada caracter ocupa un trozo del circulo proporcional a "
            "su probabilidad: la 'n' con 0.40 ocupa el 40% de la ruleta.\n\n"
            "Tiras la bola una vez (un numero aleatorio entre 0 y 1) y devuelves el "
            "caracter en cuyo trozo ha caido.\n\n"
            "No devuelvas el mas probable siempre: eso genera texto repetitivo y con "
            "bucles.",

            "Extiende la ruleta en una recta del 0 al 1:\n\n"
            "    |----'n'----|--'r'--|--' '--|-'s'-|\n"
            "    0          0.40    0.65    0.85   1.0\n\n"
            "Saca `r = rng.random()` (un float en [0,1)) y recorre los tokens llevando un "
            "acumulador. Devuelve el primero cuyo acumulado SUPERE a `r`.\n\n"
            "Y al salir del bucle, devuelve el ultimo token de todas formas: por error de "
            "redondeo el acumulado puede quedarse en 0.9999999 y no superar nunca a un `r` "
            "muy cercano a 1. Si no lo haces, la funcion devuelve None y el ejercicio 3 "
            "revienta con un error que no dice nada.",

            "    r = rng.random()\n"
            "    acumulado = 0.0\n"
            "    ultimo = None\n"
            "    para cada (token, p) en probs.items():\n"
            "        acumulado += p\n"
            "        ultimo = token\n"
            "        si r < acumulado:\n"
            "            return token\n"
            "    return ultimo\n\n"
            "Usa `<` y no `<=`. Con {a:0.5, b:0.5} y r=0.5 exacto, `<` devuelve 'b', que es "
            "lo correcto: 'a' ocupa [0, 0.5) y 'b' ocupa [0.5, 1).",
        ),
        "generate_naive": (
            "Ya sabes sacar una distribucion (ejercicio 1) y elegir un caracter de ella "
            "(ejercicio 2). Generar texto es hacer eso en bucle.\n\n"
            "La clave: cada caracter que sacas se convierte en parte de la entrada del "
            "paso siguiente. Escribes una letra, la lees como si te la hubiera dado otro, "
            "y decides la siguiente.\n\n"
            "Este bucle es el mismo que ejecuta ChatGPT. Literalmente el mismo.",

            "En cada vuelta:\n"
            "  1. coge los ultimos `len(start)` caracteres de lo que llevas generado\n"
            "  2. busca ese contexto en la tabla\n"
            "  3. si no esta, PARA (la tabla solo conoce lo que vio al entrenar)\n"
            "  4. si esta, convierte los conteos en probabilidades y muestrea\n"
            "  5. anyade el caracter elegido\n\n"
            "Cuidado con `length`: es el total devuelto INCLUYENDO `start`. Si start tiene "
            "2 caracteres y piden 5, generas 3, no 5.",

            "    context_size = len(start)\n"
            "    salida = list(start)\n"
            "    repetir max(0, length - len(start)) veces:\n"
            "        contexto = ''.join(salida[-context_size:])\n"
            "        counts = table.get(contexto)\n"
            "        si no counts: break\n"
            "        salida.append(sample_next_token(next_token_probs(counts), rng))\n"
            "    return ''.join(salida)\n\n"
            "Acumula en una lista y une al final con `''.join()`. Concatenar cadenas en un "
            "bucle crea una cadena nueva en cada vuelta; aqui da igual, pero es una "
            "costumbre que en el modulo 13 sale cara.",
        ),
    },
    # ------------------------------------------------------------------ modulo 01
    "01_entorno": {
        "measure_matmul_tflops": (
            "Quieres saber cuantas operaciones por segundo hace tu GPU DE VERDAD, no las "
            "que pone en la caja. La forma de averiguarlo es darle un trabajo grande y "
            "conocido, cronometrarlo, y dividir.\n\n"
            "El trabajo: multiplicar dos matrices cuadradas. Sabes exactamente cuantas "
            "operaciones son, asi que solo te falta el tiempo.",

            "FLOPs de multiplicar dos matrices (size x size) = 2 * size^3.\n"
            "TFLOPS = (FLOPs * numero_de_repeticiones) / segundos / 1e12\n\n"
            "Las dos cosas que hay que hacer bien:\n\n"
            "1. CALENTAR. La primera multiplicacion de un tamanyo dado es entre 10 y 100 "
            "veces mas lenta: la GPU esta eligiendo que kernel usar y reservando memoria. "
            "Haz unas cuantas que no cronometres.\n\n"
            "2. SINCRONIZAR. `a @ b` en GPU no espera a que termine: encola el trabajo y "
            "devuelve el control. Si cronometras sin sincronizar, mides lo que tarda en "
            "encolar (microsegundos) y te salen miles de TFLOPS. Usa `cfg.synchronize()` "
            "justo antes de cada marca de tiempo.",

            "    a = torch.randn(size, size, device=cfg.device, dtype=dtype)\n"
            "    b = torch.randn(size, size, device=cfg.device, dtype=dtype)\n\n"
            "    repetir `warmup` veces: a @ b\n"
            "    cfg.synchronize()\n\n"
            "    t0 = time.perf_counter()\n"
            "    repetir `iters` veces: a @ b\n"
            "    cfg.synchronize()\n"
            "    elapsed = time.perf_counter() - t0\n\n"
            "    return (2 * size**3 * iters) / elapsed / 1e12\n\n"
            "El dtype por defecto: `cfg.amp_dtype` si existe, si no `torch.float32`.",
        ),
        "transformer_flops_per_token": (
            "Quieres saber cuanto cuesta procesar UN token, para poder multiplicar por "
            "500 millones y saber si el entrenamiento dura dos horas o dos semanas.\n\n"
            "La idea que lo hace facil: casi todo el coste de una red son "
            "multiplicaciones de matrices, y una matriz de P parametros cuesta "
            "aproximadamente 2P operaciones por cada token que la atraviesa.",

            "Suma los parametros que participan en multiplicaciones:\n\n"
            "    por capa: 4*d_model^2 (las 4 proyecciones de atencion)\n"
            "            + n_ffn_matrices*d_model*d_ff (el FFN)\n"
            "    mas la proyeccion final a logits: d_model*vocab_size\n\n"
            "Forward = 2 * esos_parametros + 4*n_layers*context_length*d_model\n\n"
            "Ese segundo termino es la atencion en si (Q@K^T y softmax@V). No sale de "
            "parametros: sale de multiplicar tokens entre si, y por eso crece con el "
            "contexto y no con el tamanyo del modelo.\n\n"
            "Total con backward = 3 * forward. El backward cuesta el doble que el forward "
            "porque hace dos multiplicaciones por cada una del forward: una para el "
            "gradiente respecto a la entrada y otra respecto a los pesos.",

            "    params = n_layers * (4*d_model**2 + n_ffn_matrices*d_model*d_ff)\n"
            "    params += d_model * vocab_size\n"
            "    forward = 2*params + 4*n_layers*context_length*d_model\n"
            "    return int(3*forward if include_backward else forward)\n\n"
            "Dos avisos:\n"
            "- La proyeccion final cuenta aunque uses weight tying. Atar los pesos ahorra "
            "memoria, no calculo: el matmul se hace igual.\n"
            "- NO dividas por dos por la mascara causal. Es la convencion de nanoGPT y de "
            "los papers; si divides, tu MFU no sera comparable con la de nadie.",
        ),
        "estimate_tokens_per_second": (
            "Tienes la potencia de tu GPU (TFLOPS) y lo que cuesta un token (FLOPs). "
            "Dividir una cosa entre la otra da tokens por segundo.\n\n"
            "El unico matiz es la MFU: nunca aprovechas el 100% de la GPU, asi que hay "
            "que multiplicar por la fraccion que realmente consigues.",

            "    tokens/s = TFLOPS * 1e12 * MFU / FLOPs_por_token\n\n"
            "El 1e12 es para pasar de TeraFLOPS a FLOPS. Comprueba que "
            "`flops_per_token` sea positivo y lanza ValueError si no: una division por "
            "cero aqui produce `inf` silenciosamente y estimaciones absurdas.",

            "    si flops_per_token <= 0: raise ValueError(...)\n"
            "    return tflops * 1e12 * mfu / flops_per_token\n\n"
            "Valores de MFU realistas: 0.4-0.5 para modelos de miles de millones bien "
            "optimizados, 0.1-0.2 para nuestro modelo de 9M. Las matrices de 320x320 son "
            "demasiado pequenyas para saturar los tensor cores.",
        ),
    },
    # ------------------------------------------------------------------ modulo 02
    "02_autograd": {
        "Value": (
            "Un `Value` es un numero que ademas se acuerda de como se calculo. Cuando "
            "haces `c = a * b`, el objeto `c` guarda: su valor, quienes son `a` y `b`, y "
            "COMO repartir hacia atras el gradiente que le llegue.\n\n"
            "La parte que cuesta ver: ese 'como repartir' se guarda como una funcion que "
            "NO se ejecuta todavia. Se ejecutara mas tarde, durante el backward. Estas "
            "construyendo una lista de tareas pendientes mientras haces el forward.",

            "Todas las operaciones siguen el mismo molde:\n\n"
            "    def __mul__(self, other):\n"
            "        other = other if isinstance(other, Value) else Value(other)\n"
            "        out = Value(self.data * other.data, (self, other), '*')\n"
            "        def _backward():\n"
            "            self.grad  += other.data * out.grad\n"
            "            other.grad += self.data  * out.grad\n"
            "        out._backward = _backward\n"
            "        return out\n\n"
            "Las derivadas locales que necesitas:\n"
            "    a+b   -> da += 1*out.grad ; db += 1*out.grad\n"
            "    a*b   -> da += b.data*out.grad ; db += a.data*out.grad\n"
            "    a**n  -> da += n * a.data**(n-1) * out.grad\n"
            "    exp   -> da += out.data * out.grad\n"
            "    log   -> da += (1/a.data) * out.grad\n"
            "    tanh  -> da += (1 - out.data**2) * out.grad\n"
            "    relu  -> da += (out.data > 0) * out.grad",

            "SIEMPRE `+=`, NUNCA `=`. Es el error que hay que evitar. Pruebalo con "
            "`y = x + x`: con `=` la segunda rama pisa a la primera y sale 1; con `+=` "
            "sale 2, que es lo correcto porque y = 2x.\n\n"
            "El azucar no necesita derivadas nuevas:\n"
            "    -a      = a * -1\n"
            "    a - b   = a + (-b)\n"
            "    a / b   = a * b**-1\n\n"
            "Y `backward()` son tres lineas:\n"
            "    self.grad = 1.0\n"
            "    for node in reversed(topological_order(self)):\n"
            "        node._backward()\n\n"
            "El `self.grad = 1.0` es la semilla: la derivada de algo respecto a si mismo. "
            "Sin ella todos los gradientes salen 0.",
        ),
        "topological_order": (
            "Cuando un nodo reparte su gradiente hacia sus hijos, tiene que haber "
            "recibido ya TODO lo que le llega de sus padres. Si lo reparte antes de "
            "tiempo, manda hacia abajo un gradiente incompleto.\n\n"
            "Necesitas una lista donde cada nodo aparezca DESPUES de todos sus hijos. "
            "`root` queda el ultimo, y `backward()` recorre la lista al reves.",

            "Es un DFS post-orden: visitas los hijos de un nodo y solo despues anyades el "
            "nodo al resultado.\n\n"
            "HAZLO ITERATIVO, con una pila explicita. La version recursiva son cinco "
            "lineas pero revienta con RecursionError en cuanto el grafo tenga unos "
            "cientos de nodos, y el MLP del ejercicio 3 los tiene.\n\n"
            "Usa `id(nodo)` para el conjunto de visitados, no el nodo. Si sobrecargas "
            "operadores en una clase, apoyarte en su hash por defecto es pedir problemas.",

            "El truco es una bandera que dice si ya expandiste los hijos de ese nodo:\n\n"
            "    order, visited = [], set()\n"
            "    stack = [(root, False)]\n"
            "    while stack:\n"
            "        node, expanded = stack.pop()\n"
            "        if expanded:\n"
            "            order.append(node); continue\n"
            "        if id(node) in visited: continue\n"
            "        visited.add(id(node))\n"
            "        stack.append((node, True))          # me reencolo para DESPUES\n"
            "        for child in node._prev:\n"
            "            if id(child) not in visited:\n"
            "                stack.append((child, False))\n\n"
            "Cada nodo entra dos veces: la primera para expandir sus hijos, la segunda "
            "-que se procesa cuando ya estan todos- para anyadirse. Eso es el post-orden.\n\n"
            "Si te sale `root` el PRIMERO, tienes el orden invertido. El sintoma sera que "
            "los gradientes salen bien en grafos simples y mal en cuanto haya un nodo "
            "reutilizado.",
        ),
        "train_scalar_mlp": (
            "Es el bucle de entrenamiento en su forma mas desnuda: predecir, medir el "
            "error, calcular gradientes, mover los pesos un poco en contra del gradiente, "
            "repetir.\n\n"
            "El del modulo 11 sera este mismo con AMP, scheduler y checkpointing "
            "encima. El nucleo no cambia.",

            "En cada paso, y en este orden:\n"
            "    1. preds = [model(x) for x in xs]\n"
            "    2. loss  = media de (pred - y)**2\n"
            "    3. model.zero_grad()      <- ANTES del backward\n"
            "    4. loss.backward()\n"
            "    5. p.data -= lr * p.grad  para cada p en model.parameters()\n"
            "    6. history.append(loss.data)\n\n"
            "El paso 3 es el que se olvida todo el mundo. Los gradientes se ACUMULAN "
            "(lo hiciste asi en el ejercicio 1), asi que sin limpiarlos el paso 50 usa la "
            "suma de los gradientes de los pasos 1 a 50. No da ningun error: la perdida "
            "baja un poco y se estanca.",

            "    model = MLP(len(xs[0]), [*hidden, 1], value_cls=value_cls, seed=seed)\n"
            "    history = []\n"
            "    for _ in range(steps):\n"
            "        preds = [model(x) for x in xs]\n"
            "        loss = sum(((p - y)**2 for p, y in zip(preds, ys)),\n"
            "                   value_cls(0.0)) * (1.0/len(ys))\n"
            "        model.zero_grad()\n"
            "        loss.backward()\n"
            "        for p in model.parameters():\n"
            "            p.data -= lr * p.grad\n"
            "        history.append(loss.data)\n"
            "    return history\n\n"
            "Dos detalles:\n"
            "- El `value_cls(0.0)` como valor inicial del `sum()`: sin el, python empieza "
            "a acumular desde el entero 0 y mezclas tipos.\n"
            "- `p.data -= ...` y no `p -= ...`. Modificas el numero de dentro, no creas "
            "un nodo nuevo. Si crearas nodos, el grafo del paso siguiente colgaria del "
            "anterior y creceria sin parar. En PyTorch esto es el `torch.no_grad()`.",
        ),
    },
    # ------------------------------------------------------------------ modulo 03
    "03_tokenizacion": {
        "get_stats": (
            "Recorre la lista mirando de dos en dos y cuenta cuantas veces sale cada "
            "pareja de vecinos. Es el primer paso de BPE: para fusionar el par mas "
            "frecuente, primero hay que saber cual es.",

            "`zip(ids, ids[1:])` te da todos los pares consecutivos sin tener que "
            "manejar indices.\n\n"
            "OJO: al CONTAR, los pares SI se solapan. En [1,1,1] el par (1,1) sale 2 "
            "veces: en las posiciones 0-1 y en las 1-2. (Al FUSIONAR, en el ejercicio 2, "
            "no se solapan. Son cosas distintas.)\n\n"
            "El parametro `counts` sirve para acumular sobre un diccionario que ya "
            "existe, que es lo que necesita `train_bpe` para sumar varios chunks.",

            "    counts = {} if counts is None else counts\n"
            "    for pair in zip(ids, ids[1:]):\n"
            "        counts[pair] = counts.get(pair, 0) + 1\n"
            "    return counts\n\n"
            "Devuelve el diccionario ademas de mutarlo: asi vale para los dos usos, "
            "`stats = get_stats(ids)` y `get_stats(chunk, stats)`.\n\n"
            "El valor por defecto es `None` y no `{}` a proposito. Un `{}` como valor por "
            "defecto se crea UNA VEZ al definir la funcion y se comparte entre todas las "
            "llamadas: es el clasico bug de python con argumentos mutables.",
        ),
        "merge": (
            "Recorres la lista y cada vez que encuentras el par buscado lo sustituyes por "
            "un unico numero nuevo. Esto es lo que acorta la secuencia y crea el token.",

            "La clave esta en como avanzas: cuando hay coincidencia consumes DOS "
            "posiciones, cuando no, UNA.\n\n"
            "Por eso un `for` no vale bien: avanza siempre de uno en uno. Necesitas un "
            "`while` con un indice que controlas tu.\n\n"
            "En [1,1,1] fusionando (1,1) el resultado es [256, 1], no [256, 256]: al "
            "consumir las posiciones 0 y 1, el 1 de la posicion 2 ya no tiene pareja.",

            "    out, i, n = [], 0, len(ids)\n"
            "    while i < n:\n"
            "        if i < n - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:\n"
            "            out.append(new_id)\n"
            "            i += 2\n"
            "        else:\n"
            "            out.append(ids[i])\n"
            "            i += 1\n"
            "    return out\n\n"
            "El `i < n - 1` evita mirar `ids[i+1]` cuando estas en el ultimo elemento. "
            "Sin el, IndexError en cuanto la lista acabe justo en el primer elemento del "
            "par.\n\n"
            "Devuelve una lista NUEVA, no modifiques la de entrada.",
        ),
        "train_bpe": (
            "Es el ejemplo de 'aaabdaaabac' del TEORIA.md, en bucle. Repites: cuenta los "
            "pares, coge el mas frecuente, fusionalo, apunta el merge. Tantas veces como "
            "tokens nuevos quieras.\n\n"
            "Empiezas con los 256 bytes, asi que los ids nuevos van del 256 en adelante.",

            "Estructura:\n"
            "  1. trocear con `regex.findall(pattern, text)` (o dejarlo entero si "
            "pattern es None)\n"
            "  2. cada trozo a bytes: `list(chunk.encode('utf-8'))`\n"
            "  3. `vocab = {i: bytes([i]) for i in range(256)}`\n"
            "  4. repetir `vocab_size - 256` veces:\n"
            "       - acumular `get_stats` de TODOS los trozos en un mismo dict\n"
            "       - si el dict esta vacio -> `break` (no quedan pares)\n"
            "       - elegir el par ganador\n"
            "       - aplicar `merge` a cada trozo\n"
            "       - `merges[par] = 256 + i`\n"
            "       - `vocab[256+i] = vocab[par[0]] + vocab[par[1]]`  (bytes, no str)\n\n"
            "Se cuenta chunk a chunk para que ningun merge una el final de una palabra "
            "con el principio de la siguiente.",

            "El desempate tiene que ser EXACTAMENTE este o tus merges divergiran de los "
            "de la referencia en cuanto haya un empate:\n\n"
            "    pair = max(stats, key=lambda p: (stats[p], p))\n\n"
            "Python compara tuplas elemento a elemento: primero frecuencia, y si empata, "
            "el par. Cual gane da igual para la calidad; lo que importa es que sea "
            "determinista y el mismo criterio.\n\n"
            "El `break` cuando `stats` esta vacio NO es opcional: si pides 4096 merges "
            "sobre un texto corto, llega un momento en que no quedan pares y `max()` "
            "sobre un dict vacio lanza ValueError.\n\n"
            "Y valida `vocab_size >= 256` al principio.",
        ),
        "bpe_encode": (
            "Aplicar los merges que aprendiste a texto nuevo. El detalle que lo hace "
            "no-trivial: hay que aplicarlos EN EL ORDEN EN QUE SE APRENDIERON, no en el "
            "orden en que aparecen en este texto.\n\n"
            "Si los aplicas en otro orden, la tokenizacion es valida pero DISTINTA de la "
            "que vio el modelo al entrenar, y el modelo no la entiende.",

            "Como los ids de merge son 256, 257, 258... en orden de aprendizaje, "
            "'el que se aprendio antes' es 'el que tiene el id mas bajo'.\n\n"
            "    stats = get_stats(ids)\n"
            "    pair = min(stats, key=lambda p: merges.get(p, float('inf')))\n\n"
            "El `float('inf')` es el truco: los pares que no estan en `merges` reciben "
            "infinito y nunca ganan el `min`. Si el ganador resulta no estar en `merges`, "
            "es que ya no queda nada fusionable: para.",

            "    def _encode_chunk(ids, merges):\n"
            "        while len(ids) >= 2:\n"
            "            stats = get_stats(ids)\n"
            "            pair = min(stats, key=lambda p: merges.get(p, float('inf')))\n"
            "            if pair not in merges:\n"
            "                break\n"
            "            ids = merge(ids, pair, merges[pair])\n"
            "        return ids\n\n"
            "Y arriba: trocear con el MISMO patron con el que entrenaste, aplicar "
            "`_encode_chunk` a cada trozo, y concatenar todos los ids en una lista.\n\n"
            "El `len(ids) >= 2` evita llamar a `get_stats` sobre algo que no tiene pares.",
        ),
        "bpe_decode": (
            "Lo contrario de codificar, y mucho mas corto. Cada id tiene asociada una "
            "secuencia de bytes en `vocab`. Las juntas todas y las decodificas.",

            "Lo unico que hay que hacer bien: JUNTAR PRIMERO, DECODIFICAR DESPUES.\n\n"
            "Una 'n' es un byte pero una 'ñ' son dos (0xC3 0xB1). A BPE le da igual: "
            "puede haber aprendido un token que acaba en 0xC3 y otro que empieza por "
            "0xB1. Por separado ninguno es UTF-8 valido; juntos son una 'ñ'.\n\n"
            "Asi que NO hagas `''.join(vocab[i].decode() for i in ids)`.",

            "    raw = b''.join(vocab[i] for i in ids)\n"
            "    return raw.decode('utf-8', errors='replace')\n\n"
            "El `errors='replace'` es el BYTES FALLBACK y tampoco es opcional. Un modelo "
            "a medio entrenar genera ids al azar y muchas de esas secuencias no son UTF-8 "
            "valido. Con `replace` sale un caracter de reemplazo y la generacion sigue; "
            "sin el, una excepcion tumba el bucle entero por un byte suelto.",
        ),
    },
    # ------------------------------------------------------------------ modulo 04
    "04_datos": {
        "pack_tokens_uint16": (
            "Vas a guardar 500 millones de tokens en disco. Con el int64 que usa python "
            "por defecto son 4 GB; con uint16 son 1 GB. Como tus ids van de 0 a 4095 y "
            "uint16 llega a 65.535, sobra de largo.\n\n"
            "El ejercicio no es la conversion (una linea), es la VALIDACION.",

            "Numpy no avisa si un numero no cabe: hace wrap around en silencio.\n\n"
            "    np.array([65536], dtype=np.int64).astype(np.uint16)   ->  [0]\n\n"
            "Sin excepcion, sin warning. Los datos quedan corruptos, el modelo entrena "
            "peor y no hay nada que apunte a la causa.\n\n"
            "Hay que comprobar DOS cosas: que `vocab_size` cabe en uint16, y que ningun "
            "id es negativo ni >= `vocab_size`. Y hacerlo ANTES de convertir, en un tipo "
            "donde todo quepa.",

            "    if vocab_size > 2**16: raise ValueError(...)\n"
            "    array = np.asarray(ids, dtype=np.int64)\n"
            "    if array.size and (array.min() < 0 or array.max() >= vocab_size):\n"
            "        raise ValueError(f'... minimo={array.min()}, maximo={array.max()}')\n"
            "    return array.astype(np.uint16)\n\n"
            "El `array.size and ...` es necesario: `.min()` sobre un array vacio lanza "
            "una excepcion que no tiene nada que ver y despista.\n\n"
            "Pon los valores concretos en el mensaje de error. 'ids fuera de rango' no "
            "ayuda; 'maximo=9999' te dice al instante que tu tokenizador esta mal.",
        ),
        "train_val_split": (
            "Necesitas texto que el modelo NO vea al entrenar, para saber si esta "
            "aprendiendo o solo memorizando.\n\n"
            "La unica decision del ejercicio: el corte es CONTIGUO y POR EL FINAL, no "
            "aleatorio.",

            "Por que no aleatorio: las ventanas de entrenamiento se solapan. La que "
            "empieza en la posicion 100 y la que empieza en la 101 comparten 511 de sus "
            "512 tokens.\n\n"
            "Si repartieras al azar, validacion estaria llena de fragmentos ya vistos. La "
            "perdida saldria preciosa y no significaria nada: estarias midiendo "
            "memorizacion y llamandolo generalizacion.\n\n"
            "Cortando un bloque del final, lo que reservas son historias enteras.",

            "    if not 0.0 < val_fraction < 1.0: raise ValueError(...)\n"
            "    n_val = max(1, int(len(tokens) * val_fraction))\n"
            "    if n_val >= len(tokens): raise ValueError(...)\n"
            "    return tokens[:-n_val], tokens[-n_val:]\n\n"
            "El slicing de numpy devuelve VISTAS, no copias, y eso es lo que quieres: con "
            "500M tokens, un copy gratuito es 1 GB de RAM tirado.\n\n"
            "El `max(1, ...)` evita que con un corpus pequenyo `int(50*0.005)` de 0 y te "
            "quedes sin conjunto de validacion.",
        ),
        "get_batch": (
            "Eliges posiciones al azar en el corpus. De cada una coges una ventana como "
            "entrada, y LA MISMA ventana desplazada un token como objetivo.\n\n"
            "    x = [5, 8, 2, 9]\n"
            "    y = [8, 2, 9, 1]\n\n"
            "Leelo columna a columna: viendo [5] predice 8, viendo [5,8] predice 2... "
            "Una ventana de 4 tokens son CUATRO ejemplos de entrenamiento.",

            "    max_start = len(data) - context_length - 1\n"
            "    starts = rng.integers(0, max_start, size=batch_size)\n"
            "    x = np.stack([data[i : i+context_length]     for i in starts])\n"
            "    y = np.stack([data[i+1 : i+1+context_length] for i in starts])\n\n"
            "El `-1` de `max_start` es el off-by-one del ejercicio: `y` necesita un token "
            "MAS alla del final de `x`. Sin el, la ultima ventana desborda, y numpy no "
            "avisa (el slicing fuera de rango simplemente devuelve menos elementos); lo "
            "que ves es un `np.stack` fallando por formas incompatibles.\n\n"
            "Si `max_start < 1`, el corpus es mas corto que el contexto: ValueError.",

            "Despues del stack:\n\n"
            "    x_np = ....astype(np.int64)      # OBLIGATORIO\n"
            "    x = torch.from_numpy(x_np)\n"
            "    if device is not None:\n"
            "        device = torch.device(device)\n"
            "        if device.type == 'cuda':\n"
            "            x = x.pin_memory().to(device, non_blocking=True)\n"
            "        else:\n"
            "            x = x.to(device)\n\n"
            "El `.astype(np.int64)` hace dos cosas: nn.Embedding exige indices int64, y "
            "de paso COPIA los datos. Sin la copia te quedarias apuntando al fichero "
            "mapeado en disco y cada acceso del modelo seria una lectura.\n\n"
            "`pin_memory` + `non_blocking` solo tienen sentido en CUDA: permiten que la "
            "copia del siguiente batch se solape con el calculo del actual. En MPS la "
            "memoria es unificada y no hay copia que solapar.",
        ),
    },
    # ------------------------------------------------------------------ modulo 05
    "05_baselines": {
        "uniform_baseline_loss": (
            "Un modelo que no sabe nada reparte la probabilidad por igual entre todas las "
            "palabras del vocabulario: le da 1/V a cada una.\n\n"
            "La perdida es -ln(probabilidad que le dio al token correcto). Si esa "
            "probabilidad es siempre 1/V, la perdida es siempre la misma. Calculala.",

            "    -ln(1/V) = ln(V)\n\n"
            "Y ya esta, es una linea con `math.log`. Valida que `vocab_size >= 1` y lanza "
            "ValueError si no.\n\n"
            "Numeros que vas a ver: ln(65)=4.174 para shakespeare a nivel caracter, "
            "ln(4096)=8.317 para el modelo final.",

            "    if vocab_size < 1: raise ValueError(...)\n"
            "    return math.log(vocab_size)\n\n"
            "Guarda este numero: es tu detector de bugs mas barato. En el modulo 11, la "
            "perdida del paso 0 tiene que valer casi exactamente esto.\n"
            "  - mucho mas alta -> la inicializacion es demasiado agresiva\n"
            "  - mas baja       -> fuga de informacion (mascara causal mal puesta)",
        ),
        "bigram_counts": (
            "Una matriz V x V donde la casilla [i][j] cuenta cuantas veces el token j vino "
            "justo detras del token i.\n\n"
            "Con ids=[0,1,0,1,2] los pares son (0,1),(1,0),(0,1),(1,2), asi que "
            "counts[0][1]=2 y counts[1][0]=counts[1][2]=1.",

            "Se puede con un bucle, pero con 500M tokens tardaria una eternidad. "
            "Vectorizado:\n\n"
            "    tokens = torch.as_tensor(ids, dtype=torch.int64)\n"
            "    counts.index_put_((tokens[:-1], tokens[1:]),\n"
            "                      torch.ones(len(tokens)-1, dtype=torch.int64),\n"
            "                      accumulate=True)\n\n"
            "`tokens[:-1]` son todos los 'desde' y `tokens[1:]` todos los 'hasta'.",

            "El `accumulate=True` NO es opcional. Sin el, `index_put_` ASIGNA en vez de "
            "sumar: cada par repetido pisa al anterior y todos los conteos acaban valiendo "
            "1. Pruebalo con [0,0,0,0,0]: el resultado correcto es counts[0][0]=4.\n\n"
            "Y con menos de 2 tokens no hay ningun par: devuelve la matriz de ceros sin "
            "intentar indexar nada.",
        ),
        "bigram_nll": (
            "Ya tienes los conteos del texto de entrenamiento. Ahora mides como de bien "
            "predicen OTRO texto (el de validacion).\n\n"
            "Para cada par consecutivo de la secuencia a evaluar coges la probabilidad que "
            "el modelo le asigna, le aplicas -ln, y promedias.",

            "    P(b|a) = (C[a][b] + alpha) / (suma_b' C[a][b'] + alpha*V)\n\n"
            "El `alpha` (suavizado de Laplace) es imprescindible: sin el, un par que nunca "
            "aparecio tiene probabilidad 0, su logaritmo es -infinito, y como la perdida es "
            "una MEDIA, ese -inf se lleva por delante el resultado entero.\n\n"
            "Pasos:\n"
            "  1. smoothed = counts.double() + alpha\n"
            "  2. probs = smoothed / smoothed.sum(dim=1, keepdim=True)\n"
            "  3. seleccionar probs[tokens[:-1], tokens[1:]]\n"
            "  4. float(-torch.log(...).mean())",

            "Dos trampas:\n\n"
            "1. El `keepdim=True` del paso 2. Sin el, la suma tiene forma (V,) en vez de "
            "(V,1) y el broadcast divide por COLUMNAS en lugar de por filas. El resultado "
            "sale plausible y es completamente incorrecto.\n\n"
            "2. El denominador. Al sumar alpha a las V entradas de una fila, el total crece "
            "en alpha*V, no en alpha. Si haces la suma DESPUES de anyadir alpha (como en el "
            "paso 2), esto se resuelve solo.\n\n"
            "Usa `.double()` y no `.float()`: con corpus grandes se suman millones de "
            "conteos y float32 pierde precision.",
        ),
        "NeuralBigram": (
            "El mismo modelo del ejercicio 2, pero aprendido por descenso de gradiente en "
            "vez de contando.\n\n"
            "La idea: una tabla de V filas y V columnas donde la fila `i` son directamente "
            "los LOGITS del token que sigue al token `i`. Entrenada con cross-entropy, "
            "converge a los conteos normalizados.",

            "Un unico submodulo, y el nombre importa porque el test copia pesos por "
            "nombre:\n\n"
            "    self.token_embedding = nn.Embedding(vocab_size, vocab_size)\n\n"
            "El forward es leer la tabla:\n"
            "    logits = self.token_embedding(idx)      # (B, T) -> (B, T, V)\n\n"
            "Y si hay targets, la perdida:\n"
            "    F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))",

            "El `reshape(-1, V)` es porque `F.cross_entropy` espera (N, clases) y (N,), "
            "pero tu tienes (B, T, V) y (B, T). Aplanar batch y tiempo en una sola "
            "dimension es el patron que repetiras en TODOS los modelos del curso.\n\n"
            "Devuelve `(logits, None)` si no hay targets, no `(logits, 0)`.\n\n"
            "Curiosidad util: `nn.Embedding` y `nn.Linear` son lo mismo matematicamente (un "
            "Embedding es un Linear con entrada one-hot), pero el Embedding LEE la fila que "
            "necesita en vez de multiplicar por una matriz llena de ceros.",
        ),
        "BengioMLP": (
            "En vez de mirar solo el token anterior, mira los `block_size` anteriores. "
            "Coge sus embeddings, los PEGA uno detras de otro en un vector largo, y pasa "
            "ese vector por un MLP normal.\n\n"
            "Es el paper de 2003 que invento los word embeddings.",

            "Tres submodulos, con estos nombres exactos:\n\n"
            "    self.embedding = nn.Embedding(vocab_size, d_embed)\n"
            "    self.hidden    = nn.Linear(block_size * d_embed, n_hidden)\n"
            "    self.output    = nn.Linear(n_hidden, vocab_size)\n\n"
            "Y el forward:\n"
            "    emb    = self.embedding(idx)       # (B, block_size, d_embed)\n"
            "    flat   = emb.reshape(B, -1)        # (B, block_size*d_embed)\n"
            "    h      = torch.tanh(self.hidden(flat))\n"
            "    logits = self.output(h)            # (B, V)\n\n"
            "Ojo: aqui `targets` es `(B,)`, UN token por muestra, no una secuencia.",

            "El `reshape(B, -1)` es CONCATENAR, y eso es lo importante. Si hicieras "
            "`emb.mean(dim=1)` estarias promediando, y el modelo perderia el orden: le "
            "daria lo mismo [el, gato, come] que [come, gato, el]. Hay un test que lo "
            "comprueba pasando el contexto al reves.\n\n"
            "Y cuidado con donde va el -1: `reshape(B, -1)`, no `reshape(-1, B)`. El "
            "segundo compila y produce basura.\n\n"
            "Fijate en la capa `hidden`: sus parametros crecen LINEALMENTE con block_size. "
            "Esa es justo la limitacion que la atencion viene a resolver en el modulo 06.",
        ),
    },
    # ------------------------------------------------------------------ modulo 06
    "06_atencion": {
        "causal_mask": (
            "Al entrenar le damos al modelo la frase entera de golpe y le pedimos que "
            "prediga cada token a partir de los anteriores. Sin nada que lo impida, la "
            "posicion 3 podria mirar a la 4, que es literalmente la respuesta.\n\n"
            "Necesitas una matriz que diga, para cada par (i, j), si el token i tiene "
            "permiso para mirar al token j.",

            "Convenio del curso: True = SI puede mirar. Es el mismo que usa "
            "`F.scaled_dot_product_attention`.\n\n"
            "Para seq_len=4:\n\n"
            "    [[ T, F, F, F],\n"
            "     [ T, T, F, F],\n"
            "     [ T, T, T, F],\n"
            "     [ T, T, T, T]]\n\n"
            "Es una matriz triangular INFERIOR, con la diagonal incluida (un token si "
            "puede mirarse a si mismo). PyTorch tiene una funcion para esto.",

            "    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()\n\n"
            "`tril` = triangular lower. Su argumento `diagonal` vale 0 por defecto, que "
            "INCLUYE la diagonal: es lo que quieres.\n\n"
            "Si te sale al reves has usado `triu`. Si la diagonal sale False, has pasado "
            "`diagonal=-1`.\n\n"
            "Aviso para mas adelante: `nn.MultiheadAttention` de PyTorch usa el convenio "
            "CONTRARIO (True = prohibido). Por eso el test que compara contra ella pasa "
            "`~mask`. Es una inconsistencia real de la libreria.",
        ),
        "single_head_attention": (
            "Es el ejemplo de TEORIA.md hecho con tensores. Cada token pregunta (query), "
            "todos responden (keys), se mide el parecido con productos escalares, se "
            "convierte en pesos con softmax, y se mezclan los contenidos (values).\n\n"
            "    salida = softmax( Q K^T / sqrt(d_k) + mascara ) V",

            "Cuatro pasos:\n\n"
            "  1. scores = q @ k.transpose(-2, -1)\n"
            "     (B,T,d_k) @ (B,d_k,S) -> (B,T,S). La casilla [b,i,j] es cuanto le "
            "interesa a i el token j.\n"
            "  2. dividir por sqrt(d_k), con d_k = q.shape[-1]\n"
            "  3. scores.masked_fill(~mask, float('-inf'))  si hay mascara\n"
            "  4. weights = F.softmax(scores, dim=-1) ; salida = weights @ v\n\n"
            "Devuelve `(salida, weights)`: los pesos hacen falta para el heatmap de la demo.",

            "Tres trampas, y las tres son silenciosas (no dan error, solo resultados malos):\n\n"
            "1. Usa `transpose(-2, -1)`, con indices NEGATIVOS. Asi funciona igual con "
            "(B,T,d) que con (B,heads,T,d). Con `transpose(1,2)` este ejercicio pasa y el "
            "siguiente se rompe.\n\n"
            "2. `dim=-1` en el softmax. Normalizas sobre A QUIEN se mira, de forma que cada "
            "fila sume 1. Con `dim=-2` normalizarias sobre quien mira, que no significa "
            "nada, y las formas son identicas asi que no salta ningun error.\n\n"
            "3. La mascara va ANTES del softmax. Si borrases los pesos despues, las filas "
            "ya no sumarian 1 y estarias escalando la salida por un factor arbitrario.",
        ),
        "MultiHeadAttention": (
            "Varias atenciones en paralelo, cada una con sus propias proyecciones, para "
            "que puedan especializarse en relaciones distintas.\n\n"
            "Y no cuesta mas: con d_model=320 y 8 cabezas, cada una trabaja en 40 "
            "dimensiones. En vez de una atencion de 320 haces ocho de 40.",

            "El truco: NO hagas 8 proyecciones separadas. Haz una de d_model -> d_model y "
            "parte el resultado.\n\n"
            "    partir:  x.view(B, T, n_heads, head_dim).transpose(1, 2)\n"
            "             (B, T, d_model) -> (B, n_heads, T, head_dim)\n\n"
            "    juntar:  x.transpose(1, 2).contiguous().view(B, T, d_model)\n\n"
            "Con q, k, v ya partidos, la atencion es EXACTAMENTE la misma formula del "
            "ejercicio 2 (por eso el transpose de alli tenia que usar indices negativos).\n\n"
            "Submodulos: q_proj, k_proj, v_proj, out_proj (todos Linear d_model->d_model), "
            "attn_dropout y resid_dropout.",

            "Cuatro detalles que fallan si no los cuidas:\n\n"
            "1. El ORDEN del view: `view(B, T, n_heads, head_dim)` y LUEGO transpose. Si "
            "haces `view(B, n_heads, T, head_dim)` directamente estas mezclando posiciones "
            "con cabezas. Forma correcta, datos mal, cero errores.\n\n"
            "2. `.contiguous()` antes del view al juntar. `transpose` no mueve datos, solo "
            "cambia los strides, y `view` exige memoria contigua.\n\n"
            "3. RoPE (si cos/sin no son None) va DESPUES de partir en cabezas, porque la "
            "rotacion depende de head_dim. Y solo a q y k, nunca a v.\n\n"
            "4. Si `self.use_sdpa`, usa `F.scaled_dot_product_attention(q, k, v, "
            "attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)`. Ese "
            "`if self.training` importa: SDPA no consulta el modo por su cuenta y aplicaria "
            "dropout tambien en evaluacion.",
        ),
    },
    # ------------------------------------------------------------------ modulo 07
    "07_normalizacion": {
        "layer_norm": (
            "Los numeros que atraviesan una red profunda tienden a crecer o encogerse capa "
            "tras capa, hasta explotar o desvanecerse. La solucion es brutal en su "
            "simplicidad: despues de cada bloque, vuelve a ponerlos en una escala conocida.\n\n"
            "LayerNorm coge el vector de cada token, le resta su media y lo divide por su "
            "desviacion. Sale media 0 y varianza 1, venga de donde venga.",

            "    y = (x - media) / sqrt(varianza + eps) * weight + bias\n\n"
            "Sobre la ULTIMA dimension (las features de cada token), con `dim=-1` y "
            "`keepdim=True`. Cada token se normaliza por su cuenta, sin mirar al batch: eso "
            "es lo que lo distingue de BatchNorm.\n\n"
            "El eps va DENTRO de la raiz: sqrt(var + eps), no sqrt(var) + eps.\n\n"
            "Si `weight` o `bias` son None, no los apliques.",

            "LA TRAMPA: `torch.var` divide por (n-1) por defecto (varianza muestral). "
            "LayerNorm usa la POBLACIONAL, que divide por n.\n\n"
            "    var = x.var(dim=-1, keepdim=True, unbiased=False)\n\n"
            "Sin el `unbiased=False`, tu resultado se parecera mucho a `F.layer_norm` pero "
            "no coincidira. Con d=320 la diferencia es del 0,3% y podrias no verla; con d=4 "
            "es del 33%. El test compara contra las dos versiones y te dice a cual te "
            "pareces.\n\n"
            "Y el `keepdim=True` tampoco es opcional: sin el, la media de (4,8,32) sale "
            "(4,8) en vez de (4,8,1) y la resta se emite mal.",
        ),
        "RMSNorm": (
            "Lo mismo que LayerNorm pero SIN restar la media y SIN sesgo. Solo reescala.\n\n"
            "Zhang y Sennrich (2019) observaron que casi todo el beneficio de LayerNorm "
            "viene de reescalar, no de recentrar. Quitandolo se ahorra una pasada por los "
            "datos y un tensor intermedio. Lo usan Llama, Mistral y nuestro modelo.",

            "    y = x / sqrt( media(x^2) + eps ) * weight\n\n"
            "Con x = [2, 8, 4, 6]:\n"
            "    RMS = sqrt((4+64+16+36)/4) = sqrt(30) = 5.477\n"
            "    y   = [0.365, 1.461, 0.730, 1.096]\n\n"
            "El unico parametro es `weight`, de forma (dim,), inicializado a UNOS. Al "
            "arrancar la capa tiene que ser la normalizacion pura: si empezara aleatorio, "
            "la perdida del paso 0 no cuadraria con ln(V).\n\n"
            "`torch.rsqrt(z)` calcula 1/sqrt(z) de una vez y es mas rapido que dividir.",

            "    def _norm(self, x):\n"
            "        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)\n\n"
            "    def forward(self, x):\n"
            "        return self._norm(x.float()).type_as(x) * self.weight\n\n"
            "El `.float()` NO es paranoia. Con activaciones en fp16, elevar al cuadrado "
            "desborda antes de lo que parece: 300^2 = 90.000 y fp16 se acaba en 65.504. "
            "Saldria inf, luego la media seria inf, y rsqrt(inf) = 0: la capa devolveria "
            "ceros. Hay un test que reproduce ese caso exacto.\n\n"
            "Detalle que sorprende: aunque hagas `.type_as(x)`, la salida acaba en fp32, "
            "porque `self.weight` es fp32 y PyTorch promociona. Es correcto y es lo que "
            "hace Llama.",
        ),
        "prenorm_residual": (
            "Es UNA LINEA y es el ejercicio mas importante del modulo.\n\n"
            "En vez de que cada bloque SUSTITUYA la representacion, se le pide que la "
            "MODIFIQUE: la salida es la entrada mas una correccion. Asi hay siempre un "
            "camino directo de la entrada a la salida.",

            "Dos opciones, y solo cambian los parentesis de sitio:\n\n"
            "    post-norm (paper de 2017):   norm(x + fn(x))\n"
            "    pre-norm  (todo lo moderno): x + fn(norm(x))\n\n"
            "En pre-norm la normalizacion esta DENTRO de la rama, asi que el camino x -> x "
            "queda libre. Al derivar sale `1 + algo`, y ese 1 llega intacto a las capas de "
            "abajo por muchas que haya.\n\n"
            "En post-norm la norma esta ENCIMA de la suma, asi que el gradiente la atraviesa "
            "en cada capa y se va reescalando.",

            "    return x + fn(norm(x))\n\n"
            "Ya esta. Si te sale `norm(x + fn(x))` has hecho post-norm y hay un test que lo "
            "detecta.\n\n"
            "Consecuencia para el modulo 10: como la corriente residual nunca se normaliza "
            "por el camino, llega a la salida con una escala que crece con la profundidad. "
            "Por eso los modelos pre-norm llevan SIEMPRE una normalizacion final antes de la "
            "capa de logits. Se llamara `norm_f`.",
        ),
    },
    # ------------------------------------------------------------------ modulo 08
    "08_mlp_y_activaciones": {
        "gelu": (
            "La atencion es una media ponderada, o sea una operacion LINEAL. Y dos "
            "operaciones lineales seguidas son una sola: W2·(W1·x) = (W2·W1)·x. Sin algo "
            "no lineal entre capa y capa, cien capas equivalen a una.\n\n"
            "GELU es esa pieza. Multiplica x por la probabilidad de que una normal estandar "
            "salga menor que x: en vez de cortar en seco los negativos como hace ReLU, los "
            "atenua de forma gradual.",

            "La aproximacion por tanh, que es la que se pide:\n\n"
            "    GELU(x) ~= 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))\n\n"
            "Escrita de izquierda a derecha, sin reagrupar. Las constantes son las del "
            "paper: sqrt(2/pi) ~= 0.7978 y 0.044715.\n\n"
            "Tiene que coincidir con `F.gelu(x, approximate='tanh')`, NO con `F.gelu(x)` a "
            "secas: son funciones distintas y el test compara contra la primera.",

            "    return 0.5 * x * (1.0 + torch.tanh(\n"
            "        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))\n"
            "    ))\n\n"
            "Una linea, sin bucles ni ramas.\n\n"
            "Lo importante del ejercicio no es la formula, es la DERIVADA. Con ReLU vale "
            "cero exacto en toda la zona negativa, asi que una neurona que se vaya alli deja "
            "de recibir gradiente para siempre. Con GELU es pequenya pero no nula, y puede "
            "volver. El demo lo tabula.",
        ),
        "swiglu_hidden_dim": (
            "Aritmetica pura, pero con una decision detras. SwiGLU tiene TRES matrices donde "
            "el FFN clasico tiene dos, asi que con el mismo hidden costaria un 50% mas.\n\n"
            "Para gastar los mismos parametros se reduce el hidden a dos tercios. Este "
            "ejercicio produce el 896 del config final.",

            "Tres pasos:\n\n"
            "  1. hidden = int(2 * (4 * d_model) / 3)\n"
            "  2. si ffn_dim_multiplier no es None: hidden = int(ffn_dim_multiplier * hidden)\n"
            "  3. redondear HACIA ARRIBA al siguiente multiplo de `multiple_of`\n\n"
            "Comprobaciones:\n"
            "  d_model=320 -> int(2*1280/3) = 853 -> ceil_64 = 896   (config final)\n"
            "  d_model=128 -> int(2*512/3)  = 341 -> ceil_64 = 384   (tiny_char)",

            "El redondeo hacia arriba con enteros, sin math.ceil:\n\n"
            "    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)\n\n"
            "Sumar (multiple_of - 1) antes de la division entera fuerza el redondeo hacia "
            "arriba, y si ya era multiplo exacto no lo cambia.\n\n"
            "Por que se redondea: las dimensiones alineadas dejan que los tensor cores usen "
            "sus rutas rapidas. Una matriz de 853 columnas es mas lenta que una de 896 "
            "teniendo menos parametros.",
        ),
        "SwiGLU": (
            "Dos proyecciones en paralelo desde la misma entrada. Una de ellas, tras pasar "
            "por una activacion, actua como PUERTA: multiplica a la otra elemento a elemento "
            "y decide cuanta senyal deja pasar por cada dimension.\n\n"
            "La diferencia con una activacion normal es que ese filtrado DEPENDE DE LA "
            "ENTRADA: decide, para cada dimension y cada token, cuanto pasa.",

            "    SwiGLU(x) = down( Swish(gate(x)) * up(x) )\n\n"
            "El `*` es multiplicacion ELEMENTO A ELEMENTO, no matricial: las dos ramas salen "
            "con forma (B, T, d_ff) y se multiplican punto a punto.\n\n"
            "`Swish(z) = z * sigmoid(z)`, que en PyTorch es `F.silu(z)`.\n\n"
            "Submodulos: gate_proj y up_proj (Linear d_model -> d_ff), down_proj (Linear "
            "d_ff -> d_model) y dropout. Todos sin sesgo por defecto.",

            "    def forward(self, x):\n"
            "        return self.dropout(self.down_proj(\n"
            "            F.silu(self.gate_proj(x)) * self.up_proj(x)\n"
            "        ))\n\n"
            "La activacion va en `gate_proj`, NO en `up_proj`. Con la asignacion invertida "
            "el modulo funciona igual de bien pero no coincide con la referencia al copiar "
            "pesos, y el test falla con una diferencia dificil de interpretar. Hay un test "
            "dedicado a senyalarlo.\n\n"
            "Y ten presente que el FFN procesa cada token POR SEPARADO: no mezcla "
            "posiciones, eso es trabajo de la atencion. Aqui no hace falta ninguna mascara.",
        ),
    },
    # ------------------------------------------------------------------ modulo 09
    "09_posicion": {
        "sinusoidal_embeddings": (
            "La atencion es una suma ponderada, y una suma no tiene orden: sin informacion "
            "posicional, 'el perro muerde al hombre' y 'el hombre muerde al perro' dan "
            "exactamente la misma salida.\n\n"
            "Esta tabla se SUMA a los embeddings de token para decir en que posicion esta "
            "cada uno. La idea es la de un contador binario: cada par de dimensiones oscila "
            "a un ritmo distinto, y la combinacion identifica la posicion.",

            "    PE[pos, 2i]   = sin( pos / base^(2i/d) )\n"
            "    PE[pos, 2i+1] = cos( pos / base^(2i/d) )\n\n"
            "Dimensiones PARES con seno, IMPARES con coseno, misma frecuencia cada par.\n\n"
            "Sin bucles:\n"
            "  - `position = torch.arange(seq_len).unsqueeze(1)`  -> (T, 1)\n"
            "  - las frecuencias, una por par -> (d/2,)\n"
            "  - `position * div_term` emite a (T, d/2): todos los angulos de golpe\n"
            "  - `tabla[:, 0::2] = sin(...)` y `tabla[:, 1::2] = cos(...)` para intercalar",

            "El truco que merece la pena conocer, para las frecuencias:\n\n"
            "    div_term = torch.exp(\n"
            "        torch.arange(0, d_model, 2, dtype=torch.float32)\n"
            "        * (-math.log(base) / d_model)\n"
            "    )\n\n"
            "Es matematicamente igual a `base ** (-2i/d)` pero mucho mas estable: elevar "
            "10000 a una potencia negativa grande pierde precision en coma flotante, y "
            "hacerlo pasando por logaritmos no.\n\n"
            "Regla general que te servira en otros sitios: si ves una potencia con exponente "
            "grande, `exp(log(...))` suele ser mejor.",
        ),
        "rope_frequencies": (
            "RoPE no SUMA nada al vector: lo ROTA. Cada par de dimensiones gira un angulo "
            "proporcional a la posicion, y cada par tiene su propia velocidad de giro.\n\n"
            "Esta funcion precalcula, de una vez y para todas las posiciones, el coseno y el "
            "seno de esos angulos.",

            "El angulo del par `i` en la posicion `pos` es:\n\n"
            "    angulo = pos * theta^(-2i/head_dim)\n\n"
            "Los pasos:\n"
            "  1. validar que head_dim sea PAR (RoPE rota pares) -> ValueError si no\n"
            "  2. inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))\n"
            "     -> head_dim/2 valores\n"
            "  3. angles = torch.outer(positions, inv_freq)   -> (max_seq_len, head_dim/2)\n"
            "  4. DUPLICAR: angles = torch.cat([angles, angles], dim=-1)\n"
            "  5. devolver (angles.cos(), angles.sin())",

            "El paso 4 es el que confunde, asi que aqui va el porque.\n\n"
            "Hay dos convenios para emparejar dimensiones al rotar:\n"
            "  - el paper original empareja consecutivas: (x0,x1), (x2,x3)...\n"
            "  - Llama y HuggingFace emparejan por MITADES: (x0, x_{d/2}), (x1, x_{d/2+1})...\n\n"
            "Usamos el de mitades. Con el, la dimension `i` y la `i + head_dim/2` forman un "
            "par y necesitan EL MISMO angulo. Por eso cada frecuencia aparece duplicada y "
            "las tablas tienen head_dim columnas en vez de head_dim/2.\n\n"
            "Los dos convenios son equivalentes salvo una permutacion de dimensiones, que la "
            "red aprende sin enterarse. El de mitades gano porque hace que apply_rope sea "
            "una linea sin reordenar nada.",
        ),
        "apply_rope": (
            "Con las tablas ya calculadas esto es UNA LINEA. Pero conviene ver de donde sale "
            "antes de escribirla.\n\n"
            "Rotar un vector (x1, x2) un angulo t es la matriz de rotacion de siempre:\n\n"
            "    x1' = x1*cos(t) - x2*sin(t)\n"
            "    x2' = x2*cos(t) + x1*sin(t)",

            "Esas dos lineas se escriben de golpe como:\n\n"
            "    x' = x * cos + rotate_half(x) * sin\n\n"
            "donde `rotate_half([a, b]) = [-b, a]`, partiendo por la mitad la ultima "
            "dimension. Compruebalo:\n\n"
            "    componente 1:  x1*cos + (-x2)*sin = x1*cos - x2*sin   OK\n"
            "    componente 2:  x2*cos + ( x1)*sin = x2*cos + x1*sin   OK\n\n"
            "Y rotate_half sin bucles:\n"
            "    mitad = x.shape[-1] // 2\n"
            "    x1, x2 = x[..., :mitad], x[..., mitad:]\n"
            "    return torch.cat([-x2, x1], dim=-1)",

            "    seq_len = x.shape[-2]\n"
            "    cos = cos[:seq_len].to(dtype=x.dtype, device=x.device)\n"
            "    sin = sin[:seq_len].to(dtype=x.dtype, device=x.device)\n"
            "    return x * cos + rotate_half(x) * sin\n\n"
            "Dos detalles que fallan si los saltas:\n\n"
            "1. RECORTAR a seq_len. Las tablas se precalculan hasta max_seq_len (512 en el "
            "modelo final) y tu secuencia casi nunca mide eso. Sin recortar, el broadcast "
            "falla o -peor- acierta por casualidad con las formas equivocadas.\n\n"
            "2. Convertir dtype y device. Bajo AMP las tablas estan en fp32 y x llega en "
            "fp16; mezclarlos hace que torch promocione y acabas calculando en la precision "
            "que no querias.\n\n"
            "No hace falta ningun unsqueeze: x es (B, heads, T, head_dim) y cos es "
            "(T, head_dim); el broadcast alinea desde la derecha y se ocupa del resto.",
        ),
    },
    # ------------------------------------------------------------------ modulo 10
    "10_el_gpt_completo": {
        "expected_param_count": (
            "Calcular cuantos parametros tendra el modelo ANTES de construirlo. Sirve para "
            "disenyar (cambias d_model y ves al instante si cabe en la GPU) y para verificar "
            "que el modelo que has montado es el que creias.\n\n"
            "Hazlo con papel primero. Coge el desglose del TEORIA.md y escribe la formula; "
            "solo despues la traduces a codigo. Si vas directo al codigo acabaras probando "
            "numeros hasta que cuadre, y eso no ensenya nada.",

            "Los terminos:\n\n"
            "    embeddings  = vocab_size * d_model\n"
            "    (+ context_length * d_model solo si pos == 'learned')\n\n"
            "    por capa:\n"
            "      atencion = 4 * d_model^2                (Wq, Wk, Wv, Wo)\n"
            "      ffn      = 3 * d_model * d_ff           (SwiGLU)\n"
            "      normas   = 2 * d_model                  (dos RMSNorm)\n\n"
            "    norma final = d_model\n"
            "    lm_head     = 0 si tie_embeddings\n\n"
            "Comprobacion: 1.310.720 + 6*(409.600 + 860.160 + 640) + 320 = 8.933.440",

            "Dos cosas que se olvidan y hacen que no cuadre:\n\n"
            "1. RoPE NO aporta ni un parametro. Sus tablas salen de una formula y se guardan "
            "como buffers. Si tu cuenta incluye algo de RoPE, sobra un termino.\n\n"
            "2. RMSNorm tiene d_model parametros, no 2*d_model: solo escala, sin sesgo. Con "
            "6 capas x 2 normas + 1 final son 13 x 320 = 4.160. Es poco, pero si lo olvidas "
            "el total no cuadra.\n\n"
            "Y acuerdate de las ramas: LayerNorm en vez de RMSNorm, MLP clasico (2 matrices) "
            "en vez de SwiGLU (3), y los sesgos si cfg.bias es True. Los tests prueban todas "
            "esas combinaciones.",
        ),
        "count_parameters": (
            "Lo contrario del ejercicio 1: en vez de calcular con una formula, recorrer el "
            "modelo de verdad y sumar. Si los dos numeros coinciden, tu formula y tu modelo "
            "dicen lo mismo.\n\n"
            "Ademas se desglosa por componente, que es lo que hace util el resultado.",

            "Recorre `model.named_parameters()` y clasifica por lo que aparezca en el "
            "nombre. Los nombres son del estilo `blocks.3.attn.q_proj.weight`.\n\n"
            "    'token_embedding' / 'pos_embedding'  -> embeddings\n"
            "    'attn.'                              -> attention\n"
            "    gate_proj / up_proj / down_proj      -> ffn\n"
            "    'norm'                               -> norms\n"
            "    'lm_head'                            -> lm_head\n\n"
            "Imprime una vez `[n for n, _ in model.named_parameters()]`: vale mucho la pena "
            "para ver como esta montado el modelo por dentro.",

            "EL ORDEN DE LAS COMPROBACIONES IMPORTA. La cadena 'norm' aparece tambien en "
            "`attn_norm` y `ffn_norm`, asi que si compruebas 'norm' antes que 'attn.', la "
            "normalizacion de la atencion acaba en la categoria equivocada. Ve de mas "
            "especifico a mas general.\n\n"
            "Sobre el weight tying: `parameters()` y `named_parameters()` DEDUPLICAN por "
            "defecto (remove_duplicate=True), asi que el total sale bien sin hacer nada. Aun "
            "asi lleva un `set` de `id(param)`: deja explicito que sabes que hay pesos "
            "compartidos y protege el desglose. Con remove_duplicate=False contarias "
            "1.310.720 parametros de mas.\n\n"
            "No olvides `total` (suma de las seis categorias) y `non_embedding` "
            "(total - embeddings), que es lo que usan las leyes de escala del modulo 12.",
        ),
        "TransformerBlock": (
            "Un bloque son dos sub-bloques, cada uno con su normalizacion y su residual:\n\n"
            "    x = x + atencion(norm1(x))\n"
            "    x = x + ffn(norm2(x))\n\n"
            "La atencion MUEVE informacion entre tokens; el FFN la PROCESA token a token. "
            "Alternan. Eso es todo el bloque.",

            "Submodulos, con estos nombres (el test copia pesos por nombre y el ejercicio 2 "
            "clasifica por nombre):\n\n"
            "    self.attn_norm = make_norm(cfg)\n"
            "    self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads,\n"
            "                                   dropout=cfg.dropout, bias=cfg.bias)\n"
            "    self.ffn_norm = make_norm(cfg)\n"
            "    self.ffn = make_ffn(cfg)\n\n"
            "`make_norm` y `make_ffn` ya estan hechos en `llmfs.reference`: eligen RMSNorm o "
            "LayerNorm, SwiGLU o MLP, segun el config.",

            "    def forward(self, x, cos=None, sin=None, mask=None):\n"
            "        x = x + self.attn(self.attn_norm(x), mask=mask, cos=cos, sin=sin)\n"
            "        x = x + self.ffn(self.ffn_norm(x))\n"
            "        return x\n\n"
            "DOS residuales independientes, no uno solo alrededor de todo el bloque. Hay un "
            "test que pone a cero los pesos de salida de las dos ramas y comprueba que la "
            "salida es EXACTAMENTE la entrada: si no tienes residuales, devolveria cero.\n\n"
            "El FFN no recibe cos, sin ni mask: no mira a otros tokens y no los necesita.",
        ),
        "GPT": (
            "Ensamblar todo:\n\n"
            "    ids -> embeddings -> [bloque] x n_layers -> norma final -> logits\n\n"
            "Con RoPE no hay embedding posicional que sumar al principio: la posicion se "
            "inyecta dentro de la atencion. Por eso la primera capa es solo la tabla de "
            "tokens.\n\n"
            "Al terminar, `sum(p.numel() for p in modelo.parameters())` tiene que dar "
            "8.933.440 exactos.",

            "Las tres cosas que hay que hacer bien:\n\n"
            "1. TYING:  `self.lm_head.weight = self.token_embedding.weight`\n"
            "   Eso NO copia: hace que apunten al mismo tensor. El test comprueba `is`, no "
            "`==`.\n\n"
            "2. RoPE COMO BUFFER:\n"
            "     cos, sin = rope_frequencies(cfg.head_dim, cfg.context_length, cfg.rope_theta)\n"
            "     self.register_buffer('rope_cos', cos, persistent=False)\n"
            "   Un buffer acompanya al modelo (se mueve con .to(device)) pero no es un "
            "parametro. `persistent=False` ademas lo saca del checkpoint: se recalcula al "
            "construir.\n\n"
            "3. INIT EN DOS PASADAS: primero `self.apply(self._init_weights)` con std=0.02 "
            "en todos los Linear y Embedding, y DESPUES pisar las proyecciones que escriben "
            "en el residual (out_proj, down_proj) con std = 0.02/sqrt(2*n_layers).",

            "Por que la init escalada: cada bloque SUMA su contribucion a la corriente "
            "residual. Con 6 capas y 2 sub-bloques cada una son 12 contribuciones, asi que "
            "la varianza de la salida seria 12 veces la de la entrada. Dividir la desviacion "
            "por sqrt(2*n_layers) lo compensa. El 2 es porque cada bloque escribe dos veces.\n\n"
            "Y el orden importa: primero el apply general, LUEGO se pisa. Al reves, el apply "
            "sobrescribiria la init escalada.\n\n"
            "En el forward, la mascara se calcula UNA vez antes del bucle de bloques y se "
            "pasa a todos. Dentro de cada bloque funcionaria, pero serian 6 tensores "
            "identicos por forward.\n\n"
            "Y valida `T <= cfg.context_length` al principio, con un ValueError que diga los "
            "dos numeros.",
        ),
    },
    # ------------------------------------------------------------------ modulo 11
    "11_bucle_de_entrenamiento": {
        "AdamWScratch": (
            "El descenso de gradiente pelado (`p -= lr * g`) tiene un problema: un unico lr "
            "para todos los parametros. Los que reciben gradientes grandes constantemente "
            "dan saltos absurdos, y los que casi nunca reciben senyal no se mueven.\n\n"
            "Adam lo arregla con dos medias moviles: una del gradiente (momento, suaviza el "
            "ruido) y otra del gradiente AL CUADRADO (escalado, da a cada parametro su "
            "propio lr efectivo).",

            "    m = beta1*m + (1-beta1)*g\n"
            "    v = beta2*v + (1-beta2)*g^2\n\n"
            "    m_hat = m / (1 - beta1^t)      <- correccion de sesgo\n"
            "    v_hat = v / (1 - beta2^t)\n\n"
            "    p -= lr * m_hat / (sqrt(v_hat) + eps)\n"
            "    p -= lr * weight_decay * p     <- POR SEPARADO\n\n"
            "La estructura de un optimizador de PyTorch:\n\n"
            "    for group in self.param_groups:      # los grupos del ejercicio 4\n"
            "        for p in group['params']:\n"
            "            if p.grad is None: continue\n"
            "            state = self.state[p]        # dict por parametro, persiste\n"
            "            if len(state) == 0:          # primera vez\n"
            "                state['step'] = 0\n"
            "                state['exp_avg'] = torch.zeros_like(p)\n"
            "                state['exp_avg_sq'] = torch.zeros_like(p)\n"
            "            state['step'] += 1\n",

            "Los tres errores que cazan los tests:\n\n"
            "1. EMPEZAR t EN 0. Con t=0, 1-beta^0 = 0 y divides por cero. Incrementa "
            "`state['step']` ANTES de usarlo.\n\n"
            "2. OLVIDAR LA CORRECCION DE SESGO. Hay un test que da un solo paso con "
            "gradiente 1 y lr=0.1, y exige que el parametro se mueva 0.1. Sin correccion, "
            "con beta2=0.95 se moveria 0.447: 4,5 veces mas.\n\n"
            "3. SUMAR EL WEIGHT DECAY AL GRADIENTE. Eso es Adam+L2, no AdamW. Hazlo "
            "directamente sobre el parametro:\n\n"
            "       p.mul_(1 - lr * wd)      # ANTES de la actualizacion de Adam\n\n"
            "   El test lo distingue poniendo el gradiente a CERO: con decay desacoplado el "
            "parametro sigue encogiendo, y con L2 no.\n\n"
            "Y no olvides el `@torch.no_grad()` sobre `step`.",
        ),
        "lr_at_step": (
            "El learning rate no es constante durante el entrenamiento. Sube despacio al "
            "principio (warmup) y baja al final (coseno).\n\n"
            "El warmup existe porque en los primeros pasos los momentos de Adam estan vacios "
            "y los pesos recien inicializados dan gradientes grandes: arrancar a lr completo "
            "suele producir un pico del que a veces el modelo no se recupera.",

            "Tres tramos:\n\n"
            "  1. step < warmup_steps  ->  lr * (step + 1) / warmup_steps\n"
            "  2. step >= max_steps    ->  lr * min_lr_ratio\n"
            "  3. en medio             ->  coseno entre los dos\n\n"
            "El coseno:\n"
            "    progreso = (step - warmup_steps) / (max_steps - warmup_steps)\n"
            "    coef     = 0.5 * (1 + cos(pi * progreso))\n"
            "    return min_lr + (lr - min_lr) * coef\n\n"
            "Comprueba los extremos: progreso=0 da cos(0)=1 y coef=1, o sea `lr`. "
            "progreso=1 da cos(pi)=-1 y coef=0, o sea `min_lr`.",

            "Detalles que fallan si los saltas:\n\n"
            "- El `+1` del warmup: sin el, el paso 0 tendria lr exactamente cero y no "
            "aprenderia nada. Con warmup de 500, serian 500 pasos desperdiciados.\n\n"
            "- El orden de las guardas: `step >= max_steps` tiene que ir ANTES del coseno. "
            "Si no, progreso pasaria de 1 y el coseno empezaria a SUBIR otra vez.\n\n"
            "- Acota progreso con `min(1.0, max(0.0, progreso))` y usa `max(1, ...)` en el "
            "denominador: protege si max_steps <= warmup_steps.\n\n"
            "- `schedule` tambien admite 'linear' y 'constant', de una linea cada uno.",
        ),
        "clip_grad_norm": (
            "Ocasionalmente un batch produce gradientes enormes (una secuencia rara, un "
            "token muy poco frecuente). Sin proteccion, ese unico batch puede dar un salto "
            "que destruya horas de entrenamiento.\n\n"
            "La solucion: si la norma de TODOS los gradientes juntos supera un umbral, "
            "escalarlos todos por el mismo factor.",

            "    grads = [p.grad for p in parameters if p.grad is not None]\n"
            "    norma = sqrt( suma de (g**2).sum() sobre todos )\n"
            "    si norma > max_norm:\n"
            "        factor = max_norm / (norma + 1e-6)\n"
            "        multiplicar TODOS los gradientes por factor\n"
            "    return norma      <- la de ANTES de recortar\n\n"
            "El `1e-6` evita dividir por cero si la norma es minuscula.\n\n"
            "Devolver la norma ANTES es lo que hace `torch.nn.utils.clip_grad_norm_` y es lo "
            "util: si la registras y sube de forma sostenida, el entrenamiento se esta "
            "desestabilizando.",

            "LA NORMA ES GLOBAL, no una por tensor. Es lo unico conceptual del ejercicio.\n\n"
            "Si recortaras cada tensor por separado, cada uno se escalaria por un factor "
            "distinto y la DIRECCION del gradiente conjunto cambiaria. Y eso es justo lo que "
            "no quieres: el gradiente apunta a donde hay que ir, y tu solo estas limitando "
            "cuanto avanzas.\n\n"
            "Hay un test que calcula el coseno entre el vector de gradientes antes y despues "
            "del recorte, y exige que sea > 0.9999.\n\n"
            "Usa `g.detach()` al calcular la norma: los gradientes no requieren gradiente, "
            "pero es la costumbre correcta.",
        ),
        "build_param_groups": (
            "El weight decay empuja los pesos hacia cero. Eso tiene sentido en una matriz de "
            "proyeccion, pero NO en la escala de un RMSNorm: ese parametro arranca en 1 y su "
            "trabajo es reescalar; empujarlo hacia cero es empujar la salida de la capa "
            "hacia cero, que es lo contrario de lo que hace falta.\n\n"
            "Asi que hay que separar los parametros en dos grupos.",

            "La regla es sorprendentemente simple:\n\n"
            "    parametros de 2 dimensiones o mas  ->  CON decay   (las matrices)\n"
            "    parametros de 1 dimension          ->  SIN decay   (sesgos y escalas)\n\n"
            "`param.dim()` te da el numero de dimensiones.\n\n"
            "El formato que espera PyTorch es una lista de dicts:\n\n"
            "    [{'params': [...], 'weight_decay': wd},\n"
            "     {'params': [...], 'weight_decay': 0.0}]\n\n"
            "Cualquier clave adicional sobreescribe el valor por defecto del optimizador "
            "solo para ese grupo.",

            "    decay, no_decay = [], []\n"
            "    for param in model.parameters():\n"
            "        if not param.requires_grad:\n"
            "            continue\n"
            "        (decay if param.dim() >= 2 else no_decay).append(param)\n\n"
            "    return [\n"
            "        {'params': decay, 'weight_decay': weight_decay},\n"
            "        {'params': no_decay, 'weight_decay': 0.0},\n"
            "    ]\n\n"
            "El orden importa: primero el grupo CON decay (hay tests que dependen de ello).\n\n"
            "Salta los parametros congelados: no se van a actualizar y meterlos en el "
            "optimizador solo gasta memoria.\n\n"
            "Con el modelo final salen 43 tensores con decay (8.929.280 params) y 13 sin el "
            "(4.160, las escalas de RMSNorm).",
        ),
    },
    # ------------------------------------------------------------------ modulo 12
    "12_eficiencia_y_escalado": {
        "model_flops_per_token": (
            "El mismo calculo del modulo 01, pero devolviendo el DESGLOSE en vez de un solo "
            "numero.\n\n"
            "Merece la pena separarlo porque los dos terminos escalan distinto: el de matmul "
            "con el tamanyo del modelo, el de atencion con el CONTEXTO. Saber cual pesa mas "
            "te dice al instante si alargar el contexto te va a salir caro.",

            "    params_matmul = n_layers * (4*d^2 + n_ffn*d*d_ff) + d*vocab_size\n"
            "                    (n_ffn = 3 con SwiGLU, 2 con un FFN clasico)\n\n"
            "    matmul    = 2 * params_matmul\n"
            "    attention = 4 * n_layers * context_length * d_model\n\n"
            "Si include_backward, multiplica AMBOS por 3.\n\n"
            "Devuelve un dict con `matmul`, `attention`, `total` (la suma) y "
            "`params_matmul`.",

            "La proyeccion final `d * vocab_size` cuenta AUNQUE haya weight tying. Atar los "
            "pesos ahorra memoria, no calculo: el matmul se hace igual. Hay un test que lo "
            "comprueba.\n\n"
            "Y el `* 3` va sobre los DOS terminos, no solo sobre el de matmul: el backward "
            "de la atencion tambien cuesta el doble que su forward.\n\n"
            "Comprobacion: con la config final el total tiene que dar 65.372.160, el mismo "
            "numero del modulo 01.",
        ),
        "compute_mfu": (
            "Que fraccion de la potencia teorica de tu GPU estas aprovechando de verdad.\n\n"
            "Es LA metrica para saber si tu entrenamiento va bien optimizado, y su gracia es "
            "que no depende ni del modelo ni del hardware: puedes comparar configuraciones "
            "distintas.",

            "    MFU = tokens_per_second * flops_per_token / (peak_tflops * 1e12)\n\n"
            "El 1e12 convierte TeraFLOPS en FLOPS. Es una linea.\n\n"
            "Valida que `peak_tflops` sea positivo y lanza ValueError si no: una division por "
            "cero aqui da `inf` en silencio.",

            "    if peak_tflops <= 0:\n"
            "        raise ValueError(...)\n"
            "    return tokens_per_second * flops_per_token / (peak_tflops * 1e12)\n\n"
            "Como se interpreta:\n"
            "    0.4-0.5   modelos grandes bien optimizados en A100/H100\n"
            "    0.1-0.2   nuestro modelo de 9M\n"
            "    < 0.05    algo va mal: mira el dataloader o sube el batch\n\n"
            "NADIE llega a 1. Y con un modelo pequenyo la MFU baja es inevitable: las "
            "matrices de 320x320 no dan para saturar los tensor cores. No es culpa tuya.",
        ),
        "chinchilla_optimal_allocation": (
            "Tienes un presupuesto fijo de FLOPs. ¿Lo gastas en un modelo grande con pocos "
            "datos o en uno pequenyo con muchos?\n\n"
            "Hoffmann et al. (2022) lo midieron entrenando mas de 400 modelos: los dos deben "
            "crecer PROPORCIONALMENTE, unos 20 tokens por parametro.",

            "Partiendo de C = 6ND (modulo 01) y de D = k*N:\n\n"
            "    C = 6 * N * (k*N) = 6k * N^2\n\n"
            "    N = sqrt( C / (6*k) )\n"
            "    D = k * N\n\n"
            "Valida que el presupuesto sea positivo. Devuelve un dict con `params`, "
            "`tokens`, `tokens_per_param` y `compute`.",

            "    if compute_budget <= 0:\n"
            "        raise ValueError(...)\n"
            "    params = (compute_budget / (6 * tokens_per_param)) ** 0.5\n"
            "    tokens = tokens_per_param * params\n\n"
            "LA COMPROBACION QUE DA CONFIANZA: metele el presupuesto real de Chinchilla, "
            "5.88e23 FLOPs. La formula predice 70,0 mil millones de parametros, y el modelo "
            "real tenia exactamente 70B.\n\n"
            "Hay un test que lo verifica, y verlo funcionar con un caso historico da bastante "
            "mas confianza que leer la formula.",
        ),
    },
    # ------------------------------------------------------------------ modulo 13
    "13_entrenamiento_final": {
        "overfit_single_batch": (
            "Un modelo con millones de parametros tiene capacidad de sobra para memorizar "
            "cuatro secuencias. Si le das el MISMO batch una y otra vez, la perdida TIENE "
            "que bajar practicamente a cero.\n\n"
            "Si no baja, hay un bug. Y lo sabes en 30 segundos en vez de en cuatro horas.",

            "El bucle mas simple posible, sin scheduler, sin acumulacion y sin AMP:\n\n"
            "    opt = optimizer_factory(model.parameters())   # o AdamW si es None\n"
            "    model.train()\n"
            "    repetir `steps` veces:\n"
            "        _, perdida = model(x, y)\n"
            "        opt.zero_grad(set_to_none=True)\n"
            "        perdida.backward()\n"
            "        opt.step()\n"
            "        historial.append(float(perdida.detach()))\n\n"
            "Cuantas menos piezas, menos sitios donde pueda esconderse un bug. Por eso este "
            "bucle es deliberadamente pelado.",

            "Tres detalles:\n\n"
            "1. El `model.train()` NO es decorativo. Si el modelo venia en modo eval, el "
            "dropout estaria desactivado y no estarias probando el mismo camino de codigo "
            "que usara el entrenamiento real. Hay un test que lo comprueba.\n\n"
            "2. `float(perdida.detach())` y no `float(perdida)`: sin el detach, PyTorch "
            "lanza un warning sobre convertir tensores con gradiente a escalares.\n\n"
            "3. `optimizer_factory` es opcional: si es None, usa "
            "`torch.optim.AdamW(params, lr=lr)`.\n\n"
            "AVISO: si la perdida baja a cero DEMASIADO deprisa (en cinco pasos), sospecha de "
            "una fuga de informacion. Revisa que los targets vayan desplazados un token "
            "respecto a la entrada.",
        ),
        "format_eta": (
            "Formatear una duracion para que se lea de un vistazo. Parece cosmetico y no lo "
            "es: vas a mirar ese numero muchas veces durante una tirada de horas, y '1h 2m' "
            "se lee al instante mientras que '3725 s' hay que dividirlo mentalmente.",

            "Cuatro tramos:\n\n"
            "    < 60      ->  '{s}s'\n"
            "    < 3600    ->  '{m}m {s}s'\n"
            "    < 86400   ->  '{h}h {m}m'      <- sin segundos\n"
            "    resto     ->  '{d}d {h}h'\n\n"
            "A partir de una hora se dejan de mostrar los segundos: cuando faltan dos horas, "
            "los segundos son ruido.\n\n"
            "Division entera (`//`) y modulo (`%`) hacen todo el trabajo.",

            "    if not math.isfinite(seconds) or seconds < 0:\n"
            "        return '?'\n"
            "    segundos = int(seconds)\n"
            "    if segundos < 60:    return f'{segundos}s'\n"
            "    if segundos < 3600:  return f'{segundos//60}m {segundos%60}s'\n"
            "    if segundos < 86400: return f'{segundos//3600}h {(segundos%3600)//60}m'\n"
            "    return f'{segundos//86400}d {(segundos%86400)//3600}h'\n\n"
            "El `math.isfinite()` cubre inf, -inf y nan de una vez. Devolver '?' es mas "
            "honesto que inventarse un numero cuando todavia no hay datos para estimar, y "
            "evita imprimir cosas como '-1s' o 'infd 0h'.",
        ),
    },
    # ------------------------------------------------------------------ modulo 14
    "14_inferencia": {
        "apply_repetition_penalty": (
            "Un parche directo contra el texto repetitivo: si un token ya ha aparecido, se "
            "le baja el logit para que sea menos probable que vuelva a salir.",

            "El detalle que casi todo el mundo implementa mal:\n\n"
            "    logit > 0  ->  logit / penalty      lo acerca a cero\n"
            "    logit < 0  ->  logit * penalty      lo aleja de cero, HACIA ABAJO\n\n"
            "Con penalty=2.0:  +3.0 -> +1.5   y   -3.0 -> -6.0\n\n"
            "Si dividieras SIEMPRE, el -3.0 pasaria a -1.5 y el token se volveria MAS "
            "probable. Y como los logits negativos son la mayoria, estarias premiando casi "
            "todo lo que ya salio.",

            "    salida = logits.clone()\n"
            "    for fila in range(logits.shape[0]):\n"
            "        vistos = torch.unique(generated[fila])\n"
            "        valores = salida[fila, vistos]\n"
            "        salida[fila, vistos] = torch.where(\n"
            "            valores > 0, valores / penalty, valores * penalty\n"
            "        )\n"
            "    return salida\n\n"
            "El `torch.unique` evita penalizar dos veces un token que salio dos veces. Y el "
            "`.clone()` es para no modificar la entrada.",
        ),
        "top_k_filter": (
            "Con vocabulario de 4096 hay miles de tokens con probabilidad diminuta pero no "
            "nula. Sumadas, esa cola larga puede llevarse un 20% de la masa, y de vez en "
            "cuando sale una y descarrila la frase.\n\n"
            "Top-k la corta en seco: solo sobreviven los k mayores.",

            "    umbral = torch.topk(logits, k, dim=-1).values[..., -1:]\n"
            "    return logits.masked_fill(logits < umbral, float('-inf'))\n\n"
            "`torch.topk(...).values[..., -1:]` es el k-esimo logit mayor, o sea el umbral.\n\n"
            "Si k <= 0 o k >= vocab_size, devuelve los logits sin tocar.",

            "Dos detalles:\n\n"
            "1. El `[..., -1:]` con DOS PUNTOS conserva la dimension para que el broadcast "
            "funcione. Con `[..., -1]` la perderias y masked_fill compararia mal.\n\n"
            "2. Usa `<` y no `<=`: el propio umbral (el k-esimo logit) tiene que "
            "sobrevivir.\n\n"
            "Su defecto: k es FIJO. Si el modelo esta segurisimo, k=40 mete 39 alternativas "
            "malas; si duda entre 100, corta opciones buenas. Eso lo resuelve top-p.",
        ),
        "top_p_filter": (
            "Como top-k, pero con un numero VARIABLE de candidatos: se acumula probabilidad "
            "hasta llegar a `p` y se corta ahi.\n\n"
            "Si el modelo esta seguro, un token puede acumular el 90% y se queda solo el. Si "
            "duda entre muchos, se quedan muchos. EL NUMERO DE CANDIDATOS SE ADAPTA, y eso "
            "es lo que lo hace mejor que top-k.",

            "    ordenados, indices = torch.sort(logits, descending=True, dim=-1)\n"
            "    probs = F.softmax(ordenados, dim=-1)\n"
            "    acumulada = torch.cumsum(probs, dim=-1)\n\n"
            "    quitar = acumulada - probs > p        # la acumulada ANTES de este token\n"
            "    quitar[..., 0] = False                # el mas probable siempre se queda\n\n"
            "    a_quitar = quitar.scatter(-1, indices, quitar)\n"
            "    return logits.masked_fill(a_quitar, float('-inf'))",

            "Tres cosas:\n\n"
            "1. El `- probs`: se compara la acumulada ANTES de incluir el token actual, asi "
            "que el que CRUZA el umbral todavia entra. La definicion de Holtzman es 'el "
            "conjunto mas pequenyo cuya masa EXCEDE p', y [0.60, 0.25] = 0.85 no excede 0.9: "
            "hace falta el tercero. Es un off-by-one que yo mismo tuve mal escribiendo el "
            "modulo.\n\n"
            "2. El `quitar[..., 0] = False` NO es opcional: con p=0.5 y un token de "
            "probabilidad 0.9 te quedarias sin candidatos y multinomial reventaria.\n\n"
            "3. El `scatter` es lo que mas cuesta ver: has ordenado los logits, asi que las "
            "marcas estan en orden de probabilidad y no de token. `scatter` las devuelve a "
            "su sitio.",
        ),
        "KVCache": (
            "Al generar el token 100, la version ingenua vuelve a pasar los 100 tokens por "
            "el modelo, aunque los 99 primeros no han cambiado. Generar N tokens cuesta "
            "O(N^2) en vez de O(N).\n\n"
            "La cache guarda las claves y valores ya calculados. Lo que NO se puede cachear "
            "son las queries: cada token nuevo necesita su propia pregunta.",

            "La clase es sencilla:\n\n"
            "    __init__(n_layers):  dos listas de n_layers elementos, todos None\n"
            "    update(layer, k, v): si es None, guarda; si no, torch.cat(..., dim=-2)\n"
            "                         y devuelve las K y V COMPLETAS\n"
            "    seq_len:             0 si vacia, si no keys[0].shape[-2]\n"
            "    reset():             vuelve a poner todo a None\n"
            "    memory_bytes():      suma numel() * element_size() de lo que no sea None",

            "El `dim=-2` es la dimension de tiempo con la forma (B, n_heads, T, head_dim). "
            "Usa indice NEGATIVO: con dim=2 te funcionaria aqui y se romperia si algun dia "
            "cambia el numero de dimensiones.\n\n"
            "La dificultad de este modulo no esta aqui, esta en el ejercicio 5: hacer que la "
            "cache de EXACTAMENTE el mismo resultado.",
        ),
        "generate_with_cache": (
            "El mismo bucle autorregresivo del modulo 00 (contexto -> distribucion -> "
            "muestrear -> anyadir -> repetir), ahora con un modelo de verdad y con cache.\n\n"
            "Dos fases: PREFILL (se pasa el prompt entero y se llena la cache) y DECODE (en "
            "cada paso entra SOLO el token nuevo).",

            "El orden de los filtros importa:\n\n"
            "    1. penalizacion de repeticion   (sobre los logits crudos)\n"
            "    2. temperatura                  (dividir)\n"
            "    3. top-k\n"
            "    4. top-p\n\n"
            "La temperatura va antes de los filtros porque cambia las masas acumuladas que "
            "mira top-p.\n\n"
            "Y usa `logits[:, -1, :].float()`: bajo AMP los logits llegan en fp16 y "
            "multinomial puede dar resultados raros con probabilidades muy pequenyas.",

            "EL DETALLE QUE ROMPE TODO: RoPE tiene que rotar el token nuevo con el angulo de "
            "su POSICION REAL.\n\n"
            "Al generar el token 50 le pasas un tensor de longitud 1. Si aplicas RoPE tal "
            "cual, lo rota como si fuera la posicion 0, y la generacion con cache produce "
            "texto distinto y peor que sin ella. Nada falla: el modelo simplemente escribe "
            "mal.\n\n"
            "Por eso la atencion recibe `pos_offset` y recorta las tablas:\n"
            "    cos_t = cos[pos_offset : pos_offset + seq_len]\n\n"
            "Hay un test que compara con la generacion sin cache y exige salida IDENTICA.\n\n"
            "Y PARA al llegar al contexto maximo, en vez de recortar: recortar con cache "
            "exigiria remapear las posiciones de RoPE de todo lo que queda (sliding window "
            "attention). Parar es lo honesto.",
        ),
    },
    # ------------------------------------------------------------------ modulo 15
    "15_evaluacion": {
        "perplexity_from_loss": (
            "La perdida en nats no se lee bien: 1.60 no dice mucho. La perplejidad si, "
            "porque se interpreta como ENTRE CUANTAS OPCIONES equiprobables esta dudando el "
            "modelo.",

            "    perplejidad = exp(perdida)\n\n"
            "    perdida 8.317  ->  4096   (sin entrenar: duda entre todo el vocabulario)\n"
            "    perdida 1.60   ->  4.95   (duda entre unas 5 opciones)\n"
            "    perdida 0      ->  1      (perfecto)\n\n"
            "Es una linea. Lo que importa es saber interpretarla.",

            "    if not math.isfinite(loss):\n"
            "        return float('inf')\n"
            "    return math.exp(loss)\n\n"
            "La guarda no es decorativa: `math.exp(inf)` lanza OverflowError, y en medio de "
            "una evaluacion eso te deja sin saber que paso.\n\n"
            "La comprobacion util: con perdida ln(V), la perplejidad es exactamente V.",
        ),
        "bits_per_byte": (
            "La perplejidad depende del tokenizador: si tu vocabulario parte las palabras en "
            "trozos mas pequenyos, cada token es mas facil de predecir y tu numero sale "
            "mejor SIN QUE EL MODELO SEA MEJOR.\n\n"
            "Comparar perplejidades entre modelos con tokenizadores distintos no significa "
            "nada, y se hace constantemente.\n\n"
            "La solucion: normalizar por BYTES del texto original.",

            "    bits_por_byte = (perdida_total_en_nats / ln(2)) / n_bytes\n\n"
            "El `/ ln(2)` convierte nats a bits: un nat son 1.4427 bits.\n\n"
            "OJO: el primer argumento es la perdida TOTAL (la suma), no la media. El "
            "parametro `n_tokens` no se usa en el calculo; esta en la firma para dejarlo "
            "claro.",

            "    if n_bytes <= 0:\n"
            "        raise ValueError(...)\n"
            "    return total_loss_nats / math.log(2) / n_bytes\n\n"
            "La interpretacion, que es bonita: es cuantos bits necesitarias para transmitir "
            "el texto usando el modelo como compresor. Referencias: gzip ~2.5 bits/byte, un "
            "buen modelo pequenyo ~1.2, los mejores LLM 0.6-0.8.\n\n"
            "No es una analogia. Un modelo de lenguaje ES un compresor, y la equivalencia "
            "entre prediccion y compresion viene de Shannon (1948).",
        ),
        "run_prompt_battery": (
            "La parte de la evaluacion que ninguna metrica automatica sustituye: hacerle al "
            "modelo siempre las mismas preguntas y LEER lo que responde.\n\n"
            "Los seis prompts de `PROMPTS_TINYSTORIES` vienen del paper y cada uno prueba "
            "algo distinto: continuacion, coherencia causal, seguimiento de objeto, "
            "resolucion y cierre.",

            "    prompts = prompts or PROMPTS_TINYSTORIES\n"
            "    return [\n"
            "        {'prompt': p, 'que_prueba': etiqueta, 'completion': generate_fn(p)}\n"
            "        for p, etiqueta in prompts\n"
            "    ]\n\n"
            "`PROMPTS_TINYSTORIES` es una tupla de tuplas (prompt, etiqueta), asi que se "
            "desempaqueta directamente. Las tres claves tienen que llamarse exactamente asi.",

            "Por que se pasa `generate_fn` en vez del modelo: encapsula el modelo Y el "
            "tokenizador, asi que esta funcion no sabe nada de ninguno de los dos. Es el "
            "mismo patron que `get_batch` en el modulo 04 y `optimizer_factory` en el 13.\n\n"
            "Y hace el ejercicio testeable con un generador falso.\n\n"
            "El valor del ejercicio no esta en el codigo (son tres lineas): esta en tener "
            "una bateria FIJA que puedas volver a pasar cada vez que cambies algo.",
        ),
    },
    # ------------------------------------------------------------------ modulo 16
    "16_finetuning": {
        "build_chat_template": (
            "Un modelo preentrenado solo sabe continuar texto. Si le escribes una pregunta, "
            "lo mas probable es que responda con MAS preguntas: un documento que empieza asi "
            "suele seguir asi.\n\n"
            "Para que RESPONDA hay que enseñarle un formato. Eso son los marcadores.",

            "    [{'role': 'user', 'content': 'Hola'},\n"
            "     {'role': 'assistant', 'content': 'Que tal'}]\n\n"
            "    ->  <|user|>Hola<|end|><|assistant|>Que tal<|end|>\n\n"
            "Cada mensaje se envuelve en el marcador de su rol y se cierra con `<|end|>`. "
            "Los marcadores estan en `CHAT_MARKERS`.\n\n"
            "Con `add_generation_prompt=True` se anyade `<|assistant|>` al final y se deja la "
            "cadena ABIERTA: es lo que se usa en inferencia.",

            "    partes = []\n"
            "    for mensaje in messages:\n"
            "        rol = mensaje['role']\n"
            "        if rol not in CHAT_MARKERS:\n"
            "            raise ValueError(...)\n"
            "        partes.append(f\"{CHAT_MARKERS[rol]}{mensaje['content']}"
            "{CHAT_MARKERS['end']}\")\n"
            "    if add_generation_prompt:\n"
            "        partes.append(CHAT_MARKERS['assistant'])\n"
            "    return ''.join(partes)\n\n"
            "El `<|end|>` es lo que le enseña al modelo CUANDO PARAR. Sin un marcador de fin, "
            "generaria indefinidamente.",
        ),
        "mask_prompt_tokens": (
            "En SFT no quieres que el modelo aprenda a GENERAR las preguntas del usuario: "
            "quieres que aprenda a RESPONDERLAS.\n\n"
            "Poniendo -100 en las posiciones del prompt, `F.cross_entropy(..., "
            "ignore_index=-100)` las salta.",

            "    input_ids = [10, 11, 12, 20, 21, 22]   con prompt_len = 3\n"
            "    targets   = [-100, -100, 20, 21, 22, -100]\n\n"
            "Fijate: DOS posiciones ignoradas al principio, no tres. Y una al final.\n\n"
            "    targets = [ignore_index] * len(input_ids)\n"
            "    para i desde prompt_len - 1 hasta len(input_ids) - 2:\n"
            "        targets[i] = input_ids[i + 1]",

            "POR QUE DOS Y NO TRES, que es todo el ejercicio.\n\n"
            "Los targets van DESPLAZADOS un token. Asi que en la posicion 2 (el ULTIMO token "
            "del prompt) el objetivo ya es input_ids[3] = 20, que es el primer token de la "
            "RESPUESTA.\n\n"
            "Y ese si interesa: es justo la transicion 'se acabo la pregunta, me toca "
            "responder', que es lo mas importante que el modelo tiene que aprender del SFT. "
            "Si empezaras en prompt_len, te saltarias precisamente esa transicion.\n\n"
            "Y no da ninguna senyal: solo desperdicias la posicion mas informativa.",
        ),
        "LoRALinear": (
            "Hacer fine-tuning completo de un modelo grande necesita memoria para los pesos, "
            "los gradientes Y los estados de Adam: unos 12 bytes por parametro.\n\n"
            "LoRA parte de una observacion: los cambios que hace el fine-tuning tienen RANGO "
            "BAJO. Asi que se CONGELA W y se le suma el producto de dos matrices flacas.",

            "    salida = x @ W^T  +  (alpha/r) * x @ A^T @ B^T\n\n"
            "con A de (r, d_in) y B de (d_out, r).\n\n"
            "Con d_in = d_out = 320 y r = 8:\n"
            "    W entera:  102.400 parametros\n"
            "    A y B:       5.120 parametros    (el 5%)\n\n"
            "Submodulos: `base` (la capa original), `lora_A`, `lora_B` y `lora_dropout`.\n\n"
            "Y no olvides congelar la base:\n"
            "    for p in self.base.parameters():\n"
            "        p.requires_grad = False",

            "LA INICIALIZACION NO ES SIMETRICA, y es lo importante:\n\n"
            "    lora_A -> nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))\n"
            "    lora_B -> CEROS\n\n"
            "Con B = 0, el producto BA vale cero al empezar y la capa es EXACTAMENTE la "
            "original: el fine-tuning arranca sin perturbar nada. Hay un test que lo "
            "comprueba.\n\n"
            "¿Y por que no las dos a cero? Porque entonces el gradiente de ambas seria cero "
            "-cada una multiplica a la otra- y nunca aprenderian.\n\n"
            "Cuidado con las transpuestas: lora_A es (r, d_in), asi que `x @ lora_A.T` da "
            "(..., r), y luego `@ lora_B.T` con lora_B de (d_out, r) da (..., d_out).",
        ),
        "merge_lora_weights": (
            "Durante el entrenamiento, LoRA anyade dos matmuls por capa, y eso se nota en "
            "inferencia.\n\n"
            "Fundiendo los adaptadores en la matriz base, el modelo resultante es "
            "INDISTINGUIBLE de uno normal: mismo coste, mismas formas, y se puede servir sin "
            "ninguna dependencia de LoRA.",

            "    W_nueva = W + (alpha/r) * (B @ A)\n\n"
            "Fijate en el orden `B @ A`: B es (d_out, r) y A es (r, d_in), asi que el "
            "producto da (d_out, d_in), que es la forma de `weight` en un nn.Linear. Al reves "
            "las formas no cuadrarian.\n\n"
            "Devuelve un `nn.Linear` normal, conservando el sesgo si lo habia.",

            "    fundida = nn.Linear(d_in, d_out, bias=layer.base.bias is not None)\n"
            "    with torch.no_grad():\n"
            "        delta = (layer.lora_B @ layer.lora_A) * layer.scaling\n"
            "        fundida.weight.copy_(layer.base.weight + delta)\n"
            "        if layer.base.bias is not None:\n"
            "            fundida.bias.copy_(layer.base.bias)\n"
            "    return fundida\n\n"
            "Usa `.copy_()` en vez de asignar: asi conservas el tensor que ya creo nn.Linear "
            "con sus metadatos.\n\n"
            "Esto es lo que hace util a LoRA frente a otros metodos de fine-tuning "
            "eficiente: la adaptacion es EXACTAMENTE una suma de matrices, asi que se "
            "absorbe sin aproximar nada.",
        ),
    },
    # ------------------------------------------------------------------ modulo 17
    "17_extra": {
        "quantize_int8_symmetric": (
            "Un float32 ocupa 4 bytes y un int8 uno solo: el modelo ocupa la cuarta parte.\n\n"
            "El truco es guardar, junto a los enteros, una ESCALA que permite recuperar los "
            "valores aproximados.",

            "Con W = [0.12, -0.45, 0.03, 0.28], el mayor en valor absoluto es 0.45:\n\n"
            "    escala = 0.45 / 127 = 0.003543\n"
            "    W_int8 = round(W / escala) = [34, -127, 8, 79]\n\n"
            "Los pasos:\n"
            "    1. max_abs = weight.abs().amax(dim=-1, keepdim=True)   si es por canal\n"
            "    2. escala = (max_abs / 127.0).clamp_min(1e-12)\n"
            "    3. torch.round(weight / escala).clamp(-127, 127).to(torch.int8)",

            "Tres detalles:\n\n"
            "1. POR QUE 127 Y NO 128: int8 va de -128 a 127. Con 127 el rango queda "
            "SIMETRICO y el cero se representa exactamente. En una matriz con muchos valores "
            "pequenyos, eso evita un sesgo sistematico que se acumularia capa tras capa.\n\n"
            "2. El `clamp_min(1e-12)` evita dividir por cero si una fila es todo ceros.\n\n"
            "3. El `clamp(-127, 127)` protege del redondeo en el borde: sin el, un valor "
            "justo en el maximo podria dar 128, que no cabe en int8 y haria wrap a -128. El "
            "peso mas grande se convertiria en el mas negativo, en silencio.\n\n"
            "Por canal siempre es mejor que por tensor (0.71% frente a 1.07% de error): una "
            "sola fila con valores grandes no arrastra a las demas.",
        ),
        "dequantize_int8": (
            "Lo contrario del ejercicio 1: multiplicar por la escala para volver a float.\n\n"
            "El resultado NO es igual al original: se ha perdido informacion. Eso es lo que "
            "se paga por ocupar la cuarta parte.",

            "    return quantized.to(torch.float32) * scale\n\n"
            "Una linea.",

            "El `.to(torch.float32)` va ANTES de multiplicar. Si multiplicaras el int8 "
            "directamente, PyTorch haria la operacion en enteros y el resultado seria "
            "basura.",
        ),
        "quantization_error": (
            "Cuantizar, descuantizar, y comparar con el original. Es la forma de saber si "
            "merece la pena antes de aplicarlo al modelo entero.",

            "Las metricas que hay que devolver:\n\n"
            "    error_relativo   = ||original - recuperado|| / ||original||\n"
            "    error_maximo     = max(|diferencia|)\n"
            "    error_medio      = mean(|diferencia|)\n"
            "    compresion       = bytes por elemento original / cuantizado\n"
            "    bytes_original   = numel() * element_size()\n"
            "    bytes_cuantizado = los del int8 MAS los de las escalas\n\n"
            "`tensor.element_size()` da los bytes por elemento (4 para float32, 1 para "
            "int8). Con eso la compresion sale sola, sin numeros magicos.",

            "El error RELATIVO es la metrica que conviene mirar: es independiente de la "
            "escala de los datos, asi que puedes comparar capas distintas. Hay un test que "
            "multiplica los pesos por 1000 y comprueba que no cambia.\n\n"
            "Y las escalas TAMBIEN OCUPAN: incluirlas en `bytes_cuantizado` es lo honesto. "
            "Con una por fila son despreciables, pero contarlas cuesta nada.\n\n"
            "Con pesos de una red entrenada, int8 por canal ronda el 0.5-1% de error. Que "
            "eso apenas afecte a la calidad del modelo es un HECHO EMPIRICO, no un teorema.",
        ),
    },
}


def get_hints(module_id: str, exercise_name: str) -> tuple[str, ...]:
    """Pistas de un ejercicio, de menos a mas explicita. Tupla vacia si no hay."""
    return HINTS.get(module_id, {}).get(exercise_name, ())


def has_hints(module_id: str) -> bool:
    return bool(HINTS.get(module_id))
