"""El curriculo: que modulos hay, que ejercicios tiene cada uno y en que orden.

Esta es la unica fuente de verdad del curso. La CLI (`status`, `next`, `check`, `demo`,
`hint`) y el bridge (la red de seguridad) leen de aqui. Si anyades un ejercicio, se anyade
en este fichero y todo lo demas se entera solo.

Los tiempos (`est_minutes`) son estimaciones de trabajo con las manos en el teclado para
alguien que ya programa, SIN contar el tiempo de GPU. No incluyen leer la teoria dos veces
porque la primera no te enteraste, que pasa y es normal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from llmfs.paths import modules_dir


@dataclass(frozen=True)
class Exercise:
    """Un ejercicio: una funcion o clase que tienes que implementar en `ejercicios.py`."""

    #: Nombre exacto del simbolo en `ejercicios.py`. Es la clave que usa el bridge.
    name: str
    #: Que hace, en una linea.
    title: str
    #: `True` si es una clase (`nn.Module` u otra) en lugar de una funcion.
    is_class: bool = False


@dataclass(frozen=True)
class Module:
    """Un modulo del curso."""

    number: int
    slug: str
    title: str
    part: str
    summary: str
    est_minutes: int
    exercises: tuple[Exercise, ...] = field(default_factory=tuple)
    has_demo: bool = True
    #: Papers y enlaces que se citan en el TEORIA.md.
    references: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def id(self) -> str:
        """Identificador de directorio, p.ej. `06_atencion`."""
        return f"{self.number:02d}_{self.slug}"

    @property
    def path(self) -> Path:
        return modules_dir() / self.id

    @property
    def test_file(self) -> Path:
        return self.path / f"test_{self.number:02d}.py"

    @property
    def exercises_file(self) -> Path:
        return self.path / "ejercicios.py"

    @property
    def demo_file(self) -> Path:
        return self.path / "demo.py"

    @property
    def theory_file(self) -> Path:
        return self.path / "TEORIA.md"

    @property
    def solution_file(self) -> Path:
        return self.path / "SOLUCION.md"

    def exercise(self, name_or_index: str | int) -> Exercise:
        """Busca un ejercicio por nombre o por posicion (1-indexada, como se muestra)."""
        if isinstance(name_or_index, int):
            if not 1 <= name_or_index <= len(self.exercises):
                raise KeyError(
                    f"El modulo {self.id} tiene {len(self.exercises)} ejercicios; "
                    f"has pedido el {name_or_index}."
                )
            return self.exercises[name_or_index - 1]
        for ex in self.exercises:
            if ex.name == name_or_index:
                return ex
        raise KeyError(f"No hay ningun ejercicio {name_or_index!r} en {self.id}.")


PART_0 = "0 - Antes de empezar"
PART_I = "I - Fundamentos"
PART_II = "II - Arquitectura"
PART_III = "III - Entrenamiento"
PART_IV = "IV - Uso y evaluacion"


CURRICULUM: tuple[Module, ...] = (
    # ------------------------------------------------------------------ Parte 0
    Module(
        number=0,
        slug="que_es_un_llm",
        title="Que es un LLM, en realidad",
        part=PART_0,
        summary="Sin jerga y sin transformers: construyes un generador de texto en 20 lineas.",
        est_minutes=60,
        exercises=(
            Exercise("next_token_probs", "Convertir conteos en probabilidades que sumen 1"),
            Exercise("sample_next_token", "Elegir el siguiente caracter segun esas probabilidades"),
            Exercise("generate_naive", "Encadenar predicciones para generar texto"),
        ),
        references=(
            ("Shannon 1948, A Mathematical Theory of Communication", "https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf"),
            ("Bengio et al. 2003, A Neural Probabilistic Language Model", "https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf"),
        ),
    ),
    # ------------------------------------------------------------------ Parte I
    Module(
        number=1,
        slug="entorno",
        title="Entorno y hardware",
        part=PART_I,
        summary="uv, PyTorch, CUDA/MPS, y cuantos tokens/s da realmente tu GPU.",
        est_minutes=45,
        exercises=(
            Exercise("measure_matmul_tflops", "Medir los TFLOPS reales de un matmul grande"),
            Exercise("transformer_flops_per_token", "FLOPs por token de un transformer (6N + atencion)"),
            Exercise("estimate_tokens_per_second", "Tokens/s teoricos a partir de TFLOPS y MFU"),
        ),
        references=(
            ("Kaplan et al. 2020 (apendice de FLOPs)", "https://arxiv.org/abs/2001.08361"),
        ),
    ),
    Module(
        number=2,
        slug="autograd",
        title="Autodiferenciacion desde cero",
        part=PART_I,
        summary="Un motor escalar estilo micrograd. Backprop deja de ser magia.",
        est_minutes=180,
        exercises=(
            Exercise("Value", "Escalar con grafo de computo y regla de la cadena", is_class=True),
            Exercise("topological_order", "Orden topologico del grafo para el backward"),
            Exercise("train_scalar_mlp", "Entrenar un MLP usando solo tu motor"),
        ),
        references=(
            ("Karpathy, micrograd", "https://github.com/karpathy/micrograd"),
            ("Baydin et al. 2018, Automatic Differentiation in ML", "https://arxiv.org/abs/1502.05767"),
        ),
    ),
    Module(
        number=3,
        slug="tokenizacion",
        title="Tokenizacion y BPE",
        part=PART_I,
        summary="De caracteres a BPE propio de 4096 tokens entrenado sobre TinyStories.",
        est_minutes=240,
        exercises=(
            Exercise("get_stats", "Contar pares adyacentes"),
            Exercise("merge", "Fusionar un par en un id nuevo"),
            Exercise("train_bpe", "Entrenar los merges hasta vocab_size"),
            Exercise("bpe_encode", "Texto -> ids aplicando merges en orden"),
            Exercise("bpe_decode", "ids -> texto con bytes fallback"),
        ),
        references=(
            ("Sennrich et al. 2016, Neural MT of Rare Words with Subword Units", "https://arxiv.org/abs/1508.07909"),
            ("Karpathy, minbpe", "https://github.com/karpathy/minbpe"),
        ),
    ),
    Module(
        number=4,
        slug="datos",
        title="Datos: de texto a memmap",
        part=PART_I,
        summary="Descarga, tokenizacion paralela, uint16 en disco y dataloader de ventanas.",
        est_minutes=120,
        exercises=(
            Exercise("pack_tokens_uint16", "Empaquetar ids en un array uint16 validado"),
            Exercise("train_val_split", "Partir el corpus sin filtrar informacion"),
            Exercise("get_batch", "Muestrear ventanas (x, y) desplazadas un token"),
        ),
    ),
    # ------------------------------------------------------------------ Parte II
    Module(
        number=5,
        slug="baselines",
        title="Baselines: bigrama y MLP de Bengio",
        part=PART_II,
        summary="Cross-entropy, perplejidad y cual es el suelo de un modelo trivial.",
        est_minutes=120,
        exercises=(
            Exercise("uniform_baseline_loss", "La perdida de adivinar al azar: ln(V)"),
            Exercise("bigram_counts", "Matriz de conteos V x V"),
            Exercise("bigram_nll", "Log-verosimilitud negativa con suavizado"),
            Exercise("NeuralBigram", "El bigrama como una sola capa de embedding", is_class=True),
            Exercise("BengioMLP", "MLP con ventana de contexto (Bengio 2003)", is_class=True),
        ),
        references=(
            ("Bengio et al. 2003, A Neural Probabilistic Language Model", "https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf"),
        ),
    ),
    Module(
        number=6,
        slug="atencion",
        title="Self-attention",
        part=PART_II,
        summary="Q/K/V, mascara causal, escalado por sqrt(d_k) y multi-head.",
        est_minutes=240,
        exercises=(
            Exercise("causal_mask", "Mascara triangular que impide mirar al futuro"),
            Exercise("single_head_attention", "softmax(QK^T/sqrt(d_k) + mask) V"),
            Exercise("MultiHeadAttention", "Multi-head con proyeccion de salida", is_class=True),
        ),
        references=(
            ("Vaswani et al. 2017, Attention Is All You Need", "https://arxiv.org/abs/1706.03762"),
            ("Elhage et al. 2021, A Mathematical Framework for Transformer Circuits", "https://transformer-circuits.pub/2021/framework/index.html"),
        ),
    ),
    Module(
        number=7,
        slug="normalizacion",
        title="Normalizacion y residuales",
        part=PART_II,
        summary="LayerNorm -> RMSNorm, pre-norm vs post-norm, y por que el gradiente sobrevive.",
        est_minutes=90,
        exercises=(
            Exercise("layer_norm", "Normalizar por media y varianza, con gamma y beta"),
            Exercise("RMSNorm", "Solo escala por la raiz del cuadrado medio", is_class=True),
            Exercise("prenorm_residual", "x + f(norm(x)) frente a norm(x + f(x))"),
        ),
        references=(
            ("Ba et al. 2016, Layer Normalization", "https://arxiv.org/abs/1607.06450"),
            ("Zhang & Sennrich 2019, Root Mean Square Layer Normalization", "https://arxiv.org/abs/1910.07467"),
            ("Xiong et al. 2020, On Layer Normalization in the Transformer Architecture", "https://arxiv.org/abs/2002.04745"),
        ),
    ),
    Module(
        number=8,
        slug="mlp_y_activaciones",
        title="FFN, GELU y SwiGLU",
        part=PART_II,
        summary="El ratio 4x, por que SwiGLU usa 2/3 del hidden, y donde estan los parametros.",
        est_minutes=90,
        exercises=(
            Exercise("gelu", "GELU con la aproximacion tanh"),
            Exercise("swiglu_hidden_dim", "Calcular d_ff = round(2/3 * 4d) al multiplo de 64"),
            Exercise("SwiGLU", "FFN con puerta: (Swish(xW1) * xW3) W2", is_class=True),
        ),
        references=(
            ("Hendrycks & Gimpel 2016, Gaussian Error Linear Units", "https://arxiv.org/abs/1606.08415"),
            ("Shazeer 2020, GLU Variants Improve Transformer", "https://arxiv.org/abs/2002.05202"),
        ),
    ),
    Module(
        number=9,
        slug="posicion",
        title="Informacion posicional y RoPE",
        part=PART_II,
        summary="Aprendidas -> sinusoidales -> RoPE, y que pasa al extrapolar el contexto.",
        est_minutes=150,
        exercises=(
            Exercise("sinusoidal_embeddings", "La tabla seno/coseno del paper original"),
            Exercise("rope_frequencies", "Tablas cos/sin para theta^(-2i/d)"),
            Exercise("apply_rope", "Rotar pares de dimensiones de Q y K"),
        ),
        references=(
            ("Su et al. 2021, RoFormer: Rotary Position Embedding", "https://arxiv.org/abs/2104.09864"),
            ("Press et al. 2021, ALiBi", "https://arxiv.org/abs/2108.12409"),
        ),
    ),
    Module(
        number=10,
        slug="el_gpt_completo",
        title="El GPT completo",
        part=PART_II,
        summary="Block, GPT, weight tying, init escalada por profundidad y el conteo exacto.",
        est_minutes=180,
        exercises=(
            Exercise("expected_param_count", "La formula de parametros derivada a mano"),
            Exercise("count_parameters", "Conteo real desglosado por componente"),
            Exercise("TransformerBlock", "Atencion + FFN con pre-norm y residuales", is_class=True),
            Exercise("GPT", "El modelo entero, con tying e init escalada", is_class=True),
        ),
        references=(
            ("Radford et al. 2019, Language Models are Unsupervised Multitask Learners (GPT-2)", "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"),
            ("Press & Wolf 2017, Using the Output Embedding to Improve Language Models", "https://arxiv.org/abs/1608.05859"),
        ),
    ),
    # ------------------------------------------------------------------ Parte III
    Module(
        number=11,
        slug="bucle_de_entrenamiento",
        title="El bucle de entrenamiento",
        part=PART_III,
        summary="AdamW propio, warmup+cosine, clipping, acumulacion, AMP y checkpoints.",
        est_minutes=240,
        exercises=(
            Exercise("AdamWScratch", "AdamW desde cero, con correccion de sesgo", is_class=True),
            Exercise("lr_at_step", "Warmup lineal + decaimiento coseno hasta el 10%"),
            Exercise("clip_grad_norm", "Recorte por norma global"),
            Exercise("build_param_groups", "Sin weight decay en normas ni sesgos"),
        ),
        references=(
            ("Loshchilov & Hutter 2019, Decoupled Weight Decay Regularization", "https://arxiv.org/abs/1711.05101"),
            ("Micikevicius et al. 2018, Mixed Precision Training", "https://arxiv.org/abs/1710.03740"),
        ),
    ),
    Module(
        number=12,
        slug="eficiencia_y_escalado",
        title="Eficiencia y leyes de escala",
        part=PART_III,
        summary="MFU, memoria, KV cache, Chinchilla y por que 9M parametros y 500M tokens.",
        est_minutes=120,
        exercises=(
            Exercise("model_flops_per_token", "FLOPs por token separando matmuls y atencion"),
            Exercise("compute_mfu", "Model FLOPs Utilization frente al pico del hardware"),
            Exercise("chinchilla_optimal_allocation", "Reparto optimo de un presupuesto de computo"),
        ),
        references=(
            ("Kaplan et al. 2020, Scaling Laws for Neural Language Models", "https://arxiv.org/abs/2001.08361"),
            ("Hoffmann et al. 2022, Training Compute-Optimal LLMs (Chinchilla)", "https://arxiv.org/abs/2203.15556"),
            ("Chowdhery et al. 2022, PaLM (definicion de MFU)", "https://arxiv.org/abs/2204.02311"),
        ),
    ),
    Module(
        number=13,
        slug="entrenamiento_final",
        title="La tirada real",
        part=PART_III,
        summary="TinyStories de principio a fin: reanudable, con muestras periodicas y ETA.",
        est_minutes=60,
        exercises=(
            Exercise("overfit_single_batch", "Memorizar un batch hasta perdida ~0: el test que caza bugs"),
            Exercise("format_eta", "ETA honesta a partir del ritmo medido"),
        ),
        references=(
            ("Eldan & Li 2023, TinyStories", "https://arxiv.org/abs/2305.07759"),
        ),
    ),
    # ------------------------------------------------------------------ Parte IV
    Module(
        number=14,
        slug="inferencia",
        title="Inferencia y muestreo",
        part=PART_IV,
        summary="Temperatura, top-k, top-p, penalizacion de repeticion y KV cache.",
        est_minutes=180,
        exercises=(
            Exercise("apply_repetition_penalty", "Penalizar tokens ya generados"),
            Exercise("top_k_filter", "Quedarse con los k logits mayores"),
            Exercise("top_p_filter", "Nucleo de masa acumulada p"),
            Exercise("KVCache", "Cache de claves y valores por capa", is_class=True),
            Exercise("generate_with_cache", "Generacion incremental con la cache"),
        ),
        references=(
            ("Holtzman et al. 2020, The Curious Case of Neural Text Degeneration", "https://arxiv.org/abs/1904.09751"),
        ),
    ),
    Module(
        number=15,
        slug="evaluacion",
        title="Evaluacion",
        part=PART_IV,
        summary="Perplejidad, bits por byte y la bateria cualitativa del paper TinyStories.",
        est_minutes=120,
        exercises=(
            Exercise("perplexity_from_loss", "Perplejidad a partir de la perdida media"),
            Exercise("bits_per_byte", "Metrica comparable entre tokenizadores distintos"),
            Exercise("run_prompt_battery", "Generar completions de los prompts fijos"),
        ),
        references=(
            ("Eldan & Li 2023, TinyStories (seccion de evaluacion)", "https://arxiv.org/abs/2305.07759"),
        ),
    ),
    Module(
        number=16,
        slug="finetuning",
        title="Post-training: SFT y LoRA",
        part=PART_IV,
        summary="Chat template, perdida solo en la respuesta, y LoRA implementado a mano.",
        est_minutes=180,
        exercises=(
            Exercise("build_chat_template", "Serializar mensajes a texto con marcadores"),
            Exercise("mask_prompt_tokens", "Ignorar el prompt en la perdida (-100)"),
            Exercise("LoRALinear", "W + (alpha/r) B A con A,B de rango bajo", is_class=True),
            Exercise("merge_lora_weights", "Fundir los adaptadores en los pesos base"),
        ),
        references=(
            ("Hu et al. 2021, LoRA", "https://arxiv.org/abs/2106.09685"),
            ("Ouyang et al. 2022, InstructGPT", "https://arxiv.org/abs/2203.02155"),
        ),
    ),
    Module(
        number=17,
        slug="extra",
        title="Extras y limites honestos",
        part=PART_IV,
        summary="Cuantizacion int8, servir con FastAPI, y que separa esto de un modelo frontier.",
        est_minutes=120,
        exercises=(
            Exercise("quantize_int8_symmetric", "Pesos a int8 con escala por canal"),
            Exercise("dequantize_int8", "Vuelta a float y error de reconstruccion"),
            Exercise("quantization_error", "Medir el danyo: error relativo y perplejidad"),
        ),
    ),
)


# ---------------------------------------------------------------------------- lookup

_BY_ID = {m.id: m for m in CURRICULUM}
_BY_NUMBER = {m.number: m for m in CURRICULUM}


def all_modules() -> tuple[Module, ...]:
    return CURRICULUM


def get_module(ref: str | int) -> Module:
    """Resuelve un modulo desde `5`, `"5"`, `"05"` o `"05_atencion"`.

    Raises:
        KeyError: con la lista de modulos validos, para que el error sea util.
    """
    if isinstance(ref, int):
        if ref in _BY_NUMBER:
            return _BY_NUMBER[ref]
        raise KeyError(f"No existe el modulo {ref}. Van del 0 al {len(CURRICULUM) - 1}.")

    ref = str(ref).strip()
    if ref in _BY_ID:
        return _BY_ID[ref]
    if ref.isdigit() and int(ref) in _BY_NUMBER:
        return _BY_NUMBER[int(ref)]
    # Busqueda por prefijo del slug, para poder escribir `llmfs check aten`.
    matches = [m for m in CURRICULUM if m.slug.startswith(ref.lower())]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        f"No se reconoce el modulo {ref!r}. Validos: " + ", ".join(m.id for m in CURRICULUM)
    )


def parts() -> dict[str, list[Module]]:
    """Modulos agrupados por parte, en orden."""
    grouped: dict[str, list[Module]] = {}
    for module in CURRICULUM:
        grouped.setdefault(module.part, []).append(module)
    return grouped


def total_exercises() -> int:
    return sum(len(m.exercises) for m in CURRICULUM)


def total_minutes() -> int:
    return sum(m.est_minutes for m in CURRICULUM)
