"""Typed configuration for the model, the data and the training run.

A single place where "what exactly am I training" lives. The YAML files in `configs/` are
loaded here and validated on construction: a typo in a key blows up at startup, not three
hours into the run.

The loader accepts flat YAML or YAML split into sections. These two are equivalent:

    # flat                       # sectioned
    n_layers: 6                  model:
    d_model: 320                   n_layers: 6
    lr: 1.0e-3                     d_model: 320
                                 train:
                                   lr: 1.0e-3
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

NormKind = Literal["rmsnorm", "layernorm"]
PosKind = Literal["rope", "learned", "sinusoidal", "none"]
ActivationKind = Literal["swiglu", "gelu", "relu"]
ScheduleKind = Literal["cosine", "linear", "constant"]


@dataclass
class ModelConfig:
    """Architecture. Everything that determines the parameter count lives here."""

    vocab_size: int = 4096
    n_layers: int = 6
    d_model: int = 320
    n_heads: int = 8
    d_ff: int = 896
    context_length: int = 512
    norm: NormKind = "rmsnorm"
    pos: PosKind = "rope"
    activation: ActivationKind = "swiglu"
    tie_embeddings: bool = True
    dropout: float = 0.0
    #: Biases in the linear projections. Modern LLMs drop them: they add little and they
    #: complicate weight decay. See module 09.
    bias: bool = False
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}); "
                f"right now that gives {self.d_model / self.n_heads:.2f} per head."
            )
        if self.pos == "rope" and self.head_dim % 2 != 0:
            raise ValueError(
                f"RoPE rotates pairs of dimensions, so head_dim ({self.head_dim}) "
                "has to be even."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout out of range: {self.dropout}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class DataConfig:
    """Where the tokens come from."""

    dataset: str = "tinystories"
    data_dir: str = "data"
    tokenizer: Literal["bpe", "char"] = "bpe"
    #: How much text is used to TRAIN the BPE (not to train the model).
    #: See module 03: training merges over 2 GB in pure python is not viable.
    bpe_train_bytes: int = 150_000_000
    val_fraction: float = 0.005
    num_workers: int = 4


@dataclass
class TrainConfig:
    """The training loop."""

    optimizer: Literal["adamw", "adamw_scratch"] = "adamw"
    lr: float = 1.0e-3
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 500
    schedule: ScheduleKind = "cosine"
    #: The cosine decays down to this fraction of the lr, not to zero.
    min_lr_ratio: float = 0.1
    batch_size: int = 48
    grad_accum: int = 2
    max_tokens: int = 500_000_000
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 10
    sample_interval: int = 500
    seed: int = 1337
    #: Off by default: on Turing (sm_75) it causes more grief than it is worth.
    compile: bool = False
    #: `None` = the backend's default policy (see llmfs/device.py).
    amp: bool | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        self.betas = tuple(self.betas)  # type: ignore[assignment]
        if len(self.betas) != 2:
            raise ValueError(f"betas must have 2 elements, not {len(self.betas)}")
        if self.warmup_steps < 0 or self.grad_accum < 1 or self.batch_size < 1:
            raise ValueError("warmup_steps>=0, grad_accum>=1 and batch_size>=1")

    @property
    def tokens_per_step(self) -> int:
        """Tokens consumed per optimizer step. Needs the context_length."""
        raise NotImplementedError("Use RunConfig.tokens_per_step, which does know the context.")


@dataclass
class RunConfig:
    """A complete run: name + architecture + data + training."""

    name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    out_dir: str = "checkpoints"

    def __post_init__(self) -> None:
        # The vocabulary is dictated by the model; the tokenizer has to produce that size.
        if self.data.tokenizer == "char" and self.model.vocab_size > 512:
            raise ValueError(
                "A character tokenizer with vocab_size > 512 makes no sense. "
                "Check the config."
            )

    # ------------------------------------------------------------------ derived

    @property
    def tokens_per_step(self) -> int:
        return self.train.batch_size * self.train.grad_accum * self.model.context_length

    @property
    def max_steps(self) -> int:
        return self.train.max_tokens // self.tokens_per_step

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    # ------------------------------------------------------------------ (de)serialization

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config {path} does not exist")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = cls.from_dict(raw)
        if cfg.name == "run":
            cfg.name = path.stem
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunConfig:
        """Build from a flat dict, a sectioned one, or a mix of both."""
        sections: dict[str, type] = {
            "model": ModelConfig,
            "data": DataConfig,
            "train": TrainConfig,
        }
        buckets: dict[str, dict[str, Any]] = {k: dict(raw.get(k) or {}) for k in sections}
        top: dict[str, Any] = {}

        field_owner: dict[str, str] = {}
        for section, klass in sections.items():
            for f in fields(klass):
                if f.name in field_owner:
                    raise RuntimeError(
                        f"Ambiguous key {f.name!r}: it appears in several config sections."
                    )
                field_owner[f.name] = section

        own_fields = {"name", "out_dir"}
        unknown: list[str] = []
        for key, value in raw.items():
            if key in sections:
                continue
            if key in own_fields:
                top[key] = value
            elif key in field_owner:
                buckets[field_owner[key]].setdefault(key, value)
            else:
                unknown.append(key)

        if unknown:
            valid = sorted(own_fields | set(field_owner) | set(sections))
            raise ValueError(
                f"Unknown keys in the config: {unknown}.\nValid keys: {valid}"
            )

        return cls(
            **top,
            model=ModelConfig(**buckets["model"]),
            data=DataConfig(**buckets["data"]),
            train=TrainConfig(**buckets["train"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def summary(self) -> str:
        m, t = self.model, self.train
        return "\n".join(
            [
                f"run          : {self.name}",
                f"architecture : {m.n_layers}L x {m.d_model}d x {m.n_heads}h "
                f"(head_dim={m.head_dim}), d_ff={m.d_ff}",
                f"context      : {m.context_length} tokens, vocab {m.vocab_size}",
                f"pieces       : {m.norm}, {m.pos}, {m.activation}, "
                f"tie={'yes' if m.tie_embeddings else 'no'}, dropout={m.dropout}",
                f"batch        : {t.batch_size} x {t.grad_accum} accum x {m.context_length} "
                f"= {self.tokens_per_step:,} tokens/step",
                f"budget       : {t.max_tokens:,} tokens = {self.max_steps:,} steps",
                f"optimizer    : {t.optimizer} lr={t.lr} betas={t.betas} wd={t.weight_decay}",
            ]
        )
