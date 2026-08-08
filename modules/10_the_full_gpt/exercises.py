"""Module 10 - The full GPT.

HOW TO DO THIS MODULE
=====================

Read `THEORY.md` -> do exercise 1 ON PAPER before writing code -> implement the rest ->
`llmfs check 10` -> `llmfs hint 10 -e N` -> `SOLUTION.md` has the complete code.

WHAT YOU ARE GOING TO BUILD
===========================

The model you are going to train. Four exercises:

    expected_param_count  (ex. 1)  the formula for how many parameters it will have
    count_parameters      (ex. 2)  count them for real, broken down
    TransformerBlock      (ex. 3)  one block: attention + FFN, with their residuals
    GPT                   (ex. 4)  the whole model

The first two are counting exercises and they have to give the SAME number: 8,933,440. If
they do not match, either your formula or your model is lying. It is a cross-audit, not an
arithmetic exercise.

You can do them in this order: the tests for exercises 1 and 2 use the REFERENCE model, so you do
not need to have written exercise 4 in order to count. `THEORY.md` follows the same order and each
docstring here tells you which section it maps to.

The one thing worth doing differently is exercise 1: DO IT ON PAPER before typing.

The section "The whole model, at a glance" is the one to keep in front of you while writing
exercise 4: it is the complete model with the shape of the tensors at every point.

THE STRUCTURE
=============

    token ids
        |  embedding table
    vectors
        |  block x 6
    vectors
        |  final normalization
    vectors
        |  projection to logits
    scores over the 4096 tokens

VOCABULARY YOU ARE GOING TO NEED
================================

- **weight tying**: reusing the embedding matrix, transposed, as the output layer. It saves
  1.3 million parameters.
- **buffer**: a tensor that travels with the model (it moves with `.to(device)`) but is NOT
  a parameter and receives no gradient. RoPE's tables are buffers.
- **initialization**: the values the weights start at before training. It is not a detail:
  it decides whether the model trains well.
- **logits**: the model's final output, one score per token in the vocabulary.
- **causal**: that a token cannot see the ones that come after it.

    llmfs demo 10     breaks down the parameters and verifies the model is causal
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmfs.config import ModelConfig

# The pieces from modules 06-09. If you have not done them, the bridge uses the reference and
# this works all the same: you can assemble the GPT without having finished earlier modules.
from llmfs.bridge import resolve

MultiHeadAttention = resolve("06_attention", "MultiHeadAttention")
RMSNorm = resolve("07_normalization", "RMSNorm")
SwiGLU = resolve("08_mlp_and_activations", "SwiGLU")
rope_frequencies = resolve("09_position", "rope_frequencies")
causal_mask = resolve("06_attention", "causal_mask")


def expected_param_count(cfg: ModelConfig) -> int:
    """The parameter count, computed from the formula instead of by counting.

    Context in `THEORY.md`: section "Exercises 1 and 2: the exact count", with the term-by-term
    breakdown table and the two typical mistakes identified by the number they give.

    DO IT ON PAPER FIRST
    --------------------
    Seriously. Take THEORY.md's breakdown and write the formula by hand. Only then translate
    it into code. If you go straight to code you will end up trying numbers until they add
    up, and that teaches nothing.

    WHAT YOU HAVE TO WRITE
    ----------------------
    A sum. Accumulate into a `total` variable and return it.

        1. The token embeddings:

               total = cfg.vocab_size * cfg.d_model

        2. If `cfg.pos == "learned"`, also add `cfg.context_length * cfg.d_model`.
           (With RoPE NOTHING gets added here: see below.)

        3. What ONE layer costs, and multiply it by `cfg.n_layers`:

               attention = 4 * cfg.d_model**2                # Wq, Wk, Wv, Wo
               if cfg.bias:
                   attention += 4 * cfg.d_model

               if cfg.activation == "swiglu":
                   ffn = 3 * cfg.d_model * cfg.d_ff          # gate, up, down
               else:
                   ffn = 2 * cfg.d_model * cfg.d_ff          # the classic MLP

               norms = 2 * cfg.d_model                       # two RMSNorms per block
               # if it were LayerNorm WITH bias, it would be 2 * (2 * cfg.d_model)

               total += cfg.n_layers * (attention + ffn + norms)

        4. The final norm: `total += cfg.d_model`.

        5. The output layer:

               if not cfg.tie_embeddings:
                   total += cfg.vocab_size * cfg.d_model

        6. `return total`

    THE CHECK
    ---------
    With the default config (the final model) it has to give EXACTLY 8,933,440:

        1,310,720                    <- 4096 * 320, the embeddings
        + 6 * (409,600               <- 4 * 320²,   one block's attention
             + 860,160               <- 3 * 320 * 896, one block's SwiGLU
             +     640)              <- 2 * 320,   one block's two norms
        +       320                  <- the final norm
        = 8,933,440

    If you get 10,244,160 you forgot the weight tying (that is the exact difference:
    1,310,720). If you come out slightly over, you are probably counting biases that do not
    exist.

    WHAT DOES NOT COUNT, AND WHY IT IS WORTH UNDERSTANDING
    ------------------------------------------------------
    RoPE contributes NOT ONE parameter. Its cos/sin tables come from a closed formula and are
    stored as buffers, not as parameters: nobody trains them. If your count includes anything
    from RoPE, it is wrong.

    WHAT THIS IS REALLY FOR
    -----------------------
    For designing. You change `d_model` in the YAML and see instantly whether the model fits
    on your GPU, without waiting to build it. And for verifying: if this formula and exercise
    2 do not give the same number, either your formula or your model is lying, and you have
    to find out which.

    Args:
        cfg: the model configuration.

    Returns:
        The total parameter count, as an integer.
    """
    raise NotImplementedError("TODO: module 10, exercise 1 - expected_param_count")


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Counts the parameters for real, broken down by component.

    Context in `THEORY.md`: section "Exercises 1 and 2: the exact count", with the breakdown you
    have to get (embeddings 14.7%, attention 27.5%, FFN 57.8%) and why weight tying forces the
    `set` of ids.

    WHAT YOU HAVE TO WRITE
    ----------------------
    A walk over `named_parameters()` classifying by name.

        1. The output dict, at zero:

               out = {"embeddings": 0, "attention": 0, "ffn": 0,
                      "norms": 0, "lm_head": 0, "other": 0}

        2. The walk, skipping tensors already seen:

               seen = set()
               for name, param in model.named_parameters():
                   if id(param) in seen:
                       continue
                   seen.add(id(param))
                   n = param.numel()

        3. Inside the loop, classify by looking at SUBSTRINGS of the name. The names look
           like `blocks.3.attn.q_proj.weight`, so `in` is enough:

               "token_embedding" or "pos_embedding"         -> out["embeddings"] += n
               "attn."                                      -> out["attention"]  += n
               "gate_proj"/"up_proj"/"down_proj"/"fc_"      -> out["ffn"]        += n
               "norm"                                       -> out["norms"]      += n
               "lm_head"                                    -> out["lm_head"]    += n
               anything that fits nothing                   -> out["other"]      += n

           Watch the ORDER of the `if/elif`: `attn.` has to be checked before `norm`, because
           `blocks.0.attn_norm.weight` contains both and you want it counted as a norm. Order
           from more specific to more general and check it.

        4. The two totals:

               out["total"] = sum(v for k, v in out.items())    # before adding these two
               out["non_embedding"] = out["total"] - out["embeddings"]
               return out

    BEFORE WRITING ANYTHING, PRINT THE NAMES
    ----------------------------------------
        print([n for n, _ in model.named_parameters()])

    It is well worth it: in thirty seconds you see how the whole model is wired up and you
    know exactly which substrings to look for. Do not guess.

    WEIGHT TYING AND THE `set` OF ids
    ---------------------------------
    With `tie_embeddings=True`, `lm_head.weight` and `token_embedding.weight` are THE SAME
    tensor, not two copies. It appears under two different names.

    A fact that contradicts the common belief: both `parameters()` and `named_parameters()`
    DEDUPLICATE by identity by default (`remove_duplicate=True`), so the total comes out
    right even if you do nothing.

    Even so, keep the `set` of `id(param)`. Two reasons: it makes explicit that you know
    there are shared weights, and it protects the breakdown if one day you walk with
    `remove_duplicate=False`. It is one line and it avoids a 1,310,720-parameter error.

    And watch out for a Python trap: `if param in seen` does NOT work. The `in` operator uses
    `==`, which on tensors is elementwise and blows up with "Boolean value of Tensor is
    ambiguous". That is why we compare by `id()`.

    WHAT `non_embedding` IS AND WHY IT IS RETURNED SEPARATELY
    ---------------------------------------------------------
    It is `total - embeddings`. That is the number module 12's scaling laws use, because
    embeddings scale differently from the rest of the model: they grow with the vocabulary,
    not with depth, and they do not take part in the per-token computation the way the layers
    do.

    Args:
        model: the already built model.

    Returns:
        A dict with the keys `embeddings`, `attention`, `ffn`, `norms`, `lm_head`, `other`,
        `total` and `non_embedding`.
    """
    raise NotImplementedError("TODO: module 10, exercise 2 - count_parameters")


class TransformerBlock(nn.Module):
    """One block: attention and FFN, each with its normalization and its residual.

    Context in `THEORY.md`: section "Exercise 3: the block", with what those two lines are really
    saying and the four places where people fail, all four of them silent.

    WHAT YOU HAVE TO WRITE
    ----------------------
    **In `__init__`** (four lines besides the `super()`). The names matter: the test copies
    weights by name, and exercise 2 classifies by name.

        from llmfs.reference import make_norm, make_ffn

        self.attn_norm = make_norm(cfg)
        self.attn = MultiHeadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, bias=cfg.bias
        )
        self.ffn_norm = make_norm(cfg)
        self.ffn = make_ffn(cfg)

    `make_norm` and `make_ffn` are already written and they look at `cfg.norm` and
    `cfg.activation` to decide whether to build RMSNorm or LayerNorm, SwiGLU or the classic
    MLP. Do not reimplement them.

    **In `forward`** (two lines and a return):

        x = x + self.attn(self.attn_norm(x), cos=cos, sin=sin, mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

    And that is it. The whole block is two lines.

    WHAT THOSE TWO LINES ARE SAYING
    -------------------------------
    Each one is "pre-norm + residual" (module 07): normalize, compute something, and ADD the
    result to what was already there. The `x` to the left of the `+` is the residual stream,
    the highway that runs through the model top to bottom with nothing interrupting it.

    Attention MOVES information between tokens. The FFN PROCESSES it token by token. They
    alternate, and that alternation is the whole Transformer.

    The two residuals are independent on purpose: each sub-block can contribute a little or a
    lot without constraining the other.

    FOUR PLACES WHERE PEOPLE GO WRONG
    ---------------------------------
    **Normalizing the stream instead of the branch.** It is `x + attn(norm(x))`, not
    `norm(x + attn(x))`. The second is post-norm and it breaks the property that makes the
    gradient reach layer 1 clean. It trains, but worse and with more warmup.

    **Reusing the same norm for both sub-blocks.** They are TWO different objects with their
    own weights. `self.ffn_norm = self.attn_norm` compiles and gives plausible results, and
    it is wrong.

    **Forgetting to pass `cos`, `sin` and `mask` to attention.** Without `cos`/`sin` the model
    loses all positional information and still trains (badly). Without `mask`, every token
    sees the future and the loss drops suspiciously well. Both failures are silent.

    **Passing `cos`/`sin`/`mask` to the FFN.** It does not accept them, and it does not need
    them: the FFN does not look at other tokens, so there is nothing to mask and no position
    to inject.

    forward(self, x, cos=None, sin=None, mask=None):
        Args:
            x: `(B, T, d_model)`.
            cos, sin: RoPE tables, or None.
            mask: the causal mask, or None to let attention build it.
        Returns:
            `(B, T, d_model)`, exactly the same shape that went in.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError("TODO: module 10, exercise 3 - TransformerBlock.__init__")

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("TODO: module 10, exercise 3 - TransformerBlock.forward")


class GPT(nn.Module):
    """The complete model. 8,933,440 parameters when you finish.

    Context in `THEORY.md`: section "Exercise 4: the whole model", which goes through the four
    parts of __init__ in the same order they are written, explaining each design decision where it
    has to be written: weight tying in part 2, what a buffer is and why RoPE lives in one in part
    3, and the depth-scaled initialization in part 4. And before that section, "The whole model, at
    a glance": the route from (B, T) to (B, T, 4096) with the shapes, which is what to keep in
    front of you while writing this.

    THE STRUCTURE
    -------------
        ids -> embeddings -> [block] x n_layers -> final norm -> logits

    With RoPE there is no positional embedding to add at the start: position is injected
    inside attention. That is why the first layer is only the token table.

    WHAT YOU HAVE TO WRITE IN `__init__`
    ------------------------------------
        1. Store the config and create the submodules. The names matter:

               self.cfg = cfg
               self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
               if cfg.pos == "learned":
                   self.pos_embedding = nn.Embedding(cfg.context_length, cfg.d_model)
               self.drop = nn.Dropout(cfg.dropout)
               self.blocks = nn.ModuleList(
                   [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
               )
               self.norm_f = make_norm(cfg)
               self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        2. WEIGHT TYING:

               if cfg.tie_embeddings:
                   self.lm_head.weight = self.token_embedding.weight

        3. THE RoPE TABLES, as a buffer:

               if cfg.pos == "rope":
                   cos, sin = rope_frequencies(
                       cfg.head_dim, cfg.context_length, cfg.rope_theta
                   )
                   self.register_buffer("rope_cos", cos, persistent=False)
                   self.register_buffer("rope_sin", sin, persistent=False)

        4. THE INITIALIZATION, in two passes. First everything:

               self.apply(self._init_weights)

           with `_init_weights` doing `nn.init.normal_(m.weight, mean=0.0, std=0.02)` on the
           `nn.Linear` and `nn.Embedding`, and `nn.init.zeros_` on any biases.

           And then, OVERRIDING the above only on the projections that write into the
           residual stream:

               scale = 0.02 / math.sqrt(2 * cfg.n_layers)
               for name, param in self.named_parameters():
                   if name.endswith(("out_proj.weight", "down_proj.weight", "fc_out.weight")):
                       nn.init.normal_(param, mean=0.0, std=scale)

        The order of steps 2 and 4 matters: the tying goes AFTER creating `lm_head` and
        BEFORE the initialization.

    WHAT YOU HAVE TO WRITE IN `forward(idx, targets=None)`
    ------------------------------------------------------
        1. Validate the size, with both numbers in the message:

               B, T = idx.shape
               if T > self.cfg.context_length:
                   raise ValueError(f"sequence of {T} > context {self.cfg.context_length}")

        2. x = self.token_embedding(idx)
        3. if `cfg.pos == "learned"`, add
           `self.pos_embedding(torch.arange(T, device=idx.device))`
        4. x = self.drop(x)
        5. cos, sin = (self.rope_cos, self.rope_sin) if pos=="rope", otherwise (None, None)
        6. mask = causal_mask(T, device=idx.device)      <- ONCE, not inside the loop
        7. for block in self.blocks: x = block(x, cos=cos, sin=sin, mask=mask)
        8. x = self.norm_f(x)
        9. logits = self.lm_head(x)
       10. the loss, if there are targets:

               loss = None
               if targets is not None:
                   loss = F.cross_entropy(
                       logits.reshape(-1, logits.size(-1)),
                       targets.reshape(-1),
                       ignore_index=-100,
                   )
               return logits, loss

    WHY EACH OF THE ODD THINGS
    --------------------------
    **`self.lm_head.weight = self.token_embedding.weight`** does not COPY anything: it makes
    both layers point at the same tensor in memory. It saves 1,310,720 parameters (15% of the
    model) and it also improves quality, because each weight receives gradient along two
    paths.

    **`register_buffer`** stores a tensor that travels with the model — it moves with
    `.to(device)`, it shows up in the `state_dict` — but is NOT a parameter: it receives no
    gradient and the optimizer never sees it. `persistent=False` also keeps it out of the
    checkpoint, because it is recomputed when the model is built and storing it would waste
    space.

    **The scaled initialization.** Each block ADDS its contribution to the residual stream,
    so with 6 layers and 2 sub-blocks each the output's variance would be 12 times the
    input's. Reducing the standard deviation of the projections that write there compensates
    for it. The 2 in the denominator is because each block writes twice.

    And the 0.02 is not arbitrary either: it is what makes the step-0 loss equal ln(V). With
    `std=1` (`nn.Embedding`'s default) the model would start opinionated and random, and the
    loss would come out ABOVE ln(V). You saw it in module 05's demo.

    **`mask` outside the loop.** It is the same for all 6 layers. Computing it inside works
    and wastes work on every training step.

    **`ignore_index=-100`** does nothing right now, but you will need it in module 16 to mask
    the prompt during fine-tuning. Leave it in.

    HOW TO KNOW IF IT IS RIGHT
    --------------------------
        - `count_parameters(GPT(ModelConfig()))["total"] == 8_933_440`
        - the loss of the first forward, untrained, is around `ln(4096) = 8.32`
        - changing the token at position 6 does not move A SINGLE logit at positions 0-5
          (the demo checks it: it comes out exactly 0.00e+00)

    If the initial loss comes out WAY below 8.32, do not celebrate: it almost always means
    the model is seeing the future. Look at the mask, and then look at how the batch is
    assembled.

    forward(self, idx, targets=None):
        Args:
            idx: `(B, T)` int64, the token ids.
            targets: `(B, T)` int64 shifted by one position, or None.
        Returns:
            `(logits, loss)` with logits `(B, T, vocab_size)`. `loss` is None without
            targets.

        Raises:
            ValueError: if `T` exceeds `cfg.context_length`.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError("TODO: module 10, exercise 4 - GPT.__init__")

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError("TODO: module 10, exercise 4 - GPT.forward")
