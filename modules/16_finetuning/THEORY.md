# 16 — Post-training: teaching it to answer instead of to continue

## Why this module matters

**Because your trained model does not answer questions: it continues them.**

Write *"What is the capital of France?"* and it will most likely carry on with more questions. It
is not broken: it is doing exactly what you taught it, which is to continue plausible text. A
document that starts with a question usually goes on with more.

Turning that into something that answers is a separate phase with its own methods, and it is where
a good part of what you associate with an assistant gets installed. In this module you do it: real
SFT on your model, and you see the before and after.

And you learn LoRA, the technique that makes fine-tuning large models accessible: training **0.7%**
of the parameters instead of 100%.

### What you will know by the end

- The real difference between pretraining and post-training, and what each adds
- Why a model needs markers to know when it is its turn to speak and when to shut up
- An off-by-one that decides whether the model learns to answer or to ask
- How to train 0.7% of a model and then **merge the changes** without a trace
- What to actually expect from SFT on a toy model, so you do not misread it

### What you are going to write

Four exercises, in two independent blocks. This theory follows them in order:

| Exercise | What it does | |
|---|---|---|
| 1. `build_chat_template` | Serializing the conversation with markers | SFT |
| 2. `mask_prompt_tokens` | Making sure only the answer is learned from | SFT |
| 3. `LoRALinear` | The layer with low-rank adapters | LoRA |
| 4. `merge_lora_weights` | Merging the adapters into the base matrix | LoRA |

The first two go together and are the SFT: one gives format and the other decides what gets learned
from. The last two go together and are LoRA, and they are independent of the previous ones — you
can do SFT without LoRA and LoRA without SFT; they get combined because that is what is done in
practice.

Exercise 2 is four lines and **the range of the loop is the whole exercise**.

### What it costs

3 hours. The demo does real SFT on your module 13 model, so you will see the before and after with
your own weights.

---

## Pretraining against post-training

| | pretraining | post-training |
|---|---|---|
| **goal** | learn language | learn to behave |
| **data** | all the text you can get | curated examples, few |
| **amount** | billions of tokens | thousands or tens of thousands |
| **cost** | months and millions | hours |
| **what changes** | the knowledge | the format of the answer |

The important part, and it is an idea that takes some accepting: **post-training does not add
knowledge**. What it does is bring to the surface a behaviour that was already latent. A model that
does not know something after pretraining is not going to learn it from a thousand conversation
examples.

It is literally the same loss as always — cross-entropy over the next token, module 05's — and the
same loop as module 11. All that changes is the data and one mask.

---

## Exercise 1: the chat template (`build_chat_template`)

A pretrained model has no idea where a question ends and an answer begins. You teach it with
**markers**:

```
   training:  <|user|>Who is the king?<|end|><|assistant|>The king is Richard.<|end|>
   inference: <|user|>Who is the king?<|end|><|assistant|>
                                                         ↑ left open: the model continues here
```

Notice the difference between the two lines, because it is the reason the function has an
`add_generation_prompt` flag: when **training** you give it the complete conversation, answer
included; when **generating** you leave it open right after the assistant marker, and the model
continues from there. Same text, different ending.

The markers are not magical: they are text the model learns to recognize during SFT. It learns that
after `<|assistant|>` it is time to answer, and that `<|end|>` means stop — **without that, a model
does not know when to shut up** and generates until it exhausts the context.

Each model family uses its own and they are mutually incompatible. Using the wrong template with a
model degrades its quality noticeably, and it is a surprisingly frequent mistake.

---

## Exercise 2: masking the prompt (`mask_prompt_tokens`)

Here is the subtle part of the module. You do not want the model to learn to **generate the user's
questions**: you want it to learn to **answer them**.

The fix is putting `-100` in the targets of the prompt positions.
`F.cross_entropy(..., ignore_index=-100)` skips them — and that is the `ignore_index` you left in
place back in module 10 without using it. This is where it pays off.

With `input_ids = [10, 11, 12, 20, 21, 22]` and `prompt_len = 3`:

| position | input | target | |
|---|---|---|---|
| 0 | 10 | −100 | prompt: ignored |
| 1 | 11 | −100 | prompt: ignored |
| 2 | 12 | **20** | **the transition: it DOES learn** |
| 3 | 20 | 21 | answer |
| 4 | 21 | 22 | answer |
| 5 | 22 | −100 | there is no next token |

**Notice position 2: there are two ignored positions at the start, not three.** It is the last
token of the prompt, but since the targets are shifted by one token (module 04), its target is
already the first token of the answer.

And that transition — *"the question is over, my turn to speak"* — is the most important thing the
model has to learn in the whole SFT. Masking it would remove exactly the signal it needs.

That off-by-one is the exercise's typical mistake and it **raises no visible error**: it simply
wastes the most informative position, and the model learns worse with nothing indicating why.

---

## Exercise 3: LoRA (`LoRALinear`)

Doing full SFT on a 70B model needs memory for the weights, the gradients and Adam's states: on the
order of 12 bytes per parameter, nearly a terabyte. It is the same breakdown you did in module 10,
scaled up.

**LoRA** (Hu et al., 2021) starts from an observation: the changes fine-tuning makes are **low
rank**. You do not need to be able to modify the matrix in any direction; a few will do.

So `W` gets frozen and the product of two skinny matrices is added to it:

$$W' = W + \frac{\alpha}{r} BA$$

with $A$ of $r \times d_{in}$ and $B$ of $d_{out} \times r$, and $r$ small (4, 8, 16).

### The arithmetic that justifies it

With $d_{in} = d_{out} = 320$ and $r = 8$:

```
   full W:    320 × 320       = 102,400 parameters
   A and B:   8×320 + 320×8   =   5,120 parameters    (5%)
```

Applied to our 9M model, adapting only `q_proj` and `v_proj`:

| r | trainable | % of the model |
|---|---|---|
| 4 | 30,720 | **0.34%** |
| 8 | 61,440 | **0.68%** |
| 16 | 122,880 | **1.36%** |

And since the optimizer state only exists for what is trainable — the `requires_grad=False` that
`build_param_groups` skips in module 11 — Adam's memory drops in the same proportion. In large
models it is the difference between needing eight GPUs and needing one.

### Freezing the base is the exercise

The step to understand in `__init__` is this one:

```python
self.base = base_layer
for p in self.base.parameters():
    p.requires_grad = False
```

Without those two lines you would have a layer with adapters **and** the base training too: you
neither save memory nor achieve anything. All of LoRA is in freezing and adapting from outside.

### The initialization, which is not symmetric

```
   A ~ normal (Kaiming)
   B = ZEROS
```

With $B = 0$, the product $BA$ is zero at the start and **the layer is exactly the original one**.
Fine-tuning begins without perturbing anything, and the demo checks it: at initialization, the LoRA
layer's output is identical to the base's.

If you initialized both at random, the model would start degraded and would have to spend the first
steps recovering before it began improving. It is one of those decisions that look like a detail
and are not — the same idea as RMSNorm's `torch.ones` in module 07: on startup, the new piece must
do nothing.

---

## Exercise 4: merging the weights (`merge_lora_weights`)

When you finish, the adapters get **absorbed** into the base matrix:

$$W_{\text{new}} = W + \frac{\alpha}{r} BA$$

The resulting model is indistinguishable from a normal one: same inference cost, same shapes, and
it can be served with no LoRA dependency at all. Measured in the demo, the merged layer gives the
same as the adapted one to within `1.31e-06`, which is fp32 rounding.

That is LoRA's advantage over other parameter-efficient fine-tuning methods: the adaptation is
**exactly** a sum of matrices, so it can be absorbed without approximating anything. Other methods
add layers or change the topology, and then there is no way back to a standard model.

It is also what lets you keep **several adapters for the same base model** and load one or another
depending on the task, without duplicating the big weights.

---

## What you will actually see when doing SFT

The demo does SFT on the Shakespeare model with 96 examples in `Q: ... / A: ...` format (it uses
that format instead of the `<|user|>` markers because the character tokenizer only knows the
symbols that appear in Shakespeare). The loss drops from 1.4568 to 0.0912, and this is the before
and after:

```
   BEFORE:   Q: Who is the king?
             A:
             I have the comptaint the headen shall do logger, To hear it...

   AFTER:    Q: Who is the king?
             A:
             I say we must go.

             MARCIUS:
             A lord.
```

**What to look at is not whether the answer is correct.** With 0.8M parameters and 96 examples, it
is not going to be.

What to look at is the **format**: before, it kept writing Shakespeare indefinitely; after, it
answers something short and stops. That is exactly what post-training teaches, and it is the
module's lesson: **it does not add knowledge, it brings a behaviour to the surface**.

If you expected an assistant, the distance is not about training: it is three or four orders of
magnitude in parameters and data, plus the step that comes next.

## What we are NOT going to do, and it matters

After SFT, commercial models go through **RLHF** or **DPO**: human preferences between pairs of
answers get collected and the model is adjusted towards the preferred ones.

That is what makes a model *useful* rather than merely *format-obedient*, and it is also where a
good part of the behaviour you associate with an assistant gets installed.

We will not do it here. It requires preference data we do not have, and a 9M model does not have
the capacity to exploit it. But it is worth knowing that step exists, because it explains a good
part of the distance between what you have just built and what you use every day.

## Where the debate is

**Why LoRA works is not clear.** The "intrinsic low rank" hypothesis is reasonable and has
evidence, but it is not proven. There is work showing LoRA does worse than full fine-tuning on
tasks requiring new knowledge, and comparably on tasks that only change style — which fits the
hypothesis, but is correlation.

**How much capability post-training actually adds** is an active discussion. The *superficial
alignment* hypothesis holds that almost all the knowledge is in the pretraining and post-training
only selects the format. There is evidence for it — very decent results are achieved with a
thousand examples — and against.

And an honest one about this module: **with a 9M model trained on TinyStories, SFT is not going to
produce an assistant**. You will see it learn the format — it answers after the marker, it stops at
the end — and little else. The exercise teaches the mechanism, it does not produce a product.

---

**Further reading:** Hu et al. 2021, [LoRA](https://arxiv.org/abs/2106.09685) · Ouyang et al. 2022,
[InstructGPT](https://arxiv.org/abs/2203.02155) (RLHF) · Zhou et al. 2023,
[LIMA](https://arxiv.org/abs/2305.11206) (the superficial alignment hypothesis).
Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
