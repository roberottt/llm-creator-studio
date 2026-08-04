# 16 — Post-training: SFT and LoRA

## Why this module matters

**Because your trained model does not answer questions: it continues them.**

Write *"What is the capital of France?"* and the most likely thing is that it carries on with
more questions. It is not broken: it is doing exactly what you taught it, which is continuing
plausible text. A document that starts with a question usually carries on with more.

Turning that into something that answers is a separate phase with its own methods, and it is
where a good part of what you associate with an assistant gets installed. In this module you
do it: real SFT on your model, and you see the before and the after.

And you learn LoRA, the technique that makes fine-tuning large models accessible: training
**0.7%** of the parameters instead of 100%.

### What you will know by the end

- The real difference between pretraining and post-training, and what each one adds
- Why a model needs markers to know when it is its turn to speak and when to shut up
- An off-by-one that decides whether the model learns to answer or to ask
- How to train 0.7% of a model and then **merge the changes** without a trace

### What it costs

3 hours. The demo does real SFT on your model from module 13.

---

## Pretraining against post-training

| | pretraining | post-training |
|---|---|---|
| **goal** | learning language | learning to behave |
| **data** | all the text you can get | curated examples, few |
| **amount** | billions of tokens | thousands or tens of thousands |
| **cost** | months and millions | hours |
| **what changes** | the knowledge | the format of the answer |

The important thing, and it is an idea that is hard to accept: **post-training does not add
knowledge**. What it does is bring to the surface a behaviour that was already latent. A model
that does not know something after pretraining is not going to learn it from a thousand
conversation examples.

## SFT: teaching the format

*Supervised Fine-Tuning* is carrying on training with the same loss as always, but on pairs of
instruction and answer.

Two pieces make it work.

### The chat template

A pretrained model has no idea where a question ends and an answer begins. You teach it with
**markers**:

```
<|user|>What is the capital of France?<|end|><|assistant|>Paris.<|end|>
```

There is nothing magical about the markers: they are text the model learns to recognize during
SFT. It learns that after `<|assistant|>` it is time to answer, and that `<|end|>` means stop —
**without that, a model does not know when to shut up**.

Every model family uses its own and they are incompatible with each other. Using the wrong
template with a model degrades its quality quite a lot, and it is a surprisingly frequent
mistake.

### Masking the prompt

Here is the subtle part. You do not want the model to learn to **generate the user's
questions**: you want it to learn to **answer them**.

The solution is to put `-100` in the targets of the prompt positions.
`F.cross_entropy(..., ignore_index=-100)` skips them.

```
input_ids = [10, 11, 12, 20, 21, 22]      with prompt_len = 3
targets   = [-100, -100, 20, 21, 22, -100]
```

**Note there are two ignored positions, not three.** The targets are shifted one token, so at
position 2 —the last token of the prompt— the target is already the first token of the answer,
and that one does matter.

That off-by-one is the typical mistake, and it raises no visible error: it just wastes (or
over-uses) one position.

## LoRA: training 1% of the model

Doing full SFT on a 70B model requires memory for the weights, the gradients and Adam's
states: on the order of 12 bytes per parameter, almost a terabyte.

**LoRA** (Hu et al., 2021) solves this with one observation: the changes fine-tuning makes are
**low rank**. You do not need to be able to modify the matrix in any direction; a few are
enough.

So `W` is frozen and the product of two skinny matrices is added to it:

$$W' = W + \frac{\alpha}{r} BA$$

with $A$ of $r \times d_{in}$ and $B$ of $d_{out} \times r$, and $r$ small (4, 8, 16).

### The arithmetic that justifies it

With $d_{in} = d_{out} = 320$ and $r = 8$:

```
the whole W:  320 × 320       = 102,400 parameters
A and B:      8×320 + 320×8   =   5,120 parameters    (5%)
```

Applied to our 9M model, adapting only `q_proj` and `v_proj`:

| r | trainable | % of the model |
|---|---|---|
| 4 | 30,720 | **0.34%** |
| 8 | 61,440 | **0.68%** |
| 16 | 122,880 | **1.36%** |

And since the optimizer state only exists for what is trainable, Adam's memory drops in the
same proportion. On large models it is the difference between needing eight GPUs or one.

### The initialization, which is not symmetric

```
A ~ normal (Kaiming)
B = ZEROS
```

With $B = 0$, the product $BA$ is zero at the start and **the layer is exactly the original
one**. Fine-tuning starts without perturbing anything.

If you initialized both at random, the model would start degraded and would have to recover
before starting to improve. It is one of those decisions that look like a detail and are not.

### Merging the weights

When you finish, the adapters are **absorbed** into the base matrix:

$$W_{\text{new}} = W + \frac{\alpha}{r} BA$$

The resulting model is indistinguishable from a normal one: same inference cost, same shapes,
and it can be served with no LoRA dependency at all.

That is one of LoRA's advantages over other efficient fine-tuning methods: the adaptation is
**exactly** a sum of matrices, so it can be absorbed without approximating anything.

## What we are NOT going to do, and it matters

After SFT, commercial models go through **RLHF** or **DPO**: human preferences between pairs of
answers are collected and the model is adjusted towards the preferred ones.

That is what makes a model *useful* instead of merely *obedient to the format*, and it is also
where a good part of the behaviour you associate with an assistant gets installed.

We will not do it here. It requires preference data we do not have, and a 9M model does not
have the capacity to take advantage of it. It is worth knowing that step exists.

## Where the debate is

**Why LoRA works is not clear.** The "intrinsic low rank" hypothesis is reasonable and has
evidence, but it is not proven. There is work showing that LoRA performs worse than full
fine-tuning on tasks that require learning new knowledge, and comparably on those that only
change the style — which fits the hypothesis, but is correlation.

**How much capability post-training really adds** is an active discussion. The *superficial
alignment* hypothesis holds that almost all the knowledge is in the pretraining and
post-training only selects the format. There is evidence in favour —very decent results are
achieved with a thousand examples— and also against.

And an honest one about this module: **with a 9M model trained on TinyStories, SFT is not going
to produce an assistant**. You are going to see it learn the format —it answers after the
marker, it stops at the end— and little else. The exercise teaches the mechanism, it does not
produce a product.

---

**Further reading:** Hu et al. 2021, [LoRA](https://arxiv.org/abs/2106.09685) · Ouyang et al.
2022, [InstructGPT](https://arxiv.org/abs/2203.02155) (RLHF) · Zhou et al. 2023,
[LIMA](https://arxiv.org/abs/2305.11206) (the superficial alignment hypothesis).
Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
