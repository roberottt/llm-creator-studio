# 13 — The real run

## Why this module matters

**Because here you actually train it.**

There is no new concept. What you learn here you only learn by doing it: what to watch while
a training run is going, what is normal, and what it means when something goes wrong.

And there is one concrete technique that justifies the module on its own: **overfitting a
single batch**. Thirty seconds that catch almost any bug in the model or in the loop. It is
the best cost/benefit advice in all of deep learning, and almost nobody applies it.

It is also where you will see, in the samples file, the model learning to write step by step.
That is more informative than any loss curve.

### What you will know by the end

- A 30-second check that catches bugs you would take four hours to find
- The three numbers to look at on step 0 of any training run
- What is normal during a run and what a spike that does not recover means
- What you have to save in a checkpoint to be able to resume **without the model lurching**

### What it costs

1 hour of exercises, plus however long your training takes. `tiny_char` is 70 seconds.

---

## Before launching: the 30-second check

**Overfit a single batch.** You take four sequences, give them to the model over and over,
and check that the loss drops to almost zero.

The idea is that a model with millions of parameters has more than enough capacity to
memorize four sequences. If it cannot, there is a bug.

And you know it in **30 seconds** instead of in four hours.

### What it catches and what it does not

**It catches:** gradients that do not reach some part of the model (one `detach()` too many),
the forgotten `zero_grad()`, an absurd learning rate, a layer disconnected from the graph, the
optimizer built on the wrong parameters.

**It does not catch:** anything to do with generalization. A model that memorizes a batch can
still be completely useless.

**A warning:** if the loss drops to zero *too* fast —in five steps— suspect an information
leak. Check that the targets are shifted one token relative to the input.

It is the best cost/benefit advice in all of deep learning, and even so almost nobody does it.

## The three numbers of step 0

When training starts, look at these three before you go off and do something else:

**The initial loss** has to be $\ln(V)$. With vocabulary 4096, that is 8.317. Higher means
too aggressive an initialization; lower, an information leak. You already saw it in module 05
and it is still the most informative check there is.

**The gradient norm** should be on the order of 0.1 to 10. If it comes out $10^5$, something
is exploding. If it comes out $10^{-8}$, something is vanishing.

**The tokens per second.** Multiply by the planned duration and check that the ETA matches
what you expected. If it is ten times less than estimated, stop and find out why before
leaving it running all night.

## What is normal during the run

**The loss curve drops fast at first and then flattens out.** That is expected: learning that
spaces and vowels exist is easy; learning grammar is not. On a log scale the drop is roughly a
straight line, which is what the scaling laws say.

**The training loss is noisy and the validation one is smooth.** The first is measured on a
single batch; the second, on a hundred. The noise means nothing.

**The gap between the two grows a little.** That is incipient overfitting and it is normal.
With TinyStories and a single pass over the data it should stay small; if it takes off, the
model is memorizing.

**Occasional spikes.** An odd batch produces a loss spike and the model recovers in a few
steps. With `grad_clip` they should be small. If a spike does not recover, training has
broken: stop and resume from the last good checkpoint.

## The text samples: the part that matters

Every N steps, the script generates text and saves it to `samples.md`. That file, read top to
bottom when it finishes, is the model learning to write.

With the character model on Shakespeare the journey is roughly this one:

```
step 0      qkxJ;zW,QQjjxk vvv         pure noise
step 100    the the the and the       frequent words
step 500    I thinks crown me the      sentence structure, some punctuation
step 1500   KING RICHARD III:          names, play formatting
            That's such heaven dull
```

**It is more informative than the loss curve.** A jump from 1.6 to 1.5 does not tell you
much; seeing that the model has started closing its parentheses does.

## Checkpoints: what you have to save

The weights are not enough. A resumable checkpoint needs:

- the model **weights**
- the **optimizer** state (Adam's moments)
- the **GradScaler** state
- the **step number** and the tokens seen

If you resume with the weights only, Adam starts with its moments at zero and the model
lurches right at the resume point. It shows up as a spike in the curve, exactly where you
resumed.

**An implementation detail that matters:** write to a temporary file first and rename at the
end. If the process dies halfway through writing, the previous checkpoint is still intact. A
half-written checkpoint is worse than no checkpoint.

## The TinyStories run on your hardware

```
model     : 8,933,440 parameters
tokens    : 500,000,000
FLOPs     : 6 × 7.62M × 500M ≈ 2.3·10¹⁶
```

With the RTX 2060 at a 51.6 TFLOPS peak and a realistic MFU of 10-15%, that comes to
**between 2 and 5 hours**. It is a napkin estimate; the real number will come from your own
measurement in the first few minutes.

Before launching it, two things: run the overfit on a batch, and launch 100 steps with
`--max-steps 100` to see the real pace and the ETA. If the ETA says 40 hours, something is
wrong and it is better to know beforehand.

## Where the debate is

**When to stop is a decision with less science behind it than it seems.** The standard is to
train until the token budget runs out, but it is not clear that that is optimal: there is
evidence that carrying on past the Chinchilla point keeps improving the model, with
diminishing returns nobody has characterized well.

**Exact reproducibility is harder than it looks.** Even if you fix every seed, cuDNN picks
non-deterministic algorithms for performance, and GPU reductions sum in a non-deterministic
order. Two identical runs diverge. `torch.use_deterministic_algorithms(True)` fixes it at the
cost of speed, and for research experiments it is worth it; for training, almost never.

And one **about this course**: our 500M-token run with a single set of hyperparameters is not
a controlled experiment. If at the end the model generates decent stories, you will not know
how much is due to the architecture, how much to the learning rate and how much to the
dataset. Drawing conclusions from a single run is the most common methodological mistake in
the field, and this course is no exception: it is a learning exercise, not an experiment.

---

**Further reading:** Eldan & Li 2023, [TinyStories](https://arxiv.org/abs/2305.07759) ·
Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
(where the overfit-a-batch advice comes from). Stray terms are in
[GLOSSARY.md](../../GLOSSARY.md).
