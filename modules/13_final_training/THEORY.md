# 13 — The real run: launching it, watching it and knowing whether it is going well

## Why this module matters

**Because this is where you actually train it.**

There is no new concept. What you learn here you only learn by doing it: what to watch while a
training run is going, what is normal, and what it means when something goes wrong. It is the
least theoretical module in the course and the one that changes what you can do the most.

And there is one specific technique that earns the module on its own: **overfitting a single
batch**. Thirty seconds that catch almost any bug in the model or the loop. It is the best
cost/benefit advice in all of deep learning, and almost nobody applies it.

It is also where, in the samples file, you get to watch the model learn to write step by step.
That is more informative than any loss curve.

### What you will know by the end

- A 30-second check that catches bugs you would take four hours to find
- **How to read the log line**, field by field — each one comes from a different module and
  together they are the whole course
- The three numbers to look at at step 0 of any training run
- What is normal during a run and what a spike that does not recover means
- What has to go in a checkpoint so you can resume **without the model lurching**

### What you are going to write

Two functions, and they are the smallest in the course:

| Exercise | What it does |
|---|---|
| 1. `overfit_single_batch` | The 30-second check that catches almost everything |
| 2. `format_eta` | How much is left, in something readable |

Exercise 1 is the simplest training loop there is, four steps, and it is the one that matters.
Exercise 2 is a chain of `if`s formatting seconds — it looks cosmetic and below I explain why it
is not.

**And then comes what this module actually does, which is not an exercise: training.** The
training code is already written, assembled on top of the pieces from modules 04 to 12. Your job
here is to launch it, know how to read it and decide whether it is going well.

### What it costs

1 hour of exercises, plus however long your training takes. `tiny_char` is about 70 seconds and
runs on any machine, CPU included.

---

## Exercise 1: the 30-second check (`overfit_single_batch`)

You take four sequences, hand them to the model **over and over**, and check that the loss drops
almost to zero.

The idea is that a model with millions of parameters has more than enough capacity to memorize
four sequences. There is nothing to generalize: only to memorize. If it cannot manage it, there
is a bug. And you know in 30 seconds instead of in four hours.

What you write is the barest possible training loop: no scheduler, no gradient accumulation, no
AMP. **On purpose**: the fewer pieces there are, the fewer places a bug can hide. This loop is
the reference pattern to compare the real loop against when something fails.

### What has to come out

Measured with the toy model (`tiny_char`, vocabulary 65):

| step | loss | |
|---|---|---|
| 0 | 4.2902 | ← should be around `ln(65) = 4.174` |
| 10 | 3.1004 | |
| 50 | 2.7322 | |
| 100 | 1.2210 | |
| 200 | 0.1659 | |
| 299 | 0.0893 | ← should be almost at zero |

Two seconds on the clock. If `history[-1]` is not well below `history[0]`, **stop and find the
bug**; do not launch the long run.

### What it catches and what it does not

**It catches:** gradients not reaching some part of the model (one `detach()` too many), the
forgotten `zero_grad()` — gradients **accumulate** by default in PyTorch —, an absurd learning
rate in either direction, a layer disconnected from the graph, the optimizer built over the wrong
parameters.

**It does not catch:** anything to do with generalization. A model that memorizes a batch
perfectly can still be completely useless on new data. This checks that the **machinery** works,
not that the model is any good.

### If it drops too fast, be suspicious too

If the loss plants itself at zero in five steps, do not celebrate: look for an information leak.
Check that `y` is shifted by **one** token relative to `x`. If you passed `model(x, x)` the model
would only have to copy the input and the loss would collapse. The symptom is identical to a
broken causal mask, and you have already run into it twice in this course — modules 05 and 10:
when the loss is suspiciously good, check the mask and then check who builds the batch.

---

## Exercise 2: how much is left (`format_eta`)

A chain of `if`s turning seconds into `"1h 2m"`. Four ranges and one odd case.

**And it is not cosmetic, for two reasons.**

The first is that you are going to look at that number many times over a run that lasts hours.
`"1h 2m"` reads instantly; `"3725s"` has to be divided in your head every time.

The second is deeper: from one hour onwards **the seconds stop being shown**. When two hours are
left, seconds are noise: they change constantly, they carry no information and they make the
number dance on screen. The useful precision of an estimate is always proportional to its
magnitude, and that is a rule that serves you well beyond this exercise.

**And the `"?"` instead of a 0.** Returning `"?"` is the honest thing when there is not yet enough
data to estimate: in the first steps the average speed means nothing. It also avoids printing
things like `"-1s"` or `"infd 0h"`, which besides being ugly make you doubt whether training is
going well. `math.isfinite(x)` is `False` for `inf`, `-inf` and `nan`, and all three come from
dividing by zero when computing the rate on the first step.

Check 3725 by hand: `3725 // 3600 = 1` and `(3725 % 3600) // 60 = 125 // 60 = 2`, so `"1h 2m"`.
The five leftover seconds are lost, which is exactly what you want.

---

## And now the important part: launching the run

With both exercises green, this is what you do, and in this order:

```bash
llmfs check 13                              # both exercises green
llmfs train --config tiny_char              # the toy: ~70 s, any machine
llmfs train --config tiny_char --max-steps 100   # rate probe before the big one
llmfs train --config tinystories_9m         # the real one
llmfs train --config tinystories_9m --resume     # if it gets interrupted
```

**The toy first, always.** `tiny_char` is a character-level GPT of 861,440 parameters over
Shakespeare, and it exists precisely for this: validating the whole pipeline — data, model, loop,
sampling, checkpoints — in under a minute and on any machine. If something is broken it shows up
there, and not four hours after launching the real run.

And before the big one, that **100-step probe** with `--max-steps`. It is not for training
anything: it is to see the real rate and the ETA. If the ETA says 40 hours when you expected 4,
something is wrong and it is much better to know before going to bed.

---

## How to read the log line

This is the skill you actually take away from this module. When you launch the training, every N
steps a line like this appears (real, from a `tiny_char` run):

```
step  100/600   loss 2.2693   lr 3.00e-03   |g| 0.70   112.3k tok/s   MFU 4.8%   18s
```

Six fields, and **each one comes from a different module of the course**:

| field | what it is | where it comes from | what to watch |
|---|---|---|---|
| `loss` | cross-entropy of the last batch | module 05 | that it goes down; noisy is normal |
| `lr` | this step's learning rate | module 11, ex. 2 | rising during warmup, falling after |
| `\|g\|` | global gradient norm **before** clipping | module 11, ex. 3 | between 0.1 and 10 |
| `tok/s` | measured throughput | module 12 | that it is stable |
| `MFU` | fraction of the hardware peak | module 12, ex. 2 | that it does not suddenly collapse |
| the time | the ETA | module 13, ex. 2 | that it matches what you expected |

That is why `|g|` is logged **before** clipping and not after: if you logged the post-clipping
norm you would see `1.00` pinned forever and learn nothing. Logged before, a sustained rise warns
you that training is destabilizing **before** it blows up. It is exactly what module 11 argued,
and this is where it pays off.

And the MFU is on that line for the same reason: not for its absolute value — the 4.8% in that
run is low because the model is tiny — but because **if it collapses halfway through the run,
something has changed**: another process competing for the GPU, thermal throttling, a dataloader
that has run out of cache.

---

## The three numbers of step 0

Before the first log line, the initial loss appears. Look at these three before going off to do
something else:

**The initial loss** has to be $\ln(V)$. In the run above it came out at `4.2348` against
`ln(65) = 4.1744`, a drift of +0.060 — normal, for the reason you saw in module 10:
initialization with `std=0.02` gives nearly identical logits, not identical ones. Higher means
too aggressive an initialization; lower means an information leak.

**The gradient norm** should be on the order of 0.1 to 10. In the real run it started at 1.04. If
it comes out at $10^5$, something is exploding; if at $10^{-8}$, something is vanishing.

**The tokens per second.** Multiply by the expected duration and check the ETA matches what you
were expecting.

All three are free and all three save you hours.

---

## What is normal during the run

**The curve drops fast at the start and then flattens.** That is expected: learning that spaces
and vowels exist is easy; learning grammar is not. On a log scale the fall is roughly a straight
line, which is what module 12's scaling laws say.

**The training loss is noisy and the validation loss is smooth.** The first is measured over a
single batch; the second, over a hundred. The noise means nothing — in the demo run, the training
loss goes *up* from 1.6050 to 1.6362 between steps 500 and 600 while the validation loss goes
down from 1.7903 to 1.7131. It is not that the model got worse: it is that step 600's batch was
harder than step 500's.

**The gap between the two grows a little.** That is incipient overfitting and it is normal. With
TinyStories and a single pass over the data it should stay small; if it takes off, the model is
memorizing.

**Occasional spikes.** An odd batch produces a spike and the model recovers in a few steps. With
`grad_clip` they should be small — you measured exactly how much in module 11: without clipping, a
poisoned batch raised the loss 3×; with clipping, it did not even notice. **If a spike does not
recover, training has broken**: stop, and resume from the last good checkpoint.

---

## The text samples: the part that matters

Every N steps the script generates text and appends it to `samples.md`. That file, read top to
bottom when it finishes, **is the model learning to write**. This is what came out of a 600-step
run of the toy:

```
step 0     kUU$sbpKKMMbbbPcxfffffTjjfNLL --TJ??333OOqIwTGG33m'T.B--tuq
           D'sSSOOMBiPtB'''''wEvgRRR.vUUUHgJ;OXD3xxExqVOX$J-DUUHIiit&!

step 300   MAPCHASTING Yrace not be town, bunders.  CAMILLY: Mare striset
           mist and be doth bare Enay?  First Larry a thee slay, to I pine

step 600   Which begane of schame a loved, this show as friar, But there
           appos bementes that that will down, And my tell are whity it here
```

Look at what it has learned by step 300 without anyone telling it: that words are separated by
spaces, that sentences carry punctuation, that there are capitalised names followed by colons
because Shakespeare is published in play format. Almost every word is wrong, but **the shape** is
right.

**It is more informative than the loss curve.** A jump from 1.6 to 1.5 does not tell you much;
seeing that the model has started closing its parentheses does.

---

## Checkpoints: what gets saved and why

This is already written in `llmfs/train/checkpoint.py`, but it is worth opening it and seeing what
it puts inside, because the list is not obvious. A resumable checkpoint needs four things:

- the model's **weights**
- the **optimizer's** state, that is, module 11's Adam moments
- the **GradScaler's** state
- the **step number** and the tokens seen

If you resume with only the weights, Adam starts with its moments at zero and the model **lurches**
right at the resume point. You can see it very clearly as a spike in the curve, exactly where you
resumed. It is the same problem the bias correction solves at the start of a run, except now it
happens halfway through and with nobody expecting it.

**And an implementation detail that matters:** it writes to a temporary file first and renames at
the end. If the process dies mid-write, the previous checkpoint is still intact. **A half-written
checkpoint is worse than no checkpoint**, because it looks fine.

Training saves two: `last.pt` to resume from and `best.pt` with the best validation loss seen.
They are not the same file and it is worth knowing which one you want.

---

## The TinyStories run on your hardware

```
   model     : 8,933,440 parameters
   tokens    : 500,000,000
   FLOPs     : 6 × 8.93M × 500M ≈ 2.7·10¹⁶
```

(The 8.93M there is `params_matmul`, module 12's count: everything except the normalization
scales. It is not the 7.62M non-embedding one, which is what goes into Chinchilla and not into the
FLOPs. It is exactly the distinction that module warns about, and this is where it gets applied.)

With the RTX 2060 at a peak of 51.6 TFLOPS and a realistic MFU of 10-15%, that works out at
**between 2 and 5 hours**.

And here it is worth being explicit: **that is a back-of-the-envelope estimate, not a
measurement.** This run has not been executed while writing the course, because the development
machine has no CUDA. Everything else in this file is measured; this is not. The real number will
come from your own 100-step probe in the first few minutes, and if it does not match this
estimate, trust yours.

---

## Where the debate is

**When to stop is a decision with less science behind it than it seems.** The standard is to train
until the token budget is exhausted, but it is not clear that is optimal: there is evidence that
continuing past the Chinchilla point keeps improving the model, with diminishing returns nobody
has characterized well. It is the other side of what you saw in module 12.

**Exact reproducibility is harder than it looks.** Even if you fix every seed, cuDNN picks
non-deterministic algorithms for speed and GPU reductions sum in non-deterministic order. Two
identical runs diverge. `torch.use_deterministic_algorithms(True)` fixes it at the cost of speed;
for research experiments it is worth it, for training almost never.

And one **about this course**: our 500M-token run with a single set of hyperparameters **is not a
controlled experiment**. If the model generates decent stories at the end, you will not know how
much is due to the architecture, how much to the learning rate and how much to the dataset.
Drawing conclusions from a single run is the most common methodological error in the field, and
this course is no exception: it is a learning exercise, not an experiment.

---

**Further reading:** Eldan & Li 2023, [TinyStories](https://arxiv.org/abs/2305.07759) ·
Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
(where the overfit-a-batch advice comes from). Stray terms are in
[GLOSSARY.md](../../GLOSSARY.md).
