# 04 — Data: from text to batches on the GPU

## Why this module matters

**Because this is where it gets decided what the model learns.**

So far you have text turned into a list of numbers (module 03). What you do not have is a
*task*: nobody has yet told the model what it is supposed to do with those numbers. That
jump —from "a strip of integers" to "a question with its answer"— happens in this module,
and it happens in a single line of code.

The line is so short that it is easy to skim past without seeing what is inside it. What is
inside it is the reason a language model can be trained on text nobody has labelled: the
text **already carries the answer**, because the answer to "what comes next?" is, literally,
what comes next. It is the idea that made it possible to train on the entire internet
instead of on a hand-annotated corpus.

The rest of the module is plumbing: how to store 500 million tokens without spending 4 GB,
how to read them without loading them, and which chunk you set aside so you can tell whether
the model is learning or cheating. Plumbing that decides real things.

### What you will know by the end

- What **self-supervised learning** is and why it is what made LLMs take off
- **Why a single 512-token window is 512 training examples**, not one
- Which part of a real LLM you are building, and what a real lab does in this same phase
  that we skip here
- Why 500 million tokens take 1 GB and not 4
- A silent NumPy bug that corrupts your data without giving any error at all
- Why the validation set is NOT picked at random, with the number that proves it
- Whether the disk really is your bottleneck or it is a legend repeated without measuring

### What you are going to write

Three functions. This theory is ordered so that you read them in this order, and each one
has its own section with the matching numeric example:

| Exercise | What it does | Where it is explained |
|---|---|---|
| 1. `pack_tokens_uint16` | List of ids → a 2-bytes-per-token array, with validation | [§ Packing the corpus](#exercise-1-packing-the-corpus-pack_tokens_uint16) |
| 2. `train_val_split` | Set the end of the corpus aside for validation | [§ Setting validation aside](#exercise-2-setting-validation-aside-train_val_split) |
| 3. `get_batch` | Draw a batch of random windows and move it to the GPU | [§ Drawing a batch](#exercise-3-drawing-a-batch-get_batch) |

The first two are short and almost everything in them is error checking. The third one *is*
the module: your training run will execute it tens of thousands of times and it is where the
idea lives.

### What it costs

2 hours. Not much code, but exercise 3 has an off-by-one that, if you get it wrong, does not
raise an error where you made it but three lines further down and with a message that tells
you nothing.

But do not jump straight to them. The two sections coming up now —which part of an LLM this
is, and the underlying idea— are what stop the three exercises from looking like loose
plumbing.

---

## Which part of the LLM this is

Before getting into the code it is worth placing yourself, because this is one of those
modules that gets misunderstood when read in isolation: it builds no piece of the model, and
still it decides what the model ends up knowing.

Building an LLM is five distinct jobs, and the course walks them in this order:

```
   0. FOUNDATIONS    what an LLM is, PyTorch, autograd        modules 00-02   ✔ done
   1. TOKENIZER      text  ->  numbers                        module 03       ✔ done
   2. DATA           numbers  ->  a learning task             module 04       ← YOU ARE HERE
   3. MODEL          the architecture that does the predicting modules 05-10
   4. TRAINING       adjust the weights until it gets it right modules 11-13
   ────────────────────────────────────────────────────────────────────────────
      and afterwards: generating text (14), evaluating (15), instruction tuning (16)
```

Piece 2 is today's, and its full name in the literature is **the pretraining data pipeline**.
It is the part papers dispatch in one paragraph and that in a real lab keeps a whole team
busy for months.

### Pretraining: the phase you are preparing data for

A commercial LLM is built in two very different phases, and it is worth keeping them apart
in your head from now on:

- **Pretraining.** You give it brutal amounts of text and a single task: predict the next
  token. There is no conversation, no instructions, nobody correcting it. This is where the
  model learns grammar, facts, style and the regularities of language, and where the vast
  majority of the compute goes. **The data in this module is the data for this phase**, and
  it is the only phase this course does up to module 13.
- **Post-training** (SFT, RLHF, module 16). A comparatively tiny polish on instruction-and-
  response examples, which turns "a model that continues text" into "a model that obeys".
  That data is different, it is expensive and it has humans behind it.

When somebody says GPT-4 "was trained on the internet", they are talking about the first
phase. And the figures of the first phase all look like this:

| model | parameters (the size of the model) | pretraining tokens (the text it is shown) |
|---|---|---|
| GPT-3 (2020) | 175,000 M | 300,000 M |
| Chinchilla (2022) | 70,000 M | 1,400,000 M |
| Llama 3 405B (2024) | 405,000 M | 15,000,000 M |
| **yours** | **8.93 M** | **500 M** |

Be careful not to mix the two columns, because they are easy to confuse and they mean very
different things. **The parameters are the model**: the numbers the network adjusts and that
you end up saving to a weights file — the 8,933,440 of yours, which you assemble in module
10. **The tokens are the text you push past it during training**, and they are not stored in
the model: they get read, learned from, and discarded. A model with 8.93 M parameters trained
on 500 M tokens sees **fifty-six times its own size in text**, and still takes up what it
takes up: about 36 MB in fp32. (In module 12 you will see that same ratio quoted as 65 tokens
per parameter; there the embeddings are excluded from the count, as is customary in scaling
laws. Same arithmetic, different denominator.)

With that clear, look at the table again: your model is forty-five thousand times smaller
than Llama 3 and sees thirty thousand times less text. But the kind of object you are
preparing and the task you are defining are exactly the same, and the `get_batch` you are
about to write differs from theirs in engineering details, not in the idea.

That 500 M tokens is the right figure *for a model with 8.93 M parameters* is not arbitrary
either: that ratio between the two columns is precisely what scaling laws study, and module
12 derives them and works ours out.

### The underlying idea: text supervises itself

Here is the concept to take away from this module, and it is a machine learning concept, not
a plumbing one.

**The problem, in plain terms.** Classical machine learning needs *labelled* data: a
thousand photos with a human having written "cat" or "dog" under each one. That label is
expensive. It is the historical bottleneck of the field: there is no money to hand-label the
millions of examples needed for anything big.

Text does not come labelled. Nobody has annotated *Don Quixote* saying what is correct and
what is not. So where does the answer to compare the model's prediction against come from?

**The trick: the answer is already written, one token further along.** If the task is
"predict what comes next", the original text *is* the solution. The label does not have to
be manufactured: it has to be covered up and then uncovered.

```
    sentence:   the   cat   sleeps   on   the   sofa

    example 1   input: "the"                      answer: "cat"
    example 2   input: "the cat"                  answer: "sleeps"
    example 3   input: "the cat sleeps"           answer: "on"
    example 4   input: "the cat sleeps on"        answer: "the"
    example 5   input: "the cat sleeps on the"    answer: "sofa"
```

Six words, five training examples with their correct answers, and zero humans involved. This
is called **self-supervised learning**: the supervision exists, but it comes out of the data
itself instead of an annotator. It is what allowed the field to stop training on annotated
corpora of a few thousand examples and start training on the whole web, and it is —more than
any architectural innovation— the reason LLMs took off.

**The formula.** What that scheme formally defines is the training objective. If the corpus
is a sequence of tokens $x_1, x_2, \dots, x_n$, the model has to maximize the probability it
assigns to each token given everything before it:

$$\max_\theta \sum_{t=1}^{n} \log P_\theta(x_t \mid x_1, \dots, x_{t-1})$$

Read it with the example in front of you: each summand is a row of the table above. $\theta$
is the model's weights, and $P_\theta(x_t \mid \cdots)$ is the probability the model gives to
the word that actually came next. Maximizing that is exactly minimizing the **cross-entropy**,
which is the loss you implement in module 05 and the number you will stare at for hours in
module 13.

And that is this module's role: **`get_batch` is what materializes that sum.** Nobody writes
the summation by hand; it appears because you hand the model an `x` and a `y` shifted by one
token, and it computes the 512 predictions in one go.

### What this module deliberately does not do

So that the perimeter is clear, because a real data pipeline has more pieces than there are
here:

- **There is no model and no gradients.** Nothing from `torch.nn` is imported. All of this
  happens before the model exists.
- **The corpus is not cleaned.** A real pipeline spends most of its effort filtering junk,
  deduplicating (repeated documents make the model memorize), removing toxic content and
  deciding in what proportion the sources get mixed. We skip all of that because TinyStories
  comes clean out of the box, and that is the main reason a nine-million-parameter model can
  write anything coherent at all.
- **Tokenizing does not happen here.** That was module 03. This module starts where that one
  finished: with the list of ids already made.

## Where this module fits, concretely

With the general map clear, this is the exact route the data takes through the code you are
going to write:

```
   module 03      raw text             "Once upon a time…"
                       │
                       │  your tokenizer
                       ▼
                  list of ids          [271, 4, 88, 1902, …]
   ───────────────────┼──────────────────────────────────────────
                      │  pack_tokens_uint16   (ex. 1)
                      ▼
   module 04     .bin file             1 GB of uint16 on disk
                      │
                      │  train_val_split      (ex. 2)
                      ▼
              train  /  val            two views of the same array
                      │
                      │  get_batch            (ex. 3)   ← on every step
                      ▼
   ───────────────────┼──────────────────────────────────────────
                   (x, y)              two int64 tensors of 48×512
                      │
   module 11          ▼                the training loop
```

To the left of this module is your tokenizer. To the right is the training loop. What sits in
between —what you write today— runs at two very different moments, and that difference
explains why the three functions have opposite priorities:

- **Exercises 1 and 2 run exactly once**, when preparing the corpus. They are allowed to be
  slow. What they are not allowed to do is fail silently, because the mistake gets burned
  into the file and everything downstream trains on corrupt data without knowing it. That is
  why they are almost all validation.
- **Exercise 3 runs on every training step**, tens of thousands of times. There, expensive
  validation is out of the question.

## The three things you work with

Before typing anything, be clear about the three data structures that show up in every
signature in `exercises.py`. There are only three and there is not a single class involved.

**1. The list of ids: `list[int]`.** What module 03 gives you. Plain ordinary Python numbers,
each between 0 and 4095:

```python
[271, 4, 88, 1902, 33, 4, 271]
```

**2. The token array: an `np.ndarray` of `uint16`.** The same thing, but as a NumPy array and
taking 2 bytes per number instead of the 8 Python spends. This is what gets written to disk
and what training will read. The conversion is exercise 1.

```python
array([ 271,    4,   88, 1902,   33,    4,  271], dtype=uint16)
```

When that array lives in a file instead of in RAM it is called an `np.memmap`, but it is used
exactly the same way: `data[100:200]` just works. There is a section at the end on why.

**3. The `(x, y)` pair: two PyTorch tensors of shape `(batch_size, context_length)`.** What
gets handed to the model. `x` is what it sees and `y` is what it has to predict, and the only
difference between them is that `y` is shifted by one token. Producing them is exercise 3.

```python
x.shape   # torch.Size([48, 512])   dtype=torch.int64
y.shape   # torch.Size([48, 512])   dtype=torch.int64
```

Notice that the type changes twice along the way: Python `int` → `uint16` to store it →
`int64` to hand it to the model. It is not a whim, and both conversions have their reason.
Both are explained further down, in the exercise that does them.

---

## Exercise 1: packing the corpus (`pack_tokens_uint16`)

A list of ids goes in, a NumPy array of type `uint16` comes out. Four lines, and three of
them are checks.

### Why `uint16` and not the default type

A token in our model is a number between 0 and 4095. The question is how many bytes you give
each one, and the answer is decided by the whole corpus:

| type | range it holds | 500M tokens take |
|---|---|---|
| `int64` (what Python uses) | ±9 · 10¹⁸ | **4.0 GB** |
| `uint32` | 0 to 4,294,967,295 | 2.0 GB |
| `uint16` | **0 to 65,535** | **1.0 GB** |

`uint16` goes up to 65,535, sixteen times more than we need, and takes a quarter of the space
of the `int64` Python would use on its own. Three gigabytes of difference for writing
`dtype=np.uint16` in the right place.

(`u` is for *unsigned*: it does not spend a bit representing negatives, which do not exist in
token ids. And `16` is the bits: 2 bytes, 2¹⁶ = 65,536 distinct values.)

You could squeeze harder: with a vocabulary of 4,096, 12 bits per token would do. It is not
done, because NumPy's types go byte by byte and packing by hand would cost more CPU on every
read than it saves on disk. `uint16` is the sensible point.

### The trap: NumPy does not warn you when a number does not fit

This is what justifies the exercise existing. If you convert to `uint16` a number that does
not fit, NumPy **raises no exception and prints no warning**. It wraps the counter around and
carries on as if nothing happened. Run, not estimated:

```python
np.array([65535], dtype=np.int64).astype(np.uint16)   # -> 65535   fine
np.array([65536], dtype=np.int64).astype(np.uint16)   # -> 0
np.array([65537], dtype=np.int64).astype(np.uint16)   # -> 1
np.array([66536], dtype=np.int64).astype(np.uint16)   # -> 1000
np.array([   -1], dtype=np.int64).astype(np.uint16)   # -> 65535
```

Look at `66536 -> 1000`. That is the genuinely ugly case, not the 0 one. An id that went out
of range has become **1,000, which is a perfectly valid id** in our vocabulary of 4,096:
there is nothing in the file to give away that this token is wrong. It gets written, gets
trained on, and the only thing you notice is that the model learns slightly worse than it
should. No trace, no exception, nothing to look at. It is the kind of bug people spend days
hunting in the wrong place.

And `-1 -> 65535` is the same story from the other end: a negative id (which can only come
from a bug of yours in the tokenizer) becomes the largest possible number.

### Hence the order: convert to `int64`, validate, and only then pack

The exercise's sequence is not arbitrary:

```python
array = np.asarray(ids, dtype=np.int64)      # 1. into a type where EVERYTHING fits
if array.size and (...):                     # 2. check the range
    raise ValueError(...)
return array.astype(np.uint16)               # 3. and now, pack
```

If you validated after converting, you would be checking data **that has already been
corrupted**: 65,536 would already be a 0, and a 0 passes any range check with flying colours.
The check has to happen while the numbers are still themselves.

### The two small details

**The `array.size and ...` in step 2.** On an empty array, `.min()` does not return anything
sensible: it raises `ValueError: zero-size array to reduction operation minimum`. That is a
real error, but it has nothing to do with what you are validating and it throws off whoever
reads it. The `and` short-circuit avoids going there. An empty corpus is rare but legitimate
(a file that turned out to have nothing in it), and it should not blow up.

**The actual values in the error message.** "ids out of range" is no use to anyone.
`min=0, max=9999` tells you instantly that your tokenizer is emitting ids it should not, and
of what magnitude, which is exactly what you need in order to know where to look. There is a
test that checks the number appears in the message.

### What happens to the array afterwards

Outside this exercise, in the real pipeline, the array gets dumped to disk as it is:

```python
tokens.tofile("train.bin")
```

And that writes exactly the array's bytes, no header, no metadata, nothing. A 1 GB file that
is literally the strip of numbers. That is why, when reading it back, you have to tell NumPy
what type it was (`dtype=np.uint16`): the file does not know, there is nowhere to keep it. Get
the `dtype` wrong on the way in and you get neatly ordered garbage — one more silent failure
from this same family.

---

## Exercise 2: setting validation aside (`train_val_split`)

The whole corpus goes in, two chunks come out. Four lines.

### What the validation set is for

If you measure how well the model does on the very text you trained it on, the answer means
nothing: a model with enough memory can memorize the text and get a perfect score without
having learned anything useful for anything else. That is **overfitting**.

The way to detect it is to set a chunk of text aside, never train on it, and measure there
every so often. What you will see during training is this:

```
   loss
    │
    │ \
    │  \                       validation: goes down, bottoms out and starts going UP
    │   \        ___..--''
    │    '-.__.-'
    │     \__
    │        '--..___          training: down and down and down
    └────────────┬──────────────  steps
                 │
        it started memorizing here
```

While the two go down together, the model is learning general things. When training keeps
going down and validation flattens or rises, it is memorizing. Without a validation set **you
have no way of seeing that moment**, and it is information you will need in module 11.

### Why the cut is contiguous and from the end

The reflex of anyone who has done tabular machine learning is to shuffle and split:
`train_test_split(shuffle=True)`. Here that is **wrong**, and not a little: it is completely
broken. The reason is that the training windows overlap.

With `context_length=512`, the window starting at position 100 and the one starting at 101
share 511 of their 512 tokens. Every token in the corpus appears in 512 different windows. So
if you split **windows** at random, your validation windows are nearly identical to windows
that are in training.

How nearly. This is a simulation that was run: corpus of 100,000 tokens, context 512, random
window split 99.5% / 0.5%:

| way of splitting | validation tokens the model already saw in training |
|---|---|
| windows shuffled at random | **100.00 %** |
| contiguous cut from the end | **0.0 %** |

It is not "a bit of leakage": it is that the validation set is an exact subset of what it
already saw. 100% of the validation windows were entirely covered by training windows.

And the worst part is how it shows up, because it does not show up as a failure. It shows up
as a success:

- validation loss goes down glued to training loss,
- it never rises, it never flattens,
- you never see overfitting, however long you train.

A beautiful chart that says precisely nothing. You would be measuring memorization and
calling it generalization. It is the kind of mistake that gets discovered late, when the
model meets the world and turns out not to have been that good.

**The fix is to cut a contiguous block from the end** and not touch it. Since TinyStories is
made of stories independent from one another, that last 0.5% is whole stories the model has
never seen. There is not even leakage at the boundary of the cut, because the two halves
become separate arrays: no training window can cross to the other side.

Behind this there is a general principle worth far more than this module: **the validation
set has to be independent from the training one in the unit that matters**. Here the unit is
not the token: it is the story. In time-series data it would be the day; in medical data, the
patient. Shuffling rows is correct only when the rows really are independent, and here they
are nowhere near it.

### The two details in the code

**The `max(1, ...)`.** With a corpus of 50 tokens and `val_fraction=0.005`, `int(50 * 0.005)`
is `int(0.25)`, that is 0: you would end up with no validation set and no warning whatsoever.
The `max(1, ...)` guarantees at least one token. It is a case that only shows up in the tests
and in quick experiments, but it shows up.

**It returns views, not copies.** This:

```python
return tokens[:-n_val], tokens[-n_val:]
```

does not copy a single byte. NumPy *slicing* returns a **view**: a new array pointing at the
same memory as the original. With 500M tokens, doing a `.copy()` out of habit would be 1 GB
of RAM thrown away for nothing. And if the original is a `memmap`, the view is still a
`memmap` and still loads nothing. There is a test that verifies it with `np.shares_memory`.

What this function does **not** do, and it is worth being clear about: it does not shuffle
anything, does not reorder, does not copy, does not touch the data. It cuts and returns two
views. That is all.

---

## Exercise 3: drawing a batch (`get_batch`)

Here is the module. The corpus goes in, two tensors ready for the model come out.

### The idea, now in numbers

This is the translation into code of the self-supervised learning we saw at the start: the
"input → answer" pairs from the cat sentence, but with ids and in two matrices.

You have the corpus as one extremely long strip of numbers. You pick a position at random and
take a window. **The input is the window. The target is the same window shifted by one
token.**

```
corpus =  [ 5,  8,  2,  9,  1,  7, ... ]

     x =  [ 5,  8,  2,  9]                 the window as it is
     y =      [ 8,  2,  9,  1]             the same one, shifted by one position
```

And now read it column by column, which is where the interesting part is:

| what the model sees | what it has to predict |
|---|---|
| `[5]` | `8` |
| `[5, 8]` | `2` |
| `[5, 8, 2]` | `9` |
| `[5, 8, 2, 9]` | `1` |

**A 4-token window is not one training example: it is four.** With context 512 it is 512
predictions per window, and a batch of 48×512 is **24,576 predictions in a single pass**. All
of them are computed at once, in the same *forward*, and all of them contribute to the loss.

Each of those 24,576 predictions is a summand of the objective formula we saw above. Nobody
writes the summation: it comes from here, from having passed a `y` shifted by one token.

There is a condition for this to work, and it is not in this module: the model has to be
incapable of looking ahead. If while predicting position 2 it could see token 3, it would
copy it and learn absolutely nothing — zero loss and a useless model. What prevents that is
the **causal mask** in module 06. Keep in mind that it exists, because the construction of
`y` you are writing today depends on it.

### What the function does, with real numbers

This is a real run: corpus `[0, 1, 2, …, 999]` (so that the values tell you where they came
from), `batch_size=4`, `context_length=8`, seed 0.

```python
data = np.arange(1000, dtype=np.uint16)
x, y = get_batch(data, batch_size=4, context_length=8, rng=np.random.default_rng(0))
```

First it works out how far a window can start:

```
max_start = len(data) - context_length - 1 = 1000 - 8 - 1 = 991
```

Then it draws 4 starting positions at random. With seed 0 these are exactly the ones that
come out:

```
starts = [842, 631, 506, 267]
```

And from each one it takes two windows, the input one and the shifted one:

```
x = [[842, 843, 844, 845, 846, 847, 848, 849],
     [631, 632, 633, 634, 635, 636, 637, 638],
     [506, 507, 508, 509, 510, 511, 512, 513],
     [267, 268, 269, 270, 271, 272, 273, 274]]

y = [[843, 844, 845, 846, 847, 848, 849, 850],
     [632, 633, 634, 635, 636, 637, 638, 639],
     [507, 508, 509, 510, 511, 512, 513, 514],
     [268, 269, 270, 271, 272, 273, 274, 275]]
```

That is everything the function does. Each row is a contiguous window of the corpus, the four
of them start at unrelated places, and `y` is `x` shifted by one position. If your
implementation produces this, you are done.

Notice as well a property the tests use and that is useful to you for debugging: `x[:, 1:]`
and `y[:, :-1]` contain exactly the same thing. If that does not hold, your `y` is not the
shift of your `x` and the model would be learning to predict something else entirely.

### The `-1` is the mistake of this exercise

`max_start = len(data) - context_length - 1`. That `-1` is where most people get it wrong, so
it is worth going through slowly.

`x` needs tokens from `i` up to `i + context_length - 1`. But `y` needs **one more**: it
reaches `i + context_length`. If you computed `max_start` thinking only about `x`, the last
possible window would leave `y` asking for a token that does not exist.

And here comes the unpleasant bit: **NumPy raises no error when slicing out of range**. It
simply gives you back fewer elements than you asked for.

```python
data = np.arange(10)
data[8:14]        # -> array([8, 9])    two elements, no error
```

So your short window slips through without complaining and the failure surfaces three lines
below, in the `np.stack`, with a message about incompatible shapes that mentions neither the
index nor the end of the corpus nor anything that leads you to the cause. If you get that
error, now you know where to look.

### The `.astype(np.int64)` does two things, and both matter

```python
x_np = np.stack([data[i : i + context_length] for i in starts]).astype(np.int64)
```

**The obvious one:** `nn.Embedding` indexes its table with the tokens, and PyTorch requires
the indices to be `int64`. Your data is in `uint16`. Without the conversion, the model raises
a type error the moment you try.

**The less obvious one:** `astype` **copies**. And that copy is exactly what you want when
`data` is a `memmap`, because `torch.from_numpy` copies nothing: it wraps whatever memory you
give it. If you handed it the `memmap`'s window directly, you would have a tensor pointing at
memory mapped from a file, and every time the model read it that would potentially be a disk
read in the middle of the forward. The `astype` materializes the data in normal RAM and cuts
that dependency.

### The `device` and the `pin_memory`, CUDA only

The last step moves the tensors to wherever the model is. On CUDA there is a trick that does
not apply on CPU or MPS:

```python
if device.type == "cuda":
    x = x.pin_memory().to(device, non_blocking=True)
```

**Pinned memory** (or *page-locked*) is memory the operating system commits to not moving
around or sending to swap. That lets the GPU read it by DMA —direct access, without the CPU
acting as a middleman copying bytes— and it is what makes the second ingredient possible:
with `non_blocking=True` the `.to(device)` call **returns immediately**, without waiting for
the copy to finish. The next batch's transfer overlaps with the current computation instead
of adding to it.

On MPS it makes no sense because the memory is unified: the CPU and the GPU look at the same
RAM and there is no copy to overlap. On CPU, obviously, neither. Hence the `if`.

### Why at random and with repeats

Every call picks positions at random from the whole corpus, without keeping track of the ones
that already came up. That means some windows will come up several times and others never:
**it is not an epoch** in the classical sense of "one complete pass over the data".

In exchange, the function has no state. There is no index to maintain, no permutation to
store, no decision about what happens when you reach the end. Resuming a training run from a
checkpoint is trivial because there is nothing to resume. It is what nanoGPT does and it works
well. The other side of it is in the debate section.

A note on scale, to show why worrying about the repeats is unnecessary: a corpus of 500M
tokens with context 512 has **499,999,487 distinct starting positions**. A training run of
50,000 steps with batch 48 consumes 2.4 million windows. You are sampling 0.5% of the possible
space: collisions are anecdotal.

---

## `memmap`: what it is and why it is used here

An `np.memmap` is a NumPy array whose data lives in a file instead of in RAM. The important
part is that **it is used exactly like a normal array**: `data[100:200]` just works, `len(data)`
works, slicing works. Underneath, the operating system loads into memory only the pages you
touch and discards them when it needs the room.

```python
mm = np.memmap("train.bin", dtype=np.uint16, mode="r")
mm[1000:1512]        # the OS reads only that page from disk
```

Now the honest part, because this often gets explained badly. Our 1 GB file **would fit
perfectly well in your 16 GB of RAM**. The reason for using `memmap` is not that it does not
fit:

1. **Instant startup.** Loading 1 GB from disk into RAM takes however long it takes, and you
   pay it every time you launch the script. With `memmap` it is immediate: nothing is read
   until it is touched.
2. **The operating system's cache does the work for you.** Since you access random positions
   over and over, the OS ends up keeping in RAM what you use most. Free, and better managed
   than you would do it.
3. **It scales without changing anything.** If tomorrow you train with a 50 GB corpus, the
   same code keeps working just the same. Without `memmap` it would have to be rewritten.

If your corpus is small, loading it whole with `np.fromfile` is just as valid and simpler.
There is no magic here. And watch out when measuring it: timing `memmap` against `fromfile`
on a file that is already in the system's page cache does not measure the disk, it measures
the cache. This module's `demo.py` does that measurement and explicitly warns you to read
nothing into it.

---

## Is the disk your bottleneck? Measure it

The version repeated everywhere is that the GPU sits idle waiting for data. It is sometimes
true. Let us see whether it is true **here**, because measuring it costs thirty seconds and
repeating it without measuring costs wrong decisions.

Measured on the MacBook Pro M5 (MPS), a 50M-token corpus in a 100 MB file opened with
`memmap`, using the reference `get_batch`:

| batch × context | tokens/batch | ms per batch | tokens/s |
|---|---|---|---|
| 8 × 64 | 512 | 0.56 | 0.9 M |
| 16 × 128 | 2,048 | 0.36 | 5.7 M |
| 32 × 256 | 8,192 | 0.40 | 20.4 M |
| **48 × 512** | **24,576** | **0.45** | **54.6 M** |

And now the other side of the scales: one complete training step of the 8.9M-parameter GPT
you build in module 10, with that same 48×512 batch, on that same machine:

```
get_batch                                 0.47 ms      (same run as the line below;
full step (forward + backward + opt)      1,342 ms      hence 0.47 and not 0.45)
                                          ─────────
the data is 0.04 % of the step
```

**The disk is not your bottleneck, not by a long way.** You could make `get_batch` two
thousand times slower and it would still go unnoticed. This is the thing to take away: at
this scale, the data plumbing is already fast enough and optimizing it further is time
wasted.

When would that stop being true? When the corpus does not fit in the system cache and every
batch has to actually go to disk; when the model is so small that a step takes microseconds;
or —the realistic case, and the reason it is the first thing this module does— **when you
tokenize on the fly instead of once**. That last number is measured too: our `bpe_encode`
from module 03, in pure Python, processes about **14.5 kB of text per second**.

```
1 MB of text            ≈  70 seconds
1.7 GB (500M tokens)    ≈  33 hours
```

Thirty-three hours every time you start a training run. *That* is a bottleneck, and it is
exactly the one exercise 1 removes: you tokenize once, store the ids, and from then on you
read integers from a file.

---

## Where the debate is

**Random sampling with replacement is not the only reasonable choice.** Since it keeps no
track of the windows used, it gives no coverage guarantee: there may be chunks of the corpus
the model never sees. An ordered traversal shuffled by blocks does give one, at the cost of
keeping state and complicating resumption. With 500M tokens and a single pass the difference
is negligible; with many epochs over a small corpus it would matter considerably more, and
there most people use a `DataLoader` with a permutation.

**Fixed-size windows cut documents in half.** Our window starts at a random position, so it
almost always begins halfway through one story and ends halfway through another. The model
spends training looking at beheaded fragments. The alternatives —padding out to the end of the
document, or concatenating documents separated by an `<|endoftext|>` token and teaching the
model to reset there— have their own costs: the first wastes compute on empty tokens, the
second lets attention cross from one document to the next unless you complicate the mask. Most
large models concatenate and accept the crossing. Here, with short stories, cutting is enough.

**And the most debated one: what should be *inside* the corpus.** The TinyStories paper argues
that a small, very clean dataset with the vocabulary of a four-year-old lets tiny models
generate coherent text — something you do not get by training the same model on an equally
sized chunk of the internet. That the quality and *distribution* of the data matter as much as
or more than the quantity is today one of the most active lines in the field, and also one of
the worst documented: the big labs publish architectures and keep the data to themselves. When
you read that a new model is better, keep in mind that a good part of the difference may live
here, in the least glamorous module of them all.

---

**Further reading:** Eldan & Li 2023, [TinyStories](https://arxiv.org/abs/2305.07759) ·
[nanoGPT](https://github.com/karpathy/nanoGPT) (its `get_batch` is practically this module's) ·
[Data movement is all you need](https://arxiv.org/abs/2007.00072), on how much of training
time goes into moving data rather than computing. Stray terms are in
[GLOSSARY.md](../../GLOSSARY.md).
