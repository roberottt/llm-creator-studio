# 04 — Data: from text to batches on the GPU

## Why this module matters

**Because the GPU cannot be left waiting.**

It is the least glamorous module in the course and one of the ones that decides the most
performance. If preparing the next batch of data takes longer than processing it, your GPU
spends half its time idle and your training run takes twice as long. With a small model like
ours, that risk is real.

And there is something more important than speed: this is where **what the model learns**
gets defined. The way you pair up inputs and targets is what turns a pile of text into a
learning task. It is a three-line idea and it is what makes language models so
data-efficient.

### What you will know by the end

- Why 500 million tokens take 1 GB and not 4
- A silent NumPy bug that would corrupt your data without giving any error
- **Why a single 512-token window is 512 training examples**, not one
- Why the validation set is NOT picked at random, and what happens if you do

### What it costs

2 hours. Three short functions, but your training run will execute the batch one tens of
thousands of times.

---

## The problem: the GPU cannot be left waiting

During training, the GPU processes a batch of data and asks for the next one. If preparing
it takes longer than processing it, the GPU sits idle. With a small model like ours — which
processes a batch in hundredths of a second — this is a real risk: it is easy for the
bottleneck to be reading the disk, not computing.

And there is a second problem, sillier but more expensive: **tokenizing 2 GB of text with
your BPE in pure Python takes on the order of an hour**. Doing it every time you start a
training run is unacceptable. It has to be done **once** and the result stored.

## The solution: an array of integers on disk

The plan is: you tokenize the whole corpus once, store the ids in a flat binary file, and
from then on you always read from there.

### Choosing the type: `uint16`

A token in our model is a number between 0 and 4095. How many bytes do you give it?

| type | range | 500M tokens take |
|---|---|---|
| `int64` (Python's) | ±9·10¹⁸ | **4 GB** |
| `uint32` | 0 to 4·10⁹ | 2 GB |
| `uint16` | **0 to 65,535** | **1 GB** |

`uint16` goes up to 65,535, plenty for our 4,096. And it takes a quarter of the space of the
`int64` Python would use by default.

**Watch out for a very ugly NumPy trap:** if an id goes out of range, it does not warn. It
*wraps around* silently. 65,536 becomes 0, 65,537 becomes 1. There is no exception, no
warning: your data is simply corrupted and the model learns worse with nothing pointing at
the cause. That is why exercise 1 forces you to validate before converting. Ten lines of
checking now against days of debugging later.

### Storing it with `memmap`

An `np.memmap` is a NumPy array that lives on disk but is used **exactly like** a normal
one: `data[100:200]` just works. The operating system takes care of loading into memory only
the pages you touch, and discarding them when space is needed.

Here it is worth being honest, because this often gets explained badly. Our 1 GB file
**would fit perfectly well in your 16 GB of RAM**. The reason for using `memmap` is not that
it does not fit:

1. **Instant startup.** Loading 1 GB from disk into RAM is a few seconds every time you
   launch the script. With `memmap` it is immediate: nothing is read until it is touched.
2. **The operating system's cache does the work.** Since you access random positions
   repeatedly, the OS ends up keeping in RAM what you use most. Free, and better than you
   would do it.
3. **It scales without changing anything.** If tomorrow you train with a 50 GB corpus, the
   same code keeps working.

If your corpus is small, loading it into RAM with `np.fromfile` is just as valid and
simpler. There is no magic here.

## How a batch is drawn

Here is the idea that makes training a language model so data-efficient.

You have the corpus as one extremely long strip of numbers. You pick a position at random
and take a window. The input is the window, and the target is **the same window shifted by
one token**:

```
corpus = [ 5, 8, 2, 9, 1, 7, ...]

x      = [ 5, 8, 2, 9]
y      = [ 8, 2, 9, 1]
```

Read the correspondence column by column:

```
seeing [5]            you must predict 8
seeing [5,8]          you must predict 2
seeing [5,8,2]        you must predict 9
seeing [5,8,2,9]      you must predict 1
```

**A single 4-token window produces 4 training examples**, not one. With our context of 512,
each sample gives 512 predictions. That is why language models learn so much from each pass:
the training signal is extremely dense.

This is possible thanks to module 06's causal mask, which stops position 2 from being able
to see token 3. Without it, the model would see the answer and learn nothing.

A batch is several of these windows stacked. With `batch_size=48` and `context_length=512`,
each `x` is a `(48, 512)` matrix = 24,576 tokens.

## Training and validation: why the cut is not random

You need text the model has **not** seen, so you can tell whether it is really learning or
just memorizing.

The usual reflex is to shuffle and split. **Here that is a mistake**, and the reason is
subtle: since the windows overlap, two samples starting at positions 100 and 101 share 511
of their 512 tokens. If you split at the token or window level, your validation set would be
full of fragments the model already saw in training. The validation loss would look
beautiful and mean nothing.

The fix is to cut **contiguously and from the end**: the last 0.5% of the corpus is set
aside whole. Since TinyStories is made of independent stories, those are complete stories
the model has never seen.

It is a particular case of a general principle: the validation set has to be independent of
the training one *in the unit that matters*. Here the unit is not the token, it is the
story.

## Performance details you actually notice

**`pin_memory` and `non_blocking`** (CUDA only). "Pinned" memory is memory the operating
system promises not to move, and that lets the GPU read it by DMA without the CPU getting
involved. Combined with `non_blocking=True`, the next batch's copy overlaps with the current
computation. In a small model, where the computation is short, this shows.

**The copy with `.astype(np.int64)`.** An `nn.Embedding`'s indices have to be `int64`, so
conversion is needed. And that conversion also materializes the `memmap`: without it,
PyTorch would be left pointing at disk-mapped memory and every access would be a read.

## Where the debate is

The random sampling with replacement we are going to use is not an epoch in the strict
sense: some windows will come up several times and others never. It is what nanoGPT does and
it works well, but it is not the only reasonable choice — an ordered traversal shuffled by
blocks gives coverage guarantees this method does not. With 500M tokens and a single pass
the difference is small; with many epochs over a small corpus, it would matter more.

Even more debated is what should be *inside* the corpus. The TinyStories paper argues that a
small, very clean dataset with the vocabulary of a 4-year-old lets tiny models generate
coherent text — something you do not get by training the same model on an equally sized
chunk of the internet. That the quality and *distribution* of the data matter as much as or
more than the quantity is today one of the most active lines in the field, and also one of
the least published: the big labs do not say what is in their datasets.

---

**Further reading:** Eldan & Li 2023, [TinyStories](https://arxiv.org/abs/2305.07759) ·
[nanoGPT](https://github.com/karpathy/nanoGPT) (its `get_batch` is practically this
module's). Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
