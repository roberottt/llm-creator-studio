# 00 — What an LLM actually is

## Why this module matters

**Start here even if you are in a hurry.** It is the only module with no PyTorch, no
matrices and no derivatives, and it is the one that makes everything else make sense.

Here is the reason: the following seventeen modules build increasingly sophisticated pieces
to do **one single thing**. If you are not crystal clear on what that thing is, everything
that comes after is engineering without a purpose — you will implement a multi-head
attention that passes the tests without knowing what it is for.

In an hour you are going to write a text generator that genuinely works, using dictionaries
and one division. It is not a teaching toy to be thrown away afterwards: the loop that
drives it is *literally* the same one ChatGPT runs, and you will rewrite it almost unchanged
in module 14 on top of your own nine-million-parameter GPT. The only thing that changes
between the two is where the numbers come from.

And you are going to see, with measured data rather than hand-waving, **why that simple
model smashes into a wall**. That collision is the reason neural networks exist. Anyone who
has not seen it first-hand spends the rest of the course believing transformers are
complicated for no reason.

### What you will know by the end

- What a language model is, exactly, with a definition that fits on one line (spoiler: far
  less mystical than it looks).
- Why people say it "only predicts the next token", and what that really means.
- How that token gets chosen, and why always taking the most likely one is a bad idea.
- **How you measure whether a language model is any good**: the loss, which is the number
  you will stare at for hours in module 13.
- **Why neural networks are needed**, seeing with real numbers why the obvious alternative —
  counting — hits a wall that cannot be walked around.

### What it costs

One hour: about twenty minutes of reading and forty of code. It is the shortest module in
the course and the one with the best return per minute invested.

---

## 1. The idea, in one sentence

**A language model is a function that, given some text, tells you the probability of each
possible continuation.**

That is all. It does not "understand", does not "reason", does not "know". It receives a
piece of text and returns a list of numbers: one for each word or character that could come
next.

If you give it *"The sky is coloured "*, a decent model will return something like:

```
blue      0.72
grey      0.11
black     0.04
pink      0.02
potato    0.0000003
...
```

And that is it. That is the whole model. Notice what is **not** there: there is no decision,
no sentence, no answer. There is a distribution over the vocabulary. What you see when you
talk to ChatGPT is this step repeated thousands of times: a word gets picked according to
those probabilities, glued onto the end of the text, and the question is asked again.

That loop is called **autoregressive generation** ("auto" = itself, "regressive" = it feeds
back into itself). The consequence is worth pausing on, because it is counter-intuitive:
**the model does not plan the whole sentence**. It writes a token, reads it back to itself
as if someone else had handed it over, and decides the next one. When an LLM gives you a
nicely structured three-paragraph answer, it had not thought that answer up before starting.
It chose token by token, and each choice constrained the next.

This also explains two behaviours people find odd:

- **An LLM has no memory between conversations.** The function only sees the text you hand
  it. If the application does not send the whole conversation back on every turn, then as
  far as the model is concerned nothing ever happened. In module 14 you will see the real
  loop, which passes it all the accumulated text at every step (and the trick that avoids
  recomputing all of it).
- **Hallucinations are not a bug in the system, they are the system.** If the model assigns
  0.03 to a false continuation and that gets sampled, out comes the false continuation.
  There is no database being consulted and no verification step. There is only a
  distribution.

## 2. Where those probabilities come from: counting

The model has to get those numbers from somewhere. The dumbest way there is — and one that
works well enough that Shannon published it in 1948 — is **counting**.

Take some text and note down, for each character, which ones followed it and how many times.
With the text `"banana"` you walk over the pairs `ba`, `an`, `na`, `an`, `na`:

```
after 'b'  ->  'a' 1 time
after 'a'  ->  'n' 2 times
after 'n'  ->  'a' 2 times
```

Now turn that into probabilities by dividing each count by the total of its row:

```
after 'a'  ->  total = 2  ->  'n' with probability 2/2 = 1.0
```

With `"banana"` the table is boring because everything comes out 1.0. With real text it is
not. These numbers are real — they come from counting over the 1,115,394 letters of Tiny
Shakespeare, the corpus you will use in the `demo`:

```
after 'a'  ->  'n' 10197 times,  't' 8339,  'r' 7081,  'l' 4149,  's' 3893,  ' ' 2685, ...
              total = 55507
           ->  'n' 0.1837,  't' 0.1502,  'r' 0.1276,  'l' 0.0747,  's' 0.0701,  ' ' 0.0484
```

And one row of the corpus is a small gem:

```
after 'q'  ->  'u' 609 times, and nothing else.  ->  'u' with probability 1.0
```

Six hundred and nine `q`s in the corpus, and all 609 of them followed by a `u`. The model
has learned an English spelling rule that nobody taught it, just by counting. That is the
whole mechanism of this course in miniature: **structure comes out of the data**.

### The formula

What you just did by hand is written like this:

$$P(x_t = c \mid x_{t-1} = a) = \frac{\text{count}(a, c)}{\sum_{c'} \text{count}(a, c')}$$

Read it slowly: the probability that the next character is `c`, given that the previous one
was `a`, is the number of times you saw `ac` together divided by the number of times you saw
`a` followed by anything at all. With `a = 'a'` and `c = 'n'`: 10197 / 55507 = 0.1837. It is
exactly the arithmetic above, only with symbols.

This recipe has a name, and I am giving it to you because you will run into it again: it is
the **maximum likelihood estimator**. "Likelihood" is the probability your model assigns to
the data you actually observed; the maximum likelihood estimator is the set of parameters
that makes it as high as possible. And it turns out that for a counting model, that optimal
choice is precisely the observed frequencies. When you minimize cross-entropy with gradients
in module 05, you will be doing the same thing by another route: looking for the parameters
that give the real text the highest probability.

The result of that division is a **probability distribution**: a list of non-negative
numbers that sum to 1. It is the central object of the entire course. *How* we produce it
will change radically — from one division to nine million parameters — but not *what* it is.
If you get lost in module 10, come back to this sentence: at the end of the whole
transformer there is a list of 4096 numbers that sum to 1.

## 3. Picking one

So you have `{'n': 0.40, 'r': 0.25, ' ': 0.20, 's': 0.15}`. Which do you pick?

The obvious option is to take the most likely one, `'n'`. That is called **greedy** (or
*argmax*), and it has two problems. The first is that the model becomes deterministic: with
the same input it writes exactly the same thing every time, word for word. The second is
worse, and you will measure it in module 14: greedy gets stuck in loops. It produces things
like *"the cat sat on the mat. the cat sat on the mat."* The logic of the loop is easy to
see: if the most likely token after some context returns you to a context you have already
visited, nothing breaks the cycle, because there is no randomness anywhere.

So instead you **sample**: you roll a loaded die on which `'n'` comes up 40% of the time,
`'r'` 25%, and so on.

The method is the roulette wheel, and it is what you are going to program in exercise 2.
Split the line from 0 to 1 into slices proportional to each probability:

```
|----'n'----|--'r'--|--' '--|-'s'-|
0          0.40    0.65    0.85   1.0
```

Draw a random number in `[0, 1)` and see which slice it lands in, accumulating as you go:

```
r = 0.61

'n'  running total = 0.40    is 0.61 < 0.40?  no, keep going
'r'  running total = 0.65    is 0.61 < 0.65?  yes  ->  'r' comes out
```

Why this is correct: the slice belonging to `'r'` is 0.65 − 0.40 = 0.25 long, and a uniform
number between 0 and 1 lands inside it exactly 25% of the time. Every token comes out with
its own probability, which is what we wanted. In module 14 you will see this wheel
deliberately deformed — temperature, top-k, top-p — to make the text more creative or more
conservative, but the underlying mechanism is this one.

## 4. How you tell whether a model is any good

This is where the number you will stare at for hours in module 13 shows up, so it is worth
understanding now, while the model still fits on a napkin.

The intuition first: a model is good if the real text **does not surprise it**. You take
text the model has not seen, and at every position you ask it what probability it assigned
to the character that actually came next. High probability, good. Tiny probability, bad.

To turn "tiny" into a number you can average, we use `-ln(p)`. With concrete values:

```
p = 1.00   ->  -ln(1.00) = 0.00     dead right, no surprise at all
p = 0.50   ->  -ln(0.50) = 0.69
p = 0.10   ->  -ln(0.10) = 2.30
p = 0.01   ->  -ln(0.01) = 4.61     it gave it 1 in 100 and it happened: big surprise
p -> 0     ->  -ln(p) -> infinity   it thought it was impossible and it happened
```

The **loss** is the average of that number over every position:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \ln P(x_t \mid \text{context})$$

Training is, literally, making this number go down. Nothing else.

Two anchors so you can tell a good loss from a bad one. The first: a model that knows
absolutely nothing and spreads probability evenly across the 65 distinct characters of Tiny
Shakespeare scores `-ln(1/65) = ln(65) = 4.174`. That is the absolute-zero mark; if your
model scores above 4.17, it is doing worse than rolling dice. The second: the counting model
you are writing today, with a single character of context, scores **2.470** on text it has
never seen. It has genuinely learned something, and it did it by counting.

There is a second way to read the same number, **perplexity**, which is `e` raised to the
loss. `e^2.470 = 11.8` reads as "the model is as undecided as if it were choosing at random
between 11.8 characters". It is the same information on a different scale; the literature
uses it more than the code does.

## 5. Why this is not enough

The model in exercise 3 looks **one single character back**. To decide what comes after
*"the ca"* it looks only at the `'a'` and rolls the dice. It is hopeless, and it shows:

```
context of 1 character (65 possible contexts, all of them seen)
  FRirmpavet wis, an wok, therongushy t t atheand nturorofouceir, m tatevete ar aterd
```

It recognizes that vowels, spaces and commas exist. It knows nothing else.

The obvious reaction is to look further back: instead of counting pairs, count triples, or
windows of four or six characters. That is called an **n-gram** model, and it **works**.
These outputs are real, generated with the same code you are writing today, changing nothing
but the context size:

```
context of 2
  Fin tis fall mounto degiver he of or he were menth I to herriand my lough mord whe hat

context of 3
  First perange is ther, rumous the had to did reseralic beford,
  Why, to my back, I hair lain!

context of 4
  First Camiliar,
  And hear'd his now him in his way, 'almost thy chainstruchio is in shown your women

context of 6
  First Gentleman:
  The senator:
  No more spices of my colour half way thee,
  I have shame:
  Upon him.
```

With six characters of context you get text with character names, line breaks in the right
places and real English words. The loss agrees: it drops from 2.470 to 0.880. The recipe
looks obvious — raise the context and keep counting.

Except no. This is where the road ends, and it ends at two different walls.

### Wall 1: the table explodes

Every extra character of context multiplies the number of *possible* rows by 65. This table
is measured on the real corpus:

| context | contexts seen | possible combinations | % of the space covered |
|---|---|---|---|
| 1 character  | 65      | 65         | 100% |
| 2 characters | 1,403   | 4,225      | 33% |
| 3 characters | 11,556  | 274,625    | 4.2% |
| 4 characters | 50,712  | 17,850,625 | 0.28% |
| 6 characters | 283,313 | 7.5 · 10¹⁰ | 0.00038% |

The contexts you have seen grow slowly — they cannot grow faster than the corpus, which is
one million characters long — while the possible ones grow exponentially. And this is at the
**character** level, with a ridiculous vocabulary of 65. Your final model will work with
4096 distinct tokens and a window of 512. The equivalent table would have $4096^{512}$ rows:
a number with more than 1800 digits. There is no disk on the planet, and there never will
be.

### Wall 2: almost everything has probability zero

The first wall is about space and sounds like an engineering problem. The second is worse,
because it is about data and no amount of hardware fixes it.

If a context never appeared in the training text, the table simply **has no row for it**. It
is not that it gives a bad probability: it gives none at all. The model goes literally mute,
which is why exercise 3 needs a `break` for that case.

Measured on the 10% of the corpus held out as validation:

| context | training loss | validation loss | impossible predictions |
|---|---|---|---|
| 1 character  | 2.452 | 2.470 | 0.17% |
| 2 characters | 1.903 | 1.967 | 1.4% |
| 3 characters | 1.491 | 1.571 | 4.2% |
| 4 characters | 1.216 | 1.286 | 10.1% |
| 6 characters | 0.761 | 0.880 | **34.8%** |

Read that last column. With six characters of context, **in more than a third of the
positions of the new text the model assigns probability zero to what actually happened**
(those positions do not even enter the loss computation; if they did, it would be infinite).
Raising the context improves the loss columns and ruins the one next to them. The model is
not learning English: it is memorizing Shakespeare, and the more context you give it, the
more it memorizes and the less it generalizes. That has a name and you will meet it again:
**overfitting**.

Look at the gap between the two loss columns as well. With context 1 it is 0.018; with
context 6, 0.119. That widening distance between how the model does on data it has seen and
on data it has not is the alarm signal you will be watching in module 13.

There are patches for the zero probabilities — they are called **smoothing**: hand out a
little probability to everything unseen, or blend the 6-character model with the 3-character
one when the first has nothing to say. They work, they were used for decades, and they do
not solve the underlying problem, which is this: to a table, `"cat"` and `"dog"` are two
distinct keys **with no relationship whatsoever**, as alien to each other as `"cat"` and
`"umbrella"`. What is learned about one helps not at all with the other. And if you have to
see every combination at least once in order to know anything about it, no corpus is ever
large enough.

This is **the central problem of language modelling**, and it is called the **curse of
dimensionality**.

## 6. What a neural network does

The solution is not to count better. It is to **generalize**: if the model has seen *"the
black cat sleeps"*, it has to be able to say something sensible about *"the black dog
sleeps"* even though that sentence appears nowhere.

The idea — published by Bengio and his co-authors in 2003, and the direct ancestor of
everything that follows — is to stop using the token as a dictionary key and represent it as
a **vector of numbers learned from the data**. That vector is called an *embedding*.

With small numbers, so you can see it. Imagine each word gets only two numbers, and that
after training on a lot of text these come out:

```
              animal   object
cat            0.91     0.05
dog            0.88     0.09
umbrella       0.02     0.95
```

Nobody wrote the labels "animal" and "object"; I added them after looking at the result. The
model only adjusted numbers until it predicted better, and `cat` and `dog` ended up close
together because they appear in similar contexts. The consequence is the one we were after:
what the model computes for `cat` and what it computes for `dog` come out nearly the same,
**because the inputs are nearly the same**. What is learned about one transfers to the other
for free, without ever having seen the sentence about the dog. A count table cannot do this
even in principle.

Two properties follow, and between them they take down both walls:

- **There is no table to explode.** Instead of $4096^{512}$ rows, the model stores a few
  million numbers and *computes* the answer. Yours will store exactly 8,933,440.
- **It never goes mute.** Whatever context you hand it, including one it has never seen, the
  computation produces a distribution. It may be a bad one, but it exists. No `break`
  required: the case "I have no row for this" does not exist.

What is missing — and it is the entire rest of the course — is *how* those vectors get
combined. Because one vector per word is not enough: each position has to decide which of
the previous ones to pay attention to. In *"the cat we saw yesterday in the park sleeps"*,
the word that governs `sleeps` is `cat`, not `park`. That mechanism is **attention**, and it
is module 06.

## 7. The map: what replaces what

Everything that follows is a piece that replaces some part of what you built today. Come
back to this table whenever a module feels gratuitous:

| What you do today | What replaces it | Module |
|---|---|---|
| One character = one token | Word fragments, via BPE | 03 |
| The character as a dict key | A learned vector (embedding) | 05 |
| Looking 1 character back | Looking 512 tokens back, deciding which ones matter | 06 |
| `count / total` | `softmax(logits)` | 05, 06 |
| Counting (one pass over the text) | Driving the loss down with gradients | 02, 05, 11 |
| The `break` when there is no row | Nothing: there is always an output | — |
| Sampling from the wheel as-is | Temperature, top-k, top-p | 14 |
| The `generate_naive` loop | The same loop, with a GPT inside | 14 |

That last row is the important one. The loop does not change. It never changes.

## 8. The three numbers you will see constantly

**Token**: the unit of text the model handles. Today it will be a character; from module 03
onwards, fragments of words. Our final model will know 4096 distinct tokens.

**Parameters**: the numbers the network learns. Ours will have 8,933,440 — the count is
exact and there is a test that checks it. Large commercial models sit on the order of a
hundred thousand times above that.

**Loss**: `-ln(probability the model gave to the correct token)`, averaged. Today you have
seen 4.174 as the mark of a model that knows nothing and 2.470 as the score of the simplest
counting model. When you train for real on TinyStories, the loss will fall from about 8.3 at
the start (`ln(4096)`, an untrained model spreading probability evenly) down into the region
where readable stories start coming out. That descent, plotted, is the whole of module 13.

## Where the debate is

That an LLM "only predicts the next token" is both true and misleading, and you should know
the argument is live before somebody sells it to you as settled.

Nobody disputes the mechanical claim: the training objective is to predict the next token,
full stop. The open question — genuinely open, not rhetorical — is **what internal structure
a system has to build in order to predict well**. There is evidence that models trained only
on text prediction end up developing internal representations of things nobody taught them
explicitly: interpretable directions associated with properties of the world have been
found, and in models trained on board-game transcripts, representations of the board state
have been recovered from the activations even though the model had only ever seen lists of
moves.

Some read that as emergent understanding and some read it as very sophisticated statistics,
and the argument gets tangled because both sides use "understand" with different and not
always explicit definitions. The honest position is that it is unresolved, and to be
suspicious of anyone who asserts it confidently in either direction.

A second debate, more practical and closer to what you will actually touch: **how far
next-token prediction goes as an objective**. Some argue that scaling it is enough; others
that there are capabilities — long-range planning, correcting one's own mistakes — that need
a different training objective. In module 16 you will see the first crack concretely: making
a model follow instructions takes more than pretraining, it takes an extra phase with a
different kind of data.

---

**Further reading:**

- Shannon 1948,
  [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
  — the paper that invented all of this. Section 3 already contains the counting models you
  are going to program today, generated by hand with a book and a pencil.
- Bengio et al. 2003,
  [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
  — the paper that lays out the curse of dimensionality exactly as told here and proposes
  embeddings as the way out. Everything that came after is descended from it.

If a term is unfamiliar, it is in [GLOSSARY.md](../../GLOSSARY.md).
