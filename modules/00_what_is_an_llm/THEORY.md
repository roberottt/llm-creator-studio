# 00 — What an LLM actually is

## Why this module matters

**Start here even if you are in a hurry.** It is the only module with no PyTorch, no
matrices and no derivatives, and it is the one that makes everything else make sense.

The reason: the rest of the course builds increasingly sophisticated pieces to do **one
single thing**. If you are not crystal clear on what that thing is, the following 17
modules are engineering without a purpose.

In an hour you are going to write a text generator that genuinely works, using dictionaries
and one division. And you are going to see that the loop driving it is *literally* the same
one ChatGPT runs.

### What you will know by the end

- What a language model is, exactly (spoiler: far less mystical than it looks)
- Why people say it "only predicts the next token", and what that really means
- How that token gets chosen, and why you do not always take the most likely one
- **Why neural networks are needed**, seeing with actual numbers why the obvious
  alternative smashes into a wall

### What it costs

One hour. It is the shortest module and the one with the best return.

---

## The idea, in one sentence

**A language model is a function that, given some text, tells you the probability of each
possible continuation.**

That is all. It does not "understand", does not "reason", does not "know". It receives a
piece of text and returns a list of probabilities, one for each word or character that could
come next.

If you give it *"The sky is coloured "*, a good model will return something like:

```
blue      0.72
grey      0.11
black     0.04
pink      0.02
potato    0.0000003
...
```

And that is it. That is the whole model. What you see when you talk to ChatGPT is this step
repeated: pick a word according to those probabilities, glue it onto the end of the text,
and ask again. Over and over, word by word.

That loop is called **autoregressive generation** ("auto" = itself, "regressive" = it feeds
back into itself). It is important that you see the consequence: the model does not plan the
whole sentence. It writes a token, reads it as if someone else had handed it over, and
decides the next one.

## Let us build one right now

A model has to get those probabilities from somewhere. The dumbest way there is, and one
that works: **counting**.

Take some text and note down, for each character, which ones followed it and how many times.
With the text `"banana"`:

```
after 'b'  ->  'a' 1 time
after 'a'  ->  'n' 2 times
after 'n'  ->  'a' 2 times
```

Now turn that into probabilities by dividing by the total:

```
after 'a'  ->  'n' with probability 2/2 = 1.0
```

With real text, the table for `'a'` would be more interesting:

```
after 'a'  ->  'n' 40 times,  'r' 25,  ' ' 20,  's' 15
total = 100
        ->  'n' 0.40,  'r' 0.25,  ' ' 0.20,  's' 0.15
```

That is a **probability distribution**: a list of non-negative numbers that sum to 1. The
whole course is about producing distributions over the next token. *How* we produce them
will change radically; what they are will not.

### Picking one

You have `{'n': 0.40, 'r': 0.25, ' ': 0.20, 's': 0.15}`. Which do you pick?

If you always take the most likely one (`'n'`), the model is deterministic and desperately
boring: with the same input it always writes exactly the same thing, and it tends to get
stuck in loops. That is why you **sample**: you roll a loaded die on which `'n'` comes up
40% of the time.

The method is the roulette wheel. Draw a random number between 0 and 1 and accumulate:

```
r = 0.61

'n'  running total = 0.40    0.61 > 0.40, keep going
'r'  running total = 0.65    0.61 < 0.65, this is where I go past!  ->  'r' comes out
```

Each token occupies a slice of the line [0,1] proportional to its probability, and the
random number lands in one of them. In module 14 you will see how this wheel gets
manipulated (temperature, top-k, top-p) to make the text more creative or more conservative.

## Why this is not enough

Your model from exercise 3 will generate something like this once trained on Shakespeare:

```
QUEO: hend f th s the wive an t ourourthe
```

It recognizes that vowels and spaces exist. It knows nothing else. The problem is that it
**only looks one character back**. To decide what comes after the `'a'` in *"the ca"*,
looking only at the `'a'` is hopeless.

The obvious reaction is to look further back: count trigrams, or windows of 10 characters.
And it works a bit better, until you smash into a wall. With a 4096-token vocabulary and a
window of 10, the table would have $4096^{10} \approx 10^{36}$ entries. There is no disk on
the planet, and besides, almost all of them would be empty: most combinations of 10 tokens
never appear at all, not even across the whole internet.

This is **the central problem of language modelling**, and it has a name: the curse of
dimensionality. Counting does not scale.

## What a neural network does

The solution is not to count better, it is to **generalize**. If the model has seen *"the
black cat sleeps"*, it should be able to say something sensible about *"the black dog
sleeps"* even though it has never seen it.

Counting cannot: to a table, `"cat"` and `"dog"` are two unrelated keys, as different from
each other as `"cat"` and `"umbrella"`.

A neural network represents each token as a **vector of numbers** learned from the data (an
*embedding*). If `"cat"` and `"dog"` end up with similar vectors — because they appear in
similar contexts — then what is learned about one transfers automatically to the other. That
is the whole point. The model compresses billions of impossible counts into a few million
numbers that capture *similarity*.

And once you have vectors, you need a way for each word to decide which of the previous ones
to pay attention to. That is **attention**, and it is module 06.

## The three numbers you will see constantly

**Token**: the unit of text the model handles. Here they will be characters; from module 03
onwards they will be word fragments. Our final model will have 4096 distinct tokens.

**Parameters**: the numbers the network learns. Ours will have 8,933,440. GPT-4 has on the
order of a million times more.

**Loss**: how badly it is doing. Concretely, `-ln(probability the model gave to the correct
token)`. If it gave 1.0 to the token that actually came next, the loss is 0. If it gave
0.01, the loss is 4.6. Training is minimizing this number, and in module 05 you will see why
this particular formula and not another.

## Where the debate is

That an LLM "only predicts the next token" is both true and misleading. The open question —
genuinely open, not rhetorical — is what internal structure a system has to build in order
to predict well. There is evidence that models trained only on text prediction end up
developing internal representations of things nobody taught them explicitly. Some read that
as emergent understanding and some read it as sophisticated statistics. Nobody has the
answer, and be suspicious of anyone who tells you otherwise in either direction.

---

**Further reading:** Shannon 1948,
[A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
— the paper that invented this, and where the counting models you are going to program today
already appear. If a term is unfamiliar, it is in [GLOSSARY.md](../../GLOSSARY.md).
