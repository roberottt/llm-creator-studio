# 05 — Baselines: how "how badly it does" gets measured

## Why this module matters

**Because you need to know what you are competing against.**

You are going to train an 8.9-million-parameter model. When it finishes it will give you a
number — the loss — and you will have to decide whether that is good. Without a reference,
that number means nothing.

Here you build three pre-Transformer models, each a little less bad, and above all you
establish **the floor**: what a model that knows absolutely nothing scores. That number,
`ln(V)`, you are going to use for the rest of the course as a bug detector, and it is the
cheapest and most informative check there is.

It is also where you learn to measure. Cross-entropy and perplexity are not arbitrary
formulas: they have an exact interpretation that is worth understanding before you look at a
training curve.

### What you will know by the end

- How "it got it wrong" gets measured as a number, and **why with a logarithm**
- What perplexity is and how to read it at a glance
- The number `ln(V)` that will tell you, at step 0 of any training run, whether there is a
  bug
- That counting and learning by gradient give **exactly the same thing** when the model is
  simple

### What it costs

2 hours. And they are your first two models in PyTorch.

---

## The problem: putting a number on "it got it wrong"

A language model does not give an answer, it gives a probability distribution over the whole
vocabulary. How do you score that?

Imagine a vocabulary of only 4 words: `[cat, dog, house, blue]`. Given `"the "`, the model
says:

```
cat    0.70
dog    0.10
house  0.10
blue   0.10
```

And the word that actually came next was `cat`. It did well. If `blue` had come, it would
have done badly. But *how* badly, as a number.

The answer used throughout the field:

$$\text{loss} = -\ln(\text{probability the model gave to the correct token})$$

With the numbers above:

```
if cat came:   -ln(0.70) = 0.357     good
if blue came:  -ln(0.10) = 2.303     bad
```

And at the extremes: if the model had given 0.99 to `cat`, the loss would be 0.010. If it
had given 0.001, it would be 6.908.

## Why the logarithm, and not something else

Three reasons, and all three matter.

**1. It turns products into sums.** The probability of a whole sentence is the product of
each token's probability: $P(w_1)P(w_2|w_1)P(w_3|w_1w_2)\cdots$. With 500 tokens of
probability ~0.1 each, that product is $10^{-500}$: exactly zero in floating point. In
logarithms it is a sum of 500 numbers around $-2.3$, perfectly manageable.

**2. It punishes being confident and wrong very hard.** The curve of $-\ln(p)$ shoots to
infinity as $p \to 0$. A model that says "I am absolutely sure" and fails takes an enormous
penalty; one that spreads its bets takes a moderate one. This pushes models towards
calibration, not just towards being right.

**3. It has an exact interpretation.** $-\log_2(p)$ is the number of *bits* you would need
to transmit that token if you encoded the message using the model's probabilities. A
language model **is** a compressor: the better it predicts, the fewer bits it needs. This
equivalence between prediction and compression comes from Shannon (1948) and it is not an
analogy, it is an identity.

Averaging over every token gives the **cross-entropy**, which is the function any LLM
minimizes:

$$L = -\frac{1}{N}\sum_{i=1}^{N} \ln P(\text{token}_i \mid \text{context}_i)$$

## Perplexity: the same thing in readable units

A loss of 3.2 does not say much at a glance. **Perplexity** is simply $e^L$, and that does
have an interpretation:

$$\text{PPL} = e^{L}$$

It means **how many equally likely options the model is effectively torn between**. A
perplexity of 10 means that, on average, the model is as undecided as if it were choosing at
random among 10 words. A perplexity of 1 would be a perfect model.

## The floor: what a model that knows nothing scores

Here is the most useful number in all of training. A model that spreads probability equally
across the $V$ words of the vocabulary gives $P = 1/V$ to all of them, so:

$$L_{\text{uniform}} = -\ln(1/V) = \ln(V)$$

With our 4096-token vocabulary: $\ln(4096) = 8.317$. Perplexity 4096, which is what you
would expect.

**Use it like this:** when you start training in module 11, the loss at the first step has to
be almost exactly 8.317. No more, no less.

- If it comes out **much higher** (12, 20), the initialization is wrong: the model starts
  with strong, mistaken opinions instead of honest ignorance.
- If it comes out **lower**, there is an information leak: almost always, a badly placed
  causal mask and the model seeing the answer.

It is the cheapest and most informative check there is, and it shows up as early as module
10.

## The three baselines you are going to build

**Count-based bigram.** You count how many times each token follows each token, normalize,
and you have a model. It is module 00 with more rigour.

A serious problem appears here: if a pair never appeared in training, its probability is 0,
its logarithm is $-\infty$, and **the perplexity of the entire validation set goes to
infinity because of a single unseen pair**. The classic fix is Laplace smoothing: adding
$\alpha$ to every count before normalizing.

$$P(b \mid a) = \frac{C_{ab} + \alpha}{\sum_{b'} C_{ab'} + \alpha V}$$

It is admitting that "I have not seen it" is not the same as "it is impossible".

**Neural bigram.** The same model, written as a network: an `nn.Embedding(V, V)` where row
$i$ is directly the logits of the token that follows $i$. Trained with gradient descent it
converges to the normalized counts. It is useful for seeing that *counting* and *learning*
give the same thing when the model is simple enough — and that from there on, learning
scales and counting does not.

**Bengio's MLP (2003).** The grandparent of all this. It concatenates the embeddings of the
previous $k$ tokens and passes them through an MLP. Two of its ideas are still alive twenty
years later: representing words as learned dense vectors, and modelling probability with a
network. Its limitation is exactly what attention comes to solve: the context has a fixed
size, and since it concatenates, the parameter count grows linearly with the context length.

## Where the debate is

Perplexity is a proxy metric, not the goal. It measures how well the model predicts the
validation corpus; nobody wants a model that predicts corpora, they want one that is useful.

The correlation between perplexity and usefulness is known to break at the extremes. A model
can lower its perplexity by memorizing, or by specializing in the quirks of one particular
dataset. And comparing perplexities across models with **different tokenizers makes no sense
at all**: if your vocabulary splits words into smaller pieces, each individual token is
easier to predict and your perplexity comes out better without the model being better. That
is why in module 15 we will use *bits per byte*, which is comparable.

Even so, within the same tokenizer and the same dataset, validation perplexity is still the
most reliable signal there is for knowing whether a training run is going well. It is a good
tool with a limited scope, and it is worth being clear about where it ends.

---

**Further reading:** Bengio et al. 2003,
[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
· Shannon 1948, *A Mathematical Theory of Communication* (the equivalence between prediction
and compression). Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
