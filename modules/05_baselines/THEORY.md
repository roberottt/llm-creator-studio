# 05 — Baselines: what you are competing against, and how "it does badly" gets measured

## Why this module matters

**Because you need to know what you are competing against.**

You are going to train an 8.9-million-parameter model. When it finishes it will give you a
number — the loss — and you will have to decide whether that is good. Without a reference,
that number means nothing. A 2.49 is neither good nor bad until you know that the floor sits
at 4.13 and that a three-line model already reaches 2.49.

Here you build two different things, and it is worth not mixing them up:

- **The measuring stick.** Cross-entropy and perplexity, which are not arbitrary formulas:
  they have an exact interpretation that is worth understanding before you spend hours
  staring at a training curve. And above all **the floor**, `ln(V)`, which you are going to
  use for the rest of the course as a bug detector.
- **Three pre-Transformer models**, each a little less bad than the last. You will not use
  them for anything afterwards. They exist for two reasons: to give you something to compare
  against, and because the exact point where the third one gets stuck is the problem
  attention comes to solve in module 06. Skip this and attention looks like arbitrary magic;
  do it and attention shows up as the obvious answer to something you have watched fail with
  your own eyes.

### What you will know by the end

- How "it got it wrong" gets measured as a number, and **why with a logarithm**
- What perplexity is and how to read it at a glance
- The number `ln(V)` that will tell you, at step 0 of any training run, whether there is a
  bug
- That counting and learning by gradient give **exactly the same thing** when the model is
  simple
- How a model is written in PyTorch: `nn.Module`, `forward`, and what shape the tensors going
  in and out have. It is your first time, and the pattern does not change until the final GPT
- Why looking at more context helps, and why the naive way of doing it breaks

### What you are going to write

Five exercises. This theory is ordered so that you read them in this order, and **each one
has its own section with the matching numeric example**:

| Exercise | What it does | Where it is explained |
|---|---|---|
| 1. `uniform_baseline_loss` | The floor: `ln(V)` | [§ The floor](#the-floor-what-a-model-that-knows-nothing-scores) |
| 2. `bigram_counts` | Count which token follows each token | [§ Counting](#exercise-2-counting-the-pairs-bigram_counts) |
| 3. `bigram_nll` | Measure how well that table predicts | [§ Measuring the table](#exercise-3-measuring-the-table-bigram_nll) |
| 4. `NeuralBigram` | The same model, learned by gradient | [§ The neural bigram](#exercise-4-the-same-model-but-learned-neuralbigram) |
| 5. `BengioMLP` | The grandparent of LLMs (2003) | [§ Bengio's MLP](#exercise-5-looking-further-back-bengiomlp) |

The first three are short standalone functions. The last two are **your first two models in
PyTorch**, and between exercise 3 and exercise 4 there is a separate section
([§ What a model is in PyTorch](#a-stop-along-the-way-what-a-model-is-in-pytorch)) that
translates what you already did by hand in module 02 into the vocabulary of `torch.nn`. Read
it before you open exercise 4; without it, `nn.Embedding` and `F.cross_entropy` look like two
names you have to copy without knowing what they do.

### What it costs

2 hours. There is little code and almost all of it is dictated line by line in the
docstrings. The time goes into understanding what the numbers that come out mean, which is
what the module is about.

---

## Which part of the LLM this is

Building an LLM is five distinct jobs, and the course walks them in this order:

```
   0. FOUNDATIONS    what an LLM is, PyTorch, autograd        modules 00-02   ✔ done
   1. TOKENIZER      text  ->  numbers                        module 03       ✔ done
   2. DATA           numbers  ->  a learning task             module 04       ✔ done
   3. MODEL          the architecture that does the predicting modules 05-10  ← YOU ARE HERE
   4. TRAINING       adjust the weights until it gets it right modules 11-13
   ────────────────────────────────────────────────────────────────────────────
      and afterwards: generating text (14), evaluating (15), instruction tuning (16)
```

You are entering the model part, and this is the first stop. But watch out for one thing,
because it is confusing if nobody says it out loud: **none of the three models in this module
is part of the final GPT.** You are not building a piece that will fit in later; you are
building three dead ends, on purpose.

It is worth it because all three fail in different and very informative ways:

```
   model                what it looks at            where it breaks
   ───────────────────────────────────────────────────────────────────────────────
   uniform              nothing                     not a model, it is the floor
   bigram               1 token back                the context is ridiculous
   Bengio's MLP         k tokens back, fixed        parameters grow with k, and it
                                                    treats every position the same
   ───────────────────────────────────────────────────────────────────────────────
   attention (mod. 06)  the whole context, and it   ← what fixes both things
                        CHOOSES
```

That last row is the next module. This entire module exists so that the row reads like a
solution and not like a bright idea out of nowhere.

---

## The problem: putting a number on "it got it wrong"

A language model does not give an answer, it gives a **probability distribution** over the
whole vocabulary: one number per possible token, all positive and summing to 1. How do you
score that?

Imagine a vocabulary of only 4 words: `[cat, dog, house, blue]`. Given `"the "`, the model
says:

```
cat    0.70
dog    0.16
house  0.03
blue   0.11
```

And the word that actually came next was `cat`. It did well. If `blue` had come, it would
have done badly. But *how* badly, as a number.

The answer used throughout the field:

$$\text{loss} = -\ln(\text{probability the model gave to the correct token})$$

With the numbers above:

```
if cat came:   -ln(0.70) = 0.352     good
if blue came:  -ln(0.11) = 2.207     bad
```

Notice what is **not** looked at: what the model said about `dog` or about `house` does not
matter at all. Only the probability it gave to the token that actually came counts. And at
the extremes: if it had given 0.99 to `cat`, the loss would be 0.010; if it had given 0.001,
it would be 6.908.

### Before the probability come the logits

A detail worth having clear before exercise 4, because it is the way this actually happens
inside a network: **the model does not produce probabilities, it produces logits.** A logit
is a raw score, any number at all, positive or negative, with no constraint. The network
spits out four numbers and that is that:

```
   logits:   cat  2.0     dog  0.5     house  -1.0     blue  0.1
```

That is not a distribution: one of them is negative and they do not sum to 1. Turning them
into probabilities is the job of the **softmax**, which exponentiates each one (making them
all positive) and divides by the sum (making them sum to 1):

$$p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

```
   exponentiate:  e^2.0 = 7.389    e^0.5 = 1.649    e^-1.0 = 0.368    e^0.1 = 1.105
   the sum:       7.389 + 1.649 + 0.368 + 1.105 = 10.511
   divide:        0.7030           0.1569          0.0350            0.1051
```

Which are, with two more decimals, the numbers from the example above. And the loss if `cat`
came is `-ln(0.7030) = 0.3524`.

**Those two steps — softmax and then `-ln` — are what `F.cross_entropy` does in one go.**
That is why in exercises 4 and 5 you hand it the *logits* directly and never call softmax: if
you did, you would be applying it twice. It is a classic silent mistake, because the model
keeps training, just worse. PyTorch fuses the two into a single operation for numerical
stability: exponentiating a logit of 50 overflows in float32, and the trick that avoids it
(subtracting the maximum before exponentiating) lives inside `cross_entropy`.

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

Read it with the example in front of you: each term is one of those `-ln(0.70)` values, one
per token of the corpus, and `N` is how many tokens you evaluated. The average is what makes
the number comparable across corpora of different sizes: **the loss is nats per token**, not
total nats.

A *nat* is the unit you get from using the natural logarithm instead of log base 2. One nat
is 1.44 bits. The whole course works in nats because that is what `torch.log` returns, and
there is no more mystery to it than that.

## Perplexity: the same thing in readable units

A loss of 2.49 does not say much at a glance. **Perplexity** is simply $e^L$, and that does
have an interpretation:

$$\text{PPL} = e^{L}$$

It means **how many equally likely options the model is effectively torn between**. A
perplexity of 10 means that, on average, the model is as undecided as if it were choosing at
random among 10 words. A perplexity of 1 would be a perfect model.

With the numbers you will get from this module's demo, on character-level Shakespeare with
62 distinct characters:

| model | loss | perplexity | how to read it |
|---|---|---|---|
| uniform | 4.1271 | 62.0 | torn between all 62 characters, i.e. it knows nothing |
| bigram | 2.4916 | 12.1 | torn between 12: it has already ruled out 50 |
| Bengio's MLP (ctx 4) | 2.0939 | 8.1 | torn between 8 |

Perplexity is the quantity papers publish, and loss is the one you watch while training. They
are the same number.

## The floor: what a model that knows nothing scores

Here is the most useful number in all of training. A model that spreads probability equally
across the $V$ words of the vocabulary gives $P = 1/V$ to all of them, so:

$$L_{\text{uniform}} = -\ln(1/V) = \ln(V)$$

With our final 4096-token vocabulary: $\ln(4096) = 8.317$. Perplexity 4096, which is what you
would expect: if you are equally torn between 4096 options, you are torn between 4096
options.

**Use it like this:** when you start any training run, the loss at the first step has to be
almost exactly `ln(V)`. No more, no less.

- If it comes out **much higher** (12, 20), the initialization is wrong: the model starts
  with strong, mistaken opinions instead of honest ignorance.
- If it comes out **lower**, there is an information leak: almost always, a badly placed
  causal mask and the model seeing the answer.

The second case looks like good news and is the most expensive bug in the course. The loss
drops spectacularly, everything looks like it is going beautifully, and the trained model is
useless because at generation time that future it was peeking at does not exist.

**And you are going to see the first case live in this module's demo**, which is lucky
because it shows where it comes from. The `NeuralBigram` starts at 4.6434 when the floor is
4.1271: half a nat too much. The cause is that `nn.Embedding` initializes its weights from a
normal with standard deviation 1, and since in that model the rows of the table *are* the
logits, the model starts out with strong, random bets. A logit of +2 against one of −2 is
betting 55 to 1 before having seen a single piece of data, and getting that right by chance
is unlikely. That half nat is literally the price of having opinions without information.

That is why the GPT in module 10 initializes everything with `std=0.02`: with nearly identical
logits the softmax comes out nearly uniform and step 0 lands right on `ln(V)`.

### Exercise 1: the floor in one line (`uniform_baseline_loss`)

It is one line, `return math.log(vocab_size)`, plus a check that the vocabulary is positive.
The exercise has no difficulty whatsoever and it is the most important one in the module,
because what you are writing is not a function but **the check you will run at step 0 of
every training run left in the course**.

The two numbers you will actually run into:

```
   vocab   62   ->  ln(62)   = 4.1271     character-level Shakespeare (the demos)
   vocab 4096   ->  ln(4096) = 8.3178     the final model on TinyStories
```

(The full Shakespeare file has 65 distinct characters and `ln(65) = 4.1744`; this module's
demo keeps only the first 200,000 characters and 62 show up there. If you see both numbers
around the course, that is why: the floor depends on the vocabulary you actually use, not on
some universal constant.)

---

## Exercise 2: counting the pairs (`bigram_counts`)

**The problem.** You want a language model without training anything. The dumbest thing that
works: look at the corpus, write down how many times each character followed each character,
and to predict, consult the table.

A **bigram** is exactly that: a model whose context is a single token. To predict character
500 it looks only at 499 and ignores the previous 498. It is a terrible model and it is
surprising how far it gets.

**The example with numbers.** A three-character vocabulary, `a→0`, `b→1`, `c→2`, and the
corpus is `"ababc"`, that is, `ids = [0, 1, 0, 1, 2]`. The consecutive pairs are:

```
   a b a b c
   └─┘         (0,1)
     └─┘       (1,0)
       └─┘     (0,1)
         └─┘   (1,2)
```

Four pairs for five tokens: always one fewer than the length, because the last character has
no successor. And the count matrix, where the **row is the "from"** and the **column is the
"to"**:

```
             to a      to b      to c
   from a       0         2         0      <- b followed a twice
   from b       1         0         1
   from c       0         0         0      <- c never had a successor
```

That is exactly the result the exercise's test checks.

**The shape.** The matrix is `(V, V)`: one row per token that can be in front, one column per
token that can come after. With `V = 4096` that is 16.7 million cells, and that is the first
hint of why counting does not scale: for trigrams you would need `V³`, which is 68 billion
cells, nearly all of them zero.

**What the docstring dictates**, and why. The obvious way to fill the table is a `for` over
the pairs. It works and it is more readable, but with 500 million tokens that is 500 million
Python iterations. The vectorized version uses the fact that `tokens[:-1]` is the list of
every "from" and `tokens[1:]` the list of every "to", and asks PyTorch to add 1 at each
`(from, to)` position in one go with `index_put_(..., accumulate=True)`.

**The `accumulate=True` is not optional and it is this exercise's trap.** Without it,
`index_put_` *assigns* instead of adding: each repeated pair overwrites the previous one and
every count ends up being 1 instead of its real frequency. With the corpus `[0,0,0,0,0]`, the
correct answer is `counts[0][0] = 4`; without `accumulate` it comes out 1. There is a test
dedicated to exactly that.

---

## Exercise 3: measuring the table (`bigram_nll`)

You have counts. For them to be a model you need probabilities, and to know whether the model
is any good you need to evaluate it **on text you did not use for counting** (the validation
split from module 04; measuring on the same text you counted only tells you how much it
memorized).

**From counts to probabilities: normalize by rows.** Row `a` of the table above is
`[0, 2, 0]`, which sums to 2. Dividing: `P(a|a)=0`, `P(b|a)=1`, `P(c|a)=0`. The model claims
that after an `a` comes a `b` with absolute certainty.

And there is the disaster. If the pair `a→c`, which you never saw, shows up in validation, its
probability is 0, its logarithm is $-\infty$, and **since the loss is an average, that single
$-\infty$ takes the whole result down with it**. The perplexity of your entire validation set
goes to infinity because of one pair you did not see. This is not hypothetical: pass
`alpha=0` to your function with this data and it returns `inf`.

**The fix: Laplace smoothing.** You add a constant $\alpha$ to *every* count before
normalizing. It is admitting that "I have not seen it" is not the same as "it is impossible".

$$P(b \mid a) = \frac{C_{ab} + \alpha}{\sum_{b'} C_{ab'} + \alpha V}$$

With $\alpha = 1$ and $V = 3$, the table from exercise 2 becomes:

```
   row a:  [0,2,0] + 1  =  [1,3,1]  sums to 5  ->  [0.200, 0.600, 0.200]
   row b:  [1,0,1] + 1  =  [2,1,2]  sums to 5  ->  [0.400, 0.200, 0.400]
   row c:  [0,0,0] + 1  =  [1,1,1]  sums to 3  ->  [0.333, 0.333, 0.333]
```

Look at row `c`: a token for which you saw not a single successor ends up with a uniform
distribution. That is exactly what you want a model with no information to say. And look at
the denominator of row `a`: it is 5, not 2. By adding $\alpha$ to the $V$ entries of the row,
the total grew by $\alpha V$. **You do not write that $\alpha V$ from the formula yourself**:
it appears on its own if you add first and sum the row afterwards, which is the order the
docstring dictates. If you divided by `original_sum + alpha`, the probabilities would not sum
to 1.

**And now the loss.** Evaluate on `"abc"`, that is, `[0, 1, 2]`. The pairs are `(a,b)` and
`(b,c)`, and their probabilities in the smoothed table are 0.600 and 0.400:

```
   -ln(0.600) = 0.5108
   -ln(0.400) = 0.9163
   average    = 0.7136   <- the loss, in nats per token
   e^0.7136   = 2.041    <- the perplexity
```

And the floor for this vocabulary is `ln(3) = 1.0986`. The model goes below it, so it has
learned something. Those are the exact numbers the function returns if you implement it
correctly.

**How much to smooth.** $\alpha$ is a dial with two extremes, and both are bad:

| alpha | validation loss (Shakespeare) | what happens |
|---|---|---|
| 0.0001 | 2.4892 | almost no smoothing; with one unseen pair, `inf` |
| 0.01 | 2.4834 | the best in the table |
| 1.0 | 2.4916 | the classic value, reasonable |
| 100 | 2.9337 | the real counts start to drown |
| 10000 | 4.0430 | practically uniform: nearly the floor (4.1271) |

Raising $\alpha$ pushes the model towards ignorance. With a huge $\alpha$, the real counts are
noise next to the constant you added and every row comes out nearly uniform. This is the first
time in the course you meet a **hyperparameter**: a number you choose, that is not learned,
and whose best value is found by trying.

**The two silent traps in this exercise.** Neither raises an error; both produce plausible,
wrong numbers.

- **`keepdim=True` when summing the row.** Without it, `sum(dim=1)` returns shape `(V,)`
  instead of `(V, 1)`, and PyTorch's broadcasting rules end up dividing by **columns** instead
  of by rows. The result is perfectly believable and completely incorrect. There is a test
  that catches it by checking that every row sums to 1.
- **`.double()` and not `.float()`.** With large corpora you sum millions of counts, and
  float32 has 24 bits of mantissa: it starts losing precision sooner than one expects.

---

## A stop along the way: what a model is in PyTorch

The two remaining exercises are your first models in `torch.nn`, and everything you learn here
you will repeat unchanged all the way to the GPT in module 10. It is worth stopping to
translate the vocabulary, because conceptually **you already did this by hand in module 02**:
there you built an MLP with your own autodiff engine and wrote the whole training loop. This
is the same thing with the pieces already made.

### `nn.Module` and `forward`

A model is a class that inherits from `nn.Module` and defines two things:

```python
class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__()          # this line never gets forgotten
        self.layer = nn.Linear(...) # here the weights are CREATED

    def forward(self, x):
        return self.layer(x)        # here they are USED
```

The only thing to understand underneath: when you assign an `nn.Linear` or an `nn.Embedding`
to an attribute, `nn.Module` **registers it**. That is what makes `model.parameters()` find
them all, `model.to(device)` move them to the GPU, and saving the model save the weights. It
is the automatic equivalent of the `parameters()` you wrote yourself in module 02 by walking
over neurons by hand.

And a convention that is surprising the first time: **the model is called as `model(x)`, never
as `model.forward(x)`**. They are almost the same, but the first form goes through PyTorch's
internal hooks and the second skips them. In this course it makes no difference; the moment
you use hooks or `DataParallel`, it does.

### The shape of the tensors: `(B, T, V)`

Three letters you will see in every comment in the course:

```
   B  batch       how many sequences you process at once      (parallelism)
   T  time        how many tokens each sequence has           (the context)
   V  vocab       how many distinct tokens exist
```

The route through a language model is always this one:

```
   idx      (B, T)        integers: the ids of the input tokens
     │
     │  the model
     ▼
   logits   (B, T, V)     floats: one score for every possible token,
                          at every position, of every sequence
```

That is: for each of the `B × T` positions, the model emits `V` numbers. With `B=32`, `T=512`
and `V=4096` that is 67 million floats in a single tensor, and that is why logits are the
biggest memory consumer in the final training run, above even the activations.

### `F.cross_entropy` and the `reshape`

`F.cross_entropy` expects exactly two things: the logits with shape `(N, V)` and the targets
with shape `(N,)`, where `N` is "how many predictions you are scoring" and each target is the
**index** of the correct token (not a one-hot vector).

You have `(B, T, V)` and `(B, T)`. The translation is to flatten batch and time into a single
dimension, because the loss could not care less which sequence each prediction came from:

```python
loss = F.cross_entropy(
    logits.reshape(-1, self.vocab_size),   # (B, T, V) -> (B*T, V)
    targets.reshape(-1),                   # (B, T)    -> (B*T,)
)
```

This pair of lines is identical in the `NeuralBigram` and in the final GPT. It is worth
recognizing it now.

### Why `forward` returns `(logits, loss)`

Both models in this module return a tuple, and `loss` is `None` when you do not pass targets.
The reason is that there are two different situations:

- **Training**, where you have the right answer, you want the loss, and you do nothing with
  the logits.
- **Generating** (module 14), where there is no right answer to be had: you want the logits so
  you can sample the next token from them.

Returning `None` and not `0` in the second case is deliberate: a `0` would happily add into
anything and the bug would go unnoticed; a `None` blows up on the spot.

---

## Exercise 4: the same model, but learned (`NeuralBigram`)

**The idea.** Take the bigram from exercise 2 and, instead of filling the table by counting,
initialize it at random and let gradient descent adjust it. The whole model is one
`nn.Embedding(V, V)`: a table of `V` rows by `V` columns where **row `i` is directly the
logits of the token that follows token `i`**.

It looks like a writing trick and it is literally the same model. What is interesting is the
result: trained with cross-entropy, it converges to the normalized counts. The numbers
measured on Shakespeare:

```
   counting  (exercises 2 + 3):  2.4916
   learning  (this exercise):    2.4838
```

**The same model reaches the same place by two completely different routes**, and the 0.008
difference is smoothing noise plus how many steps you trained. Counting is instantaneous and
learning takes a few seconds, so at this point counting wins. The thing is that counting stops
here and learning does not: for the model in exercise 5 there is no longer any table to fill,
and for the GPT in module 10, far less so.

**Why `nn.Embedding` and not `nn.Linear`.** They are the same operation. An embedding is a
`Linear` whose input is a one-hot vector: multiplying a matrix by a vector that is all zeros
with a single 1 at position `i` gives, exactly, row `i` of the matrix. The difference is purely
about cost: the embedding **reads** the row it needs instead of doing the multiplication. With
`V=4096`, that is reading 4096 numbers versus doing 16.7 million multiplications, the vast
majority of them by zero.

That one-hot ↔ row equivalence is worth chewing on: it comes back in module 09 with positional
embeddings and in module 10 with weight tying.

**Careful with the name.** `self.token_embedding` is not optional: the test copies weights by
name to compare your model against the reference, and if you call it something else it fails
without there being anything wrong with the model.

---

## Exercise 5: looking further back (`BengioMLP`)

This is the model from Bengio et al. (2003), *A Neural Probabilistic Language Model*, and it is
the direct grandparent of all this: the first neural language model that actually worked,
thirteen years before the Transformer.

**The problem it solves.** The bigram looks at one token. Predicting the end of *"the cat
climbed up the ___"* by looking only at `the` is hopeless. You want to look `k` tokens back.
With counts you cannot: the table would need $V^k$ cells and they would be almost all empty
(the *curse of dimensionality* from module 00). With a network, you can.

**Bengio's two ideas**, still alive twenty years later and both present in your final GPT:

1. **Each token is represented as a learned dense vector**, not as an id with no structure. An
   id is a label: 47 and 48 are not alike in any way just for being consecutive. A vector of 24
   numbers can be like another one, and there lies all the ability to generalize that a count
   table does not have. If `dog` and `cat` end up with similar vectors, whatever the model
   learns about one is useful for the other **even if that exact combination never appeared in
   the corpus**. It is the difference between memorizing and learning.
2. **The probability of the next token is computed by a network**, not by a table.

**The route, with the shapes.** With `block_size=4` (it looks 4 characters back), `d_embed=24`
and `n_hidden=128`, on Shakespeare with `V=62`:

```
   idx       (B, 4)          the ids of the 4 previous characters
     │  embedding
     ▼
   emb       (B, 4, 24)      each one turned into its vector of 24 numbers
     │  reshape(B, -1)       ← CONCATENATE: glue the 4 vectors in a row
     ▼
   flat      (B, 96)         4 × 24 = 96
     │  hidden + tanh
     ▼
   h         (B, 128)        the hidden layer
     │  output
     ▼
   logits    (B, 62)         one logit per possible character
```

And notice the last row compared with the `NeuralBigram`: here the logits are `(B, V)`, with no
`T` dimension. This model makes **a single prediction per sample**, not one per position. That
is why `targets` is `(B,)` and why `cross_entropy` is called without any `reshape`: the shapes
already fit. It is the most important structural difference between the two exercises and the
likeliest source of confusion when writing them back to back.

The `tanh` is the one from the original paper. The reason there has to be a nonlinearity there
is the one you saw in module 02: without it, stacking two linear layers gives another linear
layer and the hidden layer would be useless.

**Concatenate, do not average.** The `reshape(batch, -1)` glues the embeddings one after
another, and that is what preserves **the order**. If you did `emb.mean(dim=1)` you would get a
perfectly valid vector of 24 numbers... in which `[the, cat, eats]` and `[eats, cat, the]` give
exactly the same thing. The model would lose any notion of what came first. There is a test
that checks this by passing the context in reverse and demanding that the output change.

And watch where the `-1` goes: `reshape(batch, -1)`, not `reshape(-1, batch)`. The second one
compiles, raises no error and produces garbage.

### Where it breaks, which is the reason it is here

The numbers measured in the demo, training all three with the same budget of 400 steps:

| model | context | loss (val) | parameters |
|---|---|---|---|
| bigram | 1 | 2.4916 | 3,844 |
| Bengio MLP | 2 | 2.1940 | 15,758 |
| Bengio MLP | 4 | **2.0939** | 21,902 |
| Bengio MLP | 8 | 2.1928 | 34,190 |

Two things to take from that, and both are module 06 poking through.

**First: the parameters grow with the context.** The hidden layer is
`Linear(block_size * d_embed, n_hidden)`, so its size is *linear* in the context length. You
can see it directly in the table:

```
   ctx 2:  Linear(48,  128)  =   6,272 weights
   ctx 4:  Linear(96,  128)  =  12,416
   ctx 8:  Linear(192, 128)  =  24,704
   ...
   ctx 512 with d_embed 320:  Linear(163840, 128)  -> impossible
```

A 512-token context, which is the one for the model you are going to train, is simply
unreachable down this road. And worse: the context is **fixed**. It is hard-wired into the
shape of the layer. You cannot feed 3 tokens to a model trained with 4, nor 5.

**Second, and deeper: the model treats each position as an independent input.** The 96 numbers
in `flat` are 96 unrelated inputs as far as the `hidden` layer is concerned. There is no way
for the model to say "of these 512 tokens the ones that matter right now are 3 and 47". The
weight of each position is fixed in the matrix and it is the same for every sentence in the
corpus, when what is needed is for it to depend on *what* is written at each position.

Solving both things at once — a long context without the parameters exploding, and deciding on
the fly which positions to pay attention to — is exactly what attention does. That is module
06.

**And a third lesson, for free, which is not about the model but about how models get
compared.** Look at the table again: context 8 comes out *worse* than context 4. This is not a
bug in the demo. All three trained for the same 400 steps, and the context-8 one has more than
twice the parameters of the context-2 one: on the same step budget, the big model is left half
trained. **Comparing architectures at equal steps is not comparing them at equal compute, and
it almost always unfairly favours the smaller model.** It is exactly the mistake the scaling
laws in module 12 come to correct, and you get to see it here with your own numbers before
anyone explains it to you in the abstract.

---

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

On smoothing there is an older and by now nearly closed discussion: Laplace is the simplest
method and it is not the best. Kneser-Ney, which distributes the leftover mass by looking at
how many distinct contexts each token appears in rather than spreading it equally, clearly
wins for n-gram models. We use Laplace here because the n-gram model is a baseline you are
going to abandon in the next module, and spending effort tuning it would be investing in the
dead end. But it is worth knowing that half a century of literature exists on how to
distribute the probability of what has not been seen, and that all of it stopped mattering
when neural models started generalizing instead of smoothing.

---

**Further reading:** Bengio et al. 2003,
[A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
· Shannon 1948, *A Mathematical Theory of Communication* (the equivalence between prediction
and compression). Stray terms are in [GLOSSARY.md](../../GLOSSARY.md).
