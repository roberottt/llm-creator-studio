# Glossary

The terms that appear in the course, explained in one or two sentences and in the order you
are going to meet them. If you are reading a `THEORY.md` and something is unfamiliar, it is
here.

In parentheses, the module where it is explained in depth.

---

## The basics

**Token** — The minimum unit of text the model handles. It can be a character, a word or
(usually) a piece of a word. `"tokenization"` might be three tokens: `token`, `iza`, `tion`.
Our final model handles 4096 distinct tokens. *(module 03)*

**Vocabulary** (`vocab_size`) — How many distinct tokens the model knows. It is a number you
choose when designing it, not something that gets discovered.

**Language model** — A function that, given a text, returns the probability of every possible
token as a continuation. That is all it is. *(module 00)*

**Autoregressive** — Generating one at a time, feeding each output into the input of the next
step. It is the reason generating text is slow and cannot be parallelized.

**Context** (`context_length`, *window*) — How many tokens back the model can look. Ours, 512.
Doubling the context quadruples the cost of attention.

**Probability distribution** — A list of non-negative numbers that add up to 1. The output of
a language model is always one of these, over the whole vocabulary.

**Sample** — Picking a token at random respecting its probabilities, instead of always taking
the most likely one. *(modules 00 and 14)*

**Greedy** (*argmax*) — The opposite of sampling: always taking the most likely token. It is
deterministic and tends to get stuck in repetitive loops. *(modules 00 and 14)*

**n-gram** — A language model that predicts by counting how many times each token followed
each sequence of `n` preceding tokens. It is what you build in module 00. It works, but the
table grows exponentially with `n`. *(module 00)*

**Curse of dimensionality** — The reason counting does not scale: as the context grows, the
possible combinations grow exponentially and the corpus covers an ever more ridiculous
fraction of them. Everything unseen is left with probability zero. It is the problem neural
networks solve by generalizing. *(module 00)*

**Smoothing** — The classic patches for the zero probabilities of an n-gram model: handing
out some mass to what was never seen, or blending models with several context sizes. It
eases the symptom, not the cause. *(module 00)*

**Maximum likelihood** — The criterion of choosing the parameters that make the observed
text most probable. Minimizing cross-entropy is exactly that. *(modules 00 and 05)*

**Hallucination** — The model generating something false with complete confidence. It is not
an added defect: it is the direct consequence of sampling from a distribution with no
verification step. *(module 00)*

**Generalize** — Getting things right on data not seen during training. It is the only thing
that separates learning from memorizing. *(module 00)*

---

## Training

**Parameter** (*weight*) — Every number the network adjusts during training. Our model has
8,933,440. GPT-4 has on the order of a million times more.

**Embedding** — The vector of numbers that represents a token. Tokens that appear in similar
contexts end up with similar vectors, and that is where the ability to generalize —which a
table of counts does not have— comes from. *(module 05)*

**Logit** — The raw score the model gives each token before turning it into a probability. It
can be any real number, positive or negative.

**Softmax** — The function that turns logits into probabilities: it exponentiates each one and
divides by the sum. Exponentiating is what lets you work with negative numbers.

**Loss** — How badly the model is doing. Concretely `-ln(the probability it gave the correct
token)`. If it gets it right with probability 1, the loss is 0. Training is minimizing this
number. *(module 05)*

**Cross-entropy** — The technical name of that loss. *(module 05)*

**Perplexity** — `e` raised to the loss. It is read as "how many options the model is
hesitating between, in practice". Perplexity 10 ≈ it is hesitating between 10 tokens.
*(module 15)*

**Gradient** — The derivative of the loss with respect to a parameter. It says which direction
to move that parameter in so the loss goes down. *(module 02)*

**Backpropagation** (*backward*) — The algorithm that computes every gradient at once, walking
the network backwards. It costs about 2 times what the forward costs, regardless of how many
parameters there are. *(module 02)*

**Forward** — Passing the data through the network and getting the output.

**Epoch** — One complete pass over the whole dataset.

**Batch** — A group of samples processed at once. Going one at a time wastes the GPU.

**Learning rate** (`lr`) — How far the parameters move at each step. The hyperparameter that
ruins the most training runs. *(module 11)*

**Optimizer** — The algorithm that decides how to apply the gradients. We will use AdamW.
*(module 11)*

**Overfitting** — When the model memorizes the training data instead of learning patterns. You
detect it because the training loss goes down and the validation one does not.

**Hyperparameter** — A number you choose (learning rate, number of layers), as opposed to a
parameter, which the model learns.

---

## The architecture

**Transformer** — The architecture of every modern LLM, published in 2017. Its central idea is
attention. *(modules 06-10)*

**Attention** (*self-attention*) — The mechanism that lets each token look at the previous ones
and decide which to pay attention to. *(module 06)*

**Query, Key, Value** (Q, K, V) — The three projections of attention. A useful metaphor: the
*query* is the question a token asks, the *key* is the label each token advertises itself with,
and the *value* is the content it contributes if it gets chosen. *(module 06)*

**Head** — Attention is done several times in parallel with different projections, so each
"head" can specialize. Ours has 8. *(module 06)*

**Causal mask** — It stops a token looking at the ones that come after. Without it the model
would cheat: it would see the answer. *(module 06)*

**Normalization** (LayerNorm, RMSNorm) — Rescales the values inside the network so they do not
grow or shrink uncontrollably layer after layer. *(module 07)*

**Residual connection** — Adding a block's input to its output (`x + f(x)`). It is what makes
deep networks trainable: it gives the gradient a direct path to the bottom. *(module 07)*

**FFN / MLP** — The part of each block that is not attention: two or three linear layers with a
non-linearity in between. It usually has more parameters than the attention. *(module 08)*

**GELU, SwiGLU** — Activation functions, the "non-linear" part without which the whole network
would collapse into a single matrix multiplication. *(module 08)*

**Positional embedding / RoPE** — How the model is told what position each token is in.
Attention on its own does not distinguish the order. *(module 09)*

**Weight tying** — Reusing the embedding matrix as the output matrix. It saves 1.3 million
parameters in our model. *(module 10)*

---

## Performance and hardware

**FLOP** — One floating point operation. It is used to measure how much training something
costs.

**MFU** (*Model FLOPs Utilization*) — What fraction of your GPU's theoretical power you are
really using. A small model rarely goes above 20%. *(modules 01 and 12)*

**fp32 / fp16 / bf16** — 32- and 16-bit numeric formats. fp16 takes half the space and runs
twice as fast, but its range is so narrow that gradients go to zero. *(module 01)*

**GradScaler** — The trick that makes fp16 viable: it multiplies the loss by a large number
before the backward so the gradients do not disappear. *(module 11)*

**AMP** (*Automatic Mixed Precision*) — Doing some operations in 16 bits and others in 32,
automatically.

**KV cache** — Saving the keys and values already computed so they are not recomputed for every
generated token. It makes generation several times faster. *(module 14)*

**Chinchilla** — The 2022 result that says how many tokens it is worth using to train a model
of a given size (about 20 per parameter). *(module 12)*

**Quantization** — Storing the weights with fewer bits (int8 instead of fp16) so the model
takes less space. *(module 17)*

---

## Post-training

**Pretraining** — The long phase: learning language by predicting the next token over an
enormous amount of text. It is what we do up to module 13.

**SFT** (*Supervised Fine-Tuning*) — Carrying on training an already pretrained model on
instruction-and-answer examples, so it obeys instead of merely continuing text. *(module 16)*

**LoRA** — Training only a few small matrices added to the model instead of all its weights.
Much cheaper. *(module 16)*

**RLHF** — Adjusting the model with human preferences. We will not do it, but it is one of the
things that separates this from a commercial model. *(module 17)*
