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

**Tokenize** — Turn text into the list of integers the model understands, and back. It
happens **before** the model sees anything: it is not part of the network. *(module 03)*

**BPE** (*Byte Pair Encoding*) — The algorithm that decides what the tokens are: it starts
from the 256 bytes and keeps merging the most frequent pair of neighbours until the
vocabulary is full. Nobody writes the list of tokens; it is discovered by counting.
*(module 03)*

**Merge** — One of those fusions: the rule "whenever you see this pair, replace it with this
new id". A trained tokenizer is an ordered list of merges, and the order matters: when
encoding, they are applied in the same order they were learned. *(module 03)*

**Pre-tokenizer** — The regular expression that splits the text into words, numbers and
punctuation **before** pairs are counted, so no merge crosses from one word into the next.
Without it, BPE learns tokens like `". the cat sleeps"`. *(module 03)*

**Bytes fallback** — Working over bytes (0-255) instead of over Unicode characters. Since all
text is a sequence of bytes and all 256 are in the vocabulary, there is no such thing as an
unencodable character. When decoding, `errors="replace"` covers the opposite case: bytes that
do not form valid UTF-8 come out as `�` instead of taking generation down. *(module 03)*

**`<UNK>`** — The "unknown word" token of classic word-level tokenizers. It destroys
information irrecoverably, and with bytes fallback it stops being necessary. *(module 03)*

**Compression** (*bytes per token*) — How much text fits in a token on average. A larger
vocabulary compresses better (shorter sequences, fewer training steps) but eats the parameter
budget in the embedding table. That trade is what decides `vocab_size`. *(module 03)*

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

**Bigram** — The smallest n-gram that is good for anything: the model that predicts by
looking **one single token back**. Its count table is `V × V`. Terrible, and it is surprising
how far it gets. *(module 05)*

**Curse of dimensionality** — The reason counting does not scale: as the context grows, the
possible combinations grow exponentially and the corpus covers an ever more ridiculous
fraction of them. Everything unseen is left with probability zero. It is the problem neural
networks solve by generalizing. *(module 00)*

**Smoothing** — The classic patches for the zero probabilities of an n-gram model: handing
out some mass to what was never seen, or blending models with several context sizes. It
eases the symptom, not the cause. *(module 00)*

**Laplace smoothing** (*add-α*) — The simplest smoothing there is: add a constant `α` to every
count before normalizing, so that no probability is zero. Without it, a single unseen pair
sends the loss to infinity, because `-ln(0)` is infinity and the loss is an average. It is
admitting that "I have not seen it" is not the same as "it is impossible". It is not the best
smoothing —**Kneser-Ney** distributes the leftover mass according to how many distinct
contexts each token appears in, rather than equally, and clearly wins— but it is the simplest,
and the n-gram is a baseline you abandon in the next module. *(module 05)*

**Baseline** — A deliberately simple model that the real model gets compared against. Without
a baseline, a loss of 2.49 means nothing. The most important one is the **uniform** baseline,
which spreads probability equally and scores exactly `ln(V)`. *(module 05)*

**Maximum likelihood** — The criterion of choosing the parameters that make the observed
text most probable. Minimizing cross-entropy is exactly that. *(modules 00 and 05)*

**Hallucination** — The model generating something false with complete confidence. It is not
an added defect: it is the direct consequence of sampling from a distribution with no
verification step. *(module 00)*

**Generalize** — Getting things right on data not seen during training. It is the only thing
that separates learning from memorizing. *(module 00)*

---

## The data

**Self-supervised learning** — Training without human-made labels, because the correct answer
is drawn from the data itself: the answer to "which token comes next?" is, literally, the
token that came next. It is what allows training on raw text from the internet and the reason
LLMs took off. *(module 04)*

**Data pipeline** — Everything that happens between raw text and the batch that enters the
model: tokenizing, packing, storing, splitting into train/validation and sampling. In a real
lab it also includes filtering, deduplicating and deciding the mix of sources. *(module 04)*

**Corpus** — All the text you train on, already tokenized: a strip of several hundred million
integers. Ours is 500M tokens of TinyStories. *(module 04)*

**Deduplicate** — Removing repeated documents from the corpus. Without it the model sees the
same text many times and memorizes it instead of learning from it. TinyStories comes clean
and here it is not needed. *(module 04)*

**TinyStories** — The course's dataset: short stories generated with the vocabulary of a
four-year-old. The point of it is that a tiny model can learn to write them coherently, which
does not happen with a chunk of internet the same size. *(module 04)*

**`uint16`** — Unsigned 2-byte integer, from 0 to 65,535. The type each token is stored in:
with `int64` the same corpus would take four times as much. *(module 04)*

**Wrap around** (*silent overflow*) — What NumPy does when you convert to a type the number
does not fit in: it wraps the counter around without warning (65,536 becomes 0). It corrupts
the data without raising any error. *(module 04)*

**`memmap`** (*memory-mapped*) — An array whose data lives in a file and not in RAM, but which
is used exactly like a normal one. The operating system loads only the pages you touch.
*(module 04)*

**Sliding window** — How samples are drawn from the corpus: a position is picked at random and
`context_length` consecutive tokens are taken. Two adjacent windows share almost all their
tokens, and that is where the rule about not shuffling the train/validation split comes from.
*(module 04)*

**Validation set** — The chunk of corpus that is NOT trained on, reserved for measuring
whether the model generalizes or is memorizing. It is cut contiguously and from the end, never
at random. *(module 04)*

**Sampling with replacement** — Picking each window at random without keeping track of the
ones already drawn. Some will come up repeated and others never, so it is not an epoch in the
strict sense; in exchange the function has no state. *(module 04)*

**Pinned memory** (*page-locked*) — Memory the operating system commits to not moving, which
lets the GPU read it by DMA without the CPU acting as a middleman. With `non_blocking=True`,
the next batch's copy overlaps with the current computation. It only makes sense on CUDA.
*(module 04)*

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

**Nat** — The unit the loss is measured in when the natural logarithm is used, which is what
`torch.log` does and therefore what this whole course uses. One nat is 1.44 bits.
*(module 05)*

**One-hot** — A vector of zeros with a single 1, at the token's position. Multiplying a matrix
by a one-hot vector gives exactly the corresponding row of the matrix, and that is why
`nn.Embedding` and `nn.Linear` are the same operation: the embedding **reads** the row instead
of multiplying by a matrix full of zeros. *(module 05)*

**Hidden layer** — Any layer of a network that is neither the input nor the output. "Hidden"
only means nobody looks at its numbers directly. *(modules 02 and 05)*

**Perplexity** — `e` raised to the loss. It is read as "how many options the model is
hesitating between, in practice". Perplexity 10 ≈ it is hesitating between 10 tokens.
*(module 15)*

**Bits per byte** — The quality metric that **can** be compared across models with different
tokenizers, because it normalizes by bytes of the original text rather than by tokens. It has an
exact interpretation: how many bits it would take to transmit the text using the model as a
compressor. gzip sits at ~2.5 and the best LLMs at 0.6-0.8. *(module 15)*

**Data contamination** — The test set being inside the training corpus. Since benchmarks are on the
internet and models are trained on the internet, telling "it learned" from "it has seen it" is
hard, and it is one of the field's serious methodological problems. *(module 15)*

**LLM-as-a-judge** — Using another language model to score the answers. It is what gets done today
and it has documented biases: it prefers long answers, prefers its own style and is sensitive to
the order the options are presented in. *(module 15)*

**Emergent abilities** — Sudden jumps in capability with scale. A 2023 analysis argues many are
artifacts of measuring with all-or-nothing metrics: a continuous metric would show a smooth curve.
Under discussion. *(module 15)*

**Entropy** — The same as perplexity but without exponentiating: it measures how spread out a
distribution is. Maximal when everything is equally likely (`ln(n)`), near zero when all the
mass is on one option. In module 06 it is used to measure whether attention spreads out or
fixates on a single token. *(modules 05 and 06)*

**Dropout** — Switching off a random fraction of the numbers during training, so the model does
not depend too much on any one of them. It is disabled during evaluation, and you have to
remember to do that by hand when the operation does not check the mode on its own.
*(modules 06 and 11)*

**Gradient** — The derivative of the loss with respect to a parameter. It says which direction
to move that parameter in so the loss goes down. *(module 02)*

**Backpropagation** (*backward*) — The algorithm that computes every gradient at once, walking
the network backwards. It costs about 2 times what the forward costs, regardless of how many
parameters there are. *(module 02)*

**Chain rule** — If `y` depends on `u` and `u` depends on `x`, then `dy/dx = (dy/du)·(du/dx)`.
All of backpropagation is this, applied operation by operation. If a variable has an influence
through several paths, its contributions are **added**. *(module 02)*

**Compute graph** — The record of which operations were done, on which operands and in what
order. It builds itself during the forward pass and it is what makes walking backwards
possible. In our engine it is the `_prev` and `_op` fields of each `Value`. *(module 02)*

**Reverse-mode autodifferentiation** — The technique that computes exact derivatives by
breaking the computation into elementary operations and walking the graph backwards. Neither
numerical (approximate and expensive) nor symbolic (unmanageable). It is what is inside
`torch.autograd`. *(module 02)*

**Topological order** — The order the graph has to be walked in so that no node passes on its
gradient before receiving it from all of its parents. A wrongly computed order gives incorrect
gradients without raising any error. *(module 02)*

**Gradient descent** — The learning rule: `p -= lr * p.grad`. Move each parameter a little
against its gradient, because the gradient points the way the loss goes up. *(module 02)*

**Neuron** — The smallest unit: a weighted sum of its inputs plus a bias,
`w₁x₁ + w₂x₂ + … + b`, passed through a non-linear function. The `w`s and the `b` are its
parameters. *(module 02)*

**Bias** — The constant term `b` of a neuron: it lets the neuron shift its output without
depending on the input. *(module 02)*

**Activation function** — The non-linear part of a neuron (`tanh`, `relu`, `gelu`). Without it,
stacking layers is pointless: the composition of linear functions is another linear function.
*(modules 02 and 08)*

**MSE** (*mean squared error*) — A loss for predicting numbers: `mean((prediction - target)²)`.
It is used in module 02; for predicting tokens, cross-entropy is used instead. *(module 02)*

**Forward** — Passing the data through the network and getting the output.

**Epoch** — One complete pass over the whole dataset.

**Batch** — A group of samples processed at once. Going one at a time wastes the GPU.

**Learning rate** (`lr`) — How far the parameters move at each step. The hyperparameter that
ruins the most training runs. *(module 11)*

**Optimizer** — The algorithm that decides how to apply the gradients. We will use AdamW.
*(module 11)*

**Adam** — The standard optimizer in deep learning. Two ideas: a moving average of the gradients
(the **momentum**, which cancels each batch's noise) and a moving average of the squared gradient
to divide by, so that **each parameter ends up with its own effective learning rate**. That is why
a single global `lr` works for the whole model. *(module 11)*

**AdamW** — Adam with the *weight decay* **decoupled**: applied directly to the parameter instead
of added to the gradient. In the coupled version the decay goes through the division by `√v` and
its effect ends up depending on each weight's gradient magnitude; decoupled, it is uniform. That is
the W. *(module 11)*

**Momentum** — The moving average of the recent gradients. Every batch is a different sample and
its gradients are noisy; averaging cancels the noise and leaves the consistent direction.
*(module 11)*

**Bias correction** — The adjustment that compensates for Adam's moving averages starting at zero
and therefore underestimating magnitudes in the first steps. You divide by `1 - β^t`, and it fades
out on its own as `t` advances. Without it training can diverge before it starts. *(module 11)*

**Weight decay** — Pushing the weights towards zero so they do not grow without control. It is
applied **only to matrices** (parameters with 2 dimensions or more): not to normalization scales or
biases, because pushing an RMSNorm scale towards zero is pushing the layer's output towards zero.
Applying it to everything gives no visible error and degrades the result. *(module 11)*

**Schedule** (*lr scheduler*) — How the learning rate changes over the run. Ours has two segments:
linear warmup up to the maximum and then cosine decay down to a 10% floor, which is not crossed
because below it the model stops learning and the compute is wasted. *(module 11)*

**Gradient clipping** — If the **global** norm of all the gradients exceeds a threshold, they all
get multiplied by the same factor. Global and not per tensor: that way you limit how far you move
without changing the direction. It caps the damage a single odd batch can do. *(module 11)*

**Parameter groups** — Subsets of the model with different hyperparameters. PyTorch takes them as a
list of dictionaries with the `"params"` key; any other key overrides the default value for that
group only. *(module 11)*

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
would cheat: it would see the answer. It is applied by putting `-inf` in the forbidden scores
**before** the softmax, not by erasing weights afterwards: that way each row still sums to 1.
*(module 06)*

**Scores** — The `T × T` matrix of dot products between queries and keys, before the softmax.
Cell `(i, j)` says how interested token `i` is in token `j`. *(module 06)*

**Dot product** — Multiplying two vectors component by component and adding. It measures how
similar they are: the more aligned, the larger the number. It is the operation attention uses
to decide who to pay attention to. *(module 06)*

**Scaling by √d_k** — Dividing the scores by the square root of the query size. The dot product
of two vectors of dimension `d_k` has variance `d_k`, and without correcting for it the softmax
saturates: the weights go to 0 and 1, their derivative `p(1-p)` vanishes and the layer stops
learning. *(module 06)*

**Output projection** (`out_proj`) — The fourth matrix of an attention layer, which does not
appear in the paper's formula. It mixes the heads' results together; without it they would be
sealed channels. *(module 06)*

**SDPA** (`scaled_dot_product_attention`) — PyTorch's fused implementation of attention: it does
the four steps in a single kernel without materializing the whole `T × T` matrix. It is the one
real training uses. *(module 06)*

**Induction head** — An attention head that learns on its own to detect the pattern "…A B … A"
and predict B. It is the best-understood component of a Transformer and the success story of
mechanistic interpretability. *(module 06)*

**Normalization** (LayerNorm, RMSNorm) — Rescales the values inside the network so they do not
grow or shrink uncontrollably layer after layer. *(module 07)*

**Residual connection** — Adding a block's input to its output (`x + f(x)`). It is what makes
deep networks trainable: it gives the gradient a direct path to the bottom. When you
differentiate, that sum contributes a `1` no layer can attenuate. *(module 07)*

**Residual stream** — The main channel carrying each token's representation from one end of the
model to the other. Every block reads from it, computes something and **adds** its contribution
back, instead of replacing it. *(module 07)*

**Pre-norm / post-norm** — Whether the normalization goes inside the branch (`x + f(norm(x))`)
or wrapped around the sum (`norm(x + f(x))`). Only the parentheses move, and it decides whether
the gradient crosses a normalization per layer or arrives at the bottom intact. Everything
modern is pre-norm. *(module 07)*

**BatchNorm** — The normalization computed along the batch, not along each token's dimensions.
Not used in Transformers: one example's result would depend on who it shares a batch with, and
at inference with a single example you have to fall back on stored statistics. *(module 07)*

**Vanishing / exploding gradient** — When the gradient gets multiplied layer after layer by a
factor smaller or larger than 1 and ends up at zero or at infinity. With 64 linear layers it
reaches **exactly** zero, by floating-point underflow. *(module 07)*

**Warmup** — Starting training with a very small learning rate and raising it gradually over the
first steps. Post-norm needs it in order not to explode; pre-norm does not. *(modules 07 and 11)*

**FFN / MLP** — The part of each block that is not attention: two or three linear layers with a
non-linearity in between. It usually has more parameters than the attention: in ours, 68% of
every block. The acronym stands for *feed-forward network*, and it describes how the information
flows: in one side and out the other, with no loops and **without looking sideways** — each token
is processed separately, with no idea the others exist. Looking sideways is the attention's job.
*(module 08)*
**MLP** (*multi-layer perceptron*) is another name for the same box, the one used in the code; a
classic FFN **is** a two-layer MLP. Careful with the name: in module 02, "MLP" is the whole
network (layers of neurons chained together); in a transformer it is only that sub-block of each
layer.

**GELU, SwiGLU** — Activation functions, the "non-linear" part without which the whole network
would collapse into a single matrix multiplication. *(module 08)*

**Non-linearity** — Any function that does not satisfy `f(ax+b) = a·f(x)+b`. It is the only thing
that makes stacking layers worth anything: without it, a hundred linear layers in a row are
exactly equivalent to a single matrix. *(module 08)*

**ReLU** — The simplest activation, `max(0, x)`. Its derivative is **exactly zero** across the
whole negative range, so a neuron that ends up always negative stops receiving gradient forever:
that is the *dead neuron* (*dying ReLU*). GELU and Swish avoid it because their derivative there
is small but not null. *(module 08)*

**Swish / SiLU** — `z · σ(z)`. Practically the same curve as GELU despite coming from a
completely different origin (an automated search over activations, not a probabilistic argument).
It is the one on SwiGLU's gate branch, and in PyTorch it is called `F.silu`. *(module 08)*

**Gate** — One of the two branches of a GLU: it multiplies the other element by element and
decides how much signal passes through each dimension. What sets it apart from a normal
activation is that the filtering **depends on the input**, and is decided for each dimension and
each token. *(module 08)*

**d_ff** — The FFN's inner dimension, the one it expands to before compressing again. In our
model, 896: two thirds of the classic `4 × d_model`, rounded up to the next multiple of 64.
*(module 08)*

**Key-value memory** (of the FFN) — The reading of the FFN in which each row of the first matrix
is a pattern to be detected and each column of the second is what gets written back into the
residual stream if that pattern shows up. It is a hypothesis with partial evidence, not an
established result. *(module 08)*

**Positional embedding** — How the model is told what position each token is in. Attention on
its own does not distinguish the order. There are three families: a learned table (GPT-2), a
fixed table of sines and cosines (the 2017 paper) and RoPE. *(module 09)*

**RoPE** (*Rotary Position Embedding*) — The positional encoding used by Llama, Mistral and our
model. Instead of **adding** something to the vector, it **rotates** it by an angle proportional
to the position, pair of dimensions by pair of dimensions. It adds not one parameter to the model
and it does not change the vectors' length. *(module 09)*

**Permutation equivariance** — The property of attention whereby shuffling the input tokens
shuffles the output the same way and changes nothing else. In other contexts it is a virtue; in
language it is a fatal flaw, and it is the problem positional encoding solves. *(module 09)*

**Absolute / relative position** — "I am token 7" versus "I am two positions behind that one".
The relative one generalizes better: what is learned at one point in the sequence works anywhere
else. The property that makes RoPE relative is that the dot product of two rotated vectors depends
only on the difference of angles. *(module 09)*

**Extrapolate** — Using the model with sequences longer than the ones it saw during training.
With a learned table it is impossible (there is no row to look up); with RoPE it is possible, but
quality degrades quite a lot, hence the family of techniques for extending the context after the
fact (position interpolation, NTK-aware scaling, YaRN). *(module 09)*

**Buffer** — A tensor that travels with the model, moves to the GPU with it and is saved with it,
but is **not a parameter**: it is not trained. RoPE's `cos` and `sin` tables are buffers.
*(module 09)*

**ALiBi** — An alternative to RoPE: instead of rotating, it subtracts from the attention scores a
penalty proportional to the distance between the two tokens. *(module 09)*

**Weight tying** — Reusing the embedding matrix as the output matrix. It saves 1.3 million
parameters in our model, 15%. It is not a copy: both modules point at the same tensor, and each
weight receives gradient by two routes. *(module 10)*

**Initialization** — The values the weights start at before training. Not a detail: it decides
whether the model trains well or does not train. In ours, everything at `std=0.02` except the
projections that write into the residual stream, which use `0.02/√(2·n_layers)`. *(module 10)*

**Non-embedding parameters** — The total minus the embeddings. It is the number the scaling laws
use, because embeddings grow with the vocabulary rather than with the depth, and they do not take
part in the per-token compute the way the layers do. In ours, 7,622,720 out of 8,933,440.
*(modules 10 and 12)*

**Checkpoint** — The file the model is saved to: the weights plus whatever is needed to resume
training, which is four things — weights, optimizer state, GradScaler state and step number. With
only the weights, Adam starts with its moments at zero and the model lurches right at the resume
point. Buffers marked `persistent=False` are not saved there, because they recompute themselves
when the model is built. *(modules 10, 11 and 13)*

**Step** — One update of the weights. Not to be confused with **epoch**, which is a full pass over
the data: our final run is 10,172 steps and less than one epoch. *(module 13)*

**Overfitting a single batch** — The cheapest sanity check there is: give the model the same batch
over and over and verify the loss drops almost to zero. A model with millions of parameters
memorizes four sequences without breaking a sweat; if it does not, there is a bug in the
machinery. It catches gradients that do not arrive, the forgotten `zero_grad` and a badly built
optimizer; it catches nothing to do with generalizing. *(module 13)*

**ETA** — How long is left, estimated from the measured rate. It is shown with precision
proportional to its magnitude: past one hour, seconds are noise. *(module 13)*

**state_dict** — PyTorch's dictionary with all of a model's tensors, parameters and buffers,
indexed by name. It is what gets saved and loaded. *(module 10)*

---

## Performance and hardware

**FLOP** — One floating point operation. It is used to measure how much training something
costs.

**TFLOPS** — Trillions of FLOPs per second. The unit GPU throughput is measured in. The peak
on the spec sheet and the one you actually get are a factor of 3 or more apart.
*(module 01)*

**MFU** (*Model FLOPs Utilization*) — What fraction of your GPU's theoretical power you are
really using. A small model rarely goes above 20%. *(modules 01 and 12)*

**Compute-bound / memory-bound** — Whether an operation is limited by arithmetic throughput
(a large matmul) or by memory bandwidth (an activation, a normalization). The FLOP formula
only sees the former, and much of the gap between estimated and real time comes from that. It
shows up very clearly in module 12's MFU curve: it rises with the batch and then flattens, and
that point is where you stop being limited by kernel launch cost and start being limited by
computation. *(modules 01 and 12)*

**Tensor cores** — The units in an NVIDIA GPU specialized in multiplying small matrices in
16 bits. They are what produces the big numbers on the spec sheet, and they only pay off
with matrices that are large enough. *(module 01)*

**Compute capability** (`sm_75`, `sm_80`…) — The generation of an NVIDIA GPU, which
determines what it can do. bf16 and FlashAttention-2 need `sm_80` (Ampere); the RTX 20
series stops at `sm_75`. *(module 01)*

**Gradient checkpointing** — Recomputing the forward pass of some blocks during the backward
pass instead of storing their activations. Saves memory and raises the cost from 6N to 8N
per token. *(modules 01 and 12)*

**fp32 / fp16 / bf16** — 32- and 16-bit numeric formats. fp16 takes half the space and runs
twice as fast, but its range is so narrow that gradients go to zero. *(module 01)*

**GradScaler** — The trick that makes fp16 viable: it multiplies the loss by a large number
before the backward so the gradients do not disappear. *(module 11)*

**AMP** (*Automatic Mixed Precision*) — Doing some operations in 16 bits and others in 32,
automatically.

**KV cache** — Storing the keys and values already computed so they do not get recomputed for
every generated token. It turns an `O(N²)` cost into `O(N)`, and its gain grows with length. The
**queries** are not cached: every new token needs its own question. *(module 14)*

**Prefill / decode** — The two phases of cached generation. In *prefill* the whole prompt goes in
at once and fills the cache — and there a causal mask **is** needed. In *decode* a single token
goes in per step, which legitimately sees the whole past and needs no mask. *(module 14)*

**Temperature** — Dividing the logits before the softmax. Below 1 it sharpens the distribution
(more deterministic), above 1 it flattens it (more variety), and in the limit at 0 it is greedy.
*(module 14)*

**Top-k** — Keeping the `k` largest logits and setting the rest to `-inf`. Its flaw is that `k` is
fixed, so it does not adapt to how sure the model is. *(module 14)*

**Top-p** (*nucleus sampling*) — Keeping the smallest set whose cumulative probability **exceeds**
`p`. The number of candidates adapts on its own: few when the model is sure, many when it hesitates.
Watch the off-by-one: the token that crosses the threshold gets in. *(module 14)*

**Repetition penalty** — Lowering the logit of already-emitted tokens to break loops. You have to
**divide if the logit is positive and multiply if it is negative**; always dividing would make
negative-logit tokens more probable, and those are the majority. *(module 14)*

**Chinchilla** — The 2022 result that says how many tokens it is worth using to train a model
of a given size (about 20 per parameter). It was derived by training over 400 models, and it
showed GPT-3 was twelve times under-trained. *(module 12)*

**Scaling laws** — The empirical relationships between model size, amount of data, compute and
loss. They predict **loss**, not capabilities, and their coefficients were fitted to a specific
range of scales: extrapolating outside it is not justified. *(module 12)*

**Compute budget** — How many total FLOPs you can afford to spend on training. It is the
constraint Chinchilla starts from: given a budget, how to split it between size and data.
Doubling it does not double the optimal model, it grows it by 41%. *(module 12)*

**Over-trained / under-trained** — Above or below those ~20 tokens per parameter. Neither is
automatically a mistake: Chinchilla optimizes **training** compute, and if the model is going to
run a lot it pays to over-train a small one, because inference is paid every time. Llama-3 is 90
times above on purpose. *(module 12)*

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
