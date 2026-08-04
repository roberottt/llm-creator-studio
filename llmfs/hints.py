"""Progressive hints, one set per exercise.

Three levels, designed to unblock you without burning the exercise:

  1. **Conceptual.** What you are trying to achieve and why. No code, no formulas. Often
     this is enough: the blocker was not technical, it was that the goal was unclear.
  2. **Technical.** The exact formula, the tensor shapes, or which PyTorch function is the
     right one. Still no code.
  3. **Structural.** The skeleton in pseudocode, including the places where people get it
     wrong (swapped axes, broadcasting, the extra `-1`). Still not the written solution:
     that part is yours.

If level 3 does not unblock you, `modules/NN_*/SOLUTION.md` explains the whole solution and
`llmfs/reference/` has the code. Looking at it is not cheating: cheating would be looking at
it without having tried anything. Learning by reading a solution you have already wrestled
with works very well; reading it cold does not work at all.
"""

from __future__ import annotations

HINTS: dict[str, dict[str, tuple[str, ...]]] = {
    # ------------------------------------------------------------------ module 00
    "00_what_is_an_llm": {
        "next_token_probs": (
            "You have counts: 'n' came up 40 times, 'r' 25... They are whole numbers that "
            "mean nothing on their own, because they depend on how long the text was.\n\n"
            "What you need is the PROPORTION: out of every time you looked, what fraction "
            "was an 'n'. And those fractions have to sum to exactly 1, because something "
            "always had to come next.",

            "Probability of each token = its count divided by the sum of ALL the counts.\n\n"
            "    total = 40 + 25 + 20 + 15 = 100\n"
            "    P(n) = 40/100 = 0.40\n\n"
            "Two warnings:\n"
            "- Check that the total is not 0 BEFORE dividing, and raise ValueError if it "
            "is. Otherwise the ZeroDivisionError will fire later and somewhere else.\n"
            "- Preserve the original key order. Exercise 2 walks this dictionary and the "
            "order changes the result of the sampling.",

            "    total = sum(counts.values())\n"
            "    if total == 0: raise ValueError(...)\n"
            "    return {token: count / total  for each (token, count) in counts.items()}\n\n"
            "A dict comprehension over `counts.items()` preserves the order by itself. Do "
            "not sort the keys.",
        ),
        "sample_next_token": (
            "Think of a roulette wheel. Each character takes up a slice of the circle "
            "proportional to its probability: 'n' with 0.40 takes up 40% of the wheel.\n\n"
            "You spin the ball once (a random number between 0 and 1) and return the "
            "character whose slice it landed in.\n\n"
            "Do not always return the most likely one: that generates repetitive text with "
            "loops.",

            "Stretch the wheel out along a line from 0 to 1:\n\n"
            "    |----'n'----|--'r'--|--' '--|-'s'-|\n"
            "    0          0.40    0.65    0.85   1.0\n\n"
            "Draw `r = rng.random()` (a float in [0,1)) and walk the tokens keeping a "
            "running total. Return the first one whose running total EXCEEDS `r`.\n\n"
            "And when the loop ends, return the last token anyway: because of rounding "
            "error the running total can end up at 0.9999999 and never exceed an `r` very "
            "close to 1. If you skip that, the function returns None and exercise 3 blows "
            "up with an error that tells you nothing.",

            "    r = rng.random()\n"
            "    running = 0.0\n"
            "    last = None\n"
            "    for each (token, p) in probs.items():\n"
            "        running += p\n"
            "        last = token\n"
            "        if r < running:\n"
            "            return token\n"
            "    return last\n\n"
            "Use `<` and not `<=`. With {a:0.5, b:0.5} and r=0.5 exactly, `<` returns 'b', "
            "which is correct: 'a' occupies [0, 0.5) and 'b' occupies [0.5, 1).",
        ),
        "generate_naive": (
            "You already know how to get a distribution (exercise 1) and pick a character "
            "from it (exercise 2). Generating text is doing that in a loop.\n\n"
            "The key: every character you draw becomes part of the input to the next step. "
            "You write a letter, read it as if someone else had given it to you, and decide "
            "the next one.\n\n"
            "This loop is the same one ChatGPT runs. Literally the same one.",

            "On each pass:\n"
            "  1. take the last `len(start)` characters of what you have generated\n"
            "  2. look that context up in the table\n"
            "  3. if it is not there, STOP (the table only knows what it saw in training)\n"
            "  4. if it is, turn the counts into probabilities and sample\n"
            "  5. append the chosen character\n\n"
            "Watch out for `length`: it is the total returned INCLUDING `start`. If start "
            "has 2 characters and 5 are asked for, you generate 3, not 5.",

            "    context_size = len(start)\n"
            "    out = list(start)\n"
            "    repeat max(0, length - len(start)) times:\n"
            "        context = ''.join(out[-context_size:])\n"
            "        counts = table.get(context)\n"
            "        if not counts: break\n"
            "        out.append(sample_next_token(next_token_probs(counts), rng))\n"
            "    return ''.join(out)\n\n"
            "Accumulate into a list and join at the end with `''.join()`. Concatenating "
            "strings in a loop creates a new string on every pass; here it does not matter, "
            "but it is a habit that gets expensive in module 13.",
        ),
    },
    # ------------------------------------------------------------------ module 01
    "01_environment": {
        "measure_matmul_tflops": (
            "You want to know how many operations per second your GPU REALLY does, not the "
            "number on the box. The way to find out is to give it a large, known amount of "
            "work, time it, and divide.\n\n"
            "The work: multiplying two square matrices. You know exactly how many operations "
            "that is, so all you are missing is the time.",

            "FLOPs of multiplying two (size x size) matrices = 2 * size^3.\n"
            "TFLOPS = (FLOPs * number_of_repeats) / seconds / 1e12\n\n"
            "The two things you have to get right:\n\n"
            "1. WARM UP. The first multiplication at a given size is between 10 and 100 "
            "times slower: the GPU is choosing which kernel to use and allocating memory. Do "
            "a few that you do not time.\n\n"
            "2. SYNCHRONIZE. `a @ b` on a GPU does not wait for the result: it queues the "
            "work and returns control. If you time without synchronizing, you measure how "
            "long it takes to enqueue (microseconds) and you get thousands of TFLOPS. Use "
            "`cfg.synchronize()` right before each timestamp.",

            "    a = torch.randn(size, size, device=cfg.device, dtype=dtype)\n"
            "    b = torch.randn(size, size, device=cfg.device, dtype=dtype)\n\n"
            "    repeat `warmup` times: a @ b\n"
            "    cfg.synchronize()\n\n"
            "    t0 = time.perf_counter()\n"
            "    repeat `iters` times: a @ b\n"
            "    cfg.synchronize()\n"
            "    elapsed = time.perf_counter() - t0\n\n"
            "    return (2 * size**3 * iters) / elapsed / 1e12\n\n"
            "The default dtype: `cfg.amp_dtype` if it exists, otherwise `torch.float32`.",
        ),
        "transformer_flops_per_token": (
            "You want to know what it costs to process ONE token, so you can multiply by 500 "
            "million and know whether training takes two hours or two weeks.\n\n"
            "The idea that makes it easy: almost all of a network's cost is matrix "
            "multiplications, and a matrix with P parameters costs roughly 2P operations for "
            "each token that passes through it.",

            "Add up the parameters that take part in multiplications:\n\n"
            "    per layer: 4*d_model^2 (the 4 attention projections)\n"
            "             + n_ffn_matrices*d_model*d_ff (the FFN)\n"
            "    plus the final projection to logits: d_model*vocab_size\n\n"
            "Forward = 2 * those_parameters + 4*n_layers*context_length*d_model\n\n"
            "That second term is attention itself (Q@K^T and softmax@V). It does not come "
            "from parameters: it comes from multiplying tokens against each other, which is "
            "why it grows with the context and not with the model size.\n\n"
            "Total with the backward pass = 3 * forward. The backward costs twice the "
            "forward because it does two multiplications for each one in the forward: one "
            "for the gradient with respect to the input and one with respect to the weights.",

            "    params = n_layers * (4*d_model**2 + n_ffn_matrices*d_model*d_ff)\n"
            "    params += d_model * vocab_size\n"
            "    forward = 2*params + 4*n_layers*context_length*d_model\n"
            "    return int(3*forward if include_backward else forward)\n\n"
            "Two warnings:\n"
            "- The final projection counts even if you use weight tying. Tying the weights "
            "saves memory, not computation: the matmul happens all the same.\n"
            "- Do NOT divide by two for the causal mask. That is the nanoGPT and paper "
            "convention; if you divide, your MFU will not be comparable with anyone's.",
        ),
        "estimate_tokens_per_second": (
            "You have your GPU's power (TFLOPS) and the cost of a token (FLOPs). Dividing "
            "one by the other gives tokens per second.\n\n"
            "The only nuance is the MFU: you never use 100% of the GPU, so you have to "
            "multiply by the fraction you actually get.",

            "    tokens/s = TFLOPS * 1e12 * MFU / FLOPs_per_token\n\n"
            "The 1e12 converts TeraFLOPS to FLOPS. Check that `flops_per_token` is positive "
            "and raise ValueError if it is not: a division by zero here silently produces "
            "`inf` and absurd estimates.",

            "    if flops_per_token <= 0: raise ValueError(...)\n"
            "    return tflops * 1e12 * mfu / flops_per_token\n\n"
            "Realistic MFU values: 0.4-0.5 for well-optimized billion-parameter models, "
            "0.1-0.2 for our 9M model. 320x320 matrices are too small to saturate the tensor "
            "cores.",
        ),
    },
    # ------------------------------------------------------------------ module 02
    "02_autograd": {
        "Value": (
            "A `Value` is a number that also remembers how it was computed. When you write "
            "`c = a * b`, the object `c` stores: its value, who `a` and `b` are, and HOW to "
            "pass backwards whatever gradient reaches it.\n\n"
            "The part that is hard to see: that 'how to pass it back' is stored as a "
            "function that does NOT run yet. It will run later, during the backward pass. "
            "You are building a to-do list while doing the forward pass.",

            "Every operation follows the same mould:\n\n"
            "    def __mul__(self, other):\n"
            "        other = other if isinstance(other, Value) else Value(other)\n"
            "        out = Value(self.data * other.data, (self, other), '*')\n"
            "        def _backward():\n"
            "            self.grad  += other.data * out.grad\n"
            "            other.grad += self.data  * out.grad\n"
            "        out._backward = _backward\n"
            "        return out\n\n"
            "The local derivatives you need:\n"
            "    a+b   -> da += 1*out.grad ; db += 1*out.grad\n"
            "    a*b   -> da += b.data*out.grad ; db += a.data*out.grad\n"
            "    a**n  -> da += n * a.data**(n-1) * out.grad\n"
            "    exp   -> da += out.data * out.grad\n"
            "    log   -> da += (1/a.data) * out.grad\n"
            "    tanh  -> da += (1 - out.data**2) * out.grad\n"
            "    relu  -> da += (out.data > 0) * out.grad",

            "ALWAYS `+=`, NEVER `=`. That is the mistake to avoid. Try it with `y = x + x`: "
            "with `=` the second branch overwrites the first and you get 1; with `+=` you get "
            "2, which is correct because y = 2x.\n\n"
            "The sugar needs no new derivatives:\n"
            "    -a      = a * -1\n"
            "    a - b   = a + (-b)\n"
            "    a / b   = a * b**-1\n\n"
            "And `backward()` is three lines:\n"
            "    self.grad = 1.0\n"
            "    for node in reversed(topological_order(self)):\n"
            "        node._backward()\n\n"
            "The `self.grad = 1.0` is the seed: the derivative of something with respect to "
            "itself. Without it every gradient comes out 0.",
        ),
        "topological_order": (
            "When a node passes its gradient down to its children, it must already have "
            "received EVERYTHING coming from its parents. If it passes it on too early, it "
            "sends an incomplete gradient downwards.\n\n"
            "You need a list where every node appears AFTER all of its children. `root` ends "
            "up last, and `backward()` walks the list in reverse.",

            "It is a post-order DFS: you visit a node's children and only then append the "
            "node to the result.\n\n"
            "MAKE IT ITERATIVE, with an explicit stack. The recursive version is five lines "
            "but it blows up with RecursionError as soon as the graph has a few hundred "
            "nodes, and the MLP in exercise 3 does.\n\n"
            "Use `id(node)` for the visited set, not the node itself. If you overload "
            "operators on a class, relying on its default hash is asking for trouble.",

            "The trick is a flag saying whether you already expanded that node's children:\n\n"
            "    order, visited = [], set()\n"
            "    stack = [(root, False)]\n"
            "    while stack:\n"
            "        node, expanded = stack.pop()\n"
            "        if expanded:\n"
            "            order.append(node); continue\n"
            "        if id(node) in visited: continue\n"
            "        visited.add(id(node))\n"
            "        stack.append((node, True))          # requeue myself for LATER\n"
            "        for child in node._prev:\n"
            "            if id(child) not in visited:\n"
            "                stack.append((child, False))\n\n"
            "Each node goes in twice: the first time to expand its children, the second one "
            "-processed once they are all done- to append itself. That is post-order.\n\n"
            "If `root` comes out FIRST, you have the order reversed. The symptom will be "
            "gradients that are right on simple graphs and wrong as soon as there is a reused "
            "node.",
        ),
        "train_scalar_mlp": (
            "This is the training loop in its most naked form: predict, measure the error, "
            "compute gradients, move the weights a little against the gradient, repeat.\n\n"
            "The one in module 11 will be this same loop with AMP, a scheduler and "
            "checkpointing on top. The core does not change.",

            "At each step, and in this order:\n"
            "    1. preds = [model(x) for x in xs]\n"
            "    2. loss  = mean of (pred - y)**2\n"
            "    3. model.zero_grad()      <- BEFORE the backward\n"
            "    4. loss.backward()\n"
            "    5. p.data -= lr * p.grad  for each p in model.parameters()\n"
            "    6. history.append(loss.data)\n\n"
            "Step 3 is the one everybody forgets. Gradients ACCUMULATE (you made them do "
            "that in exercise 1), so without clearing them step 50 uses the sum of the "
            "gradients from steps 1 to 50. It produces no error: the loss drops a little and "
            "then stalls.",

            "    model = MLP(len(xs[0]), [*hidden, 1], value_cls=value_cls, seed=seed)\n"
            "    history = []\n"
            "    for _ in range(steps):\n"
            "        preds = [model(x) for x in xs]\n"
            "        loss = sum(((p - y)**2 for p, y in zip(preds, ys)),\n"
            "                   value_cls(0.0)) * (1.0/len(ys))\n"
            "        model.zero_grad()\n"
            "        loss.backward()\n"
            "        for p in model.parameters():\n"
            "            p.data -= lr * p.grad\n"
            "        history.append(loss.data)\n"
            "    return history\n\n"
            "Two details:\n"
            "- The `value_cls(0.0)` as the initial value of `sum()`: without it, python "
            "starts accumulating from the integer 0 and you mix types.\n"
            "- `p.data -= ...` and not `p -= ...`. You modify the number inside, you do not "
            "create a new node. If you created nodes, the next step's graph would hang off "
            "the previous one and grow without end. In PyTorch this is `torch.no_grad()`.",
        ),
    },
    # ------------------------------------------------------------------ module 03
    "03_tokenization": {
        "get_stats": (
            "Walk the list two elements at a time and count how many times each pair of "
            "neighbours comes up. It is the first step of BPE: to merge the most frequent "
            "pair, you first have to know which one it is.",

            "`zip(ids, ids[1:])` gives you all the consecutive pairs without handling "
            "indices.\n\n"
            "WATCH OUT: when COUNTING, pairs DO overlap. In [1,1,1] the pair (1,1) comes up "
            "twice: at positions 0-1 and at 1-2. (When MERGING, in exercise 2, they do not "
            "overlap. They are different things.)\n\n"
            "The `counts` parameter is there so you can accumulate into a dictionary that "
            "already exists, which is what `train_bpe` needs to add up several chunks.",

            "    counts = {} if counts is None else counts\n"
            "    for pair in zip(ids, ids[1:]):\n"
            "        counts[pair] = counts.get(pair, 0) + 1\n"
            "    return counts\n\n"
            "Return the dictionary as well as mutating it: that way it works for both uses, "
            "`stats = get_stats(ids)` and `get_stats(chunk, stats)`.\n\n"
            "The default value is `None` and not `{}` on purpose. A `{}` as a default value "
            "is created ONCE when the function is defined and shared across every call: the "
            "classic python mutable-default bug.",
        ),
        "merge": (
            "You walk the list and every time you find the pair you are looking for you "
            "replace it with a single new number. This is what shortens the sequence and "
            "creates the token.",

            "The key is how you advance: on a match you consume TWO positions, otherwise "
            "ONE.\n\n"
            "That is why a `for` does not fit well: it always advances by one. You need a "
            "`while` with an index you control yourself.\n\n"
            "In [1,1,1] merging (1,1) the result is [256, 1], not [256, 256]: having consumed "
            "positions 0 and 1, the 1 at position 2 has no partner left.",

            "    out, i, n = [], 0, len(ids)\n"
            "    while i < n:\n"
            "        if i < n - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:\n"
            "            out.append(new_id)\n"
            "            i += 2\n"
            "        else:\n"
            "            out.append(ids[i])\n"
            "            i += 1\n"
            "    return out\n\n"
            "The `i < n - 1` avoids looking at `ids[i+1]` when you are on the last element. "
            "Without it, IndexError as soon as the list ends right on the first element of "
            "the pair.\n\n"
            "Return a NEW list, do not modify the input.",
        ),
        "train_bpe": (
            "It is the 'aaabdaaabac' example from THEORY.md, in a loop. You repeat: count the "
            "pairs, take the most frequent one, merge it, record the merge. As many times as "
            "you want new tokens.\n\n"
            "You start with the 256 bytes, so the new ids run from 256 upwards.",

            "Structure:\n"
            "  1. split with `regex.findall(pattern, text)` (or leave it whole if pattern is "
            "None)\n"
            "  2. each chunk to bytes: `list(chunk.encode('utf-8'))`\n"
            "  3. `vocab = {i: bytes([i]) for i in range(256)}`\n"
            "  4. repeat `vocab_size - 256` times:\n"
            "       - accumulate `get_stats` over ALL the chunks into one dict\n"
            "       - if the dict is empty -> `break` (no pairs left)\n"
            "       - pick the winning pair\n"
            "       - apply `merge` to each chunk\n"
            "       - `merges[pair] = 256 + i`\n"
            "       - `vocab[256+i] = vocab[pair[0]] + vocab[pair[1]]`  (bytes, not str)\n\n"
            "Counting chunk by chunk stops any merge from joining the end of one word with "
            "the start of the next.",

            "The tie-break has to be EXACTLY this one or your merges will diverge from the "
            "reference's as soon as there is a tie:\n\n"
            "    pair = max(stats, key=lambda p: (stats[p], p))\n\n"
            "Python compares tuples element by element: frequency first and, on a tie, the "
            "pair. Which one wins makes no difference to quality; what matters is that it is "
            "deterministic and that it is the same criterion.\n\n"
            "The `break` when `stats` is empty is NOT optional: if you ask for 4096 merges "
            "over a short text, at some point there are no pairs left and `max()` over an "
            "empty dict raises ValueError.\n\n"
            "And validate `vocab_size >= 256` at the start.",
        ),
        "bpe_encode": (
            "Applying the merges you learned to new text. The detail that makes it "
            "non-trivial: they have to be applied IN THE ORDER THEY WERE LEARNED, not in the "
            "order they appear in this particular text.\n\n"
            "If you apply them in another order, the tokenization is valid but DIFFERENT from "
            "the one the model saw during training, and the model does not understand it.",

            "Since the merge ids are 256, 257, 258... in learning order, 'the one learned "
            "first' is 'the one with the lowest id'.\n\n"
            "    stats = get_stats(ids)\n"
            "    pair = min(stats, key=lambda p: merges.get(p, float('inf')))\n\n"
            "The `float('inf')` is the trick: pairs that are not in `merges` get infinity and "
            "never win the `min`. If the winner turns out not to be in `merges`, there is "
            "nothing mergeable left: stop.",

            "    def _encode_chunk(ids, merges):\n"
            "        while len(ids) >= 2:\n"
            "            stats = get_stats(ids)\n"
            "            pair = min(stats, key=lambda p: merges.get(p, float('inf')))\n"
            "            if pair not in merges:\n"
            "                break\n"
            "            ids = merge(ids, pair, merges[pair])\n"
            "        return ids\n\n"
            "And above that: split with the SAME pattern you trained with, apply "
            "`_encode_chunk` to each chunk, and concatenate all the ids into one list.\n\n"
            "The `len(ids) >= 2` avoids calling `get_stats` on something with no pairs.",
        ),
        "bpe_decode": (
            "The opposite of encoding, and much shorter. Each id has a byte sequence "
            "associated with it in `vocab`. You join them all and decode.",

            "The one thing to get right: JOIN FIRST, DECODE AFTERWARDS.\n\n"
            "An 'n' is one byte but an 'ñ' is two (0xC3 0xB1). BPE does not care: it may have "
            "learned a token ending in 0xC3 and another starting with 0xB1. On their own "
            "neither is valid UTF-8; together they are an 'ñ'.\n\n"
            "So do NOT write `''.join(vocab[i].decode() for i in ids)`.",

            "    raw = b''.join(vocab[i] for i in ids)\n"
            "    return raw.decode('utf-8', errors='replace')\n\n"
            "The `errors='replace'` is the BYTES FALLBACK and it is not optional either. A "
            "half-trained model generates random ids and many of those sequences are not "
            "valid UTF-8. With `replace` you get a replacement character and generation "
            "continues; without it, one exception takes down the whole loop over a stray "
            "byte.",
        ),
    },
    # ------------------------------------------------------------------ module 04
    "04_data": {
        "pack_tokens_uint16": (
            "You are going to store 500 million tokens on disk. With python's default int64 "
            "that is 4 GB; with uint16 it is 1 GB. Since your ids run from 0 to 4095 and "
            "uint16 reaches 65,535, there is room to spare.\n\n"
            "The exercise is not the conversion (one line), it is the VALIDATION.",

            "Numpy does not warn you if a number does not fit: it wraps around silently.\n\n"
            "    np.array([65536], dtype=np.int64).astype(np.uint16)   ->  [0]\n\n"
            "No exception, no warning. The data is corrupted, the model trains worse, and "
            "there is nothing pointing at the cause.\n\n"
            "You have to check TWO things: that `vocab_size` fits in uint16, and that no id "
            "is negative or >= `vocab_size`. And do it BEFORE converting, in a type where "
            "everything fits.",

            "    if vocab_size > 2**16: raise ValueError(...)\n"
            "    array = np.asarray(ids, dtype=np.int64)\n"
            "    if array.size and (array.min() < 0 or array.max() >= vocab_size):\n"
            "        raise ValueError(f'... min={array.min()}, max={array.max()}')\n"
            "    return array.astype(np.uint16)\n\n"
            "The `array.size and ...` is necessary: `.min()` on an empty array raises an "
            "exception that has nothing to do with anything and sends you off track.\n\n"
            "Put the actual values in the error message. 'ids out of range' does not help; "
            "'max=9999' tells you instantly that your tokenizer is wrong.",
        ),
        "train_val_split": (
            "You need text the model does NOT see during training, so you can tell whether it "
            "is learning or just memorizing.\n\n"
            "The only decision in the exercise: the cut is CONTIGUOUS and FROM THE END, not "
            "random.",

            "Why not random: the training windows overlap. The one starting at position 100 "
            "and the one starting at 101 share 511 of their 512 tokens.\n\n"
            "If you split at random, validation would be full of fragments already seen. The "
            "loss would look beautiful and mean nothing: you would be measuring memorization "
            "and calling it generalization.\n\n"
            "By cutting a block off the end, what you set aside are whole stories.",

            "    if not 0.0 < val_fraction < 1.0: raise ValueError(...)\n"
            "    n_val = max(1, int(len(tokens) * val_fraction))\n"
            "    if n_val >= len(tokens): raise ValueError(...)\n"
            "    return tokens[:-n_val], tokens[-n_val:]\n\n"
            "Numpy slicing returns VIEWS, not copies, and that is what you want: with 500M "
            "tokens, a needless copy is 1 GB of RAM thrown away.\n\n"
            "The `max(1, ...)` stops a small corpus from making `int(50*0.005)` come out 0 "
            "and leaving you with no validation set.",
        ),
        "get_batch": (
            "You pick random positions in the corpus. From each one you take a window as the "
            "input, and THE SAME window shifted by one token as the target.\n\n"
            "    x = [5, 8, 2, 9]\n"
            "    y = [8, 2, 9, 1]\n\n"
            "Read it column by column: seeing [5] predict 8, seeing [5,8] predict 2... A "
            "4-token window is FOUR training examples.",

            "    max_start = len(data) - context_length - 1\n"
            "    starts = rng.integers(0, max_start, size=batch_size)\n"
            "    x = np.stack([data[i : i+context_length]     for i in starts])\n"
            "    y = np.stack([data[i+1 : i+1+context_length] for i in starts])\n\n"
            "The `-1` in `max_start` is the exercise's off-by-one: `y` needs one token MORE "
            "than the end of `x`. Without it the last window overflows, and numpy does not "
            "warn (out-of-range slicing simply returns fewer elements); what you see is an "
            "`np.stack` failing on incompatible shapes.\n\n"
            "If `max_start < 1`, the corpus is shorter than the context: ValueError.",

            "After the stack:\n\n"
            "    x_np = ....astype(np.int64)      # MANDATORY\n"
            "    x = torch.from_numpy(x_np)\n"
            "    if device is not None:\n"
            "        device = torch.device(device)\n"
            "        if device.type == 'cuda':\n"
            "            x = x.pin_memory().to(device, non_blocking=True)\n"
            "        else:\n"
            "            x = x.to(device)\n\n"
            "The `.astype(np.int64)` does two things: nn.Embedding requires int64 indices, "
            "and it COPIES the data along the way. Without the copy you would be left "
            "pointing at the disk-mapped file and every model access would be a read.\n\n"
            "`pin_memory` + `non_blocking` only make sense on CUDA: they let the next batch's "
            "copy overlap with the current computation. On MPS the memory is unified and "
            "there is no copy to overlap.",
        ),
    },
    # ------------------------------------------------------------------ module 05
    "05_baselines": {
        "uniform_baseline_loss": (
            "A model that knows nothing spreads probability equally across every word in the "
            "vocabulary: it gives 1/V to each one.\n\n"
            "The loss is -ln(probability it gave to the correct token). If that probability "
            "is always 1/V, the loss is always the same. Compute it.",

            "    -ln(1/V) = ln(V)\n\n"
            "And that is it, one line with `math.log`. Validate that `vocab_size >= 1` and "
            "raise ValueError if not.\n\n"
            "Numbers you will see: ln(65)=4.174 for character-level shakespeare, "
            "ln(4096)=8.317 for the final model.",

            "    if vocab_size < 1: raise ValueError(...)\n"
            "    return math.log(vocab_size)\n\n"
            "Keep this number: it is your cheapest bug detector. In module 11, the loss at "
            "step 0 has to be almost exactly this.\n"
            "  - much higher -> the initialization is too aggressive\n"
            "  - lower       -> information leak (badly placed causal mask)",
        ),
        "bigram_counts": (
            "A V x V matrix where cell [i][j] counts how many times token j came right after "
            "token i.\n\n"
            "With ids=[0,1,0,1,2] the pairs are (0,1),(1,0),(0,1),(1,2), so counts[0][1]=2 "
            "and counts[1][0]=counts[1][2]=1.",

            "You can do it with a loop, but with 500M tokens it would take forever. "
            "Vectorized:\n\n"
            "    tokens = torch.as_tensor(ids, dtype=torch.int64)\n"
            "    counts.index_put_((tokens[:-1], tokens[1:]),\n"
            "                      torch.ones(len(tokens)-1, dtype=torch.int64),\n"
            "                      accumulate=True)\n\n"
            "`tokens[:-1]` are all the 'from's and `tokens[1:]` all the 'to's.",

            "The `accumulate=True` is NOT optional. Without it, `index_put_` ASSIGNS instead "
            "of adding: each repeated pair overwrites the previous one and every count ends "
            "up at 1. Try it with [0,0,0,0,0]: the correct result is counts[0][0]=4.\n\n"
            "And with fewer than 2 tokens there are no pairs at all: return the matrix of "
            "zeros without trying to index anything.",
        ),
        "bigram_nll": (
            "You already have the counts from the training text. Now you measure how well "
            "they predict ANOTHER text (the validation one).\n\n"
            "For each consecutive pair in the sequence being evaluated you take the "
            "probability the model assigns it, apply -ln, and average.",

            "    P(b|a) = (C[a][b] + alpha) / (sum_b' C[a][b'] + alpha*V)\n\n"
            "The `alpha` (Laplace smoothing) is essential: without it, a pair that never "
            "appeared has probability 0, its logarithm is -infinity, and since the loss is a "
            "MEAN, that -inf takes the whole result with it.\n\n"
            "Steps:\n"
            "  1. smoothed = counts.double() + alpha\n"
            "  2. probs = smoothed / smoothed.sum(dim=1, keepdim=True)\n"
            "  3. select probs[tokens[:-1], tokens[1:]]\n"
            "  4. float(-torch.log(...).mean())",

            "Two traps:\n\n"
            "1. The `keepdim=True` in step 2. Without it the sum has shape (V,) instead of "
            "(V,1) and the broadcast divides by COLUMNS instead of by rows. The result looks "
            "plausible and is completely wrong.\n\n"
            "2. The denominator. Adding alpha to a row's V entries grows the total by "
            "alpha*V, not by alpha. If you do the sum AFTER adding alpha (as in step 2), this "
            "takes care of itself.\n\n"
            "Use `.double()` and not `.float()`: with large corpora millions of counts get "
            "added up and float32 loses precision.",
        ),
        "NeuralBigram": (
            "The same model as exercise 2, but learned by gradient descent instead of by "
            "counting.\n\n"
            "The idea: a table with V rows and V columns where row `i` is directly the LOGITS "
            "of the token that follows token `i`. Trained with cross-entropy, it converges to "
            "the normalized counts.",

            "A single submodule, and the name matters because the test copies weights by "
            "name:\n\n"
            "    self.token_embedding = nn.Embedding(vocab_size, vocab_size)\n\n"
            "The forward is reading the table:\n"
            "    logits = self.token_embedding(idx)      # (B, T) -> (B, T, V)\n\n"
            "And if there are targets, the loss:\n"
            "    F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))",

            "The `reshape(-1, V)` is because `F.cross_entropy` expects (N, classes) and (N,), "
            "while you have (B, T, V) and (B, T). Flattening batch and time into a single "
            "dimension is the pattern you will repeat in EVERY model in the course.\n\n"
            "Return `(logits, None)` if there are no targets, not `(logits, 0)`.\n\n"
            "A useful curiosity: `nn.Embedding` and `nn.Linear` are mathematically the same "
            "thing (an Embedding is a Linear with one-hot input), but the Embedding READS the "
            "row it needs instead of multiplying by a matrix full of zeros.",
        ),
        "BengioMLP": (
            "Instead of looking only at the previous token, it looks at the previous "
            "`block_size` ones. It takes their embeddings, GLUES them one after another into "
            "a long vector, and passes that vector through a normal MLP.\n\n"
            "It is the 2003 paper that invented word embeddings.",

            "Three submodules, with these exact names:\n\n"
            "    self.embedding = nn.Embedding(vocab_size, d_embed)\n"
            "    self.hidden    = nn.Linear(block_size * d_embed, n_hidden)\n"
            "    self.output    = nn.Linear(n_hidden, vocab_size)\n\n"
            "And the forward:\n"
            "    emb    = self.embedding(idx)       # (B, block_size, d_embed)\n"
            "    flat   = emb.reshape(B, -1)        # (B, block_size*d_embed)\n"
            "    h      = torch.tanh(self.hidden(flat))\n"
            "    logits = self.output(h)            # (B, V)\n\n"
            "Note: here `targets` is `(B,)`, ONE token per sample, not a sequence.",

            "The `reshape(B, -1)` is CONCATENATION, and that is what matters. If you did "
            "`emb.mean(dim=1)` you would be averaging, and the model would lose the order: "
            "[the, cat, eats] and [eats, cat, the] would look the same to it. There is a test "
            "that checks this by passing the context in reverse.\n\n"
            "And watch where the -1 goes: `reshape(B, -1)`, not `reshape(-1, B)`. The second "
            "compiles and produces garbage.\n\n"
            "Notice the `hidden` layer: its parameters grow LINEARLY with block_size. That is "
            "exactly the limitation attention comes to solve in module 06.",
        ),
    },
    # ------------------------------------------------------------------ module 06
    "06_attention": {
        "causal_mask": (
            "During training we give the model the whole sentence at once and ask it to "
            "predict each token from the previous ones. With nothing stopping it, position 3 "
            "could look at position 4, which is literally the answer.\n\n"
            "You need a matrix saying, for each pair (i, j), whether token i is allowed to "
            "look at token j.",

            "The course convention: True = it CAN look. It is the same one "
            "`F.scaled_dot_product_attention` uses.\n\n"
            "For seq_len=4:\n\n"
            "    [[ T, F, F, F],\n"
            "     [ T, T, F, F],\n"
            "     [ T, T, T, F],\n"
            "     [ T, T, T, T]]\n\n"
            "It is a LOWER triangular matrix, with the diagonal included (a token can look at "
            "itself). PyTorch has a function for this.",

            "    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()\n\n"
            "`tril` = triangular lower. Its `diagonal` argument is 0 by default, which "
            "INCLUDES the diagonal: that is what you want.\n\n"
            "If it comes out inverted you used `triu`. If the diagonal comes out False, you "
            "passed `diagonal=-1`.\n\n"
            "A warning for later: PyTorch's `nn.MultiheadAttention` uses the OPPOSITE "
            "convention (True = forbidden). That is why the test comparing against it passes "
            "`~mask`. It is a real inconsistency in the library.",
        ),
        "single_head_attention": (
            "It is the THEORY.md example done with tensors. Each token asks (query), everyone "
            "answers (keys), similarity is measured with dot products, turned into weights "
            "with softmax, and the contents (values) are mixed.\n\n"
            "    output = softmax( Q K^T / sqrt(d_k) + mask ) V",

            "Four steps:\n\n"
            "  1. scores = q @ k.transpose(-2, -1)\n"
            "     (B,T,d_k) @ (B,d_k,S) -> (B,T,S). Cell [b,i,j] is how interested i is in "
            "token j.\n"
            "  2. divide by sqrt(d_k), with d_k = q.shape[-1]\n"
            "  3. scores.masked_fill(~mask, float('-inf'))  if there is a mask\n"
            "  4. weights = F.softmax(scores, dim=-1) ; output = weights @ v\n\n"
            "Return `(output, weights)`: the weights are needed for the demo's heatmap.",

            "Three traps, and all three are silent (no error, just bad results):\n\n"
            "1. Use `transpose(-2, -1)`, with NEGATIVE indices. That way it works the same "
            "with (B,T,d) as with (B,heads,T,d). With `transpose(1,2)` this exercise passes "
            "and the next one breaks.\n\n"
            "2. `dim=-1` in the softmax. You normalize over WHO IS BEING LOOKED AT, so each "
            "row sums to 1. With `dim=-2` you would normalize over who is looking, which "
            "means nothing, and the shapes are identical so no error fires.\n\n"
            "3. The mask goes BEFORE the softmax. If you zeroed the weights afterwards, the "
            "rows would no longer sum to 1 and you would be scaling the output by an "
            "arbitrary factor.",
        ),
        "MultiHeadAttention": (
            "Several attentions in parallel, each with its own projections, so they can "
            "specialize in different relationships.\n\n"
            "And it costs no more: with d_model=320 and 8 heads, each one works in 40 "
            "dimensions. Instead of one attention over 320 you do eight over 40.",

            "The trick: do NOT make 8 separate projections. Make one d_model -> d_model and "
            "split the result.\n\n"
            "    split:  x.view(B, T, n_heads, head_dim).transpose(1, 2)\n"
            "            (B, T, d_model) -> (B, n_heads, T, head_dim)\n\n"
            "    merge:  x.transpose(1, 2).contiguous().view(B, T, d_model)\n\n"
            "With q, k, v already split, attention is EXACTLY the same formula as exercise 2 "
            "(which is why the transpose there had to use negative indices).\n\n"
            "Submodules: q_proj, k_proj, v_proj, out_proj (all Linear d_model->d_model), "
            "attn_dropout and resid_dropout.",

            "Four details that break things if you do not watch them:\n\n"
            "1. The ORDER of the view: `view(B, T, n_heads, head_dim)` and THEN transpose. If "
            "you do `view(B, n_heads, T, head_dim)` directly you are mixing positions with "
            "heads. Right shape, wrong data, zero errors.\n\n"
            "2. `.contiguous()` before the view when merging. `transpose` does not move data, "
            "it only changes the strides, and `view` demands contiguous memory.\n\n"
            "3. RoPE (if cos/sin are not None) goes AFTER splitting into heads, because the "
            "rotation depends on head_dim. And only to q and k, never to v.\n\n"
            "4. If `self.use_sdpa`, use `F.scaled_dot_product_attention(q, k, v, "
            "attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)`. That "
            "`if self.training` matters: SDPA does not check the mode on its own and would "
            "apply dropout at evaluation time too.",
        ),
    },
    # ------------------------------------------------------------------ module 07
    "07_normalization": {
        "layer_norm": (
            "The numbers flowing through a deep network tend to grow or shrink layer after "
            "layer, until they explode or vanish. The fix is brutal in its simplicity: after "
            "every block, put them back on a known scale.\n\n"
            "LayerNorm takes each token's vector, subtracts its mean and divides by its "
            "standard deviation. Out comes mean 0 and variance 1, wherever it came from.",

            "    y = (x - mean) / sqrt(variance + eps) * weight + bias\n\n"
            "Over the LAST dimension (each token's features), with `dim=-1` and "
            "`keepdim=True`. Each token is normalized on its own, without looking at the "
            "batch: that is what distinguishes it from BatchNorm.\n\n"
            "The eps goes INSIDE the square root: sqrt(var + eps), not sqrt(var) + eps.\n\n"
            "If `weight` or `bias` are None, do not apply them.",

            "THE TRAP: `torch.var` divides by (n-1) by default (sample variance). LayerNorm "
            "uses the POPULATION one, which divides by n.\n\n"
            "    var = x.var(dim=-1, keepdim=True, unbiased=False)\n\n"
            "Without the `unbiased=False`, your result will look a lot like `F.layer_norm` "
            "but will not match it. With d=320 the difference is 0.3% and you might not see "
            "it; with d=4 it is 33%. The test compares against both versions and tells you "
            "which one you resemble.\n\n"
            "And the `keepdim=True` is not optional either: without it, the mean of (4,8,32) "
            "comes out (4,8) instead of (4,8,1) and the subtraction broadcasts wrong.",
        ),
        "RMSNorm": (
            "The same as LayerNorm but WITHOUT subtracting the mean and WITHOUT a bias. Just "
            "rescaling.\n\n"
            "Zhang and Sennrich (2019) observed that almost all of LayerNorm's benefit comes "
            "from rescaling, not from recentering. Dropping it saves a pass over the data and "
            "an intermediate tensor. Llama, Mistral and our model use it.",

            "    y = x / sqrt( mean(x^2) + eps ) * weight\n\n"
            "With x = [2, 8, 4, 6]:\n"
            "    RMS = sqrt((4+64+16+36)/4) = sqrt(30) = 5.477\n"
            "    y   = [0.365, 1.461, 0.730, 1.096]\n\n"
            "The only parameter is `weight`, of shape (dim,), initialized to ONES. At startup "
            "the layer has to be pure normalization: if it started random, the step-0 loss "
            "would not match ln(V).\n\n"
            "`torch.rsqrt(z)` computes 1/sqrt(z) in one go and is faster than dividing.",

            "    def _norm(self, x):\n"
            "        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)\n\n"
            "    def forward(self, x):\n"
            "        return self._norm(x.float()).type_as(x) * self.weight\n\n"
            "The `.float()` is NOT paranoia. With activations in fp16, squaring overflows "
            "sooner than you would think: 300^2 = 90,000 and fp16 runs out at 65,504. You "
            "would get inf, then the mean would be inf, and rsqrt(inf) = 0: the layer would "
            "return zeros. There is a test that reproduces exactly that case.\n\n"
            "A surprising detail: even if you call `.type_as(x)`, the output ends up in fp32, "
            "because `self.weight` is fp32 and PyTorch promotes. That is correct and it is "
            "what Llama does.",
        ),
        "prenorm_residual": (
            "It is ONE LINE and it is the most important exercise in the module.\n\n"
            "Instead of each block REPLACING the representation, it is asked to MODIFY it: "
            "the output is the input plus a correction. That way there is always a direct "
            "path from the input to the output.",

            "Two options, and only the parentheses move:\n\n"
            "    post-norm (2017 paper):        norm(x + fn(x))\n"
            "    pre-norm  (everything modern): x + fn(norm(x))\n\n"
            "In pre-norm the normalization is INSIDE the branch, so the path x -> x is left "
            "clear. Differentiating gives `1 + something`, and that 1 arrives intact at the "
            "layers below however many there are.\n\n"
            "In post-norm the norm sits ON TOP of the sum, so the gradient goes through it at "
            "every layer and keeps getting rescaled.",

            "    return x + fn(norm(x))\n\n"
            "That is it. If you end up with `norm(x + fn(x))` you have written post-norm and "
            "there is a test that detects it.\n\n"
            "A consequence for module 10: since the residual stream is never normalized along "
            "the way, it reaches the output at a scale that grows with depth. That is why "
            "pre-norm models ALWAYS carry a final normalization before the logits layer. It "
            "will be called `norm_f`.",
        ),
    },
    # ------------------------------------------------------------------ module 08
    "08_mlp_and_activations": {
        "gelu": (
            "Attention is a weighted average, that is, a LINEAR operation. And two linear "
            "operations in a row are one: W2·(W1·x) = (W2·W1)·x. Without something non-linear "
            "between layers, a hundred layers are equivalent to one.\n\n"
            "GELU is that piece. It multiplies x by the probability that a standard normal "
            "comes out below x: instead of cutting the negatives dead like ReLU, it "
            "attenuates them gradually.",

            "The tanh approximation, which is the one asked for:\n\n"
            "    GELU(x) ~= 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))\n\n"
            "Written left to right, without regrouping. The constants are the paper's: "
            "sqrt(2/pi) ~= 0.7978 and 0.044715.\n\n"
            "It has to match `F.gelu(x, approximate='tanh')`, NOT plain `F.gelu(x)`: they are "
            "different functions and the test compares against the first one.",

            "    return 0.5 * x * (1.0 + torch.tanh(\n"
            "        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))\n"
            "    ))\n\n"
            "One line, no loops and no branches.\n\n"
            "The important part of the exercise is not the formula, it is the DERIVATIVE. "
            "With ReLU it is exactly zero throughout the negative region, so a neuron that "
            "drifts there stops receiving gradient forever. With GELU it is small but not "
            "zero, and it can come back. The demo tabulates it.",
        ),
        "swiglu_hidden_dim": (
            "Pure arithmetic, but with a decision behind it. SwiGLU has THREE matrices where "
            "the classic FFN has two, so with the same hidden size it would cost 50% more.\n\n"
            "To spend the same number of parameters the hidden size is cut to two thirds. "
            "This exercise produces the 896 in the final config.",

            "Three steps:\n\n"
            "  1. hidden = int(2 * (4 * d_model) / 3)\n"
            "  2. if ffn_dim_multiplier is not None: hidden = int(ffn_dim_multiplier * hidden)\n"
            "  3. round UP to the next multiple of `multiple_of`\n\n"
            "Checks:\n"
            "  d_model=320 -> int(2*1280/3) = 853 -> ceil_64 = 896   (final config)\n"
            "  d_model=128 -> int(2*512/3)  = 341 -> ceil_64 = 384   (tiny_char)",

            "Rounding up with integers, without math.ceil:\n\n"
            "    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)\n\n"
            "Adding (multiple_of - 1) before the integer division forces rounding up, and if "
            "it was already an exact multiple it leaves it alone.\n\n"
            "Why round at all: aligned dimensions let the tensor cores take their fast paths. "
            "A matrix with 853 columns is slower than one with 896 despite having fewer "
            "parameters.",
        ),
        "SwiGLU": (
            "Two projections in parallel from the same input. One of them, after going "
            "through an activation, acts as a GATE: it multiplies the other element by "
            "element and decides how much signal gets through each dimension.\n\n"
            "The difference from a normal activation is that this filtering DEPENDS ON THE "
            "INPUT: it decides, for each dimension and each token, how much gets through.",

            "    SwiGLU(x) = down( Swish(gate(x)) * up(x) )\n\n"
            "The `*` is ELEMENTWISE multiplication, not matrix multiplication: both branches "
            "come out with shape (B, T, d_ff) and are multiplied point by point.\n\n"
            "`Swish(z) = z * sigmoid(z)`, which in PyTorch is `F.silu(z)`.\n\n"
            "Submodules: gate_proj and up_proj (Linear d_model -> d_ff), down_proj (Linear "
            "d_ff -> d_model) and dropout. All without bias by default.",

            "    def forward(self, x):\n"
            "        return self.dropout(self.down_proj(\n"
            "            F.silu(self.gate_proj(x)) * self.up_proj(x)\n"
            "        ))\n\n"
            "The activation goes on `gate_proj`, NOT on `up_proj`. With the assignment "
            "swapped the module works just as well but does not match the reference when "
            "weights are copied, and the test fails with a difference that is hard to "
            "interpret. There is a dedicated test that points this out.\n\n"
            "And keep in mind that the FFN processes each token SEPARATELY: it does not mix "
            "positions, that is attention's job. No mask is needed here.",
        ),
    },
    # ------------------------------------------------------------------ module 09
    "09_position": {
        "sinusoidal_embeddings": (
            "Attention is a weighted sum, and a sum has no order: without positional "
            "information, 'the dog bites the man' and 'the man bites the dog' produce exactly "
            "the same output.\n\n"
            "This table is ADDED to the token embeddings to say which position each one is "
            "in. The idea is that of a binary counter: each pair of dimensions oscillates at "
            "a different rate, and the combination identifies the position.",

            "    PE[pos, 2i]   = sin( pos / base^(2i/d) )\n"
            "    PE[pos, 2i+1] = cos( pos / base^(2i/d) )\n\n"
            "EVEN dimensions with sine, ODD with cosine, the same frequency for each pair.\n\n"
            "Without loops:\n"
            "  - `position = torch.arange(seq_len).unsqueeze(1)`  -> (T, 1)\n"
            "  - the frequencies, one per pair -> (d/2,)\n"
            "  - `position * div_term` broadcasts to (T, d/2): every angle at once\n"
            "  - `table[:, 0::2] = sin(...)` and `table[:, 1::2] = cos(...)` to interleave",

            "The trick worth knowing, for the frequencies:\n\n"
            "    div_term = torch.exp(\n"
            "        torch.arange(0, d_model, 2, dtype=torch.float32)\n"
            "        * (-math.log(base) / d_model)\n"
            "    )\n\n"
            "It is mathematically the same as `base ** (-2i/d)` but far more stable: raising "
            "10000 to a large negative power loses floating-point precision, and going "
            "through logarithms does not.\n\n"
            "A general rule that will serve you elsewhere: if you see a power with a large "
            "exponent, `exp(log(...))` is usually better.",
        ),
        "rope_frequencies": (
            "RoPE does not ADD anything to the vector: it ROTATES it. Each pair of dimensions "
            "turns by an angle proportional to the position, and each pair has its own "
            "rotation speed.\n\n"
            "This function precomputes, once and for all positions, the cosine and sine of "
            "those angles.",

            "The angle of pair `i` at position `pos` is:\n\n"
            "    angle = pos * theta^(-2i/head_dim)\n\n"
            "The steps:\n"
            "  1. validate that head_dim is EVEN (RoPE rotates pairs) -> ValueError if not\n"
            "  2. inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))\n"
            "     -> head_dim/2 values\n"
            "  3. angles = torch.outer(positions, inv_freq)   -> (max_seq_len, head_dim/2)\n"
            "  4. DUPLICATE: angles = torch.cat([angles, angles], dim=-1)\n"
            "  5. return (angles.cos(), angles.sin())",

            "Step 4 is the confusing one, so here is why.\n\n"
            "There are two conventions for pairing dimensions when rotating:\n"
            "  - the original paper pairs consecutive ones: (x0,x1), (x2,x3)...\n"
            "  - Llama and HuggingFace pair by HALVES: (x0, x_{d/2}), (x1, x_{d/2+1})...\n\n"
            "We use the halves one. With it, dimension `i` and dimension `i + head_dim/2` "
            "form a pair and need THE SAME angle. That is why each frequency appears "
            "duplicated and the tables have head_dim columns instead of head_dim/2.\n\n"
            "The two conventions are equivalent up to a permutation of the dimensions, which "
            "the network learns without noticing. The halves one won because it makes "
            "apply_rope a one-liner with no reordering.",
        ),
        "apply_rope": (
            "With the tables already computed this is ONE LINE. But it is worth seeing where "
            "it comes from before writing it.\n\n"
            "Rotating a vector (x1, x2) by an angle t is the usual rotation matrix:\n\n"
            "    x1' = x1*cos(t) - x2*sin(t)\n"
            "    x2' = x2*cos(t) + x1*sin(t)",

            "Those two lines can be written all at once as:\n\n"
            "    x' = x * cos + rotate_half(x) * sin\n\n"
            "where `rotate_half([a, b]) = [-b, a]`, splitting the last dimension in half. "
            "Check it:\n\n"
            "    component 1:  x1*cos + (-x2)*sin = x1*cos - x2*sin   OK\n"
            "    component 2:  x2*cos + ( x1)*sin = x2*cos + x1*sin   OK\n\n"
            "And rotate_half without loops:\n"
            "    half = x.shape[-1] // 2\n"
            "    x1, x2 = x[..., :half], x[..., half:]\n"
            "    return torch.cat([-x2, x1], dim=-1)",

            "    seq_len = x.shape[-2]\n"
            "    cos = cos[:seq_len].to(dtype=x.dtype, device=x.device)\n"
            "    sin = sin[:seq_len].to(dtype=x.dtype, device=x.device)\n"
            "    return x * cos + rotate_half(x) * sin\n\n"
            "Two details that break if you skip them:\n\n"
            "1. SLICING to seq_len. The tables are precomputed up to max_seq_len (512 in the "
            "final model) and your sequence is almost never that long. Without slicing, the "
            "broadcast fails or -worse- succeeds by accident with the wrong shapes.\n\n"
            "2. Converting dtype and device. Under AMP the tables are in fp32 and x arrives "
            "in fp16; mixing them makes torch promote and you end up computing at a precision "
            "you did not want.\n\n"
            "No unsqueeze is needed: x is (B, heads, T, head_dim) and cos is (T, head_dim); "
            "the broadcast aligns from the right and takes care of the rest.",
        ),
    },
    # ------------------------------------------------------------------ module 10
    "10_the_full_gpt": {
        "expected_param_count": (
            "Computing how many parameters the model will have BEFORE building it. It is "
            "useful for designing (you change d_model and see instantly whether it fits on "
            "the GPU) and for verifying that the model you assembled is the one you thought "
            "you assembled.\n\n"
            "Do it on paper first. Take the breakdown from THEORY.md and write the formula; "
            "only then translate it into code. If you go straight to code you will end up "
            "trying numbers until they add up, and that teaches nothing.",

            "The terms:\n\n"
            "    embeddings  = vocab_size * d_model\n"
            "    (+ context_length * d_model only if pos == 'learned')\n\n"
            "    per layer:\n"
            "      attention = 4 * d_model^2                (Wq, Wk, Wv, Wo)\n"
            "      ffn       = 3 * d_model * d_ff           (SwiGLU)\n"
            "      norms     = 2 * d_model                  (two RMSNorms)\n\n"
            "    final norm = d_model\n"
            "    lm_head    = 0 if tie_embeddings\n\n"
            "Check: 1,310,720 + 6*(409,600 + 860,160 + 640) + 320 = 8,933,440",

            "Two things that get forgotten and make it not add up:\n\n"
            "1. RoPE contributes ZERO parameters. Its tables come from a formula and are "
            "stored as buffers. If your count includes anything from RoPE, there is one term "
            "too many.\n\n"
            "2. RMSNorm has d_model parameters, not 2*d_model: scale only, no bias. With 6 "
            "layers x 2 norms + 1 final that is 13 x 320 = 4,160. It is small, but if you "
            "forget it the total does not add up.\n\n"
            "And remember the branches: LayerNorm instead of RMSNorm, classic MLP (2 "
            "matrices) instead of SwiGLU (3), and the biases if cfg.bias is True. The tests "
            "check all those combinations.",
        ),
        "count_parameters": (
            "The opposite of exercise 1: instead of computing with a formula, walk the actual "
            "model and add things up. If the two numbers match, your formula and your model "
            "say the same thing.\n\n"
            "It also breaks the count down by component, which is what makes the result "
            "useful.",

            "Walk `model.named_parameters()` and classify by what appears in the name. The "
            "names look like `blocks.3.attn.q_proj.weight`.\n\n"
            "    'token_embedding' / 'pos_embedding'  -> embeddings\n"
            "    'attn.'                              -> attention\n"
            "    gate_proj / up_proj / down_proj      -> ffn\n"
            "    'norm'                               -> norms\n"
            "    'lm_head'                            -> lm_head\n\n"
            "Print `[n for n, _ in model.named_parameters()]` once: it is well worth it for "
            "seeing how the model is wired up inside.",

            "THE ORDER OF THE CHECKS MATTERS. The string 'norm' also appears in `attn_norm` "
            "and `ffn_norm`, so if you check 'norm' before 'attn.', the attention's "
            "normalization ends up in the wrong category. Go from more specific to more "
            "general.\n\n"
            "About weight tying: `parameters()` and `named_parameters()` DEDUPLICATE by "
            "default (remove_duplicate=True), so the total comes out right without doing "
            "anything. Even so, keep a `set` of `id(param)`: it makes explicit that you know "
            "there are shared weights and it protects the breakdown. With "
            "remove_duplicate=False you would count 1,310,720 parameters too many.\n\n"
            "Do not forget `total` (the sum of the six categories) and `non_embedding` "
            "(total - embeddings), which is what module 12's scaling laws use.",
        ),
        "TransformerBlock": (
            "A block is two sub-blocks, each with its own normalization and residual:\n\n"
            "    x = x + attention(norm1(x))\n"
            "    x = x + ffn(norm2(x))\n\n"
            "Attention MOVES information between tokens; the FFN PROCESSES it token by token. "
            "They alternate. That is the whole block.",

            "Submodules, with these names (the test copies weights by name and exercise 2 "
            "classifies by name):\n\n"
            "    self.attn_norm = make_norm(cfg)\n"
            "    self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads,\n"
            "                                   dropout=cfg.dropout, bias=cfg.bias)\n"
            "    self.ffn_norm = make_norm(cfg)\n"
            "    self.ffn = make_ffn(cfg)\n\n"
            "`make_norm` and `make_ffn` are already written in `llmfs.reference`: they pick "
            "RMSNorm or LayerNorm, SwiGLU or MLP, according to the config.",

            "    def forward(self, x, cos=None, sin=None, mask=None):\n"
            "        x = x + self.attn(self.attn_norm(x), mask=mask, cos=cos, sin=sin)\n"
            "        x = x + self.ffn(self.ffn_norm(x))\n"
            "        return x\n\n"
            "TWO independent residuals, not one around the whole block. There is a test that "
            "zeroes the output weights of both branches and checks that the output is EXACTLY "
            "the input: with no residuals, it would return zero.\n\n"
            "The FFN does not receive cos, sin or mask: it does not look at other tokens and "
            "does not need them.",
        ),
        "GPT": (
            "Assembling everything:\n\n"
            "    ids -> embeddings -> [block] x n_layers -> final norm -> logits\n\n"
            "With RoPE there is no positional embedding to add at the start: position is "
            "injected inside attention. That is why the first layer is only the token "
            "table.\n\n"
            "When you finish, `sum(p.numel() for p in model.parameters())` has to give "
            "exactly 8,933,440.",

            "The three things you have to get right:\n\n"
            "1. TYING:  `self.lm_head.weight = self.token_embedding.weight`\n"
            "   That does NOT copy: it makes them point at the same tensor. The test checks "
            "`is`, not `==`.\n\n"
            "2. RoPE AS A BUFFER:\n"
            "     cos, sin = rope_frequencies(cfg.head_dim, cfg.context_length, cfg.rope_theta)\n"
            "     self.register_buffer('rope_cos', cos, persistent=False)\n"
            "   A buffer travels with the model (it moves with .to(device)) but is not a "
            "parameter. `persistent=False` also keeps it out of the checkpoint: it is "
            "recomputed on construction.\n\n"
            "3. INIT IN TWO PASSES: first `self.apply(self._init_weights)` with std=0.02 on "
            "every Linear and Embedding, and THEN override the projections that write into "
            "the residual (out_proj, down_proj) with std = 0.02/sqrt(2*n_layers).",

            "Why the scaled init: every block ADDS its contribution to the residual stream. "
            "With 6 layers and 2 sub-blocks each that is 12 contributions, so the output's "
            "variance would be 12 times the input's. Dividing the standard deviation by "
            "sqrt(2*n_layers) compensates for it. The 2 is because each block writes twice.\n\n"
            "And the order matters: the general apply first, THEN the override. The other way "
            "round, the apply would overwrite the scaled init.\n\n"
            "In the forward, the mask is computed ONCE before the block loop and passed to "
            "them all. Inside each block it would work, but it would be 6 identical tensors "
            "per forward.\n\n"
            "And validate `T <= cfg.context_length` at the start, with a ValueError that "
            "states both numbers.",
        ),
    },
    # ------------------------------------------------------------------ module 11
    "11_training_loop": {
        "AdamWScratch": (
            "Bare gradient descent (`p -= lr * g`) has a problem: a single lr for every "
            "parameter. The ones that constantly receive large gradients take absurd jumps, "
            "and the ones that almost never get signal do not move.\n\n"
            "Adam fixes it with two running averages: one of the gradient (momentum, smooths "
            "the noise) and one of the SQUARED gradient (scaling, gives each parameter its "
            "own effective lr).",

            "    m = beta1*m + (1-beta1)*g\n"
            "    v = beta2*v + (1-beta2)*g^2\n\n"
            "    m_hat = m / (1 - beta1^t)      <- bias correction\n"
            "    v_hat = v / (1 - beta2^t)\n\n"
            "    p -= lr * m_hat / (sqrt(v_hat) + eps)\n"
            "    p -= lr * weight_decay * p     <- SEPARATELY\n\n"
            "The structure of a PyTorch optimizer:\n\n"
            "    for group in self.param_groups:      # the groups from exercise 4\n"
            "        for p in group['params']:\n"
            "            if p.grad is None: continue\n"
            "            state = self.state[p]        # per-parameter dict, persists\n"
            "            if len(state) == 0:          # first time\n"
            "                state['step'] = 0\n"
            "                state['exp_avg'] = torch.zeros_like(p)\n"
            "                state['exp_avg_sq'] = torch.zeros_like(p)\n"
            "            state['step'] += 1\n",

            "The three mistakes the tests catch:\n\n"
            "1. STARTING t AT 0. With t=0, 1-beta^0 = 0 and you divide by zero. Increment "
            "`state['step']` BEFORE using it.\n\n"
            "2. FORGETTING THE BIAS CORRECTION. There is a test that takes a single step with "
            "gradient 1 and lr=0.1, and requires the parameter to move by 0.1. Without the "
            "correction, with beta2=0.95 it would move 0.447: 4.5 times more.\n\n"
            "3. ADDING THE WEIGHT DECAY TO THE GRADIENT. That is Adam+L2, not AdamW. Do it "
            "directly on the parameter:\n\n"
            "       p.mul_(1 - lr * wd)      # BEFORE Adam's update\n\n"
            "   The test tells them apart by setting the gradient to ZERO: with decoupled "
            "decay the parameter keeps shrinking, with L2 it does not.\n\n"
            "And do not forget the `@torch.no_grad()` on `step`.",
        ),
        "lr_at_step": (
            "The learning rate is not constant during training. It rises slowly at the start "
            "(warmup) and falls at the end (cosine).\n\n"
            "The warmup exists because in the first steps Adam's moments are empty and the "
            "freshly initialized weights give large gradients: starting at full lr usually "
            "produces a spike the model sometimes never recovers from.",

            "Three segments:\n\n"
            "  1. step < warmup_steps  ->  lr * (step + 1) / warmup_steps\n"
            "  2. step >= max_steps    ->  lr * min_lr_ratio\n"
            "  3. in between           ->  cosine between the two\n\n"
            "The cosine:\n"
            "    progress = (step - warmup_steps) / (max_steps - warmup_steps)\n"
            "    coef     = 0.5 * (1 + cos(pi * progress))\n"
            "    return min_lr + (lr - min_lr) * coef\n\n"
            "Check the endpoints: progress=0 gives cos(0)=1 and coef=1, i.e. `lr`. "
            "progress=1 gives cos(pi)=-1 and coef=0, i.e. `min_lr`.",

            "Details that break if you skip them:\n\n"
            "- The `+1` in the warmup: without it, step 0 would have an lr of exactly zero "
            "and learn nothing. With a warmup of 500, that is 500 wasted steps.\n\n"
            "- The order of the guards: `step >= max_steps` has to come BEFORE the cosine. "
            "Otherwise progress would go past 1 and the cosine would start RISING again.\n\n"
            "- Clamp progress with `min(1.0, max(0.0, progress))` and use `max(1, ...)` in "
            "the denominator: it protects you if max_steps <= warmup_steps.\n\n"
            "- `schedule` also accepts 'linear' and 'constant', one line each.",
        ),
        "clip_grad_norm": (
            "Occasionally a batch produces enormous gradients (an odd sequence, a very rare "
            "token). With no protection, that single batch can take a jump that destroys "
            "hours of training.\n\n"
            "The fix: if the norm of ALL the gradients together exceeds a threshold, scale "
            "them all by the same factor.",

            "    grads = [p.grad for p in parameters if p.grad is not None]\n"
            "    norm = sqrt( sum of (g**2).sum() over all of them )\n"
            "    if norm > max_norm:\n"
            "        factor = max_norm / (norm + 1e-6)\n"
            "        multiply ALL the gradients by factor\n"
            "    return norm      <- the one from BEFORE clipping\n\n"
            "The `1e-6` avoids dividing by zero if the norm is tiny.\n\n"
            "Returning the norm from BEFORE is what `torch.nn.utils.clip_grad_norm_` does and "
            "it is the useful one: if you log it and it rises steadily, training is "
            "destabilizing.",

            "THE NORM IS GLOBAL, not one per tensor. It is the only conceptual point in the "
            "exercise.\n\n"
            "If you clipped each tensor separately, each would be scaled by a different "
            "factor and the DIRECTION of the combined gradient would change. And that is "
            "exactly what you do not want: the gradient points where you have to go, and you "
            "are only limiting how far you step.\n\n"
            "There is a test that computes the cosine between the gradient vector before and "
            "after clipping, and requires it to be > 0.9999.\n\n"
            "Use `g.detach()` when computing the norm: gradients do not require gradient, but "
            "it is the correct habit.",
        ),
        "build_param_groups": (
            "Weight decay pushes weights towards zero. That makes sense on a projection "
            "matrix, but NOT on an RMSNorm's scale: that parameter starts at 1 and its job is "
            "to rescale; pushing it towards zero is pushing the layer's output towards zero, "
            "which is the opposite of what is needed.\n\n"
            "So the parameters have to be split into two groups.",

            "The rule is surprisingly simple:\n\n"
            "    parameters with 2 dimensions or more  ->  WITH decay   (the matrices)\n"
            "    parameters with 1 dimension           ->  WITHOUT decay (biases and scales)\n\n"
            "`param.dim()` gives you the number of dimensions.\n\n"
            "The format PyTorch expects is a list of dicts:\n\n"
            "    [{'params': [...], 'weight_decay': wd},\n"
            "     {'params': [...], 'weight_decay': 0.0}]\n\n"
            "Any extra key overrides the optimizer's default value for that group only.",

            "    decay, no_decay = [], []\n"
            "    for param in model.parameters():\n"
            "        if not param.requires_grad:\n"
            "            continue\n"
            "        (decay if param.dim() >= 2 else no_decay).append(param)\n\n"
            "    return [\n"
            "        {'params': decay, 'weight_decay': weight_decay},\n"
            "        {'params': no_decay, 'weight_decay': 0.0},\n"
            "    ]\n\n"
            "The order matters: the group WITH decay first (there are tests that depend on "
            "it).\n\n"
            "Skip frozen parameters: they are not going to be updated and putting them in the "
            "optimizer only wastes memory.\n\n"
            "With the final model you get 43 tensors with decay (8,929,280 params) and 13 "
            "without it (4,160, the RMSNorm scales).",
        ),
    },
    # ------------------------------------------------------------------ module 12
    "12_efficiency_and_scaling": {
        "model_flops_per_token": (
            "The same calculation as module 01, but returning the BREAKDOWN instead of a "
            "single number.\n\n"
            "It is worth separating because the two terms scale differently: the matmul one "
            "with the model size, the attention one with the CONTEXT. Knowing which weighs "
            "more tells you instantly whether lengthening the context is going to be "
            "expensive.",

            "    params_matmul = n_layers * (4*d^2 + n_ffn*d*d_ff) + d*vocab_size\n"
            "                    (n_ffn = 3 with SwiGLU, 2 with a classic FFN)\n\n"
            "    matmul    = 2 * params_matmul\n"
            "    attention = 4 * n_layers * context_length * d_model\n\n"
            "If include_backward, multiply BOTH by 3.\n\n"
            "Return a dict with `matmul`, `attention`, `total` (the sum) and "
            "`params_matmul`.",

            "The final projection `d * vocab_size` counts EVEN with weight tying. Tying the "
            "weights saves memory, not computation: the matmul happens all the same. There is "
            "a test that checks this.\n\n"
            "And the `* 3` applies to BOTH terms, not just the matmul one: attention's "
            "backward also costs twice its forward.\n\n"
            "Check: with the final config the total has to give 65,372,160, the same number "
            "as in module 01.",
        ),
        "compute_mfu": (
            "What fraction of your GPU's theoretical power you are really using.\n\n"
            "It is THE metric for knowing whether your training run is well optimized, and "
            "its beauty is that it depends on neither the model nor the hardware: you can "
            "compare different configurations.",

            "    MFU = tokens_per_second * flops_per_token / (peak_tflops * 1e12)\n\n"
            "The 1e12 converts TeraFLOPS into FLOPS. It is one line.\n\n"
            "Validate that `peak_tflops` is positive and raise ValueError if not: a division "
            "by zero here silently gives `inf`.",

            "    if peak_tflops <= 0:\n"
            "        raise ValueError(...)\n"
            "    return tokens_per_second * flops_per_token / (peak_tflops * 1e12)\n\n"
            "How to read it:\n"
            "    0.4-0.5   large, well-optimized models on A100/H100\n"
            "    0.1-0.2   our 9M model\n"
            "    < 0.05    something is wrong: look at the dataloader or raise the batch\n\n"
            "NOBODY reaches 1. And with a small model a low MFU is unavoidable: 320x320 "
            "matrices are not enough to saturate the tensor cores. It is not your fault.",
        ),
        "chinchilla_optimal_allocation": (
            "You have a fixed budget of FLOPs. Do you spend it on a large model with little "
            "data or a small one with a lot?\n\n"
            "Hoffmann et al. (2022) measured it by training more than 400 models: both should "
            "grow PROPORTIONALLY, about 20 tokens per parameter.",

            "Starting from C = 6ND (module 01) and D = k*N:\n\n"
            "    C = 6 * N * (k*N) = 6k * N^2\n\n"
            "    N = sqrt( C / (6*k) )\n"
            "    D = k * N\n\n"
            "Validate that the budget is positive. Return a dict with `params`, `tokens`, "
            "`tokens_per_param` and `compute`.",

            "    if compute_budget <= 0:\n"
            "        raise ValueError(...)\n"
            "    params = (compute_budget / (6 * tokens_per_param)) ** 0.5\n"
            "    tokens = tokens_per_param * params\n\n"
            "THE CHECK THAT BUILDS CONFIDENCE: feed it Chinchilla's real budget, 5.88e23 "
            "FLOPs. The formula predicts 70.0 billion parameters, and the real model had "
            "exactly 70B.\n\n"
            "There is a test that verifies this, and seeing it work on a historical case "
            "gives a good deal more confidence than reading the formula.",
        ),
    },
    # ------------------------------------------------------------------ module 13
    "13_final_training": {
        "overfit_single_batch": (
            "A model with millions of parameters has more than enough capacity to memorize "
            "four sequences. If you give it the SAME batch over and over, the loss HAS to "
            "drop to practically zero.\n\n"
            "If it does not drop, there is a bug. And you know it in 30 seconds instead of "
            "four hours.",

            "The simplest possible loop, no scheduler, no accumulation and no AMP:\n\n"
            "    opt = optimizer_factory(model.parameters())   # or AdamW if None\n"
            "    model.train()\n"
            "    repeat `steps` times:\n"
            "        _, loss = model(x, y)\n"
            "        opt.zero_grad(set_to_none=True)\n"
            "        loss.backward()\n"
            "        opt.step()\n"
            "        history.append(float(loss.detach()))\n\n"
            "The fewer moving parts, the fewer places a bug can hide. That is why this loop "
            "is deliberately bare.",

            "Three details:\n\n"
            "1. The `model.train()` is NOT decorative. If the model came in eval mode, "
            "dropout would be off and you would not be testing the same code path real "
            "training uses. There is a test that checks it.\n\n"
            "2. `float(loss.detach())` and not `float(loss)`: without the detach, PyTorch "
            "raises a warning about converting tensors with gradients to scalars.\n\n"
            "3. `optimizer_factory` is optional: if it is None, use "
            "`torch.optim.AdamW(params, lr=lr)`.\n\n"
            "WARNING: if the loss drops to zero TOO fast (in five steps), suspect an "
            "information leak. Check that the targets are shifted by one token relative to "
            "the input.",
        ),
        "format_eta": (
            "Formatting a duration so it reads at a glance. It looks cosmetic and it is not: "
            "you are going to look at that number many times during a run that lasts hours, "
            "and '1h 2m' reads instantly while '3725 s' has to be divided in your head.",

            "Four segments:\n\n"
            "    < 60      ->  '{s}s'\n"
            "    < 3600    ->  '{m}m {s}s'\n"
            "    < 86400   ->  '{h}h {m}m'      <- no seconds\n"
            "    otherwise ->  '{d}d {h}h'\n\n"
            "Past an hour the seconds stop being shown: when two hours are left, seconds are "
            "noise.\n\n"
            "Integer division (`//`) and modulo (`%`) do all the work.",

            "    if not math.isfinite(seconds) or seconds < 0:\n"
            "        return '?'\n"
            "    secs = int(seconds)\n"
            "    if secs < 60:    return f'{secs}s'\n"
            "    if secs < 3600:  return f'{secs//60}m {secs%60}s'\n"
            "    if secs < 86400: return f'{secs//3600}h {(secs%3600)//60}m'\n"
            "    return f'{secs//86400}d {(secs%86400)//3600}h'\n\n"
            "The `math.isfinite()` covers inf, -inf and nan in one go. Returning '?' is more "
            "honest than inventing a number when there is not enough data to estimate yet, "
            "and it avoids printing things like '-1s' or 'infd 0h'.",
        ),
    },
    # ------------------------------------------------------------------ module 14
    "14_inference": {
        "apply_repetition_penalty": (
            "A blunt patch against repetitive text: if a token has already appeared, its "
            "logit is lowered so it is less likely to come out again.",

            "The detail almost everyone gets wrong:\n\n"
            "    logit > 0  ->  logit / penalty      moves it towards zero\n"
            "    logit < 0  ->  logit * penalty      moves it away from zero, DOWNWARDS\n\n"
            "With penalty=2.0:  +3.0 -> +1.5   and   -3.0 -> -6.0\n\n"
            "If you ALWAYS divided, the -3.0 would become -1.5 and the token would become "
            "MORE likely. And since negative logits are the majority, you would be rewarding "
            "almost everything that already came out.",

            "    out = logits.clone()\n"
            "    for row in range(logits.shape[0]):\n"
            "        seen = torch.unique(generated[row])\n"
            "        values = out[row, seen]\n"
            "        out[row, seen] = torch.where(\n"
            "            values > 0, values / penalty, values * penalty\n"
            "        )\n"
            "    return out\n\n"
            "The `torch.unique` avoids penalizing twice a token that came out twice. And the "
            "`.clone()` is so you do not modify the input.",
        ),
        "top_k_filter": (
            "With a 4096-token vocabulary there are thousands of tokens with tiny but "
            "non-zero probability. Added up, that long tail can carry 20% of the mass, and "
            "every so often one comes out and derails the sentence.\n\n"
            "Top-k cuts it dead: only the k largest survive.",

            "    threshold = torch.topk(logits, k, dim=-1).values[..., -1:]\n"
            "    return logits.masked_fill(logits < threshold, float('-inf'))\n\n"
            "`torch.topk(...).values[..., -1:]` is the k-th largest logit, i.e. the "
            "threshold.\n\n"
            "If k <= 0 or k >= vocab_size, return the logits untouched.",

            "Two details:\n\n"
            "1. The `[..., -1:]` with the COLON keeps the dimension so the broadcast works. "
            "With `[..., -1]` you would lose it and masked_fill would compare wrongly.\n\n"
            "2. Use `<` and not `<=`: the threshold itself (the k-th logit) has to "
            "survive.\n\n"
            "Its flaw: k is FIXED. If the model is dead sure, k=40 lets in 39 bad "
            "alternatives; if it is torn between 100, it cuts good options. top-p solves "
            "that.",
        ),
        "top_p_filter": (
            "Like top-k, but with a VARIABLE number of candidates: probability is accumulated "
            "until it reaches `p` and it is cut there.\n\n"
            "If the model is sure, one token can hold 90% and it is kept alone. If it is torn "
            "between many, many are kept. THE NUMBER OF CANDIDATES ADAPTS, and that is what "
            "makes it better than top-k.",

            "    sorted_logits, indices = torch.sort(logits, descending=True, dim=-1)\n"
            "    probs = F.softmax(sorted_logits, dim=-1)\n"
            "    cumulative = torch.cumsum(probs, dim=-1)\n\n"
            "    drop = cumulative - probs > p     # the cumulative BEFORE this token\n"
            "    drop[..., 0] = False              # the most likely one always stays\n\n"
            "    to_drop = drop.scatter(-1, indices, drop)\n"
            "    return logits.masked_fill(to_drop, float('-inf'))",

            "Three things:\n\n"
            "1. The `- probs`: you compare the cumulative BEFORE including the current token, "
            "so the one that CROSSES the threshold still gets in. Holtzman's definition is "
            "'the smallest set whose mass EXCEEDS p', and [0.60, 0.25] = 0.85 does not exceed "
            "0.9: the third one is needed. It is an off-by-one I got wrong myself while "
            "writing the module.\n\n"
            "2. The `drop[..., 0] = False` is NOT optional: with p=0.5 and a token of "
            "probability 0.9 you would be left with no candidates and multinomial would blow "
            "up.\n\n"
            "3. The `scatter` is the hardest part to see: you sorted the logits, so the marks "
            "are in probability order and not token order. `scatter` puts them back where "
            "they belong.",
        ),
        "KVCache": (
            "When generating token 100, the naive version runs all 100 tokens through the "
            "model again, even though the first 99 have not changed. Generating N tokens "
            "costs O(N^2) instead of O(N).\n\n"
            "The cache stores the keys and values already computed. What CANNOT be cached are "
            "the queries: every new token needs its own question.",

            "The class is simple:\n\n"
            "    __init__(n_layers):  two lists of n_layers elements, all None\n"
            "    update(layer, k, v): if None, store; otherwise torch.cat(..., dim=-2)\n"
            "                         and return the FULL K and V\n"
            "    seq_len:             0 if empty, otherwise keys[0].shape[-2]\n"
            "    reset():             set everything back to None\n"
            "    memory_bytes():      sum numel() * element_size() of whatever is not None",

            "The `dim=-2` is the time dimension with the shape (B, n_heads, T, head_dim). Use "
            "a NEGATIVE index: with dim=2 it would work here and break if the number of "
            "dimensions ever changed.\n\n"
            "The difficulty of this module is not here, it is in exercise 5: making the cache "
            "give EXACTLY the same result.",
        ),
        "generate_with_cache": (
            "The same autoregressive loop as module 00 (context -> distribution -> sample -> "
            "append -> repeat), now with a real model and a cache.\n\n"
            "Two phases: PREFILL (the whole prompt is passed in and the cache is filled) and "
            "DECODE (at each step ONLY the new token goes in).",

            "The order of the filters matters:\n\n"
            "    1. repetition penalty   (on the raw logits)\n"
            "    2. temperature          (divide)\n"
            "    3. top-k\n"
            "    4. top-p\n\n"
            "Temperature goes before the filters because it changes the cumulative masses "
            "top-p looks at.\n\n"
            "And use `logits[:, -1, :].float()`: under AMP the logits arrive in fp16 and "
            "multinomial can give odd results with very small probabilities.",

            "THE DETAIL THAT BREAKS EVERYTHING: RoPE has to rotate the new token by the angle "
            "of its REAL POSITION.\n\n"
            "When generating token 50 you pass in a tensor of length 1. If you apply RoPE as "
            "is, it rotates it as if it were position 0, and cached generation produces text "
            "that is different from and worse than uncached generation. Nothing fails: the "
            "model simply writes badly.\n\n"
            "That is why attention receives `pos_offset` and slices the tables:\n"
            "    cos_t = cos[pos_offset : pos_offset + seq_len]\n\n"
            "There is a test that compares against uncached generation and requires IDENTICAL "
            "output.\n\n"
            "And it STOPS on reaching the maximum context, instead of truncating: truncating "
            "with a cache would require remapping the RoPE positions of everything that "
            "remains (sliding window attention). Stopping is the honest option.",
        ),
    },
    # ------------------------------------------------------------------ module 15
    "15_evaluation": {
        "perplexity_from_loss": (
            "A loss in nats does not read well: 1.60 does not say much. Perplexity does, "
            "because it is interpreted as HOW MANY equally likely OPTIONS the model is torn "
            "between.",

            "    perplexity = exp(loss)\n\n"
            "    loss 8.317  ->  4096   (untrained: torn between the whole vocabulary)\n"
            "    loss 1.60   ->  4.95   (torn between about 5 options)\n"
            "    loss 0      ->  1      (perfect)\n\n"
            "It is one line. What matters is knowing how to read it.",

            "    if not math.isfinite(loss):\n"
            "        return float('inf')\n"
            "    return math.exp(loss)\n\n"
            "The guard is not decorative: `math.exp(inf)` raises OverflowError, and in the "
            "middle of an evaluation that leaves you with no idea what happened.\n\n"
            "The useful check: with a loss of ln(V), the perplexity is exactly V.",
        ),
        "bits_per_byte": (
            "Perplexity depends on the tokenizer: if your vocabulary splits words into "
            "smaller pieces, each token is easier to predict and your number comes out better "
            "WITHOUT THE MODEL BEING BETTER.\n\n"
            "Comparing perplexities across models with different tokenizers means nothing, "
            "and it is done constantly.\n\n"
            "The fix: normalize by BYTES of the original text.",

            "    bits_per_byte = (total_loss_in_nats / ln(2)) / n_bytes\n\n"
            "The `/ ln(2)` converts nats to bits: one nat is 1.4427 bits.\n\n"
            "WATCH OUT: the first argument is the TOTAL loss (the sum), not the mean. The "
            "`n_tokens` parameter is not used in the computation; it is in the signature to "
            "make that clear.",

            "    if n_bytes <= 0:\n"
            "        raise ValueError(...)\n"
            "    return total_loss_nats / math.log(2) / n_bytes\n\n"
            "The interpretation, which is rather nice: it is how many bits you would need to "
            "transmit the text using the model as a compressor. Reference points: gzip ~2.5 "
            "bits/byte, a good small model ~1.2, the best LLMs 0.6-0.8.\n\n"
            "It is not an analogy. A language model IS a compressor, and the equivalence "
            "between prediction and compression comes from Shannon (1948).",
        ),
        "run_prompt_battery": (
            "The part of evaluation that no automatic metric replaces: asking the model the "
            "same questions every time and READING what it answers.\n\n"
            "The six prompts in `PROMPTS_TINYSTORIES` come from the paper and each one tests "
            "something different: continuation, causal coherence, object tracking, resolution "
            "and closure.",

            "    prompts = prompts or PROMPTS_TINYSTORIES\n"
            "    return [\n"
            "        {'prompt': p, 'tests': label, 'completion': generate_fn(p)}\n"
            "        for p, label in prompts\n"
            "    ]\n\n"
            "`PROMPTS_TINYSTORIES` is a tuple of (prompt, label) tuples, so it unpacks "
            "directly. The three keys have to be named exactly like that.",

            "Why `generate_fn` is passed instead of the model: it wraps the model AND the "
            "tokenizer, so this function knows nothing about either. It is the same pattern "
            "as `get_batch` in module 04 and `optimizer_factory` in module 13.\n\n"
            "And it makes the exercise testable with a fake generator.\n\n"
            "The value of the exercise is not in the code (three lines): it is in having a "
            "FIXED battery you can rerun every time you change something.",
        ),
    },
    # ------------------------------------------------------------------ module 16
    "16_finetuning": {
        "build_chat_template": (
            "A pretrained model only knows how to continue text. If you write it a question, "
            "the most likely thing is that it answers with MORE questions: a document that "
            "starts like that usually carries on like that.\n\n"
            "For it to ANSWER you have to teach it a format. That is what the markers are.",

            "    [{'role': 'user', 'content': 'Hello'},\n"
            "     {'role': 'assistant', 'content': 'How are you'}]\n\n"
            "    ->  <|user|>Hello<|end|><|assistant|>How are you<|end|>\n\n"
            "Each message is wrapped in its role's marker and closed with `<|end|>`. The "
            "markers are in `CHAT_MARKERS`.\n\n"
            "With `add_generation_prompt=True`, `<|assistant|>` is appended at the end and "
            "the string is left OPEN: that is what is used at inference time.",

            "    parts = []\n"
            "    for message in messages:\n"
            "        role = message['role']\n"
            "        if role not in CHAT_MARKERS:\n"
            "            raise ValueError(...)\n"
            "        parts.append(f\"{CHAT_MARKERS[role]}{message['content']}"
            "{CHAT_MARKERS['end']}\")\n"
            "    if add_generation_prompt:\n"
            "        parts.append(CHAT_MARKERS['assistant'])\n"
            "    return ''.join(parts)\n\n"
            "The `<|end|>` is what teaches the model WHEN TO STOP. Without an end marker, it "
            "would generate indefinitely.",
        ),
        "mask_prompt_tokens": (
            "In SFT you do not want the model to learn to GENERATE the user's questions: you "
            "want it to learn to ANSWER them.\n\n"
            "By putting -100 at the prompt's positions, `F.cross_entropy(..., "
            "ignore_index=-100)` skips them.",

            "    input_ids = [10, 11, 12, 20, 21, 22]   with prompt_len = 3\n"
            "    targets   = [-100, -100, 20, 21, 22, -100]\n\n"
            "Note: TWO ignored positions at the start, not three. And one at the end.\n\n"
            "    targets = [ignore_index] * len(input_ids)\n"
            "    for i from prompt_len - 1 to len(input_ids) - 2:\n"
            "        targets[i] = input_ids[i + 1]",

            "WHY TWO AND NOT THREE, which is the whole exercise.\n\n"
            "The targets are SHIFTED by one token. So at position 2 (the LAST token of the "
            "prompt) the target is already input_ids[3] = 20, which is the first token of the "
            "ANSWER.\n\n"
            "And that one does matter: it is exactly the transition 'the question is over, my "
            "turn to answer', which is the most important thing the model has to learn from "
            "SFT. If you started at prompt_len, you would skip precisely that transition.\n\n"
            "And it gives no signal at all: you simply waste the most informative position.",
        ),
        "LoRALinear": (
            "Full fine-tuning of a large model needs memory for the weights, the gradients "
            "AND Adam's states: about 12 bytes per parameter.\n\n"
            "LoRA starts from an observation: the changes fine-tuning makes are LOW RANK. So "
            "W is FROZEN and the product of two thin matrices is added to it.",

            "    output = x @ W^T  +  (alpha/r) * x @ A^T @ B^T\n\n"
            "with A of size (r, d_in) and B of size (d_out, r).\n\n"
            "With d_in = d_out = 320 and r = 8:\n"
            "    full W:  102,400 parameters\n"
            "    A and B:   5,120 parameters    (5%)\n\n"
            "Submodules: `base` (the original layer), `lora_A`, `lora_B` and "
            "`lora_dropout`.\n\n"
            "And do not forget to freeze the base:\n"
            "    for p in self.base.parameters():\n"
            "        p.requires_grad = False",

            "THE INITIALIZATION IS NOT SYMMETRIC, and that is the important part:\n\n"
            "    lora_A -> nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))\n"
            "    lora_B -> ZEROS\n\n"
            "With B = 0, the product BA is zero at the start and the layer is EXACTLY the "
            "original: fine-tuning begins without perturbing anything. There is a test that "
            "checks it.\n\n"
            "So why not both at zero? Because then the gradient of both would be zero -each "
            "multiplies the other- and they would never learn.\n\n"
            "Watch the transposes: lora_A is (r, d_in), so `x @ lora_A.T` gives (..., r), and "
            "then `@ lora_B.T` with lora_B of (d_out, r) gives (..., d_out).",
        ),
        "merge_lora_weights": (
            "During training, LoRA adds two matmuls per layer, and that shows up at inference "
            "time.\n\n"
            "By merging the adapters into the base matrix, the resulting model is "
            "INDISTINGUISHABLE from a normal one: same cost, same shapes, and it can be "
            "served with no LoRA dependency at all.",

            "    W_new = W + (alpha/r) * (B @ A)\n\n"
            "Note the order `B @ A`: B is (d_out, r) and A is (r, d_in), so the product gives "
            "(d_out, d_in), which is the shape of `weight` in an nn.Linear. The other way "
            "round the shapes would not fit.\n\n"
            "Return a plain `nn.Linear`, keeping the bias if there was one.",

            "    merged = nn.Linear(d_in, d_out, bias=layer.base.bias is not None)\n"
            "    with torch.no_grad():\n"
            "        delta = (layer.lora_B @ layer.lora_A) * layer.scaling\n"
            "        merged.weight.copy_(layer.base.weight + delta)\n"
            "        if layer.base.bias is not None:\n"
            "            merged.bias.copy_(layer.base.bias)\n"
            "    return merged\n\n"
            "Use `.copy_()` instead of assigning: that way you keep the tensor nn.Linear "
            "already created, with its metadata.\n\n"
            "This is what makes LoRA useful compared with other parameter-efficient "
            "fine-tuning methods: the adaptation is EXACTLY a sum of matrices, so it is "
            "absorbed without approximating anything.",
        ),
    },
    # ------------------------------------------------------------------ module 17
    "17_extra": {
        "quantize_int8_symmetric": (
            "A float32 takes 4 bytes and an int8 only one: the model takes a quarter of the "
            "space.\n\n"
            "The trick is storing, alongside the integers, a SCALE that lets you recover the "
            "approximate values.",

            "With W = [0.12, -0.45, 0.03, 0.28], the largest in absolute value is 0.45:\n\n"
            "    scale  = 0.45 / 127 = 0.003543\n"
            "    W_int8 = round(W / scale) = [34, -127, 8, 79]\n\n"
            "The steps:\n"
            "    1. max_abs = weight.abs().amax(dim=-1, keepdim=True)   if per channel\n"
            "    2. scale = (max_abs / 127.0).clamp_min(1e-12)\n"
            "    3. torch.round(weight / scale).clamp(-127, 127).to(torch.int8)",

            "Three details:\n\n"
            "1. WHY 127 AND NOT 128: int8 runs from -128 to 127. With 127 the range stays "
            "SYMMETRIC and zero is represented exactly. In a matrix with many small values, "
            "that avoids a systematic bias that would build up layer after layer.\n\n"
            "2. The `clamp_min(1e-12)` avoids dividing by zero if a row is all zeros.\n\n"
            "3. The `clamp(-127, 127)` protects against rounding at the edge: without it, a "
            "value right at the maximum could give 128, which does not fit in int8 and would "
            "wrap to -128. The largest weight would silently become the most negative one.\n\n"
            "Per channel is always better than per tensor (0.71% versus 1.07% error): a "
            "single row with large values does not drag the others down with it.",
        ),
        "dequantize_int8": (
            "The opposite of exercise 1: multiply by the scale to get back to float.\n\n"
            "The result is NOT equal to the original: information has been lost. That is the "
            "price of taking a quarter of the space.",

            "    return quantized.to(torch.float32) * scale\n\n"
            "One line.",

            "The `.to(torch.float32)` goes BEFORE the multiplication. If you multiplied the "
            "int8 directly, PyTorch would do the operation in integers and the result would "
            "be garbage.",
        ),
        "quantization_error": (
            "Quantize, dequantize, and compare with the original. It is how you find out "
            "whether it is worth it before applying it to the whole model.",

            "The metrics to return:\n\n"
            "    relative_error  = ||original - recovered|| / ||original||\n"
            "    max_error       = max(|difference|)\n"
            "    mean_error      = mean(|difference|)\n"
            "    compression     = bytes per original element / quantized\n"
            "    original_bytes  = numel() * element_size()\n"
            "    quantized_bytes = the int8's PLUS the scales'\n\n"
            "`tensor.element_size()` gives the bytes per element (4 for float32, 1 for int8). "
            "With that, compression falls out on its own, with no magic numbers.",

            "The RELATIVE error is the metric worth looking at: it is independent of the "
            "scale of the data, so you can compare different layers. There is a test that "
            "multiplies the weights by 1000 and checks that it does not change.\n\n"
            "And the scales TAKE UP SPACE TOO: including them in `quantized_bytes` is the "
            "honest thing to do. With one per row they are negligible, but counting them "
            "costs nothing.\n\n"
            "With the weights of a trained network, per-channel int8 is around 0.5-1% error. "
            "That this barely affects the model's quality is an EMPIRICAL FACT, not a "
            "theorem.",
        ),
    },
}


def get_hints(module_id: str, exercise_name: str) -> tuple[str, ...]:
    """An exercise's hints, from least to most explicit. Empty tuple if there are none."""
    return HINTS.get(module_id, {}).get(exercise_name, ())


def has_hints(module_id: str) -> bool:
    return bool(HINTS.get(module_id))
