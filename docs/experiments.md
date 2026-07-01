This is the lab notebook. The README.md describes the model that actually shipped; this doc covers everything that led there — the dead ends, the diagnostics that explained *why* something failed, and a couple of bigger architectural ideas that were reasoned through in depth but never built. Negative results are included on purpose: most of what shaped the final design came from things that didn't work.

---

## Part 1: Early Foundations & Architecture Search

### GRU vs. Transformer — the first baseline

Before any of the file-compression-specific design work, the very first question was simply: GRU or Transformer?

```
GRU first epoch:
  loss 0.6673, 0.963 bits/token, window 512, hidden dim 512, 2 layers, size 13MB

Transformer first epoch:
  loss 1.4723, 2.124 bits/token, lr 1e-4, time 136.49s, size 7MB
```

The GRU started off with a better loss in epoch 1, but it was nearly double the size on disk. This was an early, rough comparison rather than a controlled one — different window sizes, no tuning on either side — but it set the direction toward exploring transformers further given the size advantage, since file size is the thing actually being optimized for here.

### The PAD/BOS NaN bug

An early version of the custom sliding-window scheme used a sequence mask of 255 `PAD` tokens followed by a single `BOS` token at the very start of the file (to give the first window *something* to attend to before real data exists). This produced **NaN loss**.

The cause: `PAD` tokens were being explicitly masked out in the attention function, and with 255 of them at the start, the masking made the early attention computation degenerate into effectively `-inf` everywhere — nothing left for softmax to normalize over.

**Fix:** Bookend the prefix with `BOS` at both the very start *and* the very end of the PAD run, instead of just one `BOS`. That single extra real token was enough to keep the attention computation numerically sane.

### Hidden dimension comparisons

- 128-dim hidden size worked fine on a CSV file but produced 1.7 bits/byte on enwik-style text data.
- 256-dim hidden size brought that down to 1.5 bits/byte on the same enwik-style data.
- Model size scaling at 6 layers: 96 dims → 1.5MB, 128 dims → 2.5MB.

This is the expected trade-off — bigger model, better fit, bigger file on disk — and it's the core tension the whole project sits on top of: the model itself counts against the compressed size, so every dimension added has to earn its keep in bits saved.

### The LR warmup discovery

This turned out to be one of the highest-leverage findings in the whole project, discovered early and carried through to the final training pipeline.

- At `lr=1e-4`, no warmup: reaching 0.5 bits/byte took **~20 epochs**.
- At `lr=1e-3` **with** a linear warmup ramp: the same 0.5 bits/byte was reached in **~4 epochs**.

A 5x reduction in epochs from one change. The intuition: at a high learning rate, randomly-initialized weights take a huge, destabilizing step on the very first few batches if there's no ramp-up. The warmup lets the weights settle into a reasonable region before the full learning rate kicks in, which is what makes the higher (and much faster-converging) LR usable at all.

### Failed: dropout + regularization

Added standard dropout and weight regularization to see if it would help generalization (it wasn't needed, but worth checking). As expected given the project's actual goal:

- The model was no longer able to fully memorize the file.
- Loss got stuck around **3.27 bits/byte** — nowhere close to the ~0.5 bits/byte the unregularized model reaches.

This isn't a surprising result in hindsight, but it's worth stating plainly: anything that fights memorization is actively working against the compression objective here. Confirmed and moved on.

### Failed: removing warmup (with regularization still on)

A second variant of the regularization experiment: kept dropout/regularization on, but also reverted the learning rate back to `1e-4` and removed the warmup schedule entirely.

Result: bits/byte shot up to **~100 and stayed flat** — essentially a non-functional training run. Between this and the dropout result above, the conclusion was unambiguous: warmup and zero regularization are both load-bearing, not just nice-to-haves.

### Positional encoding: sinusoidal vs. RoPE

A head-to-head comparison at a slightly different config (96 dims, 6 layers) to see whether RoPE (rotary positional embeddings) would outperform plain sinusoidal:

**With sinusoidal PE:**
```
epoch 1: loss 1.7477, 2.521 bits/token, 470.4s
epoch 2: loss 1.2084, 1.743 bits/token, 467.0s
epoch 3: loss 1.1604, 1.674 bits/token, 469.1s
epoch 4: loss 1.1349, 1.637 bits/token, 468.0s
```

**With RoPE:**
```
epoch 1: loss 1.7293, 2.495 bits/token, 508.9s
epoch 2: loss 1.2426, 1.793 bits/token, 511.0s
epoch 3: loss 1.1989, 1.730 bits/token, 509.1s
epoch 4: loss 1.1749, 1.695 bits/token, 508.9s
```

Sinusoidal PE won on every epoch past the first, and ran noticeably faster per epoch (~468s vs ~509s) on top of that. Given that every window is a fixed 256 tokens and the model only ever sees one window at a time, positional information doesn't need to generalize across arbitrary sequence lengths the way RoPE is built for — plain sinusoidal PE, reused identically for every window, turned out to be both simpler and better here. This is the version that shipped.

---

## Part 2: Chasing Long-Range Context — How the Current Window Design Came to Be

This section is less a list of failures and more a record of the actual design process — several ideas were tried, found wanting in a specific way, and that specific shortcoming pointed directly at the next idea. The end of this chain *is* the architecture described in the README.

### The starting tension

Two extremes don't work for compressing a file far larger than a model's context window:

- **A pure RNN** processes the file with constant memory, but it has to compress all history into one fixed-size vector. Over a large file, that vector becomes lossy — it can't hold exact variable names, exact syntax, exact anything indefinitely. The arithmetic coder ends up paying for the RNN's hallucinations.
- **A naive Transformer** gives flawless retrieval over whatever it can see, but standard self-attention is `O(N²)` in sequence length. Feed it a file rather than a chunk and the attention matrix alone would need more VRAM than exists.

The early instinct (covered in much more depth in [Part 5](#part-5-explored-but-not-implemented--scaling-to-much-larger-files)) was a block-recurrent design: chunk the file, run a Transformer over each chunk, and pass a small set of "memory tokens" forward between chunks to carry context without re-attending to everything. That thread is large enough that it gets its own section later — what matters for the *current* architecture is how a smaller, file-scale version of that idea evolved.

### LSTM tried, then dropped for being too slow

As part of testing the per-chunk model choice, an LSTM was tried directly: **~400 seconds per epoch**, ~12MB per model. Switching to a Transformer of comparable scale brought that down to **~135 seconds per epoch** and **~6MB** on disk — a clear win on both speed and size.

The catch: the Transformer took roughly **25 epochs** to reach the loss the LSTM had reached in far fewer. That gap was treated as a sign of an architectural problem in the Transformer setup at the time (rather than an inherent Transformer weakness) — and it's part of what motivated moving away from a plain block-recurrent design toward something with an explicit context-handoff mechanism.

### The "Markov-based transformer" / MEM token idea

The next design: instead of passing forward a vector via recurrence, generate a single trained `[MEM]` token at the end of each window that encodes "what you need to know to keep going," and prepend it to the next window. To keep it from bloating into an ever-growing context over time, the scheme was meant to be **"use and throw"** — window 1's MEM token is used as the anchor for window 2, but when window 2 generates *its* MEM token, window 1's gets masked out. Since the model is overfitting (not generalizing), the idea was that one window of lookback should be enough of an anchor to keep autoregression going indefinitely.

### A side problem: duplicate sequences

While reasoning through this, a separate issue surfaced: if the same byte sequence appears more than once in the file (very plausible at a 512-token window size in repetitive data), the model has no way to know *which* occurrence it's currently in — the MEM token and the upcoming sequence could be identical to another spot in the file, and the model would have no signal to disambiguate.

The proposed fix: add a 257th vocabulary token, `[sentinel]`, find all duplicate windows in the file, store the duplicated content once in a lookup table, and replace repeated occurrences with a pointer to it — stripping the original occurrence too so nothing is double-counted. This was reasoned through as a viable fix but became moot once the MEM token approach itself was dropped (see next).

### MEM token abandoned — too sequential

The MEM token idea was ultimately scrapped: it makes the entire training and inference pipeline tightly sequential (you can't compute window N+1's MEM token without first finishing window N), which kills the parallelism the rest of the system depends on.

**First replacement attempt:** instead of a trained MEM token, just carry the literal last 256 bytes of the previous window forward as the start of the next window (256 old + 256 new, instead of a fresh 512).

This immediately reintroduced the original forking problem in a new shape: if the model is autoregressively predicting starting from those carried-forward 256 bytes, it's right back to "given just the tail of the last window, which of several valid continuations is this?" — the same ambiguity the MEM token was supposed to solve, now reappearing because the carried-forward bytes are themselves being *predicted into* rather than just *attended to*.

### The fix: split the loss, not just the window

The resolution was to keep the 256-old + 256-new structure, but change *where loss is computed*, not just what's fed in:

- Apply a **full (non-causal) mask** over the old 256 bytes — the model can attend to all of them, but no loss is computed on them, and critically, they are never themselves predicted.
- Apply a normal **causal mask** over the new 256 bytes — these are predicted autoregressively, scored with loss, same as always.

Worked through with a concrete example: given the sequence `"hey how are you what"`, where `"hey how are you"` is the *old*, carried-forward half and `"what"` is the *new* half, the model still computes a prediction at every position internally (`hey → how`, `how → are`, etc.) but only the predictions inside the new half are scored. So instead of having to guess one token at a time starting from an ambiguous tail, the model gets the full true 256-byte context handed to it up front, and only has to extend from there. The duplicate-sequence forking problem also stops being a concern, since 256 bytes of real, true context removes the kind of short-tail ambiguity that made it possible in the first place.

This is the design that shipped — described in the README as the "second-half loss" scheme, with windows of 256 (128 old + 128 new in the final shipped config, rather than 256 + 256) and a stride of 128.

---

## Part 3: Compression-Quality Experiments That Didn't Pan Out

These were all run *after* the core architecture above was working and producing a baseline compression ratio. The goal in each case was to push bits/byte lower from there.

### Failed: slicing the file into smaller independent chunks

Hypothesis: if a 100MB file compresses to ~0.5 bits/byte, maybe slicing it into smaller independent pieces (e.g. 50MB) and training/compressing each separately would push the ratio down further — smaller, more specialized problem per model.

Result: each 50MB chunk still landed around **~0.5 bits/byte** — no proportional improvement at all. The compression ratio appears to be a property of the data's actual entropy and the model's capacity to predict it, not something that improves just by shrinking the unit being compressed.

### Failed: mixture of experts (easy/hard window routing)

**The idea:** after training the main model, run a diagnostic pass over every window with `CrossEntropyLoss(reduction='none')` to get a per-window bit-cost — essentially a difficulty map of the whole file. Bucket windows by that cost:

```
0.0 - 0.3 bits/byte : very easy
0.3 - 0.5 bits/byte : easy
0.5 - 0.7 bits/byte : medium
0.7 - 1.0 bits/byte : hard
1.0 - 1.5 bits/byte : very hard
1.5+    bits/byte    : near-random
```

The hard windows turned out to be dominated by high-precision GPS coordinates (15 decimal places) — digit sequences that are close to genuinely unpredictable from a 128-token context.

**The MoE attempt:** split into two datasets — windows in the 0.0–0.7 buckets (~84% of the file) trained one model from scratch, and windows in the 0.7+ buckets (~16%) trained a second model specifically on the hard stuff. A routing table would then determine which model encodes which byte positions during compression.

**Result: no gain at all.** The hypothesis was that Model 1, freed from the gradient noise of unlearnable GPS digits, would converge faster and lower on the patterns it *could* learn, while Model 2 would specialize on the hard regions. In practice, Model 2 struggled to push the hard windows below ~0.7 bits/byte regardless of how it was trained — suggesting those windows are fundamentally information-dense rather than just undertrained, which routing a separate model at them doesn't fix.

### Failed: random shuffling of training windows

Hypothesis: maybe the model would learn better if similar-looking chunks were grouped together as prefill context, rather than windows appearing in raw file order.

**Test:** randomized window order on a 1MB sample and compared training curves.

- Without shuffling: reached 3 bits/byte in 10 epochs.
- With shuffling (across 5–6 clean training runs): reached 3 bits/byte in 11–12 epochs.

Shuffling was, if anything, slightly *worse*, not better.

**A useful side-finding from the same investigation:** running a control where prefill context was removed entirely (each window's "context" half was genuinely new data the model had never predicted before, rather than content from a real preceding window) made things worse still. This confirms that the real, true prefill context *is* doing useful work for the model — the shuffling experiment failed not because prefill doesn't matter, but because *scrambling which* prefill goes with which target doesn't help beyond what real, in-order prefill already provides.

### Failed: bitmap-assisted selective masking ("adaptive frequency tables")

This was the most heavily developed of the failed ideas, with the most math behind it, so it's worth walking through in full.

**The motivating gap.** Even after the model is well-trained, its softmax distribution over 258 possible bytes never assigns exactly zero probability to anything — a clipping step ensures every byte gets at least some nonzero probability (required for arithmetic coding to work at all). But "near-zero" still costs bits: if 200 unlikely bytes each hold 0.001 probability, they collectively eat 20% of the coding interval, even though the true byte is never going to be any of them in a given context. The idea was to explicitly strip those structurally-impossible bytes out before encoding, and renormalize over what's left.

**The mechanism.** For a region of the file, build a "definitely not" bitmask — a 256-bit (32-byte) bitmap marking which byte values are excluded — and apply it to the model's output before arithmetic coding, recovering the wasted probability mass for the bytes that actually matter. The entropy gain from excluding a set `S` of symbols is:

```
ΔH = -log2(1 - Σ p[t, s])   for s in S
```

— a concave, increasing function of how much probability mass you can safely remove. The catch: the mask **must be derivable identically by both encoder and decoder**, or decoding diverges. The chosen approach was to derive masks straight from the raw file bytes (not the model) — both sides have access to the same raw data, so no extra metadata needs to be transmitted at all for that version.

**The greedy version actually tried:** slide through the file, tracking a running "absent byte" count — bytes that haven't appeared yet in the current run. Keep extending the run as long as at least 170 of the 258 possible byte values remain absent. The moment a new, previously-unseen byte appears and drops the absent count below 170, close the run and start a new one. Each run gets stored as 40 bytes (4 for start position, 4 for end position, 32 for the bitmap). Overhead was genuinely tiny when it worked — a 4096-token run costs 256 bits amortized, or about 0.0625 bits/token.

**A more ambitious version was also worked out on paper** (a two-pass, full-information optimization, since the entire probability matrix is known in advance): Pass 1 computes the locally optimal exclusion set per token (sort all non-correct bytes by probability, accumulate excludable mass). Pass 2 scores the trade-off of *dropping* a symbol from the mask early in order to extend a run further, weighing immediate entropy gain against how soon that symbol would otherwise appear as a correct answer and force the run to end. This was reasoned through in detail (precomputing `next_collision[t, s]` distances, etc.) but the simpler greedy version was what actually got tested end-to-end.

**Why it failed.** The real-world result, in the project's own words at the time: *"apparently for the tokens where the model is predicting bad, it's predicting it very bad — meaning it's hard to find out among N elements which one is correct, because they all have very small scores and all look identical, so there's no way to lift it from the others."* In other words: the masking idea helps most exactly where the model has a long tail of near-zero-but-not-quite-zero probabilities it's confidently *not* going to use. But in the windows where compression is actually struggling (the 0.7+ bits/byte bucket from the entropy bucketing analysis above), the model isn't confidently excluding most bytes — it's *genuinely uncertain* among a meaningfully-sized set of plausible candidates. There's no safe, low-risk exclusion set to find there, because excluding any of those candidates risks excluding the actual correct answer. The technique works fine on the easy windows, which didn't need the help, and doesn't work on the hard windows, which did.

---

## Part 4: Diagnostic Deep Dive — Why Doesn't It Go Lower?

After the masking and MoE attempts both failed in related ways, a full diagnostic pass was run across the entire file (104,857,600 evaluated tokens) to understand the actual shape of the problem, rather than guessing at more fixes.

### Overall accuracy

```
Overall Top-2 Hit Rate : 75.63%
Rank 1 hits  : 67,457,433  (64.33%)
Rank 2 hits  : 11,850,661  (11.30%)
Ranks 3-15   : 21,685,140  (20.68%)
Ranks 16-30  :  2,823,867  ( 2.69%)
Ranks 31-100 :  1,036,526  ( 0.99%)
```

### Bit cost by rank

```
Rank  1 :  67.5M hits, avg 0.43 bits/byte
Rank  2 :  11.8M hits, avg 2.51 bits/byte
Rank  3 :   5.9M hits, avg 3.38 bits/byte
Rank  5 :   2.6M hits, avg 4.40 bits/byte
Rank 10 :  0.86M hits, avg 5.60 bits/byte
Rank 20 :  0.23M hits, avg 7.11 bits/byte
21-100  :   2.3M hits, avg 8.52 bits/byte
101+    :    5.6K hits, avg 17.46 bits/byte
```

The cost climbs steeply and fast — by rank 5, a "miss" is already costing 10x what a rank-1 hit does. This is exactly the lever the bitmap-masking idea was trying to pull (cheapen the rank-2-and-beyond cases), and exactly why it couldn't: the ranks that are expensive are also the ranks where the model has no safe exclusions to offer.

### Position within the window doesn't matter

The miss rate (rank 2+) was checked at every single one of the 128 prediction positions inside a window (positions 128–255 of the 256-token window). It sits in a tight band of **~24.0% to 25.1%** at literally every position, with no meaningful trend from the start of the predicted half to the end. This rules out "the model is running low on usable context as the window goes on" — whatever's causing the misses, it's not about running out of room within a window.

### What the model gets right (Rank 1 byte profile)

The top rank-1 bytes, with how often they're preceded by a particular byte:

```
' '  (space) : 18.86% of all rank-1 hits, 89.8% of all space occurrences correctly predicted
'e'          :  8.99%,  72.5% correct — most often follows 'h'
't'          :  6.86%,  72.2% correct — most often follows ' '
'n'          :  5.80%,  76.1% correct — most often follows 'i'
'o'          :  5.14%,  64.1% correct — most often follows 'i'
'a'          :  5.12%,  57.7% correct — most often follows ' '
'['  / ']'   :  2.5% each, ~80% correct — structural/markup, highly predictable once one bracket opens
```

This is the expected shape for English/markup text — the model has clearly learned standard letter-frequency and digraph statistics well.

### What the model gets wrong (Rank 2 confusion)

When the correct byte was actually the model's *2nd* choice, here's what it guessed instead (1st choice):

```
true 'a' (1.15M instances) → guessed 'o' (23.3%), 'e' (18.3%), 't' (18.3%)
true 'i' (0.94M)           → guessed 'e' (32.5%), 'a' (18.0%), 'o' (11.9%)
true 'e' (0.92M)           → guessed 'i' (25.3%), 'a' (19.0%), 'o' (17.1%)
true 's' (0.80M)           → guessed ' ' (35.0%), 'n' (17.1%)
true ',' (0.26M)           → guessed ' ' (92.4%)  — near-total confusion with space
true '.' (0.24M)           → guessed ' ' (83.8%)
```

The pattern is consistent: when the model is wrong, it's not wrong in a random way — it's confusing vowels for vowels, common letters for common letters, punctuation for the space character that usually precedes it. This is the most direct evidence that the model's failures are about genuine semantic/word-level ambiguity, not noise or a training bug.

### The conclusion

In the project's own framing at the time: the model isn't failing because it's confused by weird syntax or unusual bytes — it's failing because it can't resolve basic English/code lexical transitions. Every time it sees a space, a `t`, or an `a`, it's facing something close to a coin-flip between a small set of very plausible next characters. A 2-layer, 128-dimension character-level model handles bigram-level statistics close to perfectly (hence the strong rank-1 numbers) but doesn't have the semantic capacity to know *which word* makes sense in context — it's capped at character-level pattern matching, and that cap is what leaves the correct byte sitting at rank 2 roughly 11–12% of the time, no matter what gets done downstream of the model's output (masking, routing, or otherwise).

This is the real reason the masking and mixture-of-experts experiments in Part 3 both came up empty: they were both attempts to extract more value from a probability distribution whose shape is already a fairly accurate reflection of genuine ambiguity, not of fixable slack.

---

### Query tokens

This was an attempt to increase the speed of inference, the idea was to somehow run the inference also in parallel for all tokens like we do in training, so the idea was that during training instead of letting model train on the loss of real tokens after the 128 prefill i train it on 256 new tokens that I will make, so during training it will work like this 

if the original sequence is 

[1,2,3,4,5,6,7,8]

then 

1,2,3,4 becomes prefill context and then instead of showing model 5,6,7,8 at once in training , i change it with T1, T2, T3, T4

and then during training model sees something like this, while predicting T4

1,2,3,4,T1,T2,T3

so when T3 tries to predict it sees the prefill and the T tokens which are arranged in a specific way using prefill premutation and then predict a token and then in target i show the correct token not T4

But it failed, the compression ratio declined a lot went from 1.5 to 5.3 for 1 mb file