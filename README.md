# pym-particles

PymParticles is an experimental neural compression system that combines an overfitted transformer with arithmetic coding to compress individual files.

Normal compressors such as zstd, gzip rely on statistical models and pattern matching algorithms. 

In PymParticles I tried a bit different approach: Overfit a neural network specifically on a file and expect model to find specific patterns because we are overfitting the model on the file and then auto repressively print the file byte by byte.

Unlike most machine learning systems, overfitting is not a failure mode here. The goal is to memorize a specific file as aggressively as possible. Things you'd normally add to help a model generalize like dropout, weight decay,  actively causes issues here, because they're fighting the thing the whole system is trying to do. 

The neural network does not perform compression directly. Its job is to predict the next byte as accurately as possible ( The more confident the model is in the next token the more compression we can squeeze) . The arithmetic coder converts those predictions into a compressed bitstream. 

The project exists just to tinker and understand how much can i compress a file using a neural network.

On a 100MB CSV file (NYC taxi trip data), it reaches **~0.5 bits/byte**, (compressing it to 7MB) .

On more varied byte-level data (enwik9 data set (sliced to 100mb)), it settles around **1.68–1.70 bits/byte** (compressing it to around 21mb). 

I tried different approaches to compress it more than 1.68 but no approach got me better compression than this.

---
### Current Architecture

**Tokenization -** The file is read as raw bytes no text tokenization, no subwords. Each byte (0–255) is one token, plus two special tokens (`PAD`, `BOS`), for a vocabulary of 258. This makes the system completely format-agnostic: a CSV, an executable, an image all just byte sequences to it.

**The model (`PymTransformer`)** A small transformer:

- 2 layers, hidden dim 128, 4 attention heads
- Vocabulary: 258
- Window size: 256 tokens
- Weight tying between the input embedding and output projection, halves the parameter count on the two largest matrices.
- Makes up the transformer size **900KB**

**Window Structure** - The model is trained using overlapping windows rather than processing the entire file as one continuous sequence.

```
Window Size : 256
Stride      : 128
```

Example:

```
Window 0:
[BOS PAD PAD ... PAD | bytes 0-127]

Window 1:
[bytes 0-127 | bytes 128-255]

Window 2:
[bytes 128-255 | bytes 256-383]
```

The first 128 tokens of every window act as context. The second 128 tokens are the region where loss is computed.

As a result, every byte in the file appears twice:

- once as context for a future prediction
- once as a prediction target

This design serves two purposes.

**1. Bounded Context Length -** A transformer cannot continuously grow its attention history forever. In a normal autoregressive setup, every new token increases the amount of context the model must attend to. For very large files this eventually becomes impractical because attention cost and KV cache memory grow with sequence length.

Instead of asking the model to remember the entire file, the file is broken into overlapping windows. The model only needs to reason about the most recent 128 bytes of history, regardless of whether the file is 1 KB, 100 MB, or larger.

From the model's perspective, every prediction task becomes:

```
Given the previous >=128 bytes,
predict the next byte.
```

This keeps memory usage fixed and prevents attention from growing without bound.

**2. Guaranteed Historical Context -** A second problem appears at the beginning of a file.

Autoregressive prediction requires previous tokens, but the first byte has no history. The model cannot predict byte 0 using "the previous 128 bytes" because those bytes do not exist.

To solve this, training begins with an artificial prefix:

```
[BOS PAD PAD PAD ... PAD]
```

This creates a synthetic 128-token history for the start of the file.

The BOS token marks the beginning of the stream while PAD tokens represent missing historical context. The model quickly learns that this pattern means "start of file".

After the first window, every prediction has access to real data.

For example:

```
Window 1

Context:
bytes 0-127

Targets:
bytes 128-255
```

When predicting byte 128, the model already sees the true preceding 128 bytes.

When predicting byte 200, the model still sees the true preceding 128 bytes.

This guarantees that every prediction is made with a full context window available.

#### Why The Overlap Matters

Without overlap, adjacent windows would become disconnected.

```
Window 0:
bytes 0-255

Window 1:
bytes 256-511
```

In this setup, the model would begin predicting byte 256 with no knowledge of bytes 128-255, even though those bytes are part of the true history of the file.

Using a stride of 128 fixes this.

```
Window 0:
context -> synthetic prefix
targets -> bytes 0-127

Window 1:
context -> bytes 0-127
targets -> bytes 128-255

Window 2:
context -> bytes 128-255
targets -> bytes 256-383
```

Each prediction region is conditioned on the exact 128 bytes that immediately precede it in the original file.

The overlap therefore acts as a sliding context buffer, allowing the model to perform autoregressive prediction with fixed memory requirements while still preserving the true sequential structure of the file.

---

**Training** uses:

```
Optimizer : Adam
Precision : FP16
Gradient Clipping : 1.0
Scheduler : Warmup + Cosine Decay
```

One of the most important findings during development was the impact of learning-rate scheduling.

Early experiments used:

```
1e-4
```

and required roughly twenty epochs to reach useful compression.

Switching to:

```
1e-3 + warmup
```

reduced convergence time dramatically, reaching comparable compression quality in roughly 7 epochs.

Had to keep warm up for initial steps because model was not able to handle such an aggressive learning rate directly.

### Compression using parallel streaming

Rather than compressing the file as one long sequential pass, it's split into `N` independent chunks (100 by default) and all of them are compressed _simultaneously_ as separate arithmetic-coded streams, batched through the model together. 

This is done so that we can start multiple streams of auto regression at once, because otherwise it takes a lot of time to de compress a 100mb file using auto regression, so we split the file in 100 chunks , and make a seed file which stores the initial 128 bytes of all the N chunks, and because our training was done in a way that model gets the first 128 tokens and predicts on the basis of them this was safe to do, because model is used to start prediction from any place, only if it has the preceding 128 token context.

So we save a seed file which contains the 128 token context of 100 splits across the file and start the auto regression from 100 different places in parallel at once and then merge those pieces at the end.

These seeds are tiny: ~25KB total overhead for a 100MB file split into 100 chunks. This is what makes both compression and decompression batchable and parallel, instead of one slow sequential crawl through the entire file.

**Inference uses a KV cache**, so generating each new byte only requires computing that one token's keys/values and appending them, instead of recomputing attention over the full context every step.

## What I tried that didn't work

A short version, full writeups with the reasoning and numbers are in [EXPERIMENTS.md](https://github.com/samyak112/pym-particles/blob/main/docs/experiments.md):

- **Mixture of experts** (route "easy" windows to one model, "hard" windows to another) — no gain.
- **Bitmap-assisted selective masking** (explicitly excluding bytes that "can't" appear next, to sharpen the arithmetic coder's distribution) — failed: exactly where this would help most, the model's concentrated most of the probability distribution in first 2 elements and kept giving lower shares after that so there was no point to reduce the probability distribution.
- **Random shuffling of training windows** — no real benefit; the idea was to check if some chunks learn better when closer to relatively similar chunks, I didnt worked on this idea because it was asking for computationally a lot of similarity work, but I tried a simple idea of randomizing the order of the sequences hoping that the over fitting will be affected either positively or negatively but there was a neutral effect, model didnt cared what was before and just made patterns with what it recieved. If anything it converged slightly slower.
- **Slicing the file into smaller independent chunks**, hoping compression would scale down proportionally — it didn't; each chunk still landed around the same ~0.5 bits/byte.
- **Dropout / weight decay** — directly counterproductive, as expected; the model got worse at memorizing and got stuck around 3.27 bits/byte.
- **Dropping the LR warmup** — catastrophic; bits/byte shot up to ~100 and stayed there.
