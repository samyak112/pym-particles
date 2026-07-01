# Pym Particles: Architecture

Pym Particles is a neural compressor, a neural network used to compress a single file. Normally we use a neural network to *generalize*: we show it many examples of a domain and teach it patterns it can apply to unseen data. Neural compression flips this goal entirely. Instead of generalizing, we want the model to **overfit** on one specific file, using its pattern-finding ability to memorize hardcoded structure rather than extract general latent features.

## Why a Transformer

You can build a neural compressor with an RNN ([Goyal et al., 2018](https://arxiv.org/pdf/1811.08162)), an LSTM, or a GRU, but empirically a transformer wins on two counts:

1. **Faster training** — parallelism across the sequence instead of sequential recurrence.
2. **Far more parameter-efficient** — a 900KB transformer matches the compression a 10MB LSTM gets.

The trained model:
- 2 layers, hidden dim 128, 4 attention heads
- Vocabulary: 258 (256 byte values + `BOS` + `PAD`)
- Window size: 256 tokens
- Weight tying between the input embedding and output projection, halves the parameter count on the two largest matrices
- Total size: ~900KB

## Why Neural Networks Alone Aren't Enough

No matter how aggressively you overfit a reasonably-sized model, it can't perfectly *cram* a file, it will still make mistakes on high-entropy regions. So the network's output isn't the compressed data itself. It's a **probability distribution**, and an arithmetic coder does the actual compressing.

### Arithmetic Coding, Concretely

Suppose the file begins with `The cat sat ...`. After reading `The`, the model predicts:

```
cat : 0.94
dog : 0.03
boy : 0.01 ...
```

The true next token is `cat`. The arithmetic coder takes this distribution plus the correct token and narrows a numeric interval according to the probabilities. Since `cat` owns 94% of the probability mass, it can be represented in very few bits. Had the model only given `cat` a 10% probability, encoding it would have cost far more bits.

Once `cat` is encoded, both encoder and decoder know the sequence so far. `The cat` and feed that back into the model for the next prediction:

```
sat   : 0.98
slept : 0.01
ran   : 0.005 ...
```

The interval refines further, consuming roughly `-log₂(P(correct token))` bits each step. This repeats for the whole file.

```
Context
   │
   ▼
Transformer
   │
   ▼
Probability distribution
   │
   ▼
Arithmetic coder + correct token
   │
   ▼
Refined interval
   │
   ▼
Next token becomes part of the context
```

The model never compresses anything directly, its only job is answering "given everything so far, how likely is every possible next token?" The arithmetic coder converts that answer into bits. The better the prediction, the fewer bits spent. This is why all the effort in this project goes into prediction accuracy: every gain in P(correct token) translates directly into a smaller file.

**Decompression** runs the same loop in reverse. The decoder has no access to the original file, only the compressed bitstream and the trained model. It starts from the same initial context, gets the same probability distribution from the transformer, and the arithmetic decoder reads just enough bits from the stream to identify which interval the encoded value falls in. Since the encoder used this exact distribution, the decoder recovers the exact same token, deterministically, never by guessing. That token is appended to the context, and the cycle repeats.

## Tokenization and Dataset Construction

Tokenization is **byte-level** (BLE). Byte-pair encoding was the first thing tried, but it gave little benefit, these files don't have enough repetitive structure at the pair level to make BPE worthwhile. So the vocabulary starts at 256 raw byte values, plus `BOS` and `PAD`, for 258 total.

Because auto-regressing an entire N-MB file in a single pass would blow up the attention window, the file is broken into **windows of 256 tokens**, auto-regressed one after another.

Each 256-token window splits into two halves:
- **First 128 tokens** — context prefill
- **Second 128 tokens** — the actual prediction target

This split exists for two reasons:

1. **Bootstrapping each window.** Since decoding happens across many separate auto-regression passes (one per window), each pass needs an initial prompt to resume from. The first half of the window is exactly that, a prefill that lets the model pick up exactly where the last window left off. The current window's predicted half becomes next window's context prefill.
2. **Avoiding branching collisions from a tiny vocabulary.** With only 256 possible byte values, short byte sequences repeat constantly across a large file — a single byte or short run isn't a reliable resume point. Loss is *not* computed on the context-prefill tokens; they're purely there to give the model a long, low-collision "signature" (127 tokens' worth) to uniquely identify where in the file it's resuming from.

For the very first window, there's no real prior context, which is what the `BOS`/`PAD` tokens are for.

**Example** — given the byte sequence `[1,2,3,4,5,6,7,8,9,10,11,12]` and a 4-token context prefill:

```
[BOS,PAD,PAD,PAD,1,2,3,4]
[1,2,3,4,5,6,7,8]
[5,6,7,8,9,10,11,12]
```

The first window's context half is a fixed synthetic prefix, and it predicts `[1,2,3,4]`. In the next window, `[1,2,3,4]` becomes the context prefill and the model predicts `[5,6,7,8]`, and so on. During training, only the *last* token of a context-prefill half contributes to loss in that half — the rest of the prefill is purely conditioning.

As a result, every byte in the file appears twice: once as context for a future prediction, once as a prediction target.

## Training

The objective is pure overfitting, so no dropout, no weight decay, no regularization of any kind, anything that helps a model generalize actively fights the goal here.

- Optimizer: Adam
- Precision: FP16
- Gradient clipping: 1.0
- LR schedule: warmup + cosine decay

Started with `1e-4`, but `5e-4` with a 500-step warmup converged to a comparable bits/byte in far fewer epochs (7–10, versus needing about 20 at the lower rate). The warmup was necessary, the model couldn't handle that higher LR directly from step zero.

The metric tracked during training is **bits/byte**: the lower it goes, the better the eventual compression. This can be read straight off the loss curve — you don't need to actually run compression to know how well a file is going to compress at a given epoch.

## Inference: Compression and Decompression

Both phases share the same underlying pipeline, repeated forward passes through the trained model. Compression uses the resulting probability distributions to write the arithmetic-coded bitstream; decompression uses them to reconstruct the file by feeding back predicted tokens as new context.

### The Speed Problem

Naively auto-regressing a 100MB file end-to-end would take **~28 hours**. Two fixes address this:

**KV Cache** — generating each new byte only requires computing that one token's key/value and appending it, rather than recomputing attention over the full context every step.

**Seed file + parallel streaming** — rather than one long sequential crawl through the file, it's split into N independent chunks (100 by default), and all of them are compressed or decompressed simultaneously as separate arithmetic-coded streams, batched through the model together. Because training used context-prefill windows, the model has no problem resuming auto-regression from any arbitrary point in the file, it only needs the preceding 128-byte context. So a seed file stores the initial 128-byte context for each of the 100 chunks (~25KB total overhead on a 100MB file), and generation kicks off from 100 points in parallel, with the pieces merged at the end.

This is what turns compression/decompression from "one slow sequential crawl" into a batchable, parallel process.

### Time Complexity (100MB file, single AMD GPU benchmark)

| Phase | Duration / Rate | Description |
|---|---|---|
| Training (per epoch) | ~180s on AMD, ~90s on CUDA (Flash Attention) | One forward/backward pass over the file slice |
| Total convergence | 7–10 epochs (~23–33 min) on AMD, 10–15 min on CUDA | Time for the LR schedule to minimize bits/byte |
| Compression (parallel) | ~45 min | Batching 100 chunk streams through the model to write the `.pym` bitstream |
| Decompression (parallel) | ~45 min | Autoregressively reconstructing the file from 100 seed points via KV cache |

End-to-end, this is roughly **1.5–2 hours per 100MB file**. Unlike standard compressors, which are I/O-bound, Pym Particles is strictly **compute-bound** the 45-minute decompression wall comes from the inherently serial nature of autoregressive generation, even split 100 ways.

## Benchmark Results

Compression scales directly with how structurally predictable the file is, the model's loss approaches zero on highly structured data, while natural language hits a semantic bottleneck.

| Dataset | Original Size | Bits/Byte | Compressed Size | Ratio | zip (for reference) |
|---|---|---|---|---|---|
| NYC Taxi Trip Data (CSV) | 100 MB | ~0.50 | 7 MB | 14.2x | 27 MB |
| enwik9 (text slice) | 100 MB | ~1.68 | 21 MB | 4.7x | 38 MB |

## What Didn't Work

- **Mixture of experts** (routing "easy" vs "hard" windows to different sub-models) — no gain.
- **Bitmap-assisted selective masking** (excluding bytes that "can't" appear next, to sharpen the distribution). the model already concentrated most probability mass on the top 1–2 candidates, so there was nothing left to sharpen.
- **Random shuffling of training windows** — neutral to slightly negative; the model didn't care about window order, it just pattern-matched on what it received, and shuffling converged marginally slower.
- **Slicing into smaller independent chunks** hoping for proportional compression gains, didn't scale down; each chunk landed at roughly the same bits/byte.
- **Dropout / weight decay** — directly counterproductive, as expected for an overfitting objective; the model got stuck around 3.27 bits/byte.