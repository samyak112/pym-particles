# pym-particles

PymParticles is an experimental neural compression system that combines an overfitted transformer with arithmetic coding to compress individual files.

The core idea: a transformer is trained to overfit on a single target file, predicting the next byte as accurately as possible. The more confident the model is in the next byte, the more compression an arithmetic coder can squeeze out of that prediction. The network never compresses anything directly, it just predicts; the arithmetic coder turns those predictions into a compressed bitstream.

Unlike most ML systems, overfitting is not a failure mode here, it's the entire goal. Things you'd normally add to help a model generalize (dropout, weight decay) actively hurt this system, since they fight the thing it's trying to do.

This project exists to explore how far a neural network can compress a single file. It's not intended as a new compression algorithm, systems combining neural prediction with entropy coding have been explored before, most notably DeepZip.

**Full writeup on how and why this works: [ARCHITECTURE.md](https://github.com/samyak112/pym-particles/blob/main/docs/architecture.md)**

```bibtex
@inproceedings{7fcb664b03ac4d6497048954d756b91f,
title = "DeepZip: Lossless Data Compression Using Recurrent Neural Networks",
author = "Mohit Goyal and Kedar Tatwawadi and Shubham Chandak and Idoia Ochoa",
year = "2019",
month = "5",
day = "10",
doi = "10.1109/DCC.2019.00087",
language = "English (US)",
series = "Data Compression Conference Proceedings",
publisher = "Institute of Electrical and Electronics Engineers Inc.",
editor = "Ali Bilgin and Storer, {James A.} and Marcellin, {Michael W.} and Joan Serra-Sagrista",
booktitle = "Proceedings - DCC 2019",
address = "United States",
}
```

## Demo

<video src="https://github.com/user-attachments/assets/2cd077f6-5a50-4f75-adad-1e1877884d94" controls width="800">Demo Video</video>

## Benchmark Results

Compression performance scales directly with the structural predictability (entropy) of the target file. Highly structured data lets the model's loss approach zero; complex natural language hits a semantic bottleneck.

| Dataset / File Type | Original Size | Bits/Byte | Compressed Size | Compression Ratio | zip |
|---|---|---|---|---|---|
| NYC Taxi Trip Data (CSV) | 100 MB | ~0.50 | 7 MB | 14.2x | 27 MB |
| enwik9 dataset (text slice) | 100 MB | ~1.68 | 21 MB | 4.7x | 38 MB |

You can download the dataset used for these benchmarks [here](https://drive.google.com/drive/folders/1P9hoPcViT2HxP7Zk5UzrSV79wiKQp44U?usp=sharing).

For a list of things that I tried and didn't worked (MoE routing, bitmap masking, window shuffling, chunk slicing), see **[EXPERIMENTS.md](https://github.com/samyak112/pym-particles/blob/main/docs/experiments.md)**.

## Running Locally

The repository is intentionally structured as a single-file workflow.

Place the file you want to compress in the project root and run:

```
python main.py
```

When prompted, enter the file name:

```
my_file.txt
```

The script will:

1. Train a transformer on that file if no trained model exists.
2. Compress the file into a `.pym` archive.
3. Decompress the archive back into a reconstructed file.
4. Verify that the reconstructed file is byte-identical to the original.

For example, given `my_file.txt`, the following files will be produced:

- `my_file.txt.pym` — the arithmetic-coded compressed stream
- `my_file.txt.bin` — the seed contexts used for parallel decompression
- `my_file_reconstructed.txt` — the recovered file

**NOTE — CUDA users:** use the `cuda_version` branch. It includes CUDA-specific optimizations like flash attention and fused kernels via `torch.compile`.

The compressor operates directly on raw bytes rather than text tokens, so it works on text files, images, archives, executables, audio, video, or any other file format.

### Config

The default configuration trains on the first 0.5 MB of the input file (for fast tests):

```
SIZE = 0.5
```

Set `SIZE = None` to train on the whole file. Increasing `SIZE` generally improves compression at the cost of longer training and compression times.

| Param | Default |
|---|---|
| `WINDOW_SIZE` | 256 |
| `STRIDE` | 128 |
| `HIDDEN_DIMS` | 128 |
| `LAYERS` | 2 |
| `BATCH_SIZE` | 64 |
| `NUM_CHUNKS` | 100 |

A CUDA GPU is strongly recommended. The code falls back to CPU automatically if CUDA is unavailable, but training and compression will be significantly slower.

For AMD users:

```
HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 main.py
```
