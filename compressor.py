import torch
from tokenizer import get_byte_ids
from data_processing import generate_windows,step_model_batched
from arithmetic_coder import BitOutputStream, ArithmeticEncoder
from tqdm import tqdm
import os
import io
import struct



def split_sizes(total, n):
    """Divide `total` windows into `n` groups as evenly as possible."""
    base = total // n
    sizes = [base] * n
    sizes[-1] += total - base * n
    return sizes

def base(size, input_file, window_size, seed_path,num_chunks):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    token_ids = get_byte_ids(chunk_path=input_file, size_mb=size)
    windows = generate_windows(token_ids, window_size, device=device)
    num_windows = len(windows)
    num_bytes = len(token_ids)

    half_window = window_size // 2

    sizes_w = split_sizes(num_windows, num_chunks)
    bounds = [0]
    custom_seeds = []

    for s in sizes_w:
        bounds.append(bounds[-1] + s)

    for c in range(num_chunks):
        if c != 0:
            first_window_of_chunk = windows[bounds[c]]
            seed = first_window_of_chunk[:half_window].cpu().tolist()
            custom_seeds.append(seed)

    with open(seed_path, 'wb') as f:
        for seed in custom_seeds:
            for token in seed:
                f.write(struct.pack('<H', token))
    print(f"Generated {num_chunks} boundary seeds → {seed_path}")

    return windows, num_bytes, num_windows, bounds


def compress(size, input_file, compressed_file, seed_path, window_size, model,num_chunks):
    print('Initiating Compression (Strict AR Mode)')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    windows, num_bytes, num_windows, bounds = base(size, input_file=input_file, window_size=window_size, seed_path=seed_path,num_chunks=num_chunks)

    half_window = window_size // 2
    sizes_w = split_sizes(num_windows, num_chunks)
    max_windows = max(sizes_w)

    streams  = [io.BytesIO() for _ in range(num_chunks)]
    bitouts  = [BitOutputStream(s) for s in streams]
    encoders = [ArithmeticEncoder(32, b) for b in bitouts]

    with torch.no_grad():
        # Iterate window by window, keeping all chunks perfectly parallel
        for w_idx in tqdm(range(max_windows), desc="compress", unit="window"):
            contexts = []
            targets_matrix = []

            # 1. Setup the exact same contexts the decompressor will see
            for i in range(num_chunks):
                if w_idx < sizes_w[i]:
                    win = windows[bounds[i] + w_idx].tolist()
                    contexts.append(win[:half_window])     # First 128 (Seed)
                    targets_matrix.append(win[half_window:]) # Next 128 (To Predict)
                else:
                    # Dummy values to maintain strict batch size of 100
                    contexts.append([0] * half_window)
                    targets_matrix.append([0] * half_window)

            # Step A: Prefill to reset KV cache (Matches Decompressor)
            freqs_batch, past_kv, past_padding = step_model_batched(model, contexts, device)

            # Step B: Autoregressively step through the targets
            for step in range(half_window):
                current_tokens = []

                for i in range(num_chunks):
                    target_token = targets_matrix[i][step]

                    # Write ground truth to stream IF chunk is active
                    if w_idx < sizes_w[i]:
                        encoders[i].write(freqs_batch[i], int(target_token))

                    current_tokens.append([target_token])

                # Feed the ground truth target back into the model to get next frequencies
                # (Skip on the very last token, as we don't need its prediction)
                if step < half_window - 1:
                    freqs_batch, past_kv, past_padding = step_model_batched(
                        model, current_tokens, device, past_kv=past_kv, past_padding=past_padding
                    )

    for enc in encoders:
        enc.finish()
    for s in streams:
        s.close = lambda: None
    for bitout in bitouts:
        bitout.close()

    compressed_chunks = [s.getvalue() for s in streams]
    sizes = [len(chunk) for chunk in compressed_chunks]

    with open(compressed_file, 'wb') as f:
        f.write(struct.pack('<Q Q I', num_bytes, num_windows, num_chunks))
        for s in sizes:
            f.write(struct.pack('<Q', s))
        for chunk in compressed_chunks:
            f.write(chunk)

    original_mb = num_bytes / 1024 / 1024
    compressed_mb = os.path.getsize(compressed_file) / 1024 / 1024
    bits_per_byte = compressed_mb * 8 / (original_mb if original_mb > 0 else 1)
    
    print(f"original     : {original_mb:.3f} MB")
    print(f"compressed   : {compressed_mb:.3f} MB")
    print(f"ratio        : {original_mb / compressed_mb:.2f}x")
    print(f"bits/byte    : {bits_per_byte:.3f}")