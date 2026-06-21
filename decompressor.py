import os
import io
import struct
import torch
import torch.nn.functional as F
from arithmetic_coder import SimpleFrequencyTable, ArithmeticDecoder, BitInputStream
from tqdm import tqdm
from compressor import split_sizes
from data_processing import BOS,PAD

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCALE      = 1_000_000      # float prob → integer frequency


# ── seed generation ────────────────────────────────────────────────────────────

def make_seed(stride):
    """Fixed static seed generated dynamically in RAM: [257, 256, 256 ... 256, 257]"""
    seed = [PAD] * stride
    seed[0]  = BOS
    seed[-1] = BOS
    return seed

def load_seeds(seed_filepath, num_chunks, window_size):
    """Reads exactly (N - 1) seeds from the integer seed file."""
    seeds = []
    half_window = window_size // 2
    
    with open(seed_filepath, 'rb') as f:
        for _ in range(num_chunks - 1):
            chunk_seed = []
            for _ in range(half_window):
                val = struct.unpack('<H', f.read(2))[0]
                chunk_seed.append(val)
            seeds.append(chunk_seed)
            
    return seeds


# ── probability helper ─────────────────────────────────────────────────────────

def step_model_batched(model, tokens_batch, device,past_kv=None, past_padding=None):
    """
    tokens_batch: List of N sequences. Shape (N, SeqLen).
    Returns N frequency tables.
    """
    inp = torch.tensor(tokens_batch, dtype=torch.long).to(device=device)
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits, next_kv, next_padding = model(
                inp,
                past_key_values=past_kv,
                past_key_padding_mask=past_padding
            )
            probs = torch.softmax(logits[:, -1, :].float(), dim=-1)

    probs_np = probs.cpu().numpy() * SCALE
    freqs_batch = []
    for b in range(probs_np.shape[0]):
        f = probs_np[b].astype(int).clip(1)
        freqs_batch.append(SimpleFrequencyTable(f.tolist()))

    return freqs_batch, next_kv, next_padding


# ── decompress ─────────────────────────────────────────────────────────────────

def decompress(model, compressed_path,stride, output_path,seed_path, window_size, num_chunks,device):
    print("Initiating Decompression")

    custom_seeds = load_seeds(seed_path, num_chunks, window_size=window_size)
    
    # ── 1. Read the updated 20-byte Header ──
    with open(compressed_path, 'rb') as f:
        header_base = f.read(20)
        num_bytes, num_windows, file_num_chunks = struct.unpack('<Q Q I', header_base)
        
        # Guard against mismatch
        assert file_num_chunks == num_chunks, "Chunk count mismatch in header"
        
        sizes = []
        for _ in range(num_chunks):
            sizes.append(struct.unpack('<Q', f.read(8))[0])
            
        stream_bytes = []
        for s in sizes:
            stream_bytes.append(f.read(s))
            
    print(f"decompressing: {num_bytes:,} bytes, {num_windows} windows ({num_chunks} parallel chunks)")
            
    bitins   = [BitInputStream(io.BytesIO(sb)) for sb in stream_bytes]
    decoders = [ArithmeticDecoder(32, b) for b in bitins]
    
    # ── 2. Determine Exact Stopping Points ──
    sizes_w = split_sizes(num_windows, num_chunks)
    max_windows = max(sizes_w)
    half_window = window_size // 2
    
    all_bytes = [[] for _ in range(num_chunks)]
    
    # Initialize the first contexts (static seed + custom seeds)
    contexts = [make_seed(stride)] + custom_seeds

    # ── 3. The Windowed Decoding Loop ──
    for w_idx in tqdm(range(max_windows), desc="decompress", unit="window"):
        
        # Step A: Prefill the first 128 tokens for the current window.
        # This completely resets the KV cache from the previous window.
        freqs_batch, past_kv, past_padding = step_model_batched(model,device=device, tokens_batch=contexts)

        # Temporary storage for the 128 bytes we are about to generate
        current_window_bytes = [[] for _ in range(num_chunks)]

        # Step B: Autoregressively decode the next 128 tokens
        for step in range(half_window):
            decoded_tokens = []
            
            for i in range(num_chunks):
                # Only read from the stream if this chunk hasn't finished its assigned windows
                if w_idx < sizes_w[i]:
                    byte_val = decoders[i].read(freqs_batch[i])
                    all_bytes[i].append(byte_val)
                else:
                    # Dummy value to maintain the strict batch size of 100 for the model
                    byte_val = 0 
                    
                current_window_bytes[i].append(byte_val)
                decoded_tokens.append([byte_val])

            # Step the model forward with KV caching
            freqs_batch, past_kv, past_padding = step_model_batched(
                model, decoded_tokens, past_kv=past_kv, past_padding=past_padding,device=device
            )

        # Step C: Shift the Context
        # The 128 bytes we just generated become the seed for the next window
        for i in range(num_chunks):
            contexts[i] = current_window_bytes[i]

    # Cleanup
    for bitin in bitins:
        bitin.close()

    # ── 4. Flatten and Truncate Padding ──
    flat_bytes = []
    for chunk_arr in all_bytes:
        flat_bytes.extend(chunk_arr)
        
    # 1. Strip out the static seed tokens (256, 257)
    valid_bytes = [b for b in flat_bytes if b < 256]

    # 2. Truncate to the exact original file size to remove window padding
    final_bytes = valid_bytes[:num_bytes]

    # 3. Write safely to disk
    with open(output_path, 'wb') as f:
        f.write(bytes(final_bytes))

    print(f"wrote → {output_path}")
    return flat_bytes