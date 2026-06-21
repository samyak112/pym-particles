import io
import struct
from arithmetic_coder import ArithmeticDecoder, BitInputStream
from tqdm import tqdm
from compressor import split_sizes
from data_processing import BOS,PAD,step_model_batched

SCALE = 1_000_000  

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

def decompress(model, compressed_path,stride, output_path,seed_path, window_size, num_chunks,device):
    print("Initiating Decompression")

    custom_seeds = load_seeds(seed_path, num_chunks, window_size=window_size)
    
    with open(compressed_path, 'rb') as f:
        header_base = f.read(20)
        num_bytes, num_windows, _ = struct.unpack('<Q Q I', header_base)
            
        sizes = []
        for _ in range(num_chunks):
            sizes.append(struct.unpack('<Q', f.read(8))[0])
            
        stream_bytes = []
        for s in sizes:
            stream_bytes.append(f.read(s))
            
    print(f"decompressing: {num_bytes:,} bytes, {num_windows} windows ({num_chunks} parallel chunks)")
            
    bitins   = [BitInputStream(io.BytesIO(sb)) for sb in stream_bytes]
    decoders = [ArithmeticDecoder(32, b) for b in bitins]
    
    # Get Exact Stopping Points
    sizes_w = split_sizes(num_windows, num_chunks)
    max_windows = max(sizes_w)
    half_window = window_size // 2
    
    all_bytes = [[] for _ in range(num_chunks)]
    
    # Initialize the first contexts (static seed + custom seeds)
    contexts = [make_seed(stride)] + custom_seeds

    for w_idx in tqdm(range(max_windows), desc="decompress", unit="window"):
        
        # Prefill the first 128 tokens for the current window.
        freqs_batch, past_kv, past_padding = step_model_batched(model,device=device, tokens_batch=contexts)

        # Temporary storage for the 128 bytes we are about to generate
        current_window_bytes = [[] for _ in range(num_chunks)]

        for step in range(half_window):
            decoded_tokens = []
            
            for i in range(num_chunks):
                if w_idx < sizes_w[i]:
                    byte_val = decoders[i].read(freqs_batch[i])
                    all_bytes[i].append(byte_val)
                else:
                    # Dummy value to maintain the strict batch size of 100 for the model
                    byte_val = 0 
                    
                current_window_bytes[i].append(byte_val)
                decoded_tokens.append([byte_val])

            freqs_batch, past_kv, past_padding = step_model_batched(
                model, decoded_tokens, past_kv=past_kv, past_padding=past_padding,device=device
            )

        for i in range(num_chunks):
            contexts[i] = current_window_bytes[i]

    for bitin in bitins:
        bitin.close()

    flat_bytes = []
    for chunk_arr in all_bytes:
        flat_bytes.extend(chunk_arr)
        
    #Strip out the static seed tokens (256, 257)
    valid_bytes = [b for b in flat_bytes if b < 256]

    final_bytes = valid_bytes[:num_bytes]

    with open(output_path, 'wb') as f:
        f.write(bytes(final_bytes))

    print(f"wrote → {output_path}")
    return flat_bytes