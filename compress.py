import os
import io
import struct
import torch
import torch.nn.functional as F
import numpy as np
from arithmetic_coder import SimpleFrequencyTable, BitOutputStream, ArithmeticDecoder, ArithmeticEncoder, BitInputStream
from pym_transformer import PymTransformer
from tqdm import tqdm

# ── constants ──────────────────────────────────────────────────────────────────

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCALE      = 1_000_000      # float prob → integer frequency
BLOCK_SIZE = 128            # bytes generated per block (Matches 256-sequence limit)
SEED_LEN   = 128            # length of static seed
TOK_A      = 257            # first and last token of seed
TOK_B      = 256            # middle tokens of seed
NUM_CHUNKS = 100              # N parallel streams for batched autoregression


# ── seed generation ────────────────────────────────────────────────────────────

def apply_bitmap(f, bitmask):
    for b in range(256):
        if (bitmask >> b) & 1:
            f[b] = 0
    return f


def make_seed():
    """Fixed static seed generated dynamically in RAM: [257, 256, 256 ... 256, 257]"""
    seed = [TOK_B] * SEED_LEN
    seed[0]  = TOK_A
    seed[-1] = TOK_A
    return seed

def generate_seed_file(input_filepath, seed_filepath, num_chunks=NUM_CHUNKS, seed_len=SEED_LEN):
    """
    Partitions the input file and extracts the 128 bytes immediately preceding 
    each boundary. Stores exactly (N - 1) seeds in the seed file.
    """
    with open(input_filepath, 'rb') as f:
        file_bytes = f.read()
        
    num_bytes = len(file_bytes)
    chunk_size = (num_bytes + num_chunks - 1) // num_chunks
    
    seeds = []
    
    # We skip Chunk 0 because it uses the dynamic inline static seed.
    # We collect seeds for Chunks 1 to (N - 1) using the 128 bytes preceding the boundary.
    for i in range(1, num_chunks):
        boundary = i * chunk_size
        
        if boundary > num_bytes:
            seeds.append([TOK_B] * seed_len)
        else:
            start = max(0, boundary - seed_len)
            chunk_seed = list(file_bytes[start:boundary])
            
            while len(chunk_seed) < seed_len:
                chunk_seed.insert(0, TOK_B)
                
            seeds.append(chunk_seed)
            
    # Write to disk as 16-bit unsigned integers to cleanly support out-of-byte tokens
    with open(seed_filepath, 'wb') as f:
        for s in seeds:
            for token in s:
                f.write(struct.pack('<H', token))
                
    print(f"Generated {num_chunks - 1} boundary seeds → {seed_filepath}")
    return seeds

def load_custom_seeds(seed_filepath, num_chunks=NUM_CHUNKS, seed_len=SEED_LEN):
    """Reads exactly (N - 1) seeds from the integer seed file."""
    seeds = []
    with open(seed_filepath, 'rb') as f:
        for _ in range(num_chunks - 1):
            chunk_seed = []
            for _ in range(seed_len):
                val = struct.unpack('<H', f.read(2))[0]
                chunk_seed.append(val)
            seeds.append(chunk_seed)
    return seeds


# ── model ──────────────────────────────────────────────────────────────────────

def convert_mha_state_dict(state_dict):
    """
    Translates PyTorch native MultiheadAttention weights into the split 
    Q, K, V format expected by the custom CausalSelfAttention module.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'attn.in_proj_weight' in key:
            q_w, k_w, v_w = value.chunk(3, dim=0)
            prefix = key.replace('attn.in_proj_weight', 'attn.')
            new_state_dict[prefix + 'q_proj.weight'] = q_w
            new_state_dict[prefix + 'k_proj.weight'] = k_w
            new_state_dict[prefix + 'v_proj.weight'] = v_w
        elif 'attn.in_proj_bias' in key:
            q_b, k_b, v_b = value.chunk(3, dim=0)
            prefix = key.replace('attn.in_proj_bias', 'attn.')
            new_state_dict[prefix + 'q_proj.bias'] = q_b
            new_state_dict[prefix + 'k_proj.bias'] = k_b
            new_state_dict[prefix + 'v_proj.bias'] = v_b
        elif 'attn.out_proj.weight' in key:
            new_key = key.replace('attn.out_proj.weight', 'attn.o_proj.weight')
            new_state_dict[new_key] = value
        elif 'attn.out_proj.bias' in key:
            new_key = key.replace('attn.out_proj.bias', 'attn.o_proj.bias')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def load_model(model_path, vocab_size=258, hidden_dim=128, num_layers=2, sequence_length=256):
    model = PymTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        sequence_length=sequence_length
    ).to(DEVICE)
    # old_state_dict = torch.load(model_path, map_location=DEVICE)
    # new_state_dict = convert_mha_state_dict(old_state_dict)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def load_bitmap(bitmap_path):
    runs = []
    with open(bitmap_path, 'rb') as f:
        num_runs = struct.unpack('>I', f.read(4))[0]
        for _ in range(num_runs):
            start, end = struct.unpack('>II', f.read(8))
            bitmask = int.from_bytes(f.read(32), byteorder='big')
            runs.append((start, end, bitmask))
    return runs

def get_bitmask_for_pos(bitmap_runs, token_pos):
    """Binary search for which run this token_pos falls in."""
    lo, hi = 0, len(bitmap_runs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end, bitmask = bitmap_runs[mid]
        if token_pos < start:
            hi = mid - 1
        elif token_pos > end:
            lo = mid + 1
        else:
            return bitmask
    return None


# ── probability helper ─────────────────────────────────────────────────────────

def step_model_batched(model, tokens_batch, bitmap_runs=None, global_positions=None, past_kv=None, past_padding=None):
    inp = torch.tensor(tokens_batch, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        logits, next_kv, next_padding = model(
            inp,
            past_key_values=past_kv,
            past_key_padding_mask=past_padding
        )
        

    if bitmap_runs is not None and global_positions is not None:
        for b in range(logits.shape[0]):
            bitmask = get_bitmask_for_pos(bitmap_runs, global_positions[b])
            
            if bitmask is not None:
                forbidden = [tok for tok in range(256) if (bitmask >> tok) & 1]
                logits[b, -1, forbidden] = float('-inf')

    with torch.no_grad():
        probs = F.softmax(logits[:, -1, :], dim=-1).float()

    probs_np = probs.cpu().numpy() * SCALE
    
    freqs_batch = []
    for b in range(probs_np.shape[0]):
        f = probs_np[b] * SCALE
        print(f"chunk {b}, global_post:{global_positions}")  # add this
        f = f.astype(int)
        f = np.where(f > 0, np.maximum(f, 1), 0)
        freqs_batch.append(SimpleFrequencyTable(f.tolist()))

    return freqs_batch, next_kv, next_padding

# ── compress ───────────────────────────────────────────────────────────────────

def compress(model, byte_ids, custom_seeds, output_path, vocab_size=258, bitmap_runs=None):

    num_bytes = len(byte_ids)
    
    # Calculate uniform chunk size and pad the input so it perfectly divides by N
    chunk_size = (num_bytes + NUM_CHUNKS - 1) // NUM_CHUNKS
    pad_amount = (chunk_size * NUM_CHUNKS) - num_bytes
    byte_ids.extend([0] * pad_amount)
    
    num_blocks = (chunk_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    print(f"compressing  : {num_bytes:,} bytes ({NUM_CHUNKS} parallel chunks)")

    # Initialize N separate memory streams and encoders
    streams  = [io.BytesIO() for _ in range(NUM_CHUNKS)]
    bitouts  = [BitOutputStream(s) for s in streams]
    encoders = [ArithmeticEncoder(32, b) for b in bitouts]

    for block_idx in tqdm(range(num_blocks), desc="compress", unit="block"):
        
        # ── Setup Batched Context ──
        if block_idx == 0:
            # Prepend the dynamic inline static seed to the N-1 loaded seeds
            context_batch = [make_seed()] + custom_seeds
        else:
            context_batch = []
            for i in range(NUM_CHUNKS):
                start = (i * chunk_size) + ((block_idx - 1) * BLOCK_SIZE)
                end   = start + BLOCK_SIZE
                context_batch.append(list(byte_ids[start:end]))

                while len(context_batch[-1]) < SEED_LEN:
                    context_batch[-1].append(TOK_B)

        # ── Prefill Phase (Batched) ──
        freqs_batch, past_kv, past_padding = step_model_batched(model, context_batch)

        # ── Decoding Phase (Batched) ──
        block_start = block_idx * BLOCK_SIZE
        block_end   = min(block_start + BLOCK_SIZE, chunk_size)

        for pos in range(block_start, block_end):
            true_bytes = []
            global_positions = []
            for i in range(NUM_CHUNKS):
                global_pos = (i * chunk_size) + pos
                true_byte = byte_ids[global_pos]
                true_bytes.append([true_byte])
                global_positions.append(global_pos)

                try:
                    encoders[i].write(freqs_batch[i], true_byte)
                except Exception:
                    print(f"chunk {i}, pos {pos}, global_pos {global_pos}, true_byte {true_byte}")
                    import sys
                    sys.exit()

            freqs_batch, past_kv, past_padding = step_model_batched(
                model, true_bytes,
                bitmap_runs=bitmap_runs,
                global_positions=global_positions,
                past_kv=past_kv,
                past_padding=past_padding
            )

    # ── Finalize Streams (Flush Race-Condition Fix) ──
    # 1. Tell all encoders to write their final mathematical intervals to the wrappers
    for enc in encoders:
        enc.finish()

    # 2. Monkey-patch the memory streams so closing wrappers won't destroy the data strings
    for s in streams:
        s.close = lambda: None

    # 3. Force the wrappers to close, which safely flushes any partial trailing bits
    for bitout in bitouts:
        bitout.close()

    # 4. Extract fully flushed, complete binary chunks from RAM
    compressed_chunks = [s.getvalue() for s in streams]
    sizes = [len(chunk) for chunk in compressed_chunks]

    # ── Write Multiplexed File with Header ──
    with open(output_path, 'wb') as f:
        # Header: <Total Bytes (8B)> <Num Chunks (4B)> <Size 1 (8B)> ... <Size N (8B)>
        f.write(struct.pack('<Q I', num_bytes, NUM_CHUNKS))
        for s in sizes:
            f.write(struct.pack('<Q', s))
            
        # Write binary chunk blocks directly
        for chunk in compressed_chunks:
            f.write(chunk)

    print(f"wrote → {output_path}")


# ── decompress ─────────────────────────────────────────────────────────────────

def decompress(model, compressed_path, custom_seeds, output_path, vocab_size=258, bitmap_runs=None):

    print(f"reading      : {compressed_path}")
    
    with open(compressed_path, 'rb') as f:
        header_base = f.read(12)
        num_bytes, num_chunks = struct.unpack('<Q I', header_base)
        
        sizes = []
        for _ in range(num_chunks):
            sizes.append(struct.unpack('<Q', f.read(8))[0])
            
        stream_bytes = []
        for s in sizes:
            stream_bytes.append(f.read(s))
            
    print(f"decompressing: {num_bytes:,} bytes ({num_chunks} parallel chunks)")
            
    bitins   = [BitInputStream(io.BytesIO(sb)) for sb in stream_bytes]
    decoders = [ArithmeticDecoder(32, b) for b in bitins]
    
    chunk_size = (num_bytes + num_chunks - 1) // num_chunks
    num_blocks = (chunk_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    all_bytes = [[] for _ in range(num_chunks)]

    for block_idx in tqdm(range(num_blocks), desc="decompress", unit="block"):
        
        # ── Setup Batched Context ──
        if block_idx == 0:
            # Mirror the compressor exactly
            context_batch = [make_seed()] + custom_seeds
        else:
            context_batch = []
            for i in range(num_chunks):
                prev_start = (block_idx - 1) * BLOCK_SIZE
                prev_end   = prev_start + BLOCK_SIZE
                context_chunk = list(all_bytes[i][prev_start:prev_end])
                
                while len(context_chunk) < SEED_LEN:
                    context_chunk.append(TOK_B)
                context_batch.append(context_chunk)

        # ── Prefill Phase (Batched) ──
        global_positions = [(i * chunk_size) + pos for i in range(num_chunks)]
        freqs_batch, past_kv, past_padding = step_model_batched(
            model, decoded_bytes,
            bitmap_runs=bitmap_runs,
            global_positions=global_positions,
            past_kv=past_kv,
            past_padding=past_padding
        )

        # ── Decoding Phase (Batched) ──
        block_start = block_idx * BLOCK_SIZE
        block_end   = min(block_start + BLOCK_SIZE, chunk_size)

        for pos in range(block_start, block_end):
            decoded_bytes = []
            global_positions = []
            for i in range(num_chunks):
                byte_val = decoders[i].read(freqs_batch[i])
                all_bytes[i].append(byte_val)
                decoded_bytes.append([byte_val])
                global_positions.append((i * chunk_size) + pos)

            freqs_batch, past_kv, past_padding = step_model_batched(
                model, decoded_bytes,
                bitmap_runs=bitmap_runs,
                global_positions=global_positions,
                past_kv=past_kv,
                past_padding=past_padding
            )

    for bitin in bitins:
        bitin.close()

    # ── Flatten and Truncate Padding ──
    flat_bytes = []
    for chunk_arr in all_bytes:
        flat_bytes.extend(chunk_arr)
        
    flat_bytes = flat_bytes[:num_bytes]

    with open(output_path, 'wb') as f:
        f.write(bytes(flat_bytes))

    print(f"wrote → {output_path}")
    return flat_bytes


# ── verify ─────────────────────────────────────────────────────────────────────

def verify(original_path, reconstructed_path, test_bytes):
    with open(original_path, 'rb') as f:
        original = f.read()[:test_bytes]
    with open(reconstructed_path, 'rb') as f:
        reconstructed = f.read()

    if original == reconstructed:
        print("lossless      : TRUE ✓")
        print("\nparallel streaming architecture verified.")
        return True
    else:
        for i, (a, b) in enumerate(zip(original, reconstructed)):
            if a != b:
                print(f"lossless      : FALSE ✗  first mismatch at byte {i:,}")
                print(f"  expected {a} got {b}")
                break
        return False


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    INPUT_FILE      = 'slice_100mb.txt'
    COMPRESSED_FILE = 'slice_100mb.pym'
    RECONSTRUCTED   = 'slice_100mb.reconstructed.txt'
    SEED_FILE       = 'seeds.bin'
    MODEL_PATH      = 'models/pym_particles.pt'
    VOCAB_SIZE      = 258

    bitmap_runs = load_bitmap(INPUT_FILE + '.bitmap')


    # Set up a target dataset size slice for validation
    TEST_BYTES = 1 * 1024 * 1024

    print("loading model...")
    model = load_model(MODEL_PATH, vocab_size=VOCAB_SIZE)

    with open(INPUT_FILE, 'rb') as f:
        byte_ids = list(f.read()[:TEST_BYTES])

    num_bytes   = len(byte_ids)
    original_mb = num_bytes / 1024 / 1024
    print(f"testing on   : {original_mb:.3f} MB  ({num_bytes:,} bytes)")

    print("extracting boundary seeds...")
    temp_subset = 'temp_input.bin'
    with open(temp_subset, 'wb') as f:
        f.write(bytes(byte_ids))
        
    # Extracts exactly N - 1 seeds from the file's segment horizons
    generate_seed_file(temp_subset, SEED_FILE, NUM_CHUNKS, SEED_LEN)
    os.remove(temp_subset)

    # Loads the N - 1 custom boundary seeds
    custom_seeds = load_custom_seeds(SEED_FILE, NUM_CHUNKS, SEED_LEN)

    # ── compress ──
    compress(model, byte_ids.copy(), custom_seeds, COMPRESSED_FILE, VOCAB_SIZE, bitmap_runs=bitmap_runs)


    compressed_mb = os.path.getsize(COMPRESSED_FILE) / 1024 / 1024
    bits_per_byte = compressed_mb * 8 / (original_mb if original_mb > 0 else 1)
    print(f"original     : {original_mb:.3f} MB")
    print(f"compressed   : {compressed_mb:.3f} MB")
    print(f"ratio        : {original_mb / compressed_mb:.2f}x")
    print(f"bits/byte    : {bits_per_byte:.3f}")

    # ── decompress ──
    decompress(model, COMPRESSED_FILE, custom_seeds, RECONSTRUCTED, VOCAB_SIZE, bitmap_runs=bitmap_runs)


    # ── verify ──
    verify(INPUT_FILE, RECONSTRUCTED, TEST_BYTES)