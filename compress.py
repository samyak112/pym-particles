import torch
from tokenizer import get_byte_ids
from data_processing import generate_windows,load_bitmap,find_run,build_ends_index,build_runs_list,decode_leb128
from pym_transformer import PymTransformer
from arithmetic_coder import SimpleFrequencyTable, BitOutputStream, ArithmeticEncoder
from tqdm import tqdm
import os
import time

INPUT_FILE      = 'slice_100mb.txt'
COMPRESSED_FILE = 'slice_100mb_current12.pym'
RECONSTRUCTED   = 'slice_100mb.reconstructed.txt'
SEED_FILE       = 'seeds.bin'
MODEL_PATH      = 'models/pym_particles_enwik_latest.pt'
VOCAB_SIZE      = 258
WINDOW_SIZE = 256
HIDDEN_DIMS = 128
BATCH_SIZE = 64
TEMPERATURE = 2.0



SCALE = 1_000_000

def base(size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Extraction Device: {device}")

    token_ids = get_byte_ids(chunk_path=INPUT_FILE,size=size)
    windows = generate_windows(token_ids, WINDOW_SIZE, device=device)
    num_windows = len(windows)

    model = PymTransformer(vocab_size=VOCAB_SIZE, hidden_dim=128, num_layers=2, sequence_length=WINDOW_SIZE).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    return model, windows,num_windows


# def compress(size=1):

#     model, windows, num_windows = base(size)

#     bit_out = BitOutputStream(open(f"{COMPRESSED_FILE}", "wb"))
#     enc = ArithmeticEncoder(32, bit_out)
#     half_window = WINDOW_SIZE // 2
#     logit_start_idx  = half_window - 1
#     target_start_idx = logit_start_idx + 1

#     ACCUMULATE_N = 1  # accumulate N batches before writing

#     accumulated_freqs   = []
#     accumulated_targets = []

#     with torch.no_grad():
#         for batch_start in tqdm(range(0, num_windows, BATCH_SIZE), desc="compressing"):

#             inp_b = windows[batch_start : batch_start + BATCH_SIZE]

#             with torch.autocast(device_type='cuda', dtype=torch.float16):
#                 logits, _, _ = model(inp_b)

#             logits_trimmed  = logits[:, logit_start_idx:-1, :].float()
#             targets_trimmed = inp_b[:, target_start_idx:]

#             accumulated_freqs.append(torch.softmax(logits_trimmed.reshape(-1, VOCAB_SIZE), dim=-1))
#             accumulated_targets.append(targets_trimmed.reshape(-1))

#             if len(accumulated_freqs) >= ACCUMULATE_N:
#                 all_freqs   = (torch.cat(accumulated_freqs,   dim=0) * 1_000_000).cpu().numpy().astype(int).clip(1)
#                 all_targets = torch.cat(accumulated_targets, dim=0).cpu().numpy()

#                 for i in range(len(all_targets)):
#                     enc.write(SimpleFrequencyTable(all_freqs[i].tolist()), int(all_targets[i]))

#                 accumulated_freqs   = []
#                 accumulated_targets = []

#         # flush remaining
#         if accumulated_freqs:
#             all_freqs   = (torch.cat(accumulated_freqs,   dim=0) * 1_000_000).cpu().numpy().astype(int).clip(1)
#             all_targets = torch.cat(accumulated_targets, dim=0).cpu().numpy()

#             for i in range(len(all_targets)):
#                 enc.write(SimpleFrequencyTable(all_freqs[i].tolist()), int(all_targets[i]))

#     enc.finish()
#     bit_out.close()

#     TEST_BYTES = size * 1024 * 1024

#     with open(INPUT_FILE, 'rb') as f:
#         byte_ids = list(f.read()[:TEST_BYTES])

#     num_bytes   = len(byte_ids)
#     original_mb = num_bytes / 1024 / 1024

#     compressed_mb = os.path.getsize(COMPRESSED_FILE) / 1024 / 1024
#     bits_per_byte = compressed_mb * 8 / (original_mb if original_mb > 0 else 1)
#     print(f"original     : {original_mb:.3f} MB")
#     print(f"compressed   : {compressed_mb:.3f} MB")
#     print(f"ratio        : {original_mb / compressed_mb:.2f}x")
#     print(f"bits/byte    : {bits_per_byte:.3f}")


def load_hit_runs(path):
    import numpy as np

    with open(path, 'rb') as f:
        data = f.read()

    values = []
    offset = 0
    while offset < len(data):
        val, offset = decode_leb128(data, offset)
        values.append(val)

    if not values:
        return np.array([], dtype=np.int64)

    # values = [end0, start1, end1, start2, end2, ...]
    # hit ranges: (0, end0), (start1, end1), (start2, end2), ...
    hit_ranges = [(0, values[0])]
    for i in range(1, len(values) - 1, 2):
        hit_ranges.append((values[i], values[i + 1]))

    # miss positions are gaps between consecutive hit ranges
    chunks = []
    for i in range(len(hit_ranges) - 1):
        gap_start = hit_ranges[i][1]
        gap_end   = hit_ranges[i + 1][0]
        if gap_end > gap_start:
            chunks.append(np.arange(gap_start, gap_end, dtype=np.int64))

    return np.concatenate(chunks) if chunks else np.array([], dtype=np.int64)

def compress(size=1, top_k=5):

    model, windows, num_windows = base(size)

    bit_out = BitOutputStream(open(f"{COMPRESSED_FILE}", "wb"))
    enc     = ArithmeticEncoder(32, bit_out)

    half_window      = WINDOW_SIZE // 2
    logit_start_idx  = half_window - 1
    target_start_idx = logit_start_idx + 1

    with torch.no_grad():
        for batch_start in tqdm(range(0, num_windows, BATCH_SIZE), desc="compressing"):

            inp_b = windows[batch_start : batch_start + BATCH_SIZE]

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _, _ = model(inp_b)

            logits_trimmed  = logits[:, logit_start_idx:-1, :].float()
            targets_trimmed = inp_b[:, target_start_idx:]

            flat_logits  = logits_trimmed.reshape(-1, VOCAB_SIZE)
            flat_targets = targets_trimmed.reshape(-1)
            N            = flat_logits.shape[0]

            topk_logits, topk_indices = flat_logits.topk(top_k, dim=-1)

            freqs = (
                torch.softmax(topk_logits, dim=-1) * SCALE
            ).cpu().numpy().astype(int).clip(1)

            topk_indices_np = topk_indices.cpu().numpy()
            targets_np = flat_targets.cpu().numpy()

            for i in range(N):
                surviving_ids = topk_indices_np[i]
                target = int(targets_np[i])

                match = (surviving_ids == target)

                if not match.any():
                    # target not in top-k
                    pass
                else:
                    enc.write(
                        SimpleFrequencyTable(freqs[i].tolist()),
                        int(match.argmax())
                    )

    enc.finish()
    bit_out.close()

    TEST_BYTES    = size * 1024 * 1024
    original_mb   = TEST_BYTES / 1024 / 1024
    compressed_mb = os.path.getsize(COMPRESSED_FILE) / 1024 / 1024
    bits_per_byte = (compressed_mb * 8 * 1024 * 1024) / TEST_BYTES

    print(f"original     : {original_mb:.3f} MB")
    print(f"compressed   : {compressed_mb:.3f} MB")
    print(f"ratio        : {original_mb / compressed_mb:.2f}x")
    print(f"bits/byte    : {bits_per_byte:.3f}")
    
compress()

