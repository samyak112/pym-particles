import torch
from tokenizer import get_byte_ids
import struct
import bisect
from pym_transformer import PymTransformer
import torch.nn as nn
import math
import time
import os

PAD = 256
BOS = 257

def generate_windows(
    token_ids: list[int],
    window_size: int,
    device
) -> list:

    print("creating windows...")
    tokens = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=device,
    )

    half_window = window_size//2

    # Adding 256 PADs because 256th PAD gets replaced by BOS in next line
    prefix = torch.full(
        (half_window,),
        PAD,
        dtype=torch.long,
        device=device,
    )
    prefix[-1] = BOS
    prefix[0] = BOS

    first_window = torch.cat([prefix, tokens[:half_window]])

    windows = [first_window]

    for i in range(half_window,len(tokens),half_window):
        new_chunk = tokens[i:i+half_window]
        older_chunk = windows[-1][half_window:]

        window = torch.cat([older_chunk, new_chunk])

        if window.numel() < window_size:
            pad_len = window_size - window.numel()
            pad = torch.full((pad_len,), PAD, dtype=torch.long, device=device)
            window = torch.cat([window, pad])
        else:
            window = window[:window_size]

        windows.append(window)

    print(len(windows))

    return torch.stack(windows, dim=0)

def make_bitmap(file_path, min_absent=170):
    with open(file_path, 'rb') as f:
        data = f.read()
    
    tokens = list(data)
    N = len(tokens)
    
    runs = []
    start = 0
    ever_seen = set()
    
    for i in range(N):
        ever_seen.add(tokens[i])
        absent = 256 - len(ever_seen)

        if absent < min_absent:
            if i > start:
                # snapshot absent set BEFORE this token corrupted it
                # ever_seen currently includes tokens[i], so remove it for the closed run
                closed_seen = ever_seen - {tokens[i]}
                absent_set = set(range(256)) - closed_seen
                runs.append((start, i - 1, absent_set))
            start = i
            ever_seen = {tokens[i]}
    
    if start < N:
        absent_set = set(range(256)) - ever_seen
        runs.append((start, N - 1, absent_set))

    # write binary
    out_path = file_path + '.bitmap'
    with open(out_path, 'wb') as f:
        # header: total runs as 4 byte int
        f.write(struct.pack('>I', len(runs)))
        for s, e, absent_set in runs:
            # start and end as 4 byte ints
            f.write(struct.pack('>II', s, e))
            # 256 bit bitmap as 32 bytes
            bitmask = 0
            for b in absent_set:
                bitmask |= (1 << b)
            f.write(bitmask.to_bytes(32, byteorder='big'))

    run_lengths = [e - s + 1 for s, e, _ in runs]
    size_kb = (len(runs) * 40) / 1024
    print(f"File          : {file_path}")
    print(f"Total tokens  : {N:,}")
    print(f"Total runs    : {len(runs):,}")
    print(f"Avg run length: {sum(run_lengths) / len(run_lengths):.1f}")
    print(f"Min run length: {min(run_lengths)}")
    print(f"Max run length: {max(run_lengths)}")
    print(f"Bitmap size   : {size_kb:.2f} KB")
    print(f"Saved → {out_path}")

    return runs

def load_bitmap(bitmap_path):
    runs = {}
    with open(bitmap_path, 'rb') as f:
        num_runs = struct.unpack('>I', f.read(4))[0]
        for _ in range(num_runs):
            start, end = struct.unpack('>II', f.read(8))
            bitmask = int.from_bytes(f.read(32), byteorder='big')
            bits = [(bitmask >> i) & 1 for i in range(256)]
            mask_tensor = torch.tensor(bits, dtype=torch.bool)
            runs[end] = (start, mask_tensor)
    return runs

def find_run(bitmap_runs, ends, global_pos):
    idx = bisect.bisect_left(ends, global_pos)
    if idx >= len(ends):
        return None
    end = ends[idx]
    start, mask = bitmap_runs[end]
    if start <= global_pos <= end:
        return mask
    return None

def build_ends_index(bitmap_runs):
    # bitmap_runs: {end: (start, mask_tensor)}
    ends = sorted(bitmap_runs.keys())
    return ends  # sorted list for bisect

def build_runs_list(bitmap_runs, ends):
    return [(bitmap_runs[e][0], e, bitmap_runs[e][1]) for e in ends]

def get_byte_at(file_path, index):
    with open(file_path, 'rb') as f:
        f.seek(index)
        return f.read(1)[0]
    
def get_high_entropy_bitmap(chunk_path, model_path, window_size=256, vocab_size=258, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    token_ids = get_byte_ids(chunk_path=chunk_path)
    windows = generate_windows(token_ids, window_size, device=device)
    num_windows = len(windows)

    model = PymTransformer(vocab_size=vocab_size, hidden_dim=128, num_layers=2, sequence_length=window_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='none')
    bitmap = torch.zeros(num_windows, dtype=torch.bool)
    num_batches = math.ceil(num_windows / batch_size)

    maps = {}

    with torch.no_grad():
        for batch_idx, batch_start in enumerate(range(0, num_windows, batch_size)):
            t0 = time.time()
            inp_b = windows[batch_start : batch_start + batch_size]
            current_batch_size = inp_b.shape[0]

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _, _ = model(inp_b)
                logit_start_idx = window_size // 2 - 1
                logits_trimmed  = logits[:, logit_start_idx:-1, :].reshape(-1, vocab_size)
                target_start_idx = logit_start_idx + 1
                targets_trimmed  = inp_b[:, target_start_idx:].reshape(-1)
                loss_array       = criterion(logits_trimmed, targets_trimmed)

            bits_array      = loss_array.float() / math.log(2)
            predict_len     = window_size - target_start_idx
            avg_bits        = bits_array.reshape(current_batch_size, predict_len).mean(dim=1)

            if(avg_bits>0.7):
                if(avg_bits in maps):
                    maps[avg_bits]+=1
                else:
                    maps[avg_bits] = 0

            bitmap[batch_start : batch_start + current_batch_size] = avg_bits > 0.7
            elapsed = time.time() - t0
            print(f"batch {batch_idx+1}/{num_batches}  [{batch_start}:{batch_start+current_batch_size}]  {elapsed*1000:.1f}ms")
    print(maps)

    torch.save(bitmap, 'high_entropy_bitmap.pt')


    return bitmap  # bool tensor, shape (num_windows,), True = high entropy


def encode_leb128(value):
    result = []
    while True:
        byte = value & 0x7F
        value >>= 7
        if value != 0:
            byte |= 0x80
        result.append(byte)
        if value == 0:
            break
    return bytes(result)

def decode_leb128(data, offset):
    result, shift = 0, 0
    while True:
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            break
        shift += 7
    return result, offset

def get_entropy_concentration_stats(chunk_path, model_path, window_size=256, vocab_size=258, batch_size=64, top_k=2, output_path='miss_runs.bin', bitmap_path='miss_bitmap.bin'):
    import math, numpy as np
    import torch
    from itertools import groupby
    from tqdm import tqdm  # Uses a single updating progress bar instead of thousands of lines

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    token_ids   = get_byte_ids(chunk_path=chunk_path)
    windows     = generate_windows(token_ids, window_size, device=device)
    num_windows = len(windows)

    model = PymTransformer(vocab_size=vocab_size, hidden_dim=128, num_layers=2, sequence_length=window_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    logit_start_idx  = window_size // 2 - 1
    target_start_idx = logit_start_idx + 1
    pred_seq_len     = window_size - target_start_idx

    miss_flags  = []
    
    # Trackers for the global average across all batches
    misses_per_position = torch.zeros(pred_seq_len, dtype=torch.long)
    counts_per_position = torch.zeros(pred_seq_len, dtype=torch.long)

    with torch.no_grad():
        # Added tqdm here so you see a single, self-updating progress bar line
        for batch_start in tqdm(range(0, num_windows, batch_size), desc="Analyzing positions"):
            inp_b              = windows[batch_start : batch_start + batch_size]
            current_batch_size = inp_b.shape[0]

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _, _ = model(inp_b)

            logits_trimmed  = logits[:, logit_start_idx:-1, :].float()
            targets_trimmed = inp_b[:, target_start_idx:]

            # Spatial Hit/Miss Detection
            topk       = logits_trimmed.topk(top_k, dim=-1).indices
            target_exp = targets_trimmed.unsqueeze(-1).expand_as(topk)
            in_topk    = (topk == target_exp).any(dim=-1)
            not_in_topk = ~in_topk 
            
            # Accumulate globally
            misses_per_position += not_in_topk.sum(dim=0).cpu()
            counts_per_position += current_batch_size
            miss_flags.append(not_in_topk.reshape(-1).cpu())

    miss_seq = torch.cat(miss_flags)
    miss_list = miss_seq.tolist()

    # Save Bitmap
    packed_bitmap = np.packbits(miss_seq.numpy().astype(np.uint8))
    with open(bitmap_path, 'wb') as f:
        f.write(packed_bitmap.tobytes())

    # RLE Calculations
    runs = [(key, sum(1 for _ in group)) for key, group in groupby(miss_list)]
    miss_ranges = []
    pos = 0
    for flag, length in runs:
        if flag:
            miss_ranges.append((pos, pos + length))
        pos += length

    delta_values = []
    for i, (start, end) in enumerate(miss_ranges):
        delta_values.append(end)
        if i + 1 < len(miss_ranges):
            delta_values.append(miss_ranges[i + 1][0])

    # Final Summary Stats
    total_positions = len(miss_list)
    total_misses    = int(miss_seq.sum().item())
    total_hits      = total_positions - total_misses

    print("\n" + "=" * 50)
    print(f"Total Evaluated Tokens     : {total_positions}")
    print(f"Overall Hit Rate (Top-2)   : {100 * total_hits / total_positions:.2f}%")
    print(f"Overall Miss Rate (Top-2)  : {100 * total_misses / total_positions:.2f}%")
    print("=" * 50)

    # --- THE SINGLE REPORT: Average Across All Batches ---
    print("=== Average Miss Rate Per Sequence Position ===")
    position_miss_rates = (misses_per_position.float() / counts_per_position.float()).numpy()
    
    row_str = ""
    for i, miss_rate in enumerate(position_miss_rates):
        actual_pos = target_start_idx + i
        row_str += f"  Pos {actual_pos:3d}: {miss_rate*100:5.2f}% |"
        # Print in a clean 4-column grid to keep terminal output compact
        if (i + 1) % 4 == 0 or (i + 1) == len(position_miss_rates):
            print(row_str)
            row_str = ""
    print("=" * 50)

    with open(output_path, 'wb') as f:
        f.write(bytearray([int(b) for b in delta_values])) 

    return delta_values

def load_failure_bitmap(path):
    with open(path, 'rb') as f:
        num_failures = struct.unpack('>I', f.read(4))[0]
        failure_set = set()
        prev = 0
        for _ in range(num_failures):
            b = struct.unpack('B', f.read(1))[0]
            if b == 255:
                delta = struct.unpack('>I', f.read(4))[0]
            else:
                delta = b
            prev += delta
            failure_set.add(prev)
    return failure_set

def find_redundant_chunks():
    test = {}

    print('test')

    with open("test.txt", "rb") as f:
        while chunk := f.read(128):
            # process chunk
            if chunk in test:
                test[chunk] += 1
            else:
                test[chunk] = 1

    y = 0

    print(len(test))

    for key,value in test.items():
        if(value > 1):
            print(value)


if __name__ == '__main__':
    # # custom_mask = generate_custom_mask(8, 8)
    # import torch

    # data = torch.load('corrupted_batch_item_11.pt') # Replace X with the actual number

    # inputs = data['input_ids']
    # print("--- RAW INPUT IDS ---")
    # print(inputs.tolist())

    # # Check for obvious input errors
    # if (inputs == 0).all():
    #     print("\nWARNING: The entire input sequence is 0 (possible padding issue).")

    # find_redundant_chunks()
    token_ids = get_byte_ids(chunk_path='slice_100mb.txt')
    # windows = generate_windows(token_ids,512,"cpu")
    # make_bitmap('slice_100mb.txt')
    # test = get_byte_at('slice_100mb.txt',307)
    # print(test)
    get_entropy_concentration_stats(chunk_path="slice_100mb.txt",
    model_path="models/pym_particles_enwik_latest.pt",
    window_size=256,
    vocab_size=258,
    batch_size=64)
    # make_topk_bitmap_from_pt(pt_file_path=pt_out_path,chunk_path='slice_100mb.txt',K=k)



