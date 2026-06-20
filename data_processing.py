import torch
from tokenizer import get_byte_ids
from pym_transformer import PymTransformer
from tqdm import tqdm
import time

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

def load_secondary_windows(path, window_size=256, device='cuda'):
    import numpy as np
    data = np.fromfile(path, dtype=np.uint16)          # uint16 to preserve 256/257
    num_windows = len(data) // window_size
    arr = torch.from_numpy(data).long()
    windows = arr.view(num_windows, window_size).to(device)
    print(len(windows))
    return windows                                      # shape: (num_windows, 256)

def construct_secondary_dataset(
    chunk_path,
    model_path,
    window_size=256,
    vocab_size=258,
    batch_size=64,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    token_ids   = get_byte_ids(chunk_path=chunk_path)
    windows     = generate_windows(token_ids, window_size, device=device)
    num_windows = len(windows)

    STRIDE           = window_size // 2       # 128
    target_start_idx = window_size // 2       # predictions start at token 128

    model = PymTransformer(
        vocab_size=vocab_size,
        hidden_dim=128,
        num_layers=2,
        sequence_length=window_size
    ).to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    total_positions = num_windows * STRIDE + window_size
    bitmap = bytearray(total_positions // 8 + 1)

    with torch.no_grad():
        for batch_start in tqdm(
            range(0, num_windows, batch_size),
            desc="Analyzing positions"
        ):
            inp_b = windows[batch_start:batch_start + batch_size]

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _, _ = model(inp_b)

            # logits for positions [127:-1], targets for positions [128:]
            logits_trimmed  = logits[:, target_start_idx - 1:-1, :].float()
            targets_trimmed = inp_b[:, target_start_idx:]

            _, top2_indices = torch.topk(logits_trimmed, k=2, dim=-1)

            rank2_correct_mask = (targets_trimmed == top2_indices[:, :, 1])

            for row_idx, seq_pos in torch.nonzero(rank2_correct_mask, as_tuple=False):
                global_window_idx = batch_start + row_idx.item()

                # window i covers tokens [i*128 : i*128+256]
                # predictions cover     [i*128+128 : i*128+256]
                pos = global_window_idx * STRIDE + target_start_idx + seq_pos.item()

                bitmap[pos >> 3] |= (1 << (pos & 7))

    bitmap_path = chunk_path + '.bitmap'
    with open(bitmap_path, 'wb') as f:
        f.write(bitmap)
    print(f"bitmap saved → {bitmap_path}  ({len(bitmap)} bytes)")

    return bitmap

def load_bitmap(bitmap_path):
    with open(bitmap_path, 'rb') as f:
        return bytearray(f.read())



def count_chars(chunk_size=1024 * 1024):
    total = 0

    with open('slice_100mb.txt', "r", encoding="utf-8") as f:
        while chunk := f.read(chunk_size):
            total += len(chunk)

    print(total)

    return total

def get_entropy_concentration_stats(chunk_path, model_path, window_size=256, vocab_size=258, batch_size=64, top_k=2, output_path='miss_runs.bin', bitmap_path='miss_bitmap.bin'):
    import numpy as np
    import torch
    from itertools import groupby
    from tqdm import tqdm  
    import time, struct

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
    misses_per_position = torch.zeros(pred_seq_len, dtype=torch.long)
    counts_per_position = torch.zeros(pred_seq_len, dtype=torch.long)

    global_target_counts = torch.zeros(vocab_size, dtype=torch.long)
    global_rank1_counts  = torch.zeros(vocab_size, dtype=torch.long)
    global_rank2_counts  = torch.zeros(vocab_size, dtype=torch.long)
    
    rank1_pairs = torch.zeros((vocab_size, vocab_size), dtype=torch.long)
    rank2_pairs = torch.zeros((vocab_size, vocab_size), dtype=torch.long)

    # NEW: when true byte lands at rank 2, what did the model put at rank 1?
    # rows = true byte, cols = model's rank-1 prediction
    rank2_confusion = torch.zeros((vocab_size, vocab_size), dtype=torch.long)

    # Discretized non-overlapping band counters
    count_rank1  = 0
    count_rank2  = 0
    count_ranks_3_15   = 0
    count_ranks_16_30  = 0
    count_ranks_31_100 = 0

    with torch.no_grad():
        for batch_start in tqdm(range(0, num_windows, batch_size), desc="Analyzing positions"):
            batch_start_time = time.perf_counter()
            
            inp_b = windows[batch_start : batch_start + batch_size]
            current_batch_size = inp_b.shape[0]

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _, _ = model(inp_b)

            logits_trimmed  = logits[:, logit_start_idx:-1, :].float()
            targets_trimmed = inp_b[:, target_start_idx:]
            prev_bytes      = inp_b[:, logit_start_idx:-1] 

            top100 = logits_trimmed.topk(100, dim=-1).indices
            tgt_exp = targets_trimmed.unsqueeze(-1)

            # Fixed: Sliced into explicit non-overlapping target segments
            is_r1   = (tgt_exp == top100[..., :1]).any(dim=-1)       # Rank 1 (Index 0)
            is_r2   = (tgt_exp == top100[..., 1:2]).any(dim=-1)      # Rank 2 (Index 1)
            is_t15  = (tgt_exp == top100[..., 2:15]).any(dim=-1)     # Ranks 3-15 (Indices 2-14)
            is_t50  = (tgt_exp == top100[..., 15:30]).any(dim=-1)    # Ranks 16-30 (Indices 15-29)
            is_t100 = (tgt_exp == top100[..., 30:100]).any(dim=-1)   # Ranks 31-100 (Indices 30-99)

            # Accumulate mutually exclusive bracket hits
            count_rank1        += int(is_r1.sum().item())
            count_rank2        += int(is_r2.sum().item())
            count_ranks_3_15   += int(is_t15.sum().item())
            count_ranks_16_30  += int(is_t50.sum().item())
            count_ranks_31_100 += int(is_t100.sum().item())

            # Configured Top-K Hit/Miss behavior tracking
            topk_config = top100[..., :top_k]
            in_topk_config = (tgt_exp == topk_config).any(dim=-1)
            not_in_topk_config = ~in_topk_config
            
            misses_per_position += not_in_topk_config.sum(dim=0).cpu()
            counts_per_position += current_batch_size
            miss_flags.append(not_in_topk_config.reshape(-1).cpu())

            tgt_f  = targets_trimmed.flatten().cpu()
            prev_f = prev_bytes.flatten().cpu()
            r1_f   = is_r1.flatten().cpu()
            r2_f   = is_r2.flatten().cpu()

            global_target_counts += torch.bincount(tgt_f, minlength=vocab_size)

            if r1_f.any():
                global_rank1_counts += torch.bincount(tgt_f[r1_f], minlength=vocab_size)
                rank1_pairs.index_put_((tgt_f[r1_f], prev_f[r1_f]), torch.tensor(1), accumulate=True)

            if r2_f.any():
                global_rank2_counts += torch.bincount(tgt_f[r2_f], minlength=vocab_size)
                rank2_pairs.index_put_((tgt_f[r2_f], prev_f[r2_f]), torch.tensor(1), accumulate=True)

                # NEW: record what model predicted at rank 1 at these rank-2 positions
                rank1_pred_flat = top100[..., 0].flatten().cpu()
                rank2_confusion.index_put_((tgt_f[r2_f], rank1_pred_flat[r2_f]), torch.tensor(1), accumulate=True)

            batch_duration = time.perf_counter() - batch_start_time
            tqdm.write(f"Batch {(batch_start // batch_size) + 1:04d} | Duration: {batch_duration:.4f}s")

    miss_seq = torch.cat(miss_flags)
    packed_bitmap = np.packbits(miss_seq.numpy().astype(np.uint8))
    with open(bitmap_path, 'wb') as f: f.write(packed_bitmap.tobytes())

    runs = [(key, sum(1 for _ in group)) for key, group in groupby(miss_seq.tolist())]
    miss_ranges, pos = [], 0
    for flag, length in runs:
        if flag: miss_ranges.append((pos, pos + length))
        pos += length

    delta_values = []
    for i, (start, end) in enumerate(miss_ranges):
        delta_values.append(end)
        if i + 1 < len(miss_ranges): delta_values.append(miss_ranges[i + 1][0])

    total_positions = len(miss_seq)
    total_misses    = int(miss_seq.sum().item())
    total_hits      = total_positions - total_misses

    print("\n" + "=" * 60)
    print(f"Total Evaluated Tokens     : {total_positions}")
    print(f"Overall Hit Rate (Top-{top_k})   : {100 * total_hits / total_positions:.2f}%")
    print(f"Overall Miss Rate (Top-{top_k})  : {100 * total_misses / total_positions:.2f}%")
    print("-" * 60)
    print("=== Global Bracket Accuracy Performance ===")
    print(f"  Rank 1 Hits              : {count_rank1:<12,} | {100 * count_rank1 / total_positions:6.2f}%")
    print(f"  Rank 2 Hits              : {count_rank2:<12,} | {100 * count_rank2 / total_positions:6.2f}%")
    print(f"  Ranks 3-15 Band Hits     : {count_ranks_3_15:<12,} | {100 * count_ranks_3_15 / total_positions:6.2f}%")
    print(f"  Ranks 16-30 Band Hits    : {count_ranks_16_30:<12,} | {100 * count_ranks_16_30 / total_positions:6.2f}%")
    print(f"  Ranks 31-100 Band Hits   : {count_ranks_31_100:<12,} | {100 * count_ranks_31_100 / total_positions:6.2f}%")
    print("=" * 60)

    print("=== Average Miss Rate Per Sequence Position ===")
    position_miss_rates = (misses_per_position.float() / counts_per_position.float()).numpy()
    row_str = ""
    for i, miss_rate in enumerate(position_miss_rates):
        actual_pos = target_start_idx + i
        row_str += f"  Pos {actual_pos:3d}: {miss_rate*100:5.2f}% |"
        if (i + 1) % 4 == 0 or (i + 1) == len(position_miss_rates):
            print(row_str)
            row_str = ""
    print("=" * 60)

    def format_char(b):
        if 32 <= b <= 126: return f"'{chr(b)}'"
        mapping = {10: r"'\n' (LF)", 13: r"'\r' (CR)", 9: r"'\t' (TAB)", 32: "' ' (SPACE)"}
        return mapping.get(b, f"Hex: {hex(b)}")

    def print_table(title, total_hits_pool, counts, pairs):
        print(f"\n=== {title} ===")
        print(f"Total Instances: {total_hits_pool}\n")
        if total_hits_pool == 0: return
        
        top_counts, top_indices = torch.topk(counts, k=min(20, vocab_size))
        print(f"{'Byte':<6} | {'ASCII / Char':<13} | {'Count':<14} | {'% of Total':<12} | {'% of Byte Occ.':<16} | {'Top Preceding Byte (% of Match)'}")
        print("-" * 115)
        
        for count, byte_idx in zip(top_counts.tolist(), top_indices.tolist()):
            if count == 0: break
            total_occ = global_target_counts[byte_idx].item()
            
            row = pairs[byte_idx]
            top_prev_idx = row.argmax().item()
            prev_pct = (row[top_prev_idx].item() / count) * 100 if count > 0 else 0.0

            print(f"{byte_idx:<6} | {format_char(byte_idx):<13} | {count:<14,} | {(count/total_hits_pool)*100:<12.2f}% | {(count/total_occ)*100:<16.2f}% | {format_char(top_prev_idx)} ({prev_pct:.1f}%)")
        print("=" * 60)

    # NEW: print confusion table
    def print_confusion_table(title, counts, confusion_matrix):
        print(f"\n=== {title} ===")
        print(f"Total Rank-2 Instances: {int(counts.sum().item())}\n")
        if counts.sum() == 0: return

        top_counts, top_indices = torch.topk(counts, k=min(20, vocab_size))
        print(f"{'True Byte':<10} | {'R2 Count':>10} | {'Rank-1 Instead #1':<25} | {'Rank-1 Instead #2':<25} | {'Rank-1 Instead #3'}")
        print("-" * 105)

        for count, byte_idx in zip(top_counts.tolist(), top_indices.tolist()):
            if count == 0: break
            row = confusion_matrix[byte_idx]
            top3_vals, top3_idxs = row.topk(3)
            confusors = [f"{format_char(idx)} ({(val/count)*100:.1f}%)" for val, idx in zip(top3_vals.tolist(), top3_idxs.tolist()) if val > 0]
            while len(confusors) < 3: confusors.append("—")
            print(f"{format_char(byte_idx):<10} | {count:>10,} | {confusors[0]:<25} | {confusors[1]:<25} | {confusors[2]}")
        print("=" * 105)

    print_table("Target Byte Profile at Rank 1", int(global_rank1_counts.sum().item()), global_rank1_counts, rank1_pairs)
    print_table("Target Byte Profile at Rank 2", int(global_rank2_counts.sum().item()), global_rank2_counts, rank2_pairs)
    print_confusion_table("Rank-2 Confusion: What Was Rank 1 Instead?", global_rank2_counts, rank2_confusion)

    with open(output_path, 'wb') as f:
        for val in delta_values: f.write(struct.pack('I', val)) 

    return delta_values

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
    construct_secondary_dataset(chunk_path="slice_100mb.txt",
    model_path="models/pym_particles_enwik_latest.pt",
    window_size=256,
    vocab_size=258,
    batch_size=64)

    # load_secondary_windows('secondary_dataset.bin')
    # count_chars()


