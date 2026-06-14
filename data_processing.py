import torch
from tokenizer import get_byte_ids
import math
import torch.nn as nn
from pym_transformer import PymTransformer
import json
import csv
import torch.nn.functional as F  # The conventional F alias
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

# def extract_high_entropy_data(chunk_path, model_path, window_size=256, vocab_size=258, batch_size=64):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Extraction Device: {device}")

#     # 1. Load Data
#     token_ids = get_byte_ids(chunk_path=chunk_path)
#     windows = generate_windows(token_ids, window_size, device=device)
#     num_windows = len(windows)

#     # 2. Load Model 1
#     model = PymTransformer(vocab_size=vocab_size, hidden_dim=128, num_layers=2, sequence_length=window_size).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
#     model.eval()

#     criterion_unreduced = nn.CrossEntropyLoss(reduction='none')

#     # 3. Define buckets: (label, min, max)
#     bucket_definitions = [
#         ("0.0-0.3", 0.0,  0.3),
#         ("0.3-0.5", 0.3,  0.5),
#         ("0.5-0.7", 0.5,  0.7),
#         ("0.7-1.0", 0.7,  1.0),
#         ("1.0-1.5", 1.0,  1.5),
#         ("1.5+",    1.5,  float('inf')),
#     ]

#     # each bucket stores list of (window_tensor, avg_bits, window_index)
#     buckets = {label: [] for label, _, _ in bucket_definitions}

#     print(f"Scanning {num_windows} windows...")

#     with torch.no_grad():
#         for batch_start in range(0, num_windows, batch_size):
#             inp_b = windows[batch_start : batch_start + batch_size]
#             current_batch_size = inp_b.shape[0]

#             with torch.autocast(device_type='cuda', dtype=torch.float16):
#                 logits, _, _ = model(inp_b)

#                 logit_start_idx  = window_size // 2 - 1
#                 logits_trimmed   = logits[:, logit_start_idx:-1, :].reshape(-1, vocab_size)
#                 target_start_idx = logit_start_idx + 1
#                 targets_trimmed  = inp_b[:, target_start_idx:].reshape(-1)
#                 loss_array       = criterion_unreduced(logits_trimmed, targets_trimmed)

#             bits_array       = loss_array.float() / math.log(2)
#             predict_len      = window_size - target_start_idx
#             bits_per_window  = bits_array.reshape(current_batch_size, predict_len)
#             avg_bits         = bits_per_window.mean(dim=1)  # (current_batch_size,)

#             for i in range(current_batch_size):
#                 cost         = avg_bits[i].item()
#                 window_idx   = batch_start + i
#                 window_tensor = inp_b[i].cpu()

#                 for label, low, high in bucket_definitions:
#                     if low <= cost < high:
#                         buckets[label].append((window_tensor, cost, window_idx))
#                         break

#     # 4. Print distribution
#     print("\n── Bit Cost Distribution ──────────────────")
#     for label, _, _ in bucket_definitions:
#         count = len(buckets[label])
#         pct   = count / num_windows * 100
#         bar   = '█' * int(pct / 2)
#         print(f"  {label:>8}  {count:>8,} windows  ({pct:5.1f}%)  {bar}")
#     print(f"  {'total':>8}  {num_windows:>8,} windows")

#     # 5. Save each bucket as json + csv
#     saved_files = {}

#     for label, _, _ in bucket_definitions:
#         group = buckets[label]
#         if len(group) == 0 or label == "0.0-0.5":
#             continue

#         safe_label = label.replace("+", "plus").replace(".", "_")
#         json_path  = f'model_2_data_{safe_label}.json'
#         csv_path   = f'model_2_data_{safe_label}.csv'

#         # ── JSON (capped at 1000) ──
#         preview_limit = 1000
#         json_data = []

#         for window_tensor, avg_bits_val, window_idx in group[:preview_limit]:
#             token_list    = window_tensor.tolist()
#             byte_array    = bytes([t for t in token_list if t < 256])
#             readable_text = byte_array.decode('utf-8', errors='replace')
#             json_data.append({
#                 "window_index":          window_idx,
#                 "average_bits_per_byte": avg_bits_val,
#                 "text_preview":          readable_text
#             })

#         truncated = len(group) - preview_limit
#         if truncated > 0:
#             json_data.append({
#                 "truncated":         True,
#                 "message":           f"{truncated:,} more windows not shown",
#                 "total_in_bucket":   len(group)
#             })

#         with open(json_path, 'w', encoding='utf-8') as f:
#             json.dump(json_data, f, indent=4, ensure_ascii=False)

#         # ── CSV (full, no truncation) ──
#         with open(csv_path, 'w', newline='', encoding='utf-8') as f:
#             writer = csv.DictWriter(f, fieldnames=['window_index', 'average_bits_per_byte', 'text_preview'])
#             writer.writeheader()
#             for window_tensor, avg_bits_val, window_idx in group:
#                 token_list    = window_tensor.tolist()
#                 byte_array    = bytes([t for t in token_list if t < 256])
#                 readable_text = byte_array.decode('utf-8', errors='replace')
#                 writer.writerow({
#                     'window_index':          window_idx,
#                     'average_bits_per_byte': avg_bits_val,
#                     'text_preview':          readable_text
#                 })

#         print(f"saved {len(group):>8,} windows → {json_path} + {csv_path}")
#         saved_files[label] = {'json': json_path, 'csv': csv_path}

# # ── build filtered windows tensors ─────────────────────────────────────
#     hard_bucket_labels = {"0.7-1.0", "1.0-1.5", "1.5+"}
#     hard_indices = set(w[2] for label in hard_bucket_labels for w in buckets[label])

#     print(f"\nbuilding filtered windows tensors...")
#     print(f"total windows    : {num_windows:,}")
#     print(f"hard windows     : {len(hard_indices):,}")

#     # model 1 tensor — excludes hard windows
#     keep_indices     = [i for i in range(num_windows) if i not in hard_indices]
#     filtered_windows = windows[keep_indices]
#     torch.save(filtered_windows.to(torch.int16), 'model1_windows.pt')
#     print(f"model 1          : {len(keep_indices):,} windows → model1_windows.pt")

#     # model 2 tensor — only hard windows
#     hard_only_indices   = sorted(hard_indices)
#     hard_windows_tensor = windows[hard_only_indices]
#     torch.save(hard_windows_tensor.to(torch.int16), 'model2_windows.pt')
#     print(f"model 2          : {len(hard_only_indices):,} windows → model2_windows.pt")
#     return buckets, saved_files


def extract_high_entropy_data(
    chunk_path,
    model_path,
    window_size=256,
    vocab_size=258,
    batch_size=64,
    out_dir="extracted_batches"
):
    import os
    import torch
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Extraction Device: {device}")

    os.makedirs(out_dir, exist_ok=True)

    total_t0 = time.perf_counter()

    token_ids = get_byte_ids(chunk_path=chunk_path)
    windows = generate_windows(token_ids, window_size, device=device).contiguous()

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

    logit_start_idx = window_size // 2 - 1

    torch.backends.cuda.matmul.allow_tf32 = True

    batch_id = 0

    with torch.inference_mode():
        for batch_start in range(0, windows.shape[0], batch_size):

            inp_b = windows[batch_start:batch_start + batch_size]

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.autocast(device_type='cuda', dtype=torch.float16):

                logits, _, _ = model(inp_b)

                logits = logits[:, logit_start_idx:-1, :]
                K, T, V = logits.shape

                logits = logits.reshape(K * T, V)

                targets = inp_b[:, logit_start_idx + 1:].reshape(-1)

                logits = logits.float()

                idx = torch.arange(logits.size(0), device=device)
                logits[idx, targets] = -1e9

                top_vals, top_idx = torch.topk(logits, k=255, dim=-1)

            if device.type == "cuda":
                torch.cuda.synchronize()

            dt = time.perf_counter() - t0
            print(f"Batch {batch_id} time: {dt:.6f}s")

            torch.save(
                {
                    "top_vals": top_vals.cpu(),
                    "top_idx": top_idx.cpu(),
                    "batch_start": batch_start,
                    "batch_size": inp_b.shape[0]
                },
                os.path.join(out_dir, f"batch_{batch_id:06d}.pt")
            )

            batch_id += 1

    total_dt = time.perf_counter() - total_t0
    print(f"Total time: {total_dt:.4f}s")
    print(f"Batches saved: {batch_id}")


def bitmap_run_analysis(out_dir):
    import os
    import torch

    print("\n── Bitmap Run Probability Analysis ──")

    files = sorted(
        f for f in os.listdir(out_dir)
        if f.endswith(".pt")
    )

    if not files:
        print("No batch files found")
        return

    # load run space = window space
    all_batches = []

    for f in files:
        data = torch.load(os.path.join(out_dir, f), map_location="cpu")

        all_batches.append({
            "start": data["batch_start"],
            "vals": data["top_vals"],   # [B, 255]
            "idx": data["top_idx"]      # [B, 255]
        })

    # sort by window position
    all_batches.sort(key=lambda x: x["start"])

    total_windows = all_batches[-1]["start"] + all_batches[-1]["vals"].shape[0]

    # build simple run heuristic in WINDOW SPACE
    runs = []
    run_start = 0
    active_score = 0.0

    WINDOW_BATCH = 0

    for b in all_batches:
        vals = b["vals"]

        # entropy proxy: how flat the top-k distribution is
        entropy_proxy = vals.mean(dim=1).mean().item()

        WINDOW_BATCH += vals.shape[0]

        if entropy_proxy < 0.002:
            if WINDOW_BATCH > run_start:
                runs.append((run_start, WINDOW_BATCH))
            run_start = WINDOW_BATCH

    if run_start < total_windows:
        runs.append((run_start, total_windows))

    runs = sorted(runs, key=lambda x: x[1] - x[0], reverse=True)[:10]

    print(f"{'Start':>10} {'End':>10} {'Length':>10}")

    for s, e in runs:
        print(f"{s:>10} {e:>10} {e - s:>10}")

def show_absent_bytes(sp_correct, start_token, N):
    end_token = start_token + N
    correct = sp_correct[start_token:end_token]  # [N]

    # which bytes appeared as correct at least once
    ever_correct = torch.zeros(258, dtype=torch.bool)
    ever_correct[correct.long()] = True

    absent = (~ever_correct).nonzero(as_tuple=True)[0].tolist()

    print(f"\nTokens {start_token} → {start_token + N - 1}  ({N} tokens)")
    print(f"Bytes never correct (safe to mask): {len(absent)} bytes")
    print(absent)

def show_continuous_token_bytes(sp_indices, sp_correct, start_token, N):
    end_token = start_token + N
    indices = sp_indices[start_token:end_token]  # [N, 255]
    correct = sp_correct[start_token:end_token]  # [N]

    # how many times each byte appeared as non-correct
    present = torch.bincount(indices.long().reshape(-1), minlength=258).int()
    # not present = N - present (max times it could appear)
    not_present = N - present

    print(f"\nTokens {start_token} → {start_token + N - 1}  ({N} tokens)")
    print(f"{'Byte':>6}  {'Present':>8}  {'Not Present':>12}")
    print("─" * 32)
    for sym in range(258):
        print(f"  {sym:>4}  {present[sym].item():>8}  {not_present[sym].item():>12}")


def show_token_distributions(sorted_indices, sorted_probs, token_positions, top_k=20):
    """
    Print the probability distribution for N token positions in a readable format.

    Args:
        sorted_indices  : tensor [N_tokens, 255]  wrong symbols sorted by prob desc
        sorted_probs    : tensor [N_tokens, 255]  their probabilities
        token_positions : list of int             which token positions to show
        top_k           : how many top symbols to display per token (default 20)

    Example:
        indices = torch.load('sorted_indices.pt')
        probs   = torch.load('sorted_probs.pt')
        show_token_distributions(indices, probs, token_positions=[0, 1, 100])
    """

    for pos in token_positions:
        syms  = sorted_indices[pos].tolist()   # 255 symbol indices
        pvals = sorted_probs[pos].tolist()     # their probabilities

        total_mass = sum(pvals)
        cumsum     = 0.0

        print(f"\n{'─'*52}")
        print(f"  Token {pos}   |   total excludable mass: {total_mass:.4f}")
        print(f"{'─'*52}")
        print(f"  {'Rank':<5}  {'Byte':>5}  {'Char':<5}  {'Prob':>8}  {'Cumsum':>8}  Bar")
        print(f"{'─'*52}")

        for rank, (sym, prob) in enumerate(zip(syms[:top_k], pvals[:top_k])):
            cumsum += prob

            # human readable character
            if 32 <= sym <= 126:
                char = f"'{chr(sym)}'"
            elif sym == 0:
                char = "NUL"
            elif sym == 9:
                char = "TAB"
            elif sym == 10:
                char = " LF"
            elif sym == 13:
                char = " CR"
            elif sym == 32:
                char = "SPC"
            else:
                char = f"x{sym:02X}"

            bar_len = int(prob / pvals[0] * 24) if pvals[0] > 0 else 0
            bar     = "█" * bar_len

            print(f"  {rank+1:<5}  {sym:>5}  {char:<5}  {prob:>8.4f}  {cumsum:>8.4f}  {bar}")

        if len(pvals) > top_k:
            remaining_mass = sum(pvals[top_k:])
            print(f"  {'...':<5}  {'':>5}  {'':5}  {'':>8}  {'':>8}")
            print(f"  remaining {255 - top_k} symbols  mass: {remaining_mass:.4f}")

        print(f"{'─'*52}")

import json

import struct

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




def extract_run_signals(
    chunk_path,
    model_path,
    window_size=256,
    vocab_size=258,
    batch_size=64,
    out_file="run_signals.pt"
):
    import torch
    import torch.nn.functional as F
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token_ids = get_byte_ids(chunk_path)
    windows = generate_windows(token_ids, window_size, device=device).contiguous()

    model = PymTransformer(
        vocab_size=vocab_size,
        hidden_dim=128,
        num_layers=2,
        sequence_length=window_size
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    logit_start_idx = window_size // 2 - 1

    raw = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)

    # build run map
    runs = []
    seen = set()
    start = 0

    for i, b in enumerate(raw):
        seen.add(b)

        if (256 - len(seen)) < 170:
            if i > start:
                runs.append((start, i, set(range(256)) - seen))
            start = i
            seen = {b}

    if start < len(raw):
        runs.append((start, len(raw) - 1, set(range(256)) - seen))

    run_map = torch.full((len(raw),), -1, dtype=torch.long)
    for r_id, (s, e, _) in enumerate(runs):
        run_map[s:e + 1] = r_id

    run_values = torch.zeros(len(runs), dtype=torch.float32)
    run_counts = torch.zeros(len(runs), dtype=torch.float32)

    all_top_vals = []
    all_top_idx = []

    with torch.inference_mode():
        global_window_idx = 0

        for batch_start in range(0, windows.shape[0], batch_size):

            inp_b = windows[batch_start:batch_start + batch_size]

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, _, _ = model(inp_b)

                logits = logits[:, logit_start_idx:-1, :]
                K, T, V = logits.shape

                logits = logits.reshape(K * T, V)

                probs = F.softmax(logits.float(), dim=-1)

                top_vals, top_idx = torch.topk(probs, k=10, dim=-1)

            all_top_vals.append(top_vals.cpu())
            all_top_idx.append(top_idx.cpu())

            top_vals_sum = top_vals.sum(dim=1).cpu()

            n = top_vals_sum.shape[0]
            idx = torch.arange(n)

            byte_pos = global_window_idx + idx + logit_start_idx
            byte_pos = byte_pos.clamp(max=len(raw) - 1)

            run_ids = run_map[byte_pos]
            valid = run_ids >= 0

            run_values.index_add_(0, run_ids[valid], top_vals_sum[valid])
            run_counts.index_add_(0, run_ids[valid], torch.ones_like(top_vals_sum[valid]))

            global_window_idx += n

    torch.save(
    {
        "run_values": run_values,
        "run_counts": run_counts,
        "runs": runs,
        "top_vals": torch.cat(all_top_vals, dim=0),
        "top_idx": torch.cat(all_top_idx, dim=0),
    },
    out_file
)

    print(f"saved: {out_file}")

def bitmap_run_analysis_stream(
    windows,
    model,
    runs,
    raw,
    batch_size=64,
    window_size=256,
    k=10
):
    import torch
    import torch.nn.functional as F
    import time

    device = next(model.parameters()).device
    logit_start_idx = window_size // 2 - 1

    # map byte position -> run id
    run_map = torch.full((len(raw),), -1, dtype=torch.long)
    for r_id, (s, e, _) in enumerate(runs):
        run_map[s:e + 1] = r_id

    run_mass = torch.zeros(len(runs), dtype=torch.float32)
    run_hits = torch.zeros(len(runs), dtype=torch.float32)
    run_total = torch.zeros(len(runs), dtype=torch.float32)

    with torch.inference_mode():
        global_idx = 0

        for batch_start in range(0, windows.shape[0], batch_size):

            t0 = time.perf_counter()

            inp_b = windows[batch_start:batch_start + batch_size]

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, _, _ = model(inp_b)

                logits = logits[:, logit_start_idx:-1, :]
                K, T, V = logits.shape

                logits = logits.reshape(K * T, V)
                probs = F.softmax(logits.float(), dim=-1)

                top_vals, top_idx = torch.topk(probs, k=k, dim=-1)

            top_vals = top_vals.cpu()
            top_idx = top_idx.cpu()

            n = top_vals.shape[0]
            pos = global_idx + torch.arange(n)

            byte_pos = pos + logit_start_idx
            byte_pos = byte_pos.clamp(max=len(raw) - 1)

            run_ids = run_map[byte_pos]

            for i in range(n):
                r = run_ids[i].item()
                if r < 0:
                    continue

                _, _, absent = runs[r]

                abs_set = absent
                idxs = top_idx[i].tolist()
                vals = top_vals[i].tolist()

                hit = 0
                mass = 0.0

                for j in range(k):
                    if idxs[j] in abs_set:
                        hit += 1
                        mass += vals[j]

                run_hits[r] += hit
                run_mass[r] += mass
                run_total[r] += k

            global_idx += n

            print(f"batch {batch_start}: {time.perf_counter() - t0:.4f}s")

    print("\n── Run Summary ──")

    for r_id, (s, e, absent) in enumerate(runs):

        if run_total[r_id] == 0:
            continue

        coverage = run_hits[r_id].item() / run_total[r_id].item()
        avg_mass = run_mass[r_id].item() / run_total[r_id].item()

        print(
            f"{s:>8} {e:>8} "
            f"absent={len(absent):>4} "
            f"coverage={coverage:.4f} "
            f"mass={avg_mass:.6f}"
        )
# ── quick usage ───────────────────────────────────────────────────────────────
if __name__ == "__main__":


    # Step 1: run extraction once (slow)
    # extract_run_signals(
    #     chunk_path="slice_100mb.txt",
    #     model_path="models/pym_particles.pt",
    #     window_size=256,
    #     vocab_size=258,
    #     batch_size=64,
    #     out_file="run_signals.pt"
    # )

    # # Step 2: instant analysis (fast, repeatable)
    # bitmap_run_analysis_from_file(
    #     pt_file="run_signals.pt"
    # )

    # sp_indices = torch.load('sp_0_7_1_0_indices.pt')
    # sp_correct = torch.load('sp_0_7_1_0_correct.pt')

    make_bitmap('slice_100mb.txt')

    # # show first 3 tokens
    # show_token_distributions(indices, probs, token_positions=[0, 1, 2], top_k=20)

    # # show random sample
    # import random
    # sample = random.sample(range(n_tokens), 3)
    # print(f"\n\nRandom sample: {sample}")
    # show_token_distributions(indices, probs, token_positions=sample, top_k=20)
        

# def find_redundant_chunks():
#     test = {}

#     print('test')

#     with open("test.txt", "rb") as f:
#         while chunk := f.read(128):
#             # process chunk
#             if chunk in test:
#                 test[chunk] += 1
#             else:
#                 test[chunk] = 1

#     y = 0

#     print(len(test))

#     for key,value in test.items():
#         if(value > 1):
#             print(value)

# def slice_file(input_path: str, output_path: str, target_mb: int):
#     target_size = target_mb * 1024 * 1024

#     with open(input_path, "rb") as f:
#         data = f.read()

#     if len(data) > target_size:
#         data = data[:target_size]
#     else:
#         data = data + b"\x00" * (target_size - len(data))

#     with open(output_path, "wb") as f:
#         f.write(data)

# if __name__ == '__main__':
#     dataset_for_model_2 = extract_high_entropy_data('slice_100mb.txt', 'models/pym_particles.pt')
#     # slice_file('test.txt','test2.txt',50)




