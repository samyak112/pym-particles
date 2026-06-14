import torch
from tokenizer import get_byte_ids
import struct
import bisect

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
    # token_ids = get_byte_ids(chunk_path='test.txt')
    # windows = generate_windows(token_ids,512,"cpu")
    # make_bitmap('slice_100mb.txt')
    test = get_byte_at('slice_100mb.txt',307)
    print(test)



