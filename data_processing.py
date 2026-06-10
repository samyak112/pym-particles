import torch

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

    return torch.stack(windows, dim=0)

def generate_custom_mask(seq_len: int, dead_rows: int, device=None) -> torch.Tensor:
    # Start clean: False means "do not mask" (allow attention)
    mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.bool)
    
    # Causal block: In-place setting of the upper triangle to True
    mask.triu_(diagonal=1)
    
    # Dead rows block: Set the entire row to True (mask it completely)
    mask[:dead_rows, :] = True
    
    return mask

def find_redundant_chunks():
    test = {}

    print('test')

    with open("test.txt", "rb") as f:
        while chunk := f.read(256):
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
    # custom_mask = generate_custom_mask(8, 8)
    import torch

    data = torch.load('corrupted_batch_item_11.pt') # Replace X with the actual number

    inputs = data['input_ids']
    print("--- RAW INPUT IDS ---")
    print(inputs.tolist())

    # Check for obvious input errors
    if (inputs == 0).all():
        print("\nWARNING: The entire input sequence is 0 (possible padding issue).")

    # token_ids = get_byte_ids(chunk_path='test.txt')
    # windows = generate_windows(token_ids,512,"cpu")



