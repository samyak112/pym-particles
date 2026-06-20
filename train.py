import math
import torch
import torch.nn as nn
from tokenizer import get_byte_ids
from data_processing import generate_windows,load_secondary_windows,load_bitmap
from pym_transformer import PymTransformer
import time
import json

model_name = 'pym_particles_non_rank1320_full_size.pt'
log_path   = "loss_log.json"


def train(
    chunk_path,
    bitmap      = None,
    epochs      = 20,
    lr          = 2e-3,
    window_size = 256,
    vocab_size  = 258,
    batch_size  = 64,
    hidden_dims = 128,
    num_layers  = 2,
    size=100
):
    STRIDE           = window_size // 2
    target_start_idx = window_size // 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device  : {device}")

    token_ids   = get_byte_ids(chunk_path=chunk_path, size_mb=size)
    windows     = generate_windows(token_ids, window_size, device=device)
    num_windows = len(windows)
    print(f"windows : {num_windows}")

    use_bitmap = bitmap is not None
    if use_bitmap:
        bitmap_tensor = torch.frombuffer(bytes(bitmap), dtype=torch.uint8).to(device)
        seq_offsets   = torch.arange(STRIDE, device=device)
        criterion     = nn.CrossEntropyLoss(reduction='none')
        print("bitmap  : active")
    else:
        criterion = nn.CrossEntropyLoss()
        print("bitmap  : none (training on all positions)")

    steps_per_epoch = math.ceil(num_windows / batch_size)
    total_steps     = epochs * steps_per_epoch

    model     = PymTransformer(
                    vocab_size=vocab_size,
                    hidden_dim=hidden_dims,
                    num_layers=num_layers,
                    sequence_length=window_size
                ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True)
    scaler    = torch.cuda.amp.GradScaler()

    warmup_steps = 500
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
            ),
            torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=total_steps - warmup_steps
            ),
        ],
        milestones=[warmup_steps]
    )

    best_loss = float('inf')

    with open(log_path, "a") as f:
        f.write(json.dumps({
            'file_name'  : chunk_path,
            'hidden_dims': hidden_dims,
            'num_layers' : num_layers,
            'bitmap'     : use_bitmap,
        }) + "\n")

    print('starting training')

    for epoch in range(epochs):
        epoch_start   = time.perf_counter()
        total_loss    = torch.tensor(0.0, device=device)
        running_steps = 0

        model.train()

        perm             = None if use_bitmap else torch.randperm(num_windows, device=device)
        windows_to_train = windows if use_bitmap else windows[perm]

        for batch_start in range(0, num_windows, batch_size):
            inp_b              = windows_to_train[batch_start:batch_start + batch_size]
            current_batch_size = inp_b.shape[0]

            logits_trimmed  = None
            targets_trimmed = None

            if use_bitmap:
                global_window_idx = batch_start + torch.arange(current_batch_size, device=device)
                positions = (
                    global_window_idx.unsqueeze(1) * STRIDE
                    + target_start_idx
                    + seq_offsets.unsqueeze(0)
                )  # [B, 128]

                mask = ((bitmap_tensor[positions >> 3] >> (positions & 7)) & 1).bool()  # [B, 128]

                if not mask.any():
                    scheduler.step()
                    continue

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, _, _ = model(inp_b)

                logits_trimmed  = logits[:, target_start_idx - 1:-1, :]  # [B, 128, vocab]
                targets_trimmed = inp_b[:, target_start_idx:]             # [B, 128]

                if use_bitmap:
                    loss_per_token = criterion(
                        logits_trimmed.reshape(-1, vocab_size),
                        targets_trimmed.reshape(-1)
                    ).reshape(current_batch_size, STRIDE)
                    loss = loss_per_token[mask].mean()
                else:
                    loss = criterion(
                        logits_trimmed.reshape(-1, vocab_size),
                        targets_trimmed.reshape(-1)
                    )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            scaler.step(optimizer)
            scaler.update()

            total_loss.add_(loss.detach())
            running_steps += 1
            scheduler.step()

        if running_steps == 0:
            print(f"epoch {epoch + 1}: no bitmap positions hit, skipping")
            continue

        avg_loss       = (total_loss / running_steps).item()
        bits_per_token = avg_loss / math.log(2)
        elapsed        = time.perf_counter() - epoch_start

        log_entry = {
            "epoch"         : epoch + 1,
            "loss"          : avg_loss,
            "bits_per_token": bits_per_token,
            "lr"            : scheduler.get_last_lr()[0],
            "time_sec"      : elapsed,
        }
        print(log_entry)

        if avg_loss < best_loss:
            best_loss  = avg_loss
            model_path = f'models/{model_name}'
            torch.save(model.state_dict(), model_path)

            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(f"saved model   → {model_path}")
            print(f"new best loss → {best_loss:.4f}")

    return model


bitmap = load_bitmap('slice_100mb.txt.rank3_20.bitmap')

train('slice_100mb.txt',bitmap=bitmap, epochs=100, size=100)